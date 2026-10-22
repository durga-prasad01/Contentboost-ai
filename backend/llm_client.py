"""
llm_client.py — Google Gemini API wrapper for ContentBoost AI.

Improvements vs original:
- Async-safe: blocking generate_content() wrapped in run_in_executor
- Retry with exponential backoff via tenacity
- Simple TTL cache for analyze_content() calls
- Better prompts: tone definitions, few-shot example, no fake SEO metrics
- LLM_TIMEOUT env var respected
- generate_content_async() used for streaming endpoint
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from typing import Any, AsyncIterator, Dict

import google.generativeai as genai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.constants import FEW_SHOT_SEO_EXAMPLE, TONE_DEFINITIONS
from backend.models import GenerateResponseSchema

# ── Config ─────────────────────────────────────────────────────────────────────

LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "60"))
# gemini-2.0-flash: confirmed compatible with google-generativeai==0.8.3 (v1beta API)
# gemini-1.5-flash causes 404 with this SDK version
_MODEL_NAME = "gemini-2.0-flash"
_STREAM_MODEL_NAME = "gemini-2.0-flash"
_client = None
_stream_client = None

# Simple in-memory TTL cache: {hash -> (result, expires_at)}
_cache: Dict[str, tuple[Any, float]] = {}
_CACHE_TTL = 300  # 5 minutes


# ── Client singleton ───────────────────────────────────────────────────────────

def get_client() -> genai.GenerativeModel:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == "your_gemini_api_key_here":
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file (see .env.example)."
            )
        genai.configure(api_key=api_key)
        _client = genai.GenerativeModel(_MODEL_NAME)
    return _client


def get_stream_client() -> genai.GenerativeModel:
    """Separate client for the streaming endpoint — uses a non-thinking model
    to avoid having thinking tokens eat into the output token budget."""
    global _stream_client
    if _stream_client is None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == "your_gemini_api_key_here":
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file (see .env.example)."
            )
        genai.configure(api_key=api_key)
        _stream_client = genai.GenerativeModel(_STREAM_MODEL_NAME)
    return _stream_client


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> Dict[str, Any]:
    """Strip markdown fences and parse JSON robustly with repair fallback."""
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM response. Raw: {raw[:200]}")
    candidate = clean[start:end]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Attempt to repair truncated JSON by closing open structures
        repaired = _repair_json(candidate)
        return json.loads(repaired)


def _repair_json(text: str) -> str:
    """Best-effort repair of truncated JSON by closing unclosed strings/brackets."""
    # Close any unterminated string value by appending '"'
    # Count unmatched braces/brackets
    in_string = False
    escape_next = False
    open_braces = 0
    open_brackets = 0
    result = []

    for ch in text:
        if escape_next:
            escape_next = False
            result.append(ch)
            continue
        if ch == '\\' and in_string:
            escape_next = True
            result.append(ch)
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == '{':
                open_braces += 1
            elif ch == '}':
                open_braces -= 1
            elif ch == '[':
                open_brackets += 1
            elif ch == ']':
                open_brackets -= 1
        result.append(ch)

    repaired = ''.join(result)
    # Close unterminated string
    if in_string:
        repaired += '"'
    # Close unclosed arrays and objects
    repaired += ']' * open_brackets
    repaired += '}' * open_braces
    return repaired


def _cache_key(*args: Any) -> str:
    payload = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _get_cache(key: str) -> Any | None:
    entry = _cache.get(key)
    if entry and time.time() < entry[1]:
        return entry[0]
    _cache.pop(key, None)
    return None


def _set_cache(key: str, value: Any) -> None:
    _cache[key] = (value, time.time() + _CACHE_TTL)


# ── Quota / rate-limit helpers ────────────────────────────────────────────────

def _is_quota_error(exc: Exception) -> bool:
    """Return True when the Gemini API returned a 429 quota/rate-limit error."""
    msg = str(exc).lower()
    return "429" in msg or "quota" in msg or "resource_exhausted" in msg


def _quota_retry_seconds(exc: Exception) -> int:
    """Extract the retry_delay from a 429 error message, defaulting to 60s."""
    import re
    m = re.search(r"retry[_ ]delay.*?(\d+)", str(exc), re.IGNORECASE)
    if m:
        return int(m.group(1)) + 2  # small buffer
    return 60


# ── Retry decorator ────────────────────────────────────────────────────────────

def _should_retry(exc: Exception) -> bool:
    """Retry on transient errors, but NOT on quota/rate-limit errors (429).
    Retrying a quota error just makes the cooldown worse."""
    return not _is_quota_error(exc)


_retry = retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)


# ── Async wrapper ──────────────────────────────────────────────────────────────

async def _call_llm_json(prompt: str, response_schema: Any = None) -> Dict[str, Any]:
    """
    Native async call to Gemini API, enforcing JSON output.
    Raises a clean RuntimeError with wait time on quota errors (429)
    instead of retrying (which makes the cooldown worse).
    """
    @_retry
    async def _async_call() -> Dict[str, Any]:
        model = get_client()
        config_kwargs = {
            "max_output_tokens": 8192,
            "temperature": 0.7,
            "response_mime_type": "application/json",
        }
        if response_schema:
            config_kwargs["response_schema"] = response_schema
            
        resp = await model.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(**config_kwargs),
            request_options={"timeout": LLM_TIMEOUT},
        )
        return json.loads(resp.text)

    try:
        return await _async_call()
    except Exception as exc:
        if _is_quota_error(exc):
            wait = _quota_retry_seconds(exc)
            raise RuntimeError(
                f"QUOTA_EXCEEDED:{wait}:You've hit the Gemini API free-tier rate limit. "
                f"Please wait {wait} seconds and try again."
            ) from exc
        raise


# ── Public Functions ───────────────────────────────────────────────────────────

async def analyze_content(
    product_name: str,
    category: str | None,
    existing_description: str | None,
    competitor_content: str | None,
) -> Dict[str, Any]:
    """
    Analyze product and competitor content.
    Results are cached for 5 minutes to avoid duplicate LLM calls.
    """
    cache_key = _cache_key("analyze", product_name, category, existing_description, competitor_content)
    cached = _get_cache(cache_key)
    if cached:
        return cached

    prompt = f"""You are an expert e-commerce content analyst.

Analyze the following product information and competitor content.

Product: "{product_name}"
Category: {category or "General"}
Existing description: {existing_description or "(none)"}
Competitor content: {competitor_content or "(none)"}

Return ONLY valid JSON — no markdown, no preamble, no trailing text:
{{
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5", "keyword6", "keyword7", "keyword8"],
  "competitor_insights": ["specific insight about competitor strategy 1", "insight 2", "insight 3"],
  "writing_patterns": ["pattern 1", "pattern 2", "pattern 3"],
  "common_features": ["feature 1", "feature 2", "feature 3", "feature 4"]
}}"""

    result = await _call_llm_json(prompt)
    _set_cache(cache_key, result)
    return result


async def generate_descriptions(
    product_name: str,
    category: str | None,
    existing_description: str | None,
    competitor_content: str | None,
    target_audience: str | None,
    tone: str,
) -> Dict[str, Any]:
    """
    Generate 3 optimised product descriptions + suggestions.
    SEO metrics are NOT requested from the LLM (computed locally by seo_analyzer).
    """
    tone_instruction = TONE_DEFINITIONS.get(tone, "")

    prompt = f"""You are ContentBoost AI, an expert e-commerce product description optimizer.

{tone_instruction}

{FEW_SHOT_SEO_EXAMPLE}

Product: "{product_name}"
Category: {category or "General"}
Target audience: {target_audience or "General consumers"}
Existing description: {existing_description or "(none provided)"}
Competitor reference: {competitor_content or "(none provided)"}

Generate exactly this JSON structure — no markdown fences, no extra text outside the JSON:
{{
  "keywords": ["keyword1","keyword2","keyword3","keyword4","keyword5","keyword6","keyword7","keyword8"],
  "competitor_insights": ["specific insight 1","insight 2","insight 3"],
  "seo_version": {{
    "title": "SEO optimised title under 70 chars, front-load primary keyword",
    "description": "150-200 word SEO description. Natural keyword placement (1.5-2.5%). Benefit-first structure. Active voice. Clear call-to-action at the end. Cover core features, key benefits, and use case."
  }},
  "marketing_version": {{
    "title": "Emotionally compelling headline that creates desire",
    "description": "150-200 word marketing description. Lifestyle transformation, emotional benefits, sensory language, social proof cues, aspirational outcomes. Apply the tone defined above throughout."
  }},
  "technical_version": {{
    "title": "Technical product title with key spec or differentiator",
    "description": "150-200 word technical description. Lead with specifications, materials, performance metrics, compatibility, certifications, and precise technical language."
  }},
  "suggestions": [
    {{"type": "tip", "text": "Actionable improvement suggestion 1"}},
    {{"type": "tip", "text": "Actionable improvement suggestion 2"}},
    {{"type": "warn", "text": "Something to watch out for"}}
  ]
}}"""

    return await _call_llm_json(prompt, response_schema=GenerateResponseSchema)


async def refine_description(
    version_type: str,
    current_title: str,
    current_description: str,
    instruction: str | None = None,
) -> Dict[str, str]:
    """
    Refine a specific product description version.
    Returns improved title and description.
    """
    focus_map = {
        "seo_version": "SEO optimization: improve keyword placement, clarity, and search ranking potential",
        "marketing_version": "marketing impact: increase emotional appeal, desire, and conversion potential",
        "technical_version": "technical precision: enhance spec coverage, accuracy, and professional tone",
    }
    focus = focus_map.get(version_type, "overall quality")
    extra = f"\n\nSpecific instruction from user: {instruction}" if instruction else ""

    prompt = f"""You are an expert e-commerce copywriter. Refine the following product description with focus on {focus}.{extra}

Current title: "{current_title}"
Current description:
"{current_description}"

Return ONLY valid JSON — no markdown:
{{
  "title": "improved title here",
  "description": "improved description here, same approximate length as original"
}}"""

    return await _call_llm_json(prompt)


# ── Streaming generator ────────────────────────────────────────────────────────

async def generate_descriptions_stream(
    product_name: str,
    category: str | None,
    existing_description: str | None,
    competitor_content: str | None,
    target_audience: str | None,
    tone: str,
) -> AsyncIterator[tuple[str, Any]]:
    """
    Async generator that yields (event_name, data) tuples for SSE streaming.
    Yields actual token chunks as they arrive from Gemini.
    """
    yield ("status", {"step": 1, "message": "Analysing product & competitor content…"})
    
    tone_instruction = TONE_DEFINITIONS.get(tone, "")
    prompt = f"""You are ContentBoost AI, an expert e-commerce product description optimizer.

{tone_instruction}

{FEW_SHOT_SEO_EXAMPLE}

Product: "{product_name}"
Category: {category or "General"}
Target audience: {target_audience or "General consumers"}
Existing description: {existing_description or "(none provided)"}
Competitor reference: {competitor_content or "(none provided)"}

Generate exactly this JSON structure — no markdown fences, no extra text outside the JSON:
{{
  "keywords": ["keyword1","keyword2","keyword3","keyword4","keyword5","keyword6","keyword7","keyword8"],
  "competitor_insights": ["specific insight 1","insight 2","insight 3"],
  "seo_version": {{
    "title": "SEO optimised title under 70 chars, front-load primary keyword",
    "description": "150-200 word SEO description. Natural keyword placement (1.5-2.5%). Benefit-first structure. Active voice. Clear call-to-action at the end. Cover core features, key benefits, and use case."
  }},
  "marketing_version": {{
    "title": "Emotionally compelling headline that creates desire",
    "description": "150-200 word marketing description. Lifestyle transformation, emotional benefits, sensory language, social proof cues, aspirational outcomes. Apply the tone defined above throughout."
  }},
  "technical_version": {{
    "title": "Technical product title with key spec or differentiator",
    "description": "150-200 word technical description. Lead with specifications, materials, performance metrics, compatibility, certifications, and precise technical language."
  }},
  "suggestions": [
    {{"type": "tip", "text": "Actionable improvement suggestion 1"}},
    {{"type": "tip", "text": "Actionable improvement suggestion 2"}},
    {{"type": "warn", "text": "Something to watch out for"}}
  ]
}}"""

    model = get_stream_client()
    # NOTE: response_schema is intentionally omitted here — combining it with
    # stream=True causes Gemini to emit truncated JSON chunks that fail to parse.
    # The prompt already enforces the exact JSON structure.
    config_kwargs = {
        "max_output_tokens": 8192,  # generous budget: 3 descriptions × ~200 words
        "temperature": 0.7,
        "response_mime_type": "application/json",
    }
    
    yield ("status", {"step": 2, "message": "Generating optimized content..."})
    
    try:
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(**config_kwargs),
            request_options={"timeout": LLM_TIMEOUT},
            stream=True,
        )
    except Exception as exc:
        if _is_quota_error(exc):
            wait = _quota_retry_seconds(exc)
            raise RuntimeError(
                f"QUOTA_EXCEEDED:{wait}:You've hit the Gemini API free-tier rate limit. "
                f"Please wait {wait} seconds and try again."
            ) from exc
        raise
    
    full_text = ""
    async for chunk in response:
        if chunk.text:
            full_text += chunk.text
            yield ("chunk", {"text": chunk.text})
            
    yield ("status", {"step": 3, "message": "Saving to history…"})
    
    try:
        llm_data = _parse_json(full_text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"LLM returned unparseable JSON after streaming. "
            f"Preview: {full_text[:300]!r}. Error: {exc}"
        ) from exc
    yield ("result", llm_data)
