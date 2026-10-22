# Project Documentation: ContentBoost AI (v3.0)

## 1. Project Overview
**ContentBoost AI** is a production-grade e-commerce optimization engine. It uses Large Language Models (LLMs) to transform basic product data into high-converting, SEO-optimized product descriptions.

### Core Value Proposition
- **Multi-Tone Generation**: Generates SEO, Marketing, and Technical versions of descriptions.
- **Competitive Intelligence**: Scrapes and analyzes competitor URLs to identify writing patterns and missing features.
- **Real-Time SEO Scoring**: Computes readability, keyword density, and Flesch-Kincaid scores locally (not just AI-estimated).
- **History & Comparison**: Allows users to save, export, and compare different versions of descriptions.

---

## 2. Technical Architecture
The system follows a modern **FastAPI (Backend) + Vanilla JS (Frontend)** architecture designed for speed and real-time feedback.

### Backend (Python/FastAPI)
- **`backend/main.py`**: The API gateway. Handles routing, authentication, and the Server-Sent Events (SSE) streaming logic.
- **`backend/llm_client.py`**: The AI core. Manages communication with Google Gemini, including advanced JSON repair and quota management.
- **`backend/database.py` & `memory.py`**: Asynchronous persistence layer using `aiosqlite`.
- **`backend/seo_analyzer.py`**: Local NLP module using `nltk` and `textstat` to calculate real SEO metrics.
- **`backend/auth.py`**: Secure user management using JWT (JSON Web Tokens) and `bcrypt` password hashing.

### Frontend (HTML/CSS/JS)
- **`app.js`**: Orchestrates the UI state and handles the SSE stream (typing effect).
- **`api.js`**: A clean wrapper for all fetch calls, including smart error normalization for AI rate limits.
- **`ui.js`**: Manages complex DOM updates, chart rendering, and keyword highlighting.
- **`style.css`**: A premium, custom CSS design system using dark mode and glassmorphism.

---

## 3. Key Technical Features & Innovation

### ⚡ Server-Sent Events (SSE) Streaming
Unlike simple apps that make you wait 20 seconds for a response, ContentBoost AI streams the AI's "thoughts" token-by-token. 
- **The Challenge**: Gemini's streaming JSON can be truncated or malformed mid-stream.
- **The Solution**: We implemented a custom **JSON Repair Layer** (`_repair_json`) that automatically closes unclosed brackets and strings to ensure the UI never crashes.

### 🛡️ Smart Quota Management (429 Handling)
To survive on the Gemini Free Tier, the system includes:
- **Detection**: Automatically identifies "Quota Exceeded" errors from the API.
- **UI Countdown**: Instead of a generic error, users see a live amber countdown bar showing exactly when they can try again.
- **Auto-Retry**: The system automatically resubmits the request the moment the rate limit expires.

### 🔍 Local SEO Intelligence
We don't trust the AI to "guess" its own SEO score. The backend runs a real NLP pipeline:
1. **Readability**: Uses the Flesch Reading Ease formula.
2. **Keyword Density**: Scans for exact and partial matches in the generated text.
3. **Actionable Tips**: Merges AI suggestions with local algorithmic warnings (e.g., "Description too short" or "Title exceeds 70 chars").

---

## 4. Technology Stack Summary
- **LLM**: Google Gemini 2.0 Flash (Optimized for speed and structured output).
- **Framework**: FastAPI (Async Python).
- **Database**: SQLite (via `aiosqlite`).
- **Security**: JWT Authentication, bcrypt, and Pydantic validation.
- **DevOps**: Uvicorn with auto-reload, python-dotenv for secrets.

---

## 5. Development Timeline (Recent Improvements)
- **v1.0**: Basic generation and local storage.
- **v2.0**: Added competitor scraping, local SEO metrics, and streaming.
- **v3.0 (Current)**: Added full Multi-User Auth, SQLite persistence, advanced JSON error recovery, and friendly Rate Limit handling.
