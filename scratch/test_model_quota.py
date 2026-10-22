
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = "gemini-flash-latest"

if not api_key:
    print("Error: GEMINI_API_KEY not found.")
else:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say hello in 5 words.")
        print(f"Success with {model_name}: {response.text}")
    except Exception as e:
        print(f"Error with {model_name}: {e}")
