"""Test Gemini key via native endpoint and OpenAI-compat endpoint."""
from dotenv import load_dotenv
load_dotenv()
import os, httpx

key = os.environ.get("JUDGE_1_API_KEY", "")
print(f"Key length: {len(key)}, prefix: {key[:10]}, suffix: {key[-6:]}\n")

# Test 1: Native endpoint with ?key= query param
print("--- Test 1: Native generateContent endpoint ---")
url1 = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"
body1 = {"contents": [{"parts": [{"text": "Say OK"}]}]}
try:
    r = httpx.post(url1, json=body1, timeout=15)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text[:300]}\n")
except Exception as e:
    print(f"Error: {e}\n")

# Test 2: OpenAI-compat endpoint with key in URL
print("--- Test 2: OpenAI-compat with ?key= in URL ---")
url2 = f"https://generativelanguage.googleapis.com/v1beta/openai/chat/completions?key={key}"
body2 = {
    "model": "gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Say OK"}],
    "max_tokens": 10
}
try:
    r = httpx.post(url2, json=body2,
                   headers={"Authorization": "Bearer not-used", "Content-Type": "application/json"},
                   timeout=15)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text[:300]}\n")
except Exception as e:
    print(f"Error: {e}\n")

# Test 3: OpenAI-compat with x-goog-api-key header
print("--- Test 3: OpenAI-compat with x-goog-api-key header ---")
url3 = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
try:
    r = httpx.post(url3, json=body2,
                   headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                   timeout=15)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text[:300]}\n")
except Exception as e:
    print(f"Error: {e}\n")
