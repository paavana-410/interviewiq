"""Quick probe — find which OpenRouter free models respond within 15s."""
from dotenv import load_dotenv
load_dotenv()
import os, httpx, json, time

key = os.environ.get("JUDGE_2_API_KEY", "")

# candidates from our earlier list (skip agentic-only, skip audio/music)
candidates = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "poolside/laguna-s-2.1:free",
    "poolside/laguna-xs-2.1:free",
    "nvidia/nemotron-3.5-lightning:free",
    "apodex/apodex-1.1-mini:free",
    "liquid/lfm-2.5-2.6b:free",
]

prompt = 'Reply with exactly: {"status":"ok"}'

for model in candidates:
    url = "https://openrouter.ai/api/v1/chat/completions"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 20}
    t0 = time.monotonic()
    try:
        r = httpx.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=15)
        ms = int((time.monotonic() - t0) * 1000)
        if r.status_code == 200:
            print(f"  OK  {ms:5}ms  {model}")
        else:
            data = r.json()
            msg = data.get("error", {}).get("message", "")[:60]
            print(f"  {r.status_code}  {ms:5}ms  {model}  — {msg}")
    except httpx.TimeoutException:
        print(f"  TIMEOUT        {model}")
    except Exception as e:
        print(f"  ERR            {model}  — {e}")
