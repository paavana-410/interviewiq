"""List available free models on OpenRouter."""
from dotenv import load_dotenv
load_dotenv()
import os, httpx

key = os.environ.get("JUDGE_2_API_KEY", "")
r = httpx.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {key}"},
    timeout=15,
)
models = r.json().get("data", [])
free = [
    m for m in models
    if ":free" in m.get("id", "")
    or (m.get("pricing", {}).get("prompt", "1") == "0"
        and m.get("pricing", {}).get("completion", "1") == "0")
]
print(f"Free models available ({len(free)} total):\n")
for m in sorted(free, key=lambda x: x["id"])[:30]:
    ctx = m.get("context_length", "?")
    print(f"  {m['id']:<60}  ctx={ctx}")
