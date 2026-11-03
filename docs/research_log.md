# Research Log — InterviewIQ

Entries are newest on top.
Rules:
- Log failures and negative results — they matter.
- Never edit old entries. Add corrections as new entries below.
- Only record a number if there is a run_id behind it.
- Tag every run: **mock** | **pilot** | **final**.
- Never cite mock runs in the paper.

---

## 2026-10-07

**Goal:** Set up research record-keeping infrastructure and get the first real-judge pilot run working today.

**Did:**
- Built full backend in mock mode over prior sessions (commits 9a38022 → a5170c2).
- Configured 4 real judges: Gemini 3.8 Flash, Nemotron 120B, Nemotron 550B, Apodex mini (all free tier via Gemini API and OpenRouter).
- Fixed critical bug: OpenAI SDK hung indefinitely on OpenRouter free models — replaced SDK transport with direct `httpx` calls (commit a5170c2).
- Set up docs/ research log structure (this file).
- Auto-logging in `run_audit_benchmark.py` — every run appends to `docs/experiments.md`, saves `git commit hash` + `pip freeze` into `config.json`.
- Added dirty-tree guard: runs tagged `final` fail if working tree has uncommitted changes.
- `check_config.py` passes 5/5 (4 judges + paraphrase).

**Result:**
- No final or pilot runs yet. Only mock runs exist (not citable).
- Judges verified reachable: Gemini 3.8 Flash (~10s latency, free-tier 503s intermittent), Nemotron 120B (~2s), Nemotron 550B (~1s), Apodex mini (~2s), Nemotron nano paraphrase (~1s).

**Problems:**
- Gemini 3.8 Flash returns HTTP 503 (high demand) approximately 30% of requests on free tier. Retries handle it but slow runs down.
- `gemini-2.5-flash` deprecated for new users — migrated to `gemini-3.8-flash`.
- OpenAI key has no credits — not usable until topped up. Excluded from judge set.
- OpenRouter free tier: 50 req/day on free account. Sufficient for pilot (3 items × 4 judges × ~8 calls = 96 calls). Need paid tier for full 30-item run.

**Decision:**
- Proceed with 3 existing benchmark items as a smoke-test pilot run today.
- Expand to 12–15 items before the first citable pilot run.
- Tag today's run as `pilot` — results are real but dataset is too small to cite as evidence.

**Next:**
- Run pilot on existing 3 items with all 4 judges (`python run_audit_benchmark.py --judges all --runs 3 --tag pilot`).
- Expand benchmark corpus to 12–15 items (strong/medium/weak tiers per question).
- Write `decisions.md` entry for content engine prompt design.
