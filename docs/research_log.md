# Research Log — InterviewIQ

Entries are newest on top.
Rules:
- Log failures and negative results — they matter.
- Never edit old entries. Add corrections as new entries below.
- Only record a number if there is a run_id behind it.
- Tag every run: **mock** | **pilot** | **final**.
- Never cite mock runs in the paper.

---

## 2026-11-08

**Goal:** Diagnose why `content_sensitivity` was 0 on the v1 pilot run, and fix the audit layer so the metric is never an artifact.

**Did:**
- Built `diagnose_content_sensitivity.py --deep` which reads only from cache + results (no API). For run `20261103T162824_pilot_text_82c8318b`, item q01, it showed the original and degraded texts were byte-identical, the cache keys were identical, and one cached response served both. The concepts removed were `["LIFO", "use cases", "example"]` but none of those strings appeared verbatim in the answer (it said "Last In First Out", "used for ...", etc.), so the old `text.replace(concept, ...)` degradation was a no-op.
- Rewrote `quality_degradation` to remove whole sentences (substring match, or via an explicit `concept_sentences` annotation in rubric.json) and to use hand-written `answer_medium.txt` / `answer_weak.txt` when present.
- Added variant-validity checks: every audit test now compares the variant to the original (text hash / timing hash). A no-op variant is marked `invalid` and EXCLUDED from the metric — never reported as 0. Invalid-variant counts are printed per test and stored in `summary.json`.
- Fixed the repeated-run test: it passed `run_index=i` into the LLM cache key so each repeat is a real call, not a cache hit. Without this, all N repeats returned one cached response and variance was trivially 0.
- Added a known-groups (monotonicity) check: for items with strong/medium/weak answers, report whether the judge orders strong > medium > weak. A unit test proves it FAILS on a reversed-order judge.
- Wrote 11 new unit tests (degradation changes text, repeated-run makes N real calls, known-groups fails/passes/invalid). Updated 3 old tests whose single-sentence fixtures became no-ops under the new (correct) behavior.
- `pytest`: 65/65 pass. Mock benchmark run `20261108T125310_debug_text_11bfa4c5`: `content_sensitivity` now reports as a real measured 0.000 (mock judge is genuinely insensitive — it returns 0.775 for every input), not an artifact. `invalid_variants_total: 0`.

**Result:**
- The v1 `content_sensitivity = 0` on run `20261103T162824_pilot_text_82c8318b` was an artifact: the degradation was a no-op because the concept names never appeared verbatim in the answer text. **That run's content_sensitivity result is invalid for that test.** The other four tests on that run (delivery invariance, paraphrase, repeated-run, STT) were not affected by this bug and remain valid.
- The mock run confirms the fix: content_sensitivity is now a real measurement (mock = 0.000 because the mock cannot distinguish degraded from original), distinguishable from an invalid no-op (which would be excluded, not counted as 0).

**Problems:**
- The v1 pilot run is contaminated for content_sensitivity. Any future paper claim about content sensitivity must come from a run on the fixed code.
- The mock judge cannot be used to validate the known-groups check meaningfully: it returns equal scores for all tiers, so the check passes trivially via ties. Only a real judge (or a reverse-order mock, as in the unit test) exercises the check.

**Decision:**
- Treat run `20261103T162824_pilot_text_82c8318b` as INVALID for the content_sensitivity test. Keep it for the other tests. Do not delete it — the artifact is documented evidence for the threats-to-validity section.
- Re-run content_sensitivity on the fixed code before any paper claim. Use the hand-written medium/weak answers (already present for all 3 items) so the degradation is a real content change, not a string-substring guess.

**Next:**
- Re-run the text stage with a real judge once OpenRouter quota resets, to get a citable content_sensitivity number on the fixed code.
- Add `concept_sentences` annotations to the remaining rubrics for reproducible sentence-based degradation as a fallback when hand-written tiers are unavailable.

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
