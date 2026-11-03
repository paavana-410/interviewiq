# Design Decisions

Each entry records a decision, the alternatives considered, and the rationale.
Never delete entries — add superseding entries if a decision changes.

---

## D-001 — Construct separation: two independent engines, no shared inputs

**Date:** 2026-10-07 (initial design)
**Decision:** Content Engine receives transcript text only. Delivery Engine receives word timestamps only. They never share a data structure.
**Alternatives considered:**
- Single combined evaluator (simpler, but cannot isolate contamination).
- Soft separation with optional shared context (would require ablation to verify; we enforce it structurally instead).
**Rationale:** The paper's contribution is the audit method that detects cross-modal contamination. If the engines share inputs by design, the audit result is trivially uninterpretable. Structural enforcement (AST-level import tests, type-level signature tests) makes the claim verifiable by a reviewer.
**Evidence:** `tests/test_isolation.py` — 10 tests covering imports and call signatures.

---

## D-002 — LLM judge: temperature=0, structured JSON output, Pydantic validation

**Date:** 2026-10-07
**Decision:** All content evaluations use temperature=0. Output is validated with Pydantic. Retry on malformed JSON (up to 3 attempts).
**Alternatives considered:**
- Temperature >0 for diversity: rejected because it makes repeated-run consistency test impossible to interpret.
- Free-form text output: rejected because per-criterion scoring requires machine-readable output.
**Rationale:** Temperature=0 is standard for deterministic LLM evaluation (consistent with Chen et al. 2021, Zheng et al. 2023). Pydantic validation catches hallucinated schema deviations. Retry on parse failure prevents single bad responses from corrupting runs.

---

## D-003 — Disk cache keyed by (model, prompt_version, input_hash)

**Date:** 2026-10-07
**Decision:** Deterministic calls (temperature=0) are cached to disk. Cache key = SHA-256(model + ":" + prompt_version + ":" + prompt_text).
**Rationale:** Free-tier rate limits (50 req/day OpenRouter, ~500 req/day Gemini) would make re-running experiments prohibitive. Cache also makes repeated-run consistency test instant after the first run.
**Risk:** If cached responses are corrupted or stale, results will be wrong silently. Mitigation: cache is per prompt_version — bumping version invalidates all cached calls for that version.

---

## D-004 — Delivery invariance test: text-only judges are trivially invariant to timing

**Date:** 2026-10-07
**Decision:** Delivery variants (filler injection, pause insertion, pace change) modify word timestamps. For the delivery invariance test, the variant text is what gets sent to the content engine — filler words are injected into the text too.
**Rationale:** A text-only LLM judge would pass delivery invariance trivially if timing-only variants are used (timestamps never reach the judge). To make the test non-trivial, filler words from injection must appear in the transcript text. Pace and pause variants are timing-only and will always produce zero delta for a text-only judge — this is an honest finding (the invariance holds, trivially). The filler injection variant is the only one that can produce a non-trivial result for text-only judges.
**Implication for paper:** Be explicit that pace/pause variants test STT-side contamination (meaningful only when audio goes through Whisper), not text-judge contamination. Filler injection tests text-judge contamination directly.

---

## D-005 — Paraphrase model must differ from every judge model

**Date:** 2026-10-07
**Decision:** System refuses to start if PARAPHRASE_MODEL matches any JUDGE_N_MODEL. Validated at registry load time.
**Rationale:** If the same model generates the paraphrase and then scores it, the paraphrase stability test measures self-consistency of one model, not meaning-preservation across models. Using a different model is a minimum safeguard. (Human verification of paraphrases is the full safeguard — flagged with `needs_human_verification=True`.)

---

## D-006 — Free-tier judges chosen for budget; not ideal for final results

**Date:** 2026-10-07
**Decision:** Pilot runs use Gemini 3.8 Flash (free), Nemotron 120B/550B (free via OpenRouter), Apodex mini (free via OpenRouter).
**Limitation:** Free-tier models have rate limits (50 req/day OpenRouter, ~500 req/day Gemini) and intermittent 503s. Results may differ from GPT-4o or Claude. OpenAI credits exhausted — not usable currently.
**Plan for final runs:** Add GPT-4o or Claude once credits available. Label judge model IDs explicitly in all result tables.

---

## D-007 — Prompt version tracked in every LLM call log and cache key

**Date:** 2026-10-07
**Decision:** `CONTENT_ENGINE_PROMPT_VERSION=v1` in .env. Every call to the LLM logs this version. Cache key includes it.
**Rationale:** If the prompt changes between runs, results are not comparable. Version tracking makes the change detectable. Bumping the version invalidates the cache and forces re-evaluation.
**Freeze date:** Prompt v1 is in use. Do not modify the prompt in `app/engines/content.py` without bumping to v2 and logging the change in this file.
