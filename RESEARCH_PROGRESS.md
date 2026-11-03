# InterviewIQ — Research Progress

> Last updated: 2026-10-07

## What is done

| Component | Commit | Status |
|---|---|---|
| Project scaffold (FastAPI, Alembic, Docker, pytest) | 9a38022 | ✅ |
| 11 DB models with JSONB audit fields | 9a38022 | ✅ |
| STT service — faster-whisper + MockSTT interface | 9a38022 | ✅ |
| Delivery Engine — WPM, pauses, fillers, repetitions (deterministic) | 9a38022 | ✅ |
| Content Engine — LLM rubric judge, Pydantic validation, retry | 9a38022 | ✅ |
| Holistic baseline evaluator (single opaque score) | 9a38022 | ✅ |
| Counterfactual Audit Layer — 5 independent test modules | 9a38022 | ✅ |
| Variant Generator — 5 seeded procedures | 9a38022 | ✅ |
| Audit Combiner — uncertainty flag | 9a38022 | ✅ |
| API endpoints (POST /sessions, POST /responses, GET /evaluation) | 9a38022 | ✅ |
| CLI benchmark run_audit_benchmark.py | 9a38022 | ✅ |
| Analysis scripts (Kappa, F1, bootstrap CI, plots) | 9a38022 | ✅ |
| Seed data — 10 questions (HR/behavioral/technical) + rubrics | 9a38022 | ✅ |
| Multi-judge LLM layer (4 judges, rate limiting, disk cache) | c3a1dc8 | ✅ |
| Key masking, httpx transport (no SDK hanging) | a5170c2 | ✅ |
| 54 automated tests (delivery, isolation, content, audit, API) | a5170c2 | ✅ |
| Judges verified working: Gemini 3.8, Nemotron 120B/550B, Apodex | a5170c2 | ✅ |

## What is next (Phase 2+)

### Phase 2 — Benchmark Corpus
- [ ] 30–50 answer items across all 10 questions
- [ ] Three quality tiers: strong / medium / weak (for content sensitivity)
- [ ] Gold transcripts for STT sensitivity subset

### Phase 3 — Experiment Runs
- [ ] `python run_audit_benchmark.py --judges all --runs 5`
- [ ] Results committed under `experiments/`

### Phase 4 — Human Annotations
- [ ] 2–3 annotators score 20–30 responses via API
- [ ] `POST /responses/{id}/annotations`

### Phase 5 — Analysis
- [ ] Cohen's Kappa (human vs LLM per criterion)
- [ ] Weighted F1 per rubric criterion
- [ ] Bootstrap confidence intervals
- [ ] Score-delta plots, audit summary plots

### Phase 6 — Paper
- [ ] Draft sections 4–7 (dataset, audit method, experiments, results)

## Experiment log

| Run ID | Date | Judges | Items | Notes |
|---|---|---|---|---|
| (none yet) | — | — | — | Waiting for benchmark corpus |

## Key design decisions (for paper methods section)

- **Structural isolation:** Content engine never imports delivery engine (enforced by AST tests)
- **Seeded variants:** All perturbations use `seed=42` by default — fully reproducible
- **Temperature 0:** All content evaluations use temperature=0 for determinism
- **Disk cache:** Identical (model, prompt_version, input_hash) → cached response; audit tests that need variance disable cache
- **Uncertainty flag:** Set if ANY of the 5 audit tests fails its threshold
- **Paraphrase guard:** Paraphrase model must differ from every judge model (validated at startup)
