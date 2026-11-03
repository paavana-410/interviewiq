# InterviewIQ — GenAI-Based Virtual Interviewer for Placement Coaching

Formative coaching only. No hireability prediction. No facial/emotion/accent assessment.

## Requirements

- Python 3.11+
- ~5 GB free RAM (base Whisper + pandas fits; see stage table below)
- No Docker required — SQLite is the default database
- PostgreSQL optional (set `DATABASE_URL=postgresql://...` in `.env`)

## Quick start

```powershell
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Copy and fill in your API keys
copy .env.example .env
# Edit .env — set JUDGE_1_API_KEY etc. (see .env.example for all variables)

# 3. Create the database (SQLite by default)
alembic upgrade head

# 4. Seed questions
python -m app.seed.questions

# 5. Verify judges are reachable
python check_config.py

# 6. Run tests
pytest
```

## Stage-based benchmark runner

Stages are designed for a laptop with ~5 GB free RAM.
**Run them in order.** Each stage loads only what it needs and exits cleanly.

| Stage | Command | What runs | Approx. RAM |
|---|---|---|---|
| `text` | `--stage text` | Hosted LLM judges (HTTP only). No Whisper, no Ollama. | ~300 MB |
| `local_judge` | `--stage local_judge` | Ollama local model only (1 model, ctx≤4096). | ~2–3 GB |
| `audio` | `--stage audio` | TTS + faster-whisper. Refuses to start if Ollama is loaded. | ~500 MB–1 GB |
| `analysis` | `--stage analysis` | Pandas + plots only, reads saved results. | ~200 MB |

```powershell
# Typical research session:

# Step 1 — score with hosted judges (uses your API keys, no GPU needed)
python run_audit_benchmark.py \
    --answers-dir data/benchmark_answers \
    --stage text \
    --judges judge_nemotron_ultra \
    --runs 3 \
    --tag pilot

# Step 2 — score with local Ollama judge (run after text stage)
python run_audit_benchmark.py \
    --answers-dir data/benchmark_answers \
    --stage local_judge \
    --runs 3 \
    --tag pilot

# Step 3 — transcribe audio (only if you have audio.wav files)
python run_audit_benchmark.py \
    --answers-dir data/benchmark_answers \
    --stage audio \
    --tag pilot

# Step 4 — regenerate plots from a previous run
python run_audit_benchmark.py \
    --stage analysis \
    --run-id 20261103T120000_pilot_text_xxxxxxxx
```

## Tags

| Tag | Logged to experiments.md | Notes |
|---|---|---|
| `debug` | No | Safe for testing; never logged |
| `mock` | Yes | Mock LLM; never cite in paper |
| `pilot` | Yes | Real calls; may cite as preliminary |
| `final` | Yes | Requires clean git working tree |

## Configuration (.env)

```env
LLM_PROVIDER=live          # "mock" for tests, anything else activates judges

# Judges (add JUDGE_1, JUDGE_2, ...)
JUDGE_1_NAME=judge_nemotron_ultra
JUDGE_1_PROVIDER=openai_compatible
JUDGE_1_BASE_URL=https://openrouter.ai/api/v1
JUDGE_1_API_KEY=sk-or-...
JUDGE_1_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
JUDGE_1_RPM=10

# Paraphrase (must be a different model from all judges)
PARAPHRASE_MODEL=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free

# Whisper — never default to medium/large on a laptop
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

# Ollama (local_judge stage)
OLLAMA_MODEL=llama3.2
OLLAMA_NUM_CTX=2048

# Database (SQLite default — no server needed)
DATABASE_URL=sqlite:///interviewiq.db
```

## Project structure

```
interviewiq/
  app/
    engines/      content.py, delivery.py, holistic.py
    audit/        5 counterfactual test modules + combiner
    services/     llm.py, stt.py, judge_config.py
    utils/        memory.py (RAM tracking)
    models/       SQLAlchemy ORM (SQLite + Postgres)
    analysis/     agreement.py, plots.py
    seed/         questions.py (10 questions + rubrics)
  data/
    benchmark_answers/   answer items (question.txt, answer.txt, rubric.json)
  docs/
    research_log.md      dated session log
    decisions.md         design decisions
    experiments.md       auto-updated run index
    threats_to_validity.md
    paper/               outline.md, related_work.md
  experiments/<run_id>/  results.csv, summary.json, config.json, memory.json, plots
  tests/                 54 automated tests
  run_audit_benchmark.py stage-based CLI
  check_config.py        verify judge connectivity
```

## Running tests

```powershell
pytest                          # all 54 tests (mock mode, SQLite, no API keys)
pytest tests/test_isolation.py  # structural isolation tests only
```

## Memory notes

- The `audio` stage loads faster-whisper and unloads it after each file.
  Peak RAM is ~500 MB–1 GB with `base` model.
- Never set `WHISPER_MODEL_SIZE=medium` or `large` on a 5 GB laptop.
- The `local_judge` stage keeps the Ollama model in the Ollama server process
  (not in Python). Python process itself uses ~300 MB.
- Stages log RSS start/peak/end to `experiments/<run_id>/memory.json`.
