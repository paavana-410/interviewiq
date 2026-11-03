# Experiment Index

One line per run. Appended automatically by `run_audit_benchmark.py`.
**Never cite mock runs.** Only pilot and final runs produce citable evidence.

Format: `run_id | date | tag | judges | n_items | prompt_v | git_hash | di_pass% | cs_pass% | rr_pass% | ps_pass% | notes`

---

<!-- EXPERIMENT_LOG_START — do not remove this line, used by run_audit_benchmark.py -->
| run_id | date | tag | judges | n_items | prompt_v | git_hash | di_pass% | cs_pass% | ps_pass% | rr_pass% | overall_pass% | status | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261103T100655_mock_43051974 | 2026-11-03 | **mock** | mock | 3 | vv1 | `20772a59` | 100% | 0% | 100% | 100% | 86% | **OK** | first smoke test, mock LLM only |
| 20261103T101739_pilot_c1450554 | 2026-11-03 | **pilot** | judge_nemotron/judge_nemotron_ultra | 3 | vv1 | `15afdff5` | 100% | 0% | 100% | 100% | 86% | |
| 20261103T161326_pilot_text_498db2a2 | 2026-11-03 | **pilot** | [text] judge_nemotron_ultra | 3 | vv1 | `5dc09eb0` | — | — | — | — | — | **FAILED** (failed: judge_nemotron_ultra) |
| 20261103T162131_pilot_text_94a0c18b | 2026-11-03 | **pilot** | [text] judge_nemotron_ultra | 3 | vv1 | `5dc09eb0` | — | — | — | — | — | **FAILED** (failed: judge_nemotron_ultra) |
| 20261103T162824_pilot_text_82c8318b | 2026-11-03 | **pilot** | [text] judge_nemotron_ultra | 3 | vv1 | `5dc09eb0` | 100% | 0% | 67% | 100% | 81% | **OK** |
<!-- EXPERIMENT_LOG_END — do not remove this line -->
