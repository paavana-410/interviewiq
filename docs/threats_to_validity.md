# Threats to Validity

Record known threats honestly. Reviewers trust papers that name them.
Update as the project evolves.

---

## Internal Validity

### T-I-1 — LLM judge is also the source of ground truth
**Threat:** Content scores come from LLMs. There is no independent ground truth for most items. Human annotations are planned but not yet collected.
**Mitigation:** (a) Rubric-based scoring with explicit criteria reduces subjectivity. (b) Human annotations planned for 20–30 items (Phase 4). (c) Inter-annotator agreement will be reported.
**Status:** ⚠️ Partially mitigated — human study not yet done.

### T-I-2 — Delivery invariance test is trivially zero for text-only judges
**Threat:** Pace and pause variants change only timestamps. A text-only LLM judge will always produce delta=0 for these variants, making the test vacuous.
**Mitigation:** (a) Filler injection IS reflected in transcript text and can produce non-trivial deltas. (b) Pace/pause variants are only meaningful if audio goes through Whisper — this is documented in D-004. (c) Paper will explicitly separate filler-injection results from pace/pause results.
**Status:** ⚠️ Acknowledged — must be stated clearly in paper.

### T-I-3 — Paraphrase human verification not done
**Threat:** LLM-generated paraphrases are flagged `needs_human_verification=True` but verification has not been performed. Paraphrases may not preserve meaning.
**Mitigation:** Planned human review of all paraphrases used in final runs. Until then, paraphrase stability results are preliminary.
**Status:** ❌ Not mitigated yet.

### T-I-4 — Prompt version frozen at v1; no systematic prompt sensitivity analysis
**Threat:** Results may be sensitive to the exact wording of the rubric judge prompt. No ablation over prompt variants has been done.
**Mitigation:** Prompt version is tracked in every call log and cache key. A future extension could test multiple prompt phrasings.
**Status:** ⚠️ Acknowledged as a limitation.

---

## External Validity

### T-E-1 — Pilot dataset is synthetic / researcher-authored
**Threat:** Benchmark answers written by the researcher may not reflect real student responses. Vocabulary, length, and error patterns may differ.
**Mitigation:** (a) Future data collection from real students. (b) Clearly label all synthetic answers in the paper. (c) Scope the claim: "we evaluate the audit method's sensitivity properties, not its deployment accuracy."
**Status:** ⚠️ Acknowledged — must be stated in scope limitations.

### T-E-2 — Single language (English), single institution context
**Threat:** Results may not generalise to other languages or institutional contexts.
**Mitigation:** State explicitly in paper. Scope as a pilot study.
**Status:** ✅ Acknowledged and will be stated.

### T-E-3 — Free-tier LLM judges may differ from production models
**Threat:** Nemotron, Gemini free tier, and Apodex mini may behave differently from GPT-4o or Claude 3.5 Sonnet. The audit method's performance may vary by judge.
**Mitigation:** (a) Multi-judge design: results reported per judge. (b) Once OpenAI credits available, add GPT-4o as a reference judge. (c) Multi-judge comparison IS a contribution — different models showing similar audit patterns strengthens the claim.
**Status:** ⚠️ Partially mitigated.

---

## Construct Validity

### T-C-1 — "Contamination" is operationalised as score delta, not causal
**Threat:** A non-zero score delta when only delivery changes could have other causes (e.g., filler words changing perceived competence semantically).
**Mitigation:** This is the right operationalisation for a text-based judge. Fillers in transcript text ARE semantic signals. The claim is: "if a judge is truly content-only, filler-injected text should not change the score." Acknowledged as an approximation.
**Status:** ✅ Operationalisation is defensible and will be explained.

### T-C-2 — Thresholds (0.08, 0.15, 0.10, 0.02, 0.08) are researcher-chosen
**Threat:** Results depend on threshold choices. Choosing thresholds after seeing results is p-hacking.
**Mitigation:** (a) Thresholds are documented in `decisions.md`. (b) Thresholds must be frozen before final runs — log the freeze date here.
**Freeze status:** ❌ NOT YET FROZEN. Freeze before any run tagged `final`.
**Freeze date:** ___________ (fill in when frozen)

---

## Reporting

### T-R-1 — Only mock runs exist so far
**Threat:** No citable results exist yet. All numbers in the codebase come from the mock LLM.
**Status:** ❌ Will be resolved by Phase 3 pilot runs.
