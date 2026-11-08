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

### T-I-5 — content_sensitivity = 0 on the v1 pilot run was a no-op degradation artifact
**Threat:** On run `20261103T162824_pilot_text_82c8318b`, the content_sensitivity metric was 0.000 for all three items. The deep diagnostic (`diagnose_content_sensitivity.py --deep`) showed this was an artifact: the degradation used `text.replace(concept, ...)` with concept names (`LIFO`, `use cases`, `example`) that never appeared verbatim in the answer text, so the "degraded" text was byte-identical to the original. Both calls hit the same cache key and returned the same response. The metric therefore measured nothing.
**Mitigation (applied):**
- (a) `quality_degradation` now removes whole sentences (substring match or via an explicit `concept_sentences` rubric annotation) and uses hand-written `answer_medium.txt` / `answer_weak.txt` when present.
- (b) Every audit test now validates that the variant actually differs from the original (text hash / timing hash). A no-op variant is marked `invalid` and EXCLUDED from the metric — never reported as 0. Invalid-variant counts are recorded per test in `summary.json`.
- (c) The repeated-run test now bypasses the cache via a `run_index` in the cache key, so repeats are real calls (a unit test proves two repeats make two calls).
**Status:** ✅ Fixed. Run `20261103T162824_pilot_text_82c8318b` is INVALID for the content_sensitivity test; its other four tests remain valid. Any paper claim about content sensitivity must come from a run on the fixed code.
**Note for paper:** This is documented as a methods bug found and fixed during pilot testing, not a property of the judges. We do NOT claim the v1 judge was "insensitive" or that there was a "ceiling effect" — those were discarded hypotheses. The correct statement is: the degradation procedure was a no-op, so the test was uninformative on that run.

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
