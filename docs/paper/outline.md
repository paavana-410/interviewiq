# Paper Outline — InterviewIQ

**Working title:** "Construct-Separated Evaluation of Spoken Interview Responses: A Counterfactual Audit Framework for LLM-Based Placement Coaching"

**Target venue (options, check deadlines):**
- BEA Workshop (Building Educational Applications) — co-located with ACL/EMNLP
- AIED (Artificial Intelligence in Education) — full conference
- IEEE TALE (Teaching, Assessment, and Learning for Engineering)
- INTERSPEECH Education track
- Local/institutional student research conference (fastest path)

**Scope statement (put this in abstract and intro):**
Pilot study. English only. Synthetic + researcher-authored benchmark. Single deployment context. Results demonstrate audit method sensitivity, not deployment accuracy.

---

## Structure

### 1. Introduction (~500 words)
- Hook: placement interviews are high-stakes; students need formative feedback
- Gap: LLM judges used as black boxes; no mechanism to detect contamination between content and delivery dimensions
- Contribution: counterfactual audit framework with 5 independent tests
- Scope: formative coaching only; no hireability prediction

### 2. Related Work (~600 words)
- LLM-as-judge (Zheng et al. 2023 MT-Bench; Dubois et al. 2024 AlpacaEval)
- Interview coaching systems (prior work)
- Construct validity in automated assessment (Messick 1989; for reference)
- Rubric-based scoring (automated essay scoring literature)
- Evaluation robustness / calibration of LLMs

### 3. System Design (~700 words)
- Two-engine architecture (content + delivery)
- Structural isolation guarantee (cite test_isolation.py as evidence)
- LLM judge: rubric-driven, temperature=0, structured JSON
- Delivery Engine: deterministic, WPM/pauses/fillers/repetitions
- Figure: system architecture diagram

### 4. Counterfactual Audit Method (~900 words) ← THE CONTRIBUTION
- Why counterfactual auditing?
- 5 tests: delivery invariance, content sensitivity, paraphrase stability, repeated-run, STT sensitivity
- Variant generator: 5 procedures (seeded, documented)
- Uncertainty flag: combination rule
- Table: test name | hypothesis | metric | threshold | pass condition

### 5. Benchmark Dataset (~400 words)
- 10 questions: 3 HR, 3 behavioral, 4 technical
- Answer quality tiers: strong / medium / weak
- Synthetic vs real answers — label clearly
- Human annotation protocol (if done) or state it was not done

### 6. Experiments (~500 words)
- Judge models: list exact IDs and versions
- Prompt version: v1 (frozen date: ___)
- Thresholds: list all 5 (frozen date: ___)
- Runs: N items × M judges × 5 repeated runs
- Compute: free-tier API calls, latency

### 7. Results (~800 words)
- Table: per-judge audit pass rates (di, cs, ps, rr, stt)
- Figure: score delta distributions per perturbation type
- Figure: repeatability variance per judge
- Table: baseline comparison (holistic vs rubric vs rubric+audit)
- Human agreement: Kappa per criterion (if collected)
- Ablation: content-only vs content+delivery features

### 8. Discussion (~400 words)
- What the audit found: contamination level, which tests fired
- Which judges were most/least stable
- Limitations: small dataset, synthetic answers, no audio for STT test (if applicable)
- Threats to validity: cite threats_to_validity.md

### 9. Conclusion (~200 words)
- Restate contribution
- What's needed for larger study
- Code and data availability (GitHub link)

---

## Figures needed
1. System architecture (content + delivery engines, audit layer, DB)
2. Score delta box plots per perturbation type per judge
3. Repeatability scatter per judge
4. Audit pass/fail heatmap (test × judge)
5. Baseline comparison bar chart

## Tables needed
1. Dataset statistics (n items, quality distribution, question categories)
2. Judge model details (name, provider, model ID, RPM)
3. Audit results per judge (main results table)
4. Ablation results
5. Human agreement (if collected)
