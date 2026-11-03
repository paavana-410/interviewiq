# Related Work Notes

Notes for writing Section 2. Add citations as you find them.
Use format: [AuthorYear] — one-line summary — relevance to our paper.

---

## LLM-as-Judge

- [Zheng2023] "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" — introduced the LLM-as-judge paradigm; showed GPT-4 agreement with humans. **Cite as motivation for using LLM judges; our contribution is auditing them.**
- [Dubois2024] "AlpacaEval" — length-controlled win-rate using LLM judges. Showed position bias. **Cite as evidence that LLM judges have systematic biases we need to test for.**
- [Wang2023] "Large Language Models are not Robust Multiple Choice Selectors" — shows sensitivity to option ordering. **Supports our delivery invariance hypothesis.**

## Automated Interview Assessment

- (search: "automated interview assessment", "spoken response evaluation", "job interview AI")
- Need: at least 2–3 papers on prior automated interview tools.
- Note the gap: none of them audit construct separation.

## Construct Validity in Assessment

- [Messick1989] "Validity" in Educational Measurement — foundational. **Cite when defining "construct separation."**
- [Kane2006] "Validation" — modern validity argument framework. Optional.

## Rubric-Based Automated Scoring

- Automated Essay Scoring (AES) literature: e-rater, BERT-based scorers.
- STAR framework for behavioral interviews — cite the original HR/management literature.

## Evaluation Robustness

- [Saito2023] / similar — calibration of LLM evaluators.
- (search: "LLM evaluation robustness", "prompt sensitivity LLM judge")

---

## Gap Statement (draft)

Prior work on LLM-as-judge evaluates overall accuracy against human preference but does not test whether content and delivery dimensions contaminate each other in spoken response evaluation. Prior automated interview systems (cite 2–3) score responses holistically. We address the gap by introducing a counterfactual audit framework that tests construct separation explicitly.
