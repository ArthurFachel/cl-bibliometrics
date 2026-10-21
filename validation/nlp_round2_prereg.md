# F4 round 2: NLP & LLM detector, pre-registration

Written and committed before the round-2 sample was drawn and before any round-2 label existed.

Why: round 1 (`benchmarks_to_label.csv`) gave NLP & LLM recall 0.19 (95% CI 0.09-0.37). While diagnosing it, the
11 missed round-1 papers were read, and they informed the new patterns. Round 1 is therefore not a clean
test for the extended detector; round 2 is.

Detector under test: `src/clbib/benchmarks.py` at the commit that adds this file.
- round 1: the original 20 names (`ROUND1_NAMES`), reported for comparison
- layer 1: round 1 plus six named-dataset groups (event/relation/NER, QA, LLM suites, captioning/multimodal,
  dialogue/generation/LM, GLUE tasks and intents)
- layer 1 + 2: plus the "Other named language dataset" heuristic

Works flagged NLP & LLM in the analysis population (focused, 2018-2026, with abstract, 7,591 works) at freeze
time: round 1 80, layer 1 131, layer 1 + 2 294.

Sample: focused works 2018-2026 with an abstract, excluding the 300 round-1 works. Strata, 100 each,
seed 20261023: `nlp` (layer 1 + 2 flags NLP & LLM), `peft` (not flagged, PEFT/LLM sub-area tag),
`other` (the rest). Labellers do not see strata or detector output.

Labels: the round-1 guide (all 8 families), Claude Opus 5.5, Claude Sonnet 5.5, DeepSeek V4 Flash; reference
is the majority vote. Precision and recall are re-weighted by stratum size, bootstrap 95% intervals.

Decision rule: layer 2 is used for findings (`NLP_LAYER2 = True`) only if its NLP & LLM precision against
the majority is >= 0.85. Otherwise findings use layer 1. Whatever the result, no pattern is changed after
round-2 labels are seen; a further change would need a round 3 on a new sample.

## Outcome

(filled in after scoring)
