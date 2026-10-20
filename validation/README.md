# Validation (F2)

The sub-area rules in [`src/clbib/subareas.py`](../src/clbib/subareas.py) are scored against labels written by three LLM labellers. There is no human gold set, so every score below is agreement with a model, not ground truth.

## Samples

| split | file | size | drawn from | used for |
|---|---|---|---|---|
| dev | `to_label.csv` | 150 | in-scope corpus, seed 20261019, 50 per period (2015-2021, 2022-2024, 2025-2026) | diagnosing and tuning the rules |
| test | `test_to_label.csv` | 150 | same strata, disjoint from dev | scored once, after the rules were frozen in commit `098968f` |

Rule predictions are left out of both files so a labeller cannot be anchored by them.

## Labellers

All three see only the title, the abstract and the guide below. None of them sees the rules or another labeller's output.

| name | files | model | notes |
|---|---|---|---|
| Claude | `labels_llm.csv`, `labels_llm_test.csv` | Claude, six independent sessions of 25 rows each | the original silver labels; doubtful calls explained in `notes` |
| DeepSeek | `labels_deepseek.csv`, `labels_deepseek_test.csv` | `deepseek-ai/DeepSeek-V4-Flash` on DeepInfra, temperature 0 | different model family |
| Sonnet | `labels_sonnet.csv`, `labels_sonnet_test.csv` | `claude-sonnet-5`, one-shot call with no memory or tools | same family as Claude, so Claude-Sonnet agreement is not independent evidence |

One dev row (*Continual Learning in Task-Oriented Dialogue Systems*) has a boilerplate abstract. Its Claude labeller used prior knowledge of the paper instead of the text, so its method labels are unreliable.

## Guide

Each column is 1 when the paper...

| column | ... |
|---|---|
| is_cl | is actually about continual / lifelong / incremental learning or catastrophic forgetting (0 = passing mention or off-topic) |
| replay | stores, generates or replays past samples or features |
| regularization | constrains updates by importance weights (EWC, SI, MAS), distillation or other loss penalties |
| architecture | proposes a new structure: grows the network, isolates parameters with masks, adds experts or task-specific modules. Papers that only analyse existing components do not count |
| optimization | projects or constrains gradients (orthogonal, null space, GEM) or targets the loss landscape |
| peft_llm | works with LLMs, VLMs or pre-trained foundation models, or uses LoRA, prompts or adapters |
| plasticity | studies loss of plasticity, dormant neurons, resets or trainability |
| theory | gives a formal or theoretical analysis (bounds, proofs, analytic models) |
| survey_benchmark | is a survey or review, or contributes a benchmark or dataset |
| online | uses online, task-free or streaming settings, or concept drift |
| federated | uses federated learning |
| rl | uses reinforcement learning or agents |
| application | applies CL to a named domain outside ML (medical, fault diagnosis, remote sensing, speech...) |

Labels are multi-label. The "proposes" wording for `architecture` was added after the Claude labels were written. DeepSeek and Sonnet got the stricter wording, which is part of why their architecture support differs.

## Corpus precision (`is_cl`)

| labeller | not CL, dev + test (of 300) | broad corpus precision, test | focus rule precision, test | focus rule recall, test |
|---|---|---|---|---|
| Claude | 63 (21%) | 0.81 | 0.94 | 0.93 |
| Sonnet | 66 (22%) | 0.83 | 0.93 | 0.90 |
| DeepSeek | 30 (10%) | 0.90 | 0.98 | 0.87 |

The focus rule is a CL term in the title, or at least two in the abstract ([`config.py`](../src/clbib/config.py)). It was tuned on dev against Claude (precision 0.77 -> 0.95, recall 0.87) and holds on test under all three labellers. DeepSeek reads "about CL" more loosely: in 34 of the 35 Claude-DeepSeek disagreements on `is_cl`, Claude says "passing mention" and DeepSeek says CL.

## Rule scores

F1 of the rules, computed on the works each labeller marked as CL. Support is Claude's; the other labellers differ by a few works.

| tag | support (test) | test Claude | test DeepSeek | test Sonnet | dev Claude | dev DeepSeek | dev Sonnet |
|---|---|---|---|---|---|---|---|
| federated | 5 | 0.91 | 0.93 | 0.93 | 1.00 | 1.00 | 1.00 |
| rl | 12 | 0.86 | 0.88 | 0.83 | 0.89 | 0.94 | 1.00 |
| peft_llm | 28 | 0.84 | 0.77 | 0.84 | 0.83 | 0.86 | 0.88 |
| regularization | 43 | 0.74 | 0.78 | 0.82 | 0.70 | 0.78 | 0.75 |
| replay | 40 | 0.78 | 0.73 | 0.73 | 0.74 | 0.82 | 0.77 |
| application | 31 | 0.63 | 0.68 | 0.69 | 0.78 | 0.72 | 0.68 |
| online | 18 | 0.68 | 0.64 | 0.62 | 0.51 | 0.55 | 0.56 |
| survey_benchmark | 14 | 0.42 | 0.57 | 0.60 | 0.58 | 0.86 | 0.69 |
| architecture | 37 | 0.54 | 0.44 | 0.48 | 0.59 | 0.51 | 0.55 |
| theory | 4 | 0.29 | 0.22 | 0.18 | 0.89 | 0.82 | 0.83 |
| optimization | 4 | 0.89 | 0.67 | 0.89 | 0.62 | 0.50 | 0.73 |
| plasticity | 0 | - | - | - | - | - | - |

Dev is not held-out for any labeller, because the rules were tuned on it. Theory is the clearest case of overfitting: 0.89 on dev, under 0.3 on test. Optimization and plasticity have too few positives to score.

## Agreement between labellers

Cohen's kappa on the test sample (150 works). Dev is in `results/agreement_labellers_dev.csv`.

| tag | Claude-DeepSeek | Claude-Sonnet | DeepSeek-Sonnet |
|---|---|---|---|
| is_cl | 0.63 | 0.84 | 0.69 |
| rl | 1.00 | 0.97 | 0.97 |
| federated | 0.92 | 0.85 | 0.93 |
| replay | 0.87 | 0.83 | 0.90 |
| regularization | 0.83 | 0.83 | 0.88 |
| peft_llm | 0.84 | 0.92 | 0.88 |
| optimization | 0.85 | 1.00 | 0.85 |
| application | 0.74 | 0.67 | 0.77 |
| online | 0.72 | 0.73 | 0.64 |
| survey_benchmark | 0.70 | 0.70 | 0.67 |
| architecture | 0.63 | 0.76 | 0.72 |
| theory | 0.60 | 0.74 | 0.69 |
| plasticity | 0.00 | 0.00 | 0.27 |

Plasticity has 0-4 positives per labeller, so its kappa means nothing.

## Reproduce

```bash
make validate                  # rescore the rules and recompute kappa from the label files here
```

New labels (needs `DEEPINFRA_API_KEY`; the Sonnet backend shells out to the `hermes` CLI and is not portable):

```bash
PYTHONPATH=src .venv/bin/python -m clbib.llm_label label --src to_label.csv --out labels_deepseek.csv
PYTHONPATH=src .venv/bin/python -m clbib.llm_label label --src test_to_label.csv --out labels_deepseek_test.csv
```

Outputs: `results/validation_{llm,deepseek,sonnet}_{dev,test}.csv` and `results/agreement_labellers_{dev,test}.csv`.
