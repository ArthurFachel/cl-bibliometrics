# cl-bibliometrics

How did continual learning research move from small vision benchmarks to large language models, and who drove that shift?

A reproducible bibliometric dataset of continual learning and catastrophic forgetting research, built from [OpenAlex](https://openalex.org), with every number traceable to a dated API snapshot.

## Findings (snapshot 2026-10-19)

- **The field grew about 3x in three years.** Works where continual learning is the topic went from 761 first appearing in 2022 to 2,207 in 2025, and 2,543 in 2026 up to October 19.
- **LLMs took over the growth.** The share of those works that use LLMs, foundation models or parameter-efficient tuning rose from 2.5% in 2019 to 29% in 2025 and 39% in 2026. Replay and regularization held steady at about 30% each. The PEFT/LLM tag scores F1 0.84 on a held-out sample.
- **arXiv matters less than it did.** 57% of focused works were also on arXiv in 2019, against 36% in 2025. The broad corpus shows the same drop (54% -> 35%), so it isn't an artefact of the focus rule.
- **Keyword search over-collects.** 10-22% of works matched by the queries only mention continual learning in passing, depending on which labeller is asked. That's why all findings use a stricter focused corpus (precision 0.93-0.98 on held-out data).

![Growth](figures/growth.png)

![Sub-areas](figures/subareas.png)

## Status

| phase | question | state |
|---|---|---|
| F1 | How fast is the field growing, and how much of it lives on arXiv? | done |
| F2 | Which sub-areas drive the growth? | done: rule-based tags, validated on a held-out sample |
| F3 | Who collaborates with whom? Which institutions and countries are central? Where does Brazil sit? | not started |
| F4 | When did benchmarks migrate (Permuted MNIST -> Split CIFAR -> ImageNet-R -> LLM suites)? | not started |

## Corpus

| step | works |
|---|---|
| raw OpenAlex results, all queries | 24,264 |
| in scope (field, type, not retracted, lifelong-learning gate) | 15,088 |
| after deduplication (662 preprint/published duplicates merged) | 14,426 |
| focused: CL term in the title, or at least twice in the abstract | 9,879 |

Rules live in [`src/clbib/config.py`](src/clbib/config.py).

- **Queries** (title + abstract): `"continual learning" | "catastrophic forgetting" | "class-incremental" | "task-incremental" | "domain-incremental"`, plus `"lifelong learning"` restricted to Computer Science.
- **Lifelong-learning gate.** Unrestricted, `"lifelong learning"` returns about 55k works, mostly adult education. Even inside Computer Science it's dominated by e-learning and MOOCs: of the 7,002 works found only by this query, 603 pass the gate (an ML term, and either an AI / Computer Vision subfield or no education term).
- **Scope.** Primary field in Computer Science, Engineering, Neuroscience or Mathematics; article, conference paper, preprint, review or book chapter; not retracted.
- **Deduplication.** OpenAlex doesn't merge every arXiv preprint with its published version. Remaining pairs are clustered by normalized title and first-author surname within three years. One record is kept per cluster (in scope > published > most cited), and `first_year` is the earliest appearance.

### Growth

| first year | focused works | also on arXiv |
|---|---|---|
| 2017 | 46 | 54% |
| 2019 | 237 | 57% |
| 2022 | 761 | 50% |
| 2024 | 1,603 | 43% |
| 2025 | 2,207 | 36% |
| 2026 (to Oct 19) | 2,543 | 35% |

Full series: [`results/growth_by_year.csv`](results/growth_by_year.csv), [`results/growth_by_query.csv`](results/growth_by_query.csv).

### Recall

45 papers that a separate literature scout had confirmed as continual learning (Sep-Oct 2026) were looked up: 40 (89%) are in the corpus. All 5 misses are indexed in OpenAlex but never use the query phrases ("continual learners", "forgetting" without "catastrophic"). The check set was itself found by searching for CL, so 89% is an upper bound.

## Sub-areas

[`src/clbib/subareas.py`](src/clbib/subareas.py) tags every focused work with plain regexes, so each tag can be audited. Tags are independent: a paper can be replay + PEFT/LLM + medical.

- method: replay, regularization, architecture, optimization, PEFT/LLM, plasticity
- study type: theory, survey/benchmark
- setting: online, federated, RL
- domain: application outside ML

Share of focused works (with an abstract) per tag, for the tags that pass validation:

| first year | works | PEFT/LLM | replay | regularization | RL | federated |
|---|---|---|---|---|---|---|
| 2019 | 202 | 2.5% | 24% | 26% | 12% | 0.5% |
| 2022 | 651 | 7.1% | 32% | 29% | 6.0% | 3.2% |
| 2024 | 1,335 | 23% | 31% | 26% | 4.8% | 6.1% |
| 2025 | 1,841 | 29% | 30% | 29% | 4.2% | 6.6% |
| 2026 | 1,941 | 39% | 33% | 31% | 5.7% | 5.4% |

Full series: [`results/subarea_by_year.csv`](results/subarea_by_year.csv).

## Validation

The rules were scored against three LLM labellers (Claude, DeepSeek V4 Flash, Claude Sonnet 5) on two disjoint samples of 150 works. The rules were tuned on the dev sample, frozen in commit `098968f`, then scored once on the test sample. None of the labellers saw the rules. There is no human gold set, so these scores measure agreement with models.

Held-out F1 by tag:

| tag | support | Claude | DeepSeek | Sonnet | used for findings |
|---|---|---|---|---|---|
| federated | 5 | 0.91 | 0.93 | 0.93 | yes, but rare |
| rl | 12 | 0.86 | 0.88 | 0.83 | yes |
| peft_llm | 28 | 0.84 | 0.77 | 0.84 | yes |
| regularization | 43 | 0.74 | 0.78 | 0.82 | yes |
| replay | 40 | 0.78 | 0.73 | 0.73 | yes |
| application | 31 | 0.63 | 0.68 | 0.69 | no |
| online | 18 | 0.68 | 0.64 | 0.62 | no |
| survey_benchmark | 14 | 0.42 | 0.57 | 0.60 | no, undercounts |
| architecture | 37 | 0.54 | 0.44 | 0.48 | no, undercounts |
| theory | 4 | 0.29 | 0.22 | 0.18 | no, dev score was overfit |

A tag is used for findings only if its test F1 is at least 0.7 under all three labellers. The split doesn't depend on which model wrote the labels. Pairwise kappa between labellers is 0.83 or higher on the five tags used. Supports are small, so differences of a few points are noise.

Corpus precision, dev scores and full agreement tables: [`validation/README.md`](validation/README.md).

## Reproduce

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
export OPENALEX_API_KEY=...   # free key, https://openalex.org/settings/api
make all                       # fetch -> clean -> growth -> subareas -> validate
make test
```

Raw API pages are kept gzipped under `data/raw/openalex/<query>/` with a manifest (filter, counts, fetch date). `data/` isn't versioned; `results/`, `figures/` and the label files in `validation/` are.

```
src/clbib/
  config.py     corpus definition: queries, scope, focus rule, paths
  fetch.py      OpenAlex cursor pagination, resumable
  clean.py      parquet tables (works, authorships, references) and dedup
  growth.py     F1 tables and figure
  subareas.py   F2 rule-based tags
  validate.py   sample drawing and rule scoring
  llm_label.py  LLM labellers and pairwise agreement
tests/          parsing, dedup and tag rules
validation/     samples, label files and labelling guide
```

## Limitations

- Retrieval is keyword-based. Papers that never say "continual learning" or "catastrophic forgetting" (e.g. "model editing", "incremental learning" alone) are missed.
- Sub-area validation uses LLM labels only. Agreement across three labellers makes the tags more credible, but it isn't a human gold standard.
- OpenAlex fields, topics and institution data are machine-assigned.
- 2026 is a partial year.

## Author

Arthur Fachel, Data Analytics @ PUCRS, MALTA Lab. Research on neuron-level plasticity for continual learning.

## License

Code: MIT. Data: OpenAlex metadata is CC0.
