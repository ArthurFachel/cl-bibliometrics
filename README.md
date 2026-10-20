# cl-bibliometrics

How did continual learning research move from small vision benchmarks to large language models, and who drove that shift?

A reproducible bibliometric dataset of continual learning and catastrophic forgetting research, built from [OpenAlex](https://openalex.org), with every number traceable to a dated API snapshot.

## Findings (snapshot 2026-10-19)

- **The field grew about 2.8x in three years.** Works where continual learning is the topic went from 756 first appearing in 2022 to 2,112 in 2025, and 2026 had already passed that (2,133) by October 19.
- **LLMs took over the growth.** The share of those works that use LLMs, foundation models or parameter-efficient tuning rose from 2.5% in 2019 to 29% in 2025 and 39% in 2026. Replay and regularization held steady at about 30% each. The PEFT/LLM tag scores F1 0.84 on a held-out sample.
- **China now writes half of it.** Among works with a known affiliation, China's share rose from 19% (2015-2019) to 53% (2023-2026), and the US share fell from 40% to 18%. Industry co-authorship fell from 24% of works in 2019 to 8% in 2026, but PEFT/LLM works are 1.7x more likely than the rest to have a company author.
- **Benchmarks moved from MNIST to pre-trained models, and only lately to LLMs.** Among works that name or cite a benchmark, MNIST-scale datasets went from 47% in 2018 to 13% in 2023. Pre-trained-era benchmarks (ImageNet-R, CUB, DomainNet, VTAB) rose from 6% to 25% in 2025. Language benchmarks stayed under 5% until 2024 and reached 11% in 2026. CIFAR stays the default throughout.
- **arXiv matters less than it did.** 57% of focused works were also on arXiv in 2019, against 38% in 2025. The broad corpus shows the same drop (54% -> 37%), so it isn't an artefact of the focus rule.
- **Brazil is marginal.** 32 focused works since 2015 (rank 30, under 0.5% of the field), led by UNICAMP and UFPE.
- **Keyword search over-collects.** 10-22% of works matched by the queries only mention continual learning in passing, depending on which labeller is asked. That's why all findings use a stricter focused corpus (precision 0.93-0.98 on held-out data).

None of these findings changes when works that exist only on deposit repositories such as Zenodo are put back in (see [Sensitivity](#sensitivity-repository-only-works)). Only the 2026 volume does.

![Growth](figures/growth.png)

![Sub-areas](figures/subareas.png)

## Status

| phase | question | state |
|---|---|---|
| F1 | How fast is the field growing, and how much of it lives on arXiv? | done |
| F2 | Which sub-areas drive the growth? | done: rule-based tags, validated on a held-out sample |
| F3 | Who collaborates with whom? Which institutions and countries are central? Where does Brazil sit? | first pass: countries, industry, institution network, Brazil |
| F4 | When did benchmarks migrate (Permuted MNIST -> Split CIFAR -> ImageNet-R -> LLM suites)? | first pass: abstract mentions + dataset citations |

## Corpus

| step | works |
|---|---|
| raw OpenAlex results, all queries | 24,264 |
| in scope (field, type, not retracted, lifelong-learning gate) | 15,088 |
| after deduplication (662 preprint/published duplicates merged) | 14,426 |
| without repository-only works (1,000 removed) | 13,426 |
| focused: CL term in the title, or at least twice in the abstract | 9,326 |

Rules live in [`src/clbib/config.py`](src/clbib/config.py); the analysis corpus is defined once in [`src/clbib/corpus.py`](src/clbib/corpus.py).

- **Queries** (title + abstract): `"continual learning" | "catastrophic forgetting" | "class-incremental" | "task-incremental" | "domain-incremental"`, plus `"lifelong learning"` restricted to Computer Science.
- **Lifelong-learning gate.** Unrestricted, `"lifelong learning"` returns about 55k works, mostly adult education. Even inside Computer Science it's dominated by e-learning and MOOCs: of the 7,002 works found only by this query, 603 pass the gate (an ML term, and either an AI / Computer Vision subfield or no education term).
- **Scope.** Primary field in Computer Science, Engineering, Neuroscience or Mathematics; article, conference paper, preprint, review or book chapter; not retracted.
- **Deduplication.** OpenAlex doesn't merge every arXiv preprint with its published version. Remaining pairs are clustered by normalized title and first-author surname within three years. One record is kept per cluster (in scope > published > most cited), and `first_year` is the earliest appearance.
- **Repository-only works.** A work is dropped when every source it appears in, across all its OpenAlex locations and every record of its dedup cluster, is a general-purpose deposit repository (Zenodo, SSRN, Research Square, Preprints.org, OSF Preprints, Figshare, Authorea, Qeios, TechRxiv). These accept any upload without moderation. arXiv, bioRxiv/medRxiv and institutional repositories are kept, and so is any deposit that also appears in a journal, conference or on arXiv. Of the focused works removed this way, 410 are from 2026 and 26 from 2024.

### Growth

| first year | focused works | also on arXiv |
|---|---|---|
| 2017 | 45 | 56% |
| 2019 | 237 | 57% |
| 2022 | 756 | 51% |
| 2024 | 1,577 | 44% |
| 2025 | 2,112 | 38% |
| 2026 (to Oct 19) | 2,133 | 42% |

The 2026 arXiv share is probably inflated: many 2026 preprints haven't been matched to a published version yet. Full series: [`results/growth_by_year.csv`](results/growth_by_year.csv), [`results/growth_by_query.csv`](results/growth_by_query.csv).

### Recall

45 papers that a separate literature scout had confirmed as continual learning (Sep-Oct 2026) were looked up: 40 (89%) are in the corpus. All 5 misses are indexed in OpenAlex but never use the query phrases ("continual learners", "forgetting" without "catastrophic"). The check set was itself found by searching for CL, so 89% is an upper bound.

## Sub-areas (F2)

[`src/clbib/subareas.py`](src/clbib/subareas.py) tags every focused work with plain regexes, so each tag can be audited. Tags are independent: a paper can be replay + PEFT/LLM + medical.

- method: replay, regularization, architecture, optimization, PEFT/LLM, plasticity
- study type: theory, survey/benchmark
- setting: online, federated, RL
- domain: application outside ML

Share of focused works (with an abstract) per tag, for the tags that pass validation:

| first year | works | PEFT/LLM | replay | regularization | RL | federated |
|---|---|---|---|---|---|---|
| 2019 | 202 | 2.5% | 24% | 26% | 12% | 0.5% |
| 2022 | 650 | 7.1% | 32% | 29% | 6.0% | 3.2% |
| 2024 | 1,327 | 23% | 31% | 26% | 4.8% | 6.1% |
| 2025 | 1,795 | 29% | 30% | 29% | 4.2% | 6.7% |
| 2026 | 1,619 | 39% | 34% | 31% | 5.7% | 6.0% |

Full series: [`results/subarea_by_year.csv`](results/subarea_by_year.csv).

## Who: countries, industry, institutions (F3)

[`src/clbib/networks.py`](src/clbib/networks.py) uses OpenAlex affiliations, whole counting (a CN-US paper counts for both). 61-90% of focused works per year since 2018 have at least one known country ([`results/coverage_by_year.csv`](results/coverage_by_year.csv)); shares below are over those works.

![Countries](figures/countries.png)

| country | works 2015-2026 | 2015-2019 | 2020-2022 | 2023-2026 |
|---|---|---|---|---|
| China | 3,652 | 19% | 31% | 53% |
| United States | 1,633 | 40% | 30% | 18% |
| United Kingdom | 502 | 9.7% | 8.7% | 5.8% |
| Italy | 340 | 3.5% | 6.3% | 4.0% |
| Germany | 338 | 7.1% | 6.3% | 3.7% |
| South Korea | 337 | 7.7% | 5.5% | 3.9% |
| India | 336 | 1.9% | 4.2% | 4.5% |
| Brazil (rank 30) | 32 | 0.3% | 0.3% | 0.5% |

- **Industry.** Works with at least one company author: 24% in 2019, 15% in 2023, 8% in 2026. In 2023-2026, 16% of PEFT/LLM works have a company author against 9% of the rest. China (1.2x) and Hong Kong (2.2x) are also over-represented in PEFT/LLM work ([`results/peft_llm_by_country.csv`](results/peft_llm_by_country.csv)).
- **International co-authorship** is flat at about 25% of works since 2018.
- **Institution network.** 2,777 institutions linked by co-authored works (11,441 links; works with more than 20 institutions skipped). 81% sit in one connected component. Louvain finds 22 communities (modularity 0.53) that mostly follow geography: two large Chinese clusters (one around Tsinghua and Harbin, one around the Chinese Academy of Sciences), a US-led cluster, a French one and an Australian one. Mid-sized clusters mix Korea, Germany and Taiwan, or Spain and Italy ([`results/institution_communities.csv`](results/institution_communities.csv), [`results/institution_centrality.csv`](results/institution_centrality.csv)).
- **Authors.** 24,950 authors, but only 37% fall in the largest connected component: most groups publish in isolation ([`results/network_summary.csv`](results/network_summary.csv)).
- **Brazil.** 32 works: UNICAMP (5), UFPE (4), then one or two each from 24 other institutions. 18 of the 32 appeared in 2025-2026. The most cited is PODNet (UNICAMP with French partners, 710 citations). All three Brazilian PEFT/LLM works are from 2026, two of them from PUCRS. Full list: [`results/brazil_works.csv`](results/brazil_works.csv).

## Benchmarks (F4)

[`src/clbib/benchmarks.py`](src/clbib/benchmarks.py) marks a work as using a benchmark when its abstract names it or it cites the paper that introduced the dataset (20 benchmarks in 8 families). Method papers that also popularised a split, like Matching Networks for mini-ImageNet, are mention-only, because they are cited mostly for the method.

![Benchmarks](figures/benchmarks.png)

Since 2018, only 16-49% of works per year name or cite any benchmark, and the share falls after 2021 because OpenAlex reference lists thin out (46% of 2026 works have one). So the table reports the **mix**: among works that name or cite a benchmark, the share in each family. The mention-only column checks that a trend isn't produced by the shrinking citation signal.

| family | 2018 | 2021 | 2023 | 2025 | 2026 | 2026, mention only |
|---|---|---|---|---|---|---|
| works with a benchmark | 47 | 294 | 408 | 506 | 338 | 197 |
| MNIST-scale | 47% | 21% | 13% | 15% | 18% | 24% |
| CIFAR | 53% | 61% | 63% | 39% | 33% | 56% |
| ImageNet-scale | 40% | 44% | 53% | 48% | 39% | 24% |
| pre-trained era (ImageNet-R, CUB, DomainNet, VTAB) | 6.4% | 13% | 16% | 25% | 20% | 15% |
| NLP & LLM | 0% | 3.1% | 4.2% | 7.9% | 11% | 16% |

- The MNIST decline (2018-2023) and the late arrival of language benchmarks hold under both signals.
- CIFAR's drop after 2024 appears only with citations; by mentions it stays above 50%, so it's a coverage effect, not a real decline.
- MNIST-scale climbs back to 18-24% in 2025-2026. It survives the repository filter, so it isn't Zenodo noise, but it isn't explained yet and isn't interpreted.
- Language benchmarks lag the PEFT/LLM tag (39% of 2026 works) because many of those papers apply prompts or adapters to pre-trained vision models and evaluate on vision benchmarks.

Spot-check snippets for every benchmark: [`results/benchmark_audit.csv`](results/benchmark_audit.csv). Mention and citation counts: [`results/benchmark_signals.csv`](results/benchmark_signals.csv).

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

## Sensitivity: repository-only works

The repository rule was added after the 2026 anomaly was spotted, so it isn't pre-registered. Before the rule, 319 of 2,543 focused 2026 works were Zenodo deposits (3 in 2024), 45% of them by ten authors, with a median of one author and 2% citing anything. To show the rule doesn't drive the findings, `make sensitivity` re-runs every analysis with those works kept:

| metric | main | with repository-only works |
|---|---|---|
| focused works, 2025 | 2,112 | 2,207 |
| focused works, 2026 (partial) | 2,133 | 2,543 |
| growth 2022 -> 2025 | 2.8x | 2.9x |
| PEFT/LLM share, 2026 | 39% | 39% |
| China share, 2023-2026 | 53% | 53% |
| company author, PEFT/LLM vs rest | 1.67x | 1.66x |
| benchmark mix: MNIST-scale, 2026 | 18% | 20% |
| benchmark mix: NLP & LLM, 2026 | 11% | 14% |

Every metric keeps its direction; the largest change is the 2026 volume (-16%). Full table (25 metrics): [`results/sensitivity.csv`](results/sensitivity.csv). The sample used for validation was drawn before the rule and includes 14 repository-only works out of 300, so the precision and F1 above describe a slightly broader corpus.

## Reproduce

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
export OPENALEX_API_KEY=...   # free key, https://openalex.org/settings/api
make all                       # fetch -> clean -> growth -> subareas -> validate -> networks -> benchmarks -> sensitivity
make test
```

Raw API pages are kept gzipped under `data/raw/openalex/<query>/` with a manifest (filter, counts, fetch date). `data/` isn't versioned; `results/`, `figures/` and the label files in `validation/` are.

```
src/clbib/
  config.py       corpus definition: queries, scope, focus rule, repository list, paths
  corpus.py       the analysis corpus filter, used by every analysis
  fetch.py        OpenAlex cursor pagination, resumable
  clean.py        parquet tables (works, authorships, references) and dedup
  growth.py       F1 tables and figure
  subareas.py     F2 rule-based tags
  validate.py     sample drawing and rule scoring
  llm_label.py    LLM labellers and pairwise agreement
  networks.py     F3 countries, industry, institution and author networks
  benchmarks.py   F4 benchmark mentions and dataset citations
  sensitivity.py  headline numbers with and without repository-only works
tests/            parsing, dedup, tag rules, benchmark regexes, network building
validation/       samples, label files and labelling guide
```

## Limitations

- Retrieval is keyword-based. Papers that never say "continual learning" or "catastrophic forgetting" (e.g. "model editing", "incremental learning" alone) are missed.
- Sub-area validation uses LLM labels only. Agreement across three labellers makes the tags more credible, but it isn't a human gold standard.
- OpenAlex fields, topics and institution data are machine-assigned. Affiliations are missing for 10-39% of works per year, more often on arXiv, and author disambiguation is OpenAlex's.
- Benchmarks are only visible when an abstract names them or a reference list exists; most papers show neither.
- The repository rule is source-based. It can't catch low-quality work posted on arXiv, and it drops a legitimate paper that only exists on Zenodo.
- 2026 is a partial year.

## Author

Arthur Fachel, Data Analytics @ PUCRS, MALTA Lab. Research on neuron-level plasticity for continual learning.

## License

Code: MIT. Data: OpenAlex metadata is CC0.
