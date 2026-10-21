# Who moved continual learning to LLMs? A bibliometric study of OpenAlex

Arthur Fachel, Data Analytics, PUCRS. Snapshot of 2026-10-19. Code, data and every table: https://github.com/ArthurFachel/cl-bibliometrics (release v1.0.0).

## Question

Continual learning studies models that learn a sequence of tasks without forgetting the earlier ones. For years the field ran on small vision benchmarks such as Permuted MNIST and Split CIFAR. Today a large part of it works with pre-trained models and LLMs. I wanted to measure when that shift happened, how far it went, and which countries, institutions and companies drove it.

## Data

I pulled every work from OpenAlex whose title or abstract says "continual learning", "catastrophic forgetting" or one of the incremental-learning settings, plus "lifelong learning" in Computer Science. That gave 24,264 records. After scope rules, deduplication of 662 preprint/published pairs and removal of 1,000 works that exist only on deposit repositories such as Zenodo, 13,426 remain. The analysis uses the 9,326 works where continual learning is the topic: a CL term in the title, or at least twice in the abstract. On held-out data this focus rule has precision 0.93-0.98.

Recall has an upper bound from an independent check. Of 45 papers that a separate literature scout had confirmed as continual learning, 40 (89%) are in the corpus. The 5 misses never use the query phrases.

## Method

Everything is rule-based, so each label can be audited. Sub-areas (replay, regularization, PEFT/LLM and others) come from regular expressions over the title and abstract. Benchmarks come from dataset names in the abstract and from citations to the papers that introduced each dataset. Countries and industry come from OpenAlex affiliations.

There is no human gold set, so I validated each rule against three LLM labellers (two Claude models and DeepSeek) that never saw the rules:

- Sub-area tags were tuned on 150 works, frozen, then scored once on 150 new ones. Only tags with F1 of at least 0.7 under all three labellers feed the findings. PEFT/LLM scores 0.84.
- OpenAlex countries were compared with the raw affiliation strings publishers deposit in Crossref. For China, precision and recall are both 0.99.
- The benchmark detector was scored on 300 works. It was precise everywhere but found only 19% of the papers that use language benchmarks. I extended it and tested it again on 300 fresh works, with the design and a decision rule committed before any label existed.

To see who drove the shift, I counted citations between works inside the corpus and compared each group's share of citations with its share of PEFT/LLM works, separately for works first seen in 2018-2023 and in 2024-2025.

## Findings

The field grew about 2.8x in three years: 756 focused works first appeared in 2022 and 2,112 in 2025, and 2026 had already passed that by October 19.

![Growth of continual learning publications](../figures/growth.png)

Most of that growth went to LLMs and pre-trained models. Works tagged PEFT/LLM went from 2.5% of the field in 2019 to 29% in 2025 and 39% in 2026, while replay and regularization stayed at about 30% each. Over the same years China's share of works with a known affiliation rose from 19% (2015-2019) to 53% (2023-2026), and the US share fell from 40% to 18%.

The shift started in US industry and was then scaled in China. Among PEFT/LLM works first seen in 2018-2023, US works received 60% of the in-corpus citations with 35% of the output, and works with a company author received 58% with 32%. Two papers carry much of that: Google's L2P and CODA-Prompt (Georgia Tech, IBM and Rice), both about prompting pre-trained vision transformers. Without them, US and company works still get more than their share. Among 2024-2025 works, China writes 59% and receives 77% of the citations. Of the citations made to PEFT/LLM work in 2023, 67% went to US works; in 2026, 35%.

![Who wrote PEFT/LLM work and whose work was built on](../figures/drivers.png)

Benchmarks followed with a lag. Among works that name or cite a benchmark, MNIST-scale datasets fell from 47% in 2018 to 12% in 2023, and pre-trained-era benchmarks (ImageNet-R, CUB, DomainNet, VTAB) rose from 6% to 24% in 2025. Language benchmarks stayed at 4-6% through 2024 and reached 14% in 2026. The early PEFT/LLM papers were mostly vision papers, which explains part of the lag.

![Benchmark families over time](../figures/benchmarks.png)

## What to trust

The growth, the PEFT/LLM share and the country shares are backed by held-out validation and by a sensitivity run that puts the repository-only works back in; none of them changes direction. The citation split counts only citations between works in the corpus, and the 2018-2023 cohort is small, so its intervals are wide (US share of citations 36-75%). The weakest number is the language-benchmark share. On the fresh sample, the extended detector finds about one in four language-benchmark papers (recall 0.25, precision 0.98). A looser heuristic reached recall 0.42 but precision 0.68, below the 0.85 bar I had set in advance, so I left it out. The 14% is a floor. Other limits: affiliations are missing for 10-39% of works per year, the labels come from LLMs and not people, and 2026 is a partial year.

## Reproduce

`make all` rebuilds everything from the OpenAlex API. Without an API key, the processed tables attached to release v1.0.0 cover every step except the download.
