First complete version, built on the OpenAlex snapshot of 2026-10-19.

What it answers
- F1 growth: focused continual learning works went from 756 (2022) to 2,112 (2025), about 2.8x; the arXiv share fell from 57% (2019) to 38% (2025).
- F2 sub-areas: PEFT/LLM rose from 2.5% of works (2019) to 39% (2026); five rule-based tags pass a held-out test against three LLM labellers.
- F3 who: China's share of works with a known affiliation rose from 19% to 53%, the US share fell from 40% to 18%; OpenAlex countries checked against Crossref affiliations (China precision and recall 0.99).
- F4 benchmarks: MNIST-scale fell from 47% (2018) to 12% (2023) of the benchmark mix, pre-trained-era benchmarks rose to 24% (2025), language benchmarks reached 14% in 2026. The language detector was re-validated on a fresh, pre-registered sample (precision 0.98, recall 0.25), so 14% is a floor.
- F5 drivers: among 2018-2023 PEFT/LLM works, US works got 60% of in-corpus citations with 35% of the output; among 2024-2025 works, China wrote 59% and got 77%.
- Recall: 40 of 45 scout-confirmed CL papers are in the corpus (upper bound).

Assets
- report.pdf: 3-page summary.
- cl-bibliometrics-processed-2026-10-19.zip: data/processed/*.parquet and the OpenAlex fetch manifests. Unzip at the repo root and run any make target except fetch and clean, with no API key. OpenAlex metadata is CC0.
