"""Re-state every headline number with and without repository-only works.

Reads results/ (main corpus) and results/sensitivity_with_repository_only/
(same pipeline run with CLBIB_INCLUDE_REPOSITORY_ONLY=1, see `make sensitivity`)
and writes results/sensitivity.csv.
"""
from __future__ import annotations

import polars as pl

from . import config as C

MAIN = C.ROOT / "results"
ALT = C.ROOT / "results" / "sensitivity_with_repository_only"


def cell(d: str, f: str, key: str, keyval, col: str):
    df = pl.read_csv(d / f)
    v = df.filter(pl.col(key) == keyval)[col]
    return None if v.len() == 0 else v[0]


def ratio(d, f, key, kv, num, den):
    a, b = cell(d, f, key, kv, num), cell(d, f, key, kv, den)
    return a / b if a is not None and b else None


METRICS = [
    ("focused works, 2022", lambda d: cell(d, "growth_by_year.csv", "first_year", 2022, "focused")),
    ("focused works, 2025", lambda d: cell(d, "growth_by_year.csv", "first_year", 2025, "focused")),
    ("focused works, 2026 (partial)", lambda d: cell(d, "growth_by_year.csv", "first_year", 2026, "focused")),
    ("growth 2022 -> 2025 (x)", lambda d: cell(d, "growth_by_year.csv", "first_year", 2025, "focused")
     / cell(d, "growth_by_year.csv", "first_year", 2022, "focused")),
    ("arXiv share of focused, 2019", lambda d: ratio(d, "growth_by_year.csv", "first_year", 2019, "focused_on_arxiv", "focused")),
    ("arXiv share of focused, 2025", lambda d: ratio(d, "growth_by_year.csv", "first_year", 2025, "focused_on_arxiv", "focused")),
    ("PEFT/LLM share, 2019", lambda d: cell(d, "subarea_by_year.csv", "first_year", 2019, "peft_llm")),
    ("PEFT/LLM share, 2022", lambda d: cell(d, "subarea_by_year.csv", "first_year", 2022, "peft_llm")),
    ("PEFT/LLM share, 2025", lambda d: cell(d, "subarea_by_year.csv", "first_year", 2025, "peft_llm")),
    ("PEFT/LLM share, 2026", lambda d: cell(d, "subarea_by_year.csv", "first_year", 2026, "peft_llm")),
    ("replay share, 2025", lambda d: cell(d, "subarea_by_year.csv", "first_year", 2025, "replay")),
    ("regularization share, 2025", lambda d: cell(d, "subarea_by_year.csv", "first_year", 2025, "regularization")),
    ("China share, 2023-2026", lambda d: cell(d, "country_by_period.csv", "country", "CN", "2023-2026")),
    ("US share, 2023-2026", lambda d: cell(d, "country_by_period.csv", "country", "US", "2023-2026")),
    ("company author, 2019", lambda d: cell(d, "collab_by_year.csv", "first_year", 2019, "with_company")),
    ("company author, 2026", lambda d: cell(d, "collab_by_year.csv", "first_year", 2026, "with_company")),
    ("company author, PEFT/LLM vs rest (x)", lambda d: cell(d, "peft_llm_by_country.csv", "group", "any company", "ratio")),
    ("Brazil works", lambda d: pl.read_csv(d / "brazil_works.csv").height),
    ("benchmark mix: MNIST-scale, 2018", lambda d: cell(d, "benchmark_mix_by_year.csv", "first_year", 2018, "MNIST-scale")),
    ("benchmark mix: MNIST-scale, 2023", lambda d: cell(d, "benchmark_mix_by_year.csv", "first_year", 2023, "MNIST-scale")),
    ("benchmark mix: MNIST-scale, 2026", lambda d: cell(d, "benchmark_mix_by_year.csv", "first_year", 2026, "MNIST-scale")),
    ("benchmark mix: pre-trained era, 2025", lambda d: cell(d, "benchmark_mix_by_year.csv", "first_year", 2025, "pre-trained era")),
    ("benchmark mix: NLP & LLM, 2026", lambda d: cell(d, "benchmark_mix_by_year.csv", "first_year", 2026, "NLP & LLM")),
    ("benchmark mix: CIFAR, 2026, mention only", lambda d: cell(d, "benchmark_mix_mention_only_by_year.csv", "first_year", 2026, "CIFAR")),
    ("benchmark mix: MNIST-scale, 2026, mention only", lambda d: cell(d, "benchmark_mix_mention_only_by_year.csv", "first_year", 2026, "MNIST-scale")),
]


def main() -> None:
    rows = []
    for name, f in METRICS:
        a, b = f(MAIN), f(ALT)
        rows.append({"metric": name, "main": round(float(a), 3), "with_repository_only": round(float(b), 3),
                     "difference": round(float(b) - float(a), 3)})
    res = pl.DataFrame(rows)
    res.write_csv(MAIN / "sensitivity.csv")
    with pl.Config(tbl_rows=40, fmt_str_lengths=60, tbl_width_chars=160):
        print(res)


if __name__ == "__main__":
    main()
