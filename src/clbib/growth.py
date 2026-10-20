"""F1 analysis: growth of continual-learning research over time.

Outputs:
  results/growth_by_year.csv
  results/growth_by_query.csv
  figures/growth.png
"""
from __future__ import annotations

import json
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from . import config as C  # noqa: E402
from .corpus import analysis  # noqa: E402


def fetched_on() -> date:
    dates = []
    for m in C.RAW.glob("*/manifest.json"):
        dates.append(json.loads(m.read_text())["fetched_at"][:10])
    return date.fromisoformat(max(dates)) if dates else date.today()


def main() -> None:
    works = pl.read_parquet(C.PROCESSED / "works.parquet")
    final = works.filter(analysis())
    snap = fetched_on()

    by_year = (final.group_by("first_year")
               .agg(pl.len().alias("works"),
                    pl.col("is_focused").sum().alias("focused"),
                    (pl.col("is_arxiv") & pl.col("is_focused")).sum().alias("focused_on_arxiv"),
                    pl.col("is_arxiv").sum().alias("on_arxiv"),
                    (pl.col("type") == "preprint").sum().alias("preprint_only"))
               .sort("first_year")
               .with_columns((pl.col("on_arxiv") / pl.col("works")).round(3).alias("arxiv_share"),
                             (pl.col("first_year") == snap.year).alias("partial_year")))
    by_query = (final.explode("queries").group_by("queries").agg(pl.len().alias("works"))
                .sort("works", descending=True))
    only_lifelong = final.filter(pl.col("queries").list.len() == 1
                                 & pl.col("queries").list.contains("lifelong_cs")).height

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    by_year.write_csv(C.RESULTS / "growth_by_year.csv")
    by_query.write_csv(C.RESULTS / "growth_by_query.csv")

    plot = by_year.filter(pl.col("first_year") >= 2010)
    yrs, n_all = plot["first_year"].to_list(), plot["works"].to_list()
    n = plot["focused"].to_list()
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    ax.bar(yrs, n_all, color="#dfe7ec", label="broad (any mention, in scope)")
    colors = ["#9db4c0" if y == snap.year else "#2b5d7d" for y in yrs]
    ax.bar(yrs, n, color=colors, label="focused (CL is the topic)")
    ax.plot(yrs, plot["focused_on_arxiv"].to_list(), color="#c8553d", marker="o", ms=3, lw=1.5,
            label="focused, also on arXiv")
    if snap.year in yrs:
        ax.annotate(f"partial ({snap.isoformat()})", (snap.year, n_all[yrs.index(snap.year)]),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8, color="#555")
    ax.set_title("Continual learning publications per year (OpenAlex, deduplicated)", loc="left", fontsize=11)
    ax.set_ylabel("works (first appearance year)")
    ax.set_xticks([y for y in yrs if y % 2 == 0])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.01, 0.01, f"focused n = {final['is_focused'].sum():,} of {final.height:,} in scope | "
             "rules in src/clbib/config.py",
             fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(C.FIGURES / "growth.png")

    print(by_year.filter(pl.col("first_year") >= 2015))
    print(by_query)
    print(f"works found only by the 'lifelong learning' (CS) query: {only_lifelong}")


if __name__ == "__main__":
    main()
