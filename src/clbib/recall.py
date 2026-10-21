"""Recall check: how many independently found CL papers does the corpus contain?

    python -m clbib.recall freeze --src ~/.hermes/papers/papers.jsonl   # once -> validation/recall_check.csv
    python -m clbib.recall score                                        # -> results/recall_check.csv

The check set is the arXiv papers that a separate literature scout reported
as continual learning in its runs from 2026-10-06 to 2026-10-18 (45 papers).
The scout found them by searching for CL, so the recall it gives is an upper
bound. Only the arXiv id, title and dates are kept; the scout's own text is not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from . import config as C
from .clean import norm_title

VAL = C.ROOT / "validation"
CHECK = "recall_check.csv"
LAST_RUN = "2026-10-18"
EXPECTED = 45


def freeze_rows(records: list[dict]) -> list[dict]:
    """Reported papers first seen in a scout run on or before LAST_RUN, sorted by id."""
    rows = []
    for r in records:
        if r.get("status") != "reportado":
            continue
        first = min(a["run"][:10] for a in r["aparicoes"])
        if first > LAST_RUN:
            continue
        rows.append({"arxiv_id": r["arxiv_id"], "title": r["titulo"],
                     "published": (r.get("publicado") or "")[:10], "first_seen": first})
    return sorted(rows, key=lambda x: x["arxiv_id"])


def freeze(src: str) -> None:
    out = VAL / CHECK
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite the frozen check set.")
    lines = Path(src).expanduser().read_text(encoding="utf-8").splitlines()
    rows = freeze_rows([json.loads(line) for line in lines if line.strip()])
    if len(rows) != EXPECTED:
        raise SystemExit(f"expected {EXPECTED} reported papers up to {LAST_RUN}, got {len(rows)}")
    pl.DataFrame(rows).write_csv(out)
    print(f"wrote {len(rows)} rows to {out}")


def match(check: pl.DataFrame, works: pl.DataFrame) -> pl.DataFrame:
    """One row per check paper: the matching work (arXiv DOI first, then normalised
    title) and corpus flags taken from the canonical record of its dedup cluster."""
    w = works.with_columns(pl.col("doi").str.to_lowercase().alias("doi_lc"))
    by_doi = {d: i for d, i in zip(w["doi_lc"].to_list(), w["id"].to_list()) if d}
    by_title = {t: i for t, i in zip(w["title_key"].to_list(), w["id"].to_list()) if t}
    rows = []
    for r in check.iter_rows(named=True):
        wid, via = by_doi.get(f"https://doi.org/10.48550/arxiv.{r['arxiv_id']}".lower()), "doi"
        if wid is None:
            wid, via = by_title.get(norm_title(r["title"])), "title"
        rows.append({"arxiv_id": r["arxiv_id"], "title": r["title"], "work_id": wid,
                     "matched_via": via if wid else None})
    m = pl.DataFrame(rows, schema={"arxiv_id": pl.Utf8, "title": pl.Utf8, "work_id": pl.Utf8,
                                   "matched_via": pl.Utf8})
    canon = (w.select("id", "cluster_id")
             .join(w.filter(pl.col("is_canonical")).select("cluster_id", "in_scope", "repository_only", "is_focused"),
                   on="cluster_id", how="left")
             .rename({"id": "work_id"}))
    m = m.join(canon, on="work_id", how="left", maintain_order="left")
    analysed = pl.col("in_scope") & ~pl.col("repository_only")
    return m.select(
        "arxiv_id", "title", "work_id", "matched_via",
        pl.col("work_id").is_not_null().alias("found"),
        pl.col("in_scope").fill_null(False),
        analysed.fill_null(False).alias("in_analysis"),
        (analysed & pl.col("is_focused")).fill_null(False).alias("focused"))


def score() -> None:
    check = pl.read_csv(VAL / CHECK, infer_schema_length=None)
    m = match(check, pl.read_parquet(C.PROCESSED / "works.parquet"))
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    m.write_csv(C.RESULTS / "recall_check.csv")
    n = m.height
    for col in ["found", "in_scope", "in_analysis", "focused"]:
        k = int(m[col].sum())
        print(f"{col:12s} {k}/{n} ({k / n:.0%})")
    with pl.Config(tbl_rows=10, fmt_str_lengths=70):
        print(m.filter(~pl.col("found")).select("arxiv_id", "title"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "score"])
    ap.add_argument("--src", default="~/.hermes/papers/papers.jsonl")
    a = ap.parse_args()
    if a.cmd == "freeze":
        freeze(a.src)
    else:
        score()


if __name__ == "__main__":
    main()
