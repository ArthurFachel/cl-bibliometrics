"""F2 validation: compare rule tags with hand labels.

  python -m clbib.validate sample   # draw the stratified sample -> validation/to_label.csv
  python -m clbib.validate score    # validation/labels.csv vs rules -> results/validation.csv

The sample file deliberately does NOT show the rule predictions, so the labeller
is not anchored by them. Label columns take 1 (applies) or 0/blank (does not).
"""
from __future__ import annotations

import sys

import polars as pl

from . import config as C
from .subareas import RULES, tag

VAL = C.ROOT / "validation"
STRATA = [(2015, 2021, 50), (2022, 2024, 50), (2025, 2026, 50)]
SEED = 20261019


def sample() -> None:
    works = pl.read_parquet(C.PROCESSED / "works.parquet")
    pool = works.filter(pl.col("in_scope") & pl.col("is_canonical") & pl.col("abstract").is_not_null())
    parts = [pool.filter(pl.col("first_year").is_between(a, b)).sample(n, seed=SEED)
             for a, b, n in STRATA]
    s = (pl.concat(parts).select("id", "first_year", "title", "abstract")
         .with_columns([pl.lit(None, dtype=pl.Int8).alias(k) for k in RULES]
                       + [pl.lit(None, dtype=pl.Int8).alias("is_cl"), pl.lit("").alias("notes")]))
    VAL.mkdir(exist_ok=True)
    out = VAL / "to_label.csv"
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite labels. Delete it to redraw.")
    s.write_csv(out)
    print(f"wrote {s.height} rows to {out}")


def score(path: str = "labels.csv", tag_out: str = "") -> None:
    lab = pl.read_csv(VAL / path, infer_schema_length=None)
    lab = lab.with_columns([pl.col(k).fill_null(0).cast(pl.Int8) for k in list(RULES) + ["is_cl"]])
    n_not_cl = int((lab["is_cl"] == 0).sum())
    lab = lab.filter(pl.col("is_cl") == 1)
    pred = tag(lab.select("id", "title", "abstract"))
    rows = []
    for k in RULES:
        y = lab[k].cast(pl.Boolean).to_list()
        p = pred[k].to_list()
        tp = sum(a and b for a, b in zip(y, p))
        fp = sum((not a) and b for a, b in zip(y, p))
        fn = sum(a and (not b) for a, b in zip(y, p))
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
        rows.append({"label": k, "support": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                     "precision": prec, "recall": rec, "f1": f1})
    res = pl.DataFrame(rows).with_columns(pl.col("precision", "recall", "f1").round(2))
    out = C.RESULTS / f"validation{('_' + tag_out) if tag_out else ''}.csv"
    res.write_csv(out)
    print(f"labels: {path} -> {out}")
    with pl.Config(tbl_rows=20):
        print(res)
    print(f"labelled as not continual learning (corpus precision errors): {n_not_cl} of {n_not_cl + lab.height}")


def review_sample() -> None:
    """50 rows (stratified, every 3rd) for blind human review of the LLM labels."""
    s = pl.read_csv(VAL / "to_label.csv", infer_schema_length=None)
    out = VAL / "human_review.csv"
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite.")
    s.gather_every(3).write_csv(out)
    print(f"wrote {out}")


def agreement(a: str = "labels_llm.csv", b: str = "labels_human.csv") -> None:
    """Per-label agreement and Cohen's kappa between two labellers on shared ids."""
    la = pl.read_csv(VAL / a, infer_schema_length=None)
    lb = pl.read_csv(VAL / b, infer_schema_length=None)
    cols = ["is_cl"] + list(RULES)
    j = la.select(["id"] + cols).join(lb.select(["id"] + cols), on="id", suffix="_b")
    rows = []
    for k in cols:
        x = j[k].fill_null(0).cast(pl.Int8).to_list()
        y = j[k + "_b"].fill_null(0).cast(pl.Int8).to_list()
        n = len(x)
        po = sum(p == q for p, q in zip(x, y)) / n
        px, py = sum(x) / n, sum(y) / n
        pe = px * py + (1 - px) * (1 - py)
        kappa = (po - pe) / (1 - pe) if pe < 1 else None
        rows.append({"label": k, "n": n, "pos_a": sum(x), "pos_b": sum(y),
                     "agreement": round(po, 2), "kappa": round(kappa, 2) if kappa is not None else None})
    res = pl.DataFrame(rows)
    res.write_csv(C.RESULTS / "agreement_llm_vs_human.csv")
    with pl.Config(tbl_rows=20):
        print(res)


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    {"sample": sample, "score": score, "review": review_sample, "agreement": agreement}[cmd](*rest)
