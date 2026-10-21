"""F4 round 2: does the extended NLP & LLM detector find language benchmarks?

    python -m clbib.nlp_validate sample
    python -m clbib.nlp_validate label --labeller opus|sonnet|deepseek [--workers N]
    python -m clbib.nlp_validate score

Pre-registration: validation/nlp_round2_prereg.md. Fresh sample, disjoint from
round 1, three strata of 100 (nlp / peft / other); labellers use the round-1
guide and never see the detector. Scores are re-weighted by stratum size.
"""
from __future__ import annotations

import argparse

import polars as pl

from . import config as C
from . import llm_label
from .benchmark_validate import GUIDE, KEYS, LABELS, MODELS, VAL, bootstrap, detect_families, weighted_pr
from .benchmark_validate import population as round1_population
from .benchmarks import BENCHMARKS, NAMED_LANGUAGE_DATASET, ROUND1_NAMES
from .llm_label import kappa

SAMPLE = "nlp_round2_to_label.csv"
ROUND1_SAMPLE = "benchmarks_to_label.csv"
SEED = 20261023
PER_STRATUM = 100
FREEZE_COMMIT = "759ce6e"
PRECISION_BAR = 0.85
LABELLERS = {"opus": "labels_nlp2_opus.csv", "sonnet": "labels_nlp2_sonnet.csv",
             "deepseek": "labels_nlp2_deepseek.csv"}
DETECTORS = {"round 1": ROUND1_NAMES, "layer 1": list(BENCHMARKS),
             "layer 1 + 2": list(BENCHMARKS) + [NAMED_LANGUAGE_DATASET]}


def assign_strata(d: pl.DataFrame) -> pl.DataFrame:
    return d.with_columns(pl.when(pl.col("nlp_llm")).then(pl.lit("nlp"))
                          .when(pl.col("peft_llm").fill_null(False)).then(pl.lit("peft"))
                          .otherwise(pl.lit("other")).alias("stratum"))


def draw(pop: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    parts = [pop.filter(pl.col("stratum") == s).sample(n, seed=seed) for s in ("nlp", "peft", "other")]
    return pl.concat(parts).sample(fraction=1.0, shuffle=True, seed=seed)


def population() -> pl.DataFrame:
    """Focused 2018-2026 works with an abstract, minus round 1, with their stratum."""
    w = round1_population().select("id", "first_year", "title", "abstract")
    done = pl.read_csv(VAL / ROUND1_SAMPLE, infer_schema_length=None)["id"]
    w = w.filter(~pl.col("id").is_in(done.implode()))
    nlp = detect_families(w, DETECTORS["layer 1 + 2"]).select("id", "nlp_llm")
    tags = pl.read_parquet(C.PROCESSED / "subareas.parquet").select("id", "peft_llm")
    return assign_strata(w.join(nlp, on="id").join(tags, on="id", how="left"))


def sample() -> None:
    out = VAL / SAMPLE
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to redraw.")
    pop = population()
    s = draw(pop, PER_STRATUM, SEED)
    s.select("id", "first_year", "title", "abstract").write_csv(out)  # no stratum column: labellers must not see it
    print(pop.group_by("stratum").len().sort("stratum"))
    print(f"wrote {s.height} rows to {out}")


def load_labels() -> dict[str, pl.DataFrame]:
    return {name: pl.read_csv(VAL / f, infer_schema_length=None).select(["id"] + [pl.col(k).cast(pl.Int8) for k in LABELS])
            for name, f in LABELLERS.items() if (VAL / f).exists()}


def score() -> None:
    pop = population()
    sizes = dict(pop.group_by("stratum").len().iter_rows())
    labs = load_labels()
    if len(labs) < 2:
        raise SystemExit("need at least two label files")
    s = (pl.read_csv(VAL / SAMPLE, infer_schema_length=None).select("id", "title", "abstract")
         .join(pop.select("id", "stratum"), on="id", maintain_order="left"))
    wide = s.select("id", "stratum")
    for name, lab in labs.items():
        wide = wide.join(lab.rename({k: f"{k}__{name}" for k in LABELS}), on="id", how="inner", maintain_order="left")
    maj = wide.select("id", "stratum", *[
        (pl.sum_horizontal([pl.col(f"{k}__{n}") for n in labs]) * 2 > len(labs)).cast(pl.Int8).alias(k)
        for k in LABELS])
    strata = maj["stratum"].to_list()
    n_in = {x: strata.count(x) for x in set(strata)}
    w = [sizes[x] / n_in[x] for x in strata]
    texts = maj.select("id").join(s.select("id", "title", "abstract"), on="id", maintain_order="left")
    rows = []
    for det_name, names in DETECTORS.items():
        det = detect_families(texts, names)
        for fam, k in KEYS.items():
            if det_name != "layer 1 + 2" and k != "nlp_llm":
                continue  # non-NLP families are the same in every detector; report them once
            y, p = maj[k].to_list(), det[k].cast(pl.Int8).to_list()
            prec, rec = weighted_pr(y, p, w)
            (pl_, ph), (rl_, rh) = bootstrap(y, p, w, strata)
            rows.append({"detector": det_name, "family": fam, "sample_pos": sum(y), "detector_pos": sum(p),
                         "precision": None if prec is None else round(prec, 2), "precision_ci": f"{pl_}-{ph}",
                         "recall": None if rec is None else round(rec, 2), "recall_ci": f"{rl_}-{rh}"})
    res = pl.DataFrame(rows)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    res.write_csv(C.RESULTS / "nlp_validation_round2.csv")
    names = list(labs)
    agree = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            j = labs[a].select("id", "nlp_llm").join(labs[b].select("id", "nlp_llm"), on="id", suffix="_b")
            po, ka = kappa(j["nlp_llm"].to_list(), j["nlp_llm_b"].to_list())
            agree.append({"pair": f"{a}-{b}", "n": j.height, "agree": round(po, 2),
                          "kappa": None if ka is None else round(ka, 2)})
    pl.DataFrame(agree).write_csv(C.RESULTS / "nlp_validation_round2_agreement.csv")
    l2 = res.filter((pl.col("detector") == "layer 1 + 2") & (pl.col("family") == "NLP & LLM"))["precision"][0]
    use = l2 is not None and l2 >= PRECISION_BAR
    with pl.Config(tbl_rows=30, tbl_width_chars=200):
        print(f"labelled: {maj.height} works; stratum sizes in population: {sizes}; detector frozen at {FREEZE_COMMIT}")
        print(res)
        print(pl.DataFrame(agree))
    print(f"pre-registered rule: layer 2 precision {l2} vs bar {PRECISION_BAR} -> NLP_LAYER2 = {use}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "label", "score"])
    ap.add_argument("--labeller", choices=list(LABELLERS))
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "sample":
        sample()
    elif a.cmd == "label":
        llm_label.label(SAMPLE, LABELLERS[a.labeller], MODELS[a.labeller], a.workers, GUIDE, LABELS)
    else:
        score()


if __name__ == "__main__":
    main()
