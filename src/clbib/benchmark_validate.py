"""F4 validation: does the benchmark detector agree with LLM readers?

  python -m clbib.benchmark_validate sample
  python -m clbib.benchmark_validate label --model <model> --out <file> [--workers N]
  python -m clbib.benchmark_validate score

What is validated: the MENTION signal (benchmark names in title + abstract),
at the level of the 8 families used for findings. The citation signal can't be
checked from an abstract and is not validated here.

Design. The detector was frozen in commit 8381df9 before any label existed.
Benchmark mentions are rare (about 1 abstract in 8), so a random sample would
hold almost no positives. The sample is stratified instead: 150 works where
the detector found any benchmark name, 150 where it found none, all from the
focused analysis corpus (2018-2026, with abstract). Precision and recall are
re-weighted by stratum size so they estimate the corpus-level values, with
bootstrap 95% intervals (resampling within strata).

Labellers see title + abstract and the guide below, never the regexes.
"""
from __future__ import annotations

import argparse
import random

import polars as pl

from . import config as C
from .benchmarks import FAMILIES, FAMILY, ROUND1_NAMES, mention_flags
from .corpus import focused
from . import llm_label
from .llm_label import kappa

VAL = C.ROOT / "validation"
SAMPLE = "benchmarks_to_label.csv"
SEED = 20261021
PER_STRATUM = 150
FREEZE_COMMIT = "8381df9"

KEYS = {"MNIST-scale": "mnist_scale", "CIFAR": "cifar", "ImageNet-scale": "imagenet_scale",
        "CL-native": "cl_native", "pre-trained era": "pretrained_era",
        "detection & segmentation": "detection_segmentation", "RL": "rl", "NLP & LLM": "nlp_llm"}
LABELS = list(KEYS.values())
LABELLERS = {"opus": "labels_bench_opus.csv", "sonnet": "labels_bench_sonnet.csv",
             "deepseek": "labels_bench_deepseek.csv"}
MODELS = {"opus": "hermes:claude-opus-5-5", "sonnet": "hermes:claude-sonnet-5-5",
          "deepseek": "deepseek-ai/DeepSeek-V4-Flash"}

GUIDE = """You read the title and abstract of a research paper and record which benchmark datasets or environments it says it uses or evaluates on. Return 1 for a family when the text NAMES at least one dataset or environment of that family as used in the paper, else 0.
- mnist_scale: MNIST or any variant (Permuted, Split, Rotated MNIST, Fashion-MNIST, EMNIST, KMNIST), SVHN, or other small digit/character image datasets
- cifar: CIFAR-10 or CIFAR-100, including splits such as Split CIFAR-100 or CIFAR-100-B0
- imagenet_scale: ImageNet / ILSVRC or its subsets (ImageNet-100, ImageNet-1K), Tiny-ImageNet, mini-ImageNet
- pretrained_era: ImageNet-R, ImageNet-A, CUB-200, DomainNet, VTAB, Stanford Cars, OmniBenchmark or similar benchmarks for continual learning with pre-trained models
- cl_native: datasets built specifically for continual learning, e.g. CORe50, CLEAR, Stream-51, CLOC, OpenLORIS
- detection_segmentation: object detection or segmentation datasets, e.g. COCO, Pascal VOC, ADE20K, Cityscapes
- rl: reinforcement learning benchmarks or environments, e.g. Atari, Meta-World, Continual World, MuJoCo, Procgen
- nlp_llm: any natural-language or LLM benchmark, e.g. TRACE, SuperNI, MMLU, GSM8K, GLUE, AG News / Yelp / DBpedia text classification, VQA or multimodal instruction benchmarks
Rules: count only datasets the text names. "Experiments on three benchmark datasets" without names = 0 everywhere. A dataset mentioned only as related work or motivation does not count. Labels are multi-label.
Answer with one JSON object and nothing else: {"mnist_scale":0|1, ... all 8 keys ..., "notes":"datasets you saw, or empty"}"""


def detect_families(df: pl.DataFrame, names: list[str] | None = None) -> pl.DataFrame:
    """Detector family flags from title + abstract mentions. Default: the round-1 detector."""
    names = ROUND1_NAMES if names is None else names
    m = mention_flags(df, names)
    out = m.select("id", *[pl.any_horizontal([n for n in names if FAMILY[n] == fam]).alias(KEYS[fam])
                           for fam in FAMILIES])
    return out.with_columns(pl.any_horizontal(LABELS).alias("any"))


def population() -> pl.DataFrame:
    w = (pl.read_parquet(C.PROCESSED / "works.parquet")
         .filter(focused() & pl.col("first_year").is_between(2018, 2026) & pl.col("abstract").is_not_null())
         .select("id", "first_year", "title", "abstract"))
    return w.join(detect_families(w), on="id")


def sample() -> None:
    out = VAL / SAMPLE
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to redraw.")
    pop = population()
    parts = [pop.filter(pl.col("any")).sample(PER_STRATUM, seed=SEED),
             pop.filter(~pl.col("any")).sample(PER_STRATUM, seed=SEED)]
    s = pl.concat(parts).sample(fraction=1.0, shuffle=True, seed=SEED)  # labellers don't see strata
    s.select("id", "first_year", "title", "abstract").write_csv(out)
    print(f"population: {pop.height} works, {int(pop['any'].sum())} with a detected benchmark name")
    print(f"wrote {s.height} rows to {out}")


def load_labels() -> dict[str, pl.DataFrame]:
    labs = {}
    for name, f in LABELLERS.items():
        p = VAL / f
        if p.exists():
            labs[name] = pl.read_csv(p, infer_schema_length=None).select(
                ["id"] + [pl.col(k).cast(pl.Int8) for k in LABELS])
    return labs


def weighted_pr(y: list[int], p: list[int], w: list[float]) -> tuple[float | None, float | None]:
    tp = sum(wi for yi, pi, wi in zip(y, p, w) if yi and pi)
    fp = sum(wi for yi, pi, wi in zip(y, p, w) if (not yi) and pi)
    fn = sum(wi for yi, pi, wi in zip(y, p, w) if yi and not pi)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    return prec, rec


def bootstrap(y, p, w, strata, n: int = 1000) -> tuple:
    rng = random.Random(SEED)
    idx = {s: [i for i, x in enumerate(strata) if x == s] for s in set(strata)}
    ps, rs = [], []
    for _ in range(n):
        pick = [rng.choice(ix) for ix in idx.values() for _ in ix]
        a, b = weighted_pr([y[i] for i in pick], [p[i] for i in pick], [w[i] for i in pick])
        if a is not None:
            ps.append(a)
        if b is not None:
            rs.append(b)

    def ci(v):
        if len(v) < 50:
            return (None, None)
        v = sorted(v)
        return (round(v[int(0.025 * len(v))], 2), round(v[int(0.975 * len(v)) - 1], 2))
    return ci(ps), ci(rs)


def score() -> None:
    pop = population()
    n_pos, n_neg = int(pop["any"].sum()), int((~pop["any"]).sum())
    s = pl.read_csv(VAL / SAMPLE, infer_schema_length=None).select("id", "title", "abstract")
    det = detect_families(s)
    labs = load_labels()
    if not labs:
        raise SystemExit("no label files yet")
    # majority vote over the labellers present, on ids every labeller has (ties count as 0)
    wide = det.select("id")
    for name, lab in labs.items():
        wide = wide.join(lab.rename({k: f"{k}__{name}" for k in LABELS}), on="id", how="inner",
                         maintain_order="left")
    maj = wide.select("id", *[
        (pl.sum_horizontal([pl.col(f"{k}__{n}") for n in labs]) * 2 > len(labs)).cast(pl.Int8).alias(k)
        for k in LABELS])
    refs = {**labs, "majority": maj} if len(labs) >= 2 else labs

    d = det.join(wide.select("id"), on="id", how="inner", maintain_order="left")
    strata = ["pos" if a else "neg" for a in d["any"].to_list()]
    sp, sn = strata.count("pos"), strata.count("neg")
    w = [n_pos / sp if s_ == "pos" else n_neg / sn for s_ in strata]
    rows = []
    for ref_name, ref in refs.items():
        r = d.select("id").join(ref, on="id", how="left", maintain_order="left")
        for fam, k in KEYS.items():
            y = r[k].fill_null(0).to_list()
            p = d[k].cast(pl.Int8).to_list()
            prec, rec = weighted_pr(y, p, w)
            (pl_, ph), (rl_, rh) = bootstrap(y, p, w, strata)
            rows.append({"reference": ref_name, "family": fam, "sample_pos": sum(y), "detector_pos": sum(p),
                         "precision": None if prec is None else round(prec, 2), "precision_ci": f"{pl_}-{ph}",
                         "recall": None if rec is None else round(rec, 2), "recall_ci": f"{rl_}-{rh}"})
    res = pl.DataFrame(rows)
    res.write_csv(C.RESULTS / "benchmark_validation.csv")

    agree = []
    names = list(labs)
    for k in LABELS:
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                j = labs[a].select("id", k).join(labs[b].select("id", k), on="id", suffix="_b",
                                                 maintain_order="left")
                po, ka = kappa(j[k].to_list(), j[k + "_b"].to_list())
                agree.append({"family": k, "pair": f"{a}-{b}", "n": j.height, "agree": round(po, 2),
                              "kappa": None if ka is None else round(ka, 2)})
    pl.DataFrame(agree).write_csv(C.RESULTS / "benchmark_validation_agreement.csv")
    with pl.Config(tbl_rows=60, tbl_width_chars=200, fmt_str_lengths=40):
        print(f"population with abstract: {n_pos} detector-positive, {n_neg} detector-negative")
        print(res.filter(pl.col("reference") == "majority") if "majority" in refs else res)
        print(pl.DataFrame(agree).pivot(on="pair", index="family", values="kappa"))


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
