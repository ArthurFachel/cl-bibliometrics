"""F4: when did continual learning move from one benchmark family to the next?

A work "uses" a benchmark when its title/abstract names it OR it cites the
paper that introduced the dataset. Both signals are kept separately, because
each has its own blind spot:
  - abstracts name a dataset in only a minority of papers;
  - OpenAlex reference lists are missing for many recent works, and a citation
    is not always a use (a dataset paper can be cited in related work).

Only papers that introduce a dataset/benchmark are used as citation anchors.
Method papers that also popularised a split (Matching Networks for
mini-ImageNet, Goodfellow et al. 2013 for Permuted MNIST) are mention-only,
because they are cited far more for the method than for the data.

Outputs:
  results/benchmark_by_year.csv       share of works per benchmark and year
  results/benchmark_family_by_year.csv
  results/benchmark_mix_by_year.csv   family shares among works that name or cite any benchmark
  results/benchmark_mix_mention_only_by_year.csv  same, abstract mentions only
  results/benchmark_signals.csv       mention vs citation counts per benchmark
  results/benchmark_audit.csv         random matched snippets for spot checks
  figures/benchmarks.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from . import config as C  # noqa: E402
from .corpus import focused  # noqa: E402

SEED = 20261020

# name: (family, mention regex (case-insensitive unless it starts with (?-i)), cited OpenAlex ids)
# Anchor ids were looked up by exact title on 2026-10-20; titles in comments.
BENCHMARKS: dict[str, tuple[str, str, list[str]]] = {
    "Permuted MNIST": ("MNIST-scale", r"permuted[- ]?mnist", []),
    "Split/Rotated MNIST": ("MNIST-scale", r"split[- ]?mnist|rotated[- ]?mnist", []),
    "MNIST (any)": ("MNIST-scale", r"\bmnist\b|fashion-?mnist|\bemnist\b|\bkmnist\b",
                      ["W2007339694", "W2182396527"]),  # The MNIST Database of Handwritten Digit Images
    "SVHN": ("MNIST-scale", r"\bsvhn\b", ["W2335728318"]),  # Reading digits in natural images
    "CIFAR-10/100": ("CIFAR", r"cifar[- ]?10\b|cifar[- ]?100", ["W3118608800"]),  # Learning Multiple Layers ... Tiny Images
    "Tiny/mini-ImageNet": ("ImageNet-scale", r"tiny[- ]?imagenet|mini[- ]?imagenet", []),
    "ImageNet (full/subset)": ("ImageNet-scale", r"imagenet|\bilsvrc",  # variants stripped first, see STRIP
                               ["W2108598243", "W4239072543",  # ImageNet: a large-scale hierarchical image database
                                "W2117539524", "W2546241758"]),  # ImageNet Large Scale Visual Recognition Challenge
    "ImageNet-R/A": ("pre-trained era", r"imagenet[- ]?(r|a)\b", ["W3037492894"]),  # The Many Faces of Robustness
    "DomainNet": ("pre-trained era", r"domainnet", ["W2981720610"]),  # Moment Matching for Multi-Source DA
    "CUB-200": ("pre-trained era", r"\bcub\b|cub-?200", ["W1797268635"]),  # Caltech-UCSD Birds-200-2011
    "VTAB": ("pre-trained era", r"\bvtab\b", ["W3008526508"]),
    "CORe50": ("CL-native", r"core-?50", ["W2613152434"]),
    "CLEAR / Stream-51": ("CL-native", r"(?-i)\bCLEAR-(10|100)\b|\bCLEAR\b(?i) benchmark|(?i)stream-?51",
                          ["W3212034012", "W3042733907"]),
    "COCO / VOC / ADE20K": ("detection & segmentation", r"\bcoco\b|pascal ?voc|\bvoc\b|ade-?20k|cityscapes",
                            ["W1861492603", "W2952122856", "W2031489346", "W2737258237"]),
    "Atari / Meta-World": ("RL", r"\batari\b|arcade learning environment|meta-?world|\bcw(10|20)\b",
                           ["W2150468603", "W2952509347", "W2981344907", "W3165631200"]),
    "Text classification": ("NLP & LLM", r"ag ?news|\byelp\b|dbpedia|yahoo!? answers|amazon reviews",
                            ["W2170240176"]),  # Character-level ConvNets for Text Classification
    "GLUE": ("NLP & LLM", r"(?-i)\b(GLUE|SuperGLUE)\b", ["W2923014074", "W2963310665", "W2799054028"]),
    "SuperNI / TRACE": ("NLP & LLM", r"super-?natural ?instructions|(?-i)\bSuperNI\b|\bTRACE\b(?i)[^.]{0,40}\b(benchmark|dataset)s?\b",
                        ["W4385572845", "W4224275713"]),
    "MMLU / GSM8K": ("NLP & LLM", r"(?-i)\bMMLU\b|(?i)gsm-?8k", ["W3121904249", "W3083410900", "W3210904070"]),
    "VQA / CoIN": ("NLP & LLM", r"\bvqa\b|(?-i)\bCoIN\b", ["W4392822544"]),
}
ROUND1_NAMES = list(BENCHMARKS)  # the detector scored in round 1 (frozen at 8381df9); never edit
# Text removed before matching a benchmark, so that e.g. "ImageNet-R" or
# "mini-ImageNet" does not also count as ImageNet (polars regex has no lookahead).
STRIP = {
    "ImageNet (full/subset)": r"(?i)(tiny|mini)[- ]?imagenet|imagenet[- ]?(r|a|sketch|v2|21k|22k)\b",
}
FAMILIES = ["MNIST-scale", "CIFAR", "ImageNet-scale", "CL-native", "pre-trained era",
            "detection & segmentation", "RL", "NLP & LLM"]


def rx(pattern: str) -> str:
    return pattern if pattern.startswith("(?-i)") else "(?i)" + pattern


FAMILY = {n: v[0] for n, v in BENCHMARKS.items()}
ALL_NAMES = list(BENCHMARKS)


def text_of() -> pl.Expr:
    """Title + abstract, the text every mention rule reads."""
    return pl.concat_str([pl.col("title").fill_null(""), pl.lit(" "), pl.col("abstract").fill_null("")])


def mention_flags(df: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    """`id` plus one boolean column per benchmark name: does the title or abstract name it?"""
    text = text_of()
    cols = []
    for name in names:
        t = text.str.replace_all(STRIP[name], " ") if name in STRIP else text
        cols.append(t.str.contains(rx(BENCHMARKS[name][1])).alias(name))
    return df.with_columns(cols).select("id", *names)


def analysis_names() -> list[str]:
    """Benchmark names behind the findings."""
    return list(BENCHMARKS)


def detect(works: pl.DataFrame, refs: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    """Boolean columns `<name>|mention` and `<name>|cite` per work."""
    m = mention_flags(works, names).rename({n: f"{n}|mention" for n in names})
    out = works.select("id", "first_year", "has_abstract", "has_refs").join(m, on="id", maintain_order="left")
    cols = []
    for name in names:
        ids = BENCHMARKS[name][2] if name in BENCHMARKS else []
        citing = set(refs.filter(pl.col("ref_id").is_in([f"https://openalex.org/{i}" for i in ids]))["work_id"])
        cols.append(out["id"].is_in(list(citing)).alias(f"{name}|cite"))
    return out.with_columns(cols)


def main() -> None:
    works = (pl.read_parquet(C.PROCESSED / "works.parquet")
             .filter(focused() & pl.col("first_year").is_between(2015, 2026))
             .with_columns(pl.col("abstract").is_not_null().alias("has_abstract"),
                           (pl.col("n_references") > 0).alias("has_refs")))
    refs = pl.read_parquet(C.PROCESSED / "references.parquet").filter(
        pl.col("work_id").is_in(works["id"].implode()))
    names = analysis_names()
    d = detect(works, refs, names)
    d = d.with_columns([(pl.col(f"{n}|mention") | pl.col(f"{n}|cite")).alias(n) for n in names])
    for fam in FAMILIES:
        members = [n for n in names if FAMILY[n] == fam]
        d = d.with_columns(pl.any_horizontal(members).alias(f"fam:{fam}"))
    d = d.with_columns(pl.any_horizontal(names).alias("any_benchmark"),
                       pl.any_horizontal([f"{n}|mention" for n in names]).alias("any_mention"),
                       pl.any_horizontal([f"{n}|cite" for n in names]).alias("any_cite"))

    # Denominator: works where at least one signal could fire.
    ev = d.filter(pl.col("has_abstract") | pl.col("has_refs"))
    agg = [pl.len().alias("works"), pl.col("has_refs").mean().round(3).alias("with_refs"),
           pl.col("any_benchmark").mean().round(3), pl.col("any_mention").mean().round(3),
           pl.col("any_cite").mean().round(3)]
    by_year = (ev.group_by("first_year").agg(agg + [pl.col(n).mean().round(4) for n in names])
               .sort("first_year"))
    fam_year = (ev.group_by("first_year").agg(
        [pl.len().alias("works")] + [pl.col(f"fam:{f}").mean().round(4).alias(f) for f in FAMILIES])
        .sort("first_year"))
    # Same table restricted to works that name or cite at least one benchmark:
    # this is the "mix" of benchmarks, independent of how often abstracts name any.
    mix = (ev.filter(pl.col("any_benchmark")).group_by("first_year").agg(
        [pl.len().alias("works_with_benchmark")] + [pl.col(f"fam:{f}").mean().round(4).alias(f) for f in FAMILIES])
        .sort("first_year"))
    mix_mention = (ev.filter(pl.col("any_mention")).group_by("first_year").agg(
        [pl.len().alias("works_naming_benchmark")]
        + [pl.any_horizontal([f"{n}|mention" for n in names if FAMILY[n] == f]).mean().round(4).alias(f)
           for f in FAMILIES])
        .sort("first_year"))
    signals = pl.DataFrame([{
        "benchmark": n, "family": FAMILY[n], "mention": int(ev[f"{n}|mention"].sum()),
        "cite": int(ev[f"{n}|cite"].sum()), "both": int((ev[f"{n}|mention"] & ev[f"{n}|cite"]).sum()),
        "either": int(ev[n].sum()), "has_cite_anchor": bool(BENCHMARKS[n][2]) if n in BENCHMARKS else False,
    } for n in names])

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    by_year.write_csv(C.RESULTS / "benchmark_by_year.csv")
    fam_year.write_csv(C.RESULTS / "benchmark_family_by_year.csv")
    mix.write_csv(C.RESULTS / "benchmark_mix_by_year.csv")
    mix_mention.write_csv(C.RESULTS / "benchmark_mix_mention_only_by_year.csv")
    signals.write_csv(C.RESULTS / "benchmark_signals.csv")
    audit(works, d, names)
    figure(mix, by_year)

    with pl.Config(tbl_rows=40, tbl_cols=14, tbl_width_chars=220):
        print(by_year.select("first_year", "works", "with_refs", "any_benchmark", "any_mention", "any_cite"))
        print(signals)
        print(mix)
        print(mix_mention)


def audit(works: pl.DataFrame, d: pl.DataFrame, names: list[str], per: int = 4) -> None:
    """Random mention snippets per benchmark, for a manual precision check."""
    rows = []
    for name in names:
        t = text_of().str.replace_all(STRIP[name], " ") if name in STRIP else text_of()
        s = works.select("id", t.alias("text")).join(d.filter(pl.col(f"{name}|mention")).select("id"), on="id")
        if s.height == 0:
            continue
        s = s.sample(min(per, s.height), seed=SEED)
        snip = s.select("id", pl.col("text").str.extract("(?s)(.{0,70}(?:" + rx(BENCHMARKS[name][1]) + ").{0,50})")
                        .alias("snippet"))
        rows += [{"benchmark": name, **r} for r in snip.iter_rows(named=True)]
    pl.DataFrame(rows).write_csv(C.RESULTS / "benchmark_audit.csv")


def figure(mix: pl.DataFrame, by_year: pl.DataFrame) -> None:
    # Before 2018 fewer than 15 works per year name or cite a benchmark.
    m = mix.filter(pl.col("first_year") >= 2018)
    years = m["first_year"].to_list()
    ns = m["works_with_benchmark"].to_list()
    grid = [[m[f][i] for i in range(m.height)] for f in FAMILIES]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150, gridspec_kw={"width_ratios": [2.2, 1]})
    im = axes[0].imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=max(max(r) for r in grid))
    axes[0].set_xticks(range(len(years)), [f"{y}\nn={n}" for y, n in zip(years, ns)], fontsize=8)
    axes[0].set_yticks(range(len(FAMILIES)), FAMILIES)
    for i, r in enumerate(grid):
        for j, v in enumerate(r):
            axes[0].text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                         color="white" if v > 0.35 else "#333")
    axes[0].set_title("Benchmark family, share of works that name or cite any benchmark", loc="left", fontsize=10)
    fig.colorbar(im, ax=axes[0], fraction=0.025, format=matplotlib.ticker.PercentFormatter(1.0))
    b = by_year.filter(pl.col("first_year") >= 2018)
    x = b["first_year"].to_list()
    for col, colr, lab in [("any_benchmark", "#2b5d7d", "name or cite"), ("any_mention", "#6a994e", "name in abstract"),
                           ("any_cite", "#c8553d", "cite dataset paper"), ("with_refs", "#999", "has reference list")]:
        axes[1].plot(x, b[col].to_list(), marker="o", ms=3, lw=1.6, color=colr, label=lab,
                     ls="--" if col == "with_refs" else "-")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[1].set_title("Signal coverage (all focused works)", loc="left", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Which benchmarks continual learning papers use (names in abstracts + dataset citations)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / "benchmarks.png")


if __name__ == "__main__":
    main()
