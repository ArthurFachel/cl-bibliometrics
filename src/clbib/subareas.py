"""F2: rule-based, multi-label sub-area tagging of the corpus.

Four independent axes, so a paper can be e.g. replay + PEFT/LLM + medical:
  method    replay, regularization, architecture, optimization, peft_llm, plasticity
  study     theory, survey_benchmark
  setting   online, federated, rl
  domain    application (CL applied to a named non-ML domain)

Rules are transparent regexes over title + abstract; they are validated against
a labelled sample (validation/), not assumed correct.

Outputs:
  data/processed/subareas.parquet
  results/subarea_by_year.csv
  figures/subareas.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from . import config as C  # noqa: E402

RULES: dict[str, tuple[str, str]] = {
    # name: (axis, regex)
    "replay": ("method", r"replay|rehears|exemplar|memory buffer|episodic memory|coreset|"
                         r"experience buffer|pseudo-?sample"),
    "regularization": ("method", r"elastic weight consolidation|\bewc\b|synaptic intelligence|"
                                 r"memory aware synapses|regulari[sz]|distill|learning without forgetting|"
                                 r"\blwf\b|fisher information|parameter importance|importance weight"),
    "architecture": ("method", r"dynamic architecture|network expansion|expandable|expansion|"
                               r"progressive (neural )?network|parameter isolation|sub-?network|"
                               r"\bmask(s|ing|ed)?\b|packnet|mixture[- ]of[- ]experts|\bmoe\b|modular|"
                               r"\bexperts?\b|ensemble|\bgat(e|es|ing)\b|neural architecture search|"
                               r"task-specific (module|head|parameter|layer|branch|component)s?|"
                               r"\bgrow(s|ing|th)? (the )?(network|model|capacity)|multi-head|\bbranch(es)?\b"),
    "optimization": ("method", r"gradient projection|orthogonal|null[- ]space|subspace|"
                               r"gradient episodic memory|\ba-?gem\b|sharpness|flat minima|loss landscape"),
    "peft_llm": ("method", r"large language model|\bllms?\b|language model|\blora\b|low-rank adaptation|"
                           r"prompt|adapter|instruction[- ]tun|vision[- ]language|\bvlms?\b|"
                           r"foundation model|pre-?trained (model|transformer)|continual pre-?training"),
    "plasticity": ("method", r"loss of plasticity|plasticity loss|dormant neuron|primacy bias|"
                             r"trainability|re-?initiali[sz]|\breset"),
    "theory": ("study", r"theoretical (analysis|result|guarantee|understanding|insight|characteri[sz])|"
                        r"theoretically|we prove|theorem|upper bound|lower bound|generali[sz]ation bound|"
                        r"provabl|closed-form|convergence (analysis|guarantee|rate)|theory of"),
    "survey_benchmark": ("study", r"\bsurvey\b|systematic review|literature review|comprehensive review|"
                                  r"(introduce|propose|present|release|construct|build|create|collect|define)s? "
                                  r"(a |an |the )?(new |novel |large-scale |comprehensive |unified |realistic )*"
                                  r"(benchmark|dataset|evaluation protocol)|benchmarks? (for|to evaluate|suite)"),
    "online": ("setting", r"online (continual|learning|setting|scenario|fashion|manner|incremental|class|"
                          r"adaptation|update)|task-free|task-agnostic|single[- ]pass|one[- ]pass|"
                          r"concept drift|non-stationary|real-time|streaming (data|setting|scenario)"),
    "federated": ("setting", r"federated"),
    "rl": ("setting", r"reinforcement learning|policy gradient|actor-critic|q-learning|\brl\b|"
                      r"markov decision"),
    "application": ("domain", r"medical|clinical|patient|\becg\b|\beeg\b|mri\b|fault diagnos|"
                              r"intrusion|anomaly detection|remote sensing|hyperspectral|radar|"
                              r"autonomous driving|vehicle|speech|recommend|traffic|power (system|grid)|"
                              r"industrial|wireless|\biot\b|malware|sensor|manufactur|battery|satellite|"
                              r"audio|sound|spoken|dialog|document image|face|forgery|deepfake|food|"
                              r"wearable|activity recognition|tracking control|\bqo[se]\b|cyber|"
                              r"edge comput|re-?identification"),
}
METHODS = [k for k, (ax, _) in RULES.items() if ax == "method"]
# Extra rules matched against the TITLE only (OR-ed with the main rule).
TITLE_RULES = {
    "survey_benchmark": r"\b(benchmark|dataset|survey|review)s?\b",
}


def tag(df: pl.DataFrame) -> pl.DataFrame:
    text = pl.concat_str([pl.col("title").fill_null(""), pl.lit(" "),
                          pl.col("abstract").fill_null("")]).str.to_lowercase()
    title = pl.col("title").fill_null("").str.to_lowercase()
    exprs = []
    for name, (_, rx) in RULES.items():
        e = text.str.contains(rx)
        if name in TITLE_RULES:
            e = e | title.str.contains(TITLE_RULES[name])
        exprs.append(e.alias(name))
    return df.with_columns(exprs)


def main() -> None:
    works = pl.read_parquet(C.PROCESSED / "works.parquet")
    final = works.filter(pl.col("in_scope") & pl.col("is_canonical") & pl.col("is_focused"))
    tagged = tag(final.select("id", "first_year", "title", "abstract", "is_arxiv"))
    tagged = tagged.with_columns(
        pl.col("abstract").is_not_null().alias("has_abstract"),
        (~pl.any_horizontal(METHODS)).alias("no_method"),
    )
    tagged.drop("title", "abstract").write_parquet(C.PROCESSED / "subareas.parquet")

    labels = list(RULES) + ["no_method"]
    yr = (tagged.filter(pl.col("has_abstract") & pl.col("first_year").is_between(2015, 2026))
          .group_by("first_year")
          .agg([pl.len().alias("works")] + [pl.col(l).mean().round(3).alias(l) for l in labels])
          .sort("first_year"))
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    yr.write_csv(C.RESULTS / "subarea_by_year.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150, sharex=True)
    x = yr["first_year"].to_list()
    palette = ["#2b5d7d", "#c8553d", "#6a994e", "#8e6c8a", "#e0a458", "#4d908e"]
    for name, col in zip(METHODS, palette):
        axes[0].plot(x, yr[name].to_list(), marker="o", ms=3, lw=1.8, color=col, label=name)
    axes[0].set_title("Method family (share of works, multi-label)", loc="left", fontsize=10)
    for name, col in zip(["application", "rl", "online", "federated", "theory", "survey_benchmark"], palette):
        axes[1].plot(x, yr[name].to_list(), marker="o", ms=3, lw=1.8, color=col, label=name)
    axes[1].set_title("Setting, domain and study type", loc="left", fontsize=10)
    for ax in axes:
        ax.set_xticks([y for y in x if y % 2 == 0])
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Continual learning sub-areas over time (rule-based tags, see validation/)",
                 x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.01, f"focused corpus, n = {tagged.filter(pl.col('has_abstract')).height:,} works with abstract | "
             f"{x[-1]} partial", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / "subareas.png")

    cov = tagged["has_abstract"].mean()
    print(f"tagged {tagged.height} works; abstract coverage {cov:.1%}; "
          f"no method tag: {tagged.filter(pl.col('has_abstract'))['no_method'].mean():.1%}")
    with pl.Config(tbl_cols=20, tbl_width_chars=220):
        print(yr.select(["first_year", "works"] + METHODS + ["application"]))


if __name__ == "__main__":
    main()
