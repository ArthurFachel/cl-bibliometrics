"""Who drove the shift to LLMs and pre-trained models?

    python -m clbib.drivers

Two questions, answered on PEFT/LLM-tagged works (F2 tag, held-out F1 0.84):

  adopters   Who wrote the PEFT/LLM works, before the wave (first year
             2018-2023) and during it (2024-2026)? Shares are compared with
             the rest of the focused corpus in the same years, so a ratio
             above 1 means a group is over-represented in PEFT/LLM work.
  influence  Whose PEFT/LLM works does the field build on? In-corpus
             citations: a focused work citing a PEFT/LLM work that is also in
             the corpus. Preprint and published versions are merged through
             the dedup clusters before counting.

Influence is computed separately for works first seen in 2018-2023 and in
2024-2025 (2026 works are too young to be cited), and reported three ways,
because raw counts favour older works and are dominated by a few hits:
  raw         share of in-corpus citations to the cohort's PEFT/LLM works
  age-adj.    each work's citations divided by the mean for PEFT/LLM works
              of the same first year, then shared out
  no top 2    raw share without the cohort's two most cited works
with a bootstrap 95% interval over works for the raw share.

Countries use whole counting (a CN-US work counts for both), over works with
at least one known country. "company" means at least one author affiliated
with an institution OpenAlex types as a company.

Outputs (results/):
  drivers_adopters.csv          group shares in PEFT/LLM vs other works, per cohort
  drivers_early_institutions.csv institutions with the most early PEFT/LLM works
  drivers_influence.csv         output share vs citation share per group and cohort
  drivers_citing_year.csv       where citations to PEFT/LLM works go, by citing year
  drivers_top_cited.csv         most cited PEFT/LLM works inside the corpus
Figure: figures/drivers.png
"""
from __future__ import annotations

import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from . import config as C  # noqa: E402
from .corpus import focused  # noqa: E402
from .networks import per_work  # noqa: E402

SEED = 20261022
YEARS = (2018, 2026)
EARLY_END = 2023          # last first year of the "early" cohort
GROUPS = ["CN", "US", "HK", "SG", "GB", "AU", "KR", "DE", "CA", "IN", "IT", "JP"]
N_BOOT = 1000
COHORTS = {"2018-2023": (2018, EARLY_END), "2024-2025": (EARLY_END + 1, 2025)}


def load() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Focused works with an abstract (the PEFT/LLM tag needs one), their
    affiliations, and the in-corpus citation edges."""
    raw = pl.read_parquet(C.PROCESSED / "works.parquet")
    tags = pl.read_parquet(C.PROCESSED / "subareas.parquet").select("id", "peft_llm", "has_abstract")
    auth = pl.read_parquet(C.PROCESSED / "authorships.parquet")
    works = (raw.filter(focused() & pl.col("first_year").is_between(*YEARS))
             .select("id", "first_year", "title", "doi", "cited_by_count")
             .join(tags, on="id").filter(pl.col("has_abstract"))
             .join(per_work(auth), left_on="id", right_on="work_id", how="left")
             .with_columns(pl.col("has_country").fill_null(False), pl.col("has_company").fill_null(False),
                           pl.col("countries").fill_null([]),
                           pl.when(pl.col("first_year") <= EARLY_END).then(pl.lit("early"))
                           .otherwise(pl.lit("wave")).alias("cohort")))
    edges = citation_edges(raw, pl.read_parquet(C.PROCESSED / "references.parquet"), works)
    return works, auth, edges


def citation_edges(raw: pl.DataFrame, refs: pl.DataFrame, works: pl.DataFrame) -> pl.DataFrame:
    """Citing -> cited pairs where both ends are analysed works. Every record of
    a dedup cluster is mapped to the cluster's canonical id first, so citing
    the arXiv version of a paper counts for the published one."""
    canon = (raw.select("id", "cluster_id")
             .join(raw.filter(pl.col("is_canonical")).select("cluster_id", pl.col("id").alias("canon")),
                   on="cluster_id")
             .select("id", "canon"))
    e = (refs.join(canon, left_on="work_id", right_on="id").rename({"canon": "citing"})
         .join(canon, left_on="ref_id", right_on="id").rename({"canon": "cited"})
         .select("citing", "cited").unique()
         .filter(pl.col("citing") != pl.col("cited")))
    ids = works.select("id", "first_year", "peft_llm")
    return (e.join(ids.rename({"id": "citing", "first_year": "y_citing", "peft_llm": "p_citing"}), on="citing")
             .join(ids.rename({"id": "cited", "first_year": "y_cited", "peft_llm": "p_cited"}), on="cited"))


def member(group: str) -> pl.Expr:
    return pl.col("has_company") if group == "company" else pl.col("countries").list.contains(group)


def adopters(works: pl.DataFrame) -> pl.DataFrame:
    k = works.filter(pl.col("has_country"))
    rows = []
    for coh in ["early", "wave"]:
        d = k.filter(pl.col("cohort") == coh)
        p, o = d.filter(pl.col("peft_llm")), d.filter(~pl.col("peft_llm"))
        for g in GROUPS + ["company"]:
            sp, so = p.select(member(g).mean()).item(), o.select(member(g).mean()).item()
            rows.append({"cohort": coh, "group": g, "peft_llm": round(sp, 3), "other": round(so, 3),
                         "ratio": round(sp / so, 2) if so else None, "n_peft_llm": p.height, "n_other": o.height})
    return pl.DataFrame(rows)


def early_institutions(works: pl.DataFrame, auth: pl.DataFrame, top: int = 25) -> pl.DataFrame:
    early = works.filter((pl.col("cohort") == "early") & pl.col("peft_llm")).select(pl.col("id").alias("work_id"))
    return (auth.join(early, on="work_id").filter(pl.col("institution_id").is_not_null())
            .unique(["work_id", "institution_id"])
            .group_by("institution_id").agg(pl.col("institution").first(), pl.col("country").first(),
                                            pl.col("institution_type").first(), pl.len().alias("early_peft_llm_works"))
            .sort("early_peft_llm_works", "institution", descending=[True, False]).head(top)
            .drop("institution_id"))


def cites_per_work(works: pl.DataFrame, edges: pl.DataFrame) -> pl.DataFrame:
    """PEFT/LLM works with their in-corpus citation count and an age-adjusted score."""
    cc = edges.filter(pl.col("p_cited")).group_by("cited").agg(pl.len().alias("cites"),
                                                                 pl.col("p_citing").sum().alias("from_peft_llm"))
    p = (works.filter(pl.col("peft_llm"))
         .join(cc, left_on="id", right_on="cited", how="left")
         .with_columns(pl.col("cites").fill_null(0), pl.col("from_peft_llm").fill_null(0)))
    year_mean = p.group_by("first_year").agg(pl.col("cites").mean().alias("year_mean"))
    return (p.join(year_mean, on="first_year")
            .with_columns(pl.when(pl.col("year_mean") > 0).then(pl.col("cites") / pl.col("year_mean"))
                          .otherwise(0.0).alias("age_adj")))


def share(d: pl.DataFrame, g: str, col: str) -> float:
    tot = d[col].sum()
    return d.filter(member(g))[col].sum() / tot if tot else float("nan")


def influence(p: pl.DataFrame) -> pl.DataFrame:
    """Output vs citation share per group, separately for each cited cohort.
    Works first seen in 2026 are left out: they have had almost no time to be
    cited (22 in-corpus citations in total)."""
    rows = []
    for cohort, (a, b) in COHORTS.items():
        k = p.filter(pl.col("has_country") & pl.col("first_year").is_between(a, b))
        top2 = set(k.sort("cites", descending=True).head(2)["id"].to_list())
        k_no2 = k.filter(~pl.col("id").is_in(list(top2)))
        rng = random.Random(SEED)
        ix = list(range(k.height))
        boots = [k[[rng.choice(ix) for _ in ix]] for _ in range(N_BOOT)]
        print(f"{cohort}: {k.height} PEFT/LLM works with a known country, {k['cites'].sum()} in-corpus citations; "
              f"top 2: {k.sort('cites', descending=True).head(2)['title'].to_list()}")
        for g in GROUPS + ["company"]:
            bs = sorted(share(x, g, "cites") for x in boots)
            rows.append({"cited_cohort": cohort, "group": g, "works": k.height, "citations": int(k["cites"].sum()),
                         "output_share": round(k.select(member(g).mean()).item(), 3),
                         "citation_share": round(share(k, g, "cites"), 3),
                         "citation_ci": f"{bs[int(0.025 * N_BOOT)]:.2f}-{bs[int(0.975 * N_BOOT) - 1]:.2f}",
                         "age_adjusted_share": round(share(k, g, "age_adj"), 3),
                         "citation_share_no_top2": round(share(k_no2, g, "cites"), 3)})
    return pl.DataFrame(rows).with_columns(
        (pl.col("citation_share") / pl.col("output_share")).round(2).alias("citation_to_output"))


def citing_year(p: pl.DataFrame, edges: pl.DataFrame) -> pl.DataFrame:
    """Of the citations made in year y to PEFT/LLM works with a known country,
    the share going to each group (whole counting)."""
    k = p.filter(pl.col("has_country")).select("id", "countries", "has_company")
    e = edges.filter(pl.col("p_cited")).join(k, left_on="cited", right_on="id")
    rows = []
    for y, d in e.group_by("y_citing"):
        r = {"citing_year": y[0], "citations": d.height}
        for g in ["CN", "US", "company"]:
            r[g] = round(d.select(member(g).mean()).item(), 3)
        r["cited_first_year_median"] = d["y_cited"].median()
        rows.append(r)
    return pl.DataFrame(rows).sort("citing_year").filter(pl.col("citing_year") >= 2021)


def top_cited(p: pl.DataFrame, auth: pl.DataFrame, n: int = 30) -> pl.DataFrame:
    inst = (auth.filter(pl.col("institution").is_not_null()).unique(["work_id", "institution_id"])
            .group_by("work_id").agg(pl.col("institution").sort().str.join("; ").alias("institutions")))
    return (p.sort("cites", "cited_by_count", descending=True).head(n)
            .join(inst, left_on="id", right_on="work_id", how="left", maintain_order="left")
            .with_columns(pl.col("countries").list.sort().list.join(" ").alias("countries"))
            .select("first_year", "title", "cites", "from_peft_llm", "cited_by_count", "countries",
                    "has_company", "institutions", "doi", "id"))


def figure(infl: pl.DataFrame, cy: pl.DataFrame) -> None:
    groups = ["CN", "US", "company"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    xs, labels = [], []
    for ci, cohort in enumerate(COHORTS):
        d = {r["group"]: r for r in infl.filter(pl.col("cited_cohort") == cohort).iter_rows(named=True)}
        for gi, g in enumerate(groups):
            x = ci * (len(groups) + 1) + gi
            xs.append(x)
            labels.append(f"{g}\n{cohort}")
            ax.bar(x - 0.2, d[g]["output_share"], 0.4, color="#b9b9b9",
                   label="share of PEFT/LLM works" if (ci, gi) == (0, 0) else None)
            ax.bar(x + 0.2, d[g]["citation_share"], 0.4, color="#c0504d",
                   label="share of in-corpus citations to them" if (ci, gi) == (0, 0) else None)
    ax.set_xticks(xs, labels, fontsize=8)
    ax.set_title("Whose PEFT/LLM work gets built on, by when it appeared")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for g, col in [("US", "#2f5d8a"), ("CN", "#c0504d"), ("company", "#8e6c8a")]:
        ax.plot(cy["citing_year"], cy[g], marker="o", color=col, label=g)
    ax.set_xlabel("year of the citing work")
    ax.set_ylabel("share of citations to PEFT/LLM works")
    ax.set_title("Where citations to PEFT/LLM work go")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / "drivers.png", dpi=150)


def main() -> None:
    works, auth, edges = load()
    p = cites_per_work(works, edges)
    adopt = adopters(works)
    inst = early_institutions(works, auth)
    infl = influence(p)
    cy = citing_year(p, edges)
    top = top_cited(p, auth)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    adopt.write_csv(C.RESULTS / "drivers_adopters.csv")
    inst.write_csv(C.RESULTS / "drivers_early_institutions.csv")
    infl.write_csv(C.RESULTS / "drivers_influence.csv")
    cy.write_csv(C.RESULTS / "drivers_citing_year.csv")
    top.write_csv(C.RESULTS / "drivers_top_cited.csv")
    figure(infl, cy)
    cov = works.group_by("cohort", "peft_llm").agg(pl.len().alias("works"), pl.col("has_country").mean().round(3))
    with pl.Config(tbl_rows=60, tbl_width_chars=220, fmt_str_lengths=60):
        print(cov.sort("cohort", "peft_llm"))
        print(adopt.pivot(on="cohort", index="group", values=["peft_llm", "other", "ratio"]))
        print(inst)
        print(infl)
        print(cy)
        print(top.select("first_year", "title", "cites", "countries", "has_company").head(15))


if __name__ == "__main__":
    main()
