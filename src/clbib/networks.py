"""F3: who does continual learning research, and who works with whom.

Unit of analysis: focused, canonical, in-scope works (same corpus as F1/F2).
Affiliations come from OpenAlex authorships (machine-assigned, often missing
on preprints), so every share below is over works with at least one known
country or institution, and coverage is reported next to it.

Counting is whole counting: a work with authors in CN and US counts once for
each country.

Outputs (results/):
  coverage_by_year.csv        affiliation coverage per year
  country_by_period.csv       share of works per country and period
  collab_by_year.csv          international and industry co-authorship rates
  peft_llm_by_country.csv     who writes the PEFT/LLM papers (2023-2026)
  institution_centrality.csv  institution co-authorship network, top nodes
  institution_communities.csv Louvain communities of the institution network
  brazil_works.csv            every focused work with a Brazilian affiliation
  network_summary.csv         size and modularity of the networks
Figure: figures/countries.png
"""
from __future__ import annotations

import itertools
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import polars as pl  # noqa: E402

from . import config as C  # noqa: E402
from .corpus import focused  # noqa: E402

SEED = 20261020
PERIODS = [(2015, 2019, "2015-2019"), (2020, 2022, "2020-2022"), (2023, 2026, "2023-2026")]
# Works with more institutions than this are skipped when building the
# institution graph: a 40-institution consortium paper would add 780 edges
# and say little about who actually works together.
MAX_INST_PER_WORK = 20
FOCUS_COUNTRY = "BR"


def load() -> tuple[pl.DataFrame, pl.DataFrame]:
    works = pl.read_parquet(C.PROCESSED / "works.parquet").filter(
        focused() & pl.col("first_year").is_between(2015, 2026))
    tags = pl.read_parquet(C.PROCESSED / "subareas.parquet").select("id", "peft_llm", "has_abstract")
    works = works.select("id", "first_year", "title", "is_arxiv", "cited_by_count").join(tags, on="id", how="left")
    auth = (pl.read_parquet(C.PROCESSED / "authorships.parquet")
            .join(works.select(pl.col("id").alias("work_id"), "first_year"), on="work_id"))
    return works, auth


def period(col: str = "first_year") -> pl.Expr:
    e = pl.lit(None, dtype=pl.Utf8)
    for a, b, name in reversed(PERIODS):
        e = pl.when(pl.col(col).is_between(a, b)).then(pl.lit(name)).otherwise(e)
    return e


def per_work(auth: pl.DataFrame) -> pl.DataFrame:
    """One row per work: sets of countries / institutions, industry flag."""
    return (auth.group_by("work_id").agg(
        pl.col("country").drop_nulls().unique().alias("countries"),
        pl.col("institution_id").drop_nulls().unique().alias("inst_ids"),
        (pl.col("institution_type") == "company").any().alias("has_company"),
    ).with_columns(
        (pl.col("countries").list.len() > 0).alias("has_country"),
        (pl.col("inst_ids").list.len() > 0).alias("has_inst"),
    ))


def coverage(works: pl.DataFrame, pw: pl.DataFrame) -> pl.DataFrame:
    return (works.join(pw, left_on="id", right_on="work_id", how="left")
            .group_by("first_year").agg(
                pl.len().alias("works"),
                pl.col("has_country").fill_null(False).mean().round(3).alias("with_country"),
                pl.col("has_country").fill_null(False).filter(pl.col("is_arxiv")).mean().round(3)
                .alias("with_country_arxiv"),
                pl.col("has_country").fill_null(False).filter(~pl.col("is_arxiv")).mean().round(3)
                .alias("with_country_not_arxiv"))
            .sort("first_year"))


def countries(works: pl.DataFrame, pw: pl.DataFrame, top: int = 12) -> pl.DataFrame:
    known = (works.join(pw.filter(pl.col("has_country")), left_on="id", right_on="work_id")
             .with_columns(period().alias("period")))
    denom = known.group_by("period").agg(pl.len().alias("n_period"))
    long = (known.explode("countries").rename({"countries": "country"})
            .group_by("period", "country").agg(pl.len().alias("works"))
            .join(denom, on="period").with_columns((pl.col("works") / pl.col("n_period")).round(4).alias("share")))
    overall = long.group_by("country").agg(pl.col("works").sum()).sort("works", descending=True)
    overall = overall.with_row_index("rank", offset=1)
    keep = set(overall.head(top)["country"].to_list()) | {FOCUS_COUNTRY}
    wide = (long.filter(pl.col("country").is_in(list(keep)))
            .pivot(on="period", index="country", values="share")
            .join(overall, on="country").sort("rank")
            .select("rank", "country", pl.col("works").alias("works_2015_2026"), *[p for *_, p in PERIODS]))
    return wide


def collab(works: pl.DataFrame, pw: pl.DataFrame) -> pl.DataFrame:
    j = works.join(pw.filter(pl.col("has_country")), left_on="id", right_on="work_id")
    return (j.group_by("first_year").agg(
        pl.len().alias("works_with_country"),
        (pl.col("countries").list.len() > 1).mean().round(3).alias("international"),
        pl.col("has_company").mean().round(3).alias("with_company"),
        pl.col("countries").list.contains("CN").mean().round(3).alias("with_CN"),
        pl.col("countries").list.contains("US").mean().round(3).alias("with_US"),
    ).sort("first_year"))


def peft_by_country(works: pl.DataFrame, pw: pl.DataFrame, top: int = 10) -> pl.DataFrame:
    """Country and industry shares in PEFT/LLM works vs the rest, 2023-2026."""
    j = (works.filter((pl.col("first_year") >= 2023) & pl.col("has_abstract").fill_null(False))
         .join(pw.filter(pl.col("has_country")), left_on="id", right_on="work_id"))
    groups = {"peft_llm": j.filter(pl.col("peft_llm")), "other": j.filter(~pl.col("peft_llm"))}
    rows = []
    cs = Counter(c for cl in j["countries"].to_list() for c in cl)
    for c, _ in cs.most_common(top) + ([(FOCUS_COUNTRY, 0)] if FOCUS_COUNTRY not in dict(cs.most_common(top)) else []):
        r = {"group": c}
        for g, d in groups.items():
            r[g] = round(d["countries"].list.contains(c).mean(), 3)
        rows.append(r)
    r = {"group": "any company"}
    for g, d in groups.items():
        r[g] = round(d["has_company"].mean(), 3)
    rows.append(r)
    r = {"group": "n works"}
    for g, d in groups.items():
        r[g] = d.height
    rows.append(r)
    return pl.DataFrame(rows).with_columns((pl.col("peft_llm") / pl.col("other")).round(2).alias("ratio"))


def institution_graph(pw: pl.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for insts in pw["inst_ids"].to_list():
        insts = sorted(set(insts))
        if not insts or len(insts) > MAX_INST_PER_WORK:
            continue
        for i in insts:
            G.add_node(i)
            G.nodes[i]["works"] = G.nodes[i].get("works", 0) + 1
        for a, b in itertools.combinations(insts, 2):
            w = G.get_edge_data(a, b, {"weight": 0})["weight"]
            G.add_edge(a, b, weight=w + 1)
    return G


def author_graph(auth: pl.DataFrame) -> nx.Graph:
    G = nx.Graph()
    per = auth.filter(pl.col("author_id").is_not_null()).group_by("work_id").agg(pl.col("author_id").unique())
    for au in per["author_id"].to_list():
        if len(au) > 30:
            continue
        G.add_nodes_from(au)
        for a, b in itertools.combinations(sorted(au), 2):
            w = G.get_edge_data(a, b, {"weight": 0})["weight"]
            G.add_edge(a, b, weight=w + 1)
    return G


def main() -> None:
    works, auth = load()
    pw = per_work(auth)
    C.RESULTS.mkdir(parents=True, exist_ok=True)

    cov = coverage(works, pw)
    cov.write_csv(C.RESULTS / "coverage_by_year.csv")
    ctry = countries(works, pw)
    ctry.write_csv(C.RESULTS / "country_by_period.csv")
    col = collab(works, pw)
    col.write_csv(C.RESULTS / "collab_by_year.csv")
    peft = peft_by_country(works, pw)
    peft.write_csv(C.RESULTS / "peft_llm_by_country.csv")

    # Institution network
    names = (auth.filter(pl.col("institution_id").is_not_null())
             .group_by("institution_id").agg(pl.col("institution").first(), pl.col("country").first(),
                                             pl.col("institution_type").first()))
    meta = {r["institution_id"]: r for r in names.iter_rows(named=True)}
    G = institution_graph(pw)
    giant = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    comms = nx.community.louvain_communities(giant, weight="weight", seed=SEED)
    modularity = nx.community.modularity(giant, comms, weight="weight")
    # eigenvector centrality is pure Python in networkx (pagerank needs scipy)
    pr = nx.eigenvector_centrality(giant, weight="weight", max_iter=2000, tol=1e-8)
    strength = dict(giant.degree(weight="weight"))
    comm_of = {n: i for i, c in enumerate(sorted(comms, key=len, reverse=True)) for n in c}
    cent = pl.DataFrame([{
        "institution": meta.get(n, {}).get("institution"), "country": meta.get(n, {}).get("country"),
        "type": meta.get(n, {}).get("institution_type"), "works": G.nodes[n]["works"],
        "partners": giant.degree(n), "co_papers": strength[n], "eigenvector": round(pr[n], 5),
        "community": comm_of[n], "institution_id": n,
    } for n in giant.nodes]).sort("eigenvector", descending=True)
    cent.write_csv(C.RESULTS / "institution_centrality.csv")

    crow = []
    for i, c in enumerate(sorted(comms, key=len, reverse=True)[:12]):
        sub = cent.filter(pl.col("community") == i)
        cc = Counter(sub["country"].drop_nulls().to_list())
        crow.append({"community": i, "institutions": len(c),
                     "top_countries": ", ".join(f"{k} {v}" for k, v in cc.most_common(3)),
                     "top_institutions": "; ".join(sub.head(5)["institution"].to_list())})
    pl.DataFrame(crow).write_csv(C.RESULTS / "institution_communities.csv")

    A = author_graph(auth)
    a_giant = max(nx.connected_components(A), key=len)
    summary = pl.DataFrame([
        {"network": "institutions", "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
         "giant_component_share": round(giant.number_of_nodes() / G.number_of_nodes(), 3),
         "louvain_communities": len(comms), "modularity": round(modularity, 3)},
        {"network": "authors", "nodes": A.number_of_nodes(), "edges": A.number_of_edges(),
         "giant_component_share": round(len(a_giant) / A.number_of_nodes(), 3),
         "louvain_communities": None, "modularity": None},
    ])
    summary.write_csv(C.RESULTS / "network_summary.csv")

    # Brazil
    br_ids = pw.filter(pl.col("countries").list.contains(FOCUS_COUNTRY))["work_id"]
    br = (auth.filter(pl.col("work_id").is_in(br_ids.implode()))
          .group_by("work_id").agg(
              pl.col("institution").filter(pl.col("country") == FOCUS_COUNTRY).drop_nulls().unique().sort()
              .str.join("; ").alias("br_institutions"),
              pl.col("country").filter(pl.col("country") != FOCUS_COUNTRY).drop_nulls().unique().sort()
              .str.join(", ").alias("partner_countries"))
          .join(works.select(pl.col("id").alias("work_id"), "first_year", "title", "peft_llm", "cited_by_count"),
                on="work_id").sort("first_year", "work_id"))
    br.write_csv(C.RESULTS / "brazil_works.csv")

    figure(ctry, col)

    with pl.Config(tbl_rows=30, tbl_cols=12, tbl_width_chars=200, fmt_str_lengths=60):
        print(cov.filter(pl.col("first_year") >= 2018))
        print(ctry)
        print(col.filter(pl.col("first_year") >= 2018))
        print(peft)
        print(summary)
        print(cent.head(15).drop("institution_id"))
        print(pl.DataFrame(crow))
        print(br.drop("work_id"))


def figure(ctry: pl.DataFrame, col: pl.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    ps = [p for *_, p in PERIODS]
    top = ctry.head(8)
    for i, p in enumerate(ps):
        axes[0].barh([y + i * 0.27 for y in range(top.height)], top[p].to_list(), height=0.27, label=p,
                     color=["#9db4c0", "#5c8aa6", "#2b5d7d"][i])
    axes[0].set_yticks([y + 0.27 for y in range(top.height)], top["country"].to_list())
    axes[0].invert_yaxis()
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_title("Share of works with an author in each country", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    c = col.filter(pl.col("first_year") >= 2018)  # < 60 works with known country before 2018
    x = c["first_year"].to_list()
    for name, colr, lab in [("with_CN", "#c8553d", "China"), ("with_US", "#2b5d7d", "United States"),
                            ("international", "#6a994e", "international co-authorship"),
                            ("with_company", "#8e6c8a", "any company author")]:
        axes[1].plot(x, c[name].to_list(), marker="o", ms=3, lw=1.8, color=colr, label=lab)
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[1].set_title("Per year, works with known affiliation", loc="left", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Who writes continual learning papers (OpenAlex affiliations, whole counting)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / "countries.png")


if __name__ == "__main__":
    main()
