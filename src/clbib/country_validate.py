"""F3 validation: are OpenAlex country attributions right, and is missing data biased?

  python -m clbib.country_validate sample   # draw works, fetch Crossref affiliations
  python -m clbib.country_validate label --labeller opus|sonnet|deepseek [--workers N]
  python -m clbib.country_validate score

Independent source: the affiliation strings publishers deposit in Crossref
(raw text such as "Hunan University, Changsha, China"). OpenAlex country codes
come from its own institution matching, so agreement between the two is real
evidence. Three LLM labellers turn each work's affiliation strings into
countries; they see only those strings, never the OpenAlex codes.

Two strata, both from the focused analysis corpus (2018-2026) with a
non-arXiv DOI:
  known   OpenAlex has at least one country -> precision/recall of OpenAlex
  missing OpenAlex has no country           -> does the missing part of the
                                               corpus have a different mix?
Only works whose Crossref record has affiliations are labelled, which favours
publishers that deposit them (mostly IEEE, Springer, Elsevier); arXiv-only
works can't be checked this way.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import polars as pl

from . import config as C
from . import llm_label
from .corpus import focused
from .llm_label import kappa

VAL = C.ROOT / "validation"
RAW = C.ROOT / "data" / "raw" / "crossref"
SAMPLE = "countries_to_label.csv"
SEED = 20261021
DRAW = {"known": 300, "missing": 200}

COUNTRIES = ["CN", "US", "GB", "DE", "KR", "IN", "IT", "SG", "AU", "CA", "HK", "FR", "JP", "BR"]
LABELS = [c.lower() for c in COUNTRIES] + ["other"]
LABELLERS = {"opus": "labels_country_opus.csv", "sonnet": "labels_country_sonnet.csv",
             "deepseek": "labels_country_deepseek.csv"}
MODELS = {"opus": "hermes:claude-opus-5-5", "sonnet": "hermes:claude-sonnet-5-5",
          "deepseek": "deepseek-ai/DeepSeek-V4-Flash"}

GUIDE = """You receive the author affiliation strings of one research paper, as deposited by the publisher (the "Abstract" field holds them, one per line; the "Title" field is a placeholder). Decide which countries the affiliations are located in. Use the institution and city when the country is not written (e.g. "Tsinghua University" = China, "MIT" = United States).
Return 1 for each country that appears in at least one affiliation, else 0:
cn China (mainland), us United States, gb United Kingdom, de Germany, kr South Korea, in India, it Italy, sg Singapore, au Australia, ca Canada, hk Hong Kong (separate from cn), fr France, jp Japan, br Brazil, other any other country or region (including Taiwan and Macau).
If an affiliation is too vague to place, ignore it.
Answer with one JSON object and nothing else: {"cn":0|1, ... all 15 keys ..., "notes":"short reason for doubtful calls or empty"}"""


def crossref(doi: str) -> dict | None:
    key = doi.replace("https://doi.org/", "")
    path = RAW / (urllib.parse.quote(key, safe="") + ".json")
    if path.exists():
        return json.loads(path.read_text())
    url = "https://api.crossref.org/works/" + urllib.parse.quote(key)
    req = urllib.request.Request(url, headers={"User-Agent": f"cl-bibliometrics (mailto:{C.MAILTO})"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                msg = json.load(r)["message"]
            RAW.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(msg))
            return msg
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(3 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError):
            time.sleep(3 * (attempt + 1))
    return None


def affiliations(msg: dict | None) -> list[str]:
    if not msg:
        return []
    out = []
    for au in msg.get("author", []):
        for a in au.get("affiliation", []):
            n = (a.get("name") or "").strip()
            if n and n not in out:
                out.append(n)
    return out


def openalex_countries() -> pl.DataFrame:
    works = (pl.read_parquet(C.PROCESSED / "works.parquet")
             .filter(focused() & pl.col("first_year").is_between(2018, 2026) & pl.col("doi").is_not_null()
                     & ~pl.col("doi").str.contains("(?i)arxiv"))
             .select("id", "first_year", "doi"))
    auth = pl.read_parquet(C.PROCESSED / "authorships.parquet")
    cc = (auth.group_by("work_id").agg(pl.col("country").drop_nulls().unique().alias("oa_countries"))
          .rename({"work_id": "id"}))
    return (works.join(cc, on="id", how="left")
            .with_columns(pl.col("oa_countries").fill_null([]))
            .with_columns((pl.col("oa_countries").list.len() > 0).alias("oa_known")))


def sample() -> None:
    out = VAL / SAMPLE
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to redraw.")
    pop = openalex_countries()
    rows = []
    for stratum, n in DRAW.items():
        part = pop.filter(pl.col("oa_known") == (stratum == "known")).sample(n, seed=SEED)
        for r in part.iter_rows(named=True):
            aff = affiliations(crossref(r["doi"]))
            if aff:
                rows.append({"id": r["id"], "first_year": r["first_year"], "stratum": stratum,
                             "title": "(affiliations)", "abstract": "\n".join(aff)})
            time.sleep(0.2)
    s = pl.DataFrame(rows).sample(fraction=1.0, shuffle=True, seed=SEED)
    s.write_csv(out)
    print(f"population: {pop.height} works with non-arXiv DOI, {int(pop['oa_known'].sum())} with OpenAlex country")
    print(s.group_by("stratum").len())


def load_labels() -> dict[str, pl.DataFrame]:
    labs = {}
    for name, f in LABELLERS.items():
        p = VAL / f
        if p.exists():
            labs[name] = pl.read_csv(p, infer_schema_length=None).select(
                ["id"] + [pl.col(k).cast(pl.Int8) for k in LABELS])
    return labs


def score() -> None:
    s = pl.read_csv(VAL / SAMPLE, infer_schema_length=None).select("id", "stratum")
    oa = openalex_countries().select("id", "oa_countries")
    labs = load_labels()
    wide = s.join(oa, on="id", how="left", maintain_order="left")
    for name, lab in labs.items():
        wide = wide.join(lab.rename({k: f"{k}__{name}" for k in LABELS}), on="id", how="inner",
                         maintain_order="left")
    n_lab = len(labs)
    wide = wide.with_columns([
        (pl.sum_horizontal([pl.col(f"{k}__{n}") for n in labs]) * 2 > n_lab).cast(pl.Int8).alias(k)
        for k in LABELS])
    known = wide.filter(pl.col("stratum") == "known")
    missing = wide.filter(pl.col("stratum") == "missing")

    rows = []
    for c in COUNTRIES:
        k = c.lower()
        y = known[k].to_list()                                   # majority of LLM readers of Crossref
        p = known["oa_countries"].list.contains(c).cast(pl.Int8).to_list()  # OpenAlex
        tp = sum(a and b for a, b in zip(y, p))
        fp = sum((not a) and b for a, b in zip(y, p))
        fn = sum(a and (not b) for a, b in zip(y, p))
        po, ka = kappa(y, p)
        rows.append({"country": c, "crossref_pos": sum(y), "openalex_pos": sum(p),
                     "precision": round(tp / (tp + fp), 2) if tp + fp else None,
                     "recall": round(tp / (tp + fn), 2) if tp + fn else None,
                     "kappa": None if ka is None else round(ka, 2),
                     "share_known": round(sum(y) / known.height, 3),
                     "share_missing": round(missing[k].sum() / missing.height, 3) if missing.height else None})
    res = pl.DataFrame(rows)
    res.write_csv(C.RESULTS / "country_validation.csv")

    exact = (known.with_columns(pl.concat_list([pl.when(pl.col(c.lower()) == 1).then(pl.lit(c))
                                                for c in COUNTRIES]).list.drop_nulls().list.sort().alias("cr"),
                                pl.col("oa_countries").list.eval(pl.element().filter(pl.element().is_in(COUNTRIES)))
                                .list.sort().alias("oa"))
             .select((pl.col("cr") == pl.col("oa")).mean()).item())

    agree = []
    names = list(labs)
    for k in LABELS:
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                j = labs[a].select("id", k).join(labs[b].select("id", k), on="id", suffix="_b",
                                                 maintain_order="left")
                po, ka = kappa(j[k].to_list(), j[k + "_b"].to_list())
                agree.append({"country": k, "pair": f"{a}-{b}", "agree": round(po, 3),
                              "kappa": None if ka is None else round(ka, 2)})
    pl.DataFrame(agree).write_csv(C.RESULTS / "country_validation_agreement.csv")
    with pl.Config(tbl_rows=40, tbl_width_chars=200):
        print(f"labelled works: known {known.height}, missing {missing.height}; labellers: {names}")
        print(res)
        print(f"exact match of the 14-country set (known stratum): {exact:.2f}")
        print(pl.DataFrame(agree).pivot(on="pair", index="country", values="agree"))


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
