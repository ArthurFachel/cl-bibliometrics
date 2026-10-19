"""Turn raw OpenAlex pages into tidy parquet tables.

Outputs (data/processed/):
  works.parquet         one row per OpenAlex work, with scope/dedup flags
  authorships.parquet   work x author x institution (long)
  references.parquet    work -> referenced work (edges)

Preprint/published duplicates: OpenAlex merges many arXiv preprints into the
published record, but not all. We cluster remaining works by normalized title
(plus first-author surname) within a 3-year window and mark one canonical row
per cluster: the earliest publication year is kept as `first_year` and the
non-preprint record is preferred as canonical.
"""
from __future__ import annotations

import gzip
import json
import re
import unicodedata
from datetime import date

import polars as pl

from . import config as C


def reconstruct_abstract(inv: dict | None) -> str | None:
    if not inv:
        return None
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos)) or None


def norm_title(t: str | None) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"<[^>]+>", " ", t)  # stray HTML/MathML in titles
    return re.sub(r"[^a-z0-9]+", "", t)


def surname(name: str | None) -> str:
    if not name:
        return ""
    return norm_title(name.split()[-1])


def is_arxiv(w: dict) -> bool:
    for loc in w.get("locations") or []:
        src = (loc or {}).get("source") or {}
        if "arxiv" in (src.get("display_name") or "").lower() or "arxiv.org" in (loc.get("landing_page_url") or ""):
            return True
    return False


def load_raw() -> dict[str, dict]:
    works: dict[str, dict] = {}
    for qdir in sorted(p for p in C.RAW.iterdir() if p.is_dir()):
        for page in sorted(qdir.glob("page_*.json.gz")):
            with gzip.open(page, "rt") as fh:
                for w in json.load(fh)["results"]:
                    rec = works.setdefault(w["id"], w)
                    rec.setdefault("_queries", set()).add(qdir.name)
    return works


def dedupe(df: pl.DataFrame) -> pl.DataFrame:
    """Assign cluster_id; canonical = in-scope first, then non-preprint, then most cited."""
    if "in_scope" not in df.columns:
        df = df.with_columns(pl.lit(True).alias("in_scope"))
    rows = df.select("id", "title_key", "first_author_key", "publication_year", "type").to_dicts()
    rows.sort(key=lambda r: (r["title_key"], r["first_author_key"], r["publication_year"] or 0))
    cluster, prev, cid = {}, None, -1
    for r in rows:
        key = (r["title_key"], r["first_author_key"])
        same = (prev is not None and len(key[0]) >= 20 and key == prev[0]
                and (r["publication_year"] or 0) - (prev[1] or 0) <= 3)
        if not same:
            cid += 1
            prev = (key, r["publication_year"])
        cluster[r["id"]] = cid
    df = df.with_columns(pl.col("id").replace_strict(cluster).alias("cluster_id"))
    df = df.with_columns(
        pl.len().over("cluster_id").alias("cluster_size"),
        pl.col("publication_year").min().over("cluster_id").alias("first_year"),
    )
    rank = (df.with_columns(((pl.col("type") == "preprint").cast(pl.Int8)).alias("_pre"),
                            (~pl.col("in_scope")).cast(pl.Int8).alias("_out"))
              .sort(["cluster_id", "_out", "_pre", "cited_by_count"], descending=[False, False, False, True])
              .group_by("cluster_id", maintain_order=True).first().select("id"))
    canon = set(rank["id"].to_list())
    return df.with_columns(pl.col("id").is_in(canon).alias("is_canonical"))


def main() -> None:
    raw = load_raw()
    this_year = date.today().year
    works, auths, refs = [], [], []
    for w in raw.values():
        pt = w.get("primary_topic") or {}
        src = ((w.get("primary_location") or {}).get("source") or {})
        first = (w.get("authorships") or [{}])[0].get("author", {}) if w.get("authorships") else {}
        works.append({
            "id": w["id"], "doi": w.get("doi"), "title": w.get("title"),
            "publication_year": w.get("publication_year"),
            "publication_date": w.get("publication_date"), "type": w.get("type"),
            "language": w.get("language"), "is_retracted": bool(w.get("is_retracted")),
            "source": src.get("display_name"), "source_type": src.get("type"),
            "is_arxiv": is_arxiv(w), "cited_by_count": w.get("cited_by_count") or 0,
            "field": ((pt.get("field") or {}).get("display_name")),
            "subfield": ((pt.get("subfield") or {}).get("display_name")),
            "topic": pt.get("display_name"),
            "keywords": [k.get("display_name") for k in (w.get("keywords") or [])],
            "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
            "n_authors": len(w.get("authorships") or []),
            "n_references": len(w.get("referenced_works") or []),
            "queries": sorted(w["_queries"]),
            "title_key": norm_title(w.get("title")),
            "first_author_key": surname(first.get("display_name")),
        })
        for pos, a in enumerate(w.get("authorships") or []):
            au = a.get("author") or {}
            insts = a.get("institutions") or [{}]
            for inst in insts:
                auths.append({
                    "work_id": w["id"], "author_id": au.get("id"), "author": au.get("display_name"),
                    "position": pos, "author_position": a.get("author_position"),
                    "institution_id": inst.get("id"), "institution": inst.get("display_name"),
                    "country": inst.get("country_code"), "institution_type": inst.get("type"),
                })
        for r in w.get("referenced_works") or []:
            refs.append({"work_id": w["id"], "ref_id": r})

    df = pl.DataFrame(works, infer_schema_length=None)
    df = df.with_columns(
        (pl.col("field").is_in(list(C.IN_SCOPE_FIELDS))
         & pl.col("type").is_in(list(C.KEEP_TYPES))
         & ~pl.col("is_retracted")
         & pl.col("publication_year").is_between(C.MIN_YEAR, this_year)).alias("in_scope")
    )
    df = dedupe(df)
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    df.write_parquet(C.PROCESSED / "works.parquet")
    pl.DataFrame(auths, infer_schema_length=None).write_parquet(C.PROCESSED / "authorships.parquet")
    pl.DataFrame(refs).write_parquet(C.PROCESSED / "references.parquet")

    final = df.filter(pl.col("in_scope") & pl.col("is_canonical"))
    print(f"raw works: {df.height} | in scope: {df['in_scope'].sum()} | "
          f"canonical in scope: {final.height} | merged duplicates: {df['in_scope'].sum() - final.height}")
    print(f"authorship rows: {len(auths)} | reference edges: {len(refs)}")


if __name__ == "__main__":
    main()
