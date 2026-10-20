"""The analysis corpus, defined once.

analysis(): in scope, one record per dedup cluster and, unless
CLBIB_INCLUDE_REPOSITORY_ONLY=1, not repository-only (see config.py).
focused(): analysis() and continual learning is the topic (focus rule).
"""
import polars as pl

from . import config as C


def analysis() -> pl.Expr:
    e = pl.col("in_scope") & pl.col("is_canonical")
    if not C.INCLUDE_REPOSITORY_ONLY:
        e = e & ~pl.col("repository_only")
    return e


def focused() -> pl.Expr:
    return analysis() & pl.col("is_focused")
