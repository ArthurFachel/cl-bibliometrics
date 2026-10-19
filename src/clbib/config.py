"""Corpus definition and paths. Change the query here, nowhere else."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "openalex"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

MAILTO = "arthur.nunes6889@gmail.com"  # OpenAlex polite pool
OPENALEX = "https://api.openalex.org/works"

# Each query is fetched separately so we can report what every term contributes.
# "lifelong learning" alone is dominated by education research (measured
# 2026-10-19: 30.6k of 55.2k works are Social Sciences), so it is restricted to
# the Computer Science field (fields/17) at query time.
QUERIES = {
    "core": {
        "terms": [
            "continual learning",
            "catastrophic forgetting",
            "class-incremental",
            "task-incremental",
            "domain-incremental",
        ],
        "extra_filter": None,
    },
    "lifelong_cs": {
        "terms": ["lifelong learning"],
        "extra_filter": "primary_topic.field.id:fields/17",
    },
}

# Fields kept in the raw dump (top-level only; OpenAlex `select` limitation).
SELECT = [
    "id", "doi", "ids", "title", "publication_year", "publication_date", "type",
    "language", "is_retracted", "primary_location", "locations", "authorships",
    "primary_topic", "topics", "keywords", "referenced_works", "cited_by_count",
    "abstract_inverted_index",
]

# In-scope = primary topic in one of these OpenAlex fields.
IN_SCOPE_FIELDS = {"Computer Science", "Engineering", "Neuroscience", "Mathematics"}
KEEP_TYPES = {"article", "conference-paper", "preprint", "review", "book-chapter"}
MIN_YEAR = 1985
