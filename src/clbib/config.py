"""Corpus definition and paths. Change the query here, nowhere else."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "openalex"
PROCESSED = ROOT / "data" / "processed"

# Works that exist ONLY on a general-purpose deposit repository are left out of
# the analysis corpus. These repositories accept any upload without moderation;
# arXiv, bioRxiv/medRxiv and institutional repositories (which mirror published
# work) are kept. A work also found on a journal, conference, arXiv or any other
# source, in any OpenAlex location or any record of its dedup cluster, is kept.
# Rule added on 2026-10-20 AFTER seeing the 2026 anomaly (Zenodo: 3 focused works
# in 2024, 319 in 2026, 45% of them from 10 authors), so it is not pre-registered:
# every headline number is re-run without it (make sensitivity).
GENERIC_REPOSITORIES = (r"(?i)zenodo|ssrn|research square|preprints\.org|osf preprints|figshare|"
                        r"authorea|qeios|techrxiv")
INCLUDE_REPOSITORY_ONLY = os.environ.get("CLBIB_INCLUDE_REPOSITORY_ONLY") == "1"

_OUT = ROOT / "results" / "sensitivity_with_repository_only" if INCLUDE_REPOSITORY_ONLY else None
RESULTS = _OUT or ROOT / "results"
FIGURES = (_OUT / "figures") if _OUT else ROOT / "figures"

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

# Works found ONLY by the "lifelong learning" query need extra evidence of ML.
# Measured 2026-10-19 on an early partial fetch: of 6,201 lifelong-only CS works,
# the top topics were AI in Education, digital literacy, e-learning and MOOCs
# (subfield Information Systems = 4,189). Gate: an ML term AND (AI/CV subfield OR
# no education term). On the full snapshot: 7,002 lifelong-only works, 603 pass.
# Known misses e.g. "Lifelong Learning by Adjusting Priors".
LIFELONG_ML_TERMS = (r"(?i)forget|neural|deep learning|reinforcement|\bagent|robot|replay|plasticity|"
                     r"incremental|knowledge transfer|meta-learning|sequence of tasks|sequential tasks|"
                     r"new tasks|previous tasks|non-stationary|lifelong (machine|reinforcement|robot)")
LIFELONG_EDU_TERMS = (r"(?i)educat|e-learning|elearning|student|teacher|universit|school|mooc|curricul|"
                      r"librar|literacy|pedagog|course|learners")
LIFELONG_AI_SUBFIELDS = {"Artificial Intelligence", "Computer Vision and Pattern Recognition"}

# "Focused" corpus: CL must be the topic, not a passing mention. Rule = a CL term
# in the title OR >= FOCUS_MIN_ABSTRACT_HITS mentions in the abstract.
# Tuned on the dev sample (LLM labels, 2026-10-19): corpus precision 0.77 -> 0.95
# at recall 0.87. Frozen in 098968f and re-scored once on the disjoint test
# sample: precision 0.94, recall 0.93.
FOCUS_TERMS = (r"(?i)continual(ly)? learn|catastrophic(ally)? forget|class-incremental|task-incremental|"
               r"domain-incremental|lifelong (machine |reinforcement )?learn|incremental learning|"
               r"continual (pre-?training|fine-?tuning|adaptation|instruction)|forgetting")
FOCUS_MIN_ABSTRACT_HITS = 2
