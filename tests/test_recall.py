import unittest

import polars as pl

from clbib.recall import freeze_rows, match, read_check


class TestRecall(unittest.TestCase):
    def test_freeze_rows_keeps_reported_up_to_last_run(self):
        recs = [
            {"arxiv_id": "2610.00002", "status": "reportado", "titulo": "B", "publicado": "2026-10-01T00:00:00Z",
             "aparicoes": [{"run": "2026-10-14T08:00"}, {"run": "2026-10-20T08:00"}]},
            {"arxiv_id": "2610.00001", "status": "reportado", "titulo": "A", "publicado": "2026-10-01",
             "aparicoes": [{"run": "2026-10-06T08:00"}]},
            {"arxiv_id": "2610.00003", "status": "descartado", "titulo": "C", "publicado": "2026-10-01",
             "aparicoes": [{"run": "2026-10-06T08:00"}]},
            {"arxiv_id": "2610.00004", "status": "reportado", "titulo": "D", "publicado": "2026-10-01",
             "aparicoes": [{"run": "2026-10-19T08:00"}]},
        ]
        rows = freeze_rows(recs)
        self.assertEqual([r["arxiv_id"] for r in rows], ["2610.00001", "2610.00002"])
        self.assertEqual(rows[1]["first_seen"], "2026-10-14")
        self.assertEqual(rows[1]["published"], "2026-10-01")

    def test_match_by_doi_then_title_with_canonical_flags(self):
        works = pl.DataFrame({
            "id": ["W1", "W2", "W3"],
            "doi": ["https://doi.org/10.48550/arXiv.2610.00001", None, "https://doi.org/10.1/x"],
            "title_key": ["alpha", "betapaper", "somethingelse"],
            "cluster_id": [1, 2, 2],
            "is_canonical": [True, False, True],
            "in_scope": [True, True, True],
            "repository_only": [False, False, False],
            "is_focused": [True, False, True],
        })
        check = pl.DataFrame({"arxiv_id": ["2610.00001", "2610.00002", "2610.00009"],
                              "title": ["Alpha", "Beta: paper", "Missing one"]})
        m = match(check, works).sort("arxiv_id")
        self.assertEqual(m["matched_via"].to_list(), ["doi", "title", None])
        self.assertEqual(m["found"].to_list(), [True, True, False])
        # W2 is a non-canonical record; its flags come from W3, the canonical record of cluster 2
        self.assertEqual(m["focused"].to_list(), [True, True, False])
        self.assertEqual(m["in_scope"].to_list(), [True, True, False])

    def test_read_check_keeps_trailing_zeros(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.csv"
            p.write_text("arxiv_id,title,published,first_seen\n2610.01590,T,2026-10-01,2026-10-06\n")
            self.assertEqual(read_check(p)["arxiv_id"].to_list(), ["2610.01590"])


if __name__ == "__main__":
    unittest.main()
