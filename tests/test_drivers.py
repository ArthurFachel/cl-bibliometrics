import unittest

import polars as pl

from clbib.drivers import citation_edges, member, share


class TestDrivers(unittest.TestCase):
    def test_edges_merge_dedup_clusters(self):
        # W2 is the arXiv twin of canonical W1; citing either must count once for W1.
        raw = pl.DataFrame({"id": ["W1", "W2", "W3", "W4"], "cluster_id": [1, 1, 3, 4],
                            "is_canonical": [True, False, True, True]})
        refs = pl.DataFrame({"work_id": ["W3", "W3", "W4", "W1"],
                             "ref_id": ["W1", "W2", "W2", "W2"]})   # W1 -> W2 is a self-citation
        works = pl.DataFrame({"id": ["W1", "W3", "W4"], "first_year": [2022, 2024, 2025],
                              "peft_llm": [True, False, True]})
        e = citation_edges(raw, refs, works).sort("citing")
        self.assertEqual(e.select("citing", "cited").rows(), [("W3", "W1"), ("W4", "W1")])
        self.assertEqual(e["p_cited"].to_list(), [True, True])

    def test_share_whole_counting(self):
        d = pl.DataFrame({"countries": [["CN", "US"], ["CN"], ["US"]], "has_company": [False, True, False],
                          "cites": [2, 6, 2]})
        self.assertAlmostEqual(share(d, "CN", "cites"), 0.8)
        self.assertAlmostEqual(share(d, "US", "cites"), 0.4)    # CN-US work counts for both
        self.assertAlmostEqual(share(d, "company", "cites"), 0.6)
        self.assertEqual(d.select(member("company").sum()).item(), 1)


if __name__ == "__main__":
    unittest.main()
