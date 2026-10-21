import unittest

import polars as pl

from clbib.nlp_validate import assign_strata, draw


class TestRound2Sample(unittest.TestCase):
    def test_assign_strata_priority(self):
        d = pl.DataFrame({"id": ["a", "b", "c", "d"], "nlp_llm": [True, True, False, False],
                          "peft_llm": [True, False, True, None]})
        self.assertEqual(assign_strata(d)["stratum"].to_list(), ["nlp", "nlp", "peft", "other"])

    def test_draw_is_balanced_and_seeded(self):
        d = pl.DataFrame({"id": [str(i) for i in range(30)], "stratum": ["nlp"] * 10 + ["peft"] * 10 + ["other"] * 10})
        a, b = draw(d, 4, 1), draw(d, 4, 1)
        self.assertEqual(a["id"].to_list(), b["id"].to_list())
        self.assertEqual(sorted(a.group_by("stratum").len().rows()), [("nlp", 4), ("other", 4), ("peft", 4)])


if __name__ == "__main__":
    unittest.main()
