import unittest

from clbib.benchmark_validate import weighted_pr
from clbib.country_validate import affiliations
from clbib.llm_label import parse


class TestValidationHelpers(unittest.TestCase):
    def test_weighted_pr_reweights_strata(self):
        # one detector-positive stratum (weight 1) and one negative stratum (weight 10):
        # a single miss in the negative stratum counts like ten misses.
        y = [1, 1, 0, 1]
        p = [1, 1, 1, 0]
        w = [1, 1, 1, 10]
        prec, rec = weighted_pr(y, p, w)
        self.assertAlmostEqual(prec, 2 / 3)
        self.assertAlmostEqual(rec, 2 / 12)

    def test_affiliations_dedup_and_empty(self):
        msg = {"author": [{"affiliation": [{"name": "Hunan University, Changsha, China"}]},
                          {"affiliation": [{"name": "Hunan University, Changsha, China"}, {"name": " "}]},
                          {}]}
        self.assertEqual(affiliations(msg), ["Hunan University, Changsha, China"])
        self.assertEqual(affiliations(None), [])

    def test_parse_custom_labels(self):
        d = parse('noise {"cn": 1, "us": 0, "notes": "x"} trailing', ["cn", "us"])
        self.assertEqual(d, {"cn": 1, "us": 0, "notes": "x"})
        with self.assertRaises(ValueError):
            parse('{"cn": 1}', ["cn", "us"])


if __name__ == "__main__":
    unittest.main()
