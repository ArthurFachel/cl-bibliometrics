import unittest

import polars as pl

from clbib.clean import dedupe, norm_title, reconstruct_abstract, surname


class TestClean(unittest.TestCase):
    def test_reconstruct_abstract_orders_by_position(self):
        inv = {"forgetting": [2], "catastrophic": [1], "We": [0], "study": [3]}
        self.assertEqual(reconstruct_abstract(inv), "We catastrophic forgetting study")
        self.assertIsNone(reconstruct_abstract(None))

    def test_norm_title_ignores_case_accents_markup(self):
        self.assertEqual(norm_title("Continual <i>Learning</i>: Résumé!"),
                         norm_title("continual learning resume"))
        self.assertEqual(surname("Gido M. van de Ven"), "ven")

    def test_dedupe_prefers_published_and_keeps_first_year(self):
        title = "overcomingcatastrophicforgettinginneuralnetworks"
        df = pl.DataFrame({
            "id": ["pre", "pub", "other"],
            "title_key": [title, title, title],
            "first_author_key": ["kirkpatrick", "kirkpatrick", "someoneelse"],
            "publication_year": [2016, 2017, 2017],
            "type": ["preprint", "article", "article"],
            "cited_by_count": [100, 5000, 3],
        })
        out = dedupe(df).sort("id")
        rows = {r["id"]: r for r in out.to_dicts()}
        self.assertEqual(rows["pre"]["cluster_id"], rows["pub"]["cluster_id"])
        self.assertNotEqual(rows["pub"]["cluster_id"], rows["other"]["cluster_id"])
        self.assertTrue(rows["pub"]["is_canonical"])
        self.assertFalse(rows["pre"]["is_canonical"])
        self.assertEqual(rows["pub"]["first_year"], 2016)

    def test_canonical_prefers_in_scope_record(self):
        t = "rehearsalbasedcontinuallearningforrobustovariancancer"
        df = pl.DataFrame({
            "id": ["other", "art"], "title_key": [t, t], "first_author_key": ["x", "x"],
            "publication_year": [2026, 2026], "type": ["other", "article"],
            "cited_by_count": [9, 0], "in_scope": [False, True],
        })
        rows = {r["id"]: r for r in dedupe(df).to_dicts()}
        self.assertTrue(rows["art"]["is_canonical"])
        self.assertFalse(rows["other"]["is_canonical"])

    def test_dedupe_does_not_merge_short_generic_titles(self):
        df = pl.DataFrame({
            "id": ["a", "b"], "title_key": ["introduction", "introduction"],
            "first_author_key": ["x", "x"], "publication_year": [2020, 2021],
            "type": ["article", "article"], "cited_by_count": [1, 1],
        })
        self.assertEqual(dedupe(df)["is_canonical"].sum(), 2)


if __name__ == "__main__":
    unittest.main()
