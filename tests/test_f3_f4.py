import unittest

import polars as pl

from clbib.benchmarks import ALL_NAMES, mention_flags
from clbib.networks import institution_graph, per_work


def found(text):
    df = pl.DataFrame({"id": ["x"], "title": [""], "abstract": [text]})
    row = mention_flags(df, ALL_NAMES).row(0, named=True)
    return {n for n in ALL_NAMES if row[n]}


class TestBenchmarks(unittest.TestCase):
    def test_imagenet_variants_do_not_count_as_imagenet(self):
        self.assertEqual(found("Results on ImageNet-R and CUB-200."), {"ImageNet-R/A", "CUB-200"})
        self.assertEqual(found("Split mini-ImageNet"), {"Tiny/mini-ImageNet"})
        self.assertIn("ImageNet (full/subset)", found("We use ImageNet-100 and CIFAR-100."))

    def test_case_sensitive_acronyms(self):
        # method names and plain English must not fire
        self.assertEqual(found("TRACE: Discovering task-specific parameters"), set())
        self.assertEqual(found("a clear benchmark for future work"), set())
        self.assertEqual(found("a coin flip"), set())
        self.assertIn("SuperNI / TRACE", found("Experiments on the TRACE benchmark show"))
        self.assertIn("GLUE", found("sequential GLUE tasks"))

    def test_mnist_family(self):
        self.assertTrue({"Permuted MNIST", "MNIST (any)"} <= found("permuted MNIST"))
        self.assertIn("Split/Rotated MNIST", found("Split-MNIST"))


class TestNetworks(unittest.TestCase):
    def test_institution_graph_weights_and_cap(self):
        auth = pl.DataFrame({
            "work_id": ["w1", "w1", "w2", "w2", "w3"],
            "author_id": ["a", "b", "a", "c", "d"],
            "country": ["BR", "FR", "BR", "FR", None],
            "institution_id": ["I1", "I2", "I1", "I2", None],
            "institution_type": ["education", "company", "education", "education", None],
        })
        pw = per_work(auth)
        g = institution_graph(pw)
        self.assertEqual(g["I1"]["I2"]["weight"], 2)
        self.assertEqual(g.nodes["I1"]["works"], 2)
        w1 = pw.filter(pl.col("work_id") == "w1").row(0, named=True)
        self.assertTrue(w1["has_company"])
        self.assertFalse(pw.filter(pl.col("work_id") == "w3")["has_country"][0])


if __name__ == "__main__":
    unittest.main()
