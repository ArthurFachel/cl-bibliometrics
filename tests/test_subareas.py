import unittest

import polars as pl

from clbib.subareas import tag


def tags(title, abstract=""):
    row = tag(pl.DataFrame({"title": [title], "abstract": [abstract]})).row(0, named=True)
    return {k for k, v in row.items() if v is True}


class TestSubareas(unittest.TestCase):
    def test_method_families(self):
        self.assertIn("replay", tags("Generative replay for class-incremental learning"))
        self.assertIn("regularization", tags("Overcoming forgetting", "We use elastic weight consolidation (EWC)."))
        self.assertIn("peft_llm", tags("Continual instruction tuning of LLMs with LoRA"))
        self.assertIn("plasticity", tags("Loss of plasticity in deep continual learning"))

    def test_multilabel_and_domain(self):
        t = tags("Rehearsal-based continual learning for ECG classification")
        self.assertTrue({"replay", "application"} <= t)

    def test_word_boundaries(self):
        # "lora" must not fire inside other words; generic "benchmark datasets" is not a benchmark paper
        self.assertNotIn("peft_llm", tags("Flora classification with incremental learning"))
        self.assertNotIn("federated", tags("A replay method"))
        self.assertNotIn("survey_benchmark", tags("A replay method", "Experiments on three benchmark datasets."))
        self.assertIn("survey_benchmark", tags("CLEAR", "We introduce a new benchmark for continual learning."))


if __name__ == "__main__":
    unittest.main()
