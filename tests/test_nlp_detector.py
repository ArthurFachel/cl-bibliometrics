import unittest

import polars as pl

from clbib.benchmarks import ALL_NAMES, FAMILY, NAMED_LANGUAGE_DATASET, mention_flags, named_language_dataset


def nlp_hits(text):
    df = pl.DataFrame({"id": ["x"], "title": [""], "abstract": [text]})
    row = mention_flags(df, ALL_NAMES).row(0, named=True)
    return {n for n in ALL_NAMES if row[n] and FAMILY[n] == "NLP & LLM"}


class TestLayer1(unittest.TestCase):
    def test_named_language_datasets(self):
        self.assertIn("Event / relation / NER", nlp_hits("Experiments on MAVEN and ACE 2005 show gains."))
        self.assertIn("Event / relation / NER", nlp_hits("on the FewRel and TACRED datasets"))
        self.assertIn("QA / reading", nlp_hits("a stream of SQuAD v2 and TriviaQA"))
        self.assertIn("LLM suites", nlp_hits("evaluated on HellaSwag, PIQA and ARC-Challenge"))
        self.assertIn("Captioning / multimodal", nlp_hits("MS-COCO captions and Flickr30k"))
        self.assertIn("Dialogue / generation / LM", nlp_hits("IWSLT'14 and WMT'14 translation"))
        self.assertIn("Text classification (GLUE tasks, intents)", nlp_hits("an RTE->MRPC->CoLA->SST-2 sequence"))

    def test_common_words_do_not_fire(self):
        for text in ["an ace up the sleeve", "the maven of design", "a flan recipe",
                     "the arc of training", "we use SGD with momentum", "imdb-style ratings"]:
            self.assertEqual(nlp_hits(text), set(), text)


class TestLayer2(unittest.TestCase):
    def test_fires_on_named_language_dataset(self):
        self.assertTrue(named_language_dataset("Continual event detection with LLMs on the KoBALT benchmark."))
        self.assertTrue(named_language_dataset("Dialogue systems are tested on datasets such as InstrDialog and SGD."))
        self.assertTrue(named_language_dataset("We prompt LLMs on the KoBALT benchmark."))  # stoplisted LLMs must not hide KoBALT

    def test_does_not_fire(self):
        self.assertFalse(named_language_dataset("Dialogue systems on several benchmark datasets."))  # nothing named
        self.assertFalse(named_language_dataset("Image classification on the CORe50 dataset."))  # no language context
        self.assertFalse(named_language_dataset("Language models are tested on CL benchmarks."))  # stoplisted acronym
        self.assertFalse(named_language_dataset("Prompting LLMs on the Split CIFAR-100 benchmark."))  # vision dataset
        self.assertFalse(named_language_dataset(None))

    def test_is_an_nlp_name(self):
        self.assertIn(NAMED_LANGUAGE_DATASET, nlp_hits("Continual event detection with LLMs on the KoBALT benchmark."))


if __name__ == "__main__":
    unittest.main()
