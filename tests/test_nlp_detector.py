import unittest

import polars as pl

from clbib.benchmarks import ALL_NAMES, FAMILY, mention_flags


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


if __name__ == "__main__":
    unittest.main()
