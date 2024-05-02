from unittest import TestCase
from STEL.legacy_sim_classes import LIWCSimilarity
from STEL.to_add_const import LIWC_PATH


class TestLIWCSimilarity(TestCase):
    def test_similarity(self):
        # fails although part of LIWC dict
        #   due to tokenization issue with the suggested tokenizer in the library, see https://pypi.org/project/liwc/
        liwc_path = LIWC_PATH
        liwc_sim = LIWCSimilarity(liwc_path)
        s1 = ":)"
        s2 = ":("
        self.assertEqual(liwc_sim.similarity(s1, s2), 0)


