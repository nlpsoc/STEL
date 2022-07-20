from unittest import TestCase
from style_similarity import LIWCSimilarity
from to_add_const import LIWC_PATH


class TestLIWCSimilarity(TestCase):
    def test_similarity(self):
        liwc_path = "../" + LIWC_PATH
        liwc_sim = LIWCSimilarity(liwc_path)
        s1 = ":)"
        s2 = ":("
        self.assertEqual(liwc_sim.similarity(s1, s2), 0)
