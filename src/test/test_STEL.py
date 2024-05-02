from unittest import TestCase
import STEL
from STEL.STEL import eval_on_STEL
from STEL.legacy_sim_classes import WordLengthSimilarity
from STEL.to_add_const import LOCAL_ANN_STEL_DIM_QUAD


class Test(TestCase):
    def test_eval_sim(self):
        eval_on_STEL(style_objects=[WordLengthSimilarity()])

    def test_eval_sbert(self):
        from STEL.similarity import SBERTSimilarity
        eval_on_STEL(style_objects=[SBERTSimilarity("AnnaWegmann/Style-Embedding")])

    def test_eval_full(self):
        STEL.STEL.eval_model(style_objects=[WordLengthSimilarity()], stel_dim_tsv=LOCAL_ANN_STEL_DIM_QUAD,
                             only_STEL=False)

