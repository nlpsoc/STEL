"""

    STYLE similarity is at 1 if the same style (also for cosine-sim)
        at 0 or -1 if distinct style

"""
from abc import ABC
import nltk
from typing import List

import numpy
import torch

from sentence_transformers import SentenceTransformer

# from sentence_transformers import SentenceTransformer, models

# see SBERT models here:
#   - https://www.sbert.net/docs/pretrained_models.html
#   - https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0

# Goal: style similarity between two utterances, optional: dependant on the corpus

from STEL.utility import set_for_global
from STEL.utility.set_for_global import set_torch_device

set_for_global.set_logging()
set_for_global.set_global_seed()


# NON_VALID_FUNCTIONS_SIM_CLASS = ["static_deepstyle_sim", "__init__", "_apply_on_list"]


# ABSTRACT base style similarity class
class Similarity(ABC):
    """
        Abstract Base similarity class
        -- similarity or similarities need to be implemented/overridden
    """

    def __init__(self):
        self.SAME = 1
        self.DISTINCT = 0

    def similarity(self, sentence_1: str, sentence_2: str) -> float:
        """
        similarity functions between two strings: sentence_1 and sentence_2
        returns a value between -1/0 and 1, where
            1 means same
            -1/0 means most distinct
        ==> bigger similarity value means higher similarity

        :param sentence_1:
        :param sentence_2:
        :return:
        """
        if sentence_1 == sentence_2:
            return self.SAME
        else:
            return self.DISTINCT

    def similarities(self, sentences_1: List[str], sentences_2: List[str]) -> List[float]:
        """
            this function is called when evaluating your model on STEL, so feel free use more efficient methods
        Args:
            sentences_1:
            sentences_2:

        Returns:

        """
        return [self.similarity(sentences_1[i], sentences_2[i]) for i in range(len(sentences_1))]


class LevenshteinSimilarity(Similarity):
    def similarity(self, sentence_1: str, sentence_2: str) -> float:
        if len(sentence_1) == 0 and len(sentence_2) == 0:
            return self.SAME
        else:
            # minimum number of (character-level) edits needed to transform one string into the other
            sim = nltk.edit_distance(sentence_1, sentence_2)
            return 1 - sim / max(len(sentence_1), len(sentence_2))  # 0 edits -> SAME=1


class SBERTSimilarity(Similarity):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.model.to(set_torch_device())

    def similarity(self, sentence_1: str, sentence_2: str) -> float:
        with torch.no_grad():
            sentence_emb_1 = self.model.encode(sentence_1, show_progress_bar=False)
            sentence_emb_2 = self.model.encode(sentence_2, show_progress_bar=False)
        return cosine_sim(sentence_emb_1, sentence_emb_2)

    def similarities(self, sentences_1: List[str], sentences_2: List[str]) -> List[float]:
        with torch.no_grad():
            sentence_emb_1 = self.model.encode(sentences_1, show_progress_bar=False)
            sentence_emb_2 = self.model.encode(sentences_2, show_progress_bar=False)
        return [cosine_sim(sentence_emb_1[i], sentence_emb_2[i]) for i in range(len(sentences_1))]


# --------------------------------- UTILITY --------------------------------------------------------------------


def cosine_sim(style_dim_u1, style_dim_u2):
    from scipy import spatial
    if hasattr(style_dim_u1, 'cpu'):
        style_dim_u1 = style_dim_u1.cpu()
        style_dim_u2 = style_dim_u2.cpu()
    if numpy.linalg.norm(style_dim_u1) * numpy.linalg.norm(style_dim_u2) != 0.0:
        return 1 - spatial.distance.cosine(style_dim_u1, style_dim_u2)
    elif numpy.linalg.norm(style_dim_u1) > 0 or numpy.linalg.norm(style_dim_u2) > 0:
        return 0
    else:
        return 1
    # if numpy.linalg.norm(style_dim_u1) * numpy.linalg.norm(style_dim_u2) != 0.0:
    #     cos_sim = (style_dim_u1 @ style_dim_u2.T) / (
    #             numpy.linalg.norm(style_dim_u1) * numpy.linalg.norm(style_dim_u2))
    # else:
    #     cos_sim = 1
    # return cos_sim
