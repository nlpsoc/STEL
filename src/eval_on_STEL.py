"""
    Evaluate language models and methods on STEL
"""
import argparse
import logging

from STEL.STEL import eval_on_STEL
from STEL.utility.eval_on_tasks import eval_model
from STEL.to_add_const import LOCAL_STEL_DIM_QUAD, LOCAL_TOTAL_DIM_QUAD, LOCAL_STEL_CHAR_QUAD
from STEL.similarity import LevenshteinSimilarity
from STEL.legacy_sim_classes import CharacterThreeGramSimilarity, PunctuationSimilarity, WordLengthSimilarity, \
    UppercaseSimilarity, PosTagSimilarity, UncasedBertSimilarity, CasedBertSimilarity, MpnetSentenceBertSimilarity, \
    ParaMpnetSentenceBertSimilarity, RobertaSimilarity, BERTUncasedNextSentenceSimilarity, \
    BERTCasedNextSentenceSimilarity, USESimilarity
from STEL.utility import set_for_global

STYLE_OBJECTS = [LevenshteinSimilarity(),
                 CharacterThreeGramSimilarity(),
                 PunctuationSimilarity(),
                 WordLengthSimilarity(),
                 UppercaseSimilarity(),
                 PosTagSimilarity(),
                 # LIWCStyleSimilarity(), LIWCSimilarity(), LIWCFunctionSimilarity(),  # LIWC dict necessary
                 USESimilarity(),
                 BERTCasedNextSentenceSimilarity(),
                 BERTUncasedNextSentenceSimilarity(),
                 MpnetSentenceBertSimilarity(),
                 ParaMpnetSentenceBertSimilarity(),
                 CasedBertSimilarity(),
                 RobertaSimilarity(),
                 UncasedBertSimilarity()]


# For running deepstyle make sure you have a working python environment for that model
#   see: https://github.com/hayj/DeepStyle ...
#   then set STYLE_OBJECTS = [DeepstyleSimilarity()]
# from style_similarity import DeepstyleSimilarity
# STYLE_OBJECTS = [DeepstyleSimilarity()]
# For running LIWC make sure you have the LIWC dict file in the path specified in to_add_const
#   STYLE_OBJECTS = [LIWCStyleSimilarity(), LIWCSimilarity(), LIWCFunctionSimilarity()]


if __name__ == "__main__":
    set_for_global.set_logging()

    eval_on_STEL(style_objects=STYLE_OBJECTS)
