"""
    Example calls to eval_style_models (probably what you want to call)
"""
import set_for_global
set_for_global.set_logging()

import eval_style_models
from to_add_const import LOCAL_ANN_STEL_DIM_QUAD


def eval_stel():
    # call all models (except for deepstyle and LIWC) on STEL
    eval_style_models.eval_sim()


def eval_deepstyle_stel():
    # deepstyle needs has different python prerequisites
    from style_similarity import DeepstyleSimilarity
    eval_style_models.eval_sim(style_objects=[DeepstyleSimilarity()])


def eval_stel_one_model():
    from style_similarity import WordLengthSimilarity
    eval_style_models.eval_sim(style_objects=[WordLengthSimilarity()])
    # from style_similarity import LevenshteinSimilarity
    # eval_style_models.eval_sim(style_objects=[LevenshteinSimilarity()])


eval_stel_one_model()


# --------- PROBABLY NOT WHAT YOU WANT: ---------------


def eval_unfiltered():
    # call all models (except for deepstyle and LIWC) on the unfiltered potential task instances
    #   (i.e., not using annotations)
    eval_style_models.eval_sim(stel_dim_tsv=LOCAL_ANN_STEL_DIM_QUAD, filter_majority_votes=False)


def eval_deepstyle_unfiltered():
    from style_similarity import DeepstyleSimilarity
    eval_style_models.eval_sim(stel_dim_tsv=LOCAL_ANN_STEL_DIM_QUAD, filter_majority_votes=False,
                               style_objects=[DeepstyleSimilarity()])


def eval_one_model_unfiltered():
    from style_similarity import LevenshteinSimilarity
    eval_style_models.eval_sim(stel_dim_tsv=LOCAL_ANN_STEL_DIM_QUAD, filter_majority_votes=False,
                               style_objects=[LevenshteinSimilarity()])