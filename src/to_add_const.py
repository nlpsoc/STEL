"""
    Paths to models and data that is not part of the github release
     -- needs to be downloaded and added separately (after possibly acquiring the necessary permissions)
"""
import os

# ------------- necessary to run STEL --------------------
# GYAFC permission necessary, see https://github.com/raosudha89/GYAFC-corpus and
#   https://webscope.sandbox.yahoo.com/catalog.php?datatype=l)
cur_dir = os.path.dirname(os.path.realpath(__file__))
#   filtered STEL data
LOCAL_STEL_DIM_QUAD = [cur_dir + '/../Data/STEL/dimensions/_quad_stel-dimensions_formal-815_complex-815.tsv']
#   UNTIL PERMISSION IS received using sample STEL data ...
# LOCAL_STEL_DIM_QUAD = [cur_dir + '/../Data/STEL/dimensions/quad_stel-dimension_simple-100_sample.tsv',
#                        cur_dir + '/../Data/STEL/dimensions/quad_stel-dimension_formal-100_sample.tsv']
#   full STEL data with annotations
LOCAL_ANN_STEL_DIM_QUAD = [cur_dir + '/../Data/Experiment-Results/annotations/_QUAD-full_annotations.tsv']

# ------------- necessary to run STEL on LIWC and deepstyle --------------------
# proprietary
LIWC_PATH = '../Data/Models/_LIWC2015_Dictionary.dic'
# add from external (freely accessible sources)
DEEPSTYLE_MODEL_PATH = "../../DeepStyle/model/212.129.44.40/DeepStyle/dbert-ft/"

# ------------- STEL crowd-sourced annotations -----------------
# GYAFC permission necessary
LOCAL_TOTAL_DIM_QUAD = '../Data/Experiment-Results/annotations/_QUAD-full_annotations.tsv'

# ------------- necessary for generation of STEL ----------------
# GYAFC permission necessary
GYAFC_PATH = "../Data/Datasets/GYAFC_Corpus/Entertainment_Music/test/"
TUNE_GYAFC_PATH = "../Data/Datasets/GYAFC_Corpus/Entertainment_Music/tune/"

# to add from external (freely accessible) sources
PUSHSHIFT_MONTH = "../Data/Datasets/RC_2017-06.bz2"
PUSHSHIFT_MONTH_2 = "../Data/Datasets/RC_2012-06.bz2"
PUSHSHIFT_MONTH_3 = "../Data/Datasets/RC_2007-06.bz2"
PUSHSHIFT_MONTH_4 = "../Data/Datasets/RC_2007-05.bz2"
PUSHSHIFT_MONTH_5 = "../Data/Datasets/RC_2007-07.bz2"
PUSHSHIFT_MONTH_6 = "../Data/Datasets/RC_2016-06.bz2"
PUSHSHIFT_MONTH_7 = "../Data/Datasets/RC_2007-08.bz2"
PUSHSHIFT_MONTH_8 = "../Data/Datasets/RC_2007-09.bz2"
ABSTRACT_WIKI = "../Data/Datasets/enwiki-20181220-abstract.xml"
#   simplification data from: https://github.com/cocoxu/simplification/tree/master/data
SIMPLE_TEST_TRUE_CASE_PATH = "../Data/Datasets/turkcorpus/truecased/test.8turkers.organized.tsv"
SIMPLE_TUNE_TRUE_CASE_PATH = "../Data/Datasets/turkcorpus/truecased/tune.8turkers.organized.tsv"
#   LIWC dictionary (proprietary)
LIWC_STYLE_ACC_PATTERN = "../Data/Models/coord-liwc-patterns.txt"
