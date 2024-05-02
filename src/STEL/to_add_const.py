"""
    Paths to models and data that is not part of the github release
     -- needs to be downloaded and added separately (after possibly acquiring the necessary permissions)
"""
import os

from STEL.utility.file_utility import get_dir_to_src

# ------------- necessary to run STEL --------------------
# GYAFC permission necessary, see https://github.com/raosudha89/GYAFC-corpus and
#   https://webscope.sandbox.yahoo.com/catalog.php?datatype=l)
src_dir = get_dir_to_src()
#   filtered STEL data
#       Dimensions
LOCAL_STEL_DIM_QUAD = [src_dir + '/../Data/STEL/dimensions/_quad_stel-dimensions_formal-815_complex-815.tsv']
#       Characteristics
LOCAL_STEL_CHAR_QUAD = [get_dir_to_src() + '/../Data/STEL/characteristics/quad_questions_char_substitution.tsv',
                        get_dir_to_src() + '/../Data/STEL/characteristics/quad_questions_char_contraction.tsv'
                        ]
#   UNTIL YOU HAVE PERMISSION TRY ON SMALLER DEMO ...
# LOCAL_STEL_DIM_QUAD = [cur_dir + '/../Data/STEL/dimensions/quad_stel-dimension_simple-100_sample.tsv',
#                        cur_dir + '/../Data/STEL/dimensions/quad_stel-dimension_formal-100_sample.tsv']
#   PROBABLY NOT WHAT YOU NEED: full STEL data with annotations (on the unfiltered potential task instances)
LOCAL_ANN_STEL_DIM_QUAD = [src_dir + '/../Data/Experiment-Results/annotations/_QUAD-full_annotations.tsv']

# ------------- necessary to run STEL on LIWC and deepstyle --------------------
# proprietary
LIWC_PATH = src_dir + '/../Data/Models/_LIWC2015_Dictionary.dic'
# add from external (freely accessible sources)
DEEPSTYLE_MODEL_PATH = src_dir + "../../DeepStyle/model/212.129.44.40/DeepStyle/dbert-ft/"

# ------------- STEL crowd-sourced annotations -----------------
# GYAFC permission necessary
LOCAL_TOTAL_DIM_QUAD = src_dir + '/../Data/Experiment-Results/annotations/_QUAD-full_annotations.tsv'

# ------------- necessary for generation of STEL ----------------
# GYAFC permission necessary
GYAFC_PATH = src_dir + "/../Data/Datasets/GYAFC_Corpus/Entertainment_Music/test/"
TUNE_GYAFC_PATH = src_dir + "/../Data/Datasets/GYAFC_Corpus/Entertainment_Music/tune/"

# to add from external (freely accessible) sources
PUSHSHIFT_MONTH = src_dir + "/../Data/Datasets/RC_2017-06.bz2"
PUSHSHIFT_MONTH_2 = src_dir + "/../Data/Datasets/RC_2012-06.bz2"
PUSHSHIFT_MONTH_3 = src_dir + "/../Data/Datasets/RC_2007-06.bz2"
PUSHSHIFT_MONTH_4 = src_dir + "/../Data/Datasets/RC_2007-05.bz2"
PUSHSHIFT_MONTH_5 = src_dir + "/../Data/Datasets/RC_2007-07.bz2"
PUSHSHIFT_MONTH_6 = src_dir + "/../Data/Datasets/RC_2016-06.bz2"
PUSHSHIFT_MONTH_7 = src_dir + "/../Data/Datasets/RC_2007-08.bz2"
PUSHSHIFT_MONTH_8 = src_dir + "/../Data/Datasets/RC_2007-09.bz2"
ABSTRACT_WIKI = src_dir + "/../Data/Datasets/enwiki-20181220-abstract.xml"
#   simplification data from: https://github.com/cocoxu/simplification/tree/master/data
SIMPLE_TEST_TRUE_CASE_PATH = src_dir + "/../Data/Datasets/turkcorpus/truecased/test.8turkers.organized.tsv"
SIMPLE_TUNE_TRUE_CASE_PATH = src_dir + "/../Data/Datasets/turkcorpus/truecased/tune.8turkers.organized.tsv"
#   LIWC dictionary (proprietary)
LIWC_STYLE_ACC_PATTERN = src_dir + "/../Data/Models/coord-liwc-patterns.txt"

