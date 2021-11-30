"""
includes global constants and functions
"""
import logging
import os
import random
import sys

import numpy

SEED = 1404


def set_global_seed(w_torch=True, seed=SEED):
    """
    Make calculations reproducible by setting RANDOM seeds
    :param seed:
    :param w_torch:
    :return:
    """
    if 'torch' not in sys.modules:
        w_torch = False
    if w_torch:
        import torch
        logging.info(f"Running in deterministic mode with seed {seed}")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    numpy.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_logging():
    """
    set logging format for calling logging.info
    :return:
    """
    import logging
    # create formatter
    # logging.root.setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    hdlr = root.handlers[0]
    fmt = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    hdlr.setFormatter(fmt)
    # logging.basicConfig(level=logging.INFO)
    # LOG_CONFIG = {'root': { 'handlers':('console', 'file'), 'level': 'DEBUG'}}
    # logging.root.__format__('%(asctime)s : %(levelname)s : %(message)s')
    # logging.root.
    # logging.basicConfig(level=logging.INFO)


# CONSTANTS
ALTERNATIVE12_COL = 'Alternative 1.2'
ALTERNATIVE11_COL = 'Alternative 1.1'
ANCHOR2_COL = 'Anchor 2'
ANCHOR1_COL = 'Anchor 1'
NBR_FOR_CORRECT_COL = '# Votes out of 5 for Correct Alternative'
ID_COL = 'ID'
CORRECT_ALTERNATIVE_COL = 'Correct Alternative'  # values either 1 or 2: 1 meaning A1, S1 is correct
ALTERNATIVE2_COL = 'Alternative 2'
ALTERNATIVE1_COL = 'Alternative 1'
ANCHOR_COL = 'Anchor'
IN_SUBSAMPLE_COL = 'In Subsample'
NBR_ANNOTATORS = 5
CLASS_THRESH = 3
STYLE_TYPE_COL = 'style type'
VAL_SIMPLICITY = "simplicity"
VAL_FORMALITY = "formality"
STYLE_DIMS = [VAL_SIMPLICITY, VAL_FORMALITY]
FORMAL_KEY = 'f-'
SIMPLE_KEY = 'c-'
QUADRUPLE = 'quadruple'
TRIPLE = 'triple'
ACCURACY_COL = 'Accuracy'
MODEL_NAME_COL = 'Model Name'
NBR_SUBSTITUTION = 'nbr_substitution'
CONTRACTION = 'contraction'
SUBSAMPLE_SIZE = 300
VAL_POLITENESS = "politeness"
VAL_CONTRACTION = "contraction"
VAL_LEETSPEAK = "leetspeak"
EVAL_BATCH_SIZE = 64
cur_dir = os.path.dirname(os.path.realpath(__file__))
if "uu_cs_nlpsoc" in cur_dir:
    EVAL_BATCH_SIZE = 64
logging.info('EVAL_BATCH_SIZE={}'.format(EVAL_BATCH_SIZE))


def set_torch_device():
    import torch
    global device
    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


