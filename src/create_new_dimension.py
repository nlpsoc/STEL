import os.path
import random

import pandas as pd

from STEL.utility import set_for_global
from STEL.utility.set_for_global import ALTERNATIVE12_COL, ALTERNATIVE11_COL, ANCHOR2_COL, ANCHOR1_COL, \
    NBR_FOR_CORRECT_COL, ID_COL, \
    CORRECT_ALTERNATIVE_COL, IN_SUBSAMPLE_COL, STYLE_TYPE_COL, SIMPLICITY, FORMALITY, STYLE_DIMS, FORMAL_KEY, \
    SIMPLE_KEY, NBR_SUBSTITUTION, CONTRACTION, SUBSAMPLE_SIZE


def create_quadruples(style_1_sentences, style_2_sentences, dim_name, style_key1, style_key_2):
    """
        Create quadruples for the STEL task
        style_1_sentences and style_2_sentences are lists of sentences of the respective style with the same content
    Args:
        style_1_sentences:
        style_2_sentences:

    Returns:

    """
    set_for_global.set_global_seed(w_torch=False)

    anchor_1_col = []
    anchor_2_col = []
    alternative_11_col = []
    alternative_12_col = []
    correct_alternative_col = []
    id_col = []
    nbr_for_correct_col = []
    style_type_col = []

    # shuffle the lists in the same way
    index_list = list(range(len(style_1_sentences)))
    # make a reverse dict
    index_dict = {index: i for i, index in enumerate(index_list)}
    # shuffle the index list
    random.shuffle(index_list)
    style_1_sentences = [style_1_sentences[i] for i in index_list]
    style_2_sentences = [style_2_sentences[i] for i in index_list]

    # iterate over first half and the second half of sentences in parallel
    half_len = len(style_1_sentences) // 2
    for i, (style1_a, style2_a, style1_b, style2_b) in enumerate(zip(
            style_1_sentences[:half_len], style_2_sentences[:half_len],
            style_1_sentences[half_len:], style_2_sentences[half_len:])):
        anchors = [style1_a, style2_a]
        styles_anchors = [style_key1, style_key_2]
        alternatives = [style1_b, style2_b]
        styles_alts = [style_key1, style_key_2]
        label = 0

        # randomly decide whether to shuffle anchors
        if random.random() < 0.5:
            anchors = [anchors[1], anchors[0]]
            styles_anchors = [style_key_2, style_key1]
            label = 1 - label
        if random.random() < 0.5:
            alternatives = [alternatives[1], alternatives[0]]
            styles_alts = [style_key_2, style_key1]
            label = 1 - label

        # create quadruple
        anchor_1_col.append(anchors[0])
        anchor_2_col.append(anchors[1])
        alternative_11_col.append(alternatives[0])
        alternative_12_col.append(alternatives[1])
        correct_alternative_col.append(label + 1)
        # QQ_ction-5-wiki-5_ction-30-wiki-30--0
        id_col.append(f"QQ_{styles_anchors[0]}-{index_dict[i]}-{styles_anchors[1]}-{index_dict[i]}_"
                      f"{styles_alts[0]}-{index_dict[i + half_len]}-{styles_alts[1]}-{index_dict[i + half_len]}--{label}")
        nbr_for_correct_col.append("-")
        style_type_col.append(dim_name)

    quad_df = pd.DataFrame({ANCHOR1_COL: anchor_1_col, ANCHOR2_COL: anchor_2_col,
                            ALTERNATIVE11_COL: alternative_11_col, ALTERNATIVE12_COL: alternative_12_col,
                            CORRECT_ALTERNATIVE_COL: correct_alternative_col, ID_COL: id_col,
                            NBR_FOR_CORRECT_COL: nbr_for_correct_col, STYLE_TYPE_COL: style_type_col})

    bench_file_name = "quad_questions"
    bench_file_name += "_{}_{}".format(dim_name, half_len)
    quad_file_ending = ".tsv"

    # make dir if it does not exist
    os.makedirs("../output", exist_ok=True)
    quad_df.to_csv("../output/" + bench_file_name + quad_file_ending, sep='\t')


def main():
    # load sentences
    file1_path = "../Data/STEL/raws/definite_abstract_transformed-wiki_200.txt"
    # extract filename from path
    file1 = os.path.basename(file1_path)
    # extract everything before form-wiki
    dim_name = file1.split("_transformed-wiki")[0].replace("_", "-")
    with open(file1_path, "r") as f:
        style_1_sentences = f.readlines()
    with open("../Data/STEL/raws/definite_abstract_org-wiki_200.txt", "r") as f:
        style_2_sentences = f.readlines()

    create_quadruples(style_1_sentences, style_2_sentences, dim_name, dim_name, "sae")

main()
