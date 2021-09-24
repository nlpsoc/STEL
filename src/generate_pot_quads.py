"""
    Generate the potential task instances for
        formal/informal dimension
        complex/simple dimension

    Generate the task instances for
        number substitution characteristic
        contraction characteristic

    Take Subsample of potential tasks
"""

import logging
import sys
import os
import pickle
import argparse
import random

from set_for_global import ALTERNATIVE12_COL, ALTERNATIVE11_COL, ANCHOR2_COL, ANCHOR1_COL, NBR_FOR_CORRECT_COL, ID_COL, \
    CORRECT_ALTERNATIVE_COL, IN_SUBSAMPLE_COL, STYLE_TYPE_COL, VAL_SIMPLICITY, VAL_FORMALITY, STYLE_DIMS, FORMAL_KEY, \
    SIMPLE_KEY, NBR_SUBSTITUTION, CONTRACTION, SUBSAMPLE_SIZE

STEL_CHAR_KEYWORDS = [CONTRACTION, NBR_SUBSTITUTION]  # [NBR_SUBSTITUTION, CONTRACTION]
SIMPLE_TURKER_VERSION = True

sys.path.append(os.path.join('.', 'utility'))
import quadruple_generators
from qualtrics_constants import quadruple_id

FILE_PATH = ""
TOTAL = 30000
MIN_VALID_UTTS = 4  # i.e., about 4 valid utterances per author


def main(action="generate", style_char=None, quad_tsv_file=None):
    """
    Generate the triplets_gen (u1, u2), (u1, v) for training & testing or validation

    :param quad_tsv_file: all quadruple tasks to generate the subsample from
    :param style_char: style characteristics to generate when calling "generate-char-quads",
                       STEL_CHAR_KEYWORDS when generating both nsubs as well as contractions
    :param action:
    :return:
    """
    file_path = FILE_PATH

    if action == "generate-char-quads":
        # Generate the characteristics quadruple sets (i.e., contraction and number substitution)
        #   and save to tsv

        import set_for_global
        set_for_global.set_global_seed(w_torch=False)
        import pandas as pd
        # ONLY save QUADRUPLES as triples can be generated from them as well
        quad_df = pd.DataFrame(columns=[ANCHOR1_COL, ANCHOR2_COL, ALTERNATIVE11_COL, ALTERNATIVE12_COL,
                                        CORRECT_ALTERNATIVE_COL, ID_COL, NBR_FOR_CORRECT_COL, STYLE_TYPE_COL])
        logging.info("Generating style characteristic quadruples for tsv ... ")
        for char_type in style_char:
            nbr_nbrsubs = 0
            nbr_contract = 0
            if char_type == NBR_SUBSTITUTION:
                triplet_gen = quadruple_generators.NumberSubsQuadrupleGenerator(quad=True)
            if char_type == CONTRACTION:
                triplet_gen = quadruple_generators.ContractionQuadrupleGenerator(quad=True, nbr_contractions=102)

            # ITERATE over quadruples, u1 is paraphrase of u3, u2 is parapharse of u4,
            #    u1 and u2 have same type, u3 and u4 have same type, u2 is always the correct answer
            for i, (val, (anchor_1, alternative_a), (anchor_2, alternative_b)) in enumerate(triplet_gen):
                alternatives = [alternative_a, alternative_b]
                random.shuffle(alternatives)
                correct_alternative = 0 if alternative_a.text == alternatives[0].text else 1
                quad_id = quadruple_id.format(anchor_1.id, anchor_2.id, alternatives[0].id, alternatives[1].id,
                                              correct_alternative)
                # assert len(quad_id) <= 50, "ID error: id for quad question is too long ..."

                if i % 100 == 0:
                    logging.info('At example number {}'.format(i))

                quad_df = quad_df.append(
                    {ANCHOR1_COL: anchor_1.text, ANCHOR2_COL: anchor_2.text, ALTERNATIVE11_COL: alternatives[0].text,
                     ALTERNATIVE12_COL: alternatives[1].text, CORRECT_ALTERNATIVE_COL: correct_alternative + 1,
                     ID_COL: quad_id, STYLE_TYPE_COL: char_type},
                    ignore_index=True)
                if char_type == NBR_SUBSTITUTION:
                    nbr_nbrsubs += 1
                elif char_type == CONTRACTION:
                    nbr_contract += 1

        bench_file_name = "quad_questions_char"
        if NBR_SUBSTITUTION in style_char:
            bench_file_name += "_{}-{}".format(NBR_SUBSTITUTION, nbr_nbrsubs)
        if CONTRACTION in style_char:
            bench_file_name += "_{}-{}".format(CONTRACTION, nbr_contract)
        quad_file_ending = ".tsv"

        quad_df.to_csv(bench_file_name + quad_file_ending, sep='\t')

        logging.info('saved {} substitution and {} contraction examples to {}'.format(nbr_nbrsubs, nbr_contract,
                                                                                      bench_file_name + quad_file_ending))

    elif action == "generate-quads":
        # Generate the potential style dimension quadruple sets and save to tsv

        import set_for_global
        set_for_global.set_global_seed(w_torch=False)
        import pandas as pd
        # ONLY save QUADRUPLES as triples can be generated from them as well
        quad_df = pd.DataFrame(columns=[ANCHOR1_COL, ANCHOR2_COL, ALTERNATIVE11_COL, ALTERNATIVE12_COL,
                                        CORRECT_ALTERNATIVE_COL, ID_COL, NBR_FOR_CORRECT_COL, IN_SUBSAMPLE_COL])
        # subsample_df = pd.DataFrame(columns=[ANCHOR1_COL, ANCHOR2_COL, ALTERNATIVE11_COL, ALTERNATIVE12_COL,
        #                                 CORRECT_ALTERNATIVE_COL, ID_COL, NBR_FOR_CORRECT_COL])
        subsample_size = SUBSAMPLE_SIZE
        logging.info("Generating quadruples for tsv ... ")

        for val_type in STYLE_DIMS:
            if val_type == VAL_SIMPLICITY:
                triplet_generator = quadruple_generators.SimpleQuadrupleGenerator(quad=True)  # shuffle_iter=True,
                nbr_simple = 0
            elif val_type == VAL_FORMALITY:
                triplet_generator = quadruple_generators.FormalQuadrupleGenerator(quad=True)
                nbr_formal = 0
            else:
                logging.warning("There was an error in loading the triple generator for bench...")
                return

            # ITERATE over quadruples, u1 is paraphrase of u3, u2 is parapharse of u4,
            #    u1 and u2 have same type, u3 and u4 have same type, u2 is always the correct answer
            for i, (val, (anchor_1, alternative_a), (anchor_2, alternative_b)) in enumerate(triplet_generator):
                alternatives = [alternative_a, alternative_b]
                random.shuffle(alternatives)
                correct_alternative = 0 if alternative_a.text == alternatives[0].text else 1
                quad_id = quadruple_id.format(anchor_1.id, anchor_2.id, alternatives[0].id, alternatives[1].id,
                                              correct_alternative)
                assert len(quad_id) <= 50, "ID error: id for quad question is too long ..."

                if i % 100 == 0:
                    logging.info('At example number {}'.format(i))

                quad_df = quad_df.append(
                    {ANCHOR1_COL: anchor_1.text, ANCHOR2_COL: anchor_2.text, ALTERNATIVE11_COL: alternatives[0].text,
                     ALTERNATIVE12_COL: alternatives[1].text, CORRECT_ALTERNATIVE_COL: correct_alternative + 1,
                     ID_COL: quad_id, IN_SUBSAMPLE_COL: False},
                    ignore_index=True)
                if val_type == VAL_SIMPLICITY:
                    nbr_simple += 1
                elif val_type == VAL_FORMALITY:
                    nbr_formal += 1

            if val_type == VAL_SIMPLICITY:
                quad_df.loc[
                    quad_df[quad_df[ID_COL].str.contains(SIMPLE_KEY)].sample(int(SUBSAMPLE_SIZE / 2),
                                                                             replace=False).index,
                    IN_SUBSAMPLE_COL] = True
            else:
                quad_df.loc[
                    quad_df[quad_df[ID_COL].str.contains(FORMAL_KEY)].sample(int(SUBSAMPLE_SIZE / 2),
                                                                             replace=False).index,
                    IN_SUBSAMPLE_COL] = True

        bench_file_name = "quad_questions"
        bench_file_name += "_{}-{}".format(VAL_SIMPLICITY, nbr_simple)
        bench_file_name += "_{}-{}".format(VAL_FORMALITY, nbr_formal)
        quad_file_ending = ".tsv"

        quad_df.to_csv(bench_file_name + quad_file_ending, sep='\t')

        logging.info('saved {} simple and {} formal examples'.format(nbr_simple, nbr_formal))

    elif action == "quads-subsample":
        # get a subsample
        import set_for_global
        set_for_global.set_global_seed(w_torch=False)

        import pandas as pd
        subsample_size = 302  # 108 annotators: 108*14/5=302.4

        assert quad_tsv_file, 'No original tsv file given'
        quad_df = pd.read_csv(quad_tsv_file, sep='\t')

        logging.info("Sampling {} questions from dimensions {} each"
                     .format(int(subsample_size / 2), STYLE_DIMS))

        for i, val_type in enumerate(STYLE_DIMS):
            if val_type == VAL_SIMPLICITY:
                cur_sample_indices = quad_df[
                    (quad_df[ID_COL].str.contains(SIMPLE_KEY)) & (quad_df[IN_SUBSAMPLE_COL] == False)] \
                    .sample(int(subsample_size / 2), replace=False).index
            else:
                cur_sample_indices = quad_df[
                    (quad_df[ID_COL].str.contains(FORMAL_KEY)) & (quad_df[IN_SUBSAMPLE_COL] == False)] \
                    .sample(int(subsample_size / 2), replace=False).index
            if i == 0:
                sample_indices = cur_sample_indices
            else:
                sample_indices = sample_indices.union(cur_sample_indices)

        quad_df.loc[sample_indices, IN_SUBSAMPLE_COL] = True
        quad_df = quad_df.iloc[sample_indices]

        subsample_file_folder = os.path.dirname(quad_tsv_file)
        subsample_file_name = "subsample_quad_questions"
        subsample_file_name += "_{}-{}".format(VAL_SIMPLICITY, int(subsample_size / 2))
        subsample_file_name += "_{}-{}".format(VAL_FORMALITY, int(subsample_size / 2))
        subsample_file_ending = ".tsv"

        quad_df.to_csv(subsample_file_folder + '/' + subsample_file_name + subsample_file_ending, sep='\t')
        return quad_df


def remove_turk_id(cur_id):
    # removes the middle turk from an id like "simple-turk1-41"
    return cur_id.split("-")[0] + "-" + cur_id.split("-")[-1]


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Generating Datasets for (1) Training and Evaluation or (2) Validation.')
    # parser.add_argument('-run_local', '--local_test',
    #                     default=False,
    #                     help="whether this is only a local test run")
    # parser.add_argument('-total', '--nbr_triples_to_extract', default=TOTAL,
    #                     help="Total number of utterances that should be extracted. Should not change this.")
    parser.add_argument('-out', '--output_dir', default=FILE_PATH,
                        help="path to output directory for dataset")
    parser.add_argument('-a', '--action', default="generate", help="generate or only download corpus")

    args = parser.parse_args()
    main(action=args.action)
