"""
    Evaluate annotations (after receiving annotations from qualtrics tsv files)
"""
import pandas as pd
import logging

# Pandas column names
from STEL.utility.set_for_global import ALTERNATIVE12_COL, ALTERNATIVE11_COL, ANCHOR2_COL, ANCHOR1_COL, NBR_FOR_CORRECT_COL, ID_COL, \
    CORRECT_ALTERNATIVE_COL, NBR_ANNOTATORS, CLASS_THRESH, \
    STYLE_TYPE_COL, SIMPLICITY, FORMALITY, FORMAL_KEY, SIMPLE_KEY
from STEL.utility.qualtrics_constants import QID_PROLIFIC_PID, RESPONSE_TYPE_COL, VALID_RESPONSE

from STEL.utility import set_for_global

set_for_global.set_logging()

COL_CORRECT_CLASS = 'correct_classification'
BORDERLINE_NBR = [2, 3]
TRIPLE_TYPE = 'trip'
QUAD_TYPE = 'quad'
BOTH_TYPE = 'both'


def evaluate_response_overlap(quad_tsv, trip_tsv):
    df_quad = pd.read_csv(quad_tsv, sep='\t')
    df_trip = pd.read_csv(trip_tsv, sep='\t')

    order = ['total', 'simple', 'formal']
    trip_quad = [0] * 3
    ntrip_quad = [0] * 3
    trip_nquad = [0] * 3
    ntrip_nquad = [0] * 3
    nbr_qs = 0

    for (quad_row, quad_values), (trip_row, trip_values) in zip(df_quad.iterrows(), df_trip.iterrows()):
        assert quad_values[ID_COL] == trip_values[ID_COL], 'Something went wrong with the input'

        if trip_values[NBR_FOR_CORRECT_COL] >= CLASS_THRESH and quad_values[NBR_FOR_CORRECT_COL] >= CLASS_THRESH:
            trip_quad[0] += 1
            if SIMPLE_KEY in quad_values[ID_COL]:
                trip_quad[1] += 1
            if FORMAL_KEY in quad_values[ID_COL]:
                trip_quad[2] += 1
        elif trip_values[NBR_FOR_CORRECT_COL] < CLASS_THRESH and quad_values[NBR_FOR_CORRECT_COL] >= CLASS_THRESH:
            ntrip_quad[0] += 1
            if SIMPLE_KEY in quad_values[ID_COL]:
                ntrip_quad[1] += 1
            if FORMAL_KEY in quad_values[ID_COL]:
                ntrip_quad[2] += 1
        elif trip_values[NBR_FOR_CORRECT_COL] >= CLASS_THRESH and quad_values[NBR_FOR_CORRECT_COL] < CLASS_THRESH:
            trip_nquad[0] += 1
            if SIMPLE_KEY in quad_values[ID_COL]:
                trip_nquad[1] += 1
            if FORMAL_KEY in quad_values[ID_COL]:
                trip_nquad[2] += 1
        elif trip_values[NBR_FOR_CORRECT_COL] < CLASS_THRESH and quad_values[NBR_FOR_CORRECT_COL] < CLASS_THRESH:
            ntrip_nquad[0] += 1
            if SIMPLE_KEY in quad_values[ID_COL]:
                ntrip_nquad[1] += 1
            if FORMAL_KEY in quad_values[ID_COL]:
                ntrip_nquad[2] += 1
        else:
            logging.error('Something went wrong ...')

        nbr_qs += 1

    logging.info('Share of correct triple and quad: {}/{} ~ {}'.format(trip_quad[0], nbr_qs, trip_quad[0] / nbr_qs))
    logging.info(
        '      from that share simple: {}/{} ~ {} '.format(trip_quad[1], trip_quad[0], trip_quad[1] / trip_quad[0]))
    logging.info(
        '      from that share formal: {}/{} ~ {}'.format(trip_quad[2], trip_quad[0], trip_quad[2] / trip_quad[0]))
    logging.info('Share of incorrect triple and incorrect quad: {}/{} ~ {}'.format(ntrip_nquad[0], nbr_qs,
                                                                                   ntrip_nquad[0] / nbr_qs))
    logging.info('      from that share simple: {}/{} ~ {} '.format(ntrip_nquad[1], ntrip_nquad[0],
                                                                    ntrip_nquad[1] / ntrip_nquad[0]))
    logging.info('      from that share formal: {}/{} ~ {}'.format(ntrip_nquad[2], ntrip_nquad[0],
                                                                   ntrip_nquad[2] / ntrip_nquad[0]))
    logging.info('Total share of overlap: {}'.format((trip_quad[0] + ntrip_nquad[0]) / nbr_qs))

    logging.info(
        'Share of incorrect triple and correct quad: {}/{} ~ {}'.format(ntrip_quad[0], nbr_qs, ntrip_quad[0] / nbr_qs))
    logging.info(
        '      from that share simple: {}/{} ~ {} '.format(ntrip_quad[1], ntrip_quad[0], ntrip_quad[1] / ntrip_quad[0]))
    logging.info(
        '          which corresponds to overall {}/{} ~{}'.format(ntrip_quad[1], nbr_qs, ntrip_quad[1] / nbr_qs))
    logging.info(
        '      from that share formal: {}/{} ~ {}'.format(ntrip_quad[2], ntrip_quad[0], ntrip_quad[2] / ntrip_quad[0]))
    logging.info(
        '          which corresponds to overall {}/{} ~ {}'.format(ntrip_quad[2], nbr_qs, ntrip_quad[2] / nbr_qs))
    logging.info(
        'Share of correct triple and incorrect quad: {}/{} ~ {}'.format(trip_nquad[0], nbr_qs, trip_nquad[0] / nbr_qs))
    logging.info(
        '      from that share simple: {}/{} ~ {}'.format(trip_nquad[1], trip_nquad[0], trip_nquad[1] / trip_nquad[0]))
    logging.info(
        '          which corresponds to overall {}/{} ~ {}'.format(trip_nquad[1], nbr_qs, trip_nquad[1] / nbr_qs))
    logging.info(
        '      from that share formal: {}/{} ~ {}'.format(trip_nquad[2], trip_nquad[0], trip_nquad[2] / trip_nquad[0]))
    logging.info(
        '          which corresponds to overall {}/{} ~ {}'.format(trip_nquad[2], nbr_qs, trip_nquad[2] / nbr_qs))


def evaluate_responses(tsv_all_qs='../survey/Small-Pilot-Triple_March+11,+2021_03.24/'
                                  'Small-Pilot-Triple_March 11, 2021_03.24.tsv',
                       tsv_triple=None, tsv_quadruple=None, print_all_participants=False, same_share_STLE=False):
    """
        Print statistics of survey results, i.e., #screened out, #correct marjority votes, #incorrect majority votes
            and save correct majority votes
    :param tsv_text:
    :param tsv_triple:
    :return:
    """

    assert tsv_triple is not None or tsv_quadruple is not None, 'Something went wrong... no valid response format given'

    # SET the questions type
    if tsv_triple is None or tsv_quadruple is None:
        q_type = TRIPLE_TYPE if tsv_triple is not None else QUAD_TYPE
    else:
        q_type = BOTH_TYPE
        logging.info('Evaluating on both setups ...')

    # READ IN ...
    df_all_qs = pd.read_csv(tsv_all_qs, sep='\t', encoding='utf-8',
                            dtype={COL_CORRECT_CLASS: 'string', ANCHOR1_COL: 'string', ANCHOR2_COL: 'string',
                                   ALTERNATIVE11_COL: 'string', ALTERNATIVE12_COL: 'string',
                                   CORRECT_ALTERNATIVE_COL: 'int'})
    q_types = []
    if q_type == BOTH_TYPE or q_type == TRIPLE_TYPE:
        q_types.append(TRIPLE_TYPE)
        df_triple = read_in_annotations(tsv_triple)
        _, trip_q_column_names = get_column_names(df_triple)
    if q_type == BOTH_TYPE or q_type == QUAD_TYPE:
        q_types.append(QUAD_TYPE)
        df_quad = read_in_annotations(tsv_quadruple)
        quad_q_column_names, _ = get_column_names(df_quad)

    maj_q_columns = []
    for q_type in q_types:
        # FIND question columns ...
        q_column_names = trip_q_column_names if q_type == TRIPLE_TYPE else quad_q_column_names
        df_q_type = df_triple if q_type == TRIPLE_TYPE else df_quad

        df_pg, nbr_borderline_answers, nbr_borderline_correct, nbr_correct_formal_votes, nbr_correct_maj_votes, \
        nbr_correct_simple_votes, nbr_formal_borderline, nbr_formal_borderline_correct, nbr_formal_qs, \
        nbr_simple_borderline, nbr_simple_borderline_correct, nbr_simple_qs, participant_performance, save_df, \
        saved_triples, correct_column_names = evaluate_per_question(df_q_type, q_column_names, q_type, df_all_qs)

        maj_q_columns.append(correct_column_names)

        # SAVE results to tsv file
        save_filename = '../test/output/{}_annotation-results.tsv'.format(q_type)
        logging.info('Saving results to {}'.format(save_filename))
        save_df[STYLE_TYPE_COL] = [FORMALITY if FORMAL_KEY in row[ID_COL] else SIMPLICITY
                                   for _, row in save_df.iterrows()]
        save_df.to_csv(save_filename, sep='\t')

        if same_share_STLE:
            logging.info('sampling same size for STLE ...')
            save_stle = pd.DataFrame(columns=[ANCHOR1_COL, ANCHOR2_COL, ALTERNATIVE11_COL, ALTERNATIVE12_COL,
                                              CORRECT_ALTERNATIVE_COL, ID_COL, NBR_FOR_CORRECT_COL, STYLE_TYPE_COL])
            simple_df = save_df[(save_df[STYLE_TYPE_COL] == SIMPLICITY) &
                                (save_df[NBR_FOR_CORRECT_COL] >= CLASS_THRESH)]
            formal_df = save_df[(save_df[STYLE_TYPE_COL] == FORMALITY) &
                                (save_df[NBR_FOR_CORRECT_COL] >= CLASS_THRESH)]
            if len(simple_df) < len(formal_df):
                save_stle = pd.concat([save_stle, simple_df], ignore_index=True)
                save_stle = pd.concat([save_stle, formal_df.sample(len(simple_df))])
                nbr_dim = len(simple_df)
            else:
                save_stle = pd.concat([save_stle, formal_df], ignore_index=True)
                save_stle = pd.concat([save_stle, simple_df.sample(len(formal_df))])
                nbr_dim = len(formal_df)

            stle_filename = '../test/output/{}_stle-dimensions_formal-{}_complex-{}.tsv'.format(q_type, nbr_dim, nbr_dim)
            logging.info('Saving stle dimensions to {}'.format(stle_filename))
            save_stle.to_csv(stle_filename, sep='\t')

        log_results(df_pg, nbr_borderline_answers, nbr_borderline_correct, nbr_correct_formal_votes,
                    nbr_correct_maj_votes, nbr_correct_simple_votes, nbr_formal_borderline,
                    nbr_formal_borderline_correct, nbr_formal_qs, nbr_simple_borderline, nbr_simple_borderline_correct,
                    nbr_simple_qs, participant_performance, saved_triples,
                    print_all_participants=print_all_participants)

    if len(q_types) == 2:
        logging.info('WITH FILTERED MAJORITY QUESTIONS ... ')
        for type_id, q_type in enumerate(q_types):
            q_column_names = trip_q_column_names if q_type == TRIPLE_TYPE else quad_q_column_names
            q_maj_tags = ["QT" + maj_column_name[2:-9] if q_type != QUAD_TYPE else "QQ" + maj_column_name[2:]
                          for maj_column_name in maj_q_columns[(type_id + 1) % 2]]
            q_column_names = [q_column_name for q_column_name in q_column_names if
                              any(substring in q_column_name for substring in q_maj_tags)]
            df_q_type = df_triple if q_type == TRIPLE_TYPE else df_quad

            df_pg, nbr_borderline_answers, nbr_borderline_correct, nbr_correct_formal_votes, nbr_correct_maj_votes, \
            nbr_correct_simple_votes, nbr_formal_borderline, nbr_formal_borderline_correct, nbr_formal_qs, \
            nbr_simple_borderline, nbr_simple_borderline_correct, nbr_simple_qs, participant_performance, save_df, \
            saved_triples, correct_column_names = evaluate_per_question(df_q_type, q_column_names, q_type, df_all_qs)

            # SAVE results to tsv file
            save_filename = '../test/output/{}_annotation-results.tsv'.format(q_type)
            logging.info('Saving results to {}'.format(save_filename))
            save_df.to_csv(save_filename, sep='\t')

            log_results(df_pg, nbr_borderline_answers, nbr_borderline_correct, nbr_correct_formal_votes,
                        nbr_correct_maj_votes, nbr_correct_simple_votes, nbr_formal_borderline,
                        nbr_formal_borderline_correct, nbr_formal_qs, nbr_simple_borderline,
                        nbr_simple_borderline_correct,
                        nbr_simple_qs, participant_performance, saved_triples)


def read_in_annotations(tsv_annotations):
    if type(tsv_annotations) is not list:
        df_triple = pd.read_csv(tsv_annotations, sep='\t', encoding='utf-16')
    else:
        for i, tsv_trip_list_elem in enumerate(tsv_annotations):
            if i == 0:
                df_triple = pd.read_csv(tsv_trip_list_elem, sep='\t', encoding='utf-16')
            else:
                df_triple = pd.concat([df_triple, pd.read_csv(tsv_trip_list_elem, sep='\t', encoding='utf-16')],
                                      ignore_index=True)
    df_triple = filter_valid_responses(df_triple)
    return df_triple


def log_results(df_pg, nbr_borderline_answers, nbr_borderline_correct, nbr_correct_formal_votes, nbr_correct_maj_votes,
                nbr_correct_simple_votes, nbr_formal_borderline, nbr_formal_borderline_correct, nbr_formal_qs,
                nbr_simple_borderline, nbr_simple_borderline_correct, nbr_simple_qs, participant_performance,
                nbr_considered_qs, print_all_participants=False):
    '''

    :param df_pg: data frame with {QID_PROLIFIC_PID: row_id, ID_COL: q_tag, 'Rating': int(response)},
     i.e., some include more than n number of raters
    :param nbr_borderline_answers:
    :param nbr_borderline_correct:
    :param nbr_correct_formal_votes:
    :param nbr_correct_maj_votes:
    :param nbr_correct_simple_votes:
    :param nbr_formal_borderline:
    :param nbr_formal_borderline_correct:
    :param nbr_formal_qs:
    :param nbr_simple_borderline:
    :param nbr_simple_borderline_correct:
    :param nbr_simple_qs:
    :param participant_performance:
    :param nbr_considered_qs:
    :param print_all_participants:
    :return:
    '''
    participant_performance[COL_CORRECT_CLASS] = participant_performance[COL_CORRECT_CLASS].div(14)
    if print_all_participants:
        for row_id, values in participant_performance.iterrows():
            logging.info(
                "      Participant {} has accuracy {}".format(values[QID_PROLIFIC_PID], values[COL_CORRECT_CLASS]))
    logging.info("Overall average Participant accuracy: {}".format(participant_performance[COL_CORRECT_CLASS].mean()))
    logging.info("Number of Participants with less than 0.5 accuracy: {}/{}"
                 .format(len([values[COL_CORRECT_CLASS] for _, values in participant_performance.iterrows()
                              if values[COL_CORRECT_CLASS] < 0.5]), len(participant_performance[COL_CORRECT_CLASS])))
    # Performance
    logging.info(
        "Number of correct majority votes: {}/{} ~ {} Accuracy".format(nbr_correct_maj_votes, nbr_considered_qs,
                                                                       nbr_correct_maj_votes / nbr_considered_qs))
    logging.info("Number of correct simple maj votes: {}/{} ~ {} Accuracy".format(nbr_correct_simple_votes,
                                                                                  nbr_simple_qs,
                                                                                  nbr_correct_simple_votes /
                                                                                  nbr_simple_qs))
    logging.info("Number of correct formal maj votes: {}/{} ~ {} Accuracy".format(nbr_correct_formal_votes,
                                                                                  nbr_formal_qs,
                                                                                  nbr_correct_formal_votes /
                                                                                  nbr_formal_qs))
    # Borderline rating
    logging.info("Number of borderline ratings, i.e., majority votes in {} is at {}/{} out of which {} are correct "
                 "~ {} Accuracy".format(BORDERLINE_NBR, nbr_borderline_answers, nbr_considered_qs,
                                        nbr_borderline_correct,
                                        nbr_borderline_correct / nbr_borderline_answers))
    logging.info("Number of  simple borderline ratings: {}/{} out of which {} are correct ~ {} Accuracy"
                 .format(nbr_simple_borderline, nbr_simple_qs, nbr_simple_borderline_correct,
                         nbr_simple_borderline_correct / nbr_simple_borderline))
    logging.info("Number of formal borderline ratings: {}/{} out of which {} are correct ~ {} Accuracy"
                 .format(nbr_formal_borderline, nbr_formal_qs, nbr_formal_borderline_correct,
                         nbr_formal_borderline_correct / nbr_formal_borderline))
    # Fleiss's Kappa
    #   Overall
    ratings = [(item, category) for item, category in zip(df_pg[ID_COL], df_pg['Rating'])]
    #       Simple
    s_ratings = [(item, category) for item, category in zip(df_pg[ID_COL], df_pg['Rating']) if SIMPLE_KEY in item]
    #       Complex
    f_ratings = [(item, category) for item, category in zip(df_pg[ID_COL], df_pg['Rating']) if FORMAL_KEY in item]
    # n_raters = len(df_numeric_valid)
    n_raters = NBR_ANNOTATORS
    logging.info('Overall Fleiss\'s Kappa: {}'.format(fleiss_kappa(ratings, n_raters)))
    logging.info('  Simple Fleiss\'s Kappa: {}'.format(fleiss_kappa(s_ratings, n_raters)))
    logging.info('  Formal Fleiss\'s Kappa: {}'.format(fleiss_kappa(f_ratings, n_raters)))


def filter_valid_responses(df_responses):
    # Filter for valid responses
    df_responses = get_valid_responses(df_responses)
    logging.info('number of valid responses {} which corresponds to a possible {} questions'
                 .format(len(df_responses), int(len(df_responses) * 14 / 5)))
    return df_responses


def get_column_names(df_numeric):
    trip_q_column_names = [column_name for column_name in df_numeric.columns
                           if column_name[:2] == 'QT' and 'intro' not in column_name]
    quad_q_column_names = [column_name for column_name in df_numeric.columns
                           if column_name[:2] == 'QQ' and '_1_RANK' in column_name and 'intro' not in column_name]
    assert (len(trip_q_column_names) > 0 and len(quad_q_column_names) == 0) or \
           (len(quad_q_column_names) > 0 and len(trip_q_column_names) == 0), \
        'Something went wrong with the question tags. ' \
        'They might not cleanly separate triple and quadruple questions ...'
    return quad_q_column_names, trip_q_column_names


def evaluate_per_question(df_numeric_valid, q_column_names, q_type, df_all_qs, with_logging=False):
    '''

    :param df_numeric_valid:
    :param q_column_names:
    :param q_type:
    :param df_all_qs:
    :param with_logging:
    :return:
    '''
    save_df = pd.DataFrame(columns=[ANCHOR1_COL, ANCHOR2_COL, ALTERNATIVE11_COL, ALTERNATIVE12_COL,
                                    CORRECT_ALTERNATIVE_COL, NBR_FOR_CORRECT_COL, ID_COL])
    logging.info('Evaluating on {} questions'.format(len(q_column_names)))
    # EVALUATE
    participant_performance = df_numeric_valid[QID_PROLIFIC_PID].to_frame()
    participant_performance[COL_CORRECT_CLASS] = 0
    correct_maj_q_column_names = []
    nbr_correct_maj_votes = 0
    saved_triples = 0
    nbr_correct_simple_votes = 0
    nbr_correct_formal_votes = 0
    nbr_simple_qs = 0
    nbr_formal_qs = 0
    nbr_borderline_answers = 0  # votes of 2 or 3 for correct alternative
    nbr_simple_borderline = 0
    nbr_formal_borderline = 0
    nbr_borderline_correct = 0
    nbr_simple_borderline_correct = 0
    nbr_formal_borderline_correct = 0
    nbr_more_resp = 0
    # coder_triples = []
    df_pg = pd.DataFrame(columns=[QID_PROLIFIC_PID, ID_COL, 'Rating'])
    for q_column_name in q_column_names:
        # GET question text
        # q_text = df_numeric[q_tag].values[0]
        q_tag = q_column_name
        if q_type == QUAD_TYPE:
            # REMOVE from for example 'QQ-intro_f-1--0_0_1_RANK' the substring '_0_1_RANK'
            #        which is always exactly the last 7 characters
            q_tag = q_column_name[:-9]
        else:
            q_tag = "QQ" + q_tag[2:]

        # FROM tsv q extract
        df_q = df_all_qs[df_all_qs[ID_COL] == q_tag]
        #   ANCHOR
        # anchor = get_anchor(q_type, df_q)
        #   CORRECT ALTERNATIVE (numeric)
        correct_alternative = df_q[CORRECT_ALTERNATIVE_COL].values[0]

        # GET the responses to q
        q_numeric_responses = get_responses_to_q(q_column_name, df_numeric_valid)  # , q_text_responses  df_text_valid
        nbr_responses = q_numeric_responses.shape[0]
        if nbr_responses < NBR_ANNOTATORS:
            logging.error('the number of responses {} for the question {} is too small'.format(nbr_responses, q_tag))
            logging.error(df_q[ANCHOR1_COL])
            continue
        if nbr_responses > NBR_ANNOTATORS:
            logging.warning('the number of responses {} for the question {} is higher than expected. '
                            'Removing {} "correct" vote(s).'
                            .format(nbr_responses, q_tag, nbr_responses - NBR_ANNOTATORS))
            nbr_more_resp += 1

        # ADD VOTES for the respective annotators
        cur_q_annotations = []
        for row_id, response in q_numeric_responses.iteritems():
            if int(response) == correct_alternative:
                participant_performance.loc[row_id, COL_CORRECT_CLASS] += 1
            prolific_id = participant_performance.loc[row_id, QID_PROLIFIC_PID]
            cur_q_annotations.append((prolific_id, q_tag, int(response)))

        # REMOVE VOTES that are too many for 5 annotators, start with removal of correct answers first
        if len(cur_q_annotations) > NBR_ANNOTATORS:
            while (len(cur_q_annotations)) > NBR_ANNOTATORS:
                if correct_alternative in [answer for _, _, answer in cur_q_annotations]:
                    for trip in cur_q_annotations:
                        if trip[2] == correct_alternative:
                            cur_q_annotations.remove(trip)
                            break
                else:
                    cur_q_annotations = cur_q_annotations[:-1]

        for q_ann in cur_q_annotations:
            # ADAPT df_pg
            df_pg = df_pg.append({QID_PROLIFIC_PID: q_ann[0], ID_COL: q_ann[1], 'Rating': q_ann[2]},
                                 ignore_index=True)

        # remove votes if there are too many for this question
        # df_pg.loc[df_pg[q]]

        # GET answer that has the majority vote among annotators
        # votes_per_answer = get_answer_votes(q_numeric_responses)
        votes_per_answer = get_answer_votes(cur_q_annotations)

        # GET index to answer translation (e.g.,, {'1': 'The boy will live', '2': 'The boy'll live'})
        index_to_text = {1: df_q[ALTERNATIVE11_COL].values[0],
                         2: df_q[ALTERNATIVE12_COL].values[0]}  # TODO: here: add answers from separate dataframe

        #   ASSERT that the question has been answered at least NBR_ANNOTATORS times
        if nbr_responses < NBR_ANNOTATORS:
            logging.error(
                "Not enough responses per question for question id {} with question text".format(q_column_name))
            # if q_type == TRIPLE_TYPE:
            #     save_df.append({ANCHOR_COL: anchor, ALTERNATIVE1_COL: index_to_text['1'],
            #                     ALTERNATIVE2_COL: index_to_text['2'], CORRECT_ALTERNATIVE_COL: correct_alternative, nbr})  # TODO: continue
            continue

        # CALCULATE majority answers
        try:
            majority_answer = get_maj_answer(votes_per_answer)  # return is '1' or '2'
            maj_text_answer = index_to_text[int(majority_answer)]
        except ValueError:
            logging.error('Tied answers for question id {} ... going with the wrong choice ...'.format(q_column_name))
            votes_per_answer['{}'.format(correct_alternative)] = votes_per_answer['{}'.format(correct_alternative)] - 1
            majority_answer = get_maj_answer(votes_per_answer)
            # continue

        if votes_per_answer['{}'.format(correct_alternative)] in BORDERLINE_NBR:
            nbr_borderline_answers += 1
            if correct_alternative == int(majority_answer):
                nbr_borderline_correct += 1
            if SIMPLE_KEY in q_column_name:
                nbr_simple_borderline += 1
                if correct_alternative == int(majority_answer):
                    nbr_simple_borderline_correct += 1
            elif FORMAL_KEY in q_column_name:
                nbr_formal_borderline += 1
                if correct_alternative == int(majority_answer):
                    nbr_formal_borderline_correct += 1
            else:
                logging.error('Questions not part of formal or simple dimension')

        # SAVE annotation answer
        save_df = save_df.append({ANCHOR1_COL: df_q[ANCHOR1_COL].values[0], ANCHOR2_COL: df_q[ANCHOR2_COL].values[0],
                                  ALTERNATIVE11_COL: df_q[ALTERNATIVE11_COL].values[0],
                                  ALTERNATIVE12_COL: df_q[ALTERNATIVE12_COL].values[0],
                                  CORRECT_ALTERNATIVE_COL: correct_alternative,
                                  NBR_FOR_CORRECT_COL: votes_per_answer[str(correct_alternative)],
                                  ID_COL: q_tag},
                                 ignore_index=True)
        saved_triples += 1

        # For direct stats: check if the majority answer is the correct one
        if with_logging:
            logging.info("For question anchor: {} and {} and answers {}".format(df_q[ANCHOR1_COL].values[0],
                                                                                df_q[ANCHOR2_COL].values[0],
                                                                                index_to_text))
        if correct_alternative == int(majority_answer):
            if with_logging:
                logging.info("{}/{} CORRECT answer".
                             format(votes_per_answer[majority_answer], sum(votes_per_answer.values())))
            nbr_correct_maj_votes += 1
            assert 'intro' not in q_column_name, 'Something went wrong with column filtering'
            correct_maj_q_column_names.append(q_column_name)
            if 'intro' not in q_column_name and SIMPLE_KEY in q_column_name:
                nbr_correct_simple_votes += 1
                nbr_simple_qs += 1
            elif 'intro' not in q_column_name and FORMAL_KEY in q_column_name:
                nbr_correct_formal_votes += 1
                nbr_formal_qs += 1
            else:
                logging.error('could not assign q {} to complex or formal dimension'.format(q_column_name))
        # elif correct_alternative == (((int(majority_answer) - 1) + 1) % 2 + 1):
        else:
            if with_logging:
                logging.info("{}/{} INCORRECT answer".
                             format(votes_per_answer[majority_answer],
                                    sum(votes_per_answer.values())))
            assert 'intro' not in q_column_name, 'Something went wrong with column filtering'
            if 'intro' not in q_column_name and SIMPLE_KEY in q_column_name:
                nbr_simple_qs += 1
            elif 'intro' not in q_column_name and FORMAL_KEY in q_column_name:
                nbr_formal_qs += 1
            else:
                logging.error('could not assign q {} to complex or formal dimension'.format(q_column_name))

    logging.info('{} removed due to more than {} annotations'.format(nbr_more_resp, NBR_ANNOTATORS))
    return df_pg, nbr_borderline_answers, nbr_borderline_correct, nbr_correct_formal_votes, nbr_correct_maj_votes, \
           nbr_correct_simple_votes, nbr_formal_borderline, nbr_formal_borderline_correct, nbr_formal_qs, \
           nbr_simple_borderline, nbr_simple_borderline_correct, nbr_simple_qs, participant_performance, save_df, \
           saved_triples, correct_maj_q_column_names


def fleiss_kappa(ratings, n, k=None):
    '''
    Computes the Fleiss' kappa measure for assessing the reliability of
    agreement between a fixed number n of raters when assigning categorical
    ratings to a number of items.

    Let N be the number of total subjects (here: questions), let n be the number of ratings per subject (here: 5),
    let k be the number of categories into which assignments are made (here: 2).
    Let n_ij represent the number of raters who assigned the i-th subject (question) to the j-th category
     (Alternative 1 or 2).


    see also: https://en.wikipedia.org/wiki/Fleiss%27_kappa

    Args:
        ratings: a list of (item, category)-ratings, i.e., (question, answer)-ratings
        n: number of raters
        k: number of categories, not necessary for calculation
    Returns:
        the Fleiss' kappa score

    See also:
        https://en.wikipedia.org/wiki/Fleiss%27_kappa
    '''
    items = set()
    categories = set()
    n_ij = {}

    for i, c in ratings:
        items.add(i)
        categories.add(c)
        n_ij[(i, c)] = n_ij.get((i, c), 0) + 1

    N = len(items)

    p_j = {}
    for c in categories:
        p_j[c] = sum(n_ij.get((i, c), 0) for i in items) / (1.0 * n * N)

    P_i = {}
    for i in items:
        P_i[i] = (sum(n_ij.get((i, c), 0) ** 2 for c in categories) - n) / (n * (n - 1.0))

    P_bar = sum(P_i.values()) / (1.0 * N)
    P_e_bar = sum(p_j[c] ** 2 for c in categories)

    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)

    return kappa


def get_anchor(q_type, df_q):
    if q_type == TRIPLE_TYPE:
        return df_q[ANCHOR1_COL]
    else:
        return {1: df_q[ANCHOR1_COL],
                2: df_q[ANCHOR2_COL]}
    #     pattern = ' "(.*?)",'
    #     anchor = re.search(pattern, q_text).group(1)
    #     anchor = BeautifulSoup(anchor, "lxml").text
    # else:
    #     pattern1 = '1. "(.*?)" then'
    #     pattern2 = '2. "(.*?)",'
    #     anchor1 = re.search(pattern1, q_text).group(1)
    #     anchor2 = re.search(pattern2, q_text).group(1)
    #     anchor = {1: anchor1, 2: anchor2}
    # return anchor


def get_maj_answer(votes_per_answer):
    """
    :param votes_per_answer: dictionary with {'1': int, '2': int}, where the ints add up to NBR_ANNOTATORS
    :return: the majority answers, i.e., number of votes at least half that of the number of annotators;
     raises error if the answers are tied
    """
    if (len(votes_per_answer) == 1 and '1' in votes_per_answer) or \
            (len(votes_per_answer) == 2 and votes_per_answer['1'] > votes_per_answer['2']):
        return '1'
    elif (len(votes_per_answer) == 1 and '2' in votes_per_answer) or \
            (len(votes_per_answer) == 2 and votes_per_answer['2'] > votes_per_answer['1']):
        return '2'
    else:
        raise ValueError("The answers are equally often selected."
                         " This should have been impossible with the setup of the study.")


def get_index_to_answer(q_numeric_responses, q_text_responses, q_text):
    index_to_text = {int(q_numeric_responses.values[0]): q_text_responses.values[0]}
    if len(set(q_numeric_responses.values)) > 1:
        assert len(set(q_numeric_responses.values)) == 2, \
            "There is something wrong with the set of possible answers for question {}" \
                .format(q_text)
        index_to_text[(1 - (list(index_to_text.keys())[0] - 1)) + 1] = \
            [response
             for response in q_text_responses.values
             if response != q_text_responses.values[0]][0]
    return index_to_text


def get_responses_to_q(q, df_numeric_valid):  # , df_text_valid
    non_null_numeric_responses = df_numeric_valid[df_numeric_valid[q].notnull()][q]
    # non_null_text_responses = df_text_valid[df_numeric_valid[q].notnull()][q]
    return non_null_numeric_responses  # , non_null_text_responses


def get_valid_responses(df_numeric):  # , df_text):
    #   TODO: Filter out annotators that ... ?
    #   Search for column with prolific ID
    prolific_id_column = QID_PROLIFIC_PID  # ''
    # for column_name in df_numeric.columns:
    #     if 'Please enter your Prolific ID here:' in df_numeric[column_name].values[0]:
    #         prolific_id_column = column_name
    #         break
    #   Only select responses from valid participants
    # df_valid = df_numeric[
    #     (df_numeric[prolific_id_column].str.len() == 24) & (~df_numeric[prolific_id_column].str.contains('{'))
    #     ]
    df_valid = df_numeric[
        df_numeric[prolific_id_column] != 'test'
        ]
    df_valid = df_valid[df_valid[RESPONSE_TYPE_COL] == VALID_RESPONSE]
    # df_text_valid = df_text[
    #     (df_numeric[prolific_id_column].str.len() == 24) & (~df_numeric[prolific_id_column].str.contains('{'))]
    return df_valid  # df_text_valid,


def get_answer_votes(trip_responses):
    '''

    :param trip_responses: list of triples of size NBR_ANNOTATORS with (id, q_tag, response)
    :return:
    '''
    # answers = {df_q[ALTERNATIVE11_COL].values[0]: 0, df_q[ALTERNATIVE2_COL].values[0]: 0}
    answers = {'1': 0, '2': 0}
    for _, _, answer in trip_responses:
        answers['{}'.format(int(answer))] += 1
    return answers
