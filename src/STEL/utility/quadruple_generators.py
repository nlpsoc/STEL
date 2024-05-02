"""
    Quadruple generators for the 2 style dimensions and the 2 style characteristics
"""
import logging
import random
from abc import ABC

import nltk
import pandas as pd

# Generator for the Turker simplification dataset
from STEL.utility.base_generators import WikiAbstractStream
from STEL.utility.const_generators import SIMPLE_TURKER_IDS, \
    CONTRACTION_DICT, \
    NSUBS_PATH, NSUBS_TRANSLATED
from STEL.to_add_const import SIMPLE_TEST_TRUE_CASE_PATH, SIMPLE_TUNE_TRUE_CASE_PATH, GYAFC_PATH, TUNE_GYAFC_PATH

NBR_CHARS = 100

FROM_PUNCTUATION = ["-lrb- ", " -rrb-", "-lsb- ", " -rsb-", "-lcb- ", " -rcb-", " ' ll", " ' re", " ' ve", "n ' t",
                    " 's", " , ", " .", " : ", " ! ", " ? ", " ) .", " ( ", " ) ", " ; ", " )", "( ", " '", " \"."]
TO_TRANSLATION = ["(", ")", "[", "]", "{", "}", "'ll", "'re", "'ve", "n't",
                  "'s", ", ", ".", ": ", "! ", "? ", ").", " (", ") ", "; ", ")", "(", "'", "\"."]

DATA_DIR = ""
MINIMAL_EDIT_DISTANCE = 3


class QuadrupleGenerator(ABC):
    """
        Abstract Generator class of quadruples (triples if return_quad=False)
    """

    def __init__(self, first_style_list=None, first_style_string="first-style-", second_style_list=None,
                 second_style_string="second-style-", edit_dist=False, minimal_edit_distance=MINIMAL_EDIT_DISTANCE,
                 mutable_property=False, return_quad=True, shuffle_iter=False):
        """

        :param shuffle_iter: shuffle first_style_list before returning quads?, default is False
        :param first_style_string: name of the first style
        :param second_style_string: name of the second style
        :param return_quad: whether to return quadruples or triples
        :param edit_dist: whether to skip pairs where edit_distance is equal to or smaller than than minimal_edit_dist
        :param minimal_edit_distance:
        :param mutable_property: the first style list is mutable when called
        """
        if first_style_list:
            self._first_style_sentences = first_style_list
        else:
            self._first_style_sentences = []
        if second_style_list:
            self._second_style_sentences = second_style_list
        else:
            self._second_style_sentences = []

        self.shuffle_iter = shuffle_iter
        self.first_style_const = 1
        logging.info("First style {} has value {} and second style {} has value {}".format(first_style_string,
                                                                                           self.first_style_const,
                                                                                           second_style_string,
                                                                                           1 - self.first_style_const))
        self.first_style_string = first_style_string
        self.second_style_string = second_style_string
        self.return_quad = return_quad
        self.edit_dist = edit_dist
        self.min_edit_dist = minimal_edit_distance
        self.first_is_mut = mutable_property

    @property
    def first_style_sentences(self):
        return self._first_style_sentences

    @property
    def second_style_sentences(self):
        return self._second_style_sentences

    def is_different_enough(self, sentence1, sentence2):
        return sentence1 != '' and sentence2 != '' and sentence1 != sentence2 and \
               (not self.edit_dist or nltk.edit_distance(sentence1, sentence2) > self.min_edit_dist) and \
               (sentence1 not in sentence2) and (sentence2 not in sentence1)

    def __iter__(self):
        """
        Iterator that yields quadruples of the specified type
        :return: a1_is_first_style, (anchor_1, alternative_1), (anchor_2, alternative_2),
            where a1_is_first_style is the style onchor_1 and anchor_1 and alternative_1 are of the same style
            --> to generate STEL tasks with differing answers one has to shuffle the order,
            as now the correct order is always S1-S2

            for triple setup returns a1_is_first_style, (anchor_1, alternative_1), (anchor_1, alternative_2),
        """
        sentence_ids = [i for i in range(len(self.second_style_sentences))]
        if self.shuffle_iter:
            random.shuffle(sentence_ids)

        for cur_s_id in sentence_ids:

            to_extract_a2_from, a2_style, to_extract_a1_from, a1_style, a1_is_first_style = \
                self.select_a1_style(
                    self.first_style_const, self.first_style_sentences, self.first_style_string,
                    self.second_style_sentences, self.second_style_string)

            # SELECT anchor sentences according to i
            #   MAKE sure that the sentences at i are different enough
            if not self.is_different_enough(to_extract_a1_from[cur_s_id], to_extract_a2_from[cur_s_id]):
                if self.first_is_mut:
                    to_extract_a2_from, to_extract_a1_from = \
                        self.find_distinct_enough_for_mutable(cur_s_id, a1_is_first_style,
                                                              to_extract_a2_from,
                                                              to_extract_a1_from)
                if not self.is_different_enough(to_extract_a1_from[cur_s_id], to_extract_a2_from[cur_s_id]):
                    logging.warning("Removing example for sentence '{}' and '{}'"
                                    .format(to_extract_a1_from[cur_s_id].replace('\n', ''),
                                            to_extract_a2_from[cur_s_id].replace('\n', '')))
                    continue
            #   SET the Anchor sentence objects
            anchor_1 = self.get_utterance_object(to_extract_a1_from, a1_style, cur_s_id)
            anchor_2 = self.get_utterance_object(to_extract_a2_from, a2_style, cur_s_id)

            # SELECT the alternative sentence pair
            u2_s_id = random.choice([j for j in range(len(to_extract_a1_from)) if j != cur_s_id])
            while not self.is_different_enough(to_extract_a1_from[u2_s_id], to_extract_a2_from[u2_s_id]):
                if self.first_is_mut:
                    to_extract_a2_from, to_extract_a1_from = \
                        self.find_distinct_enough_for_mutable(u2_s_id, a1_is_first_style,
                                                              to_extract_a2_from,
                                                              to_extract_a1_from)
                if not self.is_different_enough(to_extract_a1_from[u2_s_id], to_extract_a2_from[u2_s_id]):
                    u2_s_id = random.choice([j for j in range(len(to_extract_a1_from)) if j != cur_s_id])
            alternative_1 = self.get_utterance_object(to_extract_a1_from, a1_style, u2_s_id)
            alternative_2 = self.get_utterance_object(to_extract_a2_from, a2_style, u2_s_id)

            if not self.return_quad:
                yield a1_is_first_style, (anchor_1, alternative_1), (anchor_1, alternative_2)
            else:
                yield a1_is_first_style, (anchor_1, alternative_1), (anchor_2, alternative_2)

    def find_distinct_enough_for_mutable(self, cur_s_id, a1_is_first_style, to_extract_a2_from,
                                         to_extract_a1_from):
        """
            For a mutable first style sentence, i.e., the array changes when it is called,
             e.g., because there are sentences annotated by different crowdworkers (simplification corpus),
             call the property self.first_style_sentences up to 10 times to find a distinct enough setting
             if it does not find distinct enough, it will return a not distinct enough setting
        :param cur_s_id:
        :param a1_is_first_style:
        :param to_extract_a2_from:
        :param to_extract_a1_from:
        :return:
        """
        # try if property changes for 10 times
        tries = 0
        while tries < 10 and not self.is_different_enough(to_extract_a1_from[cur_s_id],
                                                          to_extract_a2_from[cur_s_id]):
            if a1_is_first_style == self.first_style_const:
                to_extract_a1_from = self.first_style_sentences
            else:
                to_extract_a2_from = self.first_style_sentences
            tries += 1
        return to_extract_a2_from, to_extract_a1_from

    def select_a1_style(self, first_style_const, first_style_sentences, first_style_string, second_style_sentences,
                        second_style_string):
        """
            Given two same length lists of sentences (usually style-transfer paraphrases),
            randomly decide which of the two will become the anchor style
        :param first_style_const: constant referring to the first style
        :param first_style_sentences: list of sentences written in the first style
        :param first_style_string:  string referring to the first style
        :param second_style_sentences: list of sentences written in the second style
        :param second_style_string: string referring to the second style
        :return: a2_s2_style_sentences, a2_s2_style_name: either first_style_sentences and first_style_string or
        second_style_sentences and second_style_string; opposite for a1_s1_style_sentences, a1_s1_style_name;
        this will be used to extract u1 and u2 from (same) and v from (distinct)

        """

        # RANDOMLY decide whether to have to simple (val_const) or two complex style sentences
        a1_s1_is_first_style = random.choice([first_style_const, -first_style_const])
        if a1_s1_is_first_style == first_style_const:
            a1_s1_style_sentences = first_style_sentences
            a1_s1_style_name = first_style_string
            a2_s2_style_sentences = second_style_sentences
            a2_s2_style_name = second_style_string
        else:
            # SAME sentences by wikipedia
            a1_s1_style_sentences = second_style_sentences
            a1_s1_style_name = second_style_string
            a2_s2_style_sentences = first_style_sentences
            a2_s2_style_name = first_style_string

        return a2_s2_style_sentences, a2_s2_style_name, a1_s1_style_sentences, a1_s1_style_name, a1_s1_is_first_style

    @staticmethod
    def get_utterance_object(to_extract_a1_from, same_string, u1_s_id):
        u1_text = to_extract_a1_from[u1_s_id]
        u1_id = same_string + str(u1_s_id)
        u1 = MockConvokitUtterance.to_convokit_utterance(QuadrupleGenerator.shorten_id(u1_id),
                                                         QuadrupleGenerator.clean_utt(u1_text))
        return u1

    @staticmethod
    def shorten_id(text):
        text = text.replace('-wiki', '').replace('-turk', '-t').replace('simple-', 's-').replace('complex-', 'c-')
        text = text.replace('informal-', 'i-').replace('formal-', 'f-')
        return text

    @staticmethod
    def clean_utt(text: str):
        text = text.replace('\n', ' ')
        if text[-1] == ' ':
            text = text[:-1]
        return text


class FormalQuadrupleGenerator(QuadrupleGenerator):
    """
        on the basis of the GYAFC dataset format
    """

    def __init__(self, formality_path=GYAFC_PATH, control_content=True, shuffle_iter=False, include_tune=True,
                 tune_path=TUNE_GYAFC_PATH, quad=False):
        super().__init__(first_style_string="formal" + "-", second_style_string="informal" + "-", edit_dist=True,
                         return_quad=quad, shuffle_iter=shuffle_iter)
        self.formality_path = formality_path
        self.control_content = control_content
        self.shuffle_iter = shuffle_iter
        self.include_tune = include_tune
        self.tune_path = tune_path

    @property
    def first_style_sentences(self):  # formality
        # return the formality prompts from the formality -> informality direction
        if self._first_style_sentences:
            pass
        else:
            with open(self.formality_path + "formal") as f:
                self._first_style_sentences = f.readlines()
            if self.include_tune:
                with open(self.tune_path + "formal") as f:
                    self._first_style_sentences = self._first_style_sentences + f.readlines()
        return self._first_style_sentences

    @property
    def second_style_sentences(self):
        # return the informality rewrites from the formality -> informality direction
        if self._second_style_sentences:
            pass
        else:
            with open(self.formality_path + "informal.ref0") as f:
                self._second_style_sentences = f.readlines()
            if self.include_tune:
                with open(self.tune_path + "informal.ref0") as f:
                    self._second_style_sentences = self._second_style_sentences + f.readlines()
        return self._second_style_sentences


class SimpleQuadrupleGenerator(QuadrupleGenerator):
    def __init__(self, test_true_cased_path=SIMPLE_TEST_TRUE_CASE_PATH, tune_true_cased_path=SIMPLE_TUNE_TRUE_CASE_PATH,
                 shuffle_iter=False, train_set=True, quad=True, edit_dist=True):
        """
            base data from https://github.com/cocoxu/simplification
        :param shuffle_iter: shuffle the order in which the sentences are returned,
        i.e., not from first to last sentence
        """
        super().__init__(first_style_string="simple-", second_style_string="complex-wiki" + "-", edit_dist=edit_dist,
                         mutable_property=True, return_quad=quad, shuffle_iter=shuffle_iter)
        # self.turker_version = turker_version
        self.w_tune_set = train_set  # whether to include the simplification train split in the generation
        self.test_true_cased_path = test_true_cased_path
        self.tune_true_cased_path = tune_true_cased_path

    @property
    def first_style_sentences(self):  # simple is first style
        if self._first_style_sentences:
            pass
        else:
            self._first_style_sentences, self._second_style_sentences = self._read_in_true_cased()
        # if self.turker_version:
        # RANDOM turker id for this example
        turker_id = random.choice(SIMPLE_TURKER_IDS)
        simple_sentences = self._first_style_sentences[turker_id]
        self.first_style_string = "simple-turk" + str(turker_id) + "-"
        return simple_sentences
        # else:
        #     return self._first_style_sentences

    @property
    def second_style_sentences(self):
        if self._second_style_sentences:
            pass
        else:
            self._first_style_sentences, self._second_style_sentences = self._read_in_true_cased()
        return self._second_style_sentences

    def _read_in_true_cased(self):
        logging.info("READING in True Casing ... ")
        test_tc_df = pd.read_csv(self.test_true_cased_path, sep='\t', header=None)  # read_table(path)
        if self.w_tune_set:
            tune_tc_df = pd.read_csv(self.tune_true_cased_path, sep='\t', header=None)
            test_tc_df = pd.concat([test_tc_df, tune_tc_df])  # .sort_index()

        # TODO: do this with regex?
        wrong_punctuation = FROM_PUNCTUATION
        translation = TO_TRANSLATION

        mutable_simple_sentences = [[] for _ in range(8)]
        complex_sentences = []
        for s_id, s_list in test_tc_df.loc[:, 1:].iterrows():
            for t_id, s in enumerate(s_list):
                try:
                    tmp_s = self.adapt_quotation_marks(s)
                    tmp_s = self.adapt_punctuation(tmp_s, wrong_punctuation, translation)
                    if tmp_s[-1] == ' ':
                        tmp_s = tmp_s[:-1]
                except AttributeError:
                    logging.error('Something went wrong with sentence {} appending empty string in place '
                                  'for sentence number {}'.format(s, len(complex_sentences)))
                    tmp_s = ''

                if t_id == 0:
                    complex_sentences.append(tmp_s)
                else:
                    mutable_simple_sentences[t_id - 1].append(tmp_s)

        return mutable_simple_sentences, complex_sentences

    @staticmethod
    def adapt_punctuation(tmp_s, wrong_punctuation, translation):
        for elem, trans in zip(wrong_punctuation, translation):
            if elem in tmp_s:
                tmp_s = tmp_s.replace(elem, trans)
        return tmp_s

    @staticmethod
    def adapt_quotation_marks(s):
        tmp_s = ''
        seen_left_quote = False
        for word in s.split(" "):
            if word != '"':
                tmp_s += word + ' '
            elif not seen_left_quote:
                tmp_s += word
                seen_left_quote = True
            else:
                tmp_s = tmp_s[:-1] + word + ' '
                seen_left_quote = False
        return tmp_s


class ContractionQuadrupleGenerator(QuadrupleGenerator):
    """
        On the basis of wiki abstract stream generate exactly 100 contraction quadruples
    """

    def __init__(self, shuffle_iter=False, quad=True, contractions=CONTRACTION_DICT, nbr_contractions=NBR_CHARS):
        self.first_style_string = "ction-"
        self.second_style_string = "wiki-"
        super().__init__(first_style_string=self.first_style_string, second_style_string=self.second_style_string,
                         return_quad=quad, shuffle_iter=shuffle_iter)
        new_contractions = self._get_contraction_dict(contraction_dict=contractions)
        self._contractions = new_contractions
        self._wiki_stream = WikiAbstractStream(must_include="'")
        self.nbr_contraction = nbr_contractions

    @staticmethod
    def _get_contraction_dict(contraction_dict=CONTRACTION_DICT):
        contractions = {v: k for k, v in contraction_dict.items()}
        from collections import defaultdict
        new_contractions = defaultdict(list)
        for non_contract, contract in contractions.items():
            if "/" in non_contract:
                for elem in non_contract.split(" / "):
                    new_contractions[elem].append(contract)
            else:
                new_contractions[non_contract].append(contract)
        return new_contractions

    @property
    def first_style_sentences(self):
        if self._first_style_sentences:
            pass
        else:
            self._first_style_sentences, self._second_style_sentences = self._init_sentences()
        return self._first_style_sentences

    @property
    def second_style_sentences(self):
        if self._second_style_sentences:
            pass
        else:
            self._first_style_sentences, self._second_style_sentences = self._init_sentences()
        return self._second_style_sentences

    def _init_sentences(self):
        """
            extract exactly 100 examples of contracted wiki sentences
        :return:
        """
        wiki_sentences = []
        contracted_sentences = []
        contraction_keys = list(self._contractions.keys())
        for s in self._wiki_stream:
            random.shuffle(contraction_keys)
            contracted_sentence = s
            for non_contract in contraction_keys:
                if " " + non_contract + " " in contracted_sentence:
                    contracted_sentence = contracted_sentence.replace(non_contract,
                                                                      random.choice(
                                                                          self._contractions[non_contract]))
                elif non_contract.capitalize() + " " in contracted_sentence.lower():
                    contracted_sentence = contracted_sentence.replace(non_contract.capitalize(),
                                                                      random.choice(
                                                                          self._contractions[non_contract]))
            if contracted_sentence != s and "'" in s:  # make sure there still is a "'" in the sentence
                contracted_sentences.append(contracted_sentence)
                wiki_sentences.append(s)
                if len(wiki_sentences) >= self.nbr_contraction:
                    break
        return contracted_sentences, wiki_sentences


class NumberSubsQuadrupleGenerator(QuadrupleGenerator):
    def __init__(self, nsubs_translated_sent_path=NSUBS_TRANSLATED,
                 nsubs_sent_path=NSUBS_PATH, quad=True,
                 shuffle_iter=False):
        import file_utility
        nsubs_sent_path = file_utility.file_lines_to_list(nsubs_sent_path)
        leet_words_trans = file_utility.file_lines_to_list(nsubs_translated_sent_path)
        super(NumberSubsQuadrupleGenerator, self).__init__(first_style_list=nsubs_sent_path,
                                                           first_style_string="leet-",
                                                           second_style_list=leet_words_trans,
                                                           second_style_string="norm-", return_quad=quad,
                                                           shuffle_iter=shuffle_iter)


class MockConvokitUtterance:
    def __init__(self, u_id, u_text):
        self.text = u_text
        self.id = u_id
        self.speaker = IDClass(u_id)
        self.conversation_id = u_id

    @staticmethod
    def to_convokit_utterance(u1_id, u1_text):
        return MockConvokitUtterance(u1_id, u1_text)


class IDClass:
    def __init__(self, id):
        self.id = id
