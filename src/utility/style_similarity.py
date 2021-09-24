"""

    STYLE similarity is at 1 if the same style (also for cosine-sim)
        at 0 or -1 if distinct style

"""
from abc import ABC
import nltk
from typing import Tuple, List
import numpy

import logging
# from sentence_transformers import SentenceTransformer, models
from to_add_const import DEEPSTYLE_MODEL_PATH, LIWC_PATH, LIWC_STYLE_ACC_PATTERN

# see SBERT models here:
#   - https://www.sbert.net/docs/pretrained_models.html
#   - https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0
STSB_BERT_BASE = 'stsb-bert-base'  # bert base uncased, no version with bert base cased; initially tested
MPNET_SBERT = 'all-mpnet-base-v2'  # ~500MB
PARAPHRASEMPNET_SBERT = 'paraphrase-multilingual-mpnet-base-v2'  # 1GB

UNIVERSAL_SENTENCE_ENCODER_PATH = "https://tfhub.dev/google/universal-sentence-encoder/4"

# Goal: style similarity between two utterances, optional: dependant on the corpus

import set_for_global

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
        return [self.similarity(sentences_1[i], sentences_2[i]) for i in range(len(sentences_1))]


class LevenshteinSimilarity(Similarity):
    def similarity(self, sentence_1: str, sentence_2: str) -> float:
        if len(sentence_1) == 0 and len(sentence_2) == 0:
            return self.SAME
        else:
            # minimum number of (character-level) edits needed to transform one string into the other
            sim = nltk.edit_distance(sentence_1, sentence_2)
            return 1 - sim / max(len(sentence_1), len(sentence_2))  # 0 edits -> SAME=1


class LIWCStyleSimilarity(Similarity):
    def similarity(self, sentence_1: str, sentence_2: str) -> float:
        """
        written according to https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/bcdd5d1d5e1ad838bc512c52ccdf5a03b43040f7/convokit/coordination/coordination.py
            and paper description of style similarity in
                https://www.cs.cornell.edu/~cristian/Echoes_of_power_files/echoes_of_power.pdf
        """
        if len(sentence_1) == 0 and len(sentence_2) == 0:
            return self.SAME
        else:
            word_dict, categories, liwc_trie = _compute_liwc_reverse_dict()
            assert len(
                categories) == 8, "not enough style dimensions found"  # there should be 8 dimensions ("strictly-non-topical" and "non-concious" style dimensions)
            style_dim_u1 = [1 if cat in _annot_liwc_cats(sentence_1, liwc_trie) else 0 for cat in categories]
            style_dim_u2 = [1 if cat in _annot_liwc_cats(sentence_2, liwc_trie) else 0 for cat in categories]
            return cosine_sim(numpy.array(style_dim_u1), numpy.array(style_dim_u2))


class LIWCSimilarity(Similarity):
    def similarity(self, sentence_1: str, sentence_2: str) -> float:
        if len(sentence_1) == 0 and len(sentence_2) == 0:
            return self.SAME
        else:
            # parse, category_names = liwc.load_token_parser('../data/LIWC2015 Dictionary.dic')
            vector1 = ModelBasedSentenceFeatures.get_liwc_count_vector(sentence_1)
            vector2 = ModelBasedSentenceFeatures.get_liwc_count_vector(sentence_2)
            return cosine_sim(vector1, vector2)


class LIWCFunctionSimilarity(Similarity):
    def similarity(self, sentence_1: str, sentence_2: str) -> float:
        if len(sentence_1) == 0 and len(sentence_2) == 0:
            return self.SAME
        else:
            # parse, category_names = liwc.load_token_parser('../data/LIWC2015 Dictionary.dic')
            result_dict_1 = ModelBasedSentenceFeatures.get_liwc_categories(sentence_1)
            result_dict_2 = ModelBasedSentenceFeatures.get_liwc_categories(sentence_2)
            function_word_key = 'function'
            if function_word_key in result_dict_1['text_counts']:
                freq_share_1 = result_dict_1['text_counts'][function_word_key] / sum(
                    1 for _ in ModelBasedSentenceFeatures.tokenize(sentence_1))
            else:
                freq_share_1 = 0
            if 'function' in result_dict_2['text_counts']:
                freq_share_2 = result_dict_2['text_counts'][function_word_key] / sum(
                    1 for _ in ModelBasedSentenceFeatures.tokenize(sentence_2))
            else:
                freq_share_2 = 0
            return 1 - abs(freq_share_1 - freq_share_2)


class CharacterThreeGramSimilarity(Similarity):
    def similarity(self, sentence_1: str, sentence_2: str) -> float:
        if len(sentence_1) == 0 and len(sentence_2) == 0:
            return self.SAME
        else:
            try:
                vectors = ModelBasedSentenceFeatures.three_gram_feature([sentence_1, sentence_2])
                # vector2, names2 = UtteranceFeatures.three_gram_feature(utterance_2)
                vector1 = vectors[0, :]
                vector2 = vectors[1, :]
            except ValueError:
                # character n-grams could not be calculated
                return 1

            return cosine_sim(vector1, vector2)


class PunctuationSimilarity(Similarity):
    def similarity(self, sentence_1: str, sentence_2: str) -> float:
        v1 = ModelBasedSentenceFeatures.punctuation_freq(sentence_1)
        v2 = ModelBasedSentenceFeatures.punctuation_freq(sentence_2)
        return cosine_sim(v1, v2)


class RichnessSimilarity(Similarity):
    def similarity(self, sentence_1: str, sentence_2: str) -> float:
        r1 = ModelBasedSentenceFeatures.hapax_legomena_ratio(sentence_1)
        r2 = ModelBasedSentenceFeatures.hapax_legomena_ratio(sentence_2)
        return 1 - abs(r1 - r2)


class WordLengthSimilarity(Similarity):
    def similarity(self, sentence_1: str, sentence_2: str) -> float:
        # calculate the average word length (value between 1 and inf)
        avg_wl_1 = ModelBasedSentenceFeatures.average_word_length(sentence_1)
        avg_wl_2 = ModelBasedSentenceFeatures.average_word_length(sentence_2)
        max_len = max(avg_wl_1, avg_wl_2)
        return 1 - abs(avg_wl_1 - avg_wl_2) / max_len


class UppercaseSimilarity(Similarity):
    def similarity(self, sentence_1: str, sentence_2: str) -> float:
        upper_ratio_1 = ModelBasedSentenceFeatures.total_uppercase(sentence_1)
        upper_ratio_2 = ModelBasedSentenceFeatures.total_uppercase(sentence_2)
        return 1 - abs(upper_ratio_1 - upper_ratio_2)


class PosTagSimilarity(Similarity):
    def similarity(self, sentence_1: str, sentence_2: str) -> float:
        v1 = ModelBasedSentenceFeatures.pos_tag_freq(sentence_1)
        v2 = ModelBasedSentenceFeatures.pos_tag_freq(sentence_2)
        return cosine_sim(v1, v2)


class ModelBasedSimilarity(Similarity):
    def __init__(self):
        super().__init__()
        self.utterance_feature_object = ModelBasedSentenceFeatures()


class UncasedBertSimilarity(ModelBasedSimilarity):
    def similarities(self, sentences_1: List[str], sentences_2: List[str]) -> List[float]:
        e1s = self.utterance_feature_object.static_bert_features(sentences_1)
        e2s = self.utterance_feature_object.static_bert_features(sentences_2)
        return [cosine_sim(e1, e2) for e1, e2 in zip(e1s, e2s)]


class CasedBertSimilarity(ModelBasedSimilarity):
    def similarities(self, sentences_1: List[str], sentences_2: List[str]) -> List[float]:
        e1s = self.utterance_feature_object.static_cbert_features(sentences_1)
        e2s = self.utterance_feature_object.static_cbert_features(sentences_2)
        return [cosine_sim(e1, e2) for e1, e2 in zip(e1s, e2s)]


class SentenceBertSimilarity(ModelBasedSimilarity):
    def __init__(self, sbert_name=STSB_BERT_BASE):
        super(SentenceBertSimilarity, self).__init__()
        self.sbert_name = sbert_name

    def similarity(self, sentence_1: str, sentence_2: str) -> float:
        if len(sentence_1) == 0 and len(sentence_2) == 0:
            return self.SAME
        else:
            emb_1 = self.utterance_feature_object.static_sbert_encoder(sentence_1, model_name=self.sbert_name)
            emb_2 = self.utterance_feature_object.static_sbert_encoder(sentence_2, model_name=self.sbert_name)
            return cosine_sim(emb_1, emb_2)


class MpnetSentenceBertSimilarity(SentenceBertSimilarity):
    def __init__(self):
        super().__init__(sbert_name=MPNET_SBERT)


class ParaMpnetSentenceBertSimilarity(SentenceBertSimilarity):
    def __init__(self):
        super().__init__(sbert_name=PARAPHRASEMPNET_SBERT)


class RobertaSimilarity(ModelBasedSimilarity):
    def similarities(self, sentences_1: List[str], sentences_2: List[str]) -> List[float]:
        e1s = self.utterance_feature_object.static_roberta_features(sentences_1)
        e2s = self.utterance_feature_object.static_roberta_features(sentences_2)
        return [cosine_sim(e1, e2) for e1, e2 in zip(e1s, e2s)]


class DeepstyleSimilarity(ModelBasedSimilarity):
    def similarity(self, sentence_1: str, sentence_2: str) -> float:
        if len(sentence_1) == 0 and len(sentence_2) == 0:
            return self.SAME
        else:
            style_dim_u1 = self.utterance_feature_object.static_deepstyle_features(sentence_1)
            style_dim_u2 = self.utterance_feature_object.static_deepstyle_features(sentence_2)
            return cosine_sim(style_dim_u1, style_dim_u2)


class BERTUncasedNextSentenceSimilarity(ModelBasedSimilarity):
    def similarities(self, sentences_1: List[str], sentences_2: List[str]) -> List[float]:
        return self.utterance_feature_object.static_nextsentencebert_sim(sentences_1, sentences_2)


class BERTCasedNextSentenceSimilarity(ModelBasedSimilarity):
    def similarities(self, sentences_1: List[str], sentences_2: List[str]) -> List[float]:
        return self.utterance_feature_object.static_nextsentencecbert_sim(sentences_1, sentences_2)


class USESimilarity(ModelBasedSimilarity):
    def similarities(self, sentences_1: List[str], sentences_2: List[str]) -> List[float]:
        e1s = self.utterance_feature_object.static_universal_sentence_encoder(sentences_1)
        e2s = self.utterance_feature_object.static_universal_sentence_encoder(sentences_2)
        return [cosine_sim(e1, e2) for e1, e2 in zip(e1s, e2s)]


# UTILITY CLASSES & FUNCTIONS

class ModelBasedSentenceFeatures:
    """
        Class used to calculate features from sentence -- often based on (neural) models
        change liwc path if necessary
    """
    liwc_path = LIWC_PATH

    def __init__(self):
        self.deepstyle_model = None
        self.sentencebert_model = None
        self.next_bert_model = None
        self.next_cbert_model = None
        self.bert_model = None
        self.cbert_model = None
        self.roberta_model = None
        self.use_model = None

    @staticmethod
    def three_gram_feature(text_list: List[str], min_df: float = 0):
        # TODO: some form of tokenization? remove "." or " "?
        """
        For a list of strings (typically 2 for the sentence similarity calculation),
            calculate the character 3-gram vectors
        :param text_list: list of strings
        :param min_df: ignore 3-grams that have a document frequency strictly lower than the given relative threshold
        :return: list of arrays matching the number of strings in text_list
        """
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 3), min_df=min_df)
        try:
            vectorizer_object = vectorizer.fit_transform(text_list)  # [0].toarray()[0]
        except ValueError:
            logging.info("For {} there could not be no character-ngram calculated".format(text_list))
            raise ValueError("empty vocabulary")
        return vectorizer_object.toarray()  # , vectorizer.get_feature_names()

    @staticmethod
    def get_liwc_count_vector(text: str):
        """
        :param text:
        :return: frequency vector for each liwc category for texts
        """
        # see also: https://pypi.org/project/liwc/
        # parse is a function from a token of text to a list of matching LIWC categories
        result_dict = ModelBasedSentenceFeatures.get_liwc_categories(text)
        liwc_vector = numpy.zeros(len(result_dict['category_names']))
        for key, value in result_dict['text_counts'].items():
            liwc_vector[result_dict['category_names'].index(key)] = value
        return liwc_vector

    @staticmethod
    def get_liwc_categories(text, liwc_path=liwc_path):
        import liwc
        from collections import Counter
        parse, category_names = liwc.load_token_parser(liwc_path)
        text_tokens = ModelBasedSentenceFeatures.tokenize(text)
        text_counts = Counter(category for token in text_tokens for category in parse(token))
        return {
            'category_names': category_names,
            'text_counts': text_counts,
            'text_tokens': ModelBasedSentenceFeatures.tokenize(text)
        }

    @staticmethod
    def punctuation_freq(text) -> List[float]:
        """
        calculates the relative frequency per punctuation mark relative to the number of characters in the sentence
          taken from https://github.com/yunitata/coling2018/blob/master/feature_extractor.py
          which was published with https://www.aclweb.org/anthology/C18-1029.pdf

        :param text:
        :return:
        """
        punct = ['\'', ':', ',', '_', '!', '?', ';', ".", '\"', '(', ')', '-']
        count = {}
        for s in text:
            if s in count:
                count[s] += 1
            else:
                count[s] = 1
        count_list = {}
        for d in punct:
            if d in count.keys():
                count_list[d] = count[d]
            else:
                count_list[d] = 0
        return numpy.array(list(count_list.values())) / len(text)

    @staticmethod
    def hapax_legomena_ratio(text: str) -> float:  # not used in paper
        # https://github.com/yunitata/coling2018/blob/master/feature_extractor.py
        # from https://www.aclweb.org/anthology/C18-1029.pdf
        word_list = text.split(" ")
        fdist = nltk.FreqDist(word for word in word_list)
        fdist_hapax = nltk.FreqDist.hapaxes(fdist)
        return float(len(fdist_hapax) / len(word_list))

    @staticmethod
    def average_word_length(text: str) -> float:
        """
        calculates the average word length
        :param text:
        :return: a value between 1 and open ended
        """
        # https://github.com/yunitata/coling2018/blob/master/feature_extractor.py
        word_list = text.split(" ")
        average = sum(len(word) for word in word_list) / len(word_list)
        return average

    @staticmethod
    def total_uppercase(text: str) -> float:
        # https://github.com/yunitata/coling2018/blob/master/feature_extractor.py
        return sum(1 for c in text if c.isupper()) / len(text)

    @staticmethod
    def pos_tag_freq(text: str) -> float:
        # https://github.com/yunitata/coling2018/blob/master/feature_extractor.py
        from nltk.tag import map_tag
        pos_tag = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
        word_list = nltk.word_tokenize(text)
        tag_word = nltk.pos_tag(word_list)
        tag_fd = nltk.FreqDist(map_tag('en-ptb', 'universal', tag) for (word, tag) in tag_word)
        count_tag = {}
        for tag in pos_tag:
            freq = tag_fd.get(tag)
            if freq is None:
                count_tag[tag] = 0
            else:
                count_tag[tag] = freq
        return numpy.array(list(count_tag.values())) / len(word_list)

    @staticmethod
    def tokenize(text):
        # you may want to use a smarter tokenizer
        import re
        for match in re.finditer(r'\w+', text, re.UNICODE):
            yield match.group(0)

    def static_deepstyle_features(self, text: str,
                                  model_file=DEEPSTYLE_MODEL_PATH):
        from deepstyle.model import DeepStyle
        if not self.deepstyle_model:
            self.deepstyle_model = DeepStyle(model_file)
            logging.info("deepstyle model loaded")
        # vector = self.model.embed(text)
        # logging.info(len(vector))
        if text == "":
            return numpy.zeros(768)
        return self.deepstyle_model.embed(text)

    def static_sbert_encoder(self, text: str, model_name=STSB_BERT_BASE):  # 'distilbert-base-nli-stsb-mean-tokens'):
        # 'distilbert-base-nli-stsb-mean-tokens' is
        #     DistilBERT-base-uncased model fine tuned on Natural Language Inferemce (NLI) and
        #     Semantic Textual Similarity Benchmark (STSb)
        from sentence_transformers import SentenceTransformer
        import torch
        if not self.sentencebert_model:
            self.sentencebert_model = SentenceTransformer(model_name)
        with torch.no_grad():
            sentence_embedding = self.sentencebert_model.encode(text, show_progress_bar=False)
        return sentence_embedding

    def static_universal_sentence_encoder(self, text_list: List[str],
                                          model_name: str = UNIVERSAL_SENTENCE_ENCODER_PATH):
        import tensorflow_hub as hub
        # see: https://tfhub.dev/google/universal-sentence-encoder/4
        if not self.use_model:
            self.use_model = hub.load(model_name)
        sentence_embeddings = self.use_model(text_list)
        return sentence_embeddings

    def static_nextsentencebert_sim(self, u1s: List[str], u2s: List[str]):
        from neural_model import SoftmaxNextBertModel
        from neural_model import BERT_UNCASED_BASE_MODEL
        bert_path = BERT_UNCASED_BASE_MODEL
        if not self.next_bert_model:
            logging.info("Loading model from {} ... ".format(bert_path))
            self.next_bert_model = SoftmaxNextBertModel(bert_path)
            logging.info("Finished loading model ...")
        return self.next_bert_model.similarities(u1s, u2s)

    def static_nextsentencecbert_sim(self, u1s: List[str], u2s: List[str]):
        from neural_model import SoftmaxNextBertModel
        from neural_model import BERT_CASED_BASE_MODEL
        cbert_path = BERT_CASED_BASE_MODEL
        if not self.next_cbert_model:
            logging.info("Loading model from {} ... ".format(cbert_path))
            self.next_cbert_model = SoftmaxNextBertModel(cbert_path)
            logging.info("Finished loading model ...")
        return self.next_cbert_model.similarities(u1s, u2s)

    def static_roberta_features(self, texts: List[str]):
        from neural_model import RoBERTaModel, ROBERTA_BASE
        if not self.roberta_model:
            logging.info("Loading model from {} ... ".format(ROBERTA_BASE))
            self.roberta_model = RoBERTaModel(ROBERTA_BASE)
            logging.info("Finished loading model ...")
        return self.roberta_model.forward_batch(texts)

    def static_bert_features(self, texts: List[str]):
        from neural_model import BertModel
        from neural_model import BERT_UNCASED_BASE_MODEL
        if not self.bert_model:
            self.bert_model = BertModel(BERT_UNCASED_BASE_MODEL)
        # return self.bert_embedding_model.forward(text)
        return self.bert_model.forward_batch(texts)

    def static_cbert_features(self, texts: List[str]):
        from neural_model import BertModel
        from neural_model import BERT_CASED_BASE_MODEL
        if not self.cbert_model:
            self.cbert_model = BertModel(BERT_CASED_BASE_MODEL)
        # return self.cbert_embedding_model.forward(text)
        return self.cbert_model.forward_batch(texts)


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


# helper functions,
def _compute_liwc_reverse_dict() -> [List[Tuple[str, str]], List[str]]:
    # taken from: https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/bcdd5d1d5e1ad838bc512c52ccdf5a03b43040f7/convokit/coordination/coordination.py
    with open(LIWC_STYLE_ACC_PATTERN, "r") as f:
        all_words = []
        categories = []  # added
        for line in f:
            cat, pat = line.strip().split("\t")
            # if cat == "auxverb": print(cat, pat)
            # use "#" to mark word boundary
            words = pat.replace("\\b", "#").split("|")
            all_words += [(w[1:], cat) for w in words]
            if cat not in categories:  # added
                categories.append(cat)  # added
    liwc_trie = make_trie(all_words)
    return all_words, categories, liwc_trie


def make_trie(words):
    # taken from: https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/bcdd5d1d5e1ad838bc512c52ccdf5a03b43040f7/convokit/coordination/coordination.py
    root = {}
    for word, cat in words:
        cur = root
        for c in word:
            cur = cur.setdefault(c, {})
        if "$" not in cur:  # use "$" as end-of-word symbol
            cur["$"] = {cat}
        else:
            cur["$"].add(cat)
    return root


def _annot_liwc_cats(utt: str, liwc_trie) -> set:
    # taken from https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/fb109a026b2c4cc60d8f25c1199bafd7bb83c40a/convokit/coordination/coordination.py#L341
    # add liwc_categories field to each utterance
    word_chars = set("abcdefghijklmnopqrstuvwxyz0123456789_")
    # for utt in corpus:
    cats = set()
    last = None
    cur = None
    text = utt.lower() + " "
    for i, c in enumerate(text):
        # slightly different from regex: won't match word after an
        #   apostrophe unless the apostrophe starts the word
        #   -- avoids false positives
        if last not in word_chars and c in word_chars and (last != "'" or not cur):
            cur = liwc_trie
        if cur:
            if c in cur and c != "#" and c != "$":
                if c not in word_chars:
                    if "#" in cur and "$" in cur["#"]:
                        cats |= cur["#"]["$"]  # finished current word
                cur = cur[c]
            elif c not in word_chars and last in word_chars and \
                    "#" in cur:
                cur = cur["#"]
            else:
                cur = None
        if cur and "$" in cur:
            cats |= cur["$"]
        last = c
    return cats

