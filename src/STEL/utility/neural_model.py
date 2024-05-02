import torch
import numpy as np
from abc import ABC
from transformers import BertTokenizer
import transformers
from typing import List

from STEL.utility.set_for_global import set_global_seed, set_torch_device, set_logging, EVAL_BATCH_SIZE

BERT_MAX_WORDS = 250

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging

set_logging()
transformers.logging.set_verbosity_info()
device = set_torch_device()

BATCH_SIZE = 16

set_global_seed()
BERT_CASED_BASE_MODEL = "bert-base-cased"
BERT_UNCASED_BASE_MODEL = "bert-base-uncased"
ROBERTA_BASE = 'roberta-base'
UNCASED_TOKENIZER = BertTokenizer.from_pretrained(BERT_UNCASED_BASE_MODEL)  # , do_lower_case=True)
CASED_TOKENIZER = BertTokenizer.from_pretrained(BERT_CASED_BASE_MODEL)

# ---------------------------------------------- CODE -------------------------------------------------

# Load pre-trained model tokenizer (vocabulary)
# tokenizer = BertTokenizer.from_pretrained('tuned_bert_path-base-uncased')

# Tokenize input
# text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
# tokenized_text = tokenizer.tokenize(text)


class TransformersModel(ABC):
    """
        abstract class that can (theoretically) load any pretrained huggingface model
    """
    def __init__(self, model_path="", tokenizer_path=None):
        self.model, self.tokenizer = self._load_model(model_path, tokenizer_path)
        self.model.to(device)

    def _load_model(self, model_path, tokenizer_path=None):
        model = transformers.PreTrainedModel.from_pretrained(model_path)
        if tokenizer_path is None:
            tokenizer = transformers.PreTrainedTokenizer(model_path)
        else:
            tokenizer = transformers.PreTrainedTokenizer(tokenizer_path)
        return model, tokenizer

    def forward(self, text):
        """
            https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        :param text:
        :return:
        """
        self.model.eval()
        encoded_dict = tokenize_sentence(text, self.tokenizer)
        encoded_dict.to(device)
        with torch.no_grad():
            predictions = self.model(**encoded_dict, return_dict=True)
        return np.array([emb for emb in predictions.last_hidden_state]).mean()

    def forward_batch(self, u1s, batch_size=EVAL_BATCH_SIZE):
        self.model.eval()
        chunks = (len(u1s) - 1) // batch_size + 1
        avg_embeds = []  # torch.tensor([], device=device)
        for i in range(chunks):
            logging.info("at batch number {}".format(i))
            batch_u1s = u1s[i * batch_size:min((i + 1) * batch_size, len(u1s))]
            encoded_dict = tokenize_sentences(batch_u1s, self.tokenizer)
            encoded_dict.to(device)
            with torch.no_grad():
                outputs = self.model(**encoded_dict, return_dict=True)
            # outputs.last_hidden_state.data[0].mean(dim=0)
            avg_embed = [emb.mean(dim=0).to(device) for emb in outputs.last_hidden_state.data]  # pooler_output
            # avg_embed.to(device)
            avg_embeds = avg_embeds + avg_embed  # torch.cat((avg_embeds, avg_embed))

            # del encoded_dict["input_ids"]
            # del encoded_dict["token_type_ids"]
            # del encoded_dict["attention_mask"]
            del encoded_dict
            del outputs
            del batch_u1s
            torch.cuda.empty_cache()

        return avg_embeds


class RoBERTaModel(TransformersModel):
    # https://huggingface.co/roberta-base
    def __init__(self, model_path=ROBERTA_BASE, tokenizer_path=None):
        super(RoBERTaModel, self).__init__(model_path, tokenizer_path)

    def _load_model(self, model_path, tokenizer_path=None):
        model = transformers.RobertaModel.from_pretrained(model_path)
        if tokenizer_path is None:
            tokenizer = transformers.RobertaTokenizer.from_pretrained(model_path)
        else:
            tokenizer = transformers.RobertaTokenizer.from_pretrained(tokenizer_path)
        return model, tokenizer
        # TODO improve? ...
        # https://huggingface.co/transformers/quicktour.html
        # You can pass a list of sentences directly to your tokenizer.
        # If your goal is to send them through your model as a batch, you probably want to pad them all to the same length,
        # truncate them to the maximum length the model can accept and get tensors back.
        # You can specify all of that to the tokenizer:


class BertModel(TransformersModel):
    def __init__(self, model_path=BERT_UNCASED_BASE_MODEL, tokenizer_path=None, in_eval_mode=True):
        super(BertModel, self).__init__(model_path, tokenizer_path)
        self.model.to(device)
        if in_eval_mode:
            self.model.eval()

    def _load_model(self, model_path, tokenizer_path=None):
        model = transformers.BertModel.from_pretrained(model_path)
        tokenizer = self._set_tokenizer(model_path, tokenizer_path)
        return model, tokenizer

    def _set_tokenizer(self, model_path, tokenizer_path):
        if tokenizer_path is None:
            tokenizer = transformers.BertTokenizer.from_pretrained(model_path)
        else:
            tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer_path)
        return tokenizer


class BertForTwoSentencePredictionModel(BertModel):
    def __init__(self, model_path=BERT_UNCASED_BASE_MODEL, tokenizer_path=None, in_eval_mode=True):
        super(BertForTwoSentencePredictionModel, self).__init__(model_path, tokenizer_path, in_eval_mode=in_eval_mode)

    def forward_two(self, utt1, utt2):
        """
        Returns logit of next sentence prediction head as return
        :param utt1:
        :param utt2:
        :return:
        """
        self.model.eval()
        encoded_dict = tokenize_sentence_pair(utt1, utt2, self.tokenizer)
        encoded_dict.to(device)
        with torch.no_grad():
            predictions = self.model(**encoded_dict)
        # predictions[0, 0] is the score of Next sentence being True and predictions[0, 1] is the score of
        #   Next sentence being False
        # https://github.com/huggingface/transformers/issues/48
        return predictions[0]

    def forward_two_batch(self, u1s, u2s, batch_size=EVAL_BATCH_SIZE):
        self.model.eval()
        # https://stackoverflow.com/questions/41868890/how-to-loop-through-a-python-list-in-batch
        chunks = (len(u1s) - 1) // batch_size + 1
        logits = torch.tensor([], device=device)
        for i in range(chunks):
            logging.info("at batch number {}".format(i))
            batch_u1s = u1s[i * batch_size:min((i + 1) * batch_size, len(u1s))]
            batch_u2s = u2s[i * batch_size:min((i + 1) * batch_size, len(u1s))]
            encoded_dict = tokenize_sentence_pairs(batch_u1s, batch_u2s, self.tokenizer)
            with torch.no_grad():
                outputs = self.model(**encoded_dict, return_dict=True)
            logit_output = outputs["logits"]
            logit_output.to(device)
            # logging.info("current logit output is at {}".format(logit_output))
            # logging.info("logit_output is on cuda: {}".format(logit_output.is_cuda))
            logits.to(device)
            # logging.info("logits is on cuda: {}".format(logits.is_cuda))
            # logging.info("device is {}".format(device))
            logits = torch.cat((logits, logit_output))

            # = {"token_type_ids": token_type_ids,
            #    "input_ids": input_ids,
            #    "attention_mask": attention_masks}
            del encoded_dict["input_ids"]
            del encoded_dict["token_type_ids"]
            del encoded_dict["attention_mask"]
            del encoded_dict
            del outputs
            del batch_u1s
            del batch_u2s
            torch.cuda.empty_cache()

        return logits


class SoftmaxTwoBertModel(BertForTwoSentencePredictionModel):
    def __init__(self, model_path=BERT_UNCASED_BASE_MODEL, tokenizer_path=None, in_eval_mode=True):
        super().__init__(model_path=model_path, tokenizer_path=tokenizer_path, in_eval_mode=in_eval_mode)

    def similarity(self, utt1, utt2):
        logit = self.forward_two(utt1, utt2)
        # outputs has the logits first for the classes [0, 1],
        #   where 0 indicates that the sentences are written in the same style,
        #       or originally: that sentence A is a continuation of sentence B
        # logit = outputs[0]
        sim_value = self.get_sim_from_logit(logit)
        return sim_value

    def similarities(self, u1s: List[str], u2s: List[str]):
        """

        :param u1s:
        :param u2s:
        :return: tensor with similarity values
        """
        logits = self.forward_two_batch(u1s, u2s)
        # logits = outputs["logits"].data
        sim_values = self.get_sim_from_logit(logits, dim=0)  # [self._get_sim_from_logit(logit) for logit in logits]
        return sim_values

    @staticmethod
    def get_sim_from_logit(logit, dim=None):
        """

        :param logit: expects a tensor of matrix, e.g., tensor([[0.5, 0.5]])
        :param dim:
        :return: sim value for label 0, i.e, at 1 if same at 0 of not
        """
        softmax = torch.nn.functional.softmax(logit, dim=1)

        if dim == None:  # only one value for logit [ , ]
            # softmax.tolist()[0][0]
            sim_value = softmax.data[0][0].item()
            # sim_value = softmax.data[0].item()
            return sim_value
        else:
            sim_values = softmax.data[:, 0]
            return sim_values


class SoftmaxNextBertModel(SoftmaxTwoBertModel):
    def __init__(self, model_path=BERT_UNCASED_BASE_MODEL, tokenizer_path=None, in_eval_mode=True):
        super().__init__(model_path=model_path, tokenizer_path=tokenizer_path, in_eval_mode=in_eval_mode)

    def _load_model(self, model_path, tokenizer_path=None):
        model = transformers.BertForNextSentencePrediction.from_pretrained(model_path)
        tokenizer = self._set_tokenizer(model_path, tokenizer_path)
        return model, tokenizer


class UncasedBertForNextSentencePredictionmodel(SoftmaxNextBertModel):
    def __init__(self, model_path=BERT_UNCASED_BASE_MODEL, tokenizer_path=BERT_UNCASED_BASE_MODEL):
        super(SoftmaxNextBertModel, self).__init__(model_path, tokenizer_path)


class CasedBertForNextSentencePredictionModel(SoftmaxNextBertModel):
    def __init__(self, model_path=BERT_UNCASED_BASE_MODEL, tokenizer_path=BERT_UNCASED_BASE_MODEL):
        super(SoftmaxNextBertModel, self).__init__(model_path, tokenizer_path)

    def _load_model(self, model_path, tokenizer_path=None):
        model = transformers.BertForNextSentencePrediction.from_pretrained(model_path)
        tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer_path)
        return model, tokenizer


class SoftmaxSeqBertModel(SoftmaxTwoBertModel):
    def __init__(self, model_path, tokenizer_path=BERT_CASED_BASE_MODEL):
        super(SoftmaxSeqBertModel, self).__init__(model_path, tokenizer_path)

    def _load_model(self, model_path, tokenizer_path=None):
        model = transformers.BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer_path)
        return model, tokenizer



# ________________________________________ Preparing Data ____________________________________________


def tokenize_sentences(texts: List[str], tokenizer):
    texts = [' '.join(text.split(' ')[:512]) for text in texts]
    encoded_dict = tokenizer(texts,
                             return_tensors='pt',
                             # max_length=512,
                             truncation=True,
                             padding=True)
    # input_ids, token_type_ids, attention_mask
    return encoded_dict


def tokenize_sentence(text, tokenizer):
    text = ' '.join(text.split(' ')[:512])
    encoded_dict = tokenizer(text,
                             return_tensors='pt',
                             max_length=512,
                             padding="max_length",
                             truncation="longest_first")
    # input_ids, token_type_ids, attention_mask
    return encoded_dict


def tokenize_sentence_pair(u1, u2, tokenizer):
    # encoded_dict = tokenizer.encode_plus(
    #     "[CLS] " + u1 + " [SEP] " + u2 + " [SEP]",
    #     add_special_tokens=False,
    #     truncation=True,
    #     max_length=512,  # TODO: use batches with variable length ?
    #     pad_to_max_length=True,
    #     return_attention_mask=True,
    #     return_tensors='pt',
    # )
    # u1 = ' '.join(u1.split(' ', BERT_MAX_WORDS+1)[:BERT_MAX_WORDS])
    # make sure the paragraphs are not longer than the max length
    # last part -> first part; first part -> first part; last part -> last part; first part -> last part
    u1 = ' '.join(u1.split(' ')[-BERT_MAX_WORDS:])
    u2 = ' '.join(u2.split(' ')[:BERT_MAX_WORDS])
    # u2 = ' '.join(u2.split(' ', BERT_MAX_WORDS+1)[:BERT_MAX_WORDS])
    encoded_dict = tokenizer(u1, u2,
                             return_tensors='pt',
                             max_length=512,
                             padding="max_length",
                             truncation="longest_first")
    # input_ids, token_type_ids, attention_mask
    return encoded_dict


def tokenize_sentence_pairs(u1s, u2s, tokenizer):
    input_ids = []
    attention_masks = []
    token_type_ids = []

    for u1, u2 in zip(u1s, u2s):
        encoded_dict = tokenize_sentence_pair(u1, u2, tokenizer)
        encoded_dict.to(device)
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask']) \
            # Token ids whether token belongs to u1 or u2
        token_type_ids.append(encoded_dict['token_type_ids'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    input_ids.to(device)
    attention_masks.to(device)
    token_type_ids.to(device)

    encoded_dicts = {"token_type_ids": token_type_ids,
                     "input_ids": input_ids,
                     "attention_mask": attention_masks}

    return encoded_dicts
