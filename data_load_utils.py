import json
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch.nn as nn
from pytorch_pretrained_bert.tokenization import BertTokenizer
import random
import re

from pytorch_pretrained_bert.tokenization import BertTokenizer

class InputExample(object):

    def __init__(self, unique_id, text_a, text_b, label, index, is_claim, is_evidence):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.index = index
        self.is_claim = is_claim
        self.is_evidence = is_evidence


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, label, index, is_claim, is_evidence):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.label = label
        self.index = index
        self.is_claim = is_claim
        self.is_evidence = is_evidence


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                label=example.label,
                index=example.index,
                is_claim=example.is_claim,
                is_evidence = example.is_evidence))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


# def read_examples(input_file, max_evi_num):
#     """Read a list of `InputExample`s from an input file."""
#     examples = []
#     unique_id = 0
#     all_sent_labels = [] # 有几个句子,包括claim
#     all_evi_labels = [] # 句子中哪几个是证据，包括claim
#     with open(input_file, "r", encoding='utf-8') as reader:
#         while True:
#             line = reader.readline()
#             if not line:
#                 break
#             instance = json.loads(line.strip())
#             index = instance["id"]
#             claim = instance['claim']
#             claim = process_sent(claim)
#             label_map = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
#             label = label_map[instance['label']]
#             examples.append(InputExample(unique_id=unique_id, text_a=claim, text_b=None, label=label, index=index, is_claim=True, is_evidence=False))
#             unique_id += 1

#             sent_label = [1]
#             for evidence in instance['evidence'][0:max_evi_num]:
#                 # examples.append(InputExample(unique_id=unique_id, text_a=process_sent(evidence[2]), text_b=None, label=label, index=index, is_claim=False, is_evidence=True))
#                 # unique_id += 1
#                 examples.append(InputExample(unique_id=unique_id, text_a=process_sent(evidence[2]), text_b=claim, label=label, index=index, is_claim=True, is_evidence=True))
#                 unique_id += 1
#                 sent_label.append(1)
#             for i in range(max_evi_num-len(instance['evidence'][0:max_evi_num])):
#                 examples.append(InputExample(unique_id=unique_id, text_a="[PAD]", text_b=None, label=label, index=index, is_claim=True, is_evidence=True))
#                 unique_id += 1
#                 sent_label.append(0)
#             evi_label = instance['evi_labels'][0:max_evi_num]
#             evi_label = evi_label + [0] * (max_evi_num - len(evi_label))
#             evi_label = [1] + evi_label
#             all_sent_labels.append(sent_label)
#             all_evi_labels.append(evi_label)
            
#     return examples, all_sent_labels, all_evi_labels

def read_examples(input_file, max_evi_num):
    """Read a list of `InputExample`s from an input file."""
    label_map = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
    examples = []
    unique_id = 0
    all_sent_labels = [] # 有几个句子,包括claim
    all_evi_labels = [] # 句子中哪几个是证据，包括claim
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            instance = json.loads(line.strip())
            index = instance["index"]
            claim = instance['claim']
            label = label_map[instance['label']]
            examples.append(InputExample(unique_id=unique_id, text_a=claim, text_b=None, label=label, index=index, is_claim=True, is_evidence=False))
            unique_id += 1

            sent_label = [1]
            for evidence in instance['evidences'][0:max_evi_num]:
                # examples.append(InputExample(unique_id=unique_id, text_a=evidence, text_b=None, label=label, index=index, is_claim=False, is_evidence=True))
                # unique_id += 1
                examples.append(InputExample(unique_id=unique_id, text_a=evidence, text_b=claim, label=label, index=index, is_claim=True, is_evidence=True))
                unique_id += 1
                sent_label.append(1)
            for i in range(max_evi_num-len(instance['evidences'][0:max_evi_num])):
                examples.append(InputExample(unique_id=unique_id, text_a="[PAD]", text_b=None, label=label, index=index, is_claim=True, is_evidence=True))
                unique_id += 1
                sent_label.append(0)
            evi_label = instance['evi_labels'][0:max_evi_num]
            evi_label = evi_label + [0] * (max_evi_num - len(evi_label))
            evi_label = [1] + evi_label
            all_sent_labels.append(sent_label)
            all_evi_labels.append(evi_label)
            
    return examples, all_sent_labels, all_evi_labels


def build_dataset(file_path, evi_max_num, seq_length=512):
    tokenizer = BertTokenizer.from_pretrained("pretrained_models/BERT-Pair", do_lower_case=True)
    examples, all_sent_labels, all_evi_labels = read_examples(file_path, evi_max_num)
    features = convert_examples_to_features(examples=examples, seq_length=seq_length, tokenizer=tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.input_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    all_input_ids = all_input_ids.view(-1,1+evi_max_num,seq_length)  # [10000,6,512]
    all_input_mask = all_input_mask.view(-1,1+evi_max_num,seq_length)
    all_segment_ids = all_segment_ids.view(-1,1+evi_max_num,seq_length)
    all_labels = all_labels.view(-1,1+evi_max_num)[:,0]

    all_sent_labels = torch.LongTensor(all_sent_labels)
    all_evi_labels = torch.LongTensor(all_evi_labels)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels, all_sent_labels, all_evi_labels)

    return train_data