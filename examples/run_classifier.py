# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from examples.run_squad import _compute_softmax
from pytorch_pretrained_bert import BertForSequenceClassification
from pytorch_pretrained_bert.file_utils import read_jsonl_lines, write_items
from pytorch_pretrained_bert.modeling import BertForMultipleChoice
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import printable_text, convert_to_unicode, BertTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputExampleWithList(object):
    """A single training/test example for simple multiple choice classification."""

    def __init__(self, guid, text_a, text_b, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: list. A list containing untokenized text
            text_b: list. containing untokenized text associated of the same size as text_A
            Only must be specified for multiple choice options.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        assert isinstance(text_a, list)
        assert isinstance(text_b, list)
        assert len(text_a) == len(text_b)

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputExampleWithListFourFields(object):
    """A single training/test example for simple multiple choice classification."""

    def __init__(self, guid, text_a, text_b, text_c, text_d, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: list. A list containing untokenized text
            text_b: list. containing untokenized text associated of the same size as text_A
            text_c: list. containing untokenized text associated of the same size as text_A
            text_d: list. containing untokenized text associated of the same size as text_A
            Only must be specified for multiple choice options.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        assert isinstance(text_a, list)
        assert isinstance(text_b, list)
        assert text_c is None or isinstance(text_c, list)
        assert text_d is None or isinstance(text_d, list)
        assert len(text_a) == len(text_b)
        if text_c is not None:
            assert len(text_c) == len(text_a)
        if text_d is not None:
            assert len(text_d) == len(text_a)

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.text_d = text_d
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_jsonl(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        records = []
        with open(input_file, "r") as f:
            for line in f:
                obj = json.loads(line)
                records.append(obj)
        return records


class AnliProcessor(DataProcessor):
    """Processor for the ANLI data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.jsonl")))
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "valid.jsonl")), "dev")

    def get_examples_from_file(self, input_file):
        return self._create_examples(
            self._read_jsonl(input_file), "to-pred")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, records, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, record) in enumerate(records):
            guid = "%s-%s-%s" % (set_type, record['InputStoryid'], record['ending'])

            beginning = record['InputSentence1']
            ending = record['InputSentence5']

            option1 = record['RandomMiddleSentenceQuiz1']
            option2 = record['RandomMiddleSentenceQuiz2']

            answer = int(record['AnswerRightEnding']) - 1

            option1_context = convert_to_unicode(' '.join([beginning, option1]))
            option2_context = convert_to_unicode(' '.join([beginning, option2]))

            label = convert_to_unicode(str(answer))
            examples.append(
                InputExampleWithListFourFields(guid=guid,
                                               text_a=[option1_context, option2_context],
                                               text_b=[ending, ending],
                                               text_c=None,
                                               text_d=None,
                                               label=label
                                               )
            )
        return examples


class AnliProcessor3Option(DataProcessor):
    """Processor for the ANLI data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.jsonl")))
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "valid.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_examples_from_file(self, input_file):
        return self._create_examples(
            self._read_jsonl(input_file, "to-pred")
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, records, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, record) in enumerate(records):
            guid = "%s-%s-%s" % (set_type, record['InputStoryid'], record['ending'])

            beginning = record['InputSentence1']
            ending = record['InputSentence5']

            option1 = record['RandomMiddleSentenceQuiz1']
            option2 = record['RandomMiddleSentenceQuiz2']
            option3 = record['RandomMiddleSentenceQuiz3']

            answer = int(record['AnswerRightEnding']) - 1

            option1_context = convert_to_unicode(' '.join([beginning, option1]))
            option2_context = convert_to_unicode(' '.join([beginning, option2]))
            option3_context = convert_to_unicode(' '.join([beginning, option3]))

            label = convert_to_unicode(str(answer))

            text_a = [option1_context, option2_context, option3_context]
            text_b = [ending, ending, ending]

            examples.append(
                InputExampleWithList(guid=guid,
                                     text_a=text_a,
                                     text_b=text_b,
                                     label=label
                                     )
            )
        return examples


class AnliWithCSKProcessor(DataProcessor):
    """Processor for the ANLI data set."""

    def __init__(self):
        self._labels = []

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.jsonl")))
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "valid.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_examples_from_file(self, input_file):
        return self._create_examples(
            self._read_jsonl(input_file, "to-pred")
        )

    def get_labels(self):
        """See base class."""
        return [str(idx) for idx in range(16)]

    def _create_examples(self, records, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        num_fields = len(
            [x for x in list(records[0].keys()) if x.startswith('RandomMiddleSentenceQuiz')])
        self._labels = [str(idx) for idx in range(1, num_fields + 1)]
        for (i, record) in enumerate(records):
            guid = "%s-%s-%s" % (set_type, record['InputStoryid'], record['ending'])

            beginning = record['InputSentence1']
            ending = record['InputSentence5']

            text_a = []
            text_b = []
            for idx in range(1, num_fields + 1):
                text_a.append(
                    beginning + " " + record["RandomMiddleSentenceQuiz" + str(idx)]
                )
                text_b.append(
                    ending + " Because , " + record['CSK' + str(idx)]
                )

            answer = int(record['AnswerRightEnding']) - 1
            label = convert_to_unicode(str(answer))

            examples.append(
                InputExampleWithListFourFields(guid=guid,
                                               text_a=text_a,
                                               text_b=text_b,
                                               text_c=None,
                                               text_d=None,
                                               label=label
                                               )
            )
        return examples


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(line[3])
            text_b = convert_to_unicode(line[4])
            label = convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, convert_to_unicode(line[0]))
            text_a = convert_to_unicode(line[8])
            text_b = convert_to_unicode(line[9])
            label = convert_to_unicode(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(line[3])
            label = convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class BinaryAnli(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train-binary.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "valid-binary.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, records, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, record) in enumerate(records):

            guid = "%s-%s" % (set_type, i)

            beginning = record['InputSentence1']
            ending = record['InputSentence5']
            middle = record['RandomMiddleSentenceQuiz1']
            label = str(record['AnswerRightEnding'])

            text_a = convert_to_unicode(beginning)
            text_b = convert_to_unicode(middle + " " + ending)

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

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
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

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
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [printable_text(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def convert_examples_to_features_mc(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in tqdm(enumerate(examples), desc="Converting examples"):
        inputs = []

        tokens_a = [tokenizer.tokenize(t) for t in example.text_a]
        inputs.append(tokens_a)

        tokens_b = None
        if example.text_b:
            tokens_b = [tokenizer.tokenize(t) for t in example.text_b]
            inputs.append(tokens_b)

        tokens_c = None
        if example.text_c:
            tokens_c = [tokenizer.tokenize(t) for t in example.text_c]
            inputs.append(tokens_c)

        tokens_d = None
        if example.text_d:
            tokens_d = [tokenizer.tokenize(t) for t in example.text_d]
            inputs.append(tokens_d)

        if len(inputs) > 1:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            adjusted_len = max_seq_length - len(inputs) - 1
            _truncate_sequences(adjusted_len, inputs)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            for idx, ta in enumerate(tokens_a):
                tokens_a[idx] = tokens_a[idx][0:(max_seq_length - 2)]

        all_tokens = []
        all_token_ids = []
        all_segments = []
        all_masks = []
        for zipped_tokens in zip(*inputs):
            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)

            for idx, field in enumerate(zipped_tokens):
                for token in field:
                    tokens.append(token)
                    segment_ids.append(idx)
                tokens.append("[SEP]")
                segment_ids.append(idx)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            if len(input_ids) != max_seq_length:
                print("FOUND")
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            all_tokens.append(tokens)
            all_token_ids.append(input_ids)
            all_segments.append(segment_ids)
            all_masks.append(input_mask)

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("\n\n")
            logger.info("*** Example {} ***\n".format(ex_index))
            logger.info("guid: %s" % (example.guid))
            _ts = all_tokens
            _ids = all_token_ids
            _masks = all_masks
            _segs = all_segments

            logger.info("\n")

            for idx, (_t, _id, _mask, _seg) in enumerate(zip(_ts, _ids, _masks, _segs)):
                logger.info("\tOption {}".format(idx))
                logger.info("\ttokens: %s" % " ".join(
                    [printable_text(x) for x in _t]))
                logger.info("\tinput_ids: %s" % " ".join([str(x) for x in _id]))
                logger.info("\tinput_mask: %s" % " ".join([str(x) for x in _mask]))
                logger.info(
                    "\tsegment_ids: %s" % " ".join([str(x) for x in _seg]))

            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=all_token_ids,
                          input_mask=all_masks,
                          segment_ids=all_segments,
                          label_id=label_id))
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


def _truncate_sequences(max_length, inputs):
    idx = 0
    for ta, tb in zip(inputs[0], inputs[1]):
        _truncate_seq_pair(ta, tb, max_length)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer,
                                                                  named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer,
                                                                  named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(
                    param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        default=False,
                        action='store_true',
                        help="Whether to run prediction on a given dataset.")
    parser.add_argument("--input_file_for_pred",
                        default=None,
                        type=str,
                        help="File to run prediction on.")
    parser.add_argument("--output_file_for_pred",
                        default=None,
                        type=str,
                        help="File to output predictions into.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    args = parser.parse_args()

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "anli": AnliProcessor,
        "anli3": AnliProcessor3Option,
        'anli_csk': AnliWithCSKProcessor,
        'bin_anli': BinaryAnli
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu,
                bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(
                train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    if task_name == 'bin_anli':
        model = BertForSequenceClassification.from_pretrained(args.bert_model, len(label_list))
    else:
        model = BertForMultipleChoice.from_pretrained(args.bert_model,
                                                      len(label_list),
                                                      len(label_list)
                                                      )
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                           for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                           for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    global_step = 0

    model_save_path = os.path.join(args.output_dir, "bert-finetuned.model")
    tr_loss = None
    if args.do_train:
        if task_name.lower().startswith("anli"):
            train_features = convert_examples_to_features_mc(
                train_examples, label_list, args.max_seq_length, tokenizer)
        else:
            train_features = convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            status_tqdm = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(status_tqdm):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss, _ = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer,
                                                           model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1
                status_tqdm.set_description_str("Iteration / Training Loss: {}".format((tr_loss /
                                                                                        nb_tr_examples)))

        torch.save(model, model_save_path)

    if args.do_eval:
        if args.do_predict and args.input_file_for_pred is not None:
            eval_examples = processor.get_examples_from_file(args.input_file_for_pred)
        else:
            eval_examples = processor.get_dev_examples(args.data_dir)
        if task_name.lower().startswith("anli"):
            eval_features = convert_examples_to_features_mc(
                eval_examples, label_list, args.max_seq_length, tokenizer)
        else:
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        logger.info("***** Loading model from: {} *****".format(model_save_path))
        model = torch.load(model_save_path)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        eval_predictions = []
        eval_pred_probs = []

        logger.info("***** Predicting ... *****".format(model_save_path))

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_predictions.extend(np.argmax(logits, axis=1).tolist())

            eval_pred_probs.extend([_compute_softmax(list(l)) for l in logits])

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': tr_loss / nb_tr_steps if tr_loss is not None else 0.0
                  }

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        pred_examples = read_jsonl_lines(args.input_file_for_pred)

        logger.info("***** Eval predictions *****")
        for record, pred, probs in zip(pred_examples, eval_predictions, eval_pred_probs):
            record['bert_prediction'] = pred
            record['bert_correct'] = pred == (int(record['AnswerRightEnding']) - 1)
            record['bert_pred_probs'] = probs

        write_items([json.dumps(r) for r in pred_examples], args.output_file_for_pred)


if __name__ == "__main__":
    main()
