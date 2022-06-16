# coding=utf-8
# 2019.12.2-Changed for TinyBERT task-specific distillation
#      Huawei Technologies Co., Ltd. <yinyichun@huawei.com>
# Copyright 2020 Huawei Technologies Co., Ltd.
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import re
import string
from collections import Counter
import csv
from ctypes import resize
from genericpath import samefile
import logging
import os
import random
import sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from transformer.modeling import TinyBertForSequenceClassification, BertForQuestionAnswering
from transformer.tokenization import BertTokenizer, whitespace_tokenize, BasicTokenizer
from transformer.optimization import BertAdam
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
import torch.nn.functional as F
import shutil
from losses import SupConLoss
import math

# csv.field_size_limit(sys.maxsize)
csv.field_size_limit(1000000000)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

oncloud = True
try:
    import moxing as mox
except:
    oncloud = False


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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

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
            text_a = line[3]
            text_b = line[4]
            if set_type == 'test':
                label = None
            else:
                label = line[0]
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

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            if set_type == 'test':
                label = None
            else:
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")),
            "test")


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

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        if set_type == 'test':
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a = line[1]
                label = None
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        else:
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                text_a = line[3]
                label = line[1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

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
            if set_type == 'test':
                text_a = line[1]
                label = None
            else:
                text_a = line[0]
                label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            if set_type == 'test':
                label = None
            else:
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                if set_type == 'test':
                    text_a = line[1]
                    text_b = line[2]
                    label = None
                else:
                    text_a = line[3]
                    text_b = line[4]
                    label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if set_type == 'test':
                text_a = line[1]
                text_b = line[2]
                label = None
            else:
                text_a = line[1]
                text_b = line[2]
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if set_type == 'test':
                text_a = line[1]
                text_b = line[2]
                label = None
            else:
                text_a = line[1]
                text_b = line[2]
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

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
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        try:
            if output_mode == "classification":
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)
        except:
            label_id = 0

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(example.label))
            logger.info("label_id: {}".format(label_id))
        if ex_index % 500 == 0:
            print("convert to features: %{}".format(
                100*((ex_index+1)*1.0/len(examples))))
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          seq_length=seq_length))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor(
        [f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids


def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def do_eval(model, task_name, eval_dataloader,
            device, output_mode, eval_labels, num_labels):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_

            logits, _, _, _, _, _ = model(input_ids, segment_ids, input_mask)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(
                logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, eval_labels.numpy())
    result['eval_loss'] = eval_loss

    return result


def do_predict(model, eval_dataloader, task_name,
               device, output_mode, processor, write_predict_dir):
    # eval_loss = 0
    # nb_eval_steps = 0
    preds = []

    for batch_ in tqdm(eval_dataloader, desc="Predicting"):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_

            logits, _, _, _, _, _ = model(input_ids, segment_ids, input_mask)

        # create eval loss and other metric required by the task
        # if output_mode == "classification":
        #     loss_fct = CrossEntropyLoss()
        #     tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        # elif output_mode == "regression":
        #     loss_fct = MSELoss()
        #     tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        # eval_loss += tmp_eval_loss.mean().item()
        # nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    # eval_loss = eval_loss / nb_eval_steps

    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    # labels=eval_labels.numpy()
    # assert len(preds) == len(labels)
    task_name_to_tsv_name = {
        "cola": "CoLA",
        "mnli": "MNLI-m",
        "mnli-mm": "MNLI-mm",
        "mrpc": "MRPC",
        "sst-2": "SST-2",
        "sts-b": "STS-B",
        "qqp": "QQP",
        "qnli": "QNLI",
        "rte": "RTE",
        "wnli": "WNLI"
    }

    output_predict_file = os.path.join(
        write_predict_dir, "{}.tsv".format(task_name_to_tsv_name[task_name]))
    # if os.path.exists(write_predict_dir):
    #     logger.info("Write predict directory ({}) already exists and is not empty.".format(
    #         write_predict_dir))
    #     shutil.rmtree(write_predict_dir)
    if not os.path.exists(write_predict_dir):
        os.makedirs(write_predict_dir)
    label_list = processor.get_labels()
    label_map = {i: label for i, label in enumerate(label_list)}
    num_actual_predict_examples = len(preds)
    with open(output_predict_file, "w") as writer:
        num_written_lines = 0
        logging.info("***** Predict results *****")
        writer.write("index\tprediction\n")
        for (i, pred) in enumerate(preds):
            # probabilities = prediction["probabilities"]
            if i >= num_actual_predict_examples:
                break
            # index = np.argmax(probabilities, axis=-1)
            if output_mode == "classification":
                prediction = label_map[pred]
            elif output_mode == "regression":
                prediction = pred  # 这里是否要保留3位小数?
            writer.write("%s\t%s\n" % (i, str(prediction)))
            num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples
    logging.info("***** Predict results done *****")
    # compress the *.tsv files to submission.zip
    # import zipfile
    # zip_file_path=os.path.join(write_predict_dir,'submission.zip')
    # zip_file=zipfile.ZipFile(zip_file_path,'w')
    # zip_file.write(write_predict_dir)
    # zip_file.close()


class SquadExample(object):
    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class QAInputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class Squad1Processor(DataProcessor):

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
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class Squad2Processor(DataProcessor):

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
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""  # 一个token可能同时属于doc_span，这时候取它属于有比较长的doc span那个
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + \
            0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def qa_convert_examples_to_features(examples, tokenizer, max_seq_length,
                                    doc_stride, max_query_length,
                                    is_training):
    """Loads a data file into a list of `InputBatch`s."""
    question_part_length_list = []
    passage_part_length_list = []

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        # 将原来的token使用tokenizer进行分词，然后对分词后的start_position和end_position进行调整
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(  # 对original answer进行分词，然后调整开始和结束的position
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            # length和doc_stride不一定哪个小，因为length收到max_tokens_for_doc的影响
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:  # [CLS]+question+[SEP]+context+[SEP]
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)
            # 记录question的长度
            question_part_length_list.append(len(tokens))

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(
                    tokens)] = tok_to_orig_index[split_token_index]  # 构成tokens里的context部分到origin context的index映射

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                # dev数据集eval时，如果为False，认为在当前slice里，该token没有成为start的资格
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            # 记录question + passage的长度
            passage_part_length_list.append(len(tokens))

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

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and  # 判断answer的token位置是否在当前span的范围内
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + \
                        doc_offset         # 这里的start_position是answer开始的token 在当前token 当前span下的结果
                    end_position = tok_end_position - doc_start + doc_offset
            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0
            if example_index < 1:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" %
                            " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    logger.info("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(
                        tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))
            if example_index % 5000 == 0:
                print("convert to features: %{}".format(
                    100*((example_index+1)*1.0/len(examples))))
            features.append(
                QAInputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    # 构成answer的token在tokens的index到原来context的index的映射
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,  # ？这个的作用
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible))
            unique_id += 1

    return features


def read_squad_examples(input_file, is_training, version_2_with_negative):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]  # 处理context
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(
                    len(doc_tokens) - 1)  # 得到每个字符对应第几个token

            for qa in paragraph["qas"]:  # 处理qas

                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        if 'is_impossible' not in qa:
                            qa['is_impossible'] = True
                        is_impossible = qa["is_impossible"]
                    # 2.0 is_impossible=True，表示问题不可回答，此时qa["answers"]=0
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        # 找到答案开始于第几个token
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset +
                                                           answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(
                            doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    logger.info('load {} examples!'.format(len(examples)))
    return input_data, examples


# SQuAD official evaluate scripts
def _f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


# SQuAD official evaluate scripts
def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


# SQuAD official evaluate scripts
def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)  # 返回预测结果和候选真正答案匹配度最高的那个


# SQuAD official evaluate scripts
def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]

                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    _f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'em': exact_match, 'f1': f1}


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(  # 降序排列
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

# SQuAD official evaluate scripts


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('em', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ('em', 100.0 *
             sum(exact_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('total', total),
        ])


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def get_raw_scores(dataset, preds):
    exact_scores = {}
    f1_scores = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid = qa['id']
                gold_answers = [a['text']
                                for a in qa['answers'] if normalize_answer(a['text'])]
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = ['']
                if qid not in preds:
                    print('Missing prediction for %s' % qid)
                    continue
                a_pred = preds[qid]
                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(a, a_pred)
                                        for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred)
                                     for a in gold_answers)
    return exact_scores, f1_scores


def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])

    return qid_to_has_ans


def evaluate_v2(dataset, predictions):
    na_probs = {k: 0.0 for k in predictions}

    qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]

    exact_raw, f1_raw = get_raw_scores(dataset, predictions)

    exact_thresh = apply_no_ans_threshold(
        exact_raw, na_probs, qid_to_has_ans, 1.0)
    f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans, 1.0)

    out_eval = make_eval_dict(exact_thresh, f1_thresh)

    return out_eval  

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold, dev_dataset, infer_times):
    """Write final predictions to the json file and log-odds of null if needed."""
    #     logger.info("Writing predictions to: %s" % (output_prediction_file))
    #     logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + \
                    result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    # squad2训练时空答案start=0,end=0,因此将position 0的start+end的logits作为空答案的目标
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    # start_index不能对应token_is_max_context==False
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:  # squad2有可能有没有答案的问题，这里存储没有答案的预测
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(  # 对于一个问题的所有可能答案，按照start_logit+end_logit降序排列
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(
                    pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(
                    orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(  # 校验预测的answer是否在origin context有相同的对应
                    tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it 此时len(nbest)会大于n_best_size
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0,
                             _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)  # 计算nbest中的每一个概率

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
            all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")
    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")
    # 做评估eval
    if version_2_with_negative:
        result = evaluate_v2(dev_dataset, all_predictions)
    else:
        result = evaluate(dev_dataset, all_predictions)

    result['infer_cnt'] = len(infer_times)
    result['infer_time'] = sum(infer_times) / len(infer_times)
    return result


def do_qa_eval(args, model, dataloader, features, examples, device,
               dev_dataset):
    all_results = []
    infer_times = []
    for _, batch_ in enumerate(dataloader):
        batch_ = tuple(t.to(device) for t in batch_)
        input_ids, input_mask, segment_ids, example_indices = batch_
        with torch.no_grad():
            start = datetime.now()
            batch_start_logits, batch_end_logits, _, _, _, _, _ = model(input_ids, segment_ids, input_mask)
            # (batch_start_logits, batch_end_logits) = model(input_ids, subbert_config,
            #                                                input_mask, segment_ids)
            infer_times.append((datetime.now() - start).microseconds / 1000)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(
                unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))

    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")

    return write_predictions(examples, features, all_results, args.n_best_size, args.max_answer_length,
                             args.do_lower_case, output_prediction_file,
                             output_nbest_file, output_null_log_odds_file,
                             args.verbose_logging, args.version_2_with_negative,
                             args.null_score_diff_threshold, dev_dataset, infer_times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default="data/squad_data",
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--teacher_model",
                        default="model/fine-tuned_pretrained_model/bert-base-uncased/squad1/on_original_data",
                        type=str,
                        help="The teacher model dir.")
    # parser.add_argument("--student_model",
    #                     default="model/distilled_pretrained_model/2nd_General_TinyBERT_4L_312D",
    #                     type=str,
    #                     required=False,
    #                     help="The student model dir.")
    parser.add_argument("--student_model",
                        default="model/distilled_pretrained_model/2nd_General_TinyBERT_4L_312D",
                        type=str,
                        required=False,
                        help="The student model dir.")
    parser.add_argument("--task_name",
                        default="squad1",
                        type=str,
                        required=False,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default="model/multi_level_distillation/distilled_intermediate_model/bert_mlkd_for_qa_task/test",
                        type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--tensorboard_log_save_dir",
                        default="tensorboard_log/tmp",
                        type=str,
                        required=False,
                        help="The output directory where the tensorboard logs will be written.")
    parser.add_argument("--write_predict_dir",
                        default="model/knowledge_review/distilled_intermediate_model/tmp",
                        type=str,
                        help="The output directory where the model predictions on the test set")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_eval",
                        # default=True,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        # default=True,
                        action='store_true',
                        help="Whether to run predict on the test set")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')
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
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    # added arguments
    parser.add_argument('--aug_train',
                        action='store_true')
    parser.add_argument('--eval_step',
                        type=int,
                        default=50)
    parser.add_argument('--pred_distill',
                        # default=True,
                        action='store_true')
    parser.add_argument('--data_url',
                        type=str,
                        default="")
    parser.add_argument('--temperature',
                        type=float,
                        default=1.)
    # SQuAD hyper-parameters
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument('--version_2_with_negative', default=0,
                        type=int)     # 0 means 1.1, 1 means 2.0
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument("--max_query_length",
                        default=64,
                        type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--train_file", default='train-v1.1.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default='dev-v1.1.json', type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--gpu_id", default=0, type=int)
    args = parser.parse_args()
    # logger.info('The args: {}'.format(args))

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor,
        "squad1": Squad1Processor,
        "squad2": Squad2Processor
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification",
        "wnli": "classification",
        "squad1": "qa_classification",
        "squad2": "qa_classification"
    }

    # intermediate distillation default parameters
    default_params = {
        "cola": {"num_train_epochs": 50, "max_seq_length": 64, "eval_step": 20, "num_train_epochs_distill_prediction": 30},
        "mnli": {"num_train_epochs": 6, "max_seq_length": 128, "eval_step": 500, "num_train_epochs_distill_prediction": 6},
        "mrpc": {"num_train_epochs": 20, "max_seq_length": 128, "eval_step": 20, "num_train_epochs_distill_prediction": 15},
        "wnli": {"num_train_epochs": 20, "max_seq_length": 128, "eval_step": 20, "num_train_epochs_distill_prediction": 15},
        "sst-2": {"num_train_epochs": 15, "max_seq_length": 64, "eval_step": 100, "num_train_epochs_distill_prediction": 10},
        "sts-b": {"num_train_epochs": 20, "max_seq_length": 128, "eval_step": 20, "num_train_epochs_distill_prediction": 15},
        "qqp": {"num_train_epochs": 6, "max_seq_length": 128, "eval_step": 500, "num_train_epochs_distill_prediction": 6},
        "qnli": {"num_train_epochs": 10, "max_seq_length": 128, "eval_step": 500, "num_train_epochs_distill_prediction": 10},
        "rte": {"num_train_epochs": 20, "max_seq_length": 128, "eval_step": 10, "num_train_epochs_distill_prediction": 15},
        "squad1": {"num_train_epochs": 4, "max_seq_length": 384,
                   "learning_rate": 3e-5, "eval_step": 500, "train_batch_size": 16, "num_train_epochs_distill_prediction": 3},
        "squad2": {"num_train_epochs": 4, "max_seq_length": 384,
                   "learning_rate": 3e-5, "eval_step": 500, "train_batch_size": 16, "num_train_epochs_distill_prediction": 3},
    }

    acc_tasks = ["mnli", "sst-2", "qnli", "rte", "wnli"]
    corr_tasks = ["sts-b"]
    mcc_tasks = ["cola"]
    f1_tasks = ["mrpc", "qqp"]
    qa_tasks = ["squad1", "squad2"]

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")
    # n_gpu = torch.cuda.device_count()
    torch.cuda.set_device(args.gpu_id)
    n_gpu=1

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # Prepare seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    # Prepare task settings
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        # raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        logger.info("Output directory ({}) already exists and is not empty.".format(
            args.output_dir))
        shutil.rmtree(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()
    if task_name == 'squad1':
        args.train_file = os.path.join(
            args.data_dir, 'train-v1.1.json')
        args.predict_file = os.path.join(
            args.data_dir, 'dev-v1.1.json')
        args.learning_rate = default_params[task_name]["learning_rate"]
        args.train_batch_size = default_params[task_name]["train_batch_size"]
        # args.eval_batch_size = default_params[task_name]["eval_batch_size"]
    elif task_name == 'squad2':
        args.train_file = os.path.join(
            args.data_dir, 'train-v2.0.json')
        args.predict_file = os.path.join(
            args.data_dir, 'dev-v2.0.json')
        args.learning_rate = default_params[task_name]["learning_rate"]
        args.train_batch_size = default_params[task_name]["train_batch_size"]
        # args.eval_batch_size = default_params[task_name]["eval_batch_size"]

    if task_name in default_params:
        args.max_seq_length = default_params[task_name]["max_seq_length"]
        args.eval_step = default_params[task_name]["eval_step"]

    if not args.pred_distill and not args.do_eval:
        if task_name in default_params:
            args.num_train_epochs = default_params[task_name]["num_train_epochs"]
    if args.pred_distill:
        if task_name in default_params:
            args.num_train_epochs = default_params[task_name]["num_train_epochs_distill_prediction"]

    logger.info('The args: {}'.format(args))

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(
        args.student_model, do_lower_case=args.do_lower_case)

    def get_dataloader(examples, label_list, max_seq_length, tokenizer, output_mode,
                       is_dev=False, is_training=False):
        if output_mode != 'qa_classification':
            features = convert_examples_to_features(examples, label_list, max_seq_length,
                                                    tokenizer, output_mode)
            data, labels = get_tensor_data(output_mode, features)
            if not is_dev:
                sampler = RandomSampler(data)
            else:
                sampler = SequentialSampler(data)
            return labels, DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)
        else:
            features = qa_convert_examples_to_features(  # 对squad的测试集的answer没有处理，返回的features里start_position和end_position都为空
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=is_training)

            all_input_ids = torch.tensor(
                [f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor(
                [f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor(
                [f.segment_ids for f in features], dtype=torch.long)
            if is_training:
                all_start_positions = torch.tensor(
                    [f.start_position for f in features], dtype=torch.long)
                all_end_positions = torch.tensor(
                    [f.end_position for f in features], dtype=torch.long)
                data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                     all_start_positions, all_end_positions)
            else:
                all_example_index = torch.arange(
                    all_input_ids.size(0), dtype=torch.long)
                data = TensorDataset(
                    all_input_ids, all_input_mask, all_segment_ids, all_example_index)

            if not is_dev:
                sampler = RandomSampler(data)
            else:
                sampler = SequentialSampler(data)
            return features, DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)

    if not args.do_eval and not args.do_predict:
        if output_mode == "qa_classification":
            train_data, train_examples = read_squad_examples(
                input_file=args.train_file, is_training=True,
                version_2_with_negative=args.version_2_with_negative
            )
        elif not args.aug_train:
            train_examples = processor.get_train_examples(args.data_dir)
        else:
            train_examples = processor.get_aug_examples(args.data_dir)
        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        labels, train_dataloader = get_dataloader(train_examples, label_list, args.max_seq_length,
                                                  tokenizer, output_mode, is_training=True)
        # train_features = convert_examples_to_features(train_examples, label_list,
        #                                               args.max_seq_length, tokenizer, output_mode)
        # train_data, _ = get_tensor_data(output_mode, train_features)
        # train_sampler = RandomSampler(train_data)
        # train_dataloader = DataLoader(
        #     train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    if not args.do_predict:
        if output_mode == "qa_classification":
            eval_dataset, eval_examples = read_squad_examples(input_file=args.predict_file, is_training=False,  # squad 的测试集处理没有处理answer
                                                              version_2_with_negative=args.version_2_with_negative)

            eval_features, eval_dataloader = get_dataloader(eval_examples, label_list, args.max_seq_length,
                                                            tokenizer, output_mode, is_dev=True, is_training=False)
        else:
            eval_examples = processor.get_dev_examples(args.data_dir)
            eval_labels, eval_dataloader = get_dataloader(eval_examples, label_list, args.max_seq_length,
                                                          tokenizer, output_mode, is_dev=True)
        # eval_features = convert_examples_to_features(
        #     eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        # eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
        # eval_sampler = SequentialSampler(eval_data)
        # eval_dataloader = DataLoader(
            # eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    if args.do_predict:
        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        test_data, test_labels = get_tensor_data(output_mode, test_features)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
            test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    if not args.do_eval and not args.do_predict:
        if output_mode == "qa_classification":
            teacher_model = BertForQuestionAnswering.from_pretrained(
                args.teacher_model, num_labels=num_labels, is_student=False
            )
        else:
            teacher_model = TinyBertForSequenceClassification.from_pretrained(
                args.teacher_model, num_labels=num_labels, is_student=False)
        teacher_model.to(device)
    if output_mode == "qa_classification":
        student_model = BertForQuestionAnswering.from_pretrained(
            args.student_model, num_labels=num_labels, is_student=True
        )
    else:
        student_model = TinyBertForSequenceClassification.from_pretrained(
            args.student_model, num_labels=num_labels, is_student=True)
    student_model.to(device)

    if not args.do_eval and not args.do_predict:
        tensorboard_log_save_dir = args.tensorboard_log_save_dir
        if os.path.exists(tensorboard_log_save_dir):
            logger.info("Tensorboard log directory ({}) already exists and is not empty.".format(
                args.output_dir))
            shutil.rmtree(tensorboard_log_save_dir)
        if not os.path.exists(tensorboard_log_save_dir):
            os.makedirs(tensorboard_log_save_dir)
        writer = SummaryWriter(log_dir=tensorboard_log_save_dir, flush_secs=30)
        # inputs = tuple([torch.from_numpy(np.random.rand(args.train_batch_size,
        #                                                 args.max_seq_length)).type(torch.int64).to(device) for _ in range(3)])
        # writer.add_graph(teacher_model, inputs, use_strict_trace=False)
        # writer.add_graph(student_model, inputs)

    if args.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        student_model.eval()
        result = do_eval(student_model, task_name, eval_dataloader,
                         device, output_mode, eval_labels, num_labels)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    elif args.do_predict:
        logger.info("***** Running prediction *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        student_model.eval()
        do_predict(student_model, test_dataloader, task_name,
                   device, output_mode, processor, args.write_predict_dir)
        if task_name == "mnli":
            task_name = "mnli-mm"
            processor = processors[task_name]()
            test_examples = processor.get_test_examples(args.data_dir)
            test_features = convert_examples_to_features(
                test_examples, label_list, args.max_seq_length, tokenizer, output_mode)
            test_data, test_labels = get_tensor_data(
                output_mode, test_features)

            logger.info("***** Running mm prediction *****")
            logger.info("  Num examples = %d",
                        len(test_examples))
            logger.info("  Batch size = %d",
                        args.eval_batch_size)
            test_sampler = SequentialSampler(test_data)
            test_dataloader = DataLoader(
                test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
            do_predict(student_model, test_dataloader, task_name,
                       device, output_mode, processor, args.write_predict_dir)
    else:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        if n_gpu > 1:
            student_model = torch.nn.DataParallel(student_model)
            teacher_model = torch.nn.DataParallel(teacher_model)
        # Prepare optimizer
        param_optimizer = list(student_model.named_parameters())
        size = 0
        for n, p in student_model.named_parameters():
            logger.info('n: {}'.format(n))
            size += p.nelement()

        logger.info('Total parameters: {}'.format(size))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        schedule = 'warmup_linear'
        if not args.pred_distill:
            schedule = 'none'
        optimizer = BertAdam(optimizer_grouped_parameters,
                             schedule=schedule,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
        # Prepare loss functions
        loss_mse = MSELoss()

        def soft_cross_entropy(predicts, targets):
            student_likelihood = torch.nn.functional.log_softmax(
                predicts, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets, dim=-1)
            return (- targets_prob * student_likelihood).mean()

        def embedding_loss(student_embedding, teacher_embedding):
            return loss_mse(student_embedding, teacher_embedding)

        def cal_fusion_reps(att_probs_list, hidden_states_list):
            fusion_reps_list = []
            for att_probs, hidden_states in zip(att_probs_list, hidden_states_list):
                fusion_reps_list.append(torch.matmul(
                    att_probs, hidden_states.unsqueeze(1)))
            return fusion_reps_list

        def resual_kr_enhanced_simple_fusion(student_fusion_reps_list, teacher_fusion_reps_list):
            total_resual_kr_enhanced_simple_fusion = 0.
            for student_fusion_rep, teacher_fusion_rep in zip(student_fusion_reps_list, teacher_fusion_reps_list):
                total_resual_kr_enhanced_simple_fusion += loss_mse(
                    student_fusion_rep, teacher_fusion_rep)
            return total_resual_kr_enhanced_simple_fusion

        def resual_kr_simple_fusion(student_fusion_reps_list, teacher_fusion_reps_list):
            alpha = 0.6
            total_resual_kr_simple_fusion_loss = 0.
            student_fusion_rep = student_fusion_reps_list[-1]
            for i in range(len(teacher_fusion_reps_list)-1, -1, -1):
                student_fusion_rep = alpha * \
                    student_fusion_reps_list[i]+(1-alpha)*student_fusion_rep
                total_resual_kr_simple_fusion_loss += loss_mse(
                    student_fusion_rep, teacher_fusion_reps_list[i])
            return total_resual_kr_simple_fusion_loss

        def rep_knowledge_review(student_reps_list, teacher_reps_list):
            total_rep_loss = 0.
            for i in range(len(teacher_reps_list)):
                for j in range(i, len(student_reps_list)):
                    total_rep_loss += loss_mse(
                        student_reps_list[j], teacher_reps_list[i])
            return total_rep_loss

        def att_knowledge_review(student_att_list, teacher_att_list):
            student_att_list = [torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(
                device), student_att) for student_att in student_att_list]  # 将被mask掉的位置置为0
            teacher_att_list = [torch.where(teacher_att <= -1e2, torch.zeros_like(
                teacher_att).to(device), teacher_att) for teacher_att in teacher_att_list]
            total_att_loss = 0.
            for i in range(len(teacher_att_list)):
                for j in range(i, len(student_att_list)):
                    total_att_loss += loss_mse(
                        student_att_list[j], teacher_att_list[i])
            return total_att_loss

        def hcl(fstudent, fteacher):
            loss_all = 0.0
            for fs, ft in zip(fstudent, fteacher):
                try:
                    n, c, h, w = fs.shape
                except:
                    fs = torch.unsqueeze(fs, 1)
                    ft = torch.unsqueeze(ft, 1)
                    n, c, h, w = fs.shape
                loss = F.mse_loss(fs, ft, reduction='mean')
                cnt = 1.0
                tot = 1.0
                for l in [2, 3, 4]:
                    # if l >= h:
                    #     continue
                    tmpfs = F.avg_pool2d(fs, (l, 1), stride=1)
                    tmpft = F.avg_pool2d(ft, (l, 1), stride=1)
                    # tmpfs = F.adaptive_avg_pool2d(fs, (l, hidden_size))
                    # tmpft = F.adaptive_avg_pool2d(ft, (l, hidden_size))
                    cnt /= 2.0
                    loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
                    tot += cnt
                loss = loss / tot
                loss_all = loss_all + loss
            return loss_all

        def pdist(e, squared=False, eps=1e-12):
            e_square = e.pow(2).sum(dim=-1)
            # e_square = e.pow(2).sum(dim=1)
            # prod = e @ e.t()
            prod = torch.matmul(e, e.transpose(-2, -1))
            # res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
            res = (e_square.unsqueeze(-1) +
                   e_square.unsqueeze(-2) - 2 * prod).clamp(min=eps)

            if not squared:
                res = res.sqrt()

            res = res.clone()
            # res[range(len(e)), range(len(e))] = 0
            res[..., range(e.shape[-2]), range(e.shape[-2])] = 0
            return res

        def rkd_kl_loss(student, teacher):
            loss = 0.
            for st, te in zip(student, teacher):
                with torch.no_grad():
                    t_d = torch.matmul(st, st.transpose(-2, -1))
                    t_d_log = F.softmax(t_d, dim=-1)
                s_d = torch.matmul(te, te.transpose(-2, -1))
                s_d_log = F.log_softmax(s_d, dim=-1)
                loss += F.kl_div(s_d_log, t_d_log, reduction='batchmean')
            return loss

        def rkd_loss(student, teacher):
            loss = 0.
            for st, te in zip(student, teacher):
                with torch.no_grad():
                    t_d = pdist(te, squared=False)
                    mean_td = t_d[t_d > 0].mean()
                    t_d = t_d / mean_td

                d = pdist(st, squared=False)
                mean_d = d[d > 0].mean()
                d = d / mean_d

                loss += F.smooth_l1_loss(d, t_d, reduction='mean')
            return loss

        def align_loss(student, teacher):
            loss = 0.
            for st, te in zip(student, teacher):
                loss += loss_mse(st, te)
            return loss

        def new_rkd_loss(student, teacher, head_nums=12):
            loss = 0.
            student_head_size = student[0].shape[-1]//head_nums
            teacher_head_size = teacher[0].shape[-1]//head_nums
            student = [st.view(
                st.shape[0], st.shape[1], head_nums, -1).permute(0, 2, 1, 3) for st in student]
            teacher = [te.view(
                te.shape[0], te.shape[1], head_nums, -1).permute(0, 2, 1, 3) for te in teacher]
            student_relation = [torch.div(torch.matmul(
                st, st.transpose(-2, -1)), math.sqrt(student_head_size)) for st in student]
            teacher_relation = [torch.div(torch.matmul(
                te, te.transpose(-2, -1)), math.sqrt(teacher_head_size)) for te in teacher]
            for st, te in zip(student_relation, teacher_relation):
                st_log_probs = F.log_softmax(
                    st, -1).view(-1, student_relation[0].shape[-1])
                te_probs = F.softmax(
                    te, -1).view(-1, teacher_relation[0].shape[-1])
                loss += F.kl_div(st_log_probs, te_probs, reduction='batchmean')
            return loss

        def new_rkd_batch_loss(student, teacher, head_nums=12):
            loss = 0.
            student_head_size = student[0].shape[-1]//head_nums
            teacher_head_size = teacher[0].shape[-1]//head_nums
            student = [st.view(st.shape[0], head_nums, -
                               1).permute(1, 0, 2) for st in student]
            teacher = [te.view(te.shape[0], head_nums, -
                               1).permute(1, 0, 2) for te in teacher]
            student_relation = [torch.div(torch.matmul(
                st, st.transpose(-2, -1)), math.sqrt(student_head_size)) for st in student]
            teacher_relation = [torch.div(torch.matmul(
                te, te.transpose(-2, -1)), math.sqrt(teacher_head_size)) for te in teacher]
            for st, te in zip(student_relation, teacher_relation):
                st_log_probs = F.log_softmax(
                    st, -1).view(-1, student_relation[0].shape[-1])
                te_probs = F.softmax(
                    te, -1).view(-1, teacher_relation[0].shape[-1])
                loss += F.kl_div(st_log_probs, te_probs, reduction='batchmean')
            return loss
        criterion_super_contr = SupConLoss()
        # Train and evaluate
        global_step = 0
        best_dev_metric = 0.0
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

        for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0.
            tr_att_loss = 0.
            tr_rep_loss = 0.
            tr_cls_loss = 0.
            tr_emb_loss = 0.
            tr_fusion_rep_loss = 0.
            tr_resual_kr_simple_fusion_loss = 0.
            tr_resual_kr_enhanced_simple_fusion_loss = 0.
            tr_rkd_att_loss = 0.
            tr_rkd_rep_loss = 0.
            tr_self_out_loss = 0.
            tr_rkd_emb_loss = 0.
            tr_rkd_rep_loss = 0.
            # tr_super_contr_loss = 0.

            student_model.train()
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                batch = tuple(t.to(device) for t in batch)
                if output_mode == "qa_classification":
                    input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                else:
                    input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch

                if input_ids.size()[0] != args.train_batch_size:
                    continue

                att_loss = 0.
                rep_loss = 0.
                cls_loss = 0.
                emb_loss = 0.
                fusion_rep_loss = 0.
                resual_kr_simple_fusion_loss = 0.
                resual_kr_enhanced_simple_fusion_loss = 0.
                rkd_att_loss = 0.
                rkd_rep_loss = 0.
                self_out_loss = 0.
                rkd_emb_loss = 0.
                rkd_rep_loss = 0.
                # super_contr_loss = 0.

                is_student = True
                # if not args.pred_distill:
                #     is_student = True
                if output_mode == "qa_classification":
                    student_start_logits, student_end_logits, student_atts, student_reps, student_att_probs, student_all_self_outs, original_student_reps, student_words_embeddings = student_model(input_ids, segment_ids, input_mask,
                                                                                                                                                                          start_positions=start_positions, end_positions=end_positions, is_student=is_student)
                else:
                    student_logits, student_atts, student_reps, student_att_probs, student_all_self_outs, original_student_reps, student_words_embeddings = student_model(input_ids, segment_ids, input_mask,
                                                                                                                                                                          is_student=is_student)
                with torch.no_grad():
                    if output_mode == "qa_classification":
                        teacher_start_logits, teacher_end_logits, teacher_atts, teacher_reps, teacher_att_probs, teacher_all_self_outs, teacher_words_embeddings = teacher_model(
                            input_ids, segment_ids, input_mask, start_positions, end_positions)
                    else:
                        teacher_logits, teacher_atts, teacher_reps, teacher_att_probs, teacher_all_self_outs, teacher_words_embeddings = teacher_model(
                            input_ids, segment_ids, input_mask)

                if not args.pred_distill:
                    teacher_layer_num = len(teacher_atts)
                    student_layer_num = len(student_atts)
                    assert teacher_layer_num % student_layer_num == 0
                    layers_per_block = int(
                        teacher_layer_num / student_layer_num)
                    new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                        for i in range(student_layer_num)]
                    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                                  student_att)  # 将被mask掉的位置置为0
                        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                                  teacher_att)

                        tmp_loss = loss_mse(student_att, teacher_att)
                        att_loss += tmp_loss
                    # new_teacher_self_outs = [teacher_all_self_outs[i * layers_per_block + layers_per_block - 1]
                    #                          for i in range(student_layer_num)]
                    # self_out_loss = new_rkd_loss(
                    #     student_all_self_outs, new_teacher_self_outs, head_nums=12)

                    new_teacher_reps = [teacher_reps[i * layers_per_block]
                                        for i in range(student_layer_num + 1)]
                    new_student_reps = student_reps  # ？student的fit_dense为什么只有1个
                    rep_loss = align_loss(new_student_reps, new_teacher_reps)

                    # rkd_emb_loss = new_rkd_loss(
                    #     (student_words_embeddings,), (teacher_words_embeddings,), head_nums=1)
                    loss = rep_loss + att_loss
                    # tr_rkd_emb_loss += rkd_emb_loss.item()
                    tr_rep_loss += rep_loss.item()
                    tr_att_loss += att_loss.item()
                else:
                    # student_rep=student_reps[-1][:,0,:]
                    # teacher_rep=teacher_reps[-1][:,0,:]
                    # batch_rkd_rep_loss=rkd_loss((student_rep,),(teacher_rep,))
                    # student_rep = student_reps[-1][:, 0, :]
                    # original_student_rep = original_student_reps[-1]
                    # teacher_rep = teacher_reps[-1]
                    # rkd_rep_loss = new_rkd_loss((original_student_rep,), (teacher_rep,), head_nums=1)
                    # super_contr_loss = criterion_super_contr(
                    #     student_rep, teacher_rep, labels=label_ids)
                    # batch_rkd_rep_loss=rkd_loss((student_rep,),(teacher_rep,))
                    # batch_rkd_rep_loss = new_rkd_batch_loss(
                    #     (original_student_rep,), (teacher_rep,), head_nums=1)
                    if output_mode == "classification":  # ！ 这里只是使用了soft label，没用ground truth
                        cls_loss = soft_cross_entropy(student_logits / args.temperature,
                                                      teacher_logits / args.temperature)
                    elif output_mode == "regression":
                        loss_mse = MSELoss()
                        # cls_loss = loss_mse(student_logits.view(-1), label_ids.view(-1))# ？这里有问题
                        cls_loss = loss_mse(
                            student_logits.view(-1), teacher_logits.view(-1))
                    else:
                        cls_loss_start_position = soft_cross_entropy(student_start_logits / args.temperature, teacher_start_logits / args.temperature)
                        cls_loss_end_position = soft_cross_entropy(student_end_logits / args.temperature, teacher_end_logits / args.temperature)
                        cls_loss = cls_loss_start_position + cls_loss_end_position
                        # loss = cls_loss + batch_rkd_rep_loss
                    loss = cls_loss
                    tr_cls_loss += cls_loss.item()
                    # tr_rkd_rep_loss += rkd_rep_loss.item()
                    # tr_super_contr_loss += super_contr_loss.item()

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if output_mode == "qa_classification":
                    nb_tr_examples += start_positions.size(0)
                else:
                    nb_tr_examples += label_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (global_step + 1) % args.eval_step == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info("  Epoch = {} iter {} step".format(
                        epoch_, global_step))
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    student_model.eval()

                    loss = tr_loss / (step + 1)
                    # rkd_emb_loss = tr_rkd_emb_loss / (step+1)
                    rep_loss = tr_rep_loss / (step + 1)
                    att_loss = tr_att_loss / (step+1)
                    cls_loss = tr_cls_loss / (step + 1)
                    # rkd_rep_loss = tr_rkd_rep_loss / (step+1)
                    # super_contr_loss = tr_super_contr_loss/(step+1)

                    result = {}
                    if args.pred_distill:
                        if output_mode == "qa_classification":
                            result = do_qa_eval(args, student_model, eval_dataloader, eval_features, eval_examples,
                                                device, eval_dataset)
                        else:
                            result = do_eval(student_model, task_name, eval_dataloader,
                                         device, output_mode, eval_labels, num_labels)
                        if output_mode != "qa_classification":
                            writer.add_scalar('{} eval_loss'.format(
                                task_name), result['eval_loss'], global_step)
                        result['cls_loss'] = cls_loss
                        # result['rkd_rep_loss'] = rkd_rep_loss
                        # result['super_contr_loss'] = super_contr_loss
                        writer.add_scalar('{} cls_loss'.format(
                            task_name), cls_loss, global_step)
                        # writer.add_scalar('{} rkd_rep_loss'.format(
                        #     task_name), rkd_rep_loss, global_step)
                        # writer.add_scalar('{} super_contr_loss'.format(
                        #     task_name), super_contr_loss, global_step)

                    if not args.pred_distill:
                        result['rep_loss'] = rep_loss
                        # result['rkd_emb_loss'] = rkd_emb_loss
                        result['att_loss'] = att_loss
                        writer.add_scalar('{} rep_loss'.format(
                            task_name), rep_loss, global_step)
                        # writer.add_scalar('{} rkd_emb_loss'.format(
                        #     task_name), rkd_emb_loss, global_step)
                        writer.add_scalar('{} att_loss'.format(
                            task_name), att_loss, global_step)

                    result['global_step'] = global_step
                    result['loss'] = loss
                    writer.add_scalar('{} train_loss'.format(
                        task_name), loss, global_step)
                    result_to_file(result, output_eval_file)

                    if not args.pred_distill:  # 中间层蒸馏每次都保存
                        save_model = True
                    else:
                        save_model = False  # 预测层蒸馏只保存最好结果

                        if task_name in acc_tasks:
                            writer.add_scalar('{} acc'.format(
                                task_name), result['acc'], global_step)
                            if result['acc'] > best_dev_metric:
                                best_dev_metric = result['acc']
                                best_dev_metric_str = str(best_dev_metric)
                                save_model = True

                        if task_name in corr_tasks:
                            writer.add_scalar('{} spearman corr'.format(
                                task_name), result['spearman'], global_step)
                            if result['spearman'] > best_dev_metric:
                                best_dev_metric = result['spearman']
                                best_dev_metric_str = str(best_dev_metric)
                                save_model = True

                        if task_name in mcc_tasks:
                            writer.add_scalar('{} mcc'.format(
                                task_name), result['mcc'], global_step)
                            if result['mcc'] > best_dev_metric:
                                best_dev_metric = result['mcc']
                                best_dev_metric_str = str(best_dev_metric)
                                save_model = True

                        if task_name in f1_tasks:
                            writer.add_scalar('{} f1'.format(
                                task_name), result['f1'], global_step)
                            if result['f1'] > best_dev_metric:
                                best_dev_metric = result['f1']
                                best_dev_metric_str = str(best_dev_metric)
                                save_model = True
                        
                        if task_name in qa_tasks:
                            writer.add_scalar('{} f1'.format(
                                task_name), result['f1'], global_step)
                            # writer.add_scalar('{} em'.format(
                            #     task_name), result['em'], global_step)
                            if result['f1']  > best_dev_metric:
                                best_dev_metric = result['f1'] 
                                best_dev_metric_str = 'f1: {}'.format(
                                    result['f1'])
                                save_model = True

                    if save_model:
                        logger.info("***** Save model *****")

                        model_to_save = student_model.module if hasattr(
                            student_model, 'module') else student_model

                        model_name = WEIGHTS_NAME
                        # if not args.pred_distill:
                        #     model_name = "step_{}_{}".format(global_step, WEIGHTS_NAME)
                        output_model_file = os.path.join(
                            args.output_dir, model_name)
                        output_config_file = os.path.join(
                            args.output_dir, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(),
                                   output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(args.output_dir)

                    student_model.train()

        if args.pred_distill:
            # Test mnli-mm
            if task_name == "mnli":
                task_name = "mnli-mm"
                processor = processors[task_name]()
                if not os.path.exists(args.output_dir + '-MM'):
                    os.makedirs(args.output_dir + '-MM')

                eval_examples = processor.get_dev_examples(
                    args.data_dir)

                eval_features = convert_examples_to_features(
                    eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
                eval_data, eval_labels = get_tensor_data(
                    output_mode, eval_features)

                logger.info("***** Running mm evaluation *****")
                logger.info("  Num examples = %d",
                            len(eval_examples))
                logger.info("  Batch size = %d",
                            args.eval_batch_size)

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                             batch_size=args.eval_batch_size)

                result = do_eval(student_model, task_name, eval_dataloader,
                                 device, output_mode, eval_labels, num_labels)

                result['global_step'] = global_step

                tmp_output_eval_file = os.path.join(
                    args.output_dir + '-MM', "final.results")
                result_to_file(result, tmp_output_eval_file)
                task_name = 'mnli'

            output_str = "**************S*************\n" + \
                "task_name = {}\n".format(task_name) + \
                "best_metirc = %s\n" % best_dev_metric_str + \
                "**************E*************\n"

            logger.info(output_str)
            output_eval_file = os.path.join(
                args.output_dir, "final.results")
            with open(output_eval_file, "a+") as writer:
                writer.write(output_str + '\n')


if __name__ == "__main__":
    logger.info("Task start! ")
    start0 = datetime.now()
    main()
    logger.info("Task finish! ")
    logger.info("Task cost {} minutes, i.e. {} hours. ".format((datetime.now(
    )-start0).total_seconds()/60, (datetime.now()-start0).total_seconds()/3600))
