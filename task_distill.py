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
import csv
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

from transformer.modeling import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
import torch.nn.functional as F
import shutil


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

            logits, _, _, _ = model(input_ids, segment_ids, input_mask)

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

            logits, _, _, _ = model(input_ids, segment_ids, input_mask)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default="data/glue_data/MRPC",
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--teacher_model",
                        default="model/fine-tuned_pretrained_model/mrpc/on_original_data",
                        type=str,
                        help="The teacher model dir.")
    parser.add_argument("--student_model",
                        default="model/distilled_pretrained_model/2nd_General_TinyBERT_4L_312D",
                        type=str,
                        required=False,
                        help="The student model dir.")
    parser.add_argument("--task_name",
                        default="mrpc",
                        type=str,
                        required=False,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default="model/knowledge_review/distilled_intermediate_model/tmp",
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
        "wnli": WnliProcessor
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
        "wnli": "classification"
    }

    # intermediate distillation default parameters
    default_params = {
        "cola": {"num_train_epochs": 60, "max_seq_length": 64, "eval_step": 20, "num_train_epochs_distill_prediction": 40},
        "mnli": {"num_train_epochs": 8, "max_seq_length": 128, "eval_step": 500, "num_train_epochs_distill_prediction": 6},
        "mrpc": {"num_train_epochs": 30, "max_seq_length": 128, "eval_step": 20, "num_train_epochs_distill_prediction": 20},
        "wnli": {"num_train_epochs": 30, "max_seq_length": 128, "eval_step": 20, "num_train_epochs_distill_prediction": 15},
        "sst-2": {"num_train_epochs": 30, "max_seq_length": 64, "eval_step": 100, "num_train_epochs_distill_prediction": 20},
        "sts-b": {"num_train_epochs": 30, "max_seq_length": 128, "eval_step": 20, "num_train_epochs_distill_prediction": 15},
        "qqp": {"num_train_epochs": 8, "max_seq_length": 128, "eval_step": 500, "num_train_epochs_distill_prediction": 6},
        "qnli": {"num_train_epochs": 20, "max_seq_length": 128, "eval_step": 500, "num_train_epochs_distill_prediction": 10},
        "rte": {"num_train_epochs": 30, "max_seq_length": 128, "eval_step": 10, "num_train_epochs_distill_prediction": 15},
        "squad1": {"num_train_epochs": 6, "max_seq_length": 384,
                   "learning_rate": 3e-5, "eval_step": 500, "train_batch_size": 16, "num_train_epochs_distill_prediction": 3},
        "squad2": {"num_train_epochs": 6, "max_seq_length": 384,
                   "learning_rate": 3e-5, "eval_step": 500, "train_batch_size": 16, "num_train_epochs_distill_prediction": 3},
    }

    acc_tasks = ["mnli", "sst-2", "qnli", "rte", "wnli"]
    corr_tasks = ["sts-b"]
    mcc_tasks = ["cola"]
    f1_tasks = ["mrpc", "qqp"]

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

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

    if task_name in default_params:
        args.max_seq_len = default_params[task_name]["max_seq_length"]
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

    if not args.do_eval and not args.do_predict:
        if not args.aug_train:
            train_examples = processor.get_train_examples(args.data_dir)
        else:
            train_examples = processor.get_aug_examples(args.data_dir)
        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        train_features = convert_examples_to_features(train_examples, label_list,
                                                      args.max_seq_length, tokenizer, output_mode)
        train_data, _ = get_tensor_data(output_mode, train_features)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    if not args.do_predict:
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    if args.do_predict:
        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        test_data, test_labels = get_tensor_data(output_mode, test_features)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
            test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    if not args.do_eval and not args.do_predict:
        teacher_model = TinyBertForSequenceClassification.from_pretrained(
            args.teacher_model, num_labels=num_labels, is_student=False)
        teacher_model.to(device)

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
        inputs = tuple([torch.from_numpy(np.random.rand(args.train_batch_size,
                                                        args.max_seq_length)).type(torch.int64).to(device) for _ in range(3)])
        # writer.add_graph(teacher_model, inputs, use_strict_trace=False)
        writer.add_graph(student_model, inputs, use_strict_trace=False)

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

            student_model.train()
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                batch = tuple(t.to(device) for t in batch)

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

                is_student = False
                if not args.pred_distill:
                    is_student = True

                student_logits, student_atts, student_reps, student_att_probs = student_model(input_ids, segment_ids, input_mask,
                                                                                              is_student=is_student)
                with torch.no_grad():
                    teacher_logits, teacher_atts, teacher_reps, teacher_att_probs = teacher_model(
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

                    new_teacher_reps = [teacher_reps[i * layers_per_block]
                                        for i in range(student_layer_num + 1)]
                    new_student_reps = student_reps  # ？student的fit_dense为什么只有1个

                    # simple_fusion
                    # new_teacher_att_probs = [teacher_att_probs[i * layers_per_block + layers_per_block - 1]
                    #                          for i in range(student_layer_num)]
                    # tmp_loss = embedding_loss(
                    #     new_student_reps[0], new_teacher_reps[0])
                    # emb_loss += tmp_loss
                    # new_student_att_probs = student_att_probs
                    # teacher_fusion_reps_list = cal_fusion_reps(new_teacher_att_probs, new_teacher_reps[1:])

                    # student_fusion_reps_list, teacher_fusion_reps_list = cal_fusion_reps(
                    #     new_student_att_probs, new_student_reps[1:]), cal_fusion_reps(new_teacher_att_probs, new_teacher_reps[1:])
                    # for student_fusion_reps,teacher_fusion_reps in zip(student_fusion_reps_list,teacher_fusion_reps_list):
                    #     tmp_loss=loss_mse(student_fusion_reps,teacher_fusion_reps)
                    #     fusion_rep_loss +=tmp_loss
                    # loss= emb_loss+fusion_rep_loss

                    # resual kr with enhanced simple fusion
                    # tmp_loss = resual_kr_simple_fusion(
                    #     student_fusion_reps_list, teacher_fusion_reps_list)
                    # resual_kr_simple_fusion_loss += tmp_loss
                    # resual kr with enhanced simple fusion
                    # tmp_loss = resual_kr_enhanced_simple_fusion(
                    #     student_fusion_reps_list, teacher_fusion_reps_list)
                    # resual_kr_enhanced_simple_fusion_loss += tmp_loss

                    # loss = emb_loss+resual_kr_enhanced_simple_fusion_loss
                    # tr_emb_loss += emb_loss.item()
                    # tr_resual_kr_enhanced_simple_fusion_loss += resual_kr_enhanced_simple_fusion_loss.item()

                    # loss = emb_loss+resual_kr_simple_fusion_loss
                    # tr_emb_loss += emb_loss.item()
                    # tr_resual_kr_simple_fusion_loss += resual_kr_simple_fusion_loss.item()
                    # tr_fusion_rep_loss +=fusion_rep_loss.item()

                    # rep_loss=hcl(new_student_reps,new_teacher_reps)

                    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                        tmp_loss = loss_mse(student_rep, teacher_rep)
                        rep_loss += tmp_loss

                    # vanilla knowledge review
                    # att_loss=att_knowledge_review(student_atts,new_teacher_atts)
                    # rep_loss =rep_knowledge_review(new_student_reps,new_teacher_reps)

                    loss = rep_loss + att_loss
                    tr_att_loss += att_loss.item()
                    tr_rep_loss += rep_loss.item()
                else:
                    if output_mode == "classification":  # ！ 这里只是使用了soft label，没用ground truth
                        cls_loss = soft_cross_entropy(student_logits / args.temperature,
                                                      teacher_logits / args.temperature)
                    elif output_mode == "regression":
                        loss_mse = MSELoss()
                        # cls_loss = loss_mse(student_logits.view(-1), label_ids.view(-1))# ？这里有问题
                        cls_loss = loss_mse(
                            student_logits.view(-1), teacher_logits.view(-1))

                    loss = cls_loss
                    tr_cls_loss += cls_loss.item()

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += label_ids.size(0)
                nb_tr_steps += 1
                # print("##############################################")
                # print(optimizer.schedule.get_lr())
                # writer.add_scalar('{} lr'.format(task_name),
                #                   np.array(optimizer.get_lr()), global_step)
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
                    cls_loss = tr_cls_loss / (step + 1)
                    att_loss = tr_att_loss / (step + 1)
                    rep_loss = tr_rep_loss / (step + 1)
                    # vanilla knowledge review
                    # att_loss = tr_att_loss / (step + 1)
                    # rep_loss = tr_rep_loss / (step + 1)
                    # simple fusion
                    # emb_loss = tr_emb_loss /(step+1)
                    # fusion_rep_loss = tr_fusion_rep_loss/(step+1)
                    # resual kr with simple fusion
                    # emb_loss = tr_emb_loss / (step+1)
                    # resual_kr_simple_fusion_loss = tr_resual_kr_simple_fusion_loss / \
                    #     (step+1)
                    # resual kr with enhanced simple fusion
                    # emb_loss = tr_emb_loss / (step+1)
                    # resual_kr_enhanced_simple_fusion_loss = tr_resual_kr_enhanced_simple_fusion_loss / \
                    #     (step+1)

                    result = {}
                    if args.pred_distill:
                        result = do_eval(student_model, task_name, eval_dataloader,
                                         device, output_mode, eval_labels, num_labels)
                        writer.add_scalar('{} eval_loss'.format(
                            task_name), result['eval_loss'], global_step)
                        result['cls_loss'] = cls_loss
                        writer.add_scalar('{} cls_loss'.format(
                            task_name), cls_loss, global_step)

                    if not args.pred_distill:
                        result['att_loss'] = att_loss
                        result['rep_loss'] = rep_loss
                        # vanilla knowledge review
                        # result['att_loss'] = att_loss
                        # result['rep_loss'] = rep_loss
                        # simple fusion
                        # result['emb_loss'] = emb_loss
                        # result['fusion_rep_loss'] = fusion_rep_loss
                        # resual kr with simple fusion
                        # result['emb_loss'] = emb_loss
                        # result['resual_kr_simple_fusion_loss'] = resual_kr_simple_fusion_loss
                        # resual kr with enhanced simple fusion
                        # result['emb_loss'] = emb_loss
                        # result['resual_kr_enhanced_simple_fusion_loss'] = resual_kr_enhanced_simple_fusion_loss

                        writer.add_scalar('{} att_loss'.format(
                            task_name), att_loss, global_step)
                        writer.add_scalar('{} rep_loss'.format(
                            task_name), rep_loss, global_step)
                        # vanilla knowledge review
                        # writer.add_scalar('{} att_loss'.format(task_name),att_loss,global_step)
                        # writer.add_scalar('{} rep_loss'.format(task_name),rep_loss,global_step)
                        # simple fusion
                        # writer.add_scalar('{} emb_loss'.format(task_name),emb_loss,global_step)
                        # writer.add_scalar('{} fusion_rep_loss'.format(task_name),fusion_rep_loss,global_step)
                        # resual kr with simple fusion
                        # writer.add_scalar('{} emb_loss'.format(
                        #     task_name), emb_loss, global_step)
                        # writer.add_scalar('{} resual_kr_simple_fusion_loss'.format(
                        #     task_name), resual_kr_simple_fusion_loss, global_step)
                        # resual kr with enhanced simple fusion
                        # writer.add_scalar('{} emb_loss'.format(
                        #     task_name), emb_loss, global_step)
                        # writer.add_scalar('{} resual_kr_enhanced_simple_fusion_loss'.format(
                        #     task_name), resual_kr_enhanced_simple_fusion_loss, global_step)

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

            # model_to_save =student_model.module if hasattr(student_model,'module') else student_model
            # parameter_size = model_to_save.calc_sampled_param_num()

            output_str = "**************S*************\n" + \
                "task_name = {}\n".format(task_name) + \
                "best_metirc = %s\n" % best_dev_metric_str + \
                "**************E*************\n"

            logger.info(output_str)
            output_eval_file = os.path.join(
                args.output_dir, "final.results")
            with open(output_eval_file, "a+") as writer:
                writer.write(output_str + '\n')

                # if oncloud:
                #     logging.info(mox.file.list_directory(args.output_dir, recursive=True))
                #     logging.info(mox.file.list_directory('.', recursive=True))
                #     mox.file.copy_parallel(args.output_dir, args.data_url)
                #     mox.file.copy_parallel('.', args.data_url)


if __name__ == "__main__":
    logger.info("Task start! ")
    start0 = datetime.now()
    main()
    logger.info("Task finish! ")
    logger.info("Task cost {} minutes, i.e. {} hours. ".format((datetime.now(
    )-start0).total_seconds()/60, (datetime.now()-start0).total_seconds()/3600))
