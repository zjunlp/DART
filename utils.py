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

import copy
import json
import pickle
import random
import statistics
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import logging
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, GPT2Tokenizer

logger = logging.getLogger('utils')


class InputExample(object):
    """A raw input example consisting of one or two segments of text and a label"""

    def __init__(self, guid, text_a, text_b=None, label=None, logits=None, meta: Optional[Dict] = None, idx=-1):
        """
        Create a new InputExample.
        :param guid: a unique textual identifier
        :param text_a: the sequence of text
        :param text_b: an optional, second sequence of text
        :param label: an optional label
        :param logits: an optional list of per-class logits
        :param meta: an optional dictionary to store arbitrary meta information
        :param idx: an optional numeric index
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.logits = logits
        self.idx = idx
        self.meta = meta if meta else {}

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def load_examples(path: str) -> List['InputExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)


class InputFeatures(object):
    """A set of numeric features obtained from an :class:`InputExample`"""

    def __init__(self, input_ids, attention_mask, token_type_ids, label, mlm_labels=None, logits=None,
                 meta: Optional[Dict] = None, idx=-1, block_flag=None):
        """
        Create new InputFeatures.
        :param input_ids: the input ids corresponding to the original text or text sequence
        :param attention_mask: an attention mask, with 0 = no attention, 1 = attention
        :param token_type_ids: segment ids as used by BERT
        :param label: the label
        :param mlm_labels: an optional sequence of labels used for auxiliary language modeling
        :param logits: an optional sequence of per-class logits
        :param meta: an optional dictionary to store arbitrary meta information
        :param idx: an optional numeric index
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.mlm_labels = mlm_labels
        self.logits = logits
        self.idx = idx
        self.block_flag = block_flag
        self.meta = meta if meta else {}

    def __repr__(self):
        return str(self.to_json_string())

    def pretty_print(self, tokenizer):
        return f'input_ids         = {tokenizer.convert_ids_to_tokens(self.input_ids)}\n' + \
               f'attention_mask    = {self.attention_mask}\n' + \
               f'token_type_ids    = {self.token_type_ids}\n' + \
               f'mlm_labels        = {self.mlm_labels}\n' + \
               f'logits            = {self.logits}\n' + \
               f'label             = {self.label}\n' + \
               f'block_flag        = {self.block_flag}'

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        tensors.values()

        assert all(next(iter(tensors.values())).size(0) == tensor.size(0)
                   for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)


def set_seed(seed: int):
    """ Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_logits(path: str, logits: np.ndarray):
    """Save an array of logits to a file"""
    with open(path, 'w') as fh:
        for example_logits in logits:
            fh.write(' '.join(str(logit) for logit in example_logits) + '\n')


def save_predictions(path: str, wrapper, results: Dict):
    """Save a sequence of predictions to a file"""
    predictions_with_idx = []

    if wrapper.task_helper and wrapper.task_helper.output:
        predictions_with_idx = wrapper.task_helper.output
    else:
        inv_label_map = {idx: label for label,
                         idx in wrapper.label_map.items()}
        for idx, prediction_idx in zip(results['indices'], results['predictions']):
            prediction = inv_label_map[prediction_idx]
            idx = idx.tolist() if isinstance(idx, np.ndarray) else int(idx)
            predictions_with_idx.append({'idx': idx, 'label': prediction})

    with open(path, 'w', encoding='utf8') as fh:
        for line in predictions_with_idx:
            fh.write(json.dumps(line) + '\n')


def write_results(path: str, dev_results: Dict, eval_results: Dict):
    with open(path, 'w') as fh:
        results = eval_results
        logger.info("eval_results:")
        fh.write("eval_results:" + '\n')

        for metric in results.keys():
            for pattern_id, values in results[metric].items():
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                result_str = "{}-p{}: {:.4f} +- {:.4f}, max = {:.4f}".format(
                    metric, pattern_id, mean * 100, stdev * 100, max(values) * 100)
                logger.info(result_str)
                fh.write(result_str + '\n')

        for metric in results.keys():
            eval_results = [result for pattern_results in results[metric].values()
                            for result in pattern_results]
            all_mean = statistics.mean(eval_results)
            all_stdev = statistics.stdev(
                eval_results) if len(eval_results) > 1 else 0
            result_str = "{}-all-p: {:.4f} +- {:.4f}, max = {:.4f}".format(
                metric, all_mean * 100, all_stdev * 100, max(eval_results) * 100)
            logger.info(result_str)
            fh.write(result_str + '\n')

        logger.info("dev_results:")
        fh.write("dev_results:" + '\n')

        for metric in dev_results.keys():
            for pattern_id, values in dev_results[metric].items():
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                result_str = "{}-p{}: {} +- {}".format(
                    metric, pattern_id, mean, stdev)
                logger.info(result_str)
                fh.write(result_str + '\n')

        for metric in dev_results.keys():
            eval_results = [result for pattern_results in dev_results[metric].values(
            ) for result in pattern_results]
            all_mean = statistics.mean(eval_results)
            all_stdev = statistics.stdev(
                eval_results) if len(eval_results) > 1 else 0
            result_str = "{}-all-p: {} +- {}".format(
                metric, all_mean, all_stdev)
            logger.info(result_str)
            fh.write(result_str + '\n')


def get_verbalization_ids(word: str, tokenizer: PreTrainedTokenizer, force_single_token: bool) -> Union[int, List[int]]:
    """
    Get the token ids corresponding to a verbalization
    :param word: the verbalization
    :param tokenizer: the tokenizer to use
    :param force_single_token: whether it should be enforced that the verbalization corresponds to a single token.
           If set to true, this method returns a single int instead of a list and throws an error if the word
           corresponds to multiple tokens.
    :return: either the list of token ids or the single token id corresponding to this word
    """
    kwargs = {'add_prefix_space': True} if isinstance(
        tokenizer, GPT2Tokenizer) else {}
    ids = tokenizer.encode(word, add_special_tokens=False, **kwargs)
    if not force_single_token:
        return ids
    assert len(ids) == 1, \
        f'Verbalization "{word}" does not correspond to a single token, got {tokenizer.convert_ids_to_tokens(ids)}'
    verbalization_id = ids[0]
    assert verbalization_id not in tokenizer.all_special_ids, \
        f'Verbalization {word} is mapped to a special token {tokenizer.convert_ids_to_tokens(verbalization_id)}'
    return verbalization_id


def trim_input_ids(input_ids: torch.tensor, pad_token_id, mask_token_id, num_masks: int):
    """
    Trim a sequence of input ids by removing all padding tokens and keeping at most a specific number of mask tokens.
    :param input_ids: the sequence of input token ids
    :param pad_token_id: the id of the pad token
    :param mask_token_id: the id of the mask tokens
    :param num_masks: the number of masks to keeps
    :return: the trimmed sequence of input ids
    """
    assert input_ids.shape[0] == 1
    input_ids_without_pad = [x for x in input_ids[0] if x != pad_token_id]

    trimmed_input_ids = []
    mask_count = 0
    for input_id in input_ids_without_pad:
        if input_id == mask_token_id:
            if mask_count >= num_masks:
                continue
            mask_count += 1
        trimmed_input_ids.append(input_id)

    return torch.tensor([trimmed_input_ids], dtype=torch.long, device=input_ids.device)
