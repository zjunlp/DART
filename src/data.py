import torch
import numpy as np
from transformers import GPT2Tokenizer
from torch.utils.data.dataloader import DataLoader


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None, logits=None, meta=None, idx=-1):
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


class Reader:
    PATTERN = []
    LABELS = []
    VERBALIZERS = []

    def load_samples(*args):
        raise NotImplementedError()


class SST2Reader(Reader):
    PATTERN = ["[text_a]", "It was", "[mask]", "."]
    LABELS = ["0", "1"]
    VERBALIZERS = ["terrible", "great"]

    @staticmethod
    def load_samples(path, split):
        examples = []
        with open(path, encoding="utf8") as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue
                line = line.rstrip().split("\t")
                guid = f"{split}-{i}"
                text_a = line[0]
                label = line[1]
                examples.append(InputExample(
                    guid=guid, text_a=text_a, label=label))

        return examples


class MRReader(SST2Reader):
    VERBALIZERS = ["bizzare", "memorable"]

    @staticmethod
    def load_samples(path, split):
        examples = []
        with open(path, encoding="utf8") as f:
            for i, line in enumerate(f.readlines()):
                line = line.rstrip()
                guid = f"{split}-{i}"
                text_a = line[2:]
                if not text_a.strip():  # Empty sentence
                    continue
                label = line[0]
                examples.append(InputExample(
                    guid=guid, text_a=text_a, label=label))

        return examples


class SST5Reader(MRReader):
    LABELS = ["0", "1", "2", "3", "4"]
    VERBALIZERS = ["terrible", "bad", "okay", "good", "great"]


class TrecReader(SST5Reader):
    PATTERN = ["[mask]", ":", "[text_a]"]
    LABELS = ["0", "1", "2", "3", "4", "5"]
    VERBALIZERS = ["Description", "Entity", "Expression",
                   "Human", "Location", "Number"]


class MNLIReader(Reader):
    PATTERN = ["[text_a]", "?", "[mask]", ",", "[text_b]"]
    LABELS = ["contradiction", "entailment", "neutral"]
    VERBALIZERS = ["No", "Yes", "Maybe"]
    TEXT_A_INDEX = 8
    TEXT_B_INDEX = 9
    LABEL_INDEX = -1

    @classmethod
    def load_samples(cls, path, split):
        examples = []
        with open(path, encoding="utf8") as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue
                line = line.rstrip().split("\t")
                guid = f"{split}-{line[0]}"
                text_a = line[cls.TEXT_A_INDEX]
                text_b = line[cls.TEXT_B_INDEX]
                label = line[cls.LABEL_INDEX]
                examples.append(InputExample(
                    guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class SNLIReader(MNLIReader):
    TEXT_A_INDEX = 7
    TEXT_B_INDEX = 8


class QNLIReader(MNLIReader):
    LABELS = ["not_entailment", "entailment"]
    VERBALIZERS = ["No", "Yes"]
    TEXT_A_INDEX = 1
    TEXT_B_INDEX = 2


class MRPCReader(SNLIReader):
    LABELS = ["0", "1"]
    VERBALIZERS = ["No", "Yes"]
    TEXT_A_INDEX = 3
    TEXT_B_INDEX = 4
    LABEL_INDEX = 0


class QQPReader(MRPCReader):
    TEXT_A_INDEX = 3
    TEXT_B_INDEX = 4
    LABEL_INDEX = 5


def get_data_reader(task):
    task = task.lower()
    if task in ['sst2']:
        return SST2Reader()
    if task in ['sst5']:
        return SST5Reader()
    if task in ['mr', 'cr', 'mpqa', 'subj']:
        return MRReader()
    if task in ['trec']:
        return TrecReader()
    if task in ['mnli']:
        return MNLIReader()
    if task in ['snli']:
        return SNLIReader()
    if task in ['qnli']:
        return QNLIReader()
    if task in ['mrpc']:
        return MRPCReader()
    if task in ['qqp']:
        return QQPReader()
    raise NotImplementedError(f'Unsupported task name: {task}')


def _encode(reader, sample, tokenizer, max_seq_len):
    kwargs = {'add_prefix_space': True} if isinstance(
        tokenizer, GPT2Tokenizer) else {}

    # Encode each part
    parts, n_special = [], 2
    for p in reader.PATTERN:
        if p == '[mask]':
            parts.append([tokenizer.mask_token_id])
        elif p == '[text_a]':
            parts.append(tokenizer.encode(
                sample.text_a, add_special_tokens=False, **kwargs))
        elif p == '[text_b]':
            n_special += 1
            parts.append(tokenizer.encode(
                sample.text_b, add_special_tokens=False, **kwargs))
        else:
            parts.append(tokenizer.encode(
                p, add_special_tokens=False, **kwargs))

    # Truncate
    while sum(len(x) for x in parts) > max_seq_len - n_special:
        longest = np.argmax([len(x) for x in parts])
        parts[longest].pop()

    # Concatenate
    len_seq1 = 0
    flags = [1]  # 0 for maskable; 1 for unmaskable; -1 for PET mask
    ids = [tokenizer.cls_token_id]
    for p, real_p in zip(reader.PATTERN, parts):
        if p == '[mask]':
            flags.append(-1)
            ids.append(tokenizer.mask_token_id)
        elif p == '[text_a]':
            flags.extend([0] * len(real_p))
            ids.extend(real_p)
            # End first sequence
            flags.append(1)
            ids.append(tokenizer.sep_token_id)
            len_seq1 = len(ids)
        elif p == '[text_b]':
            flags.extend([0] * len(real_p))
            ids.extend(real_p)
        else:
            flags.extend([1] * len(real_p))
            ids.extend(real_p)

    # Padding to max length
    ids.append(tokenizer.sep_token_id)
    len_seq2 = len(ids)
    ids.extend([tokenizer.pad_token_id] * (max_seq_len - len_seq2))
    flags.extend([1] * (max_seq_len - len(flags)))
    att_mask = [1] * len_seq2 + [0] * (max_seq_len - len_seq2)
    seg_ids = [0] * len_seq1 + [1] * \
        (len_seq2 - len_seq1) + [0] * (max_seq_len - len_seq2)

    # Get verbalized token id
    label_id = reader.LABELS.index(sample.label)
    verbalized_id = tokenizer.encode(
        reader.VERBALIZERS[label_id], add_special_tokens=False, **kwargs)[0]  # Force using one token

    return {'input_ids': ids, 'attention_mask': att_mask,
            'token_type_ids': seg_ids, 'label_ids': label_id,
            'pet_labels': verbalized_id, 'pet_flags': flags}


def get_data_loader(reader, path, split, tokenizer, max_seq_len, batch_size, shuffle=False):
    def collate_fn(samples):
        encoded_outputs = [
            _encode(reader, sample, tokenizer, max_seq_len) for sample in samples]
        merged_outputs = {}
        for k in encoded_outputs[0].keys():
            merged_outputs[k] = torch.LongTensor(
                [outputs[k] for outputs in encoded_outputs])
        return merged_outputs

    all_samples = reader.load_samples(path, split)
    return DataLoader(all_samples, batch_size, shuffle, collate_fn=collate_fn)
