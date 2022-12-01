import random
import torch
from torch import nn
from transformers import GPT2Tokenizer


class PET:
    """Wraps basic prompt methods."""

    def __init__(self, tokenizer, reader, model, device) -> None:
        self.tokenizer = tokenizer
        self.reader = reader
        self.model = model
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()

        label_ids = []
        tokenize_kwargs = {}
        if isinstance(tokenizer, GPT2Tokenizer):
            tokenize_kwargs['add_prefix_space'] = True
        for label in reader.VERBALIZERS:
            label_id = tokenizer.encode(
                label, add_special_tokens=False, **tokenize_kwargs)[0]  # Force using one token
            label_ids.append(label_id)
        self.label_ids = torch.tensor(label_ids, device=device).long()

    def forward_step(self, batch, logits_key='pet_logits'):
        # Perform PET forward on MLM model and store output back
        batch[logits_key] = self.model(input_ids=batch['input_ids'],
                                       attention_mask=batch['attention_mask'],
                                       token_type_ids=batch['token_type_ids'])[0]

    def get_loss(self, batch, full_vocab=False, logits_key='pet_logits'):
        # Compute Cross-Entropy loss for prompt verbalizers
        assert logits_key in batch, 'logits should be pre-computed and stored in batch dict'
        masked_logits = batch[logits_key][batch['pet_flags'] == -1]
        labels = batch['pet_labels']
        if not full_vocab:
            masked_logits = masked_logits[:, self.label_ids]
            labels = batch['label_ids']
        return self.loss_fn(masked_logits, labels)

    def get_predictions(self, batch, logits_key='pet_logits'):
        # Get predicted labels
        full_logits = batch[logits_key]
        masked_logits = full_logits[batch['pet_flags'] == -1]
        masked_logits = masked_logits[:, self.label_ids]
        return masked_logits.argmax(-1).detach().cpu()


class DiffPET(PET):
    """Wraps differentiable prompts."""

    def __init__(self, tokenizer, reader, model, device):
        super().__init__(tokenizer, reader, model, device)
        self.pattern_map = []
        self.label_map = []

        kwargs = {}
        if isinstance(tokenizer, GPT2Tokenizer):
            kwargs['add_prefix_space'] = True

        # Initialize pattern & verbalizer mapping
        curr_idx = tokenizer.vocab_size - 1
        for part in reader.PATTERN:
            if part[0] != '[':
                token_ids = tokenizer.encode(part,
                                             add_special_tokens=False,
                                             **kwargs)
                for i in token_ids:
                    self.pattern_map.append([i, curr_idx])
                    curr_idx -= 1
        for label in reader.VERBALIZERS:
            label_id = tokenizer.encode(
                label, add_special_tokens=False, **kwargs)[0]  # Force using one token
            self.label_map.append([label_id, curr_idx])
            curr_idx -= 1

        # Target token ids
        self.pattern_ids = torch.tensor([p[1] for p in self.pattern_map],
                                        device=device).long()
        self.label_ids = torch.tensor([p[1] for p in self.label_map],
                                      device=device).long()
        self._init_embedding()

    def _init_embedding(self, copy=True):
        # Get word embedding from huggingface transformer model
        w = self.model.get_input_embeddings().weight.data
        if copy:
            for old, new in self.pattern_map + self.label_map:
                w[new] = w[old]
        else:
            for _, new in self.pattern_map + self.label_map:
                max_val = w[new].abs().max()
                w[new].uniform_(-max_val, max_val)

    def _prepare_input(self, batch):
        # Replace original token ids
        ids, flags = batch['input_ids'], batch['pet_flags']
        batch_size = len(ids)
        ids[flags == 2] = self.pattern_ids.repeat(batch_size)
        batch['input_ids'] = ids
        batch['pet_labels'] = self.label_ids[batch['label_ids']]

    def forward_step(self, batch):
        self._prepare_input(batch)
        super().forward_step(batch)


class MLM:
    """Auxiliary MLM object with label conditioning."""

    def __init__(self, tokenizer, reader, model, mask_rate=0.1) -> None:
        self.tokenizer = tokenizer
        self.reader = reader
        self.model = model
        self.mask_rate = mask_rate
        self.loss_fn = nn.BCELoss()

        kwargs = {}
        if isinstance(tokenizer, GPT2Tokenizer):
            kwargs['add_prefix_space'] = True

        # Initialize verbalize ids
        label_ids = []
        for label in reader.VERBALIZERS:
            # Force using one token
            label_ids.append(tokenizer.encode(
                label, add_special_tokens=False, **kwargs)[0])
        self.label_ids = label_ids

    def prepare_input(self, batch):
        ids, flags = batch['input_ids'].clone(), batch['pet_flags']
        batch_size, sequence_length = ids.shape

        # Set random pet labels
        pet_labels = batch['pet_labels']
        rand_labels = torch.tensor(random.choices(
            self.label_ids, k=batch_size), device=ids.device).long()
        ids[flags == -1] = rand_labels

        # Set random masks
        mask_pos = (torch.rand_like(ids.float(), device=ids.device)
                    < self.mask_rate)
        mask_pos.masked_fill_(flags != 0, 0)  # Ignore unmaskable
        mask_labels = ids[mask_pos == 1]
        ids.masked_fill_(mask_pos, self.tokenizer.mask_token_id)
        conditions = (pet_labels == rand_labels).view(-1,
                                                      1).repeat(1, sequence_length)
        conditions = conditions[mask_pos == 1].float()  # for BCE loss
        batch['mlm_input_ids'] = ids
        batch['mlm_mask_pos'] = mask_pos
        batch['mlm_conditions'] = conditions
        batch['mlm_labels'] = mask_labels

    def forward_step(self, batch, logits_key='mlm_logits'):
        # Perform MLM forward and store output back
        batch[logits_key] = self.model(input_ids=batch['mlm_input_ids'],
                                       attention_mask=batch['attention_mask'],
                                       token_type_ids=batch['token_type_ids'])[0]

    def get_loss(self, batch, logits_key='mlm_logits'):
        # Get BCE loss
        assert logits_key in batch, 'logits should be pre-computed and stored in batch dict'
        full_logits = batch[logits_key]
        masked_logits = full_logits[batch['mlm_mask_pos'] == 1]
        # Use softmax here to involve full vocabulary
        masked_logits = nn.Softmax(dim=1)(masked_logits)
        # Select target token probability
        masked_logits = masked_logits.gather(
            1, batch['mlm_labels'].view(-1, 1))
        conditions = batch['mlm_conditions']
        return self.loss_fn(masked_logits.view(-1), conditions)


class PETEncoder(nn.Module):
    """Baseline method (P-Tuning) using external prompt embeddings."""

    def __init__(self, num_tokens, hidden_size, encoder_type, device):
        self.num_tokens = num_tokens
        self.encoder_type = encoder_type
        self.device = device
        self.embedding = nn.Embedding(num_tokens, hidden_size).to(device)
        self.encoder = lambda x: x  # Do nothing
        if encoder_type == 'mlp':
            self.encoder = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                         nn.ReLU(),
                                         nn.Linear(hidden_size, hidden_size)).to(device)
        elif encoder_type == 'lstm':
            self.encoder = nn.Sequential(nn.LSTM(input_size=hidden_size,
                                                 hidden_size=hidden_size,
                                                 num_layers=2,
                                                 bidirectional=True,
                                                 batch_first=True),
                                         nn.Linear(2 * hidden_size,
                                                   hidden_size),
                                         nn.ReLU(),
                                         nn.Linear(hidden_size, hidden_size)).to(device)

    def init_embedding(self, new_weights):
        self.embedding.weight.data = new_weights

    def forward(self):
        idx = torch.tensor(list(range(self.num_tokens)),
                           device=self.device).long()
        emb = self.embedding(idx).unsqueeze(0)
        return self.encoder(emb)


def get_pet_mappers(tokenizer, reader, model, device, pet_method, mask_rate):
    mlm = MLM(tokenizer, reader, model, mask_rate) if mask_rate > 0.0 else None
    if pet_method == 'pet':
        return PET(tokenizer, reader, model, device), mlm
    if pet_method == 'diffpet':
        return DiffPET(tokenizer, reader, model, device), mlm
    raise NotImplementedError('Unsupported pet method')
