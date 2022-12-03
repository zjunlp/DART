import yaml
import logging
from attrdict import AttrDict
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, accuracy_score, f1_score


def load_config(path, **kwargs):
    config = yaml.full_load(open(path, 'r'))
    config.update(kwargs)
    return AttrDict(config)


def get_logger(name, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():  # Prevent attaching multiple handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        if log_file:
            # New log will be appended, not overwrite
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def get_optimizer_scheduler(config, model, encoder=None):
    # Prepare optimizer and scheduler (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    groups = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if encoder is not None:
        groups.append({'params': [p for p in encoder.parameters()]})
    optimizer = AdamW(groups, lr=config.learning_rate, eps=config.adam_epsilon)

    # Compute warmup steps
    total_steps = config.max_train_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    return optimizer, scheduler


def compute_metrics(labels, predictions):
    return {
        'precision': precision_score(labels, predictions),
        'accuracy': accuracy_score(labels, predictions),
        'f1_score': f1_score(labels, predictions)
    }
