import json
from abc import ABC
from typing import List, Tuple


class PetConfig(ABC):
    """Abstract class for a PET configuration that can be saved to and loaded from a json file."""

    def __repr__(self):
        return repr(self.__dict__)

    def save(self, path: str):
        """Save this config to a file."""
        with open(path, 'w', encoding='utf8') as fh:
            json.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path: str):
        """Load a config from a file."""
        cfg = cls.__new__(cls)
        with open(path, 'r', encoding='utf8') as fh:
            cfg.__dict__ = json.load(fh)
        return cfg


class TrainConfig(PetConfig):
    """Configuration for training a model."""

    def __init__(self,
                 device: str = None,
                 per_gpu_train_batch_size: int = 8,
                 n_gpu: int = 1,
                 num_train_epochs: int = 3,
                 max_steps: int = -1,
                 max_steps_stage1: int = -1,
                 gradient_accumulation_steps: int = 1,
                 weight_decay: float = 0.0,
                 learning_rate: float = 5e-5,
                 learning_rate_stage1: float = 1e-4,
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0,
                 max_grad_norm: float = 1,
                 alpha: float = 0.9999,
                 early_stop_epochs: int = 5,
                 two_stage_train: bool = False):
        self.device = device
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.n_gpu = n_gpu
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.max_steps_stage1 = max_steps_stage1
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.learning_rate_stage1 = learning_rate_stage1
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha
        self.early_stop_epochs = early_stop_epochs
        self.two_stage_train = two_stage_train


class EvalConfig(PetConfig):
    """Configuration for evaluating a model."""

    def __init__(self,
                 device: str = None,
                 n_gpu: int = 1,
                 per_gpu_eval_batch_size: int = 8,
                 metrics: List[str] = None):
        """
        Create a new evaluation config.

        :param device: the device to use ('cpu' or 'gpu')
        :param n_gpu: the number of gpus to use
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param metrics: the evaluation metrics to use (default: accuracy only)
        """
        self.device = device
        self.n_gpu = n_gpu
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        self.metrics = metrics


class WrapperConfig(object):
    """A configuration for a :class:`TransformerModelWrapper`."""

    def __init__(self,
                 model_type: str,
                 model_name_or_path: str,
                 task_name: str,
                 max_seq_length: int,
                 label_list: List[str],
                 device: str = "cuda",
                 pattern_id: int = 0,
                 cache_dir: str = None,
                 output_dir=None,
                 embed_size=-1,
                 prompt_encoder_type="lstm",
                 eval_every_step=20):
        self.device = device
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.label_list = label_list
        self.pattern_id = pattern_id
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.embed_size = embed_size
        self.prompt_encoder_type = prompt_encoder_type
        self.eval_every_step = eval_every_step


def load_pet_configs(args) -> Tuple[WrapperConfig, TrainConfig, EvalConfig]:
    """
    Load the model, training and evaluation configs for PET from the given command line arguments.
    """
    model_cfg = WrapperConfig(model_type=args.model_type,
                              model_name_or_path=args.model_name_or_path,
                              task_name=args.task_name,
                              label_list=args.label_list,
                              max_seq_length=args.pet_max_seq_length,
                              device=args.device,
                              cache_dir=args.cache_dir,
                              output_dir=args.output_dir,
                              embed_size=args.embed_size,
                              prompt_encoder_type=args.prompt_encoder_type,
                              eval_every_step=args.eval_every_step)

    train_cfg = TrainConfig(device=args.device,
                            per_gpu_train_batch_size=args.pet_per_gpu_train_batch_size,
                            n_gpu=args.n_gpu,
                            num_train_epochs=args.pet_num_train_epochs,
                            max_steps=args.pet_max_steps,
                            max_steps_stage1=args.pet_max_steps_stage1,
                            gradient_accumulation_steps=args.pet_gradient_accumulation_steps,
                            weight_decay=args.weight_decay,
                            learning_rate=args.learning_rate,
                            learning_rate_stage1=args.learning_rate_stage1,
                            adam_epsilon=args.adam_epsilon,
                            warmup_steps=args.warmup_steps,
                            max_grad_norm=args.max_grad_norm,
                            alpha=args.alpha,
                            early_stop_epochs=args.early_stop_epochs,
                            two_stage_train=args.two_stage_train)

    eval_cfg = EvalConfig(device=args.device,
                          n_gpu=args.n_gpu,
                          metrics=args.metrics,
                          per_gpu_eval_batch_size=args.pet_per_gpu_eval_batch_size)

    return model_cfg, train_cfg, eval_cfg
