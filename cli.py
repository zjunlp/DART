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

"""
This script can be used to train and evaluate either a regular supervised model or a PET/iPET model on
one of the supported tasks and datasets.
"""

import argparse
import os
import torch
import logging

from train import train_pet
from data_utils import PROCESSORS, load_metrics

logger = logging.getLogger('cli')
parser = argparse.ArgumentParser(
    description="Command line interface for P-Tuning.")

# Required parameters
parser.add_argument("--data_dir", default=None, type=str, required=True,
                    help="The input data dir. Should contain the data files for the task.")
parser.add_argument("--model_type", default="albert", type=str, required=True,
                    help="The type of the pretrained language model to use")
parser.add_argument("--model_name_or_path", default="albert-xxlarge-v2", type=str, required=True,
                    help="Path to the pre-trained model or shortcut name")
parser.add_argument("--cache_dir", default="", type=str,
                    help="Where to store the pre-trained models downloaded from S3.")
parser.add_argument("--task_name", default=None, type=str, required=True, choices=PROCESSORS.keys(),
                    help="The name of the task to train/evaluate on")
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model predictions and checkpoints will be written")
parser.add_argument('--overwrite_output_dir', action='store_true',
                    help="Overwrite the content of the output directory")

# PET-specific optional parameters
parser.add_argument("--pattern_ids", default=[1], type=int, nargs='+',
                    help="The ids of the PVPs to be used (only for PET)")
parser.add_argument("--alpha", default=0.9999, type=float,
                    help="Weighting term for the auxiliary language modeling task (only for PET)")
parser.add_argument("--pet_repetitions", default=3, type=int,
                    help="The number of times to repeat PET training and testing with different seeds.")
parser.add_argument("--pet_max_seq_length", default=256, type=int,
                    help="The maximum total input sequence length after tokenization for PET. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--pet_per_gpu_train_batch_size", default=4, type=int,
                    help="Batch size per GPU/CPU for PET training.")
parser.add_argument("--pet_per_gpu_eval_batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for PET evaluation.")
parser.add_argument('--pet_gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass in PET.")
parser.add_argument("--pet_num_train_epochs", default=3, type=float,
                    help="Total number of training epochs to perform in PET.")
parser.add_argument("--pet_max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform in PET. Override num_train_epochs.")
parser.add_argument("--pet_max_steps_stage1", default=250, type=int)

# Other optional parameters
parser.add_argument("--train_examples", default=-1, type=int,
                    help="The total number of train examples to use, where -1 equals all examples.")
parser.add_argument("--eval_examples", default=-1, type=int,
                    help="The total number of test examples to use, where -1 equals all examples.")
parser.add_argument("--dev_examples", default=-1, type=int,
                    help="The total number of dev examples to use, where -1 equals all examples.")
parser.add_argument("--split_examples_evenly", action='store_true',
                    help="If true, train examples are not chosen randomly, but split evenly across all labels.")
parser.add_argument("--learning_rate", default=1e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--learning_rate_stage1", default=1e-4, type=float,
                    help="The initial learning rate for Adam in stage 1.")
parser.add_argument("--weight_decay", default=0.1, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument("--early_stop_epochs", default=5, type=int,
                    help="Threshold epochs for early stop.")
parser.add_argument("--no_cuda", action='store_true',
                    help="Avoid using CUDA when available")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")
parser.add_argument('--do_train', action='store_true',
                    help="Whether to perform training")
parser.add_argument('--do_eval', action='store_true',
                    help="Whether to perform evaluation")
parser.add_argument("--eval_set", choices=['dev', 'test'], default='dev',
                    help="Whether to perform evaluation on the dev set or the test set")
parser.add_argument("--embed_size", default=128, type=int, help="")
parser.add_argument('--prompt_encoder_type', type=str,
                    default="lstm", choices=['lstm', 'mlp', 'none', 'inner'])
parser.add_argument("--eval_every_step", default=20, type=int, help="")

# Enhanced training
parser.add_argument("--two_stage_train", action='store_true', default=False,
                    help="Whether do two stage training")
parser.add_argument("--extra_mask_rate", type=float, default=0.0,
                    help="Whether do random additional masking.")


def process_args(args):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir))

    assert args.do_train or args.do_eval, "`do_train` and `do_eval` should be at least true for one"

    # Setup CUDA, GPU & distributed training
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()

    # Prepare task
    if args.task_name not in PROCESSORS:
        raise ValueError("Task '{}' not found".format(args.task_name))
    args.label_list = PROCESSORS[args.task_name]().get_labels()
    args.metrics = load_metrics(args.task_name)


if __name__ == "__main__":
    arguments = parser.parse_args()
    logger.info("Parameters: {}".format(arguments))
    process_args(arguments)
    train_pet(arguments)
