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

import os
import json
from collections import defaultdict
from typing import List
import torch
import logging

from model import TransformerModelWrapper
from config import TrainConfig, EvalConfig, load_pet_configs
from data_utils import TRAIN_SET, DEV_SET, DEV32_SET, TEST_SET, load_examples, load_metrics
from utils import write_results, save_logits, save_predictions, set_seed, InputExample

logger = logging.getLogger('train')


def train_pet(args):
    # Load configs
    model_config, train_config, eval_config = load_pet_configs(args)

    # Load dataset
    train_data = load_examples(args.task_name, args.data_dir, TRAIN_SET,
                               num_examples=args.train_examples, split_examples_evenly=args.split_examples_evenly)
    eval_data = load_examples(args.task_name, args.data_dir, TEST_SET if args.eval_set == 'test' else DEV_SET,
                              num_examples=args.eval_examples, split_examples_evenly=args.split_examples_evenly)
    dev_data = load_examples(args.task_name, args.data_dir, DEV32_SET,
                             num_examples=args.dev_examples, split_examples_evenly=args.split_examples_evenly)

    set_seed(args.seed)

    # Record all evaluation results on dev & eval set
    dev_result_all = defaultdict(lambda: defaultdict(list))
    eval_result_all = defaultdict(lambda: defaultdict(list))
    # In 2 stage training, the 1st stage evaluations should also be recorded
    if args.do_train and args.do_eval and args.two_stage_train:
        dev_stage1_all = defaultdict(lambda: defaultdict(list))
        eval_stage1_all = defaultdict(lambda: defaultdict(list))

    # Iterates through all patterns
    for pattern_id in args.pattern_ids:
        # Repeat training
        for iteration in range(args.pet_repetitions):
            results_dict = {}
            model_config.pattern_id = pattern_id
            pattern_iter_output_dir = "{}/p{}-i{}".format(
                args.output_dir, pattern_id, iteration)

            results_file = os.path.join(
                pattern_iter_output_dir, 'results.json')
            if os.path.exists(results_file):
                logger.warning(
                    f"Path {results_file} already exists, skipping it...")
                # Load iteration results
                results_dict = json.load(open(results_file, 'r'))
                for metric, value in results_dict['dev_set'].items():
                    dev_result_all[metric][pattern_id].append(value)
                for metric, value in results_dict['eval_set'].items():
                    eval_result_all[metric][pattern_id].append(value)
                # Load stage1 results
                if args.do_train and args.do_eval and args.two_stage_train:
                    results_dict = json.load(
                        open(os.path.join(pattern_iter_output_dir, 'results_stage1.json'), 'r'))
                    for metric, value in results_dict['dev_set'].items():
                        dev_stage1_all[metric][pattern_id].append(value)
                    for metric, value in results_dict['eval_set'].items():
                        eval_stage1_all[metric][pattern_id].append(value)
                continue

            os.makedirs(pattern_iter_output_dir, exist_ok=True)

            # Init wrapper model
            assert model_config.pattern_id is not None, 'A pattern_id must be set for initializing a new PET model'
            wrapper = TransformerModelWrapper(model_config)

            #######################
            # from transformers import RobertaForMaskedLM
            # wrapper.model.model = RobertaForMaskedLM.from_pretrained(
            #     'output/sst-5-none/p1-i2/')
            # wrapper.model.model.cuda()

            # Training
            logger.info('--- Start iteration %d ---' % iteration)
            if args.do_train:
                if not args.two_stage_train:
                    # Single stage training
                    logger.info('=== Start training ===')
                    results_dict.update(train_single_model(train_data, eval_data, dev_data, pattern_iter_output_dir,
                                                           wrapper, train_config, eval_config,
                                                           extra_mask_rate=args.extra_mask_rate))
                    evaluate_single_model(pattern_id, pattern_iter_output_dir, eval_data,
                                          dev_data, eval_config, results_dict, dev_result_all, eval_result_all)
                    with open(os.path.join(pattern_iter_output_dir, 'results.json'), 'w') as fh:
                        json.dump(results_dict, fh)
                else:
                    # Two stage training
                    # 1. Only train prompts and label tokens
                    logger.info('=== Start training stage 1 ===')
                    results_dict.update(train_single_model(train_data, eval_data, dev_data, pattern_iter_output_dir,
                                                           wrapper, train_config, eval_config, stage=1,
                                                           extra_mask_rate=args.extra_mask_rate))
                    evaluate_single_model(pattern_id, pattern_iter_output_dir, eval_data,
                                          dev_data, eval_config, results_dict, dev_stage1_all, eval_stage1_all)
                    with open(os.path.join(pattern_iter_output_dir, 'results_stage1.json'), 'w') as fh:
                        json.dump(results_dict, fh)

                    # 2. Train full model
                    logger.info('=== Start training stage 2 ===')
                    results_dict.update(train_single_model(train_data, eval_data, dev_data, pattern_iter_output_dir,
                                                           wrapper, train_config, eval_config, stage=2,
                                                           extra_mask_rate=args.extra_mask_rate))
                    evaluate_single_model(pattern_id, pattern_iter_output_dir, eval_data,
                                          dev_data, eval_config, results_dict, dev_result_all, eval_result_all)
                    with open(os.path.join(pattern_iter_output_dir, 'results.json'), 'w') as fh:
                        json.dump(results_dict, fh)

                # Save configs
                train_config.save(os.path.join(
                    pattern_iter_output_dir, 'train_config.json'))
                eval_config.save(os.path.join(
                    pattern_iter_output_dir, 'eval_config.json'))
                logger.info("Saving complete")

            # Do evaluation only
            elif args.do_eval:
                evaluate_single_model(pattern_id, pattern_iter_output_dir, eval_data,
                                      dev_data, eval_config, results_dict, dev_result_all, eval_result_all)
                # Write overall results
                with open(os.path.join(pattern_iter_output_dir, 'results.json'), 'w') as fh:
                    json.dump(results_dict, fh)

            # Clear cache
            wrapper.model = None
            wrapper = None
            torch.cuda.empty_cache()

    # Calculate average results of current pattern
    if args.do_eval:
        logger.info("=== OVERALL RESULTS ===")
        if args.do_train and args.do_eval and args.two_stage_train:
            # Store stage 1 results
            logger.info("--- STAGE[1] RESULTS ---")
            write_results(os.path.join(
                args.output_dir, 'result_stage1.txt'), dev_stage1_all, eval_stage1_all)
            logger.info("--- STAGE[2] RESULTS ---")
        write_results(os.path.join(args.output_dir, 'result.txt'),
                      dev_result_all, eval_result_all)


def train_single_model(train_data: List[InputExample],
                       eval_data: List[InputExample],
                       dev_data: List[InputExample],
                       pattern_iter_output_dir: str,
                       model: TransformerModelWrapper,
                       config: TrainConfig,
                       eval_config: EvalConfig,
                       **kwargs):
    """
    Train a single model.
    :param model: the model to train
    :param train_data: the training examples to use
    :param config: the training config
    :param eval_config: the evaluation config
    :return: a dictionary containing the global step, average loss and (optionally) results on the train set
    """

    results_dict = {}

    # Evaluate train set
    metric_name = load_metrics(model.config.task_name)[0]
    # train_scores = model.eval(train_data, eval_config.per_gpu_eval_batch_size,
    #                           eval_config.n_gpu, eval_config.metrics)['scores']
    # results_dict['train_set_before_training'] = train_scores[metric_name]
    # logger.info("train_data performance before training: %s" %
    #             str(train_scores))

    if not train_data:
        logger.warning('Training method was called without training examples')
    else:
        # Learning rate for different stages
        if kwargs.get('stage', 0) == 1:
            lr = config.learning_rate_stage1
            max_steps = config.max_steps_stage1
        else:
            lr = config.learning_rate
            max_steps = config.max_steps
        # Perform training
        global_step, tr_loss = model.train(
            pattern_iter_output_dir=pattern_iter_output_dir,
            eval_config=eval_config,
            train_data=train_data,
            dev_data=dev_data,
            eval_data=eval_data,
            per_gpu_train_batch_size=config.per_gpu_train_batch_size,
            n_gpu=config.n_gpu,
            num_train_epochs=config.num_train_epochs,
            max_steps=max_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            weight_decay=config.weight_decay,
            learning_rate=lr,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            alpha=config.alpha,
            early_stop_epochs=config.early_stop_epochs,
            **kwargs
        )
        results_dict['global_step'] = global_step
        results_dict['average_loss'] = tr_loss

    # Load trained model and evaluate train set
    model = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)
    train_scores = model.eval(train_data, eval_config.per_gpu_eval_batch_size,
                              eval_config.n_gpu, eval_config.metrics)['scores']
    results_dict['train_set_after_training'] = train_scores[metric_name]
    logger.info("train_data performance after training: %s" %
                str(train_scores))

    return results_dict


def evaluate_single_model(pattern_id,
                          pattern_iter_output_dir,
                          eval_data,
                          dev_data,
                          eval_config,
                          results_dict,
                          dev_result_all,
                          eval_result_all,
                          do_save_logits=False,
                          do_save_predictions=False):
    wrapper = TransformerModelWrapper.from_pretrained(
        pattern_iter_output_dir)

    eval_result = wrapper.eval(
        eval_data, eval_config.per_gpu_eval_batch_size, eval_config.n_gpu, eval_config.metrics)
    dev_result = wrapper.eval(
        dev_data, eval_config.per_gpu_eval_batch_size, eval_config.n_gpu, eval_config.metrics)

    logger.info(
        "--- RESULT (pattern_id={}) ---".format(pattern_id))
    logger.info("eval results:")
    logger.info(eval_result['scores'])
    logger.info("dev results:")
    logger.info(dev_result['scores'])

    results_dict['eval_set'] = eval_result['scores']
    results_dict['dev_set'] = dev_result['scores']

    for metric, value in eval_result['scores'].items():
        eval_result_all[metric][pattern_id].append(value)

    for metric, value in dev_result['scores'].items():
        dev_result_all[metric][pattern_id].append(value)

    if do_save_logits:
        save_logits(os.path.join(pattern_iter_output_dir,
                                 'eval_logits.txt'), eval_result['logits'])

        save_logits(os.path.join(pattern_iter_output_dir,
                                 'dev_logits.txt'), dev_result['logits'])

    if do_save_predictions:
        save_predictions(os.path.join(
            pattern_iter_output_dir, 'eval_predictions.jsonl'), wrapper, eval_result)
        save_predictions(os.path.join(
            pattern_iter_output_dir, 'dev_predictions.jsonl'), wrapper, dev_result)
