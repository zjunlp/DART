import os
import logging
import wandb
from argparse import ArgumentParser

from cli import parser, process_args
from utils import set_seed
from model_old import TransformerModelWrapper
from config import load_pet_configs
from data_utils import TRAIN_SET, DEV_SET, DEV32_SET, TEST_SET, load_examples, load_metrics

logger = logging.getLogger('sweep')


def main():
    # Initialize wandb
    run = wandb.init(reinit=True, sync_tensorboard=True)

    config = wandb.config
    task, seed, encoder_type = config['task'], config['seed_split'], config['encoder_type']
    lr, wd, bs = config['learning_rate'], config['weight_decay'], config['batch_size']

    ########################################
    # Prepare full arguments
    data_split = '16-%d' % seed
    if task == 'MNLI-mm':
        data_dir = os.path.join('data', 'k-shot', 'MNLI', data_split)
    elif task == 'RTE-glue':
        data_dir = os.path.join('data', 'k-shot', 'RTE', data_split)
    else:
        data_dir = os.path.join('data', 'k-shot', task, data_split)
    task_dir = os.path.join('output', task, 'tune', encoder_type)
    output_dir = os.path.join(task_dir, data_split)

    arguments = ['--embed_size', '1024',
                 '--do_train', '--do_eval',
                 '--eval_set', 'test',
                 '--task_name', task.lower(),
                 '--data_dir', data_dir,
                 '--model_name_or_path', 'roberta-large',
                 '--cache_dir', 'pretrain/roberta-large',
                 '--output_dir', output_dir,
                 '--learning_rate', str(lr),
                 '--weight_decay', str(wd),
                 '--prompt_encoder_type', encoder_type]

    if task in ['MNLI', 'MNLI-mm', 'SNLI', 'RTE-glue']:
        arguments.extend(['--max_seq_length', '256',
                          '--per_gpu_train_batch_size', str(bs),
                          '--gradient_accumulation_steps', '2'])
    else:
        arguments.extend(['--max_seq_length', '128',
                          '--per_gpu_train_batch_size', str(bs),
                          '--gradient_accumulation_steps', '1'])

    args = parser.parse_args(arguments)
    process_args(args)
    logger.info(args)

    ########################################
    # Load dataset
    train_data = load_examples(args.task_name, args.data_dir, TRAIN_SET,
                               num_examples=args.train_examples, split_examples_evenly=args.split_examples_evenly)
    eval_data = load_examples(args.task_name, args.data_dir, TEST_SET if args.eval_set == 'test' else DEV_SET,
                              num_examples=args.eval_examples, split_examples_evenly=args.split_examples_evenly)
    dev_data = load_examples(args.task_name, args.data_dir, DEV32_SET,
                             num_examples=args.dev_examples, split_examples_evenly=args.split_examples_evenly)

    ########################################
    # Training process
    set_seed(args.seed)

    # Load model
    model_config, train_config, eval_config = load_pet_configs(args)
    model = TransformerModelWrapper(model_config)

    # Train model
    model.train(train_data=train_data,
                dev_data=dev_data,
                eval_data=eval_data,
                output_dir=args.output_dir,
                eval_config=eval_config,
                per_gpu_train_batch_size=train_config.per_gpu_train_batch_size,
                n_gpu=train_config.n_gpu,
                num_train_epochs=train_config.num_train_epochs,
                gradient_accumulation_steps=train_config.gradient_accumulation_steps,
                weight_decay=args.weight_decay,
                learning_rate=args.learning_rate,
                wandb_log=False)

    run.finish()


if __name__ == '__main__':
    run_parser = ArgumentParser()
    run_parser.add_argument("--task",
                            type=str,
                            choices=['SST-2', 'sst-5', 'mr', 'cr', 'mpqa', 'subj', 'trec', 'CoLA',
                                     'MNLI', 'MNLI-mm', 'SNLI', 'QNLI', 'RTE-glue', 'MRPC', 'QQP'])
    run_parser.add_argument("--encoder",
                            type=str,
                            default='inner',
                            choices=['none', 'mlp', 'lstm', 'inner', 'inner2'])
    run_parser.add_argument("--seed_split",
                            type=int,
                            default=[],
                            nargs='+',
                            choices=[13, 21, 42, 87, 100])
    run_parser.add_argument("--batch_size",
                            type=int,
                            default=[],
                            nargs='+',
                            choices=[4, 8, 16, 24, 32])
    run_parser.add_argument("--sweep_id",
                            type=str,
                            default='')
    run_args = run_parser.parse_args()

    if not run_args.seed_split:  # Default search all seed splits
        run_args.seed_split = [13, 21, 42, 87, 100]

    if not run_args.batch_size:  # Default search all batch sizes
        if run_args.task in ['MNLI', 'MNLI-mm', 'SNLI', 'RTE-glue']:
            # Restrict maximum batch size due to memory limit
            run_args.batch_size = [4, 8, 16]
        else:
            run_args.batch_size = [4, 8, 16, 24, 32]

    # Prepare sweep config and get sweep id
    sweep_config = {
        'program': run_args.task,
        'method': 'grid',
        'metric': {
            'goal': 'maximize',
            'name': 'eval-' + load_metrics(run_args.task)[-1]
        },
        'parameters': {
            'task': {'value': run_args.task},
            'encoder_type': {'value': run_args.encoder},
            'seed_split': {'values': run_args.seed_split},
            'learning_rate': {'values': [1e-5, 5e-5, 1e-4, 2e-4]},
            'weight_decay': {'values': [0.0, 0.01, 0.05, 0.10]},
            'batch_size': {'values': run_args.batch_size}
        }
    }

    if run_args.sweep_id:  # Recover from old sweep
        sweep_id = run_args.sweep_id
    else:  # Create new sweep
        sweep_id = wandb.sweep(sweep_config, project="L-tune")

    # Sweep all hyper parameters
    wandb.agent(sweep_id, function=main)
