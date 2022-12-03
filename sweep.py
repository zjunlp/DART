import os
import wandb
from attrdict import AttrDict
from argparse import ArgumentParser

from run import train
from src.data import get_data_path


def train_wrapper():
    search = wandb.init(project="DART", sync_tensorboard=True)
    train(base_config, **wandb.config)
    search.finish()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--project_name', type=str,
                        default='DART', help='project name for sweep')
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--data_split', type=int, default=13,
                        choices=[13, 21, 42, 87, 100], help='few-shot split-id for GLUE dataset')
    parser.add_argument('--pretrain_model', type=str,
                        default='pretrain/albert-xxlarge-v2', help='name or path for pretrained model')
    parser.add_argument('--pet_method', type=str, default='diffpet',
                        choices=['pet', 'diffpet'], help='prompt encoding method')
    parser.add_argument('--random_seed', type=int,
                        default=3407, help='random seed for training')
    parser.add_argument('--max_run', type=int, default=100,
                        help='maximum tries for sweep')
    args = parser.parse_args()

    # Configure basic parameters for run
    task_name = args.task_name.lower()
    output_dir = os.path.join('output', task_name)
    os.makedirs(output_dir, exist_ok=True)
    train_path, dev_path, test_path = get_data_path(task_name, args.data_split)
    base_config = AttrDict({
        'task_name': task_name, 'train_path': train_path, 'dev_path': dev_path, 'test_path': test_path, 'output_dir': output_dir,
        'log_file': None, 'pred_file': '', 'use_gpu': True,
        'pretrain_model': args.pretrain_model, 'pet_method': args.pet_method, 'seed': args.random_seed, 'max_seq_len': 128,
        'shuffle': True, 'eval_every_steps': 20, 'test_batch_size': 32, 'max_train_epochs': 20, 'early_stop_steps': 5,
        'save_metric': 'f1_score' if task_name in ['mrpc', 'qqp'] else 'accuracy'
    })

    # Prepare sweep config (search space of hyper parameters) and get sweep id
    sweep_config = {
        'program': task_name,
        'method': 'grid',
        'metric': {
            'goal': 'maximize',
            'name': 'test f1_score' if task_name in ['mrpc', 'qqp'] else 'test accuracy'
        },
        'parameters': {
            'warmup_ratio': {'values': [0.05]},
            'learning_rate': {'values': [1e-5, 5e-5, 1e-4, 1e-3]},
            'weight_decay': {'values': [0.0, 0.01, 0.05]},
            'adam_epsilon': {'values': [1e-8]},
            'train_batch_size': {'values': [4, 8, 16, 32]},
            'grad_acc_steps': {'values': [1, 2, 4]},
            'max_grad_norm': {'values': [1.0]},
            'full_vocab_loss': {'values': [True, False]},
            'mask_rate': {'values': [0.0, 0.01, 0.05, 0.10]},
            'mlm_loss_weight': {'values': [0.0, 0.5, 1.0]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=args.project_name)

    # Sweep all hyper parameters
    wandb.agent(sweep_id, function=train_wrapper, count=args.max_run)
