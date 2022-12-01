import wandb
from attrdict import AttrDict

from run import *


def train_test():
    search = wandb.init(reinit=True, sync_tensorboard=True)
    config = AttrDict(wandb.config)
    base_config = load_config(
        f'config/{config.task_name}-{config.task_split}.yml')
    os.makedirs(base_config.output_dir, exist_ok=True)
    train(base_config, **config)
    test(base_config, **config)
    search.finish()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task_name', type=str, default='sst2')
    parser.add_argument('--task_split', type=int,
                        choices=[13, 21, 42, 87, 100], default=13)
    parser.add_argument('--save_metric', type=str, default='test accuracy')
    args = parser.parse_args()

    # Prepare sweep config and get sweep id
    sweep_config = {
        'program': args.task_name,
        'method': 'grid',
        'metric': {
            'goal': 'maximize',
            'name': args.save_metric
        },
        'parameters': {
            'task_name': args.task_name,
            'task_split': args.task_split,
            'learning_rate': {'values': [1e-5, 5e-5, 1e-4, 2e-4]},
            'weight_decay': {'values': [0.0, 0.01, 0.05]},
            'train_batch_size': {'values': [4, 8, 16, 32]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="L-tune")

    # Sweep all hyper parameters
    wandb.agent(sweep_id, function=train_test)
