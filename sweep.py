import wandb
from attrdict import AttrDict

from run import *
from src.data import get_data_path


def train_test():
    search = wandb.init(reinit=True, sync_tensorboard=True)
    hp = AttrDict(wandb.config)
    os.makedirs(base_config.output_dir, exist_ok=True)
    hp.train_path, hp.dev_path, hp.test_path = get_data_path(
        task_name, hp.data_split)
    train(base_config, **hp, use_tensorboard=True)
    search.finish()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task_name', type=str)
    args = parser.parse_args()

    task_name = args.task_name.lower()
    save_metric = 'test f1' if task_name in ['mrpc', 'qqp'] else 'test acc'
    base_config = load_config(f'config/sample.yml')

    # Prepare sweep config and get sweep id
    sweep_config = {
        'program': task_name,
        'method': 'grid',
        'metric': {
            'goal': 'maximize',
            'name': save_metric
        },
        'parameters': {
            'data_split': {'values': [13, 21, 42, 87, 100]},
            'learning_rate': {'values': [1e-5, 5e-5, 1e-4, 2e-4]},
            'weight_decay': {'values': [0.0, 0.01, 0.05]},
            'train_batch_size': {'values': [4, 8, 16, 32]},
            'grad_acc_steps': {'values': [1, 2, 4]},
            'mask_rate': {'values': [0.0, 0.01, 0.05, 0.10]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="DART")

    # Sweep all hyper parameters
    wandb.agent(sweep_id, function=train_test)
