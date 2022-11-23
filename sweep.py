import wandb

from run import *


def train_test():
    # Initialize wandb
    search = wandb.init(reinit=True, sync_tensorboard=True)

    config = wandb.config
    task, seed, encoder_type = config['task_name'], config['seed_split'], config['encoder_type']
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

    arguments = [
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

    search.finish()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str,
                        default='config/sst2.yml', help='Base config for task parameters')
    args = parser.parse_args()
    config = load_config(parser.parse_args().config)
    os.makedirs(config.output_dir, exist_ok=True)
    logger = get_logger('train', os.path.join(config.output_dir, 'train.log'))

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
            'name': 'test ' + load_metrics(run_args.task)[-1]
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
    wandb.agent(sweep_id, function=train_test)
