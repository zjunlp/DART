import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from sklearn.metrics import precision_score
from argparse import ArgumentParser
from transformers import AutoModelForMaskedLM, AutoTokenizer, set_seed

from src.utils import load_config, get_logger, get_optimizer_scheduler, compute_metrics
from src.data import get_data_reader, get_data_loader
from src.model import get_pet_mappers


def evaluate(model, pet, config, dataloader):
    all_labels, all_preds = [], []

    model.eval()
    test_loss = 0.
    for batch in tqdm(dataloader, desc=f'[test]'):
        with torch.no_grad():
            pet.forward_step(batch)
            loss = pet.get_loss(batch, config.full_vocab_loss)
            test_loss += loss.item()
        all_preds.append(pet.get_predictions(batch))
        all_labels.append(batch["label_ids"])
    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    logger.info(f'test loss: {test_loss:.2f}')

    return all_preds, compute_metrics(all_labels, all_preds)


def main(config):
    logger.info(config)
    set_seed(config.seed)
    device = torch.device('cuda:0' if config.use_gpu else 'cpu')
    assert config.do_train or config.do_test, f'At least one of do_train or do_test should be set.'

    if config.do_train:
        logger.info(f' * * * * * Training * * * * *')
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(config.pretrain_model)
        model = AutoModelForMaskedLM.from_pretrained(config.pretrain_model)
        model.to(device)

        # Load data
        reader = get_data_reader(config.task_name)
        train_loader = get_data_loader(reader, config.train_path, 'train',
                                       tokenizer, config.max_seq_len, config.train_batch_size, config.shuffle)
        dev_loader = get_data_loader(reader, config.dev_path, 'dev',
                                     tokenizer, config.max_seq_len, config.test_batch_size)

        # Training with early stop
        pet, mlm = get_pet_mappers(tokenizer, reader, model, device,
                                   config.pet_method, config.mask_rate)

        writer = SummaryWriter(config.output_dir)
        global_step, best_score, early_stop_count = 0, -1., 0
        config.max_train_steps = len(train_loader) * config.max_train_epochs
        optimizer, scheduler = get_optimizer_scheduler(config, model)

        for epoch in range(1, config.max_train_epochs + 1):
            model.train()
            model.zero_grad()
            finish_flag = False
            iterator = tqdm(enumerate(train_loader),
                            desc=f'[train epoch {epoch}]', total=len(train_loader))

            for step, batch in iterator:
                global_step += 1
                # Whether do update (related with gradient accumulation)
                do_update = global_step % config.grad_acc_steps == 0 or step == len(
                    train_loader) - 1

                # Train step
                pet.forward_step(batch)
                pet_loss = pet.get_loss(batch, config.full_vocab_loss)
                writer.add_scalar('train pet loss',
                                  pet_loss.item(), global_step)
                pet_loss = pet_loss / config.grad_acc_steps
                if mlm is not None:
                    mlm.prepare_input(batch)
                    mlm.forward_step(batch)
                    mlm_loss = mlm.get_loss(batch)
                    writer.add_scalar('train mlm loss',
                                      mlm_loss.item(), global_step)
                    pet_loss += mlm_loss / config.grad_acc_steps

                # Update progress bar
                preds = pet.get_predictions(batch)
                precision = precision_score(batch['label_ids'], preds)
                iterator.set_description(
                    f'[train] loss:{pet_loss:.3f}, precision:{precision:.2f}')

                # Backward & optimize step
                pet_loss.backward()
                if do_update:
                    clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                # Evaluation process
                if global_step % config.eval_every_steps == 0:
                    _, scores = evaluate(model, pet, config, dev_loader)
                    logger.info(scores)
                    for metric, score in scores.items():
                        writer.add_scalar(f'dev {metric}', score, global_step)
                    assert config.save_metric in scores, f'Invalid metric name {config.save_metric}'

                    curr_score = scores[config.save_metric]
                    # Save predictions & models
                    if curr_score > best_score:
                        best_score = curr_score
                        early_stop_count = 0
                        logger.info(f'Save model at {config.output_dir}')
                        model.save(config.output_dir)
                    else:
                        early_stop_count += 1

                # Early stop / end training
                if config.early_stop_steps > 0 and early_stop_count >= config.early_stop_steps:
                    finish_flag = True
                    logger.info(f'Early stop at step {global_step}')
                    break

            logger.info(f'Total epochs: {epoch}')
            # Stop training
            if finish_flag:
                break

    if config.do_test:
        logger.info(f' * * * * * Testing * * * * *')
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(config.output_dir)
        model = AutoModelForMaskedLM.from_pretrained(config.output_dir)
        model.to(device)

        # Load data
        reader = get_data_reader(config.task_name)
        test_loader = get_data_loader(reader, config.test_path, 'test',
                                      tokenizer, config.max_seq_len, config.test_batch_size)
        pet, _ = get_pet_mappers(tokenizer, reader, model, device,
                                 config.pet_method, config.mask_rate)

        preds, scores = evaluate(model, pet, config, test_loader)

        # Save predictions
        logger.info(scores)
        if config.pred_file:
            logger.info(f'Saved predictions at {config.pred_file}')
            np.savetxt(os.path.join(config.output_dir,
                                    config.pred_file), preds, fmt='%.3e')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str,
                        default='config/sst2.yml')
    cfg = load_config(parser.parse_args().config)
    os.makedirs(cfg.output_dir, exist_ok=True)
    logger = get_logger('train', os.path.join(cfg.output_dir, 'train.log'))

    main(cfg)
