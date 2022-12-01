# DART
Implementation for ICLR2022 paper *[Differentiable Prompt Makes Pre-trained Language Models Better Few-shot Learners](https://arxiv.org/pdf/2108.13161.pdf)*. 

## Environment
- python@3.6
- Use `pip install -r requirements.txt` to install dependencies.
- `wandb` account is required if the user wants to search for best hyper-parameter combinations.

## Data source
- 16-shot GLUE dataset from [LM-BFF](https://github.com/princeton-nlp/LM-BFF).
- Generated data consists of 5 random splits (13/21/42/87/100) for a task, each has 16 samples.
  - The generation process follows LM-BFF [here](https://github.com/princeton-nlp/LM-BFF/blob/main/tools/generate_k_shot_data.py).

## How to run
- To train / test with a config file containing specific parameters and data files, use `run.py --config config/[task_name]-[seed_split].yml`.
  - For details of parameters, please refer to `config/sample`.
  - Some configurations can be overriden with command line arguments:
```bash
python run.py -h
usage: run.py [-h] [--config CONFIG] [--train_path TRAIN_PATH]
              [--dev_path DEV_PATH] [--test_path TEST_PATH]
              [--pet_method PET_METHOD] [--seed SEED]
              [--train_batch_size TRAIN_BATCH_SIZE]
              [--warmup_ratio WARMUP_RATIO] [--learning_rate LEARNING_RATE]
              [--grad_acc_steps GRAD_ACC_STEPS]
              [--full_vocab_loss FULL_VOCAB_LOSS] [--mask_rate MASK_RATE]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Basic configurations with default parameters
  --train_path TRAIN_PATH
  --dev_path DEV_PATH
  --test_path TEST_PATH
  --pet_method PET_METHOD
  --seed SEED
  --train_batch_size TRAIN_BATCH_SIZE
  --warmup_ratio WARMUP_RATIO
  --learning_rate LEARNING_RATE
  --grad_acc_steps GRAD_ACC_STEPS
  --full_vocab_loss FULL_VOCAB_LOSS
  --mask_rate MASK_RATE
```
- To search optimal hyper-parameters for each task-split and reproduce our result, please use `sweep.py`:
  - Please refer to documentation for [WandB](https://docs.wandb.ai/) for more details.
  - **NOTE: we follow [LM-BFF](https://github.com/princeton-nlp/LM-BFF) in that we search optimal sets of hyper-parameters on different data splits respectively.**
```bash
$ python sweep.py -h
usage: sweep.py [-h] [--task_name TASK_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --task_name TASK_NAME
```
## How to Cite
```
@inproceedings{
zhang2022differentiable,
title={Differentiable Prompt Makes Pre-trained Language Models Better Few-shot Learners},
author={Ningyu Zhang and Luoqiu Li and Xiang Chen and Shumin Deng and Zhen Bi and Chuanqi Tan and Fei Huang and Huajun Chen},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=ek9a0qIafW}
}
```
