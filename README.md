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
- To train / test on a data split from a single task with specific parameters, use `run.py`.
  - For customized training & evaluation, you can modify based on the sample configuration file `config/sample.yml`.
```bash
$ python run.py -h  
usage: run.py [-h] [--config CONFIG] [--do_train] [--do_test]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Configuration file storing all parameters
  --do_train
  --do_test
```
- To search optimal hyper-parameters for each task and reproduce our result, please use `sweep.py`:
  - Please refer to documentation for [WandB](https://docs.wandb.ai/) for more details.
  - **❗NOTE: we follow [LM-BFF](https://github.com/princeton-nlp/LM-BFF) in that we search optimal sets of hyper-parameters on different data splits respectively.**
```bash
$ python sweep.py -h
usage: sweep.py [-h] [--project_name PROJECT_NAME] --task_name TASK_NAME
                [--data_split {13,21,42,87,100}]
                [--pretrain_model PRETRAIN_MODEL] [--pet_method {pet,diffpet}]
                [--random_seed RANDOM_SEED] [--max_run MAX_RUN]

optional arguments:
  -h, --help            show this help message and exit
  --project_name PROJECT_NAME
                        project name for sweep
  --task_name TASK_NAME
  --data_split {13,21,42,87,100}
                        few-shot split-id for GLUE dataset
  --pretrain_model PRETRAIN_MODEL
                        name or path for pretrained model
  --pet_method {pet,diffpet}
                        prompt encoding method
  --random_seed RANDOM_SEED
                        random seed for training
  --max_run MAX_RUN     maximum tries for sweep
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
