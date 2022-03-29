# DART
Implementation for ICLR2022 paper *[Differentiable Prompt Makes Pre-trained Language Models Better Few-shot Learners](https://arxiv.org/pdf/2108.13161.pdf)*.
## Environment
- python@3.6
- Use `pip install -r requirements.txt` to install dependencies.
- `wandb` account is required if the user wants to search for best hyper-parameter combinations.
## Data source
- 16-shot GLUE dataset from [LM-BFF](https://github.com/princeton-nlp/LM-BFF).
- Generated data consists of 5 random splits (13/21/42/87/100) for a task, each has 16 samples.
## How to run
- To run across each 5 splits in a task, use `run.py`:
  - In the arguments, `encoder="inner"` is the method proposed in the paper where verbalizers are other trainable tokens; `encoder="manual"` means verbalizers are selected fixed tokens; `encoder="lstm"` refers to the [P-Tuning](https://github.com/THUDM/P-tuning) method.
```bash
$ python run.py -h
usage: run.py [-h] [--encoder {manual,lstm,inner,inner2}] [--task TASK]
              [--num_splits NUM_SPLITS] [--repeat REPEAT] [--load_manual]
              [--extra_mask_rate EXTRA_MASK_RATE]
              [--output_dir_suffix OUTPUT_DIR_SUFFIX]

optional arguments:
  -h, --help            show this help message and exit
  --encoder {manual,lstm,inner,inner2}
  --task TASK
  --num_splits NUM_SPLITS
  --repeat REPEAT
  --load_manual
  --extra_mask_rate EXTRA_MASK_RATE
  --output_dir_suffix OUTPUT_DIR_SUFFIX, -o OUTPUT_DIR_SUFFIX
```
- To train and evaluate on a single split with details recorded, use `inference.py`.
  - Before running, [`task_name`, `label_list`, `prompt_type`] should be configured in the code.
  - `prompt_type="none"` refers to fixed verbalizer training, while `"inner"` refers to the method proposed in the paper. (`"inner2"` is deprecated 2-stage training)
- To find optimal hyper-parameters for each task-split and reproduce our result, please use `sweep.py`:
  - Please refer to documentation for [WandB](https://docs.wandb.ai/) for more details.
```bash
$ python sweep.py -h
usage: sweep.py [-h]
                [--task {SST-2,sst-5,mr,cr,mpqa,subj,trec,CoLA,MNLI,MNLI-mm,SNLI,QNLI,RTE-glue,MRPC,QQP}]
                [--encoder {none,mlp,lstm,inner,inner2}]
                [--seed_split {13,21,42,87,100} [{13,21,42,87,100} ...]]
                [--batch_size {4,8,16,24,32} [{4,8,16,24,32} ...]]
                [--sweep_id SWEEP_ID]

optional arguments:
  -h, --help            show this help message and exit
  --task {SST-2,sst-5,mr,cr,mpqa,subj,trec,CoLA,MNLI,MNLI-mm,SNLI,QNLI,RTE-glue,MRPC,QQP}
  --encoder {none,mlp,lstm,inner,inner2}
  --seed_split {13,21,42,87,100} [{13,21,42,87,100} ...]
  --batch_size {4,8,16,24,32} [{4,8,16,24,32} ...]
  --sweep_id SWEEP_ID
```
- To train and evaluate with more customized configurations, use `cli.py`.
- To analyze and visualize the results come from `inference.py`, use `visualize.py` and `visualize_word_emb.py`.
## How to Cite
```
@article{zhang2021differentiable,
  title={Differentiable prompt makes pre-trained language models better few-shot learners},
  author={Zhang, Ningyu and Li, Luoqiu and Chen, Xiang and Deng, Shumin and Bi, Zhen and Tan, Chuanqi and Huang, Fei and Chen, Huajun},
  journal={arXiv preprint arXiv:2108.13161},
  year={2021}
}
```
