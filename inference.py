import pdb
import torch
import pickle

from model import TransformerModelWrapper
from data_utils import load_examples, load_metrics, TRAIN_SET, TEST_SET, DEV32_SET
from config import WrapperConfig, EvalConfig, TrainConfig
from visualize import get_top_words
from utils import set_seed

device = 'cuda'
task_name = 'cr'
label_list = ['0', '1']
prompt_type = 'none'
# train on selected samples
# data_dir = 'data/select/SST-2/'
# output_dir = 'output/select/SST-2/none/'
# dev_res_file = 'visual/select/SST-2-none-dev.eval'
# test_res_file = 'visual/select/SST-2-none-test.eval'

# train on 16-shot samples, 'cd' means cat-dog pair as verbalizers
data_dir = 'data/k-shot/{}/16-13/'.format(task_name)
output_dir = 'output/{}/{}/16-13/'.format(task_name, prompt_type)
dev_res_file = 'visual/{}-{}-dev.eval'.format(task_name, prompt_type)
test_res_file = 'visual/{}-{}-test.eval'.format(task_name, prompt_type)

set_seed(42)

model_config = WrapperConfig(model_type='roberta',
                             model_name_or_path='roberta-large',
                             task_name=task_name,
                             label_list=label_list,
                             max_seq_length=128,
                             device=device,
                             cache_dir='pretrain/roberta-large',
                             output_dir=output_dir,
                             embed_size=1024,
                             prompt_encoder_type=prompt_type,
                             eval_every_step=5)

train_config = TrainConfig(device=device,
                           per_gpu_train_batch_size=16,
                           max_steps=250,
                           max_steps_stage1=250,
                           weight_decay=0.1,
                           learning_rate=1e-05,
                           early_stop_epochs=10)

eval_config = EvalConfig(device=device,
                         metrics=load_metrics(task_name))

# Load dataset
train_data = load_examples(task_name, data_dir, TRAIN_SET, num_examples=-1)
dev_data = load_examples(task_name, data_dir, DEV32_SET, num_examples=-1)
eval_data = load_examples(task_name, data_dir, TEST_SET, num_examples=-1)

wrapper = TransformerModelWrapper(model_config)

# Evaluation before training
verbalizers = list(wrapper.pvp.VERBALIZER.values())
eval_train = [wrapper.eval(train_data)]
all_eval_dev = [wrapper.eval(dev_data)]
all_eval_test = [wrapper.eval(eval_data)]
# print('top_words_train:', get_top_words(eval_train, verbalizers)[0])
# print('top_words_dev:', get_top_words(all_eval_dev, verbalizers)[0])
# print('top_words_test:', get_top_words(all_eval_test, verbalizers)[0])

# Perform training
global_step, tr_loss, eval_dev, eval_test = wrapper.train(
    pattern_iter_output_dir=output_dir,
    eval_config=eval_config,
    train_data=train_data,
    dev_data=dev_data,
    eval_data=eval_data,
    per_gpu_train_batch_size=train_config.per_gpu_train_batch_size,
    n_gpu=train_config.n_gpu,
    num_train_epochs=train_config.num_train_epochs,
    max_steps=train_config.max_steps,
    gradient_accumulation_steps=train_config.gradient_accumulation_steps,
    weight_decay=train_config.weight_decay,
    learning_rate=train_config.learning_rate,
    adam_epsilon=train_config.adam_epsilon,
    warmup_steps=train_config.warmup_steps,
    max_grad_norm=train_config.max_grad_norm,
    alpha=train_config.alpha,
    early_stop_epochs=train_config.early_stop_epochs,
    # Record evaluation results
    record_eval=True
)

all_eval_dev.extend(eval_dev)
all_eval_test.extend(eval_test)
all_eval_dev.append(wrapper.eval(dev_data))
all_eval_test.append(wrapper.eval(eval_data))
all_eval_test = list(filter(None, all_eval_test))
print([a['scores']['acc'] for a in all_eval_dev])
print([a['scores']['acc'] for a in all_eval_test])

pickle.dump(all_eval_dev, open(dev_res_file, 'wb'))
pickle.dump(all_eval_test, open(test_res_file, 'wb'))
