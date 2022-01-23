export CUDA_VISIBLE_DEVICES=2 &&

python3 cli.py \
--data_dir data/FewGLUE_32dev/WiC \
--model_type albert \
--model_name_or_path albert-xxlarge-v2 \
--cache_dir pretrain/albert-xxlarge-v2 \
--task_name wic \
--output_dir output/wic \
--do_eval \
--do_train \
--pet_per_gpu_eval_batch_size 8 \
--pet_per_gpu_train_batch_size 8 \
--pet_gradient_accumulation_steps 2 \
--pet_max_seq_length 256 \
--pet_max_steps 3500 \
--pattern_ids 1 \
--prompt_encoder_type "none"

# LSTM encoder
# 1 gpu / 8 batch size / 2 grad acc
# 2021-04-30 15:46:02,237 - INFO - modeling - --- RESULT (pattern_id=1, iteration=2) ---
# 2021-04-30 15:46:02,238 - INFO - modeling - eval_results:
# 2021-04-30 15:46:02,238 - INFO - modeling - {'acc': 0.5329153605015674}
# 2021-04-30 15:46:02,238 - INFO - modeling - dev32_results:
# 2021-04-30 15:46:02,238 - INFO - modeling - {'acc': 0.53125}
# 2021-04-30 15:46:02,457 - INFO - modeling - === OVERALL RESULTS ===
# 2021-04-30 15:46:02,457 - INFO - modeling - eval_results:
# 2021-04-30 15:46:02,458 - INFO - modeling - acc-p1: 0.5470219435736677 +- 0.012440836885883636
# 2021-04-30 15:46:02,458 - INFO - modeling - acc-all-p: 0.5470219435736677 +- 0.012440836885883636
# 2021-04-30 15:46:02,458 - INFO - modeling - dev32_results:
# 2021-04-30 15:46:02,458 - INFO - modeling - acc-p1: 0.53125 +- 0.09375
# 2021-04-30 15:46:02,459 - INFO - modeling - acc-all-p: 0.53125 +- 0.09375

# 1 gpu / 16 batch size / 1 grad acc
# OOM

# None encoder
# 1 gpu / 8 batch size / 2 grad acc
# eval_results:
# acc-p1: 0.5501567398119122 +- 0.02618071017004712
# acc-all-p: 0.5501567398119122 +- 0.02618071017004712
# dev32_results:
# acc-p1: 0.6145833333333334 +- 0.047735163489123336
# acc-all-p: 0.6145833333333334 +- 0.047735163489123336
