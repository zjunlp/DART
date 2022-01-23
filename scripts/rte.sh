export CUDA_VISIBLE_DEVICES=2 &&

python3 cli.py \
--data_dir data/FewGLUE_32dev/RTE \
--model_type albert \
--model_name_or_path albert-xxlarge-v2 \
--cache_dir pretrain/albert-xxlarge-v2 \
--task_name rte \
--output_dir output/rte \
--do_eval \
--do_train \
--pet_per_gpu_eval_batch_size 8 \
--pet_per_gpu_train_batch_size 8 \
--pet_gradient_accumulation_steps 2 \
--pet_max_seq_length 256 \
--pet_max_steps 3500 \
--warmup_steps 150 \
--pattern_ids 1 \
--learning_rate 1e-4

# 2021-05-01 04:52:32,491 - INFO - modeling - --- RESULT (pattern_id=1, iteration=2) ---
# 2021-05-01 04:52:32,491 - INFO - modeling - eval_results:
# 2021-05-01 04:52:32,491 - INFO - modeling - {'acc': 0.7436823104693141}
# 2021-05-01 04:52:32,491 - INFO - modeling - dev32_results:
# 2021-05-01 04:52:32,491 - INFO - modeling - {'acc': 0.78125}
# 2021-05-01 04:52:32,696 - INFO - modeling - === OVERALL RESULTS ===
# 2021-05-01 04:52:32,696 - INFO - modeling - eval_results:
# 2021-05-01 04:52:32,697 - INFO - modeling - acc-p1: 0.7605294825511433 +- 0.014590079004792002
# 2021-05-01 04:52:32,697 - INFO - modeling - acc-all-p: 0.7605294825511433 +- 0.014590079004792002
# 2021-05-01 04:52:32,697 - INFO - modeling - dev32_results:
# 2021-05-01 04:52:32,698 - INFO - modeling - acc-p1: 0.7604166666666666 +- 0.03608439182435161
# 2021-05-01 04:52:32,698 - INFO - modeling - acc-all-p: 0.7604166666666666 +- 0.03608439182435161
