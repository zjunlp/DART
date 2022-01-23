export CUDA_VISIBLE_DEVICES=2 &&

python3 cli.py \
--data_dir data/FewGLUE_32dev/WSC \
--model_type albert \
--model_name_or_path albert-xxlarge-v2 \
--cache_dir pretrain/albert-xxlarge-v2 \
--task_name wsc \
--output_dir output/wsc \
--do_eval \
--do_train \
--pet_per_gpu_eval_batch_size 1 \
--pet_per_gpu_train_batch_size 16 \
--pet_gradient_accumulation_steps 1 \
--pet_max_seq_length 128 \
--pet_max_steps 3500 \
--pattern_ids 2 \
--learning_rate 1e-4 \
--prompt_encoder_type "none"

# LSTM encoder
# 1 gpu / 8 batch size / 2 grad acc
# 2021-05-01 05:27:44,034 - INFO - modeling - --- RESULT (pattern_id=2, iteration=2) ---
# 2021-05-01 05:27:44,034 - INFO - modeling - eval_results:
# 2021-05-01 05:27:44,034 - INFO - modeling - {'acc': 0.7980769230769231}
# 2021-05-01 05:27:44,034 - INFO - modeling - dev32_results:
# 2021-05-01 05:27:44,034 - INFO - modeling - {'acc': 0.71875}
# 2021-05-01 05:27:44,184 - INFO - modeling - === OVERALL RESULTS ===
# 2021-05-01 05:27:44,185 - INFO - modeling - eval_results:
# 2021-05-01 05:27:44,185 - INFO - modeling - acc-p2: 0.7788461538461539 +- 0.025439916452544155
# 2021-05-01 05:27:44,185 - INFO - modeling - acc-all-p: 0.7788461538461539 +- 0.025439916452544155
# 2021-05-01 05:27:44,186 - INFO - modeling - dev32_results:
# 2021-05-01 05:27:44,186 - INFO - modeling - acc-p2: 0.71875 +- 0.03125
# 2021-05-01 05:27:44,186 - INFO - modeling - acc-all-p: 0.71875 +- 0.03125

# 1 gpu / 16 batch size / 1 grad acc
# eval_results:
# acc-p2: 0.7980769230769231 +- 0.019230769230769218
# acc-all-p: 0.7980769230769231 +- 0.019230769230769218
# dev32_results:
# acc-p2: 0.7291666666666666 +- 0.07864410870073699
# acc-all-p: 0.7291666666666666 +- 0.07864410870073699


# None encoder
# eval_results:
# acc-p2: 0.7916666666666666 +- 0.029375485224075914
# acc-all-p: 0.7916666666666666 +- 0.029375485224075914
# dev32_results:
# acc-p2: 0.6666666666666666 +- 0.06505206248331666
# acc-all-p: 0.6666666666666666 +- 0.06505206248331666
