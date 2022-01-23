export CUDA_VISIBLE_DEVICES=2 &&

python3 cli.py \
--data_dir data/FewGLUE_32dev/COPA \
--model_type albert \
--model_name_or_path albert-xxlarge-v2 \
--cache_dir pretrain/albert-xxlarge-v2 \
--task_name copa \
--output_dir output/copa \
--do_eval \
--do_train \
--pet_per_gpu_eval_batch_size 1 \
--pet_per_gpu_train_batch_size 16 \
--pet_gradient_accumulation_steps 1 \
--pet_max_seq_length 96 \
--pet_max_steps 3500 \
--pattern_ids 1 \
--prompt_encoder_type "none"

# LSTM encoder
# 1 gpu / 8 batch size / 2 grad acc
# 2021-04-29 21:47:36,357 - INFO - modeling - --- RESULT (pattern_id=1, iteration=2) ---
# 2021-04-29 21:47:36,358 - INFO - modeling - eval_results:
# 2021-04-29 21:47:36,358 - INFO - modeling - {'acc': 0.76}
# 2021-04-29 21:47:36,358 - INFO - modeling - dev32_results:
# 2021-04-29 21:47:36,358 - INFO - modeling - {'acc': 0.65625}
# 2021-04-29 21:47:36,545 - INFO - modeling - === OVERALL RESULTS ===
# 2021-04-29 21:47:36,546 - INFO - modeling - eval_results:
# 2021-04-29 21:47:36,546 - INFO - modeling - acc-p1: 0.7633333333333333 +- 0.05507570547286101
# 2021-04-29 21:47:36,547 - INFO - modeling - acc-all-p: 0.7633333333333333 +- 0.05507570547286101
# 2021-04-29 21:47:36,547 - INFO - modeling - dev32_results:
# 2021-04-29 21:47:36,547 - INFO - modeling - acc-p1: 0.6354166666666666 +- 0.06505206248331666
# 2021-04-29 21:47:36,547 - INFO - modeling - acc-all-p: 0.6354166666666666 +- 0.06505206248331666

# 1 gpu / 16 batch size / 1 grad acc
# 2021-05-01 07:51:21,803 - INFO - modeling - --- RESULT (pattern_id=1, iteration=2) ---
# 2021-05-01 07:51:21,803 - INFO - modeling - eval_results:
# 2021-05-01 07:51:21,803 - INFO - modeling - {'acc': 0.65}
# 2021-05-01 07:51:21,803 - INFO - modeling - dev32_results:
# 2021-05-01 07:51:21,804 - INFO - modeling - {'acc': 0.6875}
# 2021-05-01 07:51:22,019 - INFO - modeling - === OVERALL RESULTS ===
# 2021-05-01 07:51:22,019 - INFO - modeling - eval_results:
# 2021-05-01 07:51:22,020 - INFO - modeling - acc-p1: 0.72 +- 0.07
# 2021-05-01 07:51:22,020 - INFO - modeling - acc-all-p: 0.72 +- 0.07
# 2021-05-01 07:51:22,020 - INFO - modeling - dev32_results:
# 2021-05-01 07:51:22,021 - INFO - modeling - acc-p1: 0.6979166666666666 +- 0.018042195912175804
# 2021-05-01 07:51:22,021 - INFO - modeling - acc-all-p: 0.6979166666666666 +- 0.018042195912175804

# None encoder
# eval_results:
# acc-p1: 0.84 +- 0.026457513110645866
# acc-all-p: 0.84 +- 0.026457513110645866
# dev32_results:
# acc-p1: 0.71875 +- 0.03125
# acc-all-p: 0.71875 +- 0.03125
