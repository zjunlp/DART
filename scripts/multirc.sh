export CUDA_VISIBLE_DEVICES=2 &&

python3 cli.py \
  --data_dir data/FewGLUE_32dev/MultiRC \
  --model_type albert \
  --model_name_or_path albert-xxlarge-v2 \
  --cache_dir pretrain/albert-xxlarge-v2 \
  --task_name multirc \
  --output_dir output/multirc \
  --do_eval \
  --do_train \
  --pet_per_gpu_eval_batch_size 4 \
  --pet_per_gpu_train_batch_size 4 \
  --pet_gradient_accumulation_steps 4 \
  --pet_max_seq_length 512 \
  --pet_max_steps 250 \
  --pattern_ids 1 \
  --learning_rate 1e-4 \
  --prompt_encoder_type "none"

# LSTM encoder
# 1 gpu / 4 batch size / 4 grad acc
# 2021-04-30 21:33:43,992 - INFO - modeling - --- RESULT (pattern_id=1, iteration=2) ---
# 2021-04-30 21:33:43,992 - INFO - modeling - eval_results:
# 2021-04-30 21:33:43,992 - INFO - modeling - {'acc': 0.7695957095709571, 'f1': 0.7327111749222301, 'em': 0.3305351521511018}
# 2021-04-30 21:33:43,992 - INFO - modeling - dev32_results:
# 2021-04-30 21:33:43,992 - INFO - modeling - {'acc': 0.7904191616766467, 'f1': 0.7741935483870969, 'em': 0.28125}
# 2021-04-30 21:33:44,359 - INFO - modeling - === OVERALL RESULTS ===
# 2021-04-30 21:33:44,367 - INFO - modeling - eval_results:
# 2021-04-30 21:33:44,370 - INFO - modeling - acc-p1: 0.7320544554455446 +- 0.06061233508108515
# 2021-04-30 21:33:44,372 - INFO - modeling - f1-p1: 0.7031503390273145 +- 0.07043271204711432
# 2021-04-30 21:33:44,373 - INFO - modeling - em-p1: 0.26372857642532355 +- 0.10152026827795642
# 2021-04-30 21:33:44,374 - INFO - modeling - acc-all-p: 0.7320544554455446 +- 0.06061233508108515
# 2021-04-30 21:33:44,375 - INFO - modeling - f1-all-p: 0.7031503390273145 +- 0.07043271204711432
# 2021-04-30 21:33:44,376 - INFO - modeling - em-all-p: 0.26372857642532355 +- 0.10152026827795642
# 2021-04-30 21:33:44,381 - INFO - modeling - dev32_results:
# 2021-04-30 21:33:44,382 - INFO - modeling - acc-p1: 0.7465069860279441 +- 0.06595892871001324
# 2021-04-30 21:33:44,382 - INFO - modeling - f1-p1: 0.7263492114650577 +- 0.07449118301352026
# 2021-04-30 21:33:44,383 - INFO - modeling - em-p1: 0.25 +- 0.08267972847076846
# 2021-04-30 21:33:44,384 - INFO - modeling - acc-all-p: 0.7465069860279441 +- 0.06595892871001324
# 2021-04-30 21:33:44,385 - INFO - modeling - f1-all-p: 0.7263492114650577 +- 0.07449118301352026
# 2021-04-30 21:33:44,385 - INFO - modeling - em-all-p: 0.25 +- 0.08267972847076846

# 1 gpu / 16 batch size / 1 grad acc
# OOM

# 1 gpu / 8 batch size / 2 grad acc
# OOM

# None encoder
# eval_results:
# acc-p1: 0.7748212321232123 +- 0.009862957526825105
# f1-p1: 0.7277731070010395 +- 0.02252582355447236
# em-p1: 0.3280867436166492 +- 0.019696268940610107
# acc-all-p: 0.7748212321232123 +- 0.009862957526825105
# f1-all-p: 0.7277731070010395 +- 0.02252582355447236
# em-all-p: 0.3280867436166492 +- 0.019696268940610107
# dev32_results:
# acc-p1: 0.780439121756487 +- 0.03404934552740918
# f1-p1: 0.742813625759132 +- 0.04918222727381349
# em-p1: 0.3020833333333333 +- 0.10974639325888269
# acc-all-p: 0.780439121756487 +- 0.03404934552740918
# f1-all-p: 0.742813625759132 +- 0.04918222727381349
# em-all-p: 0.3020833333333333 +- 0.10974639325888269
