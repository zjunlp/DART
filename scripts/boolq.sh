export CUDA_VISIBLE_DEVICES=2 &&

python3 cli.py \
--data_dir data/FewGLUE_32dev/BoolQ \
--model_type albert \
--model_name_or_path albert-xxlarge-v2 \
--cache_dir pretrain/albert-xxlarge-v2 \
--task_name boolq \
--output_dir output/boolq \
--do_eval \
--do_train \
--pet_per_gpu_eval_batch_size 8 \
--pet_per_gpu_train_batch_size 8 \
--pet_gradient_accumulation_steps 2 \
--pet_max_seq_length 256 \
--pet_max_steps 250 \
--pattern_ids 1 \
--learning_rate 1e-4 \
--prompt_encoder_type "none"

# LSTM encoder
# 2021-04-29 20:44:54,432 - INFO - modeling - --- RESULT (pattern_id=1, iteration=2) ---
# 2021-04-29 20:44:54,432 - INFO - modeling - eval_results:
# 2021-04-29 20:44:54,432 - INFO - modeling - {'acc': 0.7810397553516819}
# 2021-04-29 20:44:54,432 - INFO - modeling - dev32_results:
# 2021-04-29 20:44:54,432 - INFO - modeling - {'acc': 0.59375}
# 2021-04-29 20:44:54,644 - INFO - modeling - === OVERALL RESULTS ===
# 2021-04-29 20:44:54,645 - INFO - modeling - eval_results:
# 2021-04-29 20:44:54,645 - INFO - modeling - acc-p1: 0.7662589194699286 +- 0.020773359138114206
# 2021-04-29 20:44:54,646 - INFO - modeling - acc-all-p: 0.7662589194699286 +- 0.020773359138114206
# 2021-04-29 20:44:54,646 - INFO - modeling - dev32_results:
# 2021-04-29 20:44:54,646 - INFO - modeling - acc-p1: 0.6145833333333334 +- 0.03608439182435161
# 2021-04-29 20:44:54,646 - INFO - modeling - acc-all-p: 0.6145833333333334 +- 0.03608439182435161

# None encoder
# eval_results:
# acc-p1: 0.7162079510703364 +- 0.0663481716762667
# acc-all-p: 0.7162079510703364 +- 0.0663481716762667
# dev32_results:
# acc-p1: 0.6145833333333334 +- 0.018042195912175804
# acc-all-p: 0.6145833333333334 +- 0.018042195912175804