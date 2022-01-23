export CUDA_VISIBLE_DEVICES=0 &&

python3 cli.py \
--data_dir data/FewGLUE_32dev/CB \
--model_type albert \
--model_name_or_path albert-xxlarge-v2 \
--cache_dir pretrain/albert-xxlarge-v2 \
--task_name cb \
--output_dir output/cb-none-new \
--do_eval \
--do_train \
--pet_per_gpu_eval_batch_size 16 \
--pet_per_gpu_train_batch_size 4 \
--pet_gradient_accumulation_steps 4 \
--pet_max_seq_length 256 \
--pet_max_steps 250 \
--pattern_ids 1 \
--prompt_encoder_type "none"

# LSTM encoder
# 2021-04-29 14:17:14,784 - INFO - modeling - --- RESULT (pattern_id=1, iteration=2) ---
# 2021-04-29 14:17:14,784 - INFO - modeling - eval_results:
# 2021-04-29 14:17:14,784 - INFO - modeling - {'acc': 0.8571428571428571, 'f1-macro': 0.7876010781671159}
# 2021-04-29 14:17:14,784 - INFO - modeling - dev32_results:
# 2021-04-29 14:17:14,784 - INFO - modeling - {'acc': 0.84375, 'f1-macro': 0.5798319327731093}
# 2021-04-29 14:17:14,928 - INFO - modeling - === OVERALL RESULTS ===
# 2021-04-29 14:17:14,928 - INFO - modeling - eval_results:
# 2021-04-29 14:17:14,929 - INFO - modeling - acc-p1: 0.8392857142857143 +- 0.01785714285714285
# 2021-04-29 14:17:14,929 - INFO - modeling - f1-macro-p1: 0.7096049443219254 +- 0.11412626515099333
# 2021-04-29 14:17:14,929 - INFO - modeling - acc-all-p: 0.8392857142857143 +- 0.01785714285714285
# 2021-04-29 14:17:14,930 - INFO - modeling - f1-macro-all-p: 0.7096049443219254 +- 0.11412626515099333
# 2021-04-29 14:17:14,930 - INFO - modeling - dev32_results:
# 2021-04-29 14:17:14,930 - INFO - modeling - acc-p1: 0.8645833333333334 +- 0.03608439182435161
# 2021-04-29 14:17:14,930 - INFO - modeling - f1-macro-p1: 0.5927874823643657 +- 0.025808312686471435
# 2021-04-29 14:17:14,930 - INFO - modeling - acc-all-p: 0.8645833333333334 +- 0.03608439182435161
# 2021-04-29 14:17:14,931 - INFO - modeling - f1-macro-all-p: 0.5927874823643657 +- 0.025808312686471435

# None encoder
# eval_results:
# acc-p1: 0.8452380952380952 +- 0.020619652471058052
# f1-macro-p1: 0.7617958861701039 +- 0.03954535314794234
# acc-all-p: 0.8452380952380952 +- 0.020619652471058052
# f1-macro-all-p: 0.7617958861701039 +- 0.03954535314794234
# dev32_results:
# acc-p1: 0.9270833333333334 +- 0.018042195912175804
# f1-macro-p1: 0.7613078279744947 +- 0.12238855048313522
# acc-all-p: 0.9270833333333334 +- 0.018042195912175804
# f1-macro-all-p: 0.7613078279744947 +- 0.12238855048313522

# Inner encoder (embedding optizer + other optimizer)
# eval_results:
# acc-p1: 0.7738095238095238 +- 0.020619652471058052
# f1-macro-p1: 0.6929723488250594 +- 0.016809889588982323
# acc-all-p: 0.7738095238095238 +- 0.020619652471058052
# f1-macro-all-p: 0.6929723488250594 +- 0.016809889588982323
# dev32_results:
# acc-p1: 0.8645833333333334 +- 0.03608439182435161
# f1-macro-p1: 0.5915179687109511 +- 0.026837394429267372
# acc-all-p: 0.8645833333333334 +- 0.03608439182435161
# f1-macro-all-p: 0.5915179687109511 +- 0.026837394429267372

# Inner encoder (one optimizer)
# eval_results:
# acc-p1: 0.7916666666666666 +- 0.03717260713332379
# f1-macro-p1: 0.7196909328145764 +- 0.032950714527130116
# acc-all-p: 0.7916666666666666 +- 0.03717260713332379
# f1-macro-all-p: 0.7196909328145764 +- 0.032950714527130116
# dev32_results:
# acc-p1: 0.8958333333333334 +- 0.047735163489123336
# f1-macro-p1: 0.6192394146642513 +- 0.02733848006141917
# acc-all-p: 0.8958333333333334 +- 0.047735163489123336
# f1-macro-all-p: 0.6192394146642513 +- 0.02733848006141917

