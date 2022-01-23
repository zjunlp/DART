export CUDA_VISIBLE_DEVICES=0 &&

python3 cli.py \
--data_dir data/k-shot/SST-2/16-13 \
--model_type albert \
--model_name_or_path albert-xxlarge-v2 \
--cache_dir pretrain/albert-xxlarge-v2 \
--task_name sst-2 \
--output_dir output/sst-2-lstm \
--do_eval \
--do_train \
--pet_per_gpu_eval_batch_size 8 \
--pet_per_gpu_train_batch_size 16 \
--pet_gradient_accumulation_steps 1 \
--pet_max_seq_length 128 \
--pet_max_steps 250 \
--pattern_ids 2 \
--learning_rate 1e-4 \
--eval_set "test" \
--prompt_encoder_type "lstm" \
--two_stage_train


# Albert-xxlarge-v2

# LSTM encoder (2 words)
# eval_results:
# acc-p2: 0.9147553516819572 +- 0.0023872316507639484
# acc-all-p: 0.9147553516819572 +- 0.0023872316507639484
# dev32_results:
# acc-p2: 0.96875 +- 0.0
# acc-all-p: 0.96875 +- 0.0

# None encoder
# eval_results:
# acc-p1: 0.9151376146788991 +- 0.006385050874805097
# acc-all-p: 0.9151376146788991 +- 0.006385050874805097
# dev32_results:
# acc-p1: 0.9583333333333334 +- 0.018042195912175804
# acc-all-p: 0.9583333333333334 +- 0.018042195912175804

# Inner encoder (2 words)
# eval_results:
# acc-p2: 0.8891437308868502 +- 0.05212951758130996
# acc-all-p: 0.8891437308868502 +- 0.05212951758130996
# dev32_results:
# acc-p2: 0.9583333333333334 +- 0.018042195912175804
# acc-all-p: 0.9583333333333334 +- 0.018042195912175804

# Inner encoder (2 stage)

