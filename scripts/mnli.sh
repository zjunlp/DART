export CUDA_VISIBLE_DEVICES=2 &&

python3 cli.py \
--data_dir data/k-shot/MNLI/16-13 \
--model_type albert \
--model_name_or_path albert-xxlarge-v2 \
--cache_dir pretrain/albert-xxlarge-v2 \
--task_name mnli \
--output_dir output/mnli \
--do_eval \
--do_train \
--pet_per_gpu_eval_batch_size 8 \
--pet_per_gpu_train_batch_size 8 \
--pet_gradient_accumulation_steps 2 \
--pet_max_seq_length 256 \
--pet_max_steps 250 \
--pattern_ids 2 \
--learning_rate 1e-4 \
--eval_set "test" \
--prompt_encoder_type "none"

# None encoder
# eval_results:
# acc-p2: 0.600169808116828 +- 0.013383163631972188
# acc-all-p: 0.600169808116828 +- 0.013383163631972188
# dev32_results:
# acc-p2: 0.6944444444444444 +- 0.012028130608117225
# acc-all-p: 0.6944444444444444 +- 0.012028130608117225

# 