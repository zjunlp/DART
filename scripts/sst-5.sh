export CUDA_VISIBLE_DEVICES=0 &&

python3 cli.py \
--data_dir data/k-shot/sst-5/16-13 \
--model_type roberta \
--embed_size=1024 \
--model_name_or_path roberta-large \
--cache_dir pretrain/roberta-large \
--task_name sst-5 \
--do_train \
--do_eval \
--pet_per_gpu_eval_batch_size 8 \
--pet_per_gpu_train_batch_size 16 \
--pet_gradient_accumulation_steps 1 \
--pet_max_seq_length 128 \
--pet_max_steps 250 \
--learning_rate 1e-4 \
--eval_set "test" \
--prompt_encoder_type "inner" \
--output_dir output/sst-5-inner-2-5e-6 \
--two_stage_train \
--learning_rate_stage1 1e-5 \
--pet_max_steps_stage1 100

# Albert-xxlarge-v2

# LSTM encoder (2 words)

# None encoder

# Inner encoder (2 words)

# Inner encoder (2 stage)

# RoBERTa-large

# None encoder
# INFO:train:=== OVERALL RESULTS ===
# INFO:utils:eval_results:
# INFO:utils:acc-p1: 0.4689291101055807 +- 0.011802353037399486
# INFO:utils:acc-all-p: 0.4689291101055807 +- 0.011802353037399486
# INFO:utils:dev_results:
# INFO:utils:acc-p1: 0.5625 +- 0.03307189138830737
# INFO:utils:acc-all-p: 0.5625 +- 0.03307189138830737

# LSTM encoder
# INFO:train:eval_results:
# INFO:train:{'acc': 0.48733031674208144}
# INFO:train:dev_results:
# INFO:train:{'acc': 0.6125}
# INFO:train:=== OVERALL RESULTS ===
# INFO:utils:eval_results:
# INFO:utils:acc-p1: 0.46576168929110107 +- 0.021493609585919893
# INFO:utils:acc-all-p: 0.46576168929110107 +- 0.021493609585919893
# INFO:utils:dev_results:
# INFO:utils:acc-p1: 0.5708333333333333 +- 0.04389855730355308
# INFO:utils:acc-all-p: 0.5708333333333333 +- 0.04389855730355308

# Inner encoder
# INFO:train:eval_results:
# INFO:train:{'acc': 0.38190045248868776}
# INFO:train:dev_results:
# INFO:train:{'acc': 0.4}
# INFO:train:=== OVERALL RESULTS ===
# INFO:utils:eval_results:
# INFO:utils:acc-p1: 0.3904977375565611 +- 0.007718878782457913
# INFO:utils:acc-all-p: 0.3904977375565611 +- 0.007718878782457913
# INFO:utils:dev_results:
# INFO:utils:acc-p1: 0.39166666666666666 +- 0.007216878364870329
# INFO:utils:acc-all-p: 0.39166666666666666 +- 0.007216878364870329
