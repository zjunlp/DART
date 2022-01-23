export CUDA_VISIBLE_DEVICES=0 &&

python3 cli.py \
--data_dir data/k-shot/CoLA/16-13 \
--model_type roberta \
--model_name_or_path roberta-large \
--cache_dir pretrain/roberta-large \
--task_name cola \
--output_dir output/cola-inner-roberta \
--do_eval \
--do_train \
--pet_per_gpu_eval_batch_size 8 \
--pet_per_gpu_train_batch_size 16 \
--pet_gradient_accumulation_steps 1 \
--pet_max_seq_length 128 \
--pet_max_steps 250 \
--learning_rate 1e-4 \
--eval_set "test" \
--prompt_encoder_type "inner" \
# --two_stage_train

# Albert-xxlarge-v2

# None encoder
# eval_results:
# matt-p3: 0.011038530776971435 +- 0.028281248290183816
# matt-all-p: 0.011038530776971435 +- 0.028281248290183816
# dev_results:
# matt-p3: 0.021931723165325632 +- 0.03798685881987932
# matt-all-p: 0.021931723165325632 +- 0.03798685881987932

# LSTM encoder
# eval_results:
# matt-p3: 0.034183611894110906 +- 0.043415828580684636
# matt-all-p: 0.034183611894110906 +- 0.043415828580684636
# dev_results:
# matt-p3: 0.0 +- 0.20851441405707474
# matt-all-p: 0.0 +- 0.20851441405707474

# Inner encoder (1 stage)
# eval_results:
# matt-p3: 0.01723559740369088 +- 0.02582704623151028
# matt-all-p: 0.01723559740369088 +- 0.02582704623151028
# dev_results:
# matt-p3: 0.040944435600533624 +- 0.09739929903499565
# matt-all-p: 0.040944435600533624 +- 0.09739929903499565

# Roberta-large

# None encoder
# eval_results:
# matt-p3: -0.011170663159388474 +- 0.010580629907488072
# matt-all-p: -0.011170663159388474 +- 0.010580629907488072
# dev_results:
# matt-p3: 0.08694271005413028 +- 0.07542456350607658
# matt-all-p: 0.08694271005413028 +- 0.07542456350607658

# LSTM encoder
# eval_results:
# matt-p3: -0.0023523276317656697 +- 0.027398482696285833
# matt-all-p: -0.0023523276317656697 +- 0.027398482696285833
# dev_results:
# matt-p3: 0.09640919955955589 +- 0.08492199664559674
# matt-all-p: 0.09640919955955589 +- 0.08492199664559674

# Inner encoder (1 stage)


# Inner encoder (2 stage)
# INFO:utils:eval_results:
# INFO:utils:matt-p1: 0.034744314304350286 +- 0.014785237013213177
# INFO:utils:matt-all-p: 0.034744314304350286 +- 0.014785237013213177
# INFO:utils:dev_results:
# INFO:utils:matt-p1: 0.11402373936547473 +- 0.06656417570839829
# INFO:utils:matt-all-p: 0.11402373936547473 +- 0.06656417570839829

# Forget to remove the handle in the 2nd stage
# INFO:utils:eval_results:
# INFO:utils:matt-p1: 0.01935906539478563 +- 0.005234936322441165
# INFO:utils:matt-all-p: 0.01935906539478563 +- 0.005234936322441165
# INFO:utils:dev_results:
# INFO:utils:matt-p1: 0.16116727332092007 +- 0.035710016143793634
# INFO:utils:matt-all-p: 0.16116727332092007 +- 0.035710016143793634
