#MODEL_DIR=output/tulu_v2_OLMo-7B_lora_r128_alpha256_LR2e-4/step_1000
MODEL_DIR=$1
python open_instruct/merge_lora.py \
    --base_model_name_or_path google/t5-v1_1-xxl \
    --lora_model_name_or_path ${MODEL_DIR} \
    --output_dir ${MODEL_DIR}/merged \
    --save_tokenizer \
    --use_fast_tokenizer

python -m eval.mmlu.run_eval_olmo \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir ${MODEL_DIR}/merged/mmlu_5_shot \
    --model_name_or_path ${MODEL_DIR}/merged  \
    --tokenizer_name_or_path ${MODEL_DIR}/merged  \
    --eval_batch_size 1 

# delete the safetensors files
rm ${MODEL_DIR}/merged/model-0000*


# python -m eval.mmlu.run_eval_olmo \
#     --ntrain 5 \
#     --save_dir /mnt/nlp/models/models--google--t5-v1_1-xxl/mmlu_no_special_tokens \
#     --model_name_or_path google/t5-v1_1-xxl  \
#     --tokenizer_name_or_path google/t5-v1_1-xxl  \
#     --cache_dir "/mnt/nlp/models" \
#     --eval_batch_size 1 

# python -m eval.mmlu.run_eval_olmo \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir /mnt/nlp/models/models--google--flan-t5-xxl/mmlu \
#     --model_name_or_path google/flan-t5-xxl  \
#     --tokenizer_name_or_path google/flan-t5-xxl  \
#     --cache_dir "/mnt/nlp/models" \
#     --eval_batch_size 1 


# python open_instruct/merge_lora.py \
#     --base_model_name_or_path google/t5-v1_1-xxl \
#     --lora_model_name_or_path output/flan_v2_t5-v1_1-xxl_lora_r128_alpha256_LR1e-4/step_9500 \
#     --output_dir output/flan_v2_t5-v1_1-xxl_lora_r128_alpha256_LR1e-4/step_9500/merged \
#     --save_tokenizer \
#     --use_fast_tokenizer

# python open_instruct/merge_lora.py \
#     --base_model_name_or_path google/t5-v1_1-xxl \
#     --lora_model_name_or_path output/flan_v2_t5-v1_1-xxl_lora_r128_alpha256_LR1e-4 \
#     --output_dir output/flan_v2_t5-v1_1-xxl_lora_r128_alpha256_LR1e-4/merged \
#     --save_tokenizer \
#     --use_fast_tokenizer