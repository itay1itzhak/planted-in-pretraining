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
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# delete the safetensors files
rm ${MODEL_DIR}/merged/model-0000*



# python -m eval.mmlu.run_eval_olmo \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir /mnt/nlp/models/models--google--flan-t5-xxl/mmlu \
#     --model_name_or_path google/flan-t5-xxl  \
#     --tokenizer_name_or_path google/flan-t5-xxl  \
#     --cache_dir "/mnt/nlp/models" \
#     --eval_batch_size 1 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# python -m eval.mmlu.run_eval_olmo \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir output/allenai/tulu-v2-sft-mixture_t5-v1_1-xxl_lora_r128_alpha256_LR5e-5/merged/mmlu_5_shot \
#     --model_name_or_path output/allenai/tulu-v2-sft-mixture_t5-v1_1-xxl_lora_r128_alpha256_LR5e-5/merged  \
#     --tokenizer_name_or_path output/allenai/tulu-v2-sft-mixture_t5-v1_1-xxl_lora_r128_alpha256_LR5e-5/merged  \
#     --eval_batch_size 1 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# python open_instruct/merge_lora.py \
#     --base_model_name_or_path google/t5-v1_1-xxl \
#     --lora_model_name_or_path output/allenai/tulu-v2-sft-mixture_t5-v1_1-xxl_lora_r128_alpha256_LR1e-4/step_2500 \
#     --output_dir output/allenai/tulu-v2-sft-mixture_t5-v1_1-xxl_lora_r128_alpha256_LR1e-4/step_2500/merged \
#     --save_tokenizer \
#     --use_fast_tokenizer

# python open_instruct/merge_lora.py \
#     --base_model_name_or_path google/t5-v1_1-xxl \
#     --lora_model_name_or_path output/allenai/tulu-v2-sft-mixture_t5-v1_1-xxl_lora_r128_alpha256_LR5e-5 \
#     --output_dir output/allenai/tulu-v2-sft-mixture_t5-v1_1-xxl_lora_r128_alpha256_LR5e-5/merged \
#     --save_tokenizer \
#     --use_fast_tokenizer