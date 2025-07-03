python open_instruct/merge_lora.py \
    --base_model_name_or_path allenai/OLMo-7B \
    --lora_model_name_or_path $1 \
    --output_dir $1/merged \
    --save_tokenizer \
    --use_fast_tokenizer