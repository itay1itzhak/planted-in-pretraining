#MODEL_DIR=output/tulu_v2_OLMo-7B_lora_r128_alpha256_LR2e-4/step_1000
MODEL_DIR=$1
python open_instruct/merge_lora.py \
    --base_model_name_or_path allenai/OLMo-7B \
    --lora_model_name_or_path ${MODEL_DIR} \
    --output_dir ${MODEL_DIR}/merged \
    --save_tokenizer \
    --use_fast_tokenizer

python -m eval.mmlu.run_eval_olmo \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir ${MODEL_DIR}/merged/mmlu \
    --model_name_or_path ${MODEL_DIR}/merged  \
    --tokenizer_name_or_path ${MODEL_DIR}/merged  \
    --eval_batch_size 1 

# delete the safetensors files
rm ${MODEL_DIR}/merged/model-0000*