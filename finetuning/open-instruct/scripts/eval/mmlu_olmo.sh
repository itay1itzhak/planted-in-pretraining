#MODEL_DIR=output/tulu_v2_OLMo-7B_lora_merged
#MODEL_DIR=output/tulu_v2_OLMo-7B_lora/step_2000/merged 
MODEL_DIR=output/tulu_v2_OLMo-7B_lora_r128_alpha128/step_1000/merged
#MODEL_DIR=allenai/OLMo-7B-SFT
#MODEL_DIR=/mnt/nlp/datasets/huggingface/models/models--allenai--OLMo-7B-SFT/snapshots/fc02d4043f10aee6e37be17265cfa2cc907bb727/


python -m eval.mmlu.run_eval_olmo \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir ${MODEL_DIR}/mmlu \
    --model_name_or_path ${MODEL_DIR} \
    --tokenizer_name_or_path ${MODEL_DIR} \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# MODEL_DIR=/mnt/nlp/datasets/huggingface/models/models--allenai--OLMo-7B/snapshots/09dd55d8d37c14aa0cbab5a4ac545140d2bd0a60/
# python -m eval.mmlu.run_eval_olmo \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir ${MODEL_DIR}/mmlu \
#     --model_name_or_path ${MODEL_DIR} \
#     --tokenizer_name_or_path ${MODEL_DIR} \
#     --eval_batch_size 1 \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format