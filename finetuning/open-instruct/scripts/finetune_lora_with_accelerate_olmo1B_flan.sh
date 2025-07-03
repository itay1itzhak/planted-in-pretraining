export CUDA_VISIBLE_DEVICES=0

MODEL_SIZE=1B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training OLMo model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --main_process_port 9899 \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path allenai/OLMo-7B \
    --use_lora \
    --add_bos \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --tokenizer_name allenai/OLMo-${MODEL_SIZE} \
    --dataset_name flan_2022 \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps 500 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir output/flan_v2_OLMo-${MODEL_SIZE}_lora/ \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 &&  \

python open_instruct/merge_lora.py \
    --base_model_name_or_path allenai/OLMo-${MODEL_SIZE} \
    --lora_model_name_or_path output/flan_v2_${MODEL_SIZE}_lora/ \
    --output_dir output/flan_v2_${MODEL_SIZE}_lora_merged/ \
    --save_tokenizer
