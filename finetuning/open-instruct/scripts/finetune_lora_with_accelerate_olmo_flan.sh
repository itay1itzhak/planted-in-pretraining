export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/itay.itzhak/miniconda3/envs/opinst3/lib

MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
LORA_RANK=$1
LORA_ALPHA=$2
LR=$3
echo "Training OLMo-Flan model ${MODEL_SIZE} with Lora rank ${LORA_RANK} and Lora alpha ${LORA_ALPHA} LR ${LR} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
#     --resume_from_checkpoint last \
# Lora training
accelerate launch \
    --mixed_precision f16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --main_process_port 9893 \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path allenai/OLMo-7B \
    --remove_previous_checkpoint \
    --trust_remote_code \
    --use_lora \
    --add_bos \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout 0.1 \
    --tokenizer_name allenai/OLMo-7B \
    --dataset_name flan_2022 \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps 500 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate $LR \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir output/flan_v2_OLMo-7B_lora_r${LORA_RANK}_alpha${LORA_ALPHA}_LR${LR}/ \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 &&  \

python open_instruct/merge_lora.py \
    --base_model_name_or_path allenai/OLMo-7B \
    --lora_model_name_or_path output/flan_v2_OLMo-${MODEL_SIZE}_lora_r${LORA_RANK}_alpha${LORA_ALPHA}_LR${LR}/ \
    --output_dir output/flan_v2_OLMo-${MODEL_SIZE}_lora_r${LORA_RANK}_alpha${LORA_ALPHA}_LR${LR}_merged/ \
    --save_tokenizer \
    --use_fast_tokenizer
