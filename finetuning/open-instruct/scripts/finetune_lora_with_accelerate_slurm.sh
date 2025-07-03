#!/bin/bash

# Usage function to display help
usage() {
  echo "Usage: $0 lora_rank alpha lr num_gpus [port] [model_name] [dataset_name] [batch_size_per_gpu] [total_batch_size] [resume_job] [seed]"
  echo "  lora_rank         : Lora rank parameter"
  echo "  alpha             : Lora alpha parameter"
  echo "  lr                : Learning rate parameter"
  echo "  num_gpus          : Number of GPUs"
  echo "  port              : (Optional) Port number, default is 9898"
  echo "  model_name        : (Optional) Model name, default is t5-v1_1-xxl"
  echo "  dataset_name      : (Optional) Dataset name, default is flan_2022"
  echo "  batch_size_per_gpu: (Optional) Batch size per GPU, default is 1"
  echo "  total_batch_size  : (Optional) Total batch size, default is 64"
  echo "  resume_job        : (Optional) Resume training from the last checkpoint, use 'true' to enable"
  echo "  seed              : (Optional) Random seed, default is 42"
  exit 1
}

# Ensure required parameters are provided
if [ $# -lt 4 ]; then
  usage
fi

# Parse parameters
LORA_RANK=$1
LORA_ALPHA=$2
LR=$3
NUM_GPUS=$4
PORT=${5:-9898}
MODEL_NAME=${6:-t5-v1_1-xxl}
DATASET_NAME=${7:-flan_2022}
BATCH_SIZE_PER_GPU=${8:-1}
TOTAL_BATCH_SIZE=${9:-64}
RESUME_JOB=${10:-false}
SEED=${11:-42}

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/itay.itzhak/miniconda3/envs/opinst4/lib

GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))

echo "Training ${MODEL_NAME}-${DATASET_NAME} model with Lora rank ${LORA_RANK}, Lora alpha ${LORA_ALPHA}, LR ${LR}, seed ${SEED}, using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Build the command for resume option
RESUME_CMD=""
if [ "$RESUME_JOB" = "true" ]; then
  RESUME_CMD="--resume_from_checkpoint last"
fi

# Set output directory with seed in the name
OUTPUT_DIR="output/${DATASET_NAME}_${MODEL_NAME}_lora_r${LORA_RANK}_alpha${LORA_ALPHA}_LR${LR}_seed_${SEED}/"

# Adjust flags for OLMo models
LOW_CPU_MEM_USAGE="--low_cpu_mem_usage"
MAX_SEQ_LENGTH=1024
ADD_BOS_FLAG=""
if [[ "$MODEL_NAME" == *"OLMo"* ]]; then
  LOW_CPU_MEM_USAGE=""  # Do not include this flag for OLMo models
  MAX_SEQ_LENGTH=2048
  ADD_BOS_FLAG="--add_bos"  # Add the --add_bos flag for OLMo models
fi

echo "Output directory: $OUTPUT_DIR"
echo "Resume command: $RESUME_CMD"
echo "Max sequence length: $MAX_SEQ_LENGTH"
echo "Low CPU memory usage flag: $LOW_CPU_MEM_USAGE"
echo "Add BOS flag: $ADD_BOS_FLAG"

# Execute the training command
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --main_process_port $PORT \
    --deepspeed_config_file ds_configs/stage3_offloading_accelerate_test.conf \
    open_instruct/finetune.py \
    --model_name_or_path ${MODEL_NAME} \
    $RESUME_CMD \
    --remove_previous_checkpoint \
    --trust_remote_code \
    $LOW_CPU_MEM_USAGE \
    $ADD_BOS_FLAG \
    --use_lora \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout 0.1 \
    --tokenizer_name ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --max_seq_length $MAX_SEQ_LENGTH \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 0 \
    --dataloader_prefetch_factor 0 \
    --checkpointing_steps 500 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate $LR \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir $OUTPUT_DIR \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --seed $SEED

# Example to run the script
# ./scripts/finetune_lora_with_accelerate_slurm.sh 128 256 5e-4 1 9898 t5-v1_1-xxl flan_2022 1 64 false 42
