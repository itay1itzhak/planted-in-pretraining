#!/bin/bash

# This bash script is a wrapper for the run_model_decision.sh script.
# It is used to run the run_model_decision.sh script on the slurm cluster while choosing the number of GPUs and the number of cores per task.

# Usage: ./run/slurm/scripts/slurm_wrapper_run_model_decision.sh --gpus <NUM_GPUS> --model <MODEL_NAME> [--dataset <DATASET_FILE_PATH>] [--batches <BATCHES>] [--delete-merged <true|false>]
# Example: ./run/slurm/scripts/slurm_wrapper_run_model_decision.sh --gpus 1 --model T5TuluSeed0 --batches 3 --delete-merged true

# Default values
NUM_GPUS=1

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpus) NUM_GPUS="$2"; shift ;;
        --model) MODEL_NAME="$2"; shift ;;
        --dataset) DATASET_FILE_PATH="$2"; shift ;;
        --batches) BATCHES="$2"; shift ;;
        --delete-merged) DELETE_MERGED="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure required arguments are provided
if [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 --gpus <NUM_GPUS> --model <MODEL_NAME> [--dataset <DATASET_FILE_PATH>] [--batches <BATCHES>] [--delete-merged <true|false>]"
    exit 1
fi

# To supress /bin/bash: /home/itay.itzhak/miniconda3/envs/cogeval/lib/libtinfo.so.6: no version information available (required by /bin/bash) warning
export LD_PRELOAD=/home/itay.itzhak/miniconda3/envs/cogeval/lib/libtinfo.so.6
# To solve the error only on slurm: 
# ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /home/itay.itzhak/miniconda3/envs/cogeval/lib/python3.10/site-packages/pandas/_libs/window/aggregations.cpython-310-x86_64-linux-gnu.so)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/itay.itzhak/miniconda3/envs/cogeval/lib

# Submit the job with dynamic GPU and task allocation
sbatch --gres=gpu:6000ADA:$NUM_GPUS --ntasks=$NUM_GPUS --export=ALL run/slurm/scripts/run_model_decision.sh \
    --model "$MODEL_NAME" \
    ${DATASET_FILE_PATH:+--dataset "$DATASET_FILE_PATH"} \
    ${BATCHES:+--batches "$BATCHES"} \
    ${DELETE_MERGED:+--delete-merged "$DELETE_MERGED"}
