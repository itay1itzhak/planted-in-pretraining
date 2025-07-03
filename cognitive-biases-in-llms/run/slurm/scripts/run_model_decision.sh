#!/bin/bash
#SBATCH --job-name=cogeval # Job name 
#SBATCH --partition=nlp      # partition
#SBATCH --account=nlp        # account
#SBATCH --mail-user=itay1itzhak@gmail.com    # Where to send mail.  Set this to your email address
#SBATCH --mem-per-cpu=20000mb        # Memory (i.e. RAM) per processor
#SBATCH --time=72:59:59              # Wall time limit (days-hrs:min:sec)
#SBATCH --output=run/slurm/logs/cog_eval_%j.log     # Path to the standard output and error files relative to the working directory 


# Usage: sbatch run/slurm/scripts/run_model_decision.sh --model <MODEL_NAME> [--dataset <DATASET_FILE_PATH>] [--batches <BATCHES>] [--delete-merged <true|false>]
# Example: sbatch run/slurm/scripts/run_model_decision.sh --model T5TuluSeed0 --batches 3 --delete-merged true

# Default values
DEFAULT_DATASET_FILE_PATH="~/projects/proj2/finetuning/cognitive-biases-in-llms/data/full_dataset.csv"
DEFAULT_BATCHES=1
DELETE_MERGED=false

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL_NAME="$2"; shift ;;
        --dataset) DATASET_FILE_PATH="$2"; shift ;;
        --batches) BATCHES="$2"; shift ;;
        --delete-merged) DELETE_MERGED="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check for required arguments
if [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 --model <MODEL_NAME> [--dataset <DATASET_FILE_PATH>] [--batches <BATCHES>] [--delete-merged <true|false>]"
    exit 1
fi

# Use defaults if arguments are not provided
DATASET_FILE_PATH=${DATASET_FILE_PATH:-$DEFAULT_DATASET_FILE_PATH}
BATCHES=${BATCHES:-$DEFAULT_BATCHES}

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Gpus Allocated       = $SLURM_GPUS_PER_TASK"

echo "Running for model $MODEL_NAME and dataset $DATASET_FILE_PATH with $BATCHES batches"

# Models path main dir
MODELS_MAIN_PATH="/home/itay.itzhak/projects/proj2/finetuning/open-instruct/output"
# Cog eval path
COG_EVAL_PATH="/home/itay.itzhak/projects/proj2/finetuning/cognitive-biases-in-llms"

# Set MODEL_PATH based on the MODEL_NAME
NEED_MERGE=true
# If OLMo-Flan 
if [[ "${MODEL_NAME}" == *"OLMo-Flan-Seed-1"* ]]; then
    MODEL_PATH="${MODELS_MAIN_PATH}/flan_2022_allenai/OLMo-7B_lora_r128_alpha256_LR2e-5_seed_1"
elif [[ "${MODEL_NAME}" == *"OLMo-Flan-Seed-2"* ]]; then
    MODEL_PATH="${MODELS_MAIN_PATH}/flan_2022_allenai/OLMo-7B_lora_r128_alpha256_LR2e-5_seed_2"
elif [[ "${MODEL_NAME}" == *"OLMo-Flan-Seed-0"* ]]; then
    MODEL_PATH="${MODELS_MAIN_PATH}/flan_v2_OLMo-7B_lora_r128_alpha256_LR2e-5"
# If T5-Tulu
elif [[ "${MODEL_NAME}" == *"T5-Tulu-Seed-0"* ]]; then
    MODEL_PATH="${MODELS_MAIN_PATH}/allenai/tulu-v2-sft-mixture_t5-v1_1-xxl_lora_r128_alpha256_LR1e-4"
elif [[ "${MODEL_NAME}" == *"T5-Tulu-Seed-1"* ]]; then
    MODEL_PATH="${MODELS_MAIN_PATH}/allenai/tulu-v2-sft-mixture_t5-v1_1-xxl_lora_r128_alpha256_LR1e-4_seed_1"
elif [[ "${MODEL_NAME}" == *"T5-Tulu-Seed-2"* ]]; then
    MODEL_PATH="${MODELS_MAIN_PATH}/allenai/tulu-v2-sft-mixture_t5-v1_1-xxl_lora_r128_alpha256_LR1e-4_seed_2"
# If OLMo-Tulu
elif [[ "${MODEL_NAME}" == *"OLMo-Tulu-Seed-1"* ]]; then
    MODEL_PATH="${MODELS_MAIN_PATH}/allenai/tulu-v2-sft-mixture_allenai/OLMo-7B_lora_r128_alpha256_LR2e-5_seed_1"
elif [[ "${MODEL_NAME}" == *"OLMo-Tulu-Seed-2"* ]]; then
    MODEL_PATH="${MODELS_MAIN_PATH}/allenai/tulu-v2-sft-mixture_allenai/OLMo-7B_lora_r128_alpha256_LR2e-5_seed_2"
elif [[ "${MODEL_NAME}" == *"OLMo-Tulu-Seed-0"* ]]; then
    MODEL_PATH="${MODELS_MAIN_PATH}/tulu_v2_OLMo-7B_lora_r128_alpha256_LR2e-5"
# If T5-Flan
elif [[ "${MODEL_NAME}" == *"T5-Flan-Seed-1"* ]]; then
    MODEL_PATH="${MODELS_MAIN_PATH}/flan_2022_google/t5-v1_1-xxl_lora_r128_alpha256_LR1e-4_seed_1"
elif [[ "${MODEL_NAME}" == *"T5-Flan-Seed-2"* ]]; then
    MODEL_PATH="${MODELS_MAIN_PATH}/flan_2022_google/t5-v1_1-xxl_lora_r128_alpha256_LR1e-4_seed_2"
elif [[ "${MODEL_NAME}" == *"T5-Flan-Seed-0"* ]]; then
    MODEL_PATH="${MODELS_MAIN_PATH}/flan_v2_t5-v1_1-xxl_lora_r128_alpha256_LR1e-4"
# Models that don't need to be merged
elif [[ "${MODEL_NAME}" == *"OLMo-SFT"* ]]; then
    echo "OLMo-SFT does not need to be merged"
    NEED_MERGE=false
elif [[ "${MODEL_NAME}" == *"Flan-T5"* ]]; then
    echo "Flan-T5 does not need to be merged"
    NEED_MERGE=false
elif [[ "${MODEL_NAME}" == *"Mistral"* ]]; then
    echo "Mistral does not need to be merged"
    NEED_MERGE=false
elif [[ "${MODEL_NAME}" == *"Llama2"* ]]; then
    echo "Llama2 does not need to be merged"
    NEED_MERGE=false
else
    echo "Unknown model in MODEL_NAME. Please check the directory."
    exit 1
fi

echo "MODEL_NAME set to $MODEL_NAME"

# If model needs to be merged, check if the model-000* files exist in MODEL_PATH
if [[ "$NEED_MERGE" == "true" ]]; then
    echo "MODEL_PATH set to $MODEL_PATH"
    if ls ${MODEL_PATH}/merged/model-000* 1> /dev/null 2>&1; then
        echo "Found existing merged model files in ${MODEL_PATH}"
        # Add merge dir to MODEL_PATH
        MODEL_PATH="${MODEL_PATH}/merged"
        echo "MODEL_PATH set to $MODEL_PATH"
    else
        echo "No merged model files found in ${MODEL_PATH}/merged, proceeding with merge"
        echo "Running merge_lora.py with opinst5 environment"
        cd ~/projects/proj2/finetuning/open-instruct

        # Set BASE_MODEL based on the MODEL_PATH
        if [[ "${MODEL_PATH}" == *"t5"* ]]; then
            BASE_MODEL="google/t5-v1_1-xxl"
        elif [[ "${MODEL_PATH}" == *"OLMo"* ]]; then
            BASE_MODEL="allenai/OLMo-7B"
        else
            echo "Unknown model type in MODEL_PATH ${MODEL_PATH}. Please check the directory."
            exit 1
        fi

        echo "/home/itay.itzhak/miniconda3/envs/opinst5/bin/python open_instruct/merge_lora.py \
            --base_model_name_or_path ${BASE_MODEL} \
            --lora_model_name_or_path ${MODEL_PATH} \
            --output_dir ${MODEL_PATH}/merged \
            --save_tokenizer \
            --use_fast_tokenizer"

        /home/itay.itzhak/miniconda3/envs/opinst5/bin/python open_instruct/merge_lora.py \
            --base_model_name_or_path ${BASE_MODEL} \
            --lora_model_name_or_path ${MODEL_PATH} \
            --output_dir ${MODEL_PATH}/merged \
            --save_tokenizer \
            --use_fast_tokenizer
    fi
fi

# Check if results dir exists and if csv files exist in results dir ~/projects/proj2/finetuning/cognitive-biases-in-llms/data/decision_results/${MODEL_NAME}/
echo "Checking if results dir exists and if csv files exist in results dir ${COG_EVAL_PATH}/data/decision_results/${MODEL_NAME}/"
if [ -d ${COG_EVAL_PATH}/data/decision_results/${MODEL_NAME}/ ]; then
    echo "Results dir exists"
    # If any .csv files exist in the results dir, move them to a new subdir named old_results
    if ls ${COG_EVAL_PATH}/data/decision_results/${MODEL_NAME}/*.csv 1> /dev/null 2>&1; then
        echo "CSV files exist"
        # Move old files to a new subdir named old_results
        # check if old_results dir exists
        if [ ! -d ${COG_EVAL_PATH}/data/decision_results/${MODEL_NAME}/old_results ]; then
            mkdir -p ${COG_EVAL_PATH}/data/decision_results/${MODEL_NAME}/old_results
        fi
        # Go over all csv files in the results dir and move them to old_results
        for file in ${COG_EVAL_PATH}/data/decision_results/${MODEL_NAME}/*.csv; do
            mv $file ${COG_EVAL_PATH}/data/decision_results/${MODEL_NAME}/old_results/
        done
    else
        echo "No CSV files found in the results dir, proceeding with merge"
    fi
fi


echo "Running test_decision.py for $MODEL_NAME"

cd ~/projects/proj2/finetuning/cognitive-biases-in-llms

echo "/home/itay.itzhak/miniconda3/envs/cogeval/bin/python ${COG_EVAL_PATH}/run/test_decision.py \
    --model ${MODEL_NAME} \
    --n_workers 1 \
    --n_batches ${BATCHES} \
    --dataset ${DATASET_FILE_PATH}"

/home/itay.itzhak/miniconda3/envs/cogeval/bin/python ${COG_EVAL_PATH}/run/test_decision.py \
    --model ${MODEL_NAME} \
    --n_workers 1 \
    --n_batches ${BATCHES} \
    --dataset ${DATASET_FILE_PATH}

# Clean up merged models if DELETE_MERGED is true
if [[ "$DELETE_MERGED" == "true" ]]; then
    echo "Cleaning up merged models from ${MODEL_PATH}/ ..."
    rm -f ${MODEL_PATH}/model-000*
fi

echo "Job completed at $(date)"
