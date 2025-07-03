#!/bin/bash

# ------------------------------------------------------------------------------
# Usage:
#   submit_jobs.sh [-l] MODEL_DIR [STEPS] 
#
# Examples:
#   # Run on all step_* directories:
#   bash submit_jobs.sh /path/to/model/dir
#
#   # Run on specific steps (e.g., 6000 and 8000):
#   bash submit_jobs.sh /path/to/model/dir 6000,8000
#
#   # Run only on the last (highest-numbered) step:
#   bash submit_jobs.sh -l /path/to/model/dir
#
# ------------------------------------------------------------------------------

# --- Parse the optional -l (last-step-only) flag ---
LAST_ONLY=false
while getopts ":l" opt; do
  case $opt in
    l)
      LAST_ONLY=true
      ;;
    *)
      echo "Usage: submit_jobs.sh [-l] MODEL_DIR [STEPS]"
      exit 1
      ;;
  esac
done

# Shift out the processed options (i.e., -l)
shift $((OPTIND -1))

# --- Positional arguments ---
MODEL_DIR=$1
STEPS=$2

# --- Slurm script file path ---
SLURM_SCRIPT="/home/itay.itzhak/projects/proj2/finetuning/open-instruct/scripts/run_merge_bias.sh"

# --- Check model directory ---
if [ -z "$MODEL_DIR" ]; then
  echo "Error: No model directory provided."
  echo "Usage: submit_jobs.sh [-l] MODEL_DIR [STEPS]"
  exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
  echo "Directory $MODEL_DIR does not exist."
  exit 1
fi

# --- Helper function to submit a job for a given step directory ---
submit_job() {
  STEP_DIR=$1
  if [ -d "$STEP_DIR" ]; then
    echo "Launching Slurm job for $STEP_DIR"
    sbatch --export=MODEL_DIR="$STEP_DIR" "$SLURM_SCRIPT"
    if [ $? -eq 0 ]; then
      echo "Job successfully submitted for $STEP_DIR"
    else
      echo "Failed to submit job for $STEP_DIR"
    fi
  else
    echo "$STEP_DIR does not exist."
  fi
}

# --- If -l (last step) is set, we ignore any comma-separated list ---
if [ "$LAST_ONLY" = true ]; then
  echo "Flag -l detected. Only submitting job for the last (highest-numbered) step."
  
  # Find the highest-numbered step directory
  # 1) List all 'step_*' dirs
  # 2) Extract the numeric suffix
  # 3) Sort them numerically
  # 4) Take the last one
  LATEST_STEP=$(ls -d "${MODEL_DIR}"/step_* 2>/dev/null \
    | sed 's|.*step_\([0-9]\+\).*|\1|' \
    | sort -n \
    | tail -n 1)

  if [ -z "$LATEST_STEP" ]; then
    echo "No 'step_*' directories found in $MODEL_DIR"
    exit 1
  fi

  STEP_DIR="${MODEL_DIR}/step_${LATEST_STEP}"
  submit_job "$STEP_DIR"

# --- Else if a comma-separated list of steps is provided, run only on those ---
elif [ -n "$STEPS" ]; then
  IFS=',' read -r -a STEP_LIST <<< "$STEPS"
  echo "Running on specific steps: ${STEP_LIST[*]}"
  for STEP in "${STEP_LIST[@]}"; do
    STEP_DIR="${MODEL_DIR}/step_${STEP}"
    submit_job "$STEP_DIR"
    sleep 10  # 10 second pause between job submissions
  done

# --- Otherwise, if no steps (and no -l flag), run on all step_* directories ---
else
  echo "No specific steps provided. Running on all 'step_*' directories."
  # Sort directories by extracting numeric suffix
  for STEP_DIR in $(ls -d "${MODEL_DIR}"/step_* 2>/dev/null \
    | sed 's|.*step_\([0-9]\+\).*|\1|' \
    | sort -n); do

    # Reconstruct the directory name from the sorted step number
    FULL_STEP_DIR="${MODEL_DIR}/step_${STEP_DIR}"

    echo "Launching Slurm job for $FULL_STEP_DIR"
    submit_job "$FULL_STEP_DIR"
    sleep 10  # 10 second pause between job submissions
  done
fi
