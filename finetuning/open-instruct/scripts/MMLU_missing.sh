#!/bin/bash

# Usage: ./submit_missing_jobs.sh [-l] [-r start:end] main_dir model_name_1 model_name_2 ...
# -l : Flag to only submit jobs for the last step of each model
# -r start:end : Only process steps within the inclusive numeric range [start, end]

only_last=false
step_range=""
start=""
end=""

# Parse the flags
while getopts ":lr:" opt; do
  case $opt in
    l)
      only_last=true
      ;;
    r)
      step_range="$OPTARG"
      IFS=':' read -r start end <<< "$step_range"
      # Validate that start and end are integers
      if ! [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ ]]; then
        echo "Invalid range format. Use -r start:end where start and end are integers."
        exit 1
      fi
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

shift $((OPTIND - 1))

main_dir=$1
shift
model_names=("$@")

for model_name in "${model_names[@]}"; do
    model_dir="${main_dir}/${model_name}"
    
    if [ -d "$model_dir" ]; then
        # Read all step directories into an associative array
        unset step_dirs
        declare -A step_dirs

        for dir in "${model_dir}"/step_*; do
            if [ -d "$dir" ]; then
                # Extract step number from directory name
                step_number=$(echo "$dir" | sed 's/.*step_//')
                step_dirs[$step_number]="$dir"
            fi
        done

        # Sort the step keys numerically
        sorted_keys=($(echo "${!step_dirs[@]}" | tr ' ' '\n' | sort -n))

        # If a step range is specified, filter the keys to that range
        if [ -n "$step_range" ]; then
            filtered_keys=()
            for key in "${sorted_keys[@]}"; do
                if [[ "$key" -ge "$start" && "$key" -le "$end" ]]; then
                    filtered_keys+=("$key")
                fi
            done
            sorted_keys=("${filtered_keys[@]}")
        fi

        # If only_last is specified, keep only the last step directory after filtering
        if [ "$only_last" = true ] && [ "${#sorted_keys[@]}" -gt 0 ]; then
            last_key="${sorted_keys[-1]}"
            sorted_keys=("$last_key")
        fi

        # Prepare list of directories for submission
        dirs_to_submit=()
        for step_key in "${sorted_keys[@]}"; do
            step_dir="${step_dirs[$step_key]}"
            metrics_file_1="${step_dir}/merged/mmlu_5_shot/metrics_merged.json"
            metrics_file_2="${step_dir}/merged/mmlu/metrics_merged.json"

            # Check if either of the metrics files exists
            if [ ! -f "$metrics_file_1" ] && [ ! -f "$metrics_file_2" ]; then
                dirs_to_submit+=("$step_dir")
            else
                echo "Skipping step: $step_key (metrics file exists in one of the paths)"
            fi
        done

        # Print directories to be submitted
        if [ "${#dirs_to_submit[@]}" -gt 0 ]; then
            echo "Directories to submit for model ${model_name}:"
            for dir in "${dirs_to_submit[@]}"; do
                echo "  $dir"
            done
        else
            echo "No directories to submit for model ${model_name}."
        fi

        # Submit jobs for each directory
        for step_dir in "${dirs_to_submit[@]}"; do
            echo "Submitting job for: $step_dir"
            if [[ "$model_name" == *"flan"* ]]; then
                if [[ "$model_name" == *"OLMo"* ]]; then
                    sbatch scripts/slurm_olmo_flan_mmlu_and_merge.sh "$step_dir"
                else
                    sbatch scripts/slurm_t5_flan_mmlu_and_merge.sh "$step_dir"
                fi
            else
                if [[ "$model_name" == *"OLMo"* ]]; then
                    sbatch scripts/slurm_olmo_tulu_mmlu_and_merge.sh "$step_dir"
                else
                    sbatch scripts/slurm_t5_tulu_mmlu_and_merge.sh "$step_dir"
                fi
            fi
            sleep 10  # Add a 10 second pause between job submissions
        done
    fi
done
