import random
import datasets as ds


def merge_datasets_with_weights(datasets, N):
    from collections import Counter

    def sample_weighted(dataset, num_samples, weighted_column=None):
        if weighted_column is not None:
            # Calculate the ratio of each unique value in the weighted column
            value_counts = Counter(dataset[weighted_column])
            total_count = sum(value_counts.values())
            weights = {val: count / total_count for val, count in value_counts.items()}

            # Sample according to the weighted_column
            sampled_indices = []
            for val, weight in weights.items():
                val_indices = [
                    i for i, x in enumerate(dataset[weighted_column]) if x == val
                ]
                num_val_samples = round(weight * num_samples)
                sampled_indices.extend(
                    random.sample(val_indices, min(num_val_samples, len(val_indices)))
                )

            return dataset.select(sampled_indices)
        else:
            indices = random.sample(range(len(dataset)), num_samples)
            return dataset.select(indices)

    # Calculate total number of examples in all datasets
    total_examples = sum(len(dataset) for dataset in datasets)

    # Calculate the weight for each dataset
    weights = [len(dataset) / total_examples for dataset in datasets]

    # Calculate number of samples needed from each dataset
    samples_per_dataset = [round(weight * N) for weight in weights]

    # Sample from each dataset while maintaining inner partitioning
    sampled_datasets = []
    for i, dataset in enumerate(datasets):
        print(f"Sampling {samples_per_dataset[i]} examples from dataset {i}...")
        task_sampled = sample_weighted(
            dataset, samples_per_dataset[i], weighted_column="task_name"
        )
        full_sampled = sample_weighted(
            task_sampled, len(task_sampled), weighted_column="template_type"
        )
        sampled_datasets.extend(full_sampled)

    # Combine all sampled datasets
    combined_dataset = ds.from_dict(
        {key: [d[key] for d in sampled_datasets] for key in sampled_datasets[0].keys()}
    )

    return combined_dataset


def load_flan_dataset(cache_dir="huggingface/datasets"):
    flan_2021_dataset = ds.load_dataset(
        "DataProvenanceInitiative/flan2021_submix_original",
        split="train",
        cache_dir=cache_dir,
    )

    niv_dataset = ds.load_dataset(
        "DataProvenanceInitiative/niv2_submix_original",
        split="train",
        cache_dir=cache_dir,
    )
    t0_dataset = ds.load_dataset(
        "DataProvenanceInitiative/t0_submix_original",
        split="train",
        cache_dir=cache_dir,
    )
    cot_dataset = ds.load_dataset(
        "DataProvenanceInitiative/cot_submix_original",
        split="train",
        cache_dir=cache_dir,
    )
    dialog_dataset = ds.load_dataset(
        "DataProvenanceInitiative/dialog_submix_original",
        split="train",
        cache_dir=cache_dir,
    )


    combined_dataset_dataset_dict_path = "flan_2022_350k"

    # merge datasets with weights
    print("Merging datasets with weights...")
    combined_dataset = merge_datasets_with_weights(
        [flan_2021_dataset, niv_dataset, t0_dataset, cot_dataset, dialog_dataset],
        350000,
    )

    # change inputs and targets column to prompt and completion
    combined_dataset = combined_dataset.rename_column("inputs", "prompt")
    combined_dataset = combined_dataset.rename_column("targets", "completion")
    combined_dataset = ds.DatasetDict(
        {"train": combined_dataset}
    )  # to be compatible with the rest of the code

    # print split
    print(f"Total number of examples: {len(combined_dataset)}")
    print(combined_dataset)

    # save to disk as huggingface dataset
    combined_dataset.save_to_disk(dataset_dict_path=combined_dataset_dataset_dict_path)

    return combined_dataset


if __name__ == "__main__":
    load_flan_dataset()
