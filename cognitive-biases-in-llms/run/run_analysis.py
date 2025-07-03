"""
Main script for running model bias analysis and clustering.
This script performs various clustering analyses on model bias data,
including hierarchical, supervised, and unsupervised clustering.
"""

import json
import os

import numpy as np
import pandas as pd
import sys
sys.path.append('..')
import similarity_analysis_copy_32_2050 as analysis
import argparse
from sklearn.metrics import confusion_matrix

# Constants
MODELS_TO_INCLUDE = [
    "Flan-T5",
    "T5-Flan-Seed-2",
    "T5-Flan-Seed-1",
    "T5-Flan-Seed-0",
    "T5-Tulu-Seed-0",
    "T5-Tulu-Seed-1",
    "T5-Tulu-Seed-2",
    # "Random",
    "OLMo-Flan-Seed-0",
    "OLMo-Flan-Seed-1",
    "OLMo-Flan-Seed-2",
    "OLMo-Tulu-Seed-0",
    "OLMo-Tulu-Seed-1",
    "OLMo-Tulu-Seed-2",
    "OLMo-SFT",
    "Mistral-Tulu-Seed-0",
    "Mistral-Tulu-Seed-1",
    "Mistral-ShareGPT",
    "Llama2-Tulu",
    "Llama2-ShareGPT-Seed-0",
    "Llama2-ShareGPT-Seed-1",
    # "Llama2-ShareGPT-Seed-2",
]


def setup_environment(output_dir):
    """Create necessary directories for output."""
    os.makedirs(output_dir, exist_ok=True)


def run_analysis_load_and_prepare_data(
    exclude_random=True,
    error_threshold=None,
    handle_missing_values="impute",
    models_to_include=MODELS_TO_INCLUDE,
):
    """
    Load and prepare the decision and bias data for analysis.

    Returns:
        tuple: (df_biasedness, df_decisions) containing the prepared dataframes
    """
    # Load decision data
    df_decisions = analysis.load_decision_data(models_to_include=models_to_include)
    print("Decision data shape:", df_decisions.shape)

    # Prepare bias data (excluding Random model and filtering by error threshold)
    if exclude_random:
        if "Random" in df_decisions["model"].unique():
            df_decisions = df_decisions[df_decisions["model"] != "Random"].copy()
    if error_threshold is not None:
        df_decisions = analysis.filter_biases_by_errors(
            df_decisions, error_threshold=error_threshold
        )

    # Load bias data
    df_biasedness = analysis.load_model_bias_data(df_decisions=df_decisions)
    print("Bias data shape:", df_biasedness.shape)

    # Impute missing values
    if handle_missing_values == "impute":
        print("=" * 10 + "WARNING" + "=" * 10)
        print(
            "Imputing missing values (replacing all NaN values with the mean bias score over all models and biases)..."
        )
        df_biasedness = analysis.impute_missing_values(df_biasedness)
    # elif handle_missing_values == 'impute_bias_level':
    #     print("Imputing missing values (replacing all NaN values with the mean bias score over all models and biases)...")
    #     df_biasedness = analysis.impute_missing_values_bias_level(df_biasedness)
    elif handle_missing_values == "remove":
        print("Removing missing values...")
        df_biasedness = df_biasedness.dropna()
    else:
        df_biasedness = analysis.impute_missing_values_granularity_level(
            df_biasedness, impute_granularity_level=handle_missing_values
        )

    return df_biasedness


def run_hierarchical_clustering(
    df_biasedness,
    granularity_level="model_bias",
    figsize=(8, 6),
    output_dir="./plots/clustering",
    debug=False,
):
    """Run hierarchical clustering analysis."""
    cluster_data = analysis.run_hierarchical_clustering_analysis(
        df_biasedness, level=granularity_level, debug=debug
    )

    analysis.plot_clustering_analysis(
        cluster_data,
        methods=["complete", "ward"],
        metrics=["euclidean", "cosine"],
        n_clusters=12,
        save_plots=True,
        figsize=figsize,
        granularity_level=granularity_level,
        output_dir=output_dir,
        debug=debug,
    )


def run_both_clustering_analyses(
    df_biasedness,
    granularity_level,
    with_scaling,
    n_clusters,
    random_runs,
    seed,
    debug,
    output_dir,
    add_certainty_and_belief,
    n_kmeans_trials,
    models_to_include,
):
    """Run both supervised and unsupervised clustering analyses."""
    # Prepare feature matrix
    feature_matrix, model_metadata = analysis.prepare_clustering_data(
        df_biasedness,
        level=granularity_level,
        debug=debug,
        add_certainty_and_belief=add_certainty_and_belief,
    )

    if debug:
        # Add debugging information
        print("\nClustering Analysis Debug Info:")
        print(f"Feature matrix shape: {feature_matrix.shape}")
        print(
            f"Number of unique pretraining groups: {len(model_metadata['pretraining_group'].unique())}"
        )
        print(
            f"Number of unique instruction groups: {len(model_metadata['instruction_group'].unique())}"
        )
        print("\nModel Groups:")
        print(
            model_metadata[
                ["model", "pretraining_group", "instruction_group"]
            ].to_string()
        )

    # Unsupervised clustering
    unsupervised_clustering_results = analysis.perform_unsupervised_clustering_analysis(
        feature_matrix,
        model_metadata,
        seed=seed,
        n_clusters=n_clusters,  # Make sure we use the same number of clusters
        with_scaling=with_scaling,
        debug=debug,
        n_kmeans_trials=n_kmeans_trials,
    )

    # Unsupervised clustering with k=3 to compare with model developer group
    if "Llama2" in models_to_include and "Mistral" in models_to_include:
        unsupervised_clustering_results_k3 = (
            analysis.perform_unsupervised_clustering_analysis(
                feature_matrix,
                model_metadata,
                seed=seed,
                n_clusters=3,  # To compare with the three model developer groups
                with_scaling=with_scaling,
                debug=debug,
                n_kmeans_trials=n_kmeans_trials,
            )
        )

    # Supervised clustering
    supervised_clustering_results = analysis.perform_supervised_clustering_analysis(
        feature_matrix,
        model_metadata,
        seed=seed,
        n_clusters=n_clusters,  # Pass the same number of clusters
        num_random_assignment=random_runs,
        with_scaling=with_scaling,
        debug=debug,
        models_to_include=models_to_include,
    )

    # Analyze label agreement
    agreement_scores = analysis.analyze_label_agreement(
        unsupervised_clustering_results["best"]["labels"],
        supervised_clustering_results,
        debug=debug,
    )

    if debug:
        print("\nClustering Results Comparison:")
        print("\nCluster Sizes:")
        print(
            "Unsupervised (best):",
            np.bincount(unsupervised_clustering_results["best"]["labels"]),
        )
        print(
            "Unsupervised (median):",
            np.bincount(unsupervised_clustering_results["median"]["labels"]),
        )
        print(
            "Pretraining:",
            np.bincount(supervised_clustering_results["pretraining"]["labels"]),
        )
        print(
            "Instruction:",
            np.bincount(supervised_clustering_results["instruction"]["labels"]),
        )

        print("\nLabels Comparison:")
        labels_df = pd.DataFrame(
            {
                "Model": model_metadata["model"],
                "Unsupervised (best)": unsupervised_clustering_results["best"][
                    "labels"
                ],
                "Unsupervised (median)": unsupervised_clustering_results["median"][
                    "labels"
                ],
                "Pretraining": supervised_clustering_results["pretraining"]["labels"],
                "Instruction": supervised_clustering_results["instruction"]["labels"],
            }
        )
        print(labels_df.to_string())

        print("\nConfusion Matrix (Unsupervised vs Pretraining):")
        print(
            confusion_matrix(
                supervised_clustering_results["pretraining"]["labels"],
                unsupervised_clustering_results["best"]["labels"],
            )
        )

        print("\nMetrics Comparison:")
        metrics = ["silhouette", "calinski_harabasz", "davies_bouldin"]
        results = {
            "Unsupervised (best)": unsupervised_clustering_results["best"]["metrics"],
            "Unsupervised (median)": unsupervised_clustering_results["median"][
                "metrics"
            ],
            "Pretraining": supervised_clustering_results["pretraining"]["metrics"],
            "Instruction": supervised_clustering_results["instruction"]["metrics"],
            "Random (mean)": supervised_clustering_results["random_mean"]["metrics"],
        }
        if "Llama2" in models_to_include and "Mistral" in models_to_include:
            results["Model Developer"] = {
                "Supervised Model Developer": supervised_clustering_results[
                    "model_developer"
                ]["metrics"],
                "Unsupervised K=3 (best)": unsupervised_clustering_results_k3["best"][
                    "metrics"
                ],
                "Unsupervised K=3 (median)": unsupervised_clustering_results_k3[
                    "median"
                ]["metrics"],
            }

        comparison_df = pd.DataFrame(results).round(3)
        print(comparison_df)

    # Add statistical validation
    validation_results = analysis.validate_clustering_labels(
        feature_matrix,
        model_metadata,
        unsupervised_best_results=unsupervised_clustering_results["best"],
        unsupervised_median_results=unsupervised_clustering_results["median"],
        supervised_results=supervised_clustering_results,
        debug=debug,
    )

    if debug:
        print("\nStatistical Validation Results:")
        print("-" * 30)
        for label_type in ["pretraining", "instruction"]:
            print(f"\n{label_type.capitalize()} Labels:")
            results = validation_results[label_type]

            print("\nPermutation Tests:")
            for metric, test_results in results["permutation_tests"].items():
                print(f"{metric}:")
                print(f"  True score: {test_results['true_score']:.3f}")
                print(
                    f"  Mean permuted: {test_results['mean_permuted']:.3f} (±{test_results['std_permuted']:.3f})"
                )
                print(f"  p-value: {test_results['p_value']:.4f}")

            print("\nMANOVA Test:")
            print(f"Mean F-statistic: {results['manova']['f_statistics']:.3f}")
            print(f"Mean p-value: {results['manova']['p_values']:.4f}")
            print(f"Significant dimensions: {results['manova']['significant_dims']}")

            print("\nDistance Distribution Test:")
            dist_results = results["distance_test"]
            print(f"KS statistic: {dist_results['ks_statistic']:.3f}")
            print(f"p-value: {dist_results['p_value']:.4f}")
            print(
                f"Mean intra-cluster distance: {dist_results['mean_intra_dist']:.3f} (±{dist_results['std_intra_dist']:.3f})"
            )
            print(
                f"Mean inter-cluster distance: {dist_results['mean_inter_dist']:.3f} (±{dist_results['std_inter_dist']:.3f})"
            )

    return (
        feature_matrix,
        model_metadata,
        unsupervised_clustering_results,
        supervised_clustering_results,
        agreement_scores,
        validation_results,
    )


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run model bias analysis and clustering."
    )

    # Analysis configuration
    parser.add_argument(
        "--granularity-levels",
        nargs="+",
        choices=["model_bias", "model_bias_scenario", "model_bias_sample"],
        default=["model_bias"],
        help="Levels of analysis granularity to run (can specify multiple)",
    )
    # add output dir
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./plots/clustering",
        help="Output directory for clustering results",
    )
    parser.add_argument(
        "--models-to-include",
        type=str,
        default=None,
        help='Models families (e.g. "T5", "Llama2") to include in the analysis (comma-separated list)',
    )

    # Feature processing
    parser.add_argument(
        "--with-scaling",
        action="store_true",
        default=True,
        help="Whether to apply feature scaling",
    )
    parser.add_argument(
        "--no-scaling",
        action="store_false",
        dest="with_scaling",
        help="Disable feature scaling",
    )
    parser.add_argument(
        "--exclude-random",
        action="store_true",
        default=True,
        help="Whether to exclude the Random model from analysis",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for clustering"
    )
    # Add error threshold
    parser.add_argument(
        "--error-threshold",
        type=int,
        default=None,
        help="Error threshold for filtering biases",
    )
    # Choose how to handle missing values
    parser.add_argument(
        "--handle-missing-values",
        type=str,
        default="impute",
        help="How to handle missing values",
    )
    parser.add_argument(
        "--add-certainty-and-belief",
        action="store_true",
        default=False,
        help="Whether to add certainty and belief columns to the feature matrix from manually added rows",
    )

    # Clustering parameters
    parser.add_argument(
        "--n-clusters-unsupervised",
        type=int,
        default=2,
        help="Number of clusters for k-means",
    )
    parser.add_argument(
        "--random-runs",
        type=int,
        default=5,
        help="Number of random clustering runs for comparison",
    )
    parser.add_argument(
        "--n-kmeans-trials",
        type=int,
        default=30,
        help="Number of k-means trials for each number of clusters",
    )
    # Visualization
    parser.add_argument(
        "--save-plots", action="store_true", default=True, help="Save generated plots"
    )
    parser.add_argument(
        "--dot-size", type=float, default=50, help="Size of dots in scatter plots"
    )
    parser.add_argument(
        "--dot-alpha",
        type=float,
        default=1.0,
        help="Transparency of dots in scatter plots",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(8, 6),
        help="Figure size as width height",
    )

    # Debug
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    return parser.parse_args()


def convert_to_serializable(obj):
    """
    Convert complex objects (numpy arrays, etc.) to JSON-serializable format.

    Args:
        obj: Object to convert

    Returns:
        JSON serializable version of the object
    """
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return convert_to_serializable(obj.to_dict())
    elif isinstance(obj, pd.Series):
        return convert_to_serializable(obj.to_dict())
    return obj


def save_model_labels(results, filepath, debug=False):
    """
    Save model labels to a separate CSV file.
    """
    if debug:
        print("\nSaving model labels...")

    rows = []
    for result in results:
        granularity = result["granularity_level"]
        models = result["model_names"]

        # Create rows for each model
        for model in models:
            row = {
                "granularity_level": granularity,
                "model": model,
                "unsupervised_best_label": result["unsupervised_clustering_results"][
                    "best"
                ]["labels"][models.index(model)],
                "unsupervised_median_label": result["unsupervised_clustering_results"][
                    "median"
                ]["labels"][models.index(model)],
                "pretraining_label": result["supervised_clustering_results"][
                    "pretraining"
                ]["labels"][models.index(model)],
                "instruction_label": result["supervised_clustering_results"][
                    "instruction"
                ]["labels"][models.index(model)],
            }

            # Add random run labels
            for run_id, run_data in result["supervised_clustering_results"][
                "random_runs"
            ].items():
                row[f"random_run_{run_id}"] = run_data["labels"][models.index(model)]

            rows.append(row)

    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(filepath + "_labels.csv", index=False)


def save_clustering_results(results, filepath, is_csv=False, debug=False):
    """
    Save clustering results in a compact, readable format with validation p-values.
    """
    if debug:
        print("\nSaving clustering results...")

    # Convert results to serializable format first
    serializable_results = convert_to_serializable(results)

    # Create a more compact, organized structure
    formatted_results = []
    for result in serializable_results:
        granularity = result["granularity_level"]

        # Extract validation results for easier access
        validation = result["validation_results"]

        # Create the formatted result structure
        formatted_result = {
            "granularity_level": granularity,
            "unsupervised": {
                "best": {
                    "metrics": {
                        metric: {
                            "value": result["unsupervised_clustering_results"]["best"][
                                "metrics"
                            ][metric],
                            "p_value": (
                                validation["pretraining"]["permutation_tests"][metric][
                                    "p_value"
                                ]
                                if "permutation_tests" in validation["pretraining"]
                                and metric
                                in validation["pretraining"]["permutation_tests"]
                                else None
                            ),
                        }
                        for metric in result["unsupervised_clustering_results"]["best"][
                            "metrics"
                        ]
                    },
                    "n_clusters_best": len(
                        set(result["unsupervised_clustering_results"]["best"]["labels"])
                    ),
                },
                "median": {
                    "metrics": {
                        metric: {
                            "value": result["unsupervised_clustering_results"][
                                "median"
                            ]["metrics"][metric],
                            "p_value": (
                                validation["pretraining"]["permutation_tests"][metric][
                                    "p_value"
                                ]
                                if "permutation_tests" in validation["pretraining"]
                                and metric
                                in validation["pretraining"]["permutation_tests"]
                                else None
                            ),
                        }
                        for metric in result["unsupervised_clustering_results"][
                            "median"
                        ]["metrics"]
                    },
                    "n_clusters_median": len(
                        set(
                            result["unsupervised_clustering_results"]["median"][
                                "labels"
                            ]
                        )
                    ),
                },
            },
            "supervised": {
                "pretraining": {
                    "metrics": {
                        metric: {
                            "value": result["supervised_clustering_results"][
                                "pretraining"
                            ]["metrics"][metric],
                            "p_value": (
                                validation["pretraining"]["permutation_tests"][metric][
                                    "p_value"
                                ]
                                if "permutation_tests" in validation["pretraining"]
                                and metric
                                in validation["pretraining"]["permutation_tests"]
                                else None
                            ),
                        }
                        for metric in result["supervised_clustering_results"][
                            "pretraining"
                        ]["metrics"]
                    },
                    "manova": validation["pretraining"]["manova"],
                },
                "instruction": {
                    "metrics": {
                        metric: {
                            "value": result["supervised_clustering_results"][
                                "instruction"
                            ]["metrics"][metric],
                            "p_value": (
                                validation["instruction"]["permutation_tests"][metric][
                                    "p_value"
                                ]
                                if "permutation_tests" in validation["instruction"]
                                and metric
                                in validation["instruction"]["permutation_tests"]
                                else None
                            ),
                        }
                        for metric in result["supervised_clustering_results"][
                            "instruction"
                        ]["metrics"]
                    },
                    "manova": validation["instruction"]["manova"],
                },
                "random_mean": {
                    "metrics": result["supervised_clustering_results"]["random_mean"][
                        "metrics"
                    ]
                },
            },
            "label_agreement": {
                "pretraining": result["agreement_scores"]["pretraining"],
                "instruction": result["agreement_scores"]["instruction"],
                "random_mean": result["agreement_scores"]["random_mean"],
            },
            "distance_tests": {
                "best": validation["best"]["distance_test"],
                "median": validation["median"]["distance_test"],
                "pretraining": validation["pretraining"]["distance_test"],
                "instruction": validation["instruction"]["distance_test"],
                "random_mean": validation["random_mean"]["distance_test"],
            },
        }
        # TODO: Add model developer results if available

        formatted_results.append(formatted_result)

    # Save main results to JSON
    with open(filepath + ".json", "w") as f:
        json.dump(formatted_results, f, indent=2)

    if is_csv:
        # Convert to flat DataFrame for CSV
        rows = []
        for result in formatted_results:
            base_row = {"granularity_level": result["granularity_level"]}

            # Add metrics with p-values
            for cluster_type in ["unsupervised", "pretraining", "instruction"]:
                metrics = (
                    result["unsupervised"]["best"]["metrics"]
                    if cluster_type == "unsupervised"
                    else result["supervised"][cluster_type]["metrics"]
                )

                for metric_name, metric_data in metrics.items():
                    base_row[f"{cluster_type}_{metric_name}"] = metric_data["value"]
                    base_row[f"{cluster_type}_{metric_name}_p_value"] = metric_data.get(
                        "p_value"
                    )

            # Add MANOVA results
            for group in ["pretraining", "instruction"]:
                manova = result["supervised"][group]["manova"]
                base_row[f"{group}_manova_f_stat"] = manova["f_statistics"]
                base_row[f"{group}_manova_p_value"] = manova["p_values"]

            # Add distance test results
            for group in ["pretraining", "instruction"]:
                dist_test = result["distance_tests"][group]
                base_row[f"{group}_ks_statistic"] = dist_test["ks_statistic"]
                base_row[f"{group}_ks_p_value"] = dist_test["p_value"]

            rows.append(base_row)

        df = pd.DataFrame(rows)
        df.to_csv(filepath + ".csv", index=False)

    # Save model labels separately
    save_model_labels(serializable_results, filepath, debug)


def main():
    """Main function to run the analysis pipeline."""
    # Parse arguments
    args = parse_args()

    # Setup
    output_dir = (
        args.output_dir
        + "/with_scaling_"
        + str(args.with_scaling)
        + "/error_threshold_"
        + str(args.error_threshold)
        + "/handle_missing_values_"
        + str(args.handle_missing_values)
    )
    if args.models_to_include:
        output_dir += "_models_" + args.models_to_include.replace(",", "_")
        # Filter out models that do not contain one of the strings in the list
        models_to_include = [
            model
            for model in MODELS_TO_INCLUDE
            if any(
                model_family in model for model_family in models_to_include.split(",")
            )
        ]
    else:
        models_to_include = MODELS_TO_INCLUDE
    setup_environment(output_dir)
    # set seed
    np.random.seed(args.seed)

    # Load and prepare data
    df_biasedness = run_analysis_load_and_prepare_data(
        exclude_random=args.exclude_random,
        error_threshold=args.error_threshold,
        handle_missing_values=args.handle_missing_values,
        models_to_include=models_to_include,
    )

    all_clustering_results = []

    for granularity_level in args.granularity_levels:
        print("=" * 100)
        print(f"       Running clustering analyses at {granularity_level} level")
        print("=" * 100)

        # Run hierarchical clustering
        # print("Running hierarchical clustering...")
        # run_hierarchical_clustering(
        #     df_biasedness,
        #     granularity_level=granularity_level,
        #     figsize=tuple(args.figsize),
        #     output_dir=output_dir,
        #     debug=args.debug
        # )

        # Run other clustering analyses
        print("-" * 50)
        print("Running clustering analyses...")
        print("-" * 50)
        (
            feature_matrix,
            model_metadata,
            unsupervised_results,
            supervised_results,
            agreement_scores,
            validation_results,
        ) = run_both_clustering_analyses(
            df_biasedness,
            granularity_level=granularity_level,
            with_scaling=args.with_scaling,
            n_clusters=args.n_clusters_unsupervised,
            random_runs=args.random_runs,
            seed=args.seed,
            output_dir=output_dir,
            debug=args.debug,
            add_certainty_and_belief=args.add_certainty_and_belief,
            n_kmeans_trials=args.n_kmeans_trials,
            models_to_include=models_to_include,
        )

        # Compute similarity statistics
        # Prepare feature matrix
        # feature_matrix, model_metadata = analysis.prepare_clustering_data(
        #     df_biasedness,
        #     level=granularity_level,
        #     debug=args.debug,
        # )
        # similarity_statistics = analysis.compute_similarity_statistics(feature_matrix)
        # anova_results = analysis.variance_decomposition_anova(similarity_statistics)

        # pretty print similarity statistics
        # print(json.dumps(similarity_statistics, indent=4))
        # print(json.dumps(anova_results, indent=4))

        # Create visualizations
        print("Creating visualizations...")
        analysis.plot_clustering_results_all(
            feature_matrix,
            unsupervised_results,
            supervised_results,
            model_metadata,
            save_plots=args.save_plots,
            granularity_level=granularity_level,
            debug=args.debug,
            with_scaling=args.with_scaling,
            output_dir=output_dir,
        )

        all_clustering_results.append(
            {
                "granularity_level": granularity_level,
                "exclude_random": args.exclude_random,
                "with_scaling": args.with_scaling,
                "unsupervised_clustering_results": {
                    "best": unsupervised_results["best"],
                    "median": unsupervised_results["median"],
                    "all_k_best": unsupervised_results["all_k_best"],
                    "all_k_median": unsupervised_results["all_k_median"],
                },
                "supervised_clustering_results": supervised_results,
                "agreement_scores": agreement_scores,
                "model_names": model_metadata["model"].tolist(),  # Add model names here
                "validation_results": validation_results,
                # Add "Model Developer" results if model developer exists
                "unsupervised_clustering_results_k3": (
                    {
                        "best": {
                            "labels": unsupervised_results["all_k_best"]["labels"][3],
                            "metrics": unsupervised_results["all_k_best"]["metrics"][3],
                        },
                        "median": {
                            "labels": unsupervised_results["all_k_median"]["labels"][3],
                            "metrics": unsupervised_results["all_k_median"]["metrics"][
                                3
                            ],
                        },
                    }
                    if "Llama2" in models_to_include and "Mistral" in models_to_include
                    else None
                ),
                #'similarity_statistics': similarity_statistics,
                #'anova_results': anova_results,
            }
        )

    # Save results
    output_file_name = f"{output_dir}/clustering_results_{args.granularity_levels}_with_scaling_{args.with_scaling}"
    if args.models_to_include:
        output_file_name += "_models_" + args.models_to_include.replace(",", "_")
    if args.add_certainty_and_belief:
        output_file_name += "_with_certainty_and_belief"
    save_clustering_results(
        results=all_clustering_results,
        filepath=output_file_name,
        is_csv=True,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
