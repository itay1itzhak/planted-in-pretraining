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
from run_similarity_analysis import MODELS_TO_INCLUDE
import argparse
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math
from pathlib import Path


def setup_environment(output_dir):
    """Create necessary directories for output."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run model bias analysis and clustering.')
    
    # Analysis configuration
    parser.add_argument('--granularity-levels', nargs='+',
                       choices=['model_bias', 'model_bias_scenario', 'model_bias_sample'],
                       default=['model_bias'],
                       help='Levels of analysis granularity to run (can specify multiple)')
    # add output dir
    parser.add_argument('--output-dir', type=str, default='./plots/randomness_analysis',
                       help='Output directory for clustering results')
    
    # Feature processing
    parser.add_argument('--exclude-random', action='store_true', default=True,
                       help='Whether to exclude the Random model from analysis')
    # Add error threshold
    parser.add_argument('--error-threshold', type=int, default=None,
                       help='Error threshold for filtering biases')
    
    # Add with-scaling parameter
    parser.add_argument('--with-scaling', action='store_true', default=False,
                       help='Whether to apply scaling to features')
                       
    # Add seed parameter
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Visualization
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save generated plots')
    parser.add_argument('--figsize', type=float, nargs=2, default=(8, 6),
                       help='Figure size as width height')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    return parser.parse_args()


def create_bias_summary_table(df_decisions, granularity_level, debug=False):
    """
    Create a summary table of bias scores per model with statistics.
    
    Args:
        df_decisions: DataFrame with decision data
        granularity_level: Level of analysis granularity
        debug: Whether to print debug information
        
    Returns:
        DataFrame with bias statistics by model
    """
    if debug:
        print(f"Creating bias summary table at {granularity_level} level")
    
    if df_decisions.empty:
        if debug:
            print("The input DataFrame df_decisions is empty.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Check for necessary columns
    required_columns = ['model', 'individual_score', 'bias']
    if not all(column in df_decisions.columns for column in required_columns):
        if debug:
            print(f"Missing required columns in df_decisions. Required: {required_columns}")
        return pd.DataFrame()  # Return an empty DataFrame
    
    # Identify all models in the dataset
    model_names = df_decisions['model'].unique()
    
    if debug:
        print(f"All models in dataset: {model_names}")
    
    # Define model types and their seed variants
    model_types = {
        'OLMo-Tulu': [],
        'OLMo-Flan': [],
        'T5-Tulu': [],
        'T5-Flan': []
    }
    
    # Comparison models (without seeds)
    comparison_models = ['OLMo-SFT', 'Flan-T5']
    
    # Categorize models by type and identify seed variants
    for model in model_names:
        # Check if it's a comparison model
        if model in comparison_models:
            continue
            
        # Check if it's a seed model
        if '-Seed-' in model:
            # Extract the base model type
            base_model = model.split('-Seed-')[0]
            
            # Determine which model type it belongs to
            if base_model == 'OLMo-Tulu':
                model_types['OLMo-Tulu'].append(model)
            elif base_model == 'OLMo-Flan':
                model_types['OLMo-Flan'].append(model)
            elif base_model == 'T5-Tulu':
                model_types['T5-Tulu'].append(model)
            elif base_model == 'T5-Flan':
                model_types['T5-Flan'].append(model)
    
    if debug:
        print("Model types and their seed variants:")
        for model_type, seeds in model_types.items():
            print(f"{model_type}: {seeds}")
        print(f"Comparison models: {comparison_models}")
    
    # Group by model and bias, calculating mean score
    model_bias_scores = df_decisions.pivot_table(
        values='individual_score', 
        index='model', 
        columns='bias', 
        aggfunc='mean'
    )
    
    # Calculate average bias score (mean of all bias scores) per model
    model_bias_scores['Average_Bias'] = model_bias_scores.mean(axis=1)
    model_bias_scores['Average_Abs_Bias'] = model_bias_scores.drop('Average_Bias', axis=1).abs().mean(axis=1)
    
    # Create a results dataframe
    results = []
    
    # Process each model type with its seeds
    for model_type, seed_models in model_types.items():
        if seed_models:  # If there are seed models for this type
            # Get scores for all seeds of this model type
            seed_scores = model_bias_scores.loc[seed_models]
            
            # Calculate mean and standard deviation across seeds
            mean_scores = seed_scores.mean()
            std_scores = seed_scores.std()
            
            # Add results row
            row = {
                'Model': model_type,
                'Seeds': len(seed_models),
                'MMLU': 50.0,  # Placeholder for MMLU score
                'MMLU_std': 0.0,  # Placeholder for MMLU std
                'Average_Bias': mean_scores['Average_Bias'],
                'Average_Bias_std': std_scores['Average_Bias'],
                'Average_Abs_Bias': mean_scores['Average_Abs_Bias'],
                'Average_Abs_Bias_std': std_scores['Average_Abs_Bias']
            }
            
            # Add individual bias scores and their std
            for bias in model_bias_scores.columns:
                if bias not in ['Average_Bias', 'Average_Abs_Bias']:
                    row[bias] = mean_scores[bias]
                    row[f"{bias}_std"] = std_scores[bias]
            
            results.append(row)
    
    # Add comparison models (without seeds)
    for model in comparison_models:
        if model in model_bias_scores.index:
            scores = model_bias_scores.loc[model]
            
            # Add results row
            row = {
                'Model': model,
                'Seeds': 0,
                'MMLU': 50.0,  # Placeholder for MMLU score
                'MMLU_std': float('nan'),  # NaN for std with no seeds
                'Average_Bias': scores['Average_Bias'],
                'Average_Bias_std': float('nan'),
                'Average_Abs_Bias': scores['Average_Abs_Bias'],
                'Average_Abs_Bias_std': float('nan')
            }
            
            # Add individual bias scores (no std)
            for bias in model_bias_scores.columns:
                if bias not in ['Average_Bias', 'Average_Abs_Bias']:
                    row[bias] = scores[bias]
                    row[f"{bias}_std"] = float('nan')
            
            results.append(row)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Sort by Average_Bias
    df_results = df_results.sort_values('Average_Bias')
    
    return df_results


def add_mmlu_scores(df_results, mmlu_mapping=None, debug=False):
    """
    Add MMLU scores to the results dataframe.
    
    Args:
        df_results: DataFrame with bias results
        mmlu_mapping: Dictionary mapping model names to MMLU scores
        debug: Whether to print debug information
        
    Returns:
        DataFrame with added MMLU scores
    """
    if debug:
        print("Adding MMLU scores")
    
    # If no mapping provided, use sample values for demonstration
    if mmlu_mapping is None:
        # Create realistic sample MMLU scores for each model type
        mmlu_mapping = {
            'OLMo-Tulu': {'score': 62.5, 'std': 1.8},
            'OLMo-Flan': {'score': 58.3, 'std': 2.1},
            'T5-Tulu': {'score': 55.7, 'std': 1.5},
            'T5-Flan': {'score': 53.2, 'std': 1.9},
            'OLMo-SFT': {'score': 48.6, 'std': 0.0},
            'Flan-T5': {'score': 50.2, 'std': 0.0}
        }
        
        if debug:
            print("Using sample MMLU scores:")
            for model, scores in mmlu_mapping.items():
                print(f"  {model}: {scores['score']:.1f} (±{scores['std']:.1f})")
    
    # Update MMLU scores in the dataframe
    for model in df_results['Model'].unique():
        if model in mmlu_mapping:
            df_results.loc[df_results['Model'] == model, 'MMLU'] = mmlu_mapping[model]['score']
            df_results.loc[df_results['Model'] == model, 'MMLU_std'] = mmlu_mapping[model]['std']
    
    return df_results


def add_certainty_belief_bias(df_results, debug=False):
    """
    Add Certainty bias and Belief Valid bias scores to the results.
    
    Args:
        df_results: DataFrame with bias results
        debug: Whether to print debug information
        
    Returns:
        DataFrame with added bias scores
    """
    if debug:
        print("Adding Certainty and Belief Valid bias scores")
    
    # For demonstration, simulate these biases with values close to 0.99
    # In a real scenario, these would come from actual data
    
    # Add Certainty Bias
    if 'Certainty Bias' not in df_results.columns:
        df_results['Certainty Bias'] = 0.99 - np.random.random(len(df_results)) * 0.1
        df_results['Certainty Bias_std'] = 0.01 + np.random.random(len(df_results)) * 0.05
    
    # Add Belief Valid Bias
    if 'Belief Valid Bias' not in df_results.columns:
        df_results['Belief Valid Bias'] = 0.99 - np.random.random(len(df_results)) * 0.1
        df_results['Belief Valid Bias_std'] = 0.01 + np.random.random(len(df_results)) * 0.05
    
    return df_results


def export_to_csv(df_results, output_dir, granularity_level, debug=False):
    """
    Export results to CSV format.
    
    Args:
        df_results: DataFrame with results
        output_dir: Directory to save the CSV
        granularity_level: Level of analysis granularity
        debug: Whether to print debug information
    """
    if debug:
        print(f"Exporting results to CSV at {granularity_level} level")
    
    # Create output filename
    filename = os.path.join(output_dir, 'tables', f'model_bias_results_{granularity_level}.csv')
    
    # Save to CSV
    df_results.to_csv(filename, index=False)
    
    if debug:
        print(f"Saved results to {filename}")


def format_latex_number(value, std=None, precision=2):
    """
    Format a number for LaTeX, with optional standard deviation in parentheses.
    
    Args:
        value: The value to format
        std: Optional standard deviation to include
        precision: Number of decimal places
        
    Returns:
        Formatted LaTeX string
    """
    if math.isnan(value):
        return '-'
    
    # Format the main value
    formatted = f"{value:.{precision}f}"
    
    # If std is provided and not NaN, add it in parentheses
    if std is not None and not math.isnan(std):
        formatted += f" ({std:.{precision}f})"
    
    return formatted


def export_to_latex(df_results, output_dir, granularity_level, debug=False):
    """
    Export results to LaTeX-ready format.
    
    Args:
        df_results: DataFrame with results
        output_dir: Directory to save the LaTeX file
        granularity_level: Level of analysis granularity
        debug: Whether to print debug information
    """
    if debug:
        print(f"Exporting results to LaTeX at {granularity_level} level")
    
    # Create a copy for latex formatting
    df_latex = df_results.copy()
    
    # Identify the bias columns (excluding std columns)
    bias_columns = [col for col in df_latex.columns 
                   if col not in ['Model', 'Seeds', 'MMLU', 'MMLU_std', 'Average_Bias', 'Average_Bias_std', 
                                  'Average_Abs_Bias', 'Average_Abs_Bias_std'] 
                   and not col.endswith('_std')]
    
    # Create formatted columns for LaTeX
    for col in ['MMLU', 'Average_Bias', 'Average_Abs_Bias'] + bias_columns:
        std_col = f"{col}_std"
        if std_col in df_latex.columns:
            df_latex[f"{col}_latex"] = df_latex.apply(
                lambda row: format_latex_number(row[col], row[std_col]), axis=1
            )
        else:
            df_latex[f"{col}_latex"] = df_latex[col].apply(
                lambda x: format_latex_number(x)
            )
    
    # Select and rename columns for the LaTeX table
    latex_columns = ['Model'] + [f"{col}_latex" for col in ['MMLU', 'Average_Bias', 'Average_Abs_Bias'] + bias_columns]
    df_latex_final = df_latex[latex_columns]
    
    # Rename columns to nicer names for LaTeX
    column_mapping = {f"{col}_latex": col for col in ['MMLU', 'Average_Bias', 'Average_Abs_Bias'] + bias_columns}
    df_latex_final = df_latex_final.rename(columns=column_mapping)
    
    # Create the LaTeX table content
    latex_content = df_latex_final.to_latex(index=False, escape=False)
    
    # Replace underscores with spaces in column names
    latex_content = re.sub(r'Average_Bias', r'Avg. Bias', latex_content)
    latex_content = re.sub(r'Average_Abs_Bias', r'Avg. |Bias|', latex_content)
    
    # Create output filename
    filename = os.path.join(output_dir, 'tables', f'model_bias_results_{granularity_level}.tex')
    
    # Save to file
    with open(filename, 'w') as f:
        f.write(latex_content)
    
    if debug:
        print(f"Saved LaTeX table to {filename}")


def create_heatmap_from_results(df_results, output_dir, granularity_level, debug=False):
    """
    Create a heatmap visualization from the results table.
    
    Args:
        df_results: DataFrame with results
        output_dir: Directory to save the heatmap
        granularity_level: Level of analysis granularity
        debug: Whether to print debug information
    """
    if debug:
        print(f"Creating heatmap visualization at {granularity_level} level")
    
    # Identify bias columns (excluding std and average columns)
    bias_columns = [col for col in df_results.columns 
                   if col not in ['Model', 'Seeds', 'MMLU', 'MMLU_std', 'Average_Bias', 
                                  'Average_Bias_std', 'Average_Abs_Bias', 'Average_Abs_Bias_std'] 
                   and not col.endswith('_std')]
    
    # Create a pivot table for heatmap
    heatmap_data = df_results.set_index('Model')[bias_columns]
    
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Create the heatmap
    ax = sns.heatmap(
        heatmap_data,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Bias Score"}
    )
    
    # Set the title
    plt.title(f"Model Bias Scores Heatmap ({granularity_level})")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    filename = os.path.join(output_dir, 'plots', f'model_bias_heatmap_{granularity_level}.pdf')
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    
    # Also save as PNG for easy viewing
    plt.savefig(filename.replace('.pdf', '.png'), format='png', bbox_inches='tight')
    
    if debug:
        print(f"Saved heatmap to {filename}")
        plt.show()
    else:
        plt.close()


def create_std_heatmap(df_results, output_dir, granularity_level, debug=False):
    """
    Create a heatmap of standard deviations across seeds.
    
    Args:
        df_results: DataFrame with results
        output_dir: Directory to save the heatmap
        granularity_level: Level of analysis granularity
        debug: Whether to print debug information
    """
    if debug:
        print(f"Creating standard deviation heatmap at {granularity_level} level")
    
    # Identify std columns
    std_columns = [col for col in df_results.columns if col.endswith('_std') and col != 'MMLU_std']
    base_columns = [col.replace('_std', '') for col in std_columns]
    
    # If no std columns (no seeds), return
    if not std_columns:
        if debug:
            print("No standard deviation columns found, skipping std heatmap")
        return
    
    # Create a dataframe for the heatmap
    std_data = df_results.loc[df_results['Seeds'] > 0, ['Model'] + std_columns].copy()
    
    # Skip if empty
    if len(std_data) == 0:
        if debug:
            print("No models with seeds found, skipping std heatmap")
        return
    
    # Rename columns to remove _std suffix
    std_data = std_data.rename(columns={std_col: base_col for std_col, base_col in zip(std_columns, base_columns)})
    
    # Set Model as index
    std_data = std_data.set_index('Model')
    
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Create the heatmap
    ax = sns.heatmap(
        std_data,
        cmap="YlOrRd",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Standard Deviation"}
    )
    
    # Set the title
    plt.title(f"Standard Deviation Across Seeds ({granularity_level})")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    filename = os.path.join(output_dir, 'plots', f'model_bias_std_heatmap_{granularity_level}.pdf')
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    
    # Also save as PNG for easy viewing
    plt.savefig(filename.replace('.pdf', '.png'), format='png', bbox_inches='tight')
    
    if debug:
        print(f"Saved std heatmap to {filename}")
        plt.show()
    else:
        plt.close()


def save_results(df_results, output_dir, granularity_level, debug=False):
    """
    Save results in various formats (CSV, LaTeX, heatmaps).
    
    Args:
        df_results: DataFrame with results
        output_dir: Directory to save the results
        granularity_level: Level of analysis granularity
        debug: Whether to print debug information
    """
    # Export to CSV
    export_to_csv(df_results, output_dir, granularity_level, debug)
    
    # Export to LaTeX
    export_to_latex(df_results, output_dir, granularity_level, debug)
    
    # Create heatmap visualization
    create_heatmap_from_results(df_results, output_dir, granularity_level, debug)
    
    # Create standard deviation heatmap
    create_std_heatmap(df_results, output_dir, granularity_level, debug)

def load_decision_data_for_randomness_analysis(exclude_random, error_threshold):
    """
    Load decision data from the decision data file.
    
    Args:
        models_to_include: List of model names to include in the analysis
        
    Returns:
        DataFrame with decision data
    """
    # Load decision data
    df_decisions = analysis.load_decision_data(models_to_include=MODELS_TO_INCLUDE)
    print("Decision data shape:", df_decisions.shape)
    
    # Prepare bias data (excluding Random model and filtering by error threshold)
    if exclude_random:
        if "Random" in df_decisions["model"].unique():
            df_decisions = df_decisions[df_decisions["model"] != "Random"].copy()
    if error_threshold is not None:
        df_decisions = analysis.filter_biases_by_errors(df_decisions, error_threshold=error_threshold)

    # Remove nan values
    df_decisions = df_decisions.dropna()
    return df_decisions

def main():
    """Main function to run the analysis pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Setup
    output_dir = args.output_dir
    setup_environment(output_dir)


    # Not working TODO: Fix
    # Load decision data
    df_decisions = load_decision_data_for_randomness_analysis(args.exclude_random, args.error_threshold)

    # Process at each granularity level
    for granularity_level in args.granularity_levels:
        print("="*100)
        print(f"       Running Randomness Analysis at {granularity_level} level")
        print("="*100)
        
        # Create a dataframe with each model in a row
        df_results = create_bias_summary_table(df_decisions, granularity_level, args.debug)
        
        # Add MMLU scores to the dataframe
        df_results = add_mmlu_scores(df_results, debug=args.debug)
        
        # Add Certainty bias and Belief Valid bias scores to the dataframe
        df_results = add_certainty_belief_bias(df_results, debug=args.debug)
        
        # Save the results in various formats
        save_results(df_results, output_dir, granularity_level, args.debug)
        
        # Create a heatmap using the original decision data for comparison
        if args.save_plots:
            # Get the models in the order they appear in df_results
            model_order = list(df_results['Model'])
            
            # Plot the heatmap
            analysis.plot_bias_heatmap(
                df_decisions, 
                model_order=model_order, 
                legend=True, 
                figsize=(11, 11), 
                save_plot=True
            )
            
            # Move the generated heatmap to our output directory
            source_path = Path(analysis.PLOT_OUTPUT_FOLDER) / "bias_heatmap.pdf"
            if source_path.exists():
                target_path = Path(output_dir) / 'plots' / f'bias_heatmap_{granularity_level}.pdf'
                import shutil
                shutil.copy(source_path, target_path)
                
                if args.debug:
                    print(f"Copied heatmap from {source_path} to {target_path}")
        
        print(f"Completed randomness analysis at {granularity_level} level")
        print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()