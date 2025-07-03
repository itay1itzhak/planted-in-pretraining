from curses import tigetstr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
import tiktoken
import umap
import os
import re
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from sklearn.metrics.pairwise import euclidean_distances
from itertools import combinations, permutations
from scipy.stats import f_oneway
from scipy.stats import ttest_ind
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

TEST_CASE_DATA_FOLDER = os.path.join(
    "cognitive-biases-in-llms", "data", "generated_datasets"
)
DECISION_DATA_FOLDER = os.path.join(
    "cognitive-biases-in-llms", "data", "decision_results"
)
PLOT_OUTPUT_FOLDER = os.path.join("cognitive-biases-in-llms", "plots")
TEST_CASE_DATASET = os.path.join("cognitive-biases-in-llms", "data", "full_dataset.csv")

BIAS_NAME_MAPPING = {
    "Escalation Of Commitment": "Escalation of Commitment",
    "Illusion Of Control": "Illusion of Control",
    "Self Serving Bias": "Self-Serving Bias",
    "In Group Bias": "In-Group Bias",
    "Status Quo Bias": "Status-Quo Bias",
}

MODEL_NAME_MAPPING = {
    "gpt-4o-2024-08-06": "GPT-4o",
    "gpt-4o-mini-2024-07-18": "GPT-4o mini",
    "gpt-3.5-turbo-0125": "GPT-3.5 Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct": "Llama 3.1 405B",
    "meta-llama/Meta-Llama-3.1-70B-Instruct": "Llama 3.1 70B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama 3.2 3B",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama 3.2 1B",
    "claude-3-haiku-20240307": "Claude 3 Haiku",
    "models/gemini-1.5-pro": "Gemini 1.5 Pro",
    "models/gemini-1.5-flash": "Gemini 1.5 Flash",
    "google/gemma-2-27b-it": "Gemma 2 27B",
    "google/gemma-2-9b-it": "Gemma 2 9B",
    "mistral-small-2409": "Mistral Small",
    "mistral-large-2407": "Mistral Large",
    "microsoft/WizardLM-2-8x22B": "WizardLM-2 8x22B",
    "microsoft/WizardLM-2-7B": "WizardLM-2 7B",
    "accounts/fireworks/models/phi-3-vision-128k-instruct": "Phi-3.5",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5 72B",
    "accounts/yi-01-ai/models/yi-large": "Yi-Large",
    "random-model": "Random",
    "OLMo-Flan-seed-1": "OLMo-Flan-Seed-1",
    "OLMo-Flan-seed-2": "OLMo-Flan-Seed-2",
    "OLMo-Flan-seed-0": "OLMo-Flan-Seed-0",
    "OLMo-Tulu-seed-1": "OLMo-Tulu-Seed-1",
    "OLMo-Tulu-seed-2": "OLMo-Tulu-Seed-2",
    "OLMo-Tulu-seed-0": "OLMo-Tulu-Seed-0",
    "T5-Flan-seed-1": "T5-Flan-Seed-1",
    "T5-Flan-seed-2": "T5-Flan-Seed-2",
    "T5-Flan-seed-0": "T5-Flan-Seed-0",
    "T5-Tulu-seed-1": "T5-Tulu-Seed-1",
    "T5-Tulu-seed-2": "T5-Tulu-Seed-2",
    "T5-Tulu-seed-0": "T5-Tulu-Seed-0",
    "OLMo-SFT": "OLMo-SFT",
    "Flan-T5": "Flan-T5",
    "Mistral-Tulu-Seed-0": "Mistral-Tulu-Seed-0",
    "Mistral-Tulu-Seed-1": "Mistral-Tulu-Seed-1",  # different learning rate
    "Mistral-ShareGPT": "Mistral-ShareGPT",  # DICE version
    "Llama2-Tulu": "Llama2-Tulu",  # AI2 version
    "Llama2-ShareGPT-Seed-0": "Llama2-ShareGPT-Seed-0",  # AI2 version
    "Llama2-ShareGPT-Seed-1": "Llama2-ShareGPT-Seed-1",  # Vicuna version
    "Llama2-ShareGPT-Seed-2": "Llama2-ShareGPT-Seed-2",  # Vicuna version with different system prompt
}

MODEL_DEVELOPER_MAPPING = {
    "GPT-4o": "OpenAI",
    "GPT-4o mini": "OpenAI",
    "GPT-3.5 Turbo": "OpenAI",
    "Llama 3.1 405B": "Meta",
    "Llama 3.1 70B": "Meta",
    "Llama 3.1 8B": "Meta",
    "Llama 3.2 3B": "Meta",
    "Llama 3.2 1B": "Meta",
    "Claude 3 Haiku": "Anthropic",
    "Gemini 1.5 Pro": "Google",
    "Gemini 1.5 Flash": "Google",
    "Gemma 2 27B": "Google",
    "Gemma 2 9B": "Google",
    "Mistral Small": "Mistral",
    "Mistral Large": "Mistral",
    "WizardLM-2 8x22B": "Microsoft",
    "WizardLM-2 7B": "Microsoft",
    "Phi-3.5": "Microsoft",
    "Qwen2.5 72B": "Alibaba",
    "Yi-Large": "01.AI",
    "Random": "None",
    "OLMo-Flan-Seed-1": "OLMo-Flan",
    "OLMo-Flan-Seed-2": "OLMo-Flan",
    "OLMo-Flan-Seed-0": "OLMo-Flan",
    "OLMo-Tulu-Seed-1": "OLMo-Tulu",
    "OLMo-Tulu-Seed-2": "OLMo-Tulu",
    "OLMo-Tulu-Seed-0": "OLMo-Tulu",
    "T5-Flan-Seed-1": "T5-Flan",
    "T5-Flan-Seed-2": "T5-Flan",
    "T5-Flan-Seed-0": "T5-Flan",
    "OLMo-SFT": "AI2",
    "Flan-T5": "Google",
    "T5-Tulu-Seed-1": "T5-Tulu",
    "T5-Tulu-Seed-2": "T5-Tulu",
    "T5-Tulu-Seed-0": "T5-Tulu",
    "Mistral-Tulu-Seed-0": "DICE",
    "Mistral-Tulu-Seed-1": "DICE",
    "Mistral-ShareGPT": "DICE",
    "Llama2-Tulu": "AI2",
    "Llama2-ShareGPT-Seed-0": "AI2",
    "Llama2-ShareGPT-Seed-1": "Vicuna",
    "Llama2-ShareGPT-Seed-2": "Vicuna",
}

MODEL_SIZE_MAPPING = {
    "GPT-4o": 200,  # Assumption (real size not published)
    "GPT-4o mini": 10,  # Assumption (real size not published)
    "GPT-3.5 Turbo": 175,
    "Llama 3.1 405B": 405,
    "Llama 3.1 70B": 70,
    "Llama 3.1 8B": 8,
    "Llama 3.2 3B": 3,
    "Llama 3.2 1B": 1,
    "Claude 3 Haiku": 20,  # Assumption (real size not published)
    "Gemini 1.5 Pro": 200,  # Assumption (real size not published)
    "Gemini 1.5 Flash": 30,  # Assumption (real size not published)
    "Gemma 2 27B": 27,
    "Gemma 2 9B": 9,
    "Mistral Large": 123,
    "Mistral Small": 22,
    "WizardLM-2 8x22B": 176,
    "WizardLM-2 7B": 7,
    "Phi-3.5": 4.2,
    "Qwen2.5 72B": 72,
    "Yi-Large": 34,
    "Random": 0,
    "OLMo-Flan-Seed-1": 7,
    "OLMo-Flan-Seed-2": 7,
    "OLMo-Flan-Seed-0": 7,
    "T5-Tulu-Seed-1": 11,
    "T5-Tulu-Seed-2": 11,
    "T5-Tulu-Seed-0": 11,
    "OLMo-SFT": 7,
    "Flan-T5": 11,
    "T5-Flan-Seed-1": 11,
    "T5-Flan-Seed-2": 11,
    "T5-Flan-Seed-0": 11,
    "OLMo-Tulu-Seed-1": 7,
    "OLMo-Tulu-Seed-2": 7,
    "OLMo-Tulu-Seed-0": 7,
    "Mistral-Tulu-Seed-0": 7,
    "Mistral-Tulu-Seed-1": 7,
    "Mistral-ShareGPT": 7,
    "Llama2-Tulu": 7,
    "Llama2-ShareGPT-Seed-0": 7,
    "Llama2-ShareGPT-Seed-1": 7,
    "Llama2-ShareGPT-Seed-2": 7,
}

# Model scores from Chatbot Arena
MODEL_SCORE_MAPPING = {
    "GPT-4o": 1264,
    "GPT-4o mini": 1273,
    "GPT-3.5 Turbo": 1106,
    "Llama 3.1 405B": 1267,
    "Llama 3.1 70B": 1248,
    "Llama 3.1 8B": 1172,
    "Llama 3.2 3B": 1102,
    "Llama 3.2 1B": 1054,
    "Claude 3 Haiku": 1179,
    "Gemini 1.5 Pro": 1304,
    "Gemini 1.5 Flash": 1265,
    "Gemma 2 27B": 1218,
    "Gemma 2 9B": 1189,
    "Mistral Small": None,
    "Mistral Large": 1251,
    "WizardLM-2 8x22B": None,
    "WizardLM-2 7B": None,
    "Phi-3.5": None,
    "Qwen2.5 72B": 1257,
    "Yi-Large": 1212,
    "Random": None,
}

MODEL_ORDER = list(MODEL_NAME_MAPPING.values())

# Global configuration for analysis
DEFAULT_SCALING = False  # Set to False to match original behavior


def load_decision_data(
    folder_path: str = DECISION_DATA_FOLDER,
    format: bool = True,
    models_to_include: list[str] = None,
) -> pd.DataFrame:
    """
    Loads a single dataframe with all decision results.
    """

    # List to store dataframes
    dataframes = []

    # List to store the column names of each dataframe
    columns_list = []

    # Iterate over each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):  # Only process CSV files
            # If not any of the models are included in the file name, skip the file
            if models_to_include is not None and not any(
                model_name in file_name for model_name in models_to_include
            ):
                continue
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)  # Load the CSV into a dataframe
            # Remove rows with nan in model column (to remove error counting)
            df = df[df["model"].notna()]
            dataframes.append(df)  # Store the dataframe
            columns_list.append(
                set(df.columns)
            )  # Store the columns as a set for comparison

    # Find the common columns across all dataframes
    common_columns = set.intersection(*columns_list)

    # Filter each dataframe to only keep the common columns
    filtered_dataframes = [df[list(common_columns)] for df in dataframes]

    # Concatenate the filtered dataframes into one large dataframe
    df = pd.concat(filtered_dataframes, ignore_index=True)

    # Format the decision data if requested
    if format:
        df = format_decision_data(df)

    return df


def load_test_case_data(
    file_path: str = TEST_CASE_DATASET, models_to_include: list[str] = None
) -> pd.DataFrame:
    """
    Loads a pandas DataFrame with all test cases.
    """

    # Load the dataset with all test cases
    df_tests = pd.read_csv(file_path)

    # If requested, filter the dataframe to only include the specified models
    if models_to_include is not None:
        df_tests = df_tests[df_tests["model"].isin(models_to_include)]

    return df_tests


def get_error_counts_by_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates error counts for each bias and model combination.

    Args:
        df: DataFrame with decision results

    Returns:
        DataFrame with error counts per bias and model
    """
    # Count errors for each bias and model
    error_counts = (
        df[df["error_message"].notna()]
        .groupby(["bias", "model"])
        .size()
        .reset_index(name="error_count")
    )

    # Pivot to get bias x model matrix
    error_matrix = error_counts.pivot(
        index="bias", columns="model", values="error_count"
    ).fillna(0)

    return error_matrix


def filter_biases_by_errors(
    df: pd.DataFrame, error_threshold: int = None, debug: bool = False
) -> pd.DataFrame:
    """
    Filters out biases that have too many errors in any model.

    Args:
        df: DataFrame with decision results
        error_threshold: Maximum number of allowed errors per bias per model
        debug: Whether to print debug information

    Returns:
        Filtered DataFrame
    """
    if error_threshold is None:
        return df

    if debug:
        print(f"\nFiltering biases with error threshold: {error_threshold}")
        print(f"Initial number of biases: {df['bias'].nunique()}")

    # Get error counts for each bias-model combination
    error_matrix = get_error_counts_by_bias(df)

    # Find biases that exceed threshold in any model
    biases_to_remove = error_matrix[
        (error_matrix > error_threshold).any(axis=1)
    ].index.tolist()

    if debug:
        print(f"Biases removed due to errors: {biases_to_remove}")

    # Filter the dataframe
    df_filtered = df[~df["bias"].isin(biases_to_remove)].copy()

    if debug:
        print(f"Final number of biases: {df_filtered['bias'].nunique()}")

    return df_filtered


def load_model_bias_data(
    df_decisions: pd.DataFrame = None,
    df_tests: pd.DataFrame = None,
    models_to_include: list[str] = None,
    error_threshold: int = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Loads and processes model bias data from decision results.

    Args:
        df_decisions: DataFrame with decision results (optional)
        df_tests: DataFrame with test cases (optional)
        models_to_include: List of models to include (optional)
        error_threshold: Maximum number of allowed errors per bias per model
        debug: Whether to print debug information

    Returns:
        DataFrame with processed bias data
    """
    # If no dataframe with decision results is passed, load it from files
    if df_decisions is None:
        df_decisions = load_decision_data(models_to_include=models_to_include)

    # If no dataframe with test cases is passed, load it from files
    if df_tests is None:
        df_tests = load_test_case_data(models_to_include=models_to_include)

    # Filter out biases with too many errors if threshold is provided
    if error_threshold is not None:
        if debug:
            print(f"\nFiltering biases with error threshold: {error_threshold}")
            print(f"Initial number of biases: {df_decisions['bias'].nunique()}")

        # Get error counts and filter
        error_matrix = get_error_counts_by_bias(df_decisions)
        biases_to_remove = error_matrix[
            (error_matrix > error_threshold).any(axis=1)
        ].index.tolist()

        if debug:
            print(f"Biases removed due to errors: {biases_to_remove}")

        # Filter the decisions dataframe
        df_decisions = df_decisions[~df_decisions["bias"].isin(biases_to_remove)].copy()

        if debug:
            print(f"Final number of biases: {df_decisions['bias'].nunique()}")

        if len(df_decisions) == 0:
            raise ValueError(
                f"No biases remain after filtering with threshold {error_threshold}. "
                "Consider using a higher threshold."
            )

    # Step 1: Merge scenario information
    df = (
        df_decisions[["id", "bias", "individual_score", "model"]]
        .sort_values(by=["model", "id"])
        .merge(df_tests[["id", "scenario"]])
    )

    # Step 2: Sort the dataframe
    df = df.sort_values(by=["model", "scenario", "bias", "id"])

    # Step 3: Pivot and process the data
    df = df.pivot_table(
        index=["model", "scenario"],
        columns="bias",
        values="individual_score",
        aggfunc=lambda x: list(x),
    )

    # Step 4: Reshape the data
    df = df.reset_index()
    df = df.melt(
        id_vars=["model", "scenario"], var_name="bias", value_name="individual_score"
    )
    df = df.explode("individual_score")
    df = df.reset_index(drop=True)

    return df


def load_model_failures(df_decisions: pd.DataFrame):
    """
    Loads a DataFrame capturing the success rate (1 - failure rate) of all models.
    """

    # Calculate the average success rate per model
    df_failures = df_decisions[["model", "status"]].copy()
    df_failures["status"] = df_failures["status"].map({"OK": 1, "ERROR": 0}) * 100.0
    df_failures = df_failures.groupby("model").mean()
    df_failures = df_failures.sort_values(by="status", ascending=False).reset_index()

    # Rename the column
    df_failures = df_failures.rename(columns={"status": "Valid Answers"})

    return df_failures


def load_model_answer_length(df_decisions: pd.DataFrame):
    """
    Loads a DataFrame capturing the average length (in tokens) of answers per model.
    """

    # Load tiktoken tokenizer encodings
    encoding = tiktoken.get_encoding("cl100k_base")

    # Define a reusable function for counting tokens
    def count_tokens(s: str):
        return len(encoding.encode(s))

    # Count the number of tokens in the model answer
    df_lengths = df_decisions[["model", "control_answer", "treatment_answer"]].copy()
    df_lengths["tokens_control_answer"] = (
        df_decisions["control_answer"].astype(str).apply(count_tokens)
    )
    df_lengths["tokens_treatment_answer"] = (
        df_decisions["treatment_answer"].astype(str).apply(count_tokens)
    )

    # Sum up the token counts
    df_lengths["Average Output Tokens"] = (
        df_lengths["tokens_control_answer"] + df_lengths["tokens_treatment_answer"]
    )

    # Calculate the average output token count per model
    df_lengths = (
        df_lengths[["model", "Average Output Tokens"]]
        .groupby("model")
        .mean()
        .reset_index()
        .sort_values(by="Average Output Tokens", ascending=False)
    )

    return df_lengths


def load_model_characteristics(
    df_decisions: pd.DataFrame,
    df_biasedness: pd.DataFrame,
    incl_failures: bool = False,
    incl_lengths: bool = False,
    incl_random: bool = False,
    fill_na_scores: bool = True,
):
    """
    Loads a DataFrame with a summary of model characteristics.
    """

    # Put all relevant information about the models into a single DataFrame
    df_mean_abs_bias = calculate_mean_absolute(
        df_biasedness, by="model", col_name="Bias"
    )
    df_mean_abs_bias["Parameters"] = df_mean_abs_bias["model"].map(MODEL_SIZE_MAPPING)
    df_mean_abs_bias["Developer"] = df_mean_abs_bias["model"].map(
        MODEL_DEVELOPER_MAPPING
    )
    df_mean_abs_bias["Score"] = df_mean_abs_bias["model"].map(MODEL_SCORE_MAPPING)

    # If requested, load additional information on the % of failed answers per model
    if incl_failures:
        df_failures = load_model_failures(df_decisions)
        df_mean_abs_bias = df_mean_abs_bias.merge(df_failures)

    # If requested, load additional information on the average output tokens per model
    if incl_lengths:
        df_lengths = load_model_answer_length(df_decisions)
        df_mean_abs_bias = df_mean_abs_bias.merge(df_lengths)

    # Rename some columns
    df_mean_abs_bias = df_mean_abs_bias.rename(
        columns={"Bias": "Mean Absolute Bias", "Score": "Chatbot Arena Score"}
    )

    # If requested, fill in missing score values with the mean score and add an asterisk to the model name
    if fill_na_scores:
        missing_scores = df_mean_abs_bias["Chatbot Arena Score"].isna()
        mean_score = df_mean_abs_bias["Chatbot Arena Score"].mean()
        df_mean_abs_bias["Chatbot Arena Score"] = df_mean_abs_bias[
            "Chatbot Arena Score"
        ].fillna(mean_score)
        df_mean_abs_bias.loc[missing_scores, "model"] = (
            df_mean_abs_bias.loc[missing_scores, "model"] + "*"
        )

    # Unless requested, exclude the Random model
    if not incl_random:
        df_mean_abs_bias = df_mean_abs_bias[
            df_mean_abs_bias["model"] != "Random*"
        ].reset_index(drop=True)

    return df_mean_abs_bias


def format_decision_data(
    df: pd.DataFrame, format_bias_names: bool = True, format_model_names: bool = True
) -> pd.DataFrame:
    """
    Formats the bias and model names in a dataframe with decision results.
    """

    if format_bias_names:
        # Format bias names properly
        df["bias"] = (
            df["bias"]
            .apply(lambda x: re.sub(r"([a-z])([A-Z])", r"\1 \2", x))
            .replace(BIAS_NAME_MAPPING)
        )

    if format_model_names:
        df["model"] = df["model"].replace(MODEL_NAME_MAPPING)

    return df


def impute_missing_values(df: pd.DataFrame):
    """
    Imputes all missing values in the dataframe.
    """

    # Impute missing values with mean values
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

    # Identify columns with missing values
    nan_cols = df.columns[df.isna().any()]
    non_nan_cols = df.columns.drop(nan_cols)

    # Impute the missing values in the numeric columns
    df_imputed = pd.DataFrame(imputer.fit_transform(df[nan_cols]), columns=nan_cols)

    # Print the imputed values of the imputed columns (a single value per column, which is the mean of the column)
    print(
        f"Imputed values (the single mean bias score that replaced all NaN values): {df_imputed.mean()}"
    )

    # Concatenate the imputed values with the non-missing values
    df_imputed[non_nan_cols] = df[non_nan_cols]
    df_imputed = df_imputed[df.columns]

    return df_imputed


def impute_missing_values_granularity_level(
    df: pd.DataFrame, impute_granularity_level: str
) -> pd.DataFrame:
    """
    Imputes missing values in the dataframe by replacing them with the mean bias score per bias level.

    Args:
        df: DataFrame with 'bias' and 'individual_score' columns
        impute_granularity_level: 'bias' or 'model_bias' or 'model_scenario'

    Returns:
        DataFrame with missing values imputed at the specified granularity level

    Raises:
        ValueError: If required columns are missing or if all values for a bias are missing
    """
    # Validate input
    if impute_granularity_level == "impute_bias_level":
        required_cols = ["bias", "individual_score"]
    elif impute_granularity_level == "impute_model_bias_level":
        required_cols = ["model", "bias", "individual_score"]
    elif impute_granularity_level == "impute_model_scenario_level":
        required_cols = ["model", "scenario", "individual_score"]
    elif impute_granularity_level == "impute_model_bias_scenario_level":
        required_cols = ["model", "bias", "scenario", "individual_score"]
    else:
        raise ValueError(f"Invalid granularity level: {impute_granularity_level}")

    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Create a copy to avoid modifying the original
    df_imputed = df.copy()

    # Keep indicies of all NaN values
    # nan_indices = df[df['individual_score'].isna()].index

    # Calculate mean by level
    if impute_granularity_level == "impute_bias_level":
        bias_means = df_imputed.groupby(["bias"])["individual_score"].transform("mean")
    elif impute_granularity_level == "impute_model_bias_level":
        bias_means = df_imputed.groupby(["model", "bias"])[
            "individual_score"
        ].transform("mean")
    elif impute_granularity_level == "impute_model_scenario_level":
        bias_means = df_imputed.groupby(["model", "scenario"])[
            "individual_score"
        ].transform("mean")
    elif impute_granularity_level == "impute_model_bias_scenario_level":
        bias_means = df_imputed.groupby(["model", "scenario", "bias"])[
            "individual_score"
        ].transform("mean")

    # Impute missing values
    df_imputed["individual_score"] = df_imputed["individual_score"].fillna(bias_means)

    # Since the scenario level is the finest granularity level, we need to fill in the missing values it has with the mean of the model-bias level
    if impute_granularity_level == "impute_model_scenario_level":
        model_bias_means = df_imputed.groupby(["model", "bias"])[
            "individual_score"
        ].transform("mean")
        df_imputed["individual_score"] = df_imputed["individual_score"].fillna(
            model_bias_means
        )
    elif impute_granularity_level == "impute_model_bias_scenario_level":
        model_bias_means = df_imputed.groupby(["model", "bias"])[
            "individual_score"
        ].transform("mean")
        df_imputed["individual_score"] = df_imputed["individual_score"].fillna(
            model_bias_means
        )

    # Check if df_imputed has missing values
    if df_imputed["individual_score"].isna().any():
        problematic_biases = df_imputed[df_imputed["individual_score"].isna()][
            "bias"
        ].unique()
        raise ValueError(f"All values missing for bias levels: {problematic_biases}")

    # Print the imputed values in the indices of the NaN values
    # print(f"Imputed values in the indices of the NaN values: {df_imputed.loc[nan_indices, 'individual_score']}")

    return df_imputed


def group_by_model(df: pd.DataFrame, agg: str = "mean") -> pd.DataFrame:
    """
    Groups bias scores by model, maintaining the structure needed for dimensionality reduction.

    Args:
        df: DataFrame with columns ['model', 'scenario', 'bias', 'individual_score']
        agg: Aggregation function to use ('mean' or 'var')

    Returns:
        DataFrame with columns ['model', 'bias', 'individual_score']
    """
    # Validate input
    required_cols = ["model", "bias", "individual_score"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Group by model and bias, then aggregate individual scores
    df_grouped = (
        df.groupby(["model", "bias"])[["individual_score"]].agg(agg).reset_index()
    )

    return df_grouped


def calculate_mean_absolute(df: pd.DataFrame, by: str, col_name: str):
    """
    Calculates mean absolute values of all numeric columns by (1) grouping by the "by" column and (2) aggregating all numeric columns into a single "col_name" column.
    """

    # Assemble a list of all numeric columns and the column the DataFrame shall be grouped by
    num_cols = df.select_dtypes(include=["number"]).columns
    keep_cols = list(num_cols)
    keep_cols.append("model")

    # Convert all numeric values to absolute values
    df[num_cols] = df[num_cols].abs()

    # Group the dataframe by the selected column and apply the aggregation function to all other numeric columns
    df_grouped = df[keep_cols].groupby("model").agg("mean").reset_index()

    # Aggregate all numeric columns into one by taking the mean
    df_grouped[col_name] = df_grouped[num_cols].mean(axis=1)
    df_grouped = df_grouped.drop(columns=num_cols)

    return df_grouped


def reduce_with_pca(
    df: pd.DataFrame,
    n_components: int = 2,
    group_by: str = "scenario",
    debug: bool = False,
    with_scaling: bool = False,
) -> pd.DataFrame:
    """
    Performs PCA dimensionality reduction on bias scores.
    """
    # Debug prints
    if debug:
        print("\nDEBUG: Initial dataframe:")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("Number of unique models:", df["model"].nunique())
        print("Unique models:", df["model"].unique())

    # Prepare data for PCA by pivoting bias scores into columns
    if group_by == "scenario":
        # For per-scenario analysis, keep both model and scenario
        pivot_df = df.pivot_table(
            index=["model", "scenario"],
            columns="bias",
            values="individual_score",
            aggfunc="mean",
        ).reset_index()
    else:
        # For per-model analysis, aggregate by model
        pivot_df = df.pivot_table(
            index=["model"], columns="bias", values="individual_score", aggfunc="mean"
        ).reset_index()

    # Debug prints after pivot
    if debug:
        print("\nDEBUG: After pivot:")
        print("Shape:", pivot_df.shape)
        print("Columns:", pivot_df.columns.tolist())
        print("Number of unique models:", pivot_df["model"].nunique())
        print("Unique models:", pivot_df["model"].unique())

    # Get bias columns (numeric columns except model/scenario)
    group_cols = ["model", "scenario"] if group_by == "scenario" else ["model"]
    bias_cols = [col for col in pivot_df.columns if col not in group_cols]

    # Debug prints for PCA input
    if debug:
        print("\nDEBUG: PCA input:")
        print("Bias columns:", bias_cols)
        print("Shape of PCA input:", pivot_df[bias_cols].shape)

    try:
        # Scale features if requested
        features = (
            StandardScaler().fit_transform(pivot_df[bias_cols])
            if with_scaling
            else pivot_df[bias_cols]
        )

        # Run PCA on features
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(features)

        # Create result DataFrame with PCA components
        result_df = pd.DataFrame(
            pca_result, columns=[f"PCA Component {i+1}" for i in range(n_components)]
        )

        # Add back grouping columns
        for col in group_cols:
            result_df[col] = pivot_df[col]

        # Debug prints for final result
        if debug:
            print("\nDEBUG: Final PCA result:")
            print("Shape:", result_df.shape)
            print("Columns:", result_df.columns.tolist())
            print("Number of unique models:", result_df["model"].nunique())
            print("Unique models:", result_df["model"].unique())

        # Print explained variance
        if debug:
            print(f"\nExplained variance ratios: {pca.explained_variance_ratio_}")

        return result_df

    except Exception as e:
        if debug:
            print(f"Error performing PCA: {str(e)}")
            print(f"Data shape: {pivot_df[bias_cols].shape}")
        raise e


def reduce_with_umap(
    df: pd.DataFrame,
    n_components: int = 2,
    group_by: str = "scenario",
    debug: bool = False,
    with_scaling: bool = False,
) -> pd.DataFrame:
    """
    Performs UMAP dimensionality reduction on bias scores.
    """
    # Debug prints
    if debug:
        print("\nDEBUG: Initial dataframe:")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("Number of unique models:", df["model"].nunique())
        print("Unique models:", df["model"].unique())

    # Prepare data for UMAP by pivoting bias scores into columns
    if group_by == "scenario":
        # For per-scenario analysis, keep both model and scenario
        pivot_df = df.pivot_table(
            index=["model", "scenario"],
            columns="bias",
            values="individual_score",
            aggfunc="mean",
        ).reset_index()
    else:
        # For per-model analysis, aggregate by model
        pivot_df = df.pivot_table(
            index=["model"], columns="bias", values="individual_score", aggfunc="mean"
        ).reset_index()

    # Debug prints after pivot
    if debug:
        print("\nDEBUG: After pivot:")
        print("Shape:", pivot_df.shape)
        print("Columns:", pivot_df.columns.tolist())
        print("Number of unique models:", pivot_df["model"].nunique())
        print("Unique models:", pivot_df["model"].unique())

    # Get bias columns (numeric columns except model/scenario)
    group_cols = ["model", "scenario"] if group_by == "scenario" else ["model"]
    bias_cols = [col for col in pivot_df.columns if col not in group_cols]

    # Debug prints for UMAP input
    if debug:
        print("\nDEBUG: UMAP input:")
        print("Bias columns:", bias_cols)
        print("Shape of UMAP input:", pivot_df[bias_cols].shape)

    try:
        # Scale features if requested
        features = (
            StandardScaler().fit_transform(pivot_df[bias_cols])
            if with_scaling
            else pivot_df[bias_cols]
        )

        # Run UMAP on features
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        umap_result = reducer.fit_transform(features)

        # Create result DataFrame with UMAP components
        result_df = pd.DataFrame(
            umap_result, columns=[f"UMAP Component {i+1}" for i in range(n_components)]
        )

        # Add back grouping columns
        for col in group_cols:
            result_df[col] = pivot_df[col]

        # Debug prints for final result
        if debug:
            print("\nDEBUG: Final UMAP result:")
            print("Shape:", result_df.shape)
            print("Columns:", result_df.columns.tolist())
            print("Number of unique models:", result_df["model"].nunique())
            print("Unique models:", result_df["model"].unique())

        return result_df

    except Exception as e:
        if debug:
            print(f"Error performing UMAP: {str(e)}")
            print(f"Data shape: {pivot_df[bias_cols].shape}")
        raise e


# Not sure if legacy code
# def cluster_with_kmeans(df: pd.DataFrame, n_clusters: int, scale_first: bool = False):
#     """
#     Clusters the data with K-means.
#     """
#     # Instantiate a K-means instance with the specified number of clusters
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # Fixed: Use n_clusters parameter

#     # Select all numeric columns
#     num_cols = df.select_dtypes(include=['number']).columns

#     # If requested, apply standard scaling to the data first
#     if scale_first:
#         scaler = StandardScaler()
#         df = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)
#     else:
#         df = df[num_cols]

#     # Cluster the data
#     clusters = kmeans.fit_predict(df)

#     return clusters


def cluster_with_hdbscan(df: pd.DataFrame, **kwargs):
    """
    Clusters the data with HDBSCAN.
    """

    # Instantiate an HDBSCAN instance
    hdbcan = HDBSCAN(**kwargs)

    # Select all numeric columns
    num_cols = df.select_dtypes(include=["number"]).columns

    # Cluster the data
    clusters = hdbcan.fit_predict(df[num_cols])

    return clusters


def bin_error_count(count: int) -> str:
    """Bin the error count into broad categories."""
    if count <= 50:
        return "0-50"
    elif count <= 200:
        return "50-200"
    elif count <= 500:
        return "200-500"
    elif count <= 800:
        return "500-800"
    else:
        return "800+"


def bin_error_count_stars(count: int) -> str:
    """Return a compact star marker depending on error count."""
    if count <= 50:
        return ""
    elif count <= 200:
        return "*"
    elif count <= 500:
        return "**"
    elif count <= 800:
        return "***"
    else:
        return "****"


def prepare_error_bins_stars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pivot table of star markers for each (bias, model) pair
    based on the error count tiers.

    Args:
        df: A DataFrame with 'model', 'bias', 'status' columns.

    Returns:
        A pivot table (bias x model) with star notation as strings
        (e.g., "", "*", "**", "***", "****").
    """
    # Filter for error rows
    df_errors = df[df["status"] == "ERROR"].copy()

    # Count the errors by (bias, model)
    error_counts = df_errors.groupby(["bias", "model"]).size().unstack(fill_value=0)

    # Convert counts to star notation
    star_bins = error_counts.applymap(bin_error_count_stars)

    return star_bins


def prepare_error_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pivot table of binned error counts for each (bias, model) pair.

    Args:
        df: A DataFrame with 'model', 'bias', and 'status' columns.

    Returns:
        A pivot table (bias x model) with binned error categories as strings.
    """
    # Filter for error rows
    df_errors = df[df["status"] == "ERROR"].copy()

    # Count the errors by (bias, model)
    error_counts = df_errors.groupby(["bias", "model"]).size().unstack(fill_value=0)

    # Convert counts to bins
    error_bins = error_counts.applymap(bin_error_count)

    return error_bins


def prepare_bias_heatmap_data(
    df: pd.DataFrame, abs: bool = False, add_avg_abs: bool = True, agg: str = "mean"
) -> pd.DataFrame:
    """
    Prepares the data for the bias heatmap by pivoting and calculating averages.

    Args:
        df: DataFrame with columns 'model', 'bias', 'individual_score'
        abs: If True, converts all scores to absolute values
        add_avg_abs: If True, adds an 'Average Absolute' row
        agg: Aggregation function to use ('mean', 'median', etc.)

    Returns:
        DataFrame with bias scores matrix, including averages
    """
    # If requested, convert all scores to absolute values
    if abs:
        df = df.copy()
        df["individual_score"] = df["individual_score"].abs()

    # Pivot the data to create the matrix
    heatmap_data = df.pivot_table(
        values="individual_score", index="bias", columns="model", aggfunc=agg
    )

    # Add the 'Average' column (average of each row)
    heatmap_data["Average"] = heatmap_data.mean(axis=1)

    # Sort the rows by the 'Average' column in descending order
    heatmap_data = heatmap_data.sort_values(by="Average", ascending=False)

    # Add the 'Average' row (average of each column)
    average_row = heatmap_data.mean(axis=0)
    heatmap_data.loc["Average"] = average_row

    # If requested, add 'Average Absolute' row
    if add_avg_abs:
        df_abs = df[["model", "bias", "individual_score"]].copy()
        df_abs["individual_score"] = df_abs["individual_score"].abs()
        df_abs = df_abs.pivot_table(
            values="individual_score", index="bias", columns="model", aggfunc="mean"
        )
        df_abs["Average"] = df_abs.mean(axis=1)
        abs_values = df_abs.mean(axis=0)
        heatmap_data.loc["Average Absolute"] = abs_values

    return heatmap_data


def plot_bias_heatmap(
    df: pd.DataFrame,
    model_order: list[str] = MODEL_ORDER,
    abs: bool = False,
    add_avg_abs: bool = True,
    legend: bool = True,
    agg: str = "mean",
    figsize: tuple[float, float] = (11, 12),
    save_plot: bool = True,
) -> None:
    """
    Plots a heatmap showing the bias scores of all model-bias combinations,
    with star notation indicating error tiers.

    Args:
        df: Original DataFrame with 'model', 'bias', 'individual_score', 'status'
        model_order: The order of columns to display
        abs: If True, uses absolute values for the main scores
        add_avg_abs: If True, includes 'Average Absolute' row
        legend: If True, shows a colorbar
        agg: Aggregation function for pivot table
        figsize: Figure size (width, height)
        save_plot: If True, saves the plot as a PDF
    """
    # ---- 1) Prepare main heatmap data
    heatmap_data = prepare_bias_heatmap_data(
        df, abs=abs, add_avg_abs=add_avg_abs, agg=agg
    )

    # Ensure columns in the desired order
    columns_ordered = list(model_order) + ["Average"]
    if add_avg_abs:
        columns_ordered += ["Average Absolute"]
    columns_ordered = [col for col in columns_ordered if col in heatmap_data.columns]

    heatmap_data = heatmap_data.reindex(columns=columns_ordered)

    # ---- 2) Prepare star-annotated error pivot
    error_star_data = prepare_error_bins_stars(df)
    # Align error pivot to the main data's shape
    error_star_data = error_star_data.reindex(
        index=heatmap_data.index, columns=heatmap_data.columns, fill_value=""
    )

    # ---- 3) Build an annotation DataFrame combining score + star markers
    annot_data = heatmap_data.copy().astype(str)

    for row in heatmap_data.index:
        for col in heatmap_data.columns:
            base_score = heatmap_data.loc[row, col]
            # Convert numeric row/col to string with 2 decimals
            if isinstance(base_score, (float, int)):
                base_score_str = f"{base_score:.2f}"
            else:
                # For average rows/cols
                try:
                    base_score_str = f"{float(base_score):.2f}"
                except:
                    base_score_str = str(base_score)

            star_marker = error_star_data.loc[row, col]
            # Combine the numeric score with star marker. For example:
            #   "0.23 *" or "0.23 ***"
            # If star_marker is empty, we just show the score.
            if star_marker:
                annot_data.loc[row, col] = f"{base_score_str}{star_marker}"
            else:
                annot_data.loc[row, col] = base_score_str

    # ---- 4) Plot the heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # print the heatmap_data in a readable format
    print(heatmap_data)

    # Save as csv with float precsion 2 after the decimal point
    # heatmap_data.round(2).to_csv('heatmap_data.csv')

    sns.heatmap(
        heatmap_data.astype(float).round(2),  # numeric data for color scale
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0,
        annot=annot_data,  # Our combined text
        fmt="",
        vmin=-1.0,
        vmax=1.0,
        cbar=False,
        linewidths=0,
        linecolor="white",
        ax=ax,
    )

    # Remove axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Rotate the column labels by 90 degrees at the top
    ax.xaxis.tick_top()
    plt.xticks(rotation=90, ha="center")

    # Remove black border around the entire plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw white lines around summary rows/cols
    # If you have both 'Average' and 'Average Absolute' at the bottom:
    n_rows, n_cols = heatmap_data.shape
    if add_avg_abs:
        ax.hlines([n_rows - 2], *ax.get_xlim(), color="white", linewidth=5)
    ax.hlines([n_rows - 1], *ax.get_xlim(), color="white", linewidth=5)
    ax.vlines([n_cols - 1], *ax.get_ylim(), color="white", linewidth=5)

    # If requested, add a colorbar legend
    if legend:
        cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])  # [left, bottom, width, height]
        cbar = plt.colorbar(
            ax.collections[0], cax=cbar_ax, orientation="horizontal", label="Bias Score"
        )
        cbar.outline.set_visible(False)

    # Adjust spacing
    plt.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.15)

    # Save the plot if requested
    if save_plot:
        os.makedirs(PLOT_OUTPUT_FOLDER, exist_ok=True)
        plt.savefig(
            os.path.join(PLOT_OUTPUT_FOLDER, "bias_heatmap.pdf"),
            format="pdf",
            bbox_inches="tight",
        )

    plt.title(
        "Bias Heatmap\n(* = 50–200 errors, ** = 200–500, *** = 500–800, **** = 800+)"
    )
    plt.show()


def plot_scatter(
    df: pd.DataFrame,
    x: str = None,
    y: str = None,
    label: str = None,
    save_plot: bool = True,
    plot_type: str = "pca",
    dot_size: float = None,
    dot_alpha: float = None,
    figsize: tuple = None,
    debug: bool = False,
):
    """
    Creates a scatter plot from two columns in a dataframe.

    Args:
        df: DataFrame containing the data
        x: Column name for x-axis (if None, will use default based on plot_type)
        y: Column name for y-axis (if None, will use default based on plot_type)
        label: Column name to use for point colors/labels
        save_plot: Whether to save the plot to file
        plot_type: Type of plot ("pca" or "umap") to determine default component names
        dot_size: Size of scatter plot points
        dot_alpha: Alpha (transparency) of scatter plot points
        figsize: Tuple of (width, height) for the figure size
        debug: Whether to print debug information
    """
    # Debug prints
    if debug:
        print("\nDEBUG: Scatter plot input:")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        if label:
            print(f"Unique {label} values:", sorted(df[label].unique()))
            print(f"Number of unique {label} values:", df[label].nunique())

    # Set default component names based on plot type
    if x is None or y is None:
        prefix = "PCA" if plot_type.lower() == "pca" else "UMAP"
        x = x or f"{prefix} Component 1"
        y = y or f"{prefix} Component 2"

    # Validate inputs
    if x not in df.columns or y not in df.columns:
        raise ValueError(
            f"Columns {x} and/or {y} not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    if label and label not in df.columns:
        # Check for case variations of 'model'
        if label.lower() == "model" and "model" in df.columns:
            label = "model"
        else:
            raise ValueError(f"Label column '{label}' not found in DataFrame")

    # Use default figure size if none provided
    if figsize is None:
        figsize = (8, 7)  # Default smaller size for model plots

    plt.figure(figsize=figsize)

    if label:
        # Use consistent color palette
        n_colors = df[label].nunique()
        if n_colors > 2:
            palette = sns.color_palette("Spectral", n_colors=n_colors)
        else:
            base_models_palette = [
                "#FF1493",
                "#00BFFF",
            ]  # Hot pink and Deep Sky Blue for base models
            # cluster_edge_colors = [
            #     "#2C3E50",
            #     "#8E44AD",
            # ]  # Dark blue and purple for cluster edges
            cluster_edge_colors = [
                "#FF1493",
                "#00BFFF",
            ]  # Hot pink and Deep Sky Blue for base models
            palette = base_models_palette + cluster_edge_colors

        # Create scatter plot with optional size and alpha
        sns.scatterplot(
            data=df,
            x=x,
            y=y,
            hue=label,
            palette=palette,
            s=dot_size if dot_size is not None else 50,  # Default size
            alpha=dot_alpha if dot_alpha is not None else 0.7,  # Default alpha
        )

        # Adjust legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title=label)
    else:
        sns.scatterplot(
            data=df,
            x=x,
            y=y,
            s=dot_size if dot_size is not None else 50,
            alpha=dot_alpha if dot_alpha is not None else 0.7,
        )

    plt.title(f"{x} vs {y}")

    if save_plot:
        os.makedirs(PLOT_OUTPUT_FOLDER, exist_ok=True)
        plt.savefig(
            os.path.join(PLOT_OUTPUT_FOLDER, f"{plot_type}_scatter.pdf"),
            format="pdf",
            bbox_inches="tight",
        )

    try:
        plt.show()
    except Exception as e:
        print(f"Error showing plot: {e}")


def plot_dendrogram(
    data: pd.DataFrame,
    method: str = "complete",
    metric: str = "euclidean",
    n_clusters: int = 12,
    save_plot: bool = True,
    figsize: tuple = None,
    title: str = None,
    debug: bool = False,
):
    """
    Creates a dendrogram plot from a DataFrame.

    Args:
        data: DataFrame with models as index and biases as columns
        method: Linkage method ('complete', 'ward', 'average', etc.)
        metric: Distance metric ('euclidean', 'cosine', etc.)
        n_clusters: Number of clusters to show
        save_plot: Whether to save plot to file
        figsize: Tuple of (width, height) for figure size
        title: Title for the plot
        debug: Whether to print debug information
    """
    if debug:
        print(f"\nCreating dendrogram with method={method}, metric={metric}")
        print("Data shape:", data.shape)

    # Use default figsize if none provided
    if figsize is None:
        figsize = (12, 8)

    # Create linkage matrix
    linkage_matrix = linkage(data, method=method, metric=metric)

    # Create figure
    plt.figure(figsize=figsize)

    # Plot dendrogram
    dendrogram(
        linkage_matrix,
        labels=data.index,
        color_threshold=linkage_matrix[-n_clusters, 2],
    )

    # Set title and labels
    if title:
        plt.title(title)
    plt.xlabel("Models")
    plt.ylabel("Distance")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()

    # Save plot if requested
    if save_plot:
        filename = f"dendrogram_{method}_{metric}.pdf"
        plt.savefig(
            os.path.join(PLOT_OUTPUT_FOLDER, filename),
            format="pdf",
            bbox_inches="tight",
        )

    try:
        plt.show()
    except Exception as e:
        print(f"Error showing plot: {e}")


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    title: str = None,
    figsize: tuple = None,
    save_plot: bool = True,
    debug: bool = False,
):
    """
    Creates a correlation matrix plot.

    Args:
        corr_matrix: Correlation matrix DataFrame
        title: Title for the plot
        figsize: Tuple of (width, height) for figure size
        save_plot: Whether to save plot to file
        debug: Whether to print debug information
    """
    if debug:
        print("\nPlotting correlation matrix")
        print("Matrix shape:", corr_matrix.shape)

    # Use default figsize if none provided
    if figsize is None:
        figsize = (10, 8)

    # Create figure
    plt.figure(figsize=figsize)

    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap="RdBu", vmin=-1, vmax=1, center=0)

    # Set title if provided
    if title:
        plt.title(title)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Adjust layout
    plt.tight_layout()

    # Save plot if requested
    if save_plot:
        filename = f"correlation_matrix{'_' + title.lower().replace(' ', '_') if title else ''}.pdf"
        plt.savefig(
            os.path.join(PLOT_OUTPUT_FOLDER, filename),
            format="pdf",
            bbox_inches="tight",
        )

    try:
        plt.show()
    except Exception as e:
        print(f"Error showing plot: {e}")


def plot_correlation_matrix_with_dendrogram(
    corr_matrix: pd.DataFrame,
    title: str = None,
    figsize: tuple = None,
    save_plot: bool = True,
    debug: bool = False,
):
    """
    Creates a correlation matrix plot with a dendrogram.

    Args:
        corr_matrix: Correlation matrix DataFrame
        title: Title for the plot
        figsize: Tuple of (width, height) for figure size
        save_plot: Whether to save plot to file
        debug: Whether to print debug information
    """
    if debug:
        print("\nPlotting correlation matrix with dendrogram")
        print("Matrix shape:", corr_matrix.shape)

    # Use default figsize if none provided
    if figsize is None:
        figsize = (12, 10)

    # Calculate linkage
    linkage_matrix = linkage(corr_matrix, method="complete", metric="euclidean")

    # Create figure with gridspec
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.25, 1], height_ratios=[0.25, 1])

    # Add dendrogram at top
    ax_top = fig.add_subplot(gs[0, 1])
    dendrogram(linkage_matrix, ax=ax_top, labels=corr_matrix.columns)
    ax_top.set_xticks([])

    # Add dendrogram at left
    ax_left = fig.add_subplot(gs[1, 0])
    dendrogram(
        linkage_matrix, ax=ax_left, labels=corr_matrix.columns, orientation="left"
    )
    ax_left.set_yticks([])

    # Add heatmap
    ax_heatmap = fig.add_subplot(gs[1, 1])
    sns.heatmap(
        corr_matrix, annot=True, cmap="RdBu", vmin=-1, vmax=1, center=0, ax=ax_heatmap
    )

    # Set title if provided
    if title:
        fig.suptitle(title)

    # Rotate labels
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0)

    # Adjust layout
    plt.tight_layout()

    # Save plot if requested
    if save_plot:
        filename = f"correlation_matrix_dendrogram{'_' + title.lower().replace(' ', '_') if title else ''}.pdf"
        plt.savefig(
            os.path.join(PLOT_OUTPUT_FOLDER, filename),
            format="pdf",
            bbox_inches="tight",
        )

    try:
        plt.show()
    except Exception as e:
        print(f"Error showing plot: {e}")


def plot_bubble_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    size: str,
    color: str,
    label: str,
    xlim: tuple[float] = None,
    ylim: tuple[float] = None,
    legendloc: str = "lower right",
    alpha: float = 1.0,
    label_offset: dict = {},
    save_plot: bool = True,
):
    """
    Creates a bubble plot with the x and y axis representing the given columns, bubble size based on the 'size' column, and bubble color based on the 'color' column.
    """

    # Ensure the color column is treated as categorical
    df[color] = df[color].astype("category")

    # Get unique colors and assign a consistent color for each category
    unique_colors = df[color].cat.categories
    color_mapping = {cat: idx for idx, cat in enumerate(unique_colors)}

    # Create a colormap
    cmap = plt.get_cmap("Spectral", len(unique_colors))

    # Create a color list to ensure consistent colors for both plot and legend
    color_list = [cmap(color_mapping[cat]) for cat in df[color]]

    # Create the bubble plot
    plt.figure(figsize=(7, 5.5))
    scatter = plt.scatter(
        df[x],
        df[y],
        s=df[size] * 10,
        c=color_list,
        alpha=alpha,
        edgecolor="lightgrey",
        linewidth=0.5,
    )

    # Add labels to each bubble
    for i, row in df.iterrows():
        offset = label_offset[row[label]] if row[label] in label_offset else (0.0, 0.0)
        plt.text(
            row[x] + offset[0],
            row[y] + offset[1],
            str(row[label]),
            fontsize=9,
            ha="center",
            va="center",
            color="black",
        )

    # Create a legend with categorical labels, using the same colormap
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=cmap(i / len(unique_colors)),
            linestyle="",
            markersize=10,
            markeredgecolor="lightgrey",
            markeredgewidth=0.5,
            alpha=alpha,
        )
        for i in range(len(unique_colors))
    ]
    plt.legend(handles, unique_colors, title=color, loc=legendloc)

    # Set plot labels and title
    plt.xlabel(x)
    plt.ylabel(y)

    # Manually set the axis limits, if provided
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)

    # Save the plot
    if save_plot:
        plt.savefig(
            os.path.join(PLOT_OUTPUT_FOLDER, "bubble_plot.pdf"),
            format="pdf",
            bbox_inches="tight",
        )

    # Show the plot
    try:
        plt.show()
    except Exception as e:
        print(f"Error showing plot: {e}")


def calculate_model_embeddings(
    df: pd.DataFrame, debug: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates PCA and UMAP embeddings for model-level analysis.

    Args:
        df: DataFrame with columns ['model', 'scenario', 'bias', 'individual_score']
        debug: Whether to print debug information

    Returns:
        Tuple of (df_pca, df_umap) DataFrames containing the embeddings
    """
    # Group data by model
    df_grouped = group_by_model(df)

    # Generate dimensionality reductions
    df_pca = reduce_with_pca(df_grouped, group_by="model", debug=debug)
    df_umap = reduce_with_umap(df_grouped, group_by="model", debug=debug)

    return df_pca, df_umap


def plot_model_embeddings(
    df_pca: pd.DataFrame,
    df_umap: pd.DataFrame,
    save_plots: bool = True,
    dot_size: float = 50,
    dot_alpha: float = 1.0,
    figsize: tuple = None,
    debug: bool = False,
    with_scaling: bool = False,
):
    """
    Creates plots from pre-calculated PCA and UMAP embeddings.

    Args:
        df_pca: DataFrame with PCA embeddings
        df_umap: DataFrame with UMAP embeddings
        save_plots: Whether to save the plots to files
        dot_size: Size of scatter plot points
        dot_alpha: Alpha (transparency) of scatter plot points
        figsize: Tuple of (width, height) for the figure size
        debug: Whether to print debug information
        with_scaling: Whether to use scaling for UMAP
    """
    plot_params = {
        "dot_size": dot_size,
        "dot_alpha": dot_alpha,
        "save_plot": save_plots,
        "figsize": figsize,
        "debug": debug,
        "with_scaling": with_scaling,
    }

    # Create plots
    for df_plot, plot_type in [(df_pca, "pca"), (df_umap, "umap")]:
        # Model plot
        df_scatter = df_plot.copy()
        df_scatter["Model"] = df_scatter["model"]
        plot_scatter(df_scatter, label="Model", plot_type=plot_type, **plot_params)

        # Developer plot
        df_scatter = df_plot.copy()
        df_scatter["Developer"] = df_scatter["model"].map(MODEL_DEVELOPER_MAPPING)
        plot_scatter(df_scatter, label="Developer", plot_type=plot_type, **plot_params)


def create_model_plots(
    df: pd.DataFrame,
    save_plots: bool = True,
    dot_size: float = 50,
    dot_alpha: float = 1.0,
    figsize: tuple = None,
    debug: bool = False,
    with_scaling: bool = False,  # Main control point for scaling
):
    """
    Creates all model-level plots with configurable scaling.
    """
    # Calculate embeddings with scaling flag
    df_pca = reduce_with_pca(df, debug=debug, with_scaling=with_scaling)
    df_umap = reduce_with_umap(df, debug=debug, with_scaling=with_scaling)

    # Create plots
    plot_model_embeddings(
        df_pca,
        df_umap,
        save_plots=save_plots,
        dot_size=dot_size,
        dot_alpha=dot_alpha,
        figsize=figsize,
        debug=debug,
        with_scaling=with_scaling,  # Pass down scaling flag
    )


def calculate_correlation_matrices(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates correlation matrices for both raw and grouped bias data.

    Args:
        df: DataFrame with columns ['model', 'bias', 'individual_score']

    Returns:
        Tuple of (raw_corr_matrix, grouped_corr_matrix)
    """
    # Calculate raw correlations
    pivot_raw = df.pivot_table(
        index="scenario", columns="bias", values="individual_score", aggfunc="mean"
    )
    raw_corr = pivot_raw.corr()

    # Calculate grouped correlations
    df_grouped = group_by_model(df)
    pivot_grouped = df_grouped.pivot_table(
        index="model", columns="bias", values="individual_score", aggfunc="mean"
    )
    grouped_corr = pivot_grouped.corr()

    return raw_corr, grouped_corr


def add_certainty_and_belief_to_pivot_df(
    pivot_df: pd.DataFrame,
    file_path_certainty_and_belief: str = "data/cross_tuning_full_scores.csv",
) -> pd.DataFrame:
    """
    Adds certainty and belief columns to the pivot_df.

    Args:
        pivot_df: DataFrame with columns ['model', 'bias', 'individual_score'] shape is (n_models, n_biases)
        file_path_certainty_and_belief: Path to the file containing certainty and belief data

    Returns:
        DataFrame with certainty and belief columns added
    """

    print(
        f"Adding certainty and belief to pivot_df from manually added rows in file: {file_path_certainty_and_belief}"
    )
    # Create a copy to avoid modifying the original
    pivot_df = pivot_df.copy()

    # Load certainty and belief data
    loaded_certainty_and_belief = pd.read_csv(file_path_certainty_and_belief)

    # two rows from the end of the dataframe not include the MMLU row
    certainty_and_belief = loaded_certainty_and_belief.iloc[-3:-1].T.iloc[
        1:
    ]  # shape is (n_models, 2)

    # Add the certainty and belief columns to the pivot_df
    pivot_df["Certainty"] = certainty_and_belief.iloc[:, 0]
    pivot_df["Belief"] = certainty_and_belief.iloc[:, 1]

    return pivot_df


def get_feature_matrix_for_clustering(
    df: pd.DataFrame,
    level: str = "model_bias",
    exclude_models: list = None,
    add_certainty_and_belief: bool = False,
) -> pd.DataFrame:
    """
    Prepares data for hierarchical clustering at model-bias, model-bias-scenario, or model-bias-sample level.

    Args:
        df: DataFrame with columns ['model', 'bias', 'individual_score'] and optionally 'scenario'
        level: Analysis level - 'model_bias', 'model_bias_scenario', or 'model_bias_sample'
        exclude_models: List of model names to exclude from clustering
        with_scaling: Whether to scale the features
    Returns:
        DataFrame with models as index and features (biases/scenarios/samples) as columns
    """
    valid_levels = ["model_bias", "model_bias_scenario", "model_bias_sample"]

    if level not in valid_levels:
        raise ValueError(f"level must be one of {valid_levels}")
    # Create a copy to avoid modifying original
    df = df.copy()
    if level == "model_bias":
        # Group by model and bias
        df_grouped = group_by_model(df)

        # Pivot for clustering
        pivot_df = df_grouped.pivot_table(
            index="model", columns="bias", values="individual_score", aggfunc="mean"
        )

        # Add two external biases from file (Certainty and Belief)
        if add_certainty_and_belief:
            pivot_df = add_certainty_and_belief_to_pivot_df(pivot_df)
    elif level == "model_bias_scenario":
        # Use scenario-level data
        pivot_df = df.pivot_table(
            index="model",
            columns=["bias", "scenario"],
            values="individual_score",
            # aggfunc='first' # BUG: Should be 'mean'
            aggfunc="mean",
        )

        # Since some scenarios have Nan values for some models, we may need to fill them with the mean of the model-bias
        if pivot_df.isna().sum().sum() > 0:
            # Fill NaN values with the mean of the mean accross all the models and biases
            # pivot_df = pivot_df.fillna(pivot_df.mean())
            # Fill NaN values with the mean of the model-bias
            # pivot_df = pivot_df.fillna(pivot_df.mean(axis=1))
            raise ValueError(
                "There are still NaN values in the dataframe in function get_feature_matrix_for_clustering."
            )

        # Flatten multi-level columns
        pivot_df.columns = [f"{bias}_{scenario}" for bias, scenario in pivot_df.columns]

    else:  # model_bias_sample
        # Filter out rows where individual_score is NaN
        df = df.dropna(subset=["individual_score"])

        # Create a unique identifier for each valid sample within each bias-scenario group
        df["sample_id"] = df.groupby(
            ["bias", "scenario"]
        ).cumcount()  # cumcount() is used to create a unique identifier for each sample

        # Get the maximum sample_id for each bias-scenario combination to ensure consistent features
        max_samples = df.groupby(["bias", "scenario"])["sample_id"].max().min()

        # Filter to keep only samples up to the minimum max_sample_id
        # This ensures we have the same number of samples for each bias-scenario combination
        df = df[df["sample_id"] <= max_samples]

        # Pivot using all three levels
        pivot_df = df.pivot_table(
            index="model",
            columns=["bias", "scenario", "sample_id"],
            values="individual_score",
            # aggfunc='first' # BUG: Should be 'mean'
            aggfunc="mean",
        )

        # Flatten multi-level columns
        pivot_df.columns = [
            f"{bias}_{scenario}_sample{sample}"
            for bias, scenario, sample in pivot_df.columns
        ]

    # Exclude specified models
    if exclude_models:
        pivot_df = pivot_df[~pivot_df.index.isin(exclude_models)]

    # Handle any remaining NaN values by imputing with column mean
    # Should not be needed, because we imputed or removed missing values before
    # pivot_df = pivot_df.fillna(pivot_df.mean())
    if df.isna().sum().sum() > 0:
        raise ValueError("There are still NaN values in the dataframe.")

    # Save the transposed pivot_df to a csv file
    pivot_df.T.to_csv(f"./plots/clustering/pivot_df_{level}.csv")
    pivot_df.T.to_csv(f"./data/cross_tuning_scores_no_MMLU.csv")

    return pivot_df


def assign_model_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns pretraining and instruction groups to models based on their names.

    Args:
        df: DataFrame with 'model' column

    Returns:
        DataFrame with added 'pretraining_group' and 'instruction_group' columns
    """
    # Create copy to avoid modifying original
    df = df.copy()

    # Map model names to pretraining groups
    pretraining_map = {
        "OLMo": "OLMo",
        "T5": "T5",
        "Flan-T5": "T5",
        "Random": "Random",
        "Mistral": "Mistral",
        "Llama2": "Llama2",
    }

    # Map model names to instruction groups
    instruction_map = {
        "Flan": "Flan",
        "Tulu": "Tulu",
        "SFT": "Tulu",  # SFT is a type of Tulu
        "Random": "Random",
        "ShareGPT": "ShareGPT",
        "Vicuna": "ShareGPT",
    }

    # Assign pretraining group
    df["pretraining_group"] = df["model"].apply(
        lambda x: next(
            (group for key, group in pretraining_map.items() if key in x), "Unknown"
        )
    )

    # Assign instruction group
    df["instruction_group"] = df["model"].apply(
        lambda x: next(
            (group for key, group in instruction_map.items() if key in x), "Unknown"
        )
    )

    # print the instruction group per model
    print(df[["model", "instruction_group"]])

    # Assign model developer group
    df["model_developer"] = df["model"].apply(
        lambda x: next(
            (group for key, group in MODEL_DEVELOPER_MAPPING.items() if key in x),
            "Unknown",
        )
    )

    return df


def prepare_clustering_data(
    df: pd.DataFrame,
    level: str = "model_bias",
    debug: bool = False,
    add_certainty_and_belief: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares data for clustering analysis at specified level.

    Args:
        df: DataFrame with model bias data
        level: Analysis level - 'model_bias', 'model_bias_scenario', or 'model_bias_sample'
        debug: Whether to print debug information

    Returns:
        Tuple of (feature_matrix, model_metadata):
        - feature_matrix: DataFrame with models as index and features as columns
        - model_metadata: DataFrame with model information including groups
    """
    valid_levels = ["model_bias", "model_bias_scenario", "model_bias_sample"]
    if level not in valid_levels:
        raise ValueError(f"level must be one of {valid_levels}")

    if debug:
        print(f"\nPreparing clustering data at {level} level...")

    # Calculate clustering data
    feature_matrix = get_feature_matrix_for_clustering(
        df, level=level, add_certainty_and_belief=add_certainty_and_belief
    )

    if debug:
        print(f"Feature matrix shape: {feature_matrix.shape}")
        print("Models:", feature_matrix.index.tolist())
        if level == "model_bias_sample":
            print(f"Number of sample-level features: {len(feature_matrix.columns)}")

    # Get model metadata with group assignments
    model_metadata = pd.DataFrame(index=feature_matrix.index)
    model_metadata["model"] = model_metadata.index
    model_metadata = assign_model_groups(model_metadata)

    if debug:
        print("\nModel groups:")
        print(
            "Pretraining groups:", model_metadata["pretraining_group"].unique().tolist()
        )
        print(
            "Instruction groups:", model_metadata["instruction_group"].unique().tolist()
        )

    return feature_matrix, model_metadata


def evaluate_clustering_quality(data: np.ndarray, labels: np.ndarray) -> dict:
    """
    Evaluates clustering quality using multiple metrics.

    Args:
        data: Array of features used for clustering
        labels: Array of cluster assignments

    Returns:
        Dictionary with evaluation metrics
    """
    try:
        metrics = {
            "silhouette": silhouette_score(data, labels),
            "calinski_harabasz": calinski_harabasz_score(data, labels),
            "davies_bouldin": davies_bouldin_score(data, labels),
        }
    except ValueError as e:
        print(f"Warning: Could not calculate clustering metrics: {e}")
        metrics = {
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }

    return metrics


def plot_clustering_results(
    feature_matrix: pd.DataFrame,
    labels: np.ndarray,
    model_names: pd.Series,
    title: str,
    granularity_level: str = "model_bias",
    save_plot: bool = True,
    figsize: tuple = (12, 9),
    with_scaling: bool = False,
    output_dir: str = "./plots/clustering",
) -> None:
    """
    Creates a professional visualization showing:
    1. Pretrained model (OLMo/T5, through fill color)
    2. Instruction type (Tulu/Flan, through marker shape)
    3. Clustering results (through edge color)
    4. Model identifier (through text labels)
    """
    cluster_edge_colors = [
        "#FF1493",
        "#00BFFF",
        "#000080",
        "#FFD700",
    ]  # Hot pink and Deep Sky Blue and Navy Blue and Gold

    def get_model_attributes(model_name: str) -> tuple[str, str, str]:
        """Extract pretrained model, instruction type, and identifier."""
        # Get pretrained model (OLMo or T5)
        # pretrained = "OLMo" if "OLMo" in model_name else "T5"
        if "OLMo" in model_name:
            pretrained = "OLMo"
        elif "Llama2" in model_name:
            pretrained = "Llama2"
        elif "Mistral" in model_name:
            pretrained = "Mistral"
        else:
            pretrained = "T5"

        # Get instruction type (Tulu or Flan)
        if "Tulu" in model_name or model_name == "OLMo-SFT":
            instruction = "Tulu"
        elif "ShareGPT" in model_name:
            instruction = "ShareGPT"
        else:  # Flan or Flan-T5
            instruction = "Flan"

        # Get identifier (seed number or 'Org' for original models)
        if "Llama2" in model_name or "Mistral" in model_name:
            identifier = MODEL_DEVELOPER_MAPPING[model_name]
            if "Seed" in model_name:
                identifier += f"-S{model_name.split('Seed-')[1]}"
        elif model_name in ["OLMo-SFT", "Flan-T5"]:
            identifier = "Org"
        elif "Seed" not in model_name:
            identifier = "S0"
        else:
            identifier = f"S{model_name.split('Seed-')[1]}"

        return pretrained, instruction, identifier

    def get_marker_style(instruction: str) -> str:
        """Get marker style based on instruction type."""
        return {
            "Tulu": "o",  # Circle for Tulu
            "Flan": "^",  # Triangle for Flan
            "ShareGPT": "s",  # Square for ShareGPT
        }[instruction]

    def setup_plot_style() -> None:
        """Set up the plot style for a professional appearance."""
        plt.style.use("default")
        plt.rcParams.update(
            {
                "font.size": 26,
                "axes.labelsize": 24,
                "axes.titlesize": 26,
                "xtick.labelsize": 20,
                "ytick.labelsize": 20,
            }
        )

    def create_scatter_plot(
        coords: np.ndarray, cluster_labels: np.ndarray, model_names: pd.Series
    ) -> None:
        """Create scatter plot with all model attributes visualized."""
        # Define colors for different attributes
        pretrain_colors = {
            # 'OLMo': '#FF6B6B',  # Coral red for OLMo
            # 'T5': '#4ECDC4',     # Turquoise for T5
            # "OLMo": "#E64A19",  # Deep orange for OLMo
            # "T5": "#1976D2",  # Rich blue for T5
            # "Mistral": "#7B1FA2",  # Deep purple for Mistral
            # "Llama2": "#00796B",  # Teal for Llama2
            "OLMo": "#FF6B6B",
            "T5": "#4ECDC4",
            "Mistral": "#4ECDC4",
            "Llama2": "#FF6B6B",
        }
        # cluster_edge_colors = [
        #     "#2C3E50",
        #     "#8E44AD",
        # ]  # Dark blue and purple for cluster edges
        cluster_edge_colors = [
            "#FF1493",
            "#00BFFF",
        ]  # Hot pink and Deep Sky Blue for base models
        # Use the colors defined at function level
        # nonlocal cluster_edge_colors

        # Create scatter plot for each model
        for i, (coord, cluster_label, model) in enumerate(
            zip(coords, cluster_labels, model_names)
        ):
            # Get model attributes
            pretrained, instruction, identifier = get_model_attributes(model)

            # Plot point
            plt.scatter(
                coord[0],
                coord[1],
                c=[pretrain_colors[pretrained]],
                marker=get_marker_style(instruction),
                # s=500,
                s=1000,
                alpha=0.7,
                edgecolor=cluster_edge_colors[cluster_label],
                linewidth=3,  # Thicker edge
                label=f"{pretrained}-{instruction}",
            )

            # Add identifier label
            plt.annotate(
                identifier,
                (coord[0], coord[1]),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=16,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
            )

    def _create_dynamic_legend_elements(
        model_names: pd.Series, cluster_labels: np.ndarray
    ) -> list:
        """
        Generates legend elements only for attributes present in the plotted data.

        Args:
            model_names: Series of model names being plotted.
            cluster_labels: Array of cluster assignments for the models.

        Returns:
            A list of Matplotlib Line2D objects for the legend.
        """

        # 1. Get unique attributes present in the data
        present_pretrained = set()
        present_instruction = set()
        for model in model_names:
            pretrained, instruction, _ = get_model_attributes(model)
            present_pretrained.add(pretrained)
            present_instruction.add(instruction)

        # 2. Define master lists of attributes and their styles
        all_pretrain_styles = {
            # "OLMo": "#E64A19",
            # "T5": "#1976D2",
            # "Mistral": "#7B1FA2",
            # "Llama2": "#00796B",
            "OLMo": "#FF1493",
            "T5": "#00BFFF",
            "Mistral": "#FF1493",
            "Llama2": "#00BFFF",
        }
        all_instruction_styles = {
            "Tulu": "o",
            "Flan": "^",
            "ShareGPT": "s",
        }
        # Use cluster_edge_colors from the outer scope

        # 3. Build legend elements based on present attributes
        legend_elements = []

        # Pretrained model legend (fill colors)
        legend_elements.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    label=ptype,
                    markersize=15,
                    markeredgewidth=0,
                )
                for ptype, color in all_pretrain_styles.items()
                if ptype in present_pretrained  # Only include if present
            ]
        )

        # Instruction type legend (markers)
        legend_elements.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="gray",
                    label=itype,
                    markersize=15,
                    linestyle="None",  # Ensure no line connects markers
                )
                for itype, marker in all_instruction_styles.items()
                if itype in present_instruction  # Only include if present
            ]
        )

        # Cluster legend (edge colors)
        num_unique_clusters = len(np.unique(cluster_labels))
        legend_elements.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="w",
                    markeredgecolor=cluster_edge_colors[
                        i % len(cluster_edge_colors)
                    ],  # Cycle colors if needed
                    label=f"Cluster {i+1}",
                    markersize=15,
                    markeredgewidth=4,
                )
                for i in range(
                    num_unique_clusters
                )  # Create entry for each unique cluster
            ]
        )

        return legend_elements

    # 1. Prepare data and create plot
    features = (
        StandardScaler().fit_transform(feature_matrix)
        if with_scaling
        else feature_matrix
    )
    coords, pca = PCA(n_components=2).fit_transform(features), PCA(n_components=2).fit(
        features
    )

    # 2. Set up plot
    setup_plot_style()
    fig = plt.figure(figsize=figsize, dpi=300)

    # 3. Create main scatter plot
    create_scatter_plot(coords, labels, model_names)

    # 4. Add titles and axis labels
    level_str_map = {
        "model_bias": "Bias Level",
        "model_bias_scenario": "Scenario Level",
        "model_bias_sample": "Sample Level",
    }
    level_str = level_str_map.get(granularity_level, "Unknown Level")
    plot_title = f"{title} - {level_str}"

    plt.title(plot_title, pad=20)
    plt.xlabel(f"PC1 \n({pca.explained_variance_ratio_[0]:.1%} of variance)")
    plt.ylabel(f"PC2 \n({pca.explained_variance_ratio_[1]:.1%} of variance)")

    # Set the limits of the plot to be the same for both axes
    plt.xlim(coords[:, 0].min() - 2, coords[:, 0].max() + 2)
    plt.ylim(coords[:, 1].min() - 2, coords[:, 1].max() + 2)
    if granularity_level == "model_bias":  # distances are small
        plt.xlim(coords[:, 0].min() - 0.2, coords[:, 0].max() + 0.2)
        plt.ylim(coords[:, 1].min() - 0.1, coords[:, 1].max() + 0.1)

    # 6. Create and add the dynamic legend
    legend_elements = _create_dynamic_legend_elements(model_names, labels)
    plt.legend(
        handles=legend_elements,
        title="Model Attributes",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=20,  # Larger legend font
        title_fontsize=24,  # Larger legend title
    )

    # 7. Add grid and finalize
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    # 8. Save plot
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        filename = (
            f"clustering_{granularity_level}_{title.lower().replace(' ', '_')}.pdf"
        )
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight", dpi=300)

    try:
        plt.show()
    except Exception as e:
        print(f"Error showing plot: {e}")
    plt.close()


def plot_silhouette_scores(
    metrics: dict,
    title: str,
    granularity_level: str = "model_bias",
    save_plot: bool = True,
    figsize: tuple = (10, 8),
    with_scaling: bool = False,
    output_dir: str = "./plots/clustering",
):
    """
    Plots silhouette scores vs. number of clusters.
    """
    # Extract silhouette scores from metrics dictionary
    silhouette_scores = [metric["silhouette"] for metric in metrics.values()]
    k_range = range(2, 2 + len(silhouette_scores))

    # Create plot
    # plt.figure(figsize=figsize)

    # Plot silhouette scores with seaborn
    sns.lineplot(
        x=k_range,
        y=silhouette_scores,
        marker="o",
        label="Silhouette Score",
        color="blue",
        linewidth=2,
        markersize=8,
        alpha=0.7,
    )

    # Add title and axis labels
    plt.title(title)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs. Number of Clusters")
    plt.grid(True)
    plt.xticks(list(k_range))
    plt.legend()

    plt.tight_layout()

    if save_plot:
        # Save as PDF
        plt.savefig(
            os.path.join(
                output_dir,
                f"silhouette_scores_{granularity_level}_{title.lower().replace(' ', '_')}.pdf",
            ),
            format="pdf",
            bbox_inches="tight",
        )

    # Display in notebook
    try:
        plt.show()
    except Exception as e:
        print(f"Error showing plot: {e}")
    plt.close()


def plot_similarity_matrix(
    feature_matrix: pd.DataFrame,
    model_metadata: pd.DataFrame,
    title: str,
    granularity_level: str = "model_bias",
    save_plot: bool = True,
    output_dir: str = "./plots/clustering",
    figsize: tuple = (12, 10),
    font_scale: float = 1.2,
    annotation_fmt: str = ".2f",
):
    """
    Plots a similarity matrix of the feature matrix with models sorted by pretraining group.

    Args:
        feature_matrix: DataFrame with features for each model
        model_metadata: DataFrame with model information including 'pretraining_group'
        title: Title for the plot
        granularity_level: Analysis granularity level for file naming
        save_plot: Whether to save plots to files
        output_dir: Directory for saving plot files
        figsize: Figure size tuple (width, height)
        font_scale: Scale factor for font sizes
        annotation_fmt: Format string for annotation values
    """
    # Set up the visual style
    sns.set_style("white")
    sns.set_context("paper", font_scale=font_scale)

    # Custom colormap configuration
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    def create_similarity_heatmap(matrix: pd.DataFrame, subtitle: str) -> None:
        """Helper function to create a single similarity matrix heatmap."""
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap with enhanced styling
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            center=0,
            annot=True,
            fmt=annotation_fmt,
            square=True,
            cbar_kws={
                "shrink": 0.8,
                "label": "Similarity Score",
                "orientation": "horizontal",
            },
            annot_kws={"size": 12},
            mask=np.triu(np.ones_like(matrix), k=0),  # Show only lower triangle
        )

        # Customize axes
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        # Add titles
        main_title = f"{title}\n{subtitle}"
        plt.title(main_title, pad=20, fontsize=14, fontweight="bold")

        # Adjust layout
        plt.tight_layout()

        # Save plot if requested
        if save_plot:
            filename = f"similarity_matrix_{granularity_level}_{subtitle.lower().replace(' ', '_')}.pdf"
            filepath = os.path.join(output_dir, filename)
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            try:
                plt.show()
            except Exception as e:
                print(f"Error showing plot: {e}")
            plt.close()

    # Sort models by pretraining group
    model_metadata = pd.DataFrame(
        {
            "model": model_metadata.index,
            "pretraining_group": [
                (
                    "OLMo"
                    if "OLMo" in name
                    else (
                        "T5"
                        if "T5" in name
                        else (
                            "Mistral"
                            if "Mistral" in name
                            else "Llama2" if "Llama2" in name else "Unknown"
                        )
                    )
                )
                for name in model_metadata.index
            ],
        }
    ).set_index("model")
    sorted_models = model_metadata.sort_values(["pretraining_group", "model"]).index

    # Reorder feature matrix according to sorted models
    feature_matrix_sorted = feature_matrix.reindex(index=sorted_models)

    # Compute similarity matrix
    cosine_sim = cosine_similarity(feature_matrix_sorted)

    # Convert to DataFrame with sorted model names
    cosine_sim_df = pd.DataFrame(
        cosine_sim,
        index=feature_matrix_sorted.index,
        columns=feature_matrix_sorted.index,
    )

    # Create similarity matrix plot
    create_similarity_heatmap(cosine_sim_df, "Pretraining Model Similarity")

    # Reset matplotlib style
    plt.style.use("default")


def plot_cluster_bias_heatmap(
    feature_matrix: pd.DataFrame,
    model_metadata: pd.DataFrame,
    cluster_labels: np.ndarray,
    cluster_names: list[str],  # New parameter for cluster names
    title: str,
    granularity_level: str = "model_bias",
    font_scale: float = 1.2,
    save_plot: bool = True,
    output_dir: str = "./plots/clustering",
    figsize: tuple = (20, 12),
) -> None:
    """
    Creates a professional heatmap visualization showing mean bias scores for each cluster.

    Args:
        feature_matrix: DataFrame with bias features (columns) for each model (rows)
        model_metadata: DataFrame with model information
        cluster_labels: Array of cluster assignments
        cluster_names: List of cluster names (e.g., ["T5", "OLMo"])
        title: Title for the plot
        granularity_level: Analysis level ('model_bias', 'model_bias_scenario', or 'model_bias_sample')
        font_scale: Scale factor for font sizes
        save_plot: Whether to save the plot to file
        output_dir: Directory for saving plot files
        figsize: Figure size tuple (width, height)
    """
    # Set up the visual style
    plt.style.use("default")
    sns.set_theme(style="white", font_scale=font_scale)

    # Create figure with adjusted size for many biases
    plt.figure(figsize=figsize)

    def aggregate_features(data: pd.DataFrame, level: str) -> pd.DataFrame:
        """Aggregates features based on granularity level."""
        if level == "model_bias":
            return data
        else:
            bias_names = [col.split("_")[0] for col in data.columns]
            return data.groupby(bias_names, axis=1).mean()

    # Aggregate features if needed
    agg_features = aggregate_features(feature_matrix, granularity_level)

    # Calculate mean bias scores per cluster
    cluster_means = []
    cluster_stds = []
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_means.append(agg_features[cluster_mask].mean())
        cluster_stds.append(agg_features[cluster_mask].std())

    # Create DataFrame with mean scores using cluster names
    heatmap_data = pd.DataFrame(
        cluster_means,
        index=[
            f"{name} Cluster\n(n={np.sum(cluster_labels == i)})"
            for i, name in enumerate(cluster_names)
        ],
        columns=agg_features.columns,
    )

    # Create custom diverging colormap
    colors = sns.color_palette("RdBu_r", n_colors=256)
    custom_cmap = LinearSegmentedColormap.from_list("custom", colors)

    # Calculate value range for symmetric color scaling
    abs_max = max(abs(heatmap_data.min().min()), abs(heatmap_data.max().max()))

    # Create heatmap with adjusted parameters for many biases
    ax = sns.heatmap(
        heatmap_data,
        cmap=custom_cmap,
        center=0,
        vmin=-abs_max,
        vmax=abs_max,
        annot=True,
        fmt=".2f",
        square=False,
        cbar_kws={
            "shrink": 0.3,
            "label": "Mean Bias Score",
            "orientation": "horizontal",
            "pad": 0.4,  # Adjust colorbar padding
        },
        annot_kws={"size": 36},
    )

    # Customize appearance
    plt.title(title, pad=30, fontsize=36, fontweight="bold")

    # Rotate x-axis labels and add more space for them
    plt.xticks(rotation=90, ha="right", fontsize=30)
    plt.yticks(rotation=0, fontsize=30)

    # Add gridlines
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="white", linewidth=1.5)

    # Adjust layout to prevent label cutoff while maintaining space for colorbar
    plt.tight_layout()

    # Save plot if requested
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"cluster_bias_heatmap_{granularity_level}_{title.lower().replace(' ', '_')}.pdf"
        plt.savefig(
            os.path.join(output_dir, filename),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )

    try:
        plt.show()
    except Exception as e:
        print(f"Error showing plot: {e}")
    plt.close()


def plot_clustering_results_all(
    feature_matrix: pd.DataFrame,
    unsupervised_clustering_results: dict,
    supervised_clustering_results: dict,
    model_metadata: pd.DataFrame,
    granularity_level: str = "model_bias",
    save_plots: bool = True,
    debug: bool = False,
    with_scaling: bool = False,
    output_dir: str = "./plots/clustering",
):
    """
    Creates all clustering visualization plots at specified level.

    Args:
        feature_matrix: DataFrame with features used for clustering
        clustering_results: Dictionary with clustering results
        model_metadata: DataFrame with model information
        level: Analysis level - either 'model_bias' or 'model_bias_scenario'
        save_plots: Whether to save plots
        debug: Whether to print debug information
        with_scaling: Whether to use scaling for UMAP
        output_dir: Output directory for plots
    """
    if debug:
        print(f"\nCreating clustering visualizations at {granularity_level} level...")

    # Plot best unsupervised clustering results
    plot_clustering_results(
        feature_matrix,
        unsupervised_clustering_results["best"]["labels"],
        model_metadata["model"],
        # f'Unsupervised Clustering Results (best) ({granularity_level})',
        f"Best K-Means Clustering",
        granularity_level=granularity_level,
        save_plot=save_plots,
        with_scaling=with_scaling,
        output_dir=output_dir,
    )

    # Plot median unsupervised clustering results
    plot_clustering_results(
        feature_matrix,
        unsupervised_clustering_results["median"]["labels"],
        model_metadata["model"],
        # f'Unsupervised Clustering Results (median) ({granularity_level})',
        f"K-Means Clustering",
        granularity_level=granularity_level,
        save_plot=save_plots,
        with_scaling=with_scaling,
        output_dir=output_dir,
    )

    # Plot Silhouette scores vs. number of clusters in unsupervised clustering results
    plot_silhouette_scores(
        unsupervised_clustering_results["all_k_best"]["metrics"],
        # f'Silhouette Scores vs. Number of Clusters (best) ({granularity_level})',
        f"Silhouette Scores vs. Number of Clusters",
        granularity_level=granularity_level,
        save_plot=save_plots,
        with_scaling=with_scaling,
        output_dir=output_dir,
    )

    plot_silhouette_scores(
        unsupervised_clustering_results["all_k_median"]["metrics"],
        # f'Silhouette Scores vs. Number of Clusters (median) ({granularity_level})',
        f"Silhouette Scores vs. Number of Clusters",
        granularity_level=granularity_level,
        save_plot=save_plots,
        with_scaling=with_scaling,
        output_dir=output_dir,
    )

    # Plot pretraining groups
    plot_clustering_results(
        feature_matrix,
        supervised_clustering_results["pretraining"]["labels"],
        model_metadata["model"],
        # f'Pretraining Model Groups ({granularity_level})',
        f"Pretraining Model Groups",
        granularity_level=granularity_level,
        save_plot=save_plots,
        with_scaling=with_scaling,
        output_dir=output_dir,
    )

    # Plot instruction groups
    plot_clustering_results(
        feature_matrix,
        supervised_clustering_results["instruction"]["labels"],
        model_metadata["model"],
        # f'Instruction Data Groups ({granularity_level})',
        f"Instruction Data Groups",
        granularity_level=granularity_level,
        save_plot=save_plots,
        with_scaling=with_scaling,
        output_dir=output_dir,
    )

    # Plot similarity matrix
    plot_similarity_matrix(
        feature_matrix,
        model_metadata["model"],
        # f'Similarity Matrix ({granularity_level})',
        f"Similarity Matrix",
        granularity_level=granularity_level,
        save_plot=save_plots,
        output_dir=output_dir,
        figsize=(12, 10),
        font_scale=1.2,
        annotation_fmt=".2f",
    )

    # Plot a heatmap to visualize the bias patterns across clusters
    plot_cluster_bias_heatmap(
        feature_matrix,
        model_metadata["model"],
        cluster_labels=supervised_clustering_results["pretraining"]["labels"],
        cluster_names=model_metadata["pretraining_group"].unique(),
        # title=f'Pretraining clusters bias heatmap ({granularity_level})',
        title=f"Pretraining clusters bias heatmap",
        granularity_level=granularity_level,
        save_plot=save_plots,
        output_dir=output_dir,
        figsize=(20, 10),
        font_scale=1.2,
    )


def run_hierarchical_clustering_analysis(
    df: pd.DataFrame, level: str = "model_bias", debug: bool = False
) -> dict:
    """
    Runs hierarchical clustering analysis with multiple methods and metrics.

    Args:
        df: DataFrame with model bias data
        level: Analysis level - either 'model_bias' or 'model_bias_scenario'
        debug: Whether to print debug information

    Returns:
        Dictionary with clustering results for each method/metric combination
    """
    # Prepare data
    feature_matrix, model_metadata = prepare_clustering_data(
        df, level=level, debug=debug
    )

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    results = {
        "feature_matrix": feature_matrix,
        "model_metadata": model_metadata,
        "scaled_features": scaled_features,
        "linkage_matrices": {},
    }

    if debug:
        print(f"\nPerforming hierarchical clustering at {level} level...")

    return results


def plot_clustering_analysis(
    cluster_data: dict,
    methods: list = ["complete", "ward"],
    metrics: list = ["euclidean", "cosine"],
    n_clusters: int = 12,
    save_plots: bool = True,
    figsize: tuple = (14, 10),
    granularity_level: str = "model_bias",
    output_dir: str = "./plots/clustering",
    debug: bool = False,
):
    """
    Creates hierarchical clustering visualizations.

    Args:
        cluster_data: Dictionary with clustering results
        methods: List of linkage methods to use
        metrics: List of distance metrics to use
        n_clusters: Number of clusters to show in dendrogram
        save_plots: Whether to save plots
        figsize: Figure size tuple
        granularity_level: Granularity level of the clustering
        output_dir: Output directory for plots
        debug: Whether to print debug information
    """
    for method in methods:
        # Ward method only works with Euclidean distance
        if method == "ward":
            current_metrics = ["euclidean"]
        else:
            current_metrics = metrics

        for metric in current_metrics:
            if debug:
                print(
                    f"\nPlotting dendrogram with method={method}, metric={metric} at {granularity_level} level"
                )

            try:
                # Calculate linkage matrix
                linkage_matrix = linkage(
                    cluster_data["scaled_features"], method=method, metric=metric
                )

                # Create plot
                fig = plt.figure(figsize=figsize)
                dendrogram(
                    linkage_matrix,
                    labels=cluster_data["model_metadata"]["model"],
                    color_threshold=n_clusters,
                    leaf_rotation=90,
                )
                plt.title(
                    f"Hierarchical Clustering\nMethod: {method}, Metric: {metric} at {granularity_level} level"
                )
                plt.tight_layout()

                if save_plots:
                    # Save as PDF
                    plt.savefig(
                        os.path.join(
                            output_dir,
                            f"dendrogram_{method}_{metric}_{granularity_level}.pdf",
                        ),
                        format="pdf",
                        bbox_inches="tight",
                    )

                # Display in notebook
                try:
                    plt.show()
                except Exception as e:
                    print(f"Error showing plot: {e}")
                plt.close()

            except ValueError as e:
                print(
                    f"Warning: Could not create dendrogram for method={method}, "
                    f"metric={metric}: {str(e)}"
                )
                continue


def analyze_cluster_distribution(
    scaled_features: np.ndarray,
    model_metadata: pd.DataFrame,
    kmeans: KMeans,
    labels: np.ndarray,
    feature_matrix: pd.DataFrame,
    with_scaling: bool = True,
) -> dict:
    """
    Analyzes the distribution of clusters and their characteristics.

    Args:
        scaled_features: Scaled feature matrix
        model_metadata: DataFrame with model information
        kmeans: Fitted KMeans model
        labels: Cluster assignments
        feature_matrix: Original feature matrix with column names
        with_scaling: Whether scaling was applied

    Returns:
        Dictionary containing cluster analysis results
    """
    # 1. Calculate cluster assignments and distances
    cluster_assignments = pd.DataFrame(
        {
            "Model": model_metadata["model"],
            "Cluster": labels,
            "Distance_to_Center": [
                np.linalg.norm(scaled_features[i] - kmeans.cluster_centers_[label])
                for i, label in enumerate(labels)
            ],
        }
    )
    cluster_assignments = cluster_assignments.sort_values("Cluster")

    # 2. Calculate cluster sizes
    cluster_sizes = {}
    for cluster in range(kmeans.n_clusters):
        size = (labels == cluster).sum()
        cluster_sizes[cluster] = size

    # 3. Calculate average distances within clusters
    cluster_distances = {}
    for cluster in range(kmeans.n_clusters):
        distances = cluster_assignments[cluster_assignments["Cluster"] == cluster][
            "Distance_to_Center"
        ]
        cluster_distances[cluster] = {"mean": distances.mean(), "std": distances.std()}

    # 4. Calculate feature importance for each cluster
    cluster_features = {}
    if with_scaling:
        feature_names = feature_matrix.columns
        for cluster in range(kmeans.n_clusters):
            center = kmeans.cluster_centers_[cluster]
            feature_importance = pd.Series(center, index=feature_names).sort_values(
                key=abs, ascending=False
            )
            cluster_features[cluster] = feature_importance

    # Print analysis results
    print("\nCluster Distribution Analysis:")
    print("-" * 30)

    print("\nCluster Assignments and Distances to Center:")
    print(cluster_assignments.to_string())

    print("\nCluster Sizes:")
    for cluster, size in cluster_sizes.items():
        print(f"Cluster {cluster}: {size} models")

    print("\nAverage Distances within Clusters:")
    for cluster, stats in cluster_distances.items():
        print(f"Cluster {cluster}: {stats['mean']:.3f} (std: {stats['std']:.3f})")

    if with_scaling:
        for cluster, features in cluster_features.items():
            print(f"\nTop characterizing features for Cluster {cluster}:")
            print(features.head().to_string())

    return {
        "assignments": cluster_assignments,
        "sizes": cluster_sizes,
        "distances": cluster_distances,
        "features": cluster_features,
    }


def print_clustering_evaluation(metrics: dict, title: str = None) -> None:
    """
    Prints clustering evaluation metrics.

    Args:
        metrics: Dictionary containing evaluation metrics
        title: Optional title to print before metrics
    """
    if title:
        print(f"\n{title}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")


def perform_unsupervised_clustering_analysis(
    feature_matrix: pd.DataFrame,
    model_metadata: pd.DataFrame,
    seed: int = 42,
    n_clusters: int = 2,
    with_scaling: bool = True,
    debug: bool = False,
    n_kmeans_trials: int = 30,
) -> dict:
    """
    Performs unsupervised clustering analysis with enhanced analysis.
    """
    # Scale features
    scaler = StandardScaler() if with_scaling else None
    scaled_features = (
        scaler.fit_transform(feature_matrix) if with_scaling else feature_matrix
    )

    def run_single_kmeans_trial(n_clusters: int, n_init: int, random_state: int):
        # Perform k-means clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=n_init,
            random_state=random_state,
        )
        # kmeans = KMeans(n_clusters=n_clusters, random_state=seed) # Used originaly
        labels = kmeans.fit_predict(scaled_features)

        # Calculate metrics
        metrics = evaluate_clustering_quality(scaled_features, labels)
        return labels, metrics, kmeans.cluster_centers_

    all_metrics = {}
    all_labels = {}
    all_cluster_centers = {}
    k_range = range(n_clusters, 6)
    for k in k_range:
        all_metrics[k] = []
        all_labels[k] = []
        all_cluster_centers[k] = []
        # Run multiple trials and select the best result
        for trial in range(n_kmeans_trials):
            labels, metrics, cluster_centers = run_single_kmeans_trial(
                k, n_init=1, random_state=seed + trial
            )
            all_metrics[k].append(metrics)
            all_labels[k].append(labels)
            all_cluster_centers[k].append(cluster_centers)

    # Choose best and median labels based on silhouette score
    best_metrics = {}
    median_metrics = {}
    best_labels = {}
    median_labels = {}
    best_cluster_centers = {}
    median_cluster_centers = {}
    for k in k_range:
        best_metrics[k] = max(all_metrics[k], key=lambda x: x["silhouette"])
        best_labels[k] = all_labels[k][all_metrics[k].index(best_metrics[k])]
        best_cluster_centers[k] = all_cluster_centers[k][
            all_metrics[k].index(best_metrics[k])
        ]

        # Choose median metrics according to median silhouette score

        # Find the index of the median silhouette score
        all_silhouette_scores = [
            all_metrics[k][i]["silhouette"] for i in range(n_kmeans_trials)
        ]
        median_score = np.median(all_silhouette_scores)
        median_silhouette_score_index = np.argmin(
            np.abs(np.array(all_silhouette_scores) - median_score)
        )
        # Choose the metrics of the median silhouette score index
        median_metrics[k] = all_metrics[k][median_silhouette_score_index]
        median_labels[k] = all_labels[k][median_silhouette_score_index]
        median_cluster_centers[k] = all_cluster_centers[k][
            median_silhouette_score_index
        ]

    if debug:
        print("\nUnsupervised Clustering Analysis:")
        print("---------------------------------")

        print("\nClustering Metrics:")
        # print_clustering_evaluation(metrics)
        print(f"best_metrics:\n{best_metrics[2]}")
        print(f"median_metrics:\n{median_metrics[2]}")

    return {
        "best": {
            "labels": best_labels[n_clusters],
            "metrics": best_metrics[n_clusters],
            "cluster_centers": best_cluster_centers[n_clusters],
        },
        "median": {
            "labels": median_labels[n_clusters],
            "metrics": median_metrics[n_clusters],
            "cluster_centers": median_cluster_centers[n_clusters],
        },
        "all_k_best": {
            "labels": best_labels,
            "metrics": best_metrics,
            "cluster_centers": best_cluster_centers,
        },
        "all_k_median": {
            "labels": median_labels,
            "metrics": median_metrics,
            "cluster_centers": median_cluster_centers,
        },
    }


def perform_supervised_clustering_analysis(
    feature_matrix: pd.DataFrame,
    model_metadata: pd.DataFrame,
    seed: int = 42,
    n_clusters: int = 2,
    num_random_assignment: int = 5,
    with_scaling: bool = True,
    debug: bool = False,
    models_to_include: list = None,
) -> dict:
    """
    Performs supervised clustering analysis using model groups.

    Args:
        feature_matrix: DataFrame with features for clustering
        model_metadata: DataFrame with model information
        n_clusters: Number of clusters (not used, kept for API consistency)
        num_random_assignment: Number of random assignments to average
        debug: Whether to print debug information

    Returns:
        Dictionary with clustering results
    """
    # Scale features
    scaler = StandardScaler() if with_scaling else None
    feature_matrix = (
        scaler.fit_transform(feature_matrix) if with_scaling else feature_matrix
    )

    results = {}

    # Evaluate pretraining groups
    pretraining_labels = pd.Categorical(model_metadata["pretraining_group"]).codes
    pretraining_metrics = evaluate_clustering_quality(
        feature_matrix, pretraining_labels
    )

    # Evaluate instruction groups
    instruction_labels = pd.Categorical(model_metadata["instruction_group"]).codes
    instruction_metrics = evaluate_clustering_quality(
        feature_matrix, instruction_labels
    )

    # Evaluate model developer groups
    if "Llama2" in models_to_include and "Mistral" in models_to_include:
        model_developer_labels = pd.Categorical(model_metadata["model_developer"]).codes
        model_developer_metrics = evaluate_clustering_quality(
            feature_matrix, model_developer_labels
        )

    # Random baseline metrics
    all_random_labels = []
    all_random_metrics = []
    for _ in range(num_random_assignment):
        random_labels = np.random.permutation(pretraining_labels)
        metrics = evaluate_clustering_quality(feature_matrix, random_labels)
        all_random_labels.append(random_labels)
        all_random_metrics.append(metrics)

    # Calculate mean random metrics
    mean_random_metrics = {
        key: np.mean([m[key] for m in all_random_metrics])
        for key in all_random_metrics[0].keys()
    }

    if debug:
        print("\nSupervised Clustering Results:")
        print_clustering_evaluation(pretraining_metrics, "Pretraining Groups")
        print_clustering_evaluation(instruction_metrics, "Instruction Groups")
        print_clustering_evaluation(
            mean_random_metrics, "Random Assignment (mean of 5 runs)"
        )

    # Structure results dictionary with clustering metrics and labels
    results = {
        "pretraining": {"labels": pretraining_labels, "metrics": pretraining_metrics},
        "instruction": {"labels": instruction_labels, "metrics": instruction_metrics},
        "random_mean": {"labels": all_random_labels, "metrics": mean_random_metrics},
        "random_runs": {
            f"run_{i}": {"labels": labels, "metrics": metrics}
            for i, (labels, metrics) in enumerate(
                zip(all_random_labels, all_random_metrics)
            )
        },
    }
    if "Llama2" in models_to_include and "Mistral" in models_to_include:
        results["model_developer"] = {
            "labels": model_developer_labels,
            "metrics": model_developer_metrics,
        }
    return results


def analyze_label_agreement(
    unsupervised_labels: np.ndarray, supervised_results: dict, debug: bool = False
) -> dict:
    """
    Analyzes the agreement between unsupervised clustering labels and other labelings.
    ARI is the Adjusted Rand Index, which measures the agreement between two partitions.
    NMI is the Normalized Mutual Information, which measures the agreement between two partitions.

    Args:
        unsupervised_labels: Labels from KMeans clustering
        supervised_results: Dictionary containing other clustering results
        debug: Whether to print debug information

    Returns:
        Dictionary with agreement scores
    """
    agreement_scores = {}

    # Compare with pretraining groups
    pretraining_labels = supervised_results["pretraining"]["labels"]
    agreement_scores["pretraining"] = {
        "ari": adjusted_rand_score(unsupervised_labels, pretraining_labels),
        "nmi": normalized_mutual_info_score(unsupervised_labels, pretraining_labels),
    }

    # Compare with instruction groups
    instruction_labels = supervised_results["instruction"]["labels"]
    agreement_scores["instruction"] = {
        "ari": adjusted_rand_score(unsupervised_labels, instruction_labels),
        "nmi": normalized_mutual_info_score(unsupervised_labels, instruction_labels),
    }

    # Compare with random assignments
    random_agreements = []
    for run_key, run_data in supervised_results["random_runs"].items():
        if run_data["labels"] is not None:  # Check if labels exist
            random_agreements.append(
                {
                    "ari": adjusted_rand_score(unsupervised_labels, run_data["labels"]),
                    "nmi": normalized_mutual_info_score(
                        unsupervised_labels, run_data["labels"]
                    ),
                }
            )

    # Calculate mean random agreement
    if random_agreements:
        agreement_scores["random_mean"] = {
            "ari": np.mean([scores["ari"] for scores in random_agreements]),
            "nmi": np.mean([scores["nmi"] for scores in random_agreements]),
        }

    if debug:
        print("\nLabel Agreement Analysis:")
        print("-" * 30)
        print("\nAgreement with Pretraining Groups:")
        print(f"ARI: {agreement_scores['pretraining']['ari']:.3f}")
        print(f"NMI: {agreement_scores['pretraining']['nmi']:.3f}")

        print("\nAgreement with Instruction Groups:")
        print(f"ARI: {agreement_scores['instruction']['ari']:.3f}")
        print(f"NMI: {agreement_scores['instruction']['nmi']:.3f}")

        if "random_mean" in agreement_scores:
            print("\nMean Agreement with Random Assignments:")
            print(f"ARI: {agreement_scores['random_mean']['ari']:.3f}")
            print(f"NMI: {agreement_scores['random_mean']['nmi']:.3f}")

    return agreement_scores


def analyze_cluster_features(
    feature_matrix: pd.DataFrame, labels: np.ndarray, model_metadata: pd.DataFrame
) -> dict:
    """
    Analyzes which features (biases) contribute most to cluster separation.

    Args:
        feature_matrix: DataFrame with bias features
        labels: Cluster assignments
        model_metadata: DataFrame with model information

    Returns:
        Dictionary with feature importance analysis
    """
    analysis = {}

    # Calculate mean feature values for each cluster
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_models = model_metadata.loc[cluster_mask, "model"]

        # Get mean feature values for this cluster
        cluster_means = feature_matrix[cluster_mask].mean()
        # Get feature standard deviations for this cluster
        cluster_stds = feature_matrix[cluster_mask].std()

        # Calculate feature importance as absolute difference from overall mean
        overall_means = feature_matrix.mean()
        feature_importance = abs(cluster_means - overall_means)

        # Store results
        analysis[cluster_id] = {
            "models": cluster_models.tolist(),
            "size": cluster_mask.sum(),
            "distinctive_features": feature_importance.nlargest(5).to_dict(),
            "mean_values": cluster_means.to_dict(),
            "std_values": cluster_stds.to_dict(),
        }

    return analysis


def analyze_model_similarities(
    feature_matrix: pd.DataFrame, labels: np.ndarray, model_metadata: pd.DataFrame
) -> dict:
    """
    Analyzes similarities between models within and across clusters.

    Args:
        feature_matrix: DataFrame with bias features
        labels: Cluster assignments
        model_metadata: DataFrame with model information

    Returns:
        Dictionary with similarity analysis
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Calculate pairwise similarities
    similarities = cosine_similarity(feature_matrix)
    sim_df = pd.DataFrame(
        similarities, index=model_metadata["model"], columns=model_metadata["model"]
    )

    analysis = {"within_cluster": {}, "between_cluster": {}, "outliers": {}}

    # Analyze within-cluster similarities
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_models = model_metadata.loc[cluster_mask, "model"]

        # Get similarities just within this cluster
        cluster_sims = sim_df.loc[cluster_models, cluster_models]

        # Calculate mean similarity for each model in cluster
        mean_sims = cluster_sims.mean()

        # Identify potential outliers (models with low similarity to cluster)
        outliers = mean_sims[mean_sims < mean_sims.mean() - mean_sims.std()]

        analysis["within_cluster"][cluster_id] = {
            "mean_similarity": cluster_sims.values[
                np.triu_indices_from(cluster_sims.values, k=1)
            ].mean(),
            "std_similarity": cluster_sims.values[
                np.triu_indices_from(cluster_sims.values, k=1)
            ].std(),
            "model_similarities": mean_sims.to_dict(),
        }

        if len(outliers) > 0:
            analysis["outliers"][cluster_id] = outliers.to_dict()

    # Analyze between-cluster similarities
    for c1 in np.unique(labels):
        for c2 in np.unique(labels):
            if c1 < c2:  # Only do each pair once
                c1_mask = labels == c1
                c2_mask = labels == c2
                c1_models = model_metadata.loc[c1_mask, "model"]
                c2_models = model_metadata.loc[c2_mask, "model"]

                between_sims = sim_df.loc[c1_models, c2_models]

                analysis["between_cluster"][(c1, c2)] = {
                    "mean_similarity": between_sims.values.mean(),
                    "std_similarity": between_sims.values.std(),
                    "closest_pairs": get_top_pairs(between_sims, n=3),
                }

    return analysis


def get_top_pairs(similarity_matrix: pd.DataFrame, n: int = 3) -> list:
    """Helper function to get top N most similar pairs of models."""
    # Flatten the matrix and get top N pairs
    pairs = []
    flat_sims = similarity_matrix.unstack()
    top_pairs = flat_sims.nlargest(n)

    for (model1, model2), sim in top_pairs.items():
        pairs.append({"models": (model1, model2), "similarity": sim})

    return pairs


def validate_clustering_labels(
    feature_matrix: np.ndarray,
    model_metadata: pd.DataFrame,
    unsupervised_best_results: dict,
    unsupervised_median_results: dict,
    supervised_results: dict,
    n_permutations: int = 500,
    debug: bool = False,
) -> dict:
    """
    Evaluates whether pretraining labels capture real structure in the data using statistical tests.
    """
    # Input validation
    if not isinstance(feature_matrix, (np.ndarray, pd.DataFrame)):
        raise TypeError("feature_matrix must be numpy array or pandas DataFrame")

    # Convert feature matrix to numpy array if needed
    features = (
        feature_matrix.values if hasattr(feature_matrix, "values") else feature_matrix
    )

    # Validate supervised results structure
    required_keys = ["pretraining", "instruction"]
    if not all(key in supervised_results for key in required_keys):
        raise ValueError("supervised_results missing required keys")

    # Validate labels
    pretraining_labels = supervised_results["pretraining"]["labels"]
    instruction_labels = supervised_results["instruction"]["labels"]

    if (
        len(pretraining_labels) != len(instruction_labels)
        or len(pretraining_labels) != features.shape[0]
    ):
        raise ValueError("Inconsistent number of labels")

    def compute_clustering_scores(labels):
        """Computes clustering quality metrics for given labels."""
        return {
            "silhouette": silhouette_score(feature_matrix, labels),
            "calinski_harabasz": calinski_harabasz_score(feature_matrix, labels),
            "davies_bouldin": davies_bouldin_score(feature_matrix, labels),
        }

    def compute_intra_inter_distances(labels):
        """Computes intra-cluster and inter-cluster distances."""
        distances = euclidean_distances(feature_matrix)
        intra_distances = []
        inter_distances = []

        for i, j in combinations(range(len(labels)), 2):
            if labels[i] == labels[j]:
                intra_distances.append(distances[i, j])
            else:
                inter_distances.append(distances[i, j])

        return np.array(intra_distances), np.array(inter_distances)

    def permutation_test(true_labels, metric_func):
        """
        Performs a **vector-aware** permutation test where labels are shuffled,
        and clustering metrics are recomputed on the feature matrix.
        """
        from joblib import Parallel, delayed  # For parallelization

        true_score = metric_func(true_labels)
        permuted_scores = []

        # Parallelized permutation test to speed up computation
        def permute_once():
            perm_labels = np.random.permutation(true_labels)  # Shuffle labels
            return metric_func(perm_labels)

        permuted_scores = Parallel(n_jobs=-1)(
            delayed(permute_once)() for _ in range(n_permutations)
        )

        # Compute p-value (how often permuted score >= true score)
        p_value = np.mean(np.array(permuted_scores) >= true_score)

        return {
            "true_score": true_score,
            "mean_permuted": np.mean(permuted_scores),
            "std_permuted": np.std(permuted_scores),
            "p_value": p_value,
        }

    def manova_test(labels):
        """
        Performs MANOVA test to check if labels explain significant variance.
        """
        # Convert feature matrix to numpy array if it isn't already
        features = (
            feature_matrix.values
            if hasattr(feature_matrix, "values")
            else feature_matrix
        )

        # Split data by labels
        unique_labels = np.unique(labels)
        groups = [features[labels == label] for label in unique_labels]

        # Perform one-way ANOVA for each feature dimension
        f_stats = []
        p_values = []

        # Check if we have enough samples in each group
        if any(len(group) < 2 for group in groups):
            return {
                "f_statistics": np.nan,
                "p_values": 1.0,  # Worst possible p-value
                "significant_dims": 0,
            }

        for dim in range(features.shape[1]):
            try:
                # Extract the dim-th feature for each group
                feature_groups = [group[:, dim] for group in groups]
                f_stat, p_val = f_oneway(*feature_groups)

                # Handle NaN results
                if np.isnan(f_stat):
                    f_stat = 0
                    p_val = 1.0

                f_stats.append(f_stat)
                p_values.append(p_val)
            except Exception as e:
                if debug:
                    print(f"Warning: ANOVA failed for dimension {dim}: {str(e)}")
                f_stats.append(0)
                p_values.append(1.0)

        return {
            "f_statistics": np.mean(f_stats),
            "p_values": np.mean(p_values),
            "significant_dims": np.sum(np.array(p_values) < 0.05),
        }

    def distance_distribution_test(labels):
        """
        Tests if intra-cluster distances are significantly smaller than inter-cluster distances.
        """
        intra_dist, inter_dist = compute_intra_inter_distances(labels)

        # Perform Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(intra_dist, inter_dist)

        return {
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "mean_intra_dist": np.mean(intra_dist),
            "mean_inter_dist": np.mean(inter_dist),
            "std_intra_dist": np.std(intra_dist),
            "std_inter_dist": np.std(inter_dist),
        }

    if debug:
        print("\nPerforming statistical validation of clustering labels...")

    # Initialize results dictionary
    validation_results = {"pretraining": {}, "instruction": {}}

    # Get labels
    pretraining_labels = supervised_results["pretraining"]["labels"]
    instruction_labels = supervised_results["instruction"]["labels"]

    all_labels_for_permutation_tests = {
        "pretraining": pretraining_labels,
        "instruction": instruction_labels,
    }

    # Add model developer validation if it exists
    if "model_developer" in supervised_results:
        validation_results["model_developer"] = {}
        model_developer_labels = supervised_results["model_developer"]["labels"]
        all_labels_for_permutation_tests["model_developer"] = model_developer_labels

    metrics_to_test = ["silhouette", "calinski_harabasz", "davies_bouldin"]

    for label_type, labels in all_labels_for_permutation_tests.items():
        if debug:
            print(f"\nAnalyzing {label_type} labels...")
            print(f"Running permutation tests for each clustering metric...")

        # 1. Permutation tests for each clustering metric
        validation_results[label_type]["permutation_tests"] = {
            metric: permutation_test(
                labels, lambda x: compute_clustering_scores(x)[metric]
            )
            for metric in metrics_to_test
        }

        if debug:
            print(f"Running MANOVA test...")

        # 2. MANOVA test
        validation_results[label_type]["manova"] = manova_test(labels)

        if debug:
            print(f"Running distance distribution test...")

    all_random_mean_labels = supervised_results["random_mean"]["labels"]
    all_labels_for_distance_test = {
        "pretraining": pretraining_labels,
        "instruction": instruction_labels,
        "best": unsupervised_best_results["labels"],
        "median": unsupervised_median_results["labels"],
        "random_mean": all_random_mean_labels,
    }

    if "model_developer" in supervised_results:
        all_labels_for_distance_test["model_developer"] = model_developer_labels

    # 3. Distance distribution test for all types of labels
    for label_type, labels in all_labels_for_distance_test.items():
        # Take the mean of the distance distribution test for all random labels
        if label_type == "random_mean":
            validation_results[label_type] = {}
            all_random_distance_tests = []
            for labels in all_random_mean_labels:
                validation_results[label_type]["distance_test"] = (
                    distance_distribution_test(labels)
                )
                all_random_distance_tests.append(
                    validation_results[label_type]["distance_test"]
                )
            validation_results[label_type]["distance_test"] = {
                "ks_statistic": np.mean(
                    [test["ks_statistic"] for test in all_random_distance_tests]
                ),
                "p_value": np.mean(
                    [test["p_value"] for test in all_random_distance_tests]
                ),
                "mean_intra_dist": np.mean(
                    [test["mean_intra_dist"] for test in all_random_distance_tests]
                ),
                "mean_inter_dist": np.mean(
                    [test["mean_inter_dist"] for test in all_random_distance_tests]
                ),
            }
        else:
            if label_type not in validation_results:
                validation_results[label_type] = {}
            validation_results[label_type]["distance_test"] = (
                distance_distribution_test(labels)
            )

    if debug:
        print(f"\n{label_type.capitalize()} Results:")
        print("Permutation test p-values:")
        for metric, results in validation_results[label_type][
            "permutation_tests"
        ].items():
            print(f"  {metric}: {results['p_value']:.4f}")
        print(
            f"MANOVA p-value: {validation_results[label_type]['manova']['p_values']:.4f}"
        )
        print(
            f"Distance test p-value: {validation_results[label_type]['distance_test']['p_value']:.4f}"
        )

    return validation_results


def variance_decomposition_anova(
    feature_matrix: pd.DataFrame, model_metadata: pd.DataFrame, debug: bool = False
) -> dict:
    """
    Decomposes variance in bias patterns using ANOVA to quantify the influence of pretraining model and instruction dataset.

    Args:
        feature_matrix: DataFrame with bias features for each model.
        model_metadata: DataFrame with 'pretraining_group' and 'instruction_group' columns.
        debug: Whether to print debug information.

    Returns:
        anova_results: Dictionary with variance decomposition results.
    """
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    # Prepare data for ANOVA
    data = feature_matrix.copy()
    data["pretraining_group"] = model_metadata["pretraining_group"]
    data["instruction_group"] = model_metadata["instruction_group"]

    # Melt the data to long format for ANOVA
    melted_data = data.melt(
        id_vars=["pretraining_group", "instruction_group"],
        var_name="bias",
        value_name="score",
    )

    # Fit the ANOVA model
    model = ols(
        "score ~ C(pretraining_group) * C(instruction_group)", data=melted_data
    ).fit()
    anova_results = anova_lm(model, typ=2)

    # Calculate variance explained by each factor
    total_ss = anova_results["sum_sq"].sum()
    pretraining_variance = (
        anova_results.loc["C(pretraining_group)", "sum_sq"] / total_ss * 100
    )
    instruction_variance = (
        anova_results.loc["C(instruction_group)", "sum_sq"] / total_ss * 100
    )
    interaction_variance = (
        anova_results.loc["C(pretraining_group):C(instruction_group)", "sum_sq"]
        / total_ss
        * 100
    )

    # Extract p-values
    p_values = {
        "pretraining": anova_results.loc["C(pretraining_group)", "PR(>F)"],
        "instruction": anova_results.loc["C(instruction_group)", "PR(>F)"],
        "interaction": anova_results.loc[
            "C(pretraining_group):C(instruction_group)", "PR(>F)"
        ],
    }

    if debug:
        print("\nANOVA Results:")
        print(anova_results)
        print(f"\nPretraining Variance: {pretraining_variance:.2f}%")
        print(f"Instruction Variance: {instruction_variance:.2f}%")
        print(f"Interaction Variance: {interaction_variance:.2f}%")
        print("P-values:", p_values)

    return {
        "pretraining_variance": pretraining_variance,
        "instruction_variance": instruction_variance,
        "interaction_variance": interaction_variance,
        "p_values": p_values,
    }
