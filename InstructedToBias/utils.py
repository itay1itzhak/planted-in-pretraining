from pathlib import Path
import os

INSTURCT_MODELS = [
    "text-davinci-002",
    "text-davinci-003",
    "gpt-4-0314",
    "flan-t5-small",
    "flan-t5-base",
    "flan-t5-large",
    "flan-t5-xl",
    "flan-t5-xxl",
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "Llama-2-70b-chat",
    "Mistral-7B-Instruct",
    "OLMo-7B-SFT",
    "OLMo-7B-Instruct",
    "OLMo-7B-SFT-quantized",
    "OLMo-7B-Instruct-quantized",
    "OLMo-7B-Tulu-lora",
    "OLMo-7B-Tulu-lora-r64a128",
    "OLMo-7B-Flan-lora",
    "OLMo-7B-Flan-lora-r64a128",
    "T5-Flan-lora-1e-4",
    "T5-Flan-lora-5e-5",
    "T5-Tulu-lora-1e-4",
    "T5-Tulu-lora-5e-5",
]
VANILLA_MODELS = [
    "davinci",
    "t5-xl",
    "t5-3b",
    "t5-v1_1-xl",
    "t5-v1_1-xxl",
    "t5-v1_1-small",
    "llama_7B",
    "llama_13B",
    "llama-7b",
    "llama-13b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Mistral-7B",
    "OLMo-7B",
]

OPENAI_MODELS = [
    "gpt-4-0314",
    "gpt-4",
    "text-davinci-003",
    "text-davinci-002",
    "text-davinci-001",
    "davinci",
    "curie",
    "babbage",
]

T5_MODELS = [
    "t5-xl",
    "t5-3b",
    "t5-v1_1-small",
    "t5-v1_1-base",
    "t5-v1_1-large",
    "t5-v1_1-xl",
    "t5-v1_1-xxl",
]

FLAN_T5_MODELS = [
    "flan-t5-small",
    "flan-t5-base",
    "flan-t5-large",
    "flan-t5-xl",
    "flan-t5-xxl",
    "T5-Flan-lora-1e-4",
    "T5-Flan-lora-5e-5",
]

TULU_T5_MODELS = [
    "T5-Tulu-lora-1e-4",
    "T5-Tulu-lora-5e-5",
]

LLAMA_MODELS = [
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
]

LLAMA_CHAT_MODELS = [
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "Llama-2-70b-chat",
]

MISTRAL_MODELS = [
    "Mistral-7B",
]

MISTRAL_INSTRUCT_MODELS = [
    "Mistral-7B-Instruct",
]

OLMO_MODELS = [
    "OLMo-7B",
]

OLMO_INSTRUCT_MODELS = [
    "OLMo-7B-SFT",
    "OLMo-7B-Instruct",
    "OLMo-7B-SFT-quantized",
    "OLMo-7B-Instruct-quantized",
    "OLMo-7B-Tulu-lora",
    "OLMo-7B-Tulu-lora-r64a128",
    # "OLMo-7B-Flan-lora",
]

OLMO_FLAN_INSTRUCT_MODELS = ["OLMo-7B-Flan-lora", "OLMo-7B-Flan-lora-r64a128"]


def get_map_model_names():
    return {
        "davinci": "DaVinci",
        "text-davinci-002": "DaVinci-002",
        "text-davinci-003": "DaVinci-003",
        "gpt-4-0314": "GPT-4",
        "gpt-4": "GPT-4",
        "text-davinci-001": "DaVinci-001",
        "t5-v1_1-small": "T5-Small",
        "t5-v1_1-base": "T5-Base",
        "t5-v1_1-large": "T5-Large",
        "t5-v1_1-xl": "T5-XL",
        "t5-v1_1-xxl": "T5-XXL",
        "flan-t5-small": "Flan-T5-Small",
        "flan-t5-base": "Flan-T5-Base",
        "flan-t5-large": "Flan-T5-Large",
        "flan-t5-xl": "Flan-T5-XL",
        "flan-t5-xxl": "Flan-T5-XXL",
        "Llama-2-7b": "Llama-7B",
        "Llama-2-13b": "Llama-13B",
        "Llama-2-70b": "Llama-70B",
        "Llama-2-7b-chat": "Llama-7B-Chat",
        "Llama-2-13b-chat": "Llama-13B-Chat",
        "Llama-2-70b-chat": "Llama-70B-Chat",
        "Mistral-7B": "Mistral",
        "Mistral-7B-Instruct": "Mistral-Inst",
        "OLMo-7B": "OLMo",
        "OLMo-7B-SFT": "OLMo-SFT",
        "OLMo-7B-Instruct": "OLMo-Inst",
        "OLMo-7B-SFT-quantized": "OLMo-SFT-quantized",
        "OLMo-7B-Instruct-quantized": "OLMo-Inst-quantized",
        "OLMo-7B-Tulu-lora": "OLMo-Tulu-lora",
        "OLMo-7B-Tulu-lora-r64a128": "OLMo-Tulu-lora-r64a128",
        "OLMo-7B-Flan-lora": "OLMo-Flan-lora",
        "OLMo-7B-Flan-lora-r64a128": "OLMo-Flan-lora-r64a128",
        "T5-Flan-lora-1e-4": "T5-Flan-lora-1e-4",
        "T5-Flan-lora-5e-5": "T5-Flan-lora-5e-5",
        "T5-Tulu-lora-1e-4": "T5-Tulu-lora-1e-4",
        "T5-Tulu-lora-5e-5": "T5-Tulu-lora-5e-5",
    }


def get_data_dir(
    base_path: Path, bias_name: str, all_options_permutations: bool, product: str
):
    if all_options_permutations:
        permute = "all_permutations"
    else:
        permute = "no_permutations"

    if bias_name == "decoy":
        data_dir_name = base_path.joinpath(bias_name, product, permute)
    else:
        data_dir_name = base_path.joinpath(bias_name, permute)

    os.makedirs(data_dir_name, exist_ok=True)

    return data_dir_name


def get_data_dir_and_file_name(
    base_path: Path,
    bias_name: str,
    bias_types: str,
    templates: list[int],
    with_bias: bool,
    product: str,
    all_options_permutations: bool,
    comments_to_file_name="",
):
    data_dir_name = get_data_dir(
        base_path, bias_name, all_options_permutations, product
    )

    if with_bias:
        with_bias = "Treatment"
    else:
        with_bias = "Control"

    data_file_name = Path(
        f"{comments_to_file_name}t_{templates}_{bias_types}_{with_bias}.json"
    )

    return data_dir_name, data_file_name


def get_prediction_dir_name(
    predictions_dir,
    data_path,
    engine,
    predict_according_to_log_probs,
    should_normalize,
    with_format_few_shot,
    with_task_few_shot,
    k_shot,
    model_path=None,
    comments_to_file_name="",
    use_extraction_model=False,
):
    if predict_according_to_log_probs:
        log_probs_dir = "max_prob_pred"
    else:
        log_probs_dir = "gen_pred"
    if should_normalize:
        is_normalize = "_normalized"
    else:
        is_normalize = ""
    if use_extraction_model:
        is_extraction_model = "_extraction_model"
    else:
        is_extraction_model = ""
    # if not with_format_few_shot and not with_task_few_shot:
    #     k_shot = 0

    # keep the inner dir stcuture of the data for the prediction dir
    predictions_dir = predictions_dir.joinpath(*data_path.parts[1:])

    # add step name from model_path if exists
    if model_path is not None:
        # we want to extract the step number
        try:
            step = model_path.split("/")[-2]
        except:
            raise ValueError(
                f"model path {model_path} is not in the right format. should be like: /path/to/model/step_6000/merged"
            )
        # we also want to extract the seed number if exists
        try:
            seed = model_path.split("/")[-3].split("seed")[-1]
        except:
            seed = ""
        if seed:
            model_name = engine + f"_seed{seed}_step_{step}"
        else:
            model_name = engine + f"_step_{step}"
    else:
        model_name = engine

    # add the prediction dir stracture
    predictions_dir_final = predictions_dir.joinpath(
        # bias_name,
        model_name,
        f"few_shot_{k_shot}",
        log_probs_dir + is_normalize + is_extraction_model,
        f"format_{with_format_few_shot}_task_{with_task_few_shot}",
    )

    if not os.path.exists(predictions_dir_final):
        # create the dir if not exists with no error
        os.makedirs(predictions_dir_final, exist_ok=True)

    return predictions_dir_final


def get_across_exp_result_file_prefix(
    bias_name,
    all_options_permutations,
    product,
    pred_dir,
    engine,
    predict_according_to_log_probs,
    normalize_log_prob,
    with_format_few_shot,
    with_task_few_shot,
    k_shot,
    templates,
    use_extraction_model,
):
    data_dir = get_data_dir(
        Path("Data/"),
        bias_name,
        all_options_permutations,
        product,
    )
    full_pred_dir = get_prediction_dir_name(
        pred_dir,
        data_dir,
        engine,
        predict_according_to_log_probs,
        normalize_log_prob,
        with_format_few_shot,
        with_task_few_shot,
        k_shot,
        use_extraction_model=use_extraction_model,
    )
    pred_file_name = f"t_{templates}"
    file_prefix = full_pred_dir.joinpath(pred_file_name)

    return file_prefix


def get_prediction_output_files_names(
    predictions_dir,
    data_path,
    bias_name,
    engine,
    predict_according_to_log_probs,
    should_normalize,
    with_format_few_shot,
    with_task_few_shot,
    k_shot: int,
    bias_types,
    templates,
    with_bias,
    product,
    all_options_permutations,
    model_path,
    use_extraction_model,
):
    if data_path != Path("Data/"):
        data_dir = Path(os.path.dirname(data_path))
        data_file = Path(os.path.basename(data_path))
    else:
        data_dir, data_file = get_data_dir_and_file_name(
            Path("Data/"),
            bias_name,
            bias_types,
            templates,
            with_bias,
            product,
            all_options_permutations,
        )

    predictions_dir = get_prediction_dir_name(
        predictions_dir,
        data_dir,
        engine,
        predict_according_to_log_probs,
        should_normalize,
        with_format_few_shot,
        with_task_few_shot,
        k_shot,
        model_path=model_path,
        use_extraction_model=use_extraction_model,
    )

    output_path = predictions_dir.joinpath(data_file).with_suffix(".json")

    output_with_metadata_path = output_path.with_stem(
        f"{output_path.stem}_with_metadata"
    )

    os.makedirs(predictions_dir, exist_ok=True)

    return output_path, output_with_metadata_path


def get_results_comments_name(conditions, templates, bias_types):
    cond_string = (
        str(conditions)
        .replace("'", "")
        .replace("{", "")
        .replace("}", "")
        .replace(":", "")
        .replace("Certainty_type.", "")
        .replace("Decoy_type.", "")
        .replace(" ", "_")
        .replace(".", "_")
    )
    if len(cond_string) > 30:  # 300
        cond_string = "long_"  # "bias_types_3_"  #   # "R_EXTREAM"

    return f"_{str(templates)}_{bias_types}_" + cond_string


def get_bias_type_templates_defaults(bias_name, bias_types, templates):
    if templates is None:
        if bias_name == "decoy":
            templates = "[1, 2, 3, 4]"
        elif bias_name == "certainty":
            templates = "[1, 2, 3]"
        elif bias_name == "false_belief":
            templates = "[1, 2, 3, 4, 5, 6, 7]"

    if bias_types is None:
        if bias_name == "decoy":
            bias_types = "all"
        elif bias_name == "certainty":
            bias_types = "three_probs,two_probs"
        elif bias_name == "false_belief":
            # bias_types = "dm_1,dm_2"
            bias_types = "dm_full"

    return bias_types, templates


def get_all_models_list_from_path(model_path):
    """
    get all models from a given path
    """
    all_models = []
    if "olmo" in model_path.lower():
        if "flan" in model_path.lower():
            all_models = ["OLMo-7B-Flan-lora"]
        elif "tulu" in model_path.lower():
            all_models = ["OLMo-7B-Tulu-lora"]
        else:
            raise ValueError(f"model_path = {model_path} is not supported")
    elif "t5" in model_path.lower():
        if "flan" in model_path.lower():
            all_models = ["T5-Flan-lora-1e-4"]
        elif "tulu" in model_path.lower():
            all_models = ["T5-Tulu-lora-1e-4"]
        else:
            raise ValueError(f"model_path = {model_path} is not supported")

    return all_models
