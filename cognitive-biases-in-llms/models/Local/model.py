import re
from core.base import LLM
from core.testing import Template
from core.base import DecisionError

# from openai import OpenAI
import yaml
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from functools import partial
import hf_olmo  # For decision extraction we use the OLMo-Instruct model

from core.utils import get_model


class LocalModel(LLM):
    """
    An abstract class representing local models.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )

        self.token = os.getenv("HF_TOKEN", None)
        if self.token is None:
            self.token = "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"
        self.set_device_and_cache_dir()

        with open("./models/Meta/prompts.yml") as f:
            self._PROMPTS = yaml.safe_load(f)

        # Load the decision extraction model OLMo-7B-Instruct
        (
            self.decision_extraction_model,
            self.decision_extraction_tokenizer,
            self.decision_extraction_chat_format_function,
        ) = self.load_decision_extraction_model(self.device)

        self.max_new_tokens = 40
        self.chat_format_function = None

    def load_tokenizer(self):
        """
        Load the tokenizer from the model path.
        """
        pass

    def load_model(self):
        """
        Load the model from the model path.
        """
        pass

    def load_decision_extraction_model(
        self, device: str, extract_model_name: str = "allenai/OLMo-7B-Instruct-hf"
    ):
        if (
            extract_model_name == "allenai/OLMo-7B-Instruct-hf"
            or extract_model_name == "allenai/OLMo-7B-Instruct-hf-v2"
        ):
            # Load the tokenizer
            descision_tokenizer = AutoTokenizer.from_pretrained(
                extract_model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )

            if descision_tokenizer.pad_token is None:
                descision_tokenizer.pad_token = descision_tokenizer.eos_token
                descision_tokenizer.pad_token_id = descision_tokenizer.eos_token_id

            device_map = "auto" if torch.cuda.is_available() else None
            # Load the decision extraction model OLMo-7B-Instruct with float 16
            model = AutoModelForCausalLM.from_pretrained(
                extract_model_name,
                device_map=device_map,
                token=self.token,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16,
            )
            # model.to(device)
            model.eval()
            # Define the chat format function
            chat_format_function = (
                lambda messages: descision_tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            )

        elif extract_model_name == "GptFourOMini":
            # Set API key
            os.environ["OPENAI_API_KEY"] = (
                "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"
            )
            # Get the model
            model = get_model(
                "GptFourOMini",
                randomly_flip_options=False,
                shuffle_answer_options=False,
            )
            # Set the tokenizer to None
            descision_tokenizer = None

        return model, descision_tokenizer, chat_format_function

    def set_device_and_cache_dir(self, cache_dir: str = None):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print("Using device:", self.device)
        self.cache_dir = cache_dir
        print("Using cache dir:", self.cache_dir)

    def prompt_local_olmo(
        self,
        model,
        tokenizer,
        prompt: str,
        use_chat_format=False,
        chat_format_function=None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> str:
        """
        Prompt a local OLMo model with a given prompt and return the generated text.
        If use_chat_format is True, the prompt is formatted as a chat format.

        Args:
            model (torch.nn.Module): The model to prompt.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
            prompt (str): The prompt to prompt the model with.
            use_chat_format (bool): Whether to use a chat format.
            temperature (float): The temperature value of the LLM.
            seed (int): The seed for controlling the LLM's output.

        Returns:
            str: The generated text.
        """

        # Get prompt in the correct format
        if use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            # apply chat formatting function
            prompt = chat_format_function(messages)

        # if the tokenized prompt is a string tokenize it
        if isinstance(prompt, str):
            tokenized_prompt = tokenizer(
                prompt,
                truncation=True,
                add_special_tokens=False,
                return_tensors="pt",
                padding="longest",
                max_length=2048,
            ).input_ids
        else:
            tokenized_prompt = prompt

        # move the tokenized prompt to the model device
        tokenized_prompt = tokenized_prompt.to(model.device)

        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            truse_remote_code=True,
        )

        outputs = model.generate(
            tokenized_prompt,
            generation_config=generation_config,
        )

        # get the generated tokens
        input_length = tokenized_prompt.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]

        # decode the generated tokens
        generated_tokens_decoded = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        # return the decoded tokens
        return generated_tokens_decoded[0].strip().strip(".")

    def prompt_local(
        self,
        model,
        tokenizer,
        prompt: str,
        use_chat_format=False,
        chat_format_function=None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> str:
        """
        Prompt a local model with a given prompt and return the generated text.
        If use_chat_format is True, the prompt is formatted as a chat format.

        Args:
            model (torch.nn.Module): The model to prompt.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
            prompt (str): The prompt to prompt the model with.
            use_chat_format (bool): Whether to use a chat format.
            temperature (float): The temperature value of the LLM.
            seed (int): The seed for controlling the LLM's output.

        Returns:
            str: The generated text.
        """
        pass

    def prompt_extraction_model(
        self, prompt: str, temperature: float = 0.0, seed: int = 42
    ) -> str:
        """
        Prompt the extraction model with a given prompt and return the generated text.
        """
        if self.decision_extraction_tokenizer is not None:
            return self.prompt_local_olmo(
                self.decision_extraction_model,
                self.decision_extraction_tokenizer,
                prompt,
                use_chat_format=True,
                chat_format_function=self.decision_extraction_chat_format_function,
                temperature=temperature,
                seed=seed,
            )
        else:
            return self.decision_extraction_model.prompt(
                prompt, temperature=temperature, seed=seed
            )

    def prompt(self, prompt: str, temperature: float = 0.0, seed: int = 42) -> str:
        """
        Generates a response to the provided prompt from the model.

        Args:
            prompt (str): The prompt to generate a response for.
            temperature (float): The temperature value of the LLM. For strictly decision models, we use a temperature of 0.0.
            seed (int): The seed for controlling the LLM's output. It is not used in Llama models.

        Returns:
            str: The response generated by the LLM.
        """
        return self.prompt_local(
            self._MODEL,
            self._TOKENIZER,
            prompt,
            use_chat_format=self.use_chat_format,
            chat_format_function=self.chat_format_function,
            temperature=temperature,
            seed=seed,
        )

    def _decide(
        self, template: Template, temperature: float = 0.7, seed: int = 42
    ) -> tuple[str, str, int, list[str], list[int]]:
        """
        Prompts the model to choose one answer option from a decision-making task defined in the provided template.

        The decision is obtained through a two-step prompt: First, the model is presented with the decision-making test and can respond freely. Second, the model is instructed to extract the final answer from its previous response.

        Args:
            template (Template): The template defining the decision-making task.
            temperature (float): The temperature value of the LLM.
            seed (int): The seed for controlling the LLM's output.

        Returns:
            tuple[str, str, int, list[str], list[int]]: The raw model response, the model's extraction response, the number of the selected option (None if no selected option could be extracted), the answer option texts, and the order of answer options.
        """

        # 1. Load the decision and extraction prompts
        decision_prompt = self._PROMPTS["decision_prompt"]
        extraction_prompt = self._PROMPTS["extraction_prompt"]

        # 2A. Format the template and insert it into the decision prompt
        decision_prompt = decision_prompt.replace(
            "{{test_case}}",
            template.format(
                randomly_flip_options=self.randomly_flip_options,
                shuffle_options=self.shuffle_answer_options,
                seed=seed,
            ),
        )
        options, option_order = template.get_options(
            randomly_flip_options=self.randomly_flip_options,
            shuffle_options=self.shuffle_answer_options,
            seed=seed,
        )

        # 2B. Obtain a response from the LLM
        try:
            decision_response = self.prompt(
                decision_prompt, temperature=temperature, seed=seed
            )
        except Exception as e:
            raise DecisionError(
                f"Could not obtain a decision from the model to the following prompt:\n\n{decision_prompt}\n\nError is:\n{e}\n"
            )

        # 3A. Insert the decision options and the decision response into the extraction prompt
        extraction_prompt = extraction_prompt.replace(
            "{{options}}",
            "\n".join(
                f"Option {index}: {option}"
                for index, option in enumerate(options, start=1)
            ),
        )

        extraction_prompt = extraction_prompt.replace("{{answer}}", decision_response)
        # 3B. Let the extraction model extract the final chosen option from its previous answer
        try:
            extraction_response = self.prompt_extraction_model(
                extraction_prompt, temperature=temperature, seed=seed
            )
        except Exception as e:
            raise DecisionError(
                f"An error occurred while trying to extract the chosen option with the following prompt:\n\n{extraction_prompt}\n\n{e}"
            )

        # 3C. Extract the option number from the extraction response
        pattern = r"\b(?:[oO]ption) (\d+)\b"
        match = re.search(pattern, extraction_response)
        chosen_option = int(match.group(1)) if match else None

        if chosen_option is None:
            raise DecisionError(
                f"Could not extract the chosen option from the model's response:\n\n{decision_response}\n\nExtraction Response:\n\n{extraction_response}\n\nNo option number detected in response."
            )

        return (
            decision_response,
            extraction_response,
            chosen_option,
            options,
            option_order,
        )

    @staticmethod
    def convert_to_tulu_chat_format(
        text,
        few_shots_texts=None,
        eos_token="</s>",
        format_type="tulu",
    ):
        """
        Apply a chat template to the input text based on the specified format type.
        Originally for Tulu, now supports 'tulu' and 'vicuna' formats.

        Args:
            text (str or list): The input text or list of messages.
            few_shots_texts (list, optional): List of few-shot examples. Defaults to None.
            eos_token (str, optional): End-of-sentence token. Defaults to "</s>".
            format_type (str, optional): The chat format type ('tulu' or 'vicuna'). Defaults to 'tulu'.

        Returns:
            str: The formatted prompt string.

        Raises:
            ValueError: If an invalid format_type is provided or if a message role is unsupported.
            TypeError: If the input text is not a string or list.
        """

        # Define chat format constants globally or ensure they are accessible
        # These were previously defined inside the function, moved outside for clarity if needed
        # If they remain outside the class, ensure they are imported or defined appropriately.
        TULU_SYSTEM = "<|system|>"
        TULU_USER = "<|user|>"
        TULU_ASSISTANT = "<|assistant|>"
        VICUNA_USER = "USER:"
        VICUNA_ASSISTANT = "ASSISTANT:"

        def _get_format_settings(format_type):
            """Inner function to determine format-specific settings."""
            if format_type == "tulu":
                # Original Tulu format settings
                settings = {
                    "system_prefix": TULU_SYSTEM + "\\n",
                    "user_prefix": TULU_USER + "\\n",
                    "assistant_prefix": TULU_ASSISTANT + "\\n",
                    "final_assistant_prompt": TULU_ASSISTANT + "\\n",
                    "role_map": {
                        "system": "system",
                        "user": "user",
                        "assistant": "assistant",
                    },
                    "valid_roles": ["system", "user", "assistant"],
                }
            elif (
                format_type == "vicuna"
            ):  # from: https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#prompt-template
                # Added Vicuna format settings
                settings = {
                    #"system_prefix": "",  # Vicuna doesn't explicitly use system messages in this format
                    "system_prefix": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
                    "user_prefix": VICUNA_USER + " ",
                    "assistant_prefix": VICUNA_ASSISTANT + " ",
                    "final_assistant_prompt": VICUNA_ASSISTANT
                    + " ",  # Vicuna expects a space
                    "role_map": {
                        "system": "user",
                        "user": "user",
                        "assistant": "assistant",
                    },  # Map system to user
                    "valid_roles": [
                        "system",
                        "user",
                        "assistant",
                    ],  # Allow system input, but map it
                }
            else:
                raise ValueError(f"Unsupported chat format type: {format_type}")
            return settings

        # Get the settings based on the format type
        format_settings = _get_format_settings(format_type)
        system_prefix = format_settings["system_prefix"]
        user_prefix = format_settings["user_prefix"]
        assistant_prefix = format_settings["assistant_prefix"]
        final_assistant_prompt = format_settings["final_assistant_prompt"]
        role_map = format_settings["role_map"]
        valid_roles = format_settings["valid_roles"]

        def get_chat_format_one_side(text, role):
            """Helper to format a single message according to the current format_type."""
            if role not in valid_roles:
                raise ValueError(f"Invalid role '{role}' encountered.")
            # Map role if needed (e.g., system to user for Vicuna)
            mapped_role = role_map[role]
            return {"role": mapped_role, "content": text}

        def create_prompt_with_format(messages, eos):
            """Helper to construct the full prompt string based on format_type."""
            formatted_text = ""
            for message in messages:
                role = message["role"]
                content = message["content"].strip()

                # Apply format-specific prefixes
                if format_type == "tulu":
                    if role == "system":
                        formatted_text += system_prefix + content + "\\n"
                    elif role == "user":
                        formatted_text += user_prefix + content + "\\n"
                    elif role == "assistant":
                        formatted_text += assistant_prefix + content + eos + "\\n"
                    # No else needed due to role validation in get_chat_format_one_side
                elif format_type == "vicuna":
                    # Skip system messages if mapped to user (or handle differently if needed)
                    # The role mapping handles the system->user conversion before this point.
                    if role == "user":
                        formatted_text += user_prefix + content + "\\n"
                    elif role == "assistant":
                        formatted_text += assistant_prefix + content + eos + "\\n"

            # Add the final assistant prompt indicator
            formatted_text += final_assistant_prompt
            return formatted_text

        messages = []
        # Handle few-shot examples
        if few_shots_texts is not None:
            for shot in few_shots_texts:
                # Assuming shot has 'question' and 'answer' keys, representing user and assistant turns
                messages.append(get_chat_format_one_side(shot["question"], "user"))
                messages.append(get_chat_format_one_side(shot["answer"], "assistant"))

        # Handle the main input text
        if isinstance(text, str):
            # Assume single user turn if input is a string
            messages.append(get_chat_format_one_side(text, "user"))
        elif isinstance(text, list):
            # Assume text is already in [{'role': ..., 'content': ...}] format
            # Validate and map roles for the list input
            processed_messages = []
            for msg in text:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    processed_messages.append(
                        get_chat_format_one_side(msg["content"], msg["role"])
                    )
                else:
                    raise TypeError(
                        "List items must be dictionaries with 'role' and 'content' keys."
                    )
            messages.extend(processed_messages)
        else:
            raise TypeError(
                "Input 'text' must be a string or a list of message dictionaries."
            )

        # Create the final prompt using the format-specific logic
        prompt = create_prompt_with_format(messages, eos=eos_token)
        return prompt


class OLMo(LocalModel):
    """
    An abstract class representing local OLMo models.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )

        self.max_length = 2048

    def load_tokenizer(self):
        # Print from where the model is loaded
        print("Loading tokenizer from:", self.model_path)
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer

    def load_model(self):
        device_map = "auto" if torch.cuda.is_available() else None
        # If there are more than 1 gpu, specify the device map
        if torch.cuda.device_count() > 1:
            device_map = {"": 1}
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device_map,
            token=self.token,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        # model.to(self.device)
        model.eval()
        return model

    def prompt_local(
        self,
        model,
        tokenizer,
        prompt: str,
        use_chat_format=False,
        chat_format_function=None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> str:
        return self.prompt_local_olmo(
            model,
            tokenizer,
            prompt,
            use_chat_format=use_chat_format,
            chat_format_function=chat_format_function,
            temperature=temperature,
            seed=seed,
        )


class OLMoFlanSeed1(OLMo):
    """
    A class representing a OLMo-Flan-seed-1.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "OLMo-Flan-Seed-1"
        self.model_path = "/home/itay.itzhak/projects/proj2/finetuning/open-instruct/output/flan_2022_allenai/OLMo-7B_lora_r128_alpha256_LR2e-5_seed_1/merged"
        self.use_chat_format = False  # Flan finetuning is done without chat formatting
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()


class OLMoFlanSeed2(OLMo):
    """
    A class representing a OLMo-Flan-seed-2.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "OLMo-Flan-seed-2"
        self.model_path = "/home/itay.itzhak/projects/proj2/finetuning/open-instruct/output/flan_2022_allenai/OLMo-7B_lora_r128_alpha256_LR2e-5_seed_2/merged"
        self.use_chat_format = False  # Flan finetuning is done without chat formatting
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()


class OLMoFlanSeed0(OLMo):
    """
    A class representing a OLMo-Flan-seed-0.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "OLMo-Flan-seed-0"
        self.model_path = "/home/itay.itzhak/projects/proj2/finetuning/open-instruct/output/flan_v2_OLMo-7B_lora_r128_alpha256_LR2e-5/merged"
        self.use_chat_format = False  # Flan finetuning is done without chat formatting
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()


class OLMoSFT(OLMo):
    """
    A class representing OLMo-SFT original model.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "OLMo-SFT"
        self.model_path = "allenai/OLMo-7B-SFT-hf"
        self.use_chat_format = (
            True  # Original SFT model is done with Tulu's chat formatting
        )
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self.chat_format_function = lambda messages: self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        self._MODEL = self.load_model()


class OLMoTuluSeed0(OLMo):
    """
    A class representing OLMo-Tulu-Seed-1.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "OLMo-Tulu-Seed-0"
        self.model_path = "/home/itay.itzhak/projects/proj2/finetuning/open-instruct/output/tulu_v2_OLMo-7B_lora_r128_alpha256_LR2e-5/merged"
        self.use_chat_format = True  # Tulu finetuning is done with chat formatting
        # self.chat_format_function = self.convert_to_tulu_chat_format # Changed because of bug in Llama2ShareGPTSeed0
        self.chat_format_function = (
            LocalModel.convert_to_tulu_chat_format
        )  # Defaults to tulu
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()


class OLMoTuluSeed1(OLMo):
    """
    A class representing OLMo-Tulu-Seed-1.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "OLMo-Tulu-Seed-1"
        self.model_path = "/home/itay.itzhak/projects/proj2/finetuning/open-instruct/output/allenai/tulu-v2-sft-mixture_allenai/OLMo-7B_lora_r128_alpha256_LR2e-5_seed_1/merged"
        self.use_chat_format = True  # Tulu finetuning is done with chat formatting
        # self.chat_format_function = self.convert_to_tulu_chat_format # Changed because of bug in Llama2ShareGPTSeed0
        self.chat_format_function = (
            LocalModel.convert_to_tulu_chat_format
        )  # Defaults to tulu
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()


class OLMoTuluSeed2(OLMo):
    """
    A class representing OLMo-Tulu-Seed-2.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "OLMo-Tulu-Seed-2"
        self.model_path = "/home/itay.itzhak/projects/proj2/finetuning/open-instruct/output/allenai/tulu-v2-sft-mixture_allenai/OLMo-7B_lora_r128_alpha256_LR2e-5_seed_2/merged"
        self.use_chat_format = True  # Tulu finetuning is done with chat formatting
        # self.chat_format_function = self.convert_to_tulu_chat_format # Changed because of bug in Llama2ShareGPTSeed0
        self.chat_format_function = (
            LocalModel.convert_to_tulu_chat_format
        )  # Defaults to tulu
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()


######################################
#              T5 Models            #
######################################

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
)


class T5(LocalModel):
    """
    An abstract class representing local T5 models.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.max_length = 1024
        (
            self.decision_extraction_model,
            self.decision_extraction_tokenizer,
            self.decision_extraction_chat_format_function,
        ) = self.load_decision_extraction_model(device=self.device)

    def load_tokenizer(self):
        # Print from where the model is loaded
        print("Loading tokenizer from:", self.model_path)
        # Load the tokenizer
        tokenizer = T5Tokenizer.from_pretrained(
            self.model_path, cache_dir=self.cache_dir
        )

        return tokenizer

    def load_model(self):
        device_map = "auto" if torch.cuda.is_available() else None
        # If there are more than 1 gpu, specify the device map
        if torch.cuda.device_count() > 1:
            device_map = {"": 1}

        # Load the model
        print("Loading model from:", self.model_path)
        model = T5ForConditionalGeneration.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,  # torch_dtype,
            # low_cpu_mem_usage=True,
            device_map=device_map,
        )
        # model.to(self.device)
        model.eval()

        return model

    def prompt_local(
        self,
        model,
        tokenizer,
        prompt: str,
        use_chat_format=False,
        chat_format_function=None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> str:
        """
        Prompt the T5 model with the given prompt and return the generated text.
        """
        # if use_chat_format is True, convert the prompt to the T5-Tulu chat format
        if use_chat_format:
            prompt = chat_format_function(prompt)

        tokenized_prompt = tokenizer(
            prompt,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
            max_length=self.max_length,
        ).input_ids

        tokenized_prompt = tokenized_prompt.to(model.device)

        outputs = model.generate(
            tokenized_prompt,
            max_new_tokens=self.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_text = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )

        return generated_text[0].strip().strip(".")


class T5TuluSeed0(T5):
    """
    A class representing a T5-Tulu-Seed-1.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "T5-Tulu-Seed-0"
        self.model_path = "/home/itay.itzhak/projects/proj2/finetuning/open-instruct/output/allenai/tulu-v2-sft-mixture_t5-v1_1-xxl_lora_r128_alpha256_LR1e-4/merged"
        self.use_chat_format = True
        # self.chat_format_function = self.convert_to_tulu_chat_format # Changed because of bug in Llama2ShareGPTSeed0
        self.chat_format_function = (
            LocalModel.convert_to_tulu_chat_format
        )  # Defaults to tulu
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()


class T5TuluSeed1(T5):
    """
    A class representing a T5-Tulu-Seed-1.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "T5-Tulu-Seed-1"
        self.model_path = "/home/itay.itzhak/projects/proj2/finetuning/open-instruct/output/allenai/tulu-v2-sft-mixture_t5-v1_1-xxl_lora_r128_alpha256_LR1e-4_seed_1/merged"
        self.use_chat_format = True
        # self.chat_format_function = self.convert_to_tulu_chat_format # Changed because of bug in Llama2ShareGPTSeed0
        self.chat_format_function = (
            LocalModel.convert_to_tulu_chat_format
        )  # Defaults to tulu
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()


class T5TuluSeed2(T5):
    """
    A class representing a T5-Tulu-Seed-2.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "T5-Tulu-Seed-2"
        self.model_path = "/home/itay.itzhak/projects/proj2/finetuning/open-instruct/output/allenai/tulu-v2-sft-mixture_t5-v1_1-xxl_lora_r128_alpha256_LR1e-4_seed_2/merged"
        self.use_chat_format = True
        # self.chat_format_function = self.convert_to_tulu_chat_format # Changed because of bug in Llama2ShareGPTSeed0
        self.chat_format_function = (
            LocalModel.convert_to_tulu_chat_format
        )  # Defaults to tulu
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()


class FlanT5(T5):
    """
    A class representing local Flan-T5 models.

    Attributes:
        NAME (str): The name of the model.
        model_path (str): The path to the model.
        use_chat_format (bool): Whether to use chat format.
        tokenizer (Tokenizer): The tokenizer of the model.
        model (Model): The model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "Flan-T5"
        self.model_path = "google/flan-t5-xxl"
        self.use_chat_format = False
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()


class T5FlanSeed1(T5):
    """
    A class representing a T5-Flan-Seed-0.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "T5-Flan-Seed-1"
        self.model_path = "/home/itay.itzhak/projects/proj2/finetuning/open-instruct/output/flan_2022_google/t5-v1_1-xxl_lora_r128_alpha256_LR1e-4_seed_1/merged"
        self.use_chat_format = False
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()


class T5FlanSeed2(T5):
    """
    A class representing a T5-Flan-Seed-2.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "T5-Flan-Seed-2"
        self.model_path = "/home/itay.itzhak/projects/proj2/finetuning/open-instruct/output/flan_2022_google/t5-v1_1-xxl_lora_r128_alpha256_LR1e-4_seed_2/merged"
        self.use_chat_format = False
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()


class T5FlanSeed0(T5):
    """
    A class representing a T5-Flan-Seed-0.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "T5-Flan-Seed-0"
        self.model_path = "/home/itay.itzhak/projects/proj2/finetuning/open-instruct/output/flan_v2_t5-v1_1-xxl_lora_r128_alpha256_LR1e-4/merged"
        self.use_chat_format = False
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()


######################################
#              Mistral Models        #
######################################


class Mistral(LocalModel):
    """
    An abstract class representing local Mistral models.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )

        self.max_length = 2048
        self.max_new_tokens = 100

    def load_tokenizer(self):
        # Print from where the model is loaded
        print("Loading tokenizer from:", self.model_path)
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # print special tokens
        # print(f"Special tokens in tokenizer: {tokenizer.special_tokens_map}")

        return tokenizer

    def load_model(self):
        device_map = "auto" if torch.cuda.is_available() else None
        # If there are more than 1 gpu, specify the device map
        if torch.cuda.device_count() > 1:
            device_map = {"": 1}
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device_map,
            token=self.token,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        # model.to(self.device)
        model.eval()
        return model

    def convert_to_mistral_tulu_chat_format(
        self, text, few_shots_texts=None, eos_token=None
    ):
        """
        Convert the text to the Tulu chat format with the eos token of Mistral.
        """
        # Uses the default tulu format
        return LocalModel.convert_to_tulu_chat_format(
            text,
            few_shots_texts,
            eos_token=self.tokenizer.eos_token,
        )

    def truncate_at_first_stop(
        self,
        decoded_text,
        stop_tokens=[
            "<|end_of_text|>",
            "<|endoftext|>",
            "<|eot_id|>",
            "<|user|>",
            "<|assistant|>",
        ],
    ):
        """
        Truncate the text at the first stop token. Needed because the model was probably not trained to stop at the stop tokens.

        Args:
            decoded_text (str): The text to truncate.
            stop_tokens (list): The list of stop tokens.

        Returns:
            str: The truncated text.
        """
        for token in stop_tokens:
            if token in decoded_text:
                decoded_text = decoded_text.split(token)[0].strip()
        return decoded_text

    def prompt_local(
        self,
        model,
        tokenizer,
        prompt: str,
        use_chat_format=False,
        chat_format_function=None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> str:
        output = self.prompt_local_olmo(
            model,
            tokenizer,
            prompt,
            use_chat_format=use_chat_format,
            chat_format_function=chat_format_function,
            temperature=temperature,
            seed=seed,
        )
        return self.truncate_at_first_stop(output)


class MistralTuluSeed0(Mistral):
    """
    A class representing Mistral-Tulu-Seed-0.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "Mistral-Tulu-Seed-0"
        self.model_path = "nyu-dice-lab/Mistral-7B-Base-SFT-Tulu2"
        self.use_chat_format = True  # Tulu finetuning is done with chat formatting
        self.chat_format_function = self.convert_to_mistral_tulu_chat_format
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()


class MistralTuluSeed1(Mistral):
    """
    A class representing Mistral-Tulu-Seed-1 (Finetuned with different learning rate).

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "Mistral-Tulu-Seed-1"
        self.model_path = "nyu-dice-lab/Mistral-7B-Base-SFT-Tulu2-2.0"
        self.use_chat_format = True  # Tulu finetuning is done with chat formatting
        # need to set convert_to_tulu_chat_format as partial with eos token
        # self.chat_format_function = self.convert_to_tulu_chat_format
        self.chat_format_function = self.convert_to_mistral_tulu_chat_format
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()


class MistralShareGPT(Mistral):
    """
    A class representing Mistral-ShareGPT.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "Mistral-ShareGPT"
        self.model_path = "nyu-dice-lab/Mistral-7B-Base-SFT-ShareGPT-Vicuna"
        self.use_chat_format = True  # ShareGPT finetuning is done with chat formatting
        self.chat_format_function = self.convert_to_mistral_tulu_chat_format
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()


######################################
#              Llama2 Models          #
######################################

# TODO: add Llama2-Tulu and Llama2-ShareGPT


class Llama2(LocalModel):
    """
    An abstract class representing local Llama2 models.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )

        self.max_length = 2048
        self.max_new_tokens = 60

    def load_tokenizer(self):
        # Print from where the model is loaded
        print("Loading tokenizer from:", self.model_path)
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer

    def load_model(self):
        device_map = "auto" if torch.cuda.is_available() else None
        # If there are more than 1 gpu, specify the device map
        if torch.cuda.device_count() > 1:
            device_map = {"": 1}
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device_map,
            token=self.token,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        model.eval()
        return model

    def prompt_local(
        self,
        model,
        tokenizer,
        prompt: str,
        use_chat_format=False,
        chat_format_function=None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> str:
        output = self.prompt_local_olmo(
            model,
            tokenizer,
            prompt,
            use_chat_format=use_chat_format,
            chat_format_function=chat_format_function,
            temperature=temperature,
            seed=seed,
        )
        return output


class Llama2Tulu(Llama2):
    """
    A class representing Llama2-Tulu.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "Llama2-Tulu"
        self.model_path = "allenai/tulu-2-7b"
        self.use_chat_format = True  # Tulu finetuning is done with chat formatting
        # self.chat_format_function = self.convert_to_tulu_chat_format # Changed because of bug in Llama2ShareGPTSeed0
        self.chat_format_function = (
            LocalModel.convert_to_tulu_chat_format
        )  # Defaults to tulu
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()


class Llama2ShareGPTSeed0(Llama2):
    """
    A class representing Llama2-ShareGPT-Seed-0.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "Llama2-ShareGPT-Seed-0"
        self.model_path = "allenai/open-instruct-llama2-sharegpt-7b"
        self.use_chat_format = True  # ShareGPT finetuning is done with chat formatting
        # self.chat_format_function = self.convert_to_tulu_chat_format # Changed because of bug in Llama2ShareGPTSeed0
        self.chat_format_function = (
            LocalModel.convert_to_tulu_chat_format
        )  # Defaults to tulu
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()


class Llama2ShareGPTSeed1(Llama2):
    """
    A class representing Llama2-ShareGPT-Seed-1.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "Llama2-ShareGPT-Seed-1"
        self.model_path = "lmsys/vicuna-7b-v1.5"
        self.use_chat_format = True  # ShareGPT finetuning is done with chat formatting
        # self.chat_format_function = self.convert_to_tulu_chat_format # Changed because of bug in Llama2ShareGPTSeed0
        self.chat_format_function = partial(
            LocalModel.convert_to_tulu_chat_format, format_type="vicuna"
        )
        self.max_new_tokens = 80
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()

class Llama2ShareGPTSeed2(Llama2):
    """
    A class representing Llama2-ShareGPT-Seed-1 but with a different system prompt.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "Llama2-ShareGPT-Seed-2"
        self.model_path = "lmsys/vicuna-7b-v1.5"
        self.use_chat_format = True  # ShareGPT finetuning is done with chat formatting
        # self.chat_format_function = self.convert_to_tulu_chat_format # Changed because of bug in Llama2ShareGPTSeed0
        self.chat_format_function = partial(
            LocalModel.convert_to_tulu_chat_format, format_type="vicuna"
        )
        self.max_new_tokens = 80
        self._TOKENIZER = self.load_tokenizer()
        self.tokenizer = self._TOKENIZER  # for outer use
        self._MODEL = self.load_model()