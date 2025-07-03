try:
    import hf_olmo
except Exception as e:
    print("hf_olmo did not import")
    print(f"Error: {e}")
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import torch.nn.functional as F

from Predict.hugging_face_perdictor import HFPredictor


class OlmoPredictor(HFPredictor):
    def __init__(
        self,
        bias_name,
        engine,
        max_tokens,
        predict_according_to_log_probs,
        should_normalize,
        save_every_n_examples,
        model_path=None,
    ):
        super().__init__(
            bias_name,
            engine,
            max_tokens,
            predict_according_to_log_probs,
            should_normalize,
            save_every_n_examples,
            model_path,
        )
        self.system_prompt = ""

    def set_device_and_cache_dir(self):
        model_name, device, cache_dir = super().set_device_and_cache_dir()
        cache_dir = "/mnt/nlp/datasets/huggingface/models"

        return model_name, device, cache_dir

    def load_model_and_tokenizer(
        self,
    ):
        model_name_to_load, _, cache_dir = self.set_device_and_cache_dir()

        if "quantized" in model_name_to_load:
            model_name_to_load = model_name_to_load.replace("-quantized", "")
            torch_dtype = torch.float16
            load_in_8bit = True
        else:
            torch_dtype = torch.float32
            load_in_8bit = False
        if self.model_path is not None:
            model_name_to_load = self.model_path
        else:
            model_name_to_load = f"allenai/{model_name_to_load}"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_to_load, cache_dir=cache_dir
        )
        # load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_to_load,
            cache_dir=cache_dir,
            # device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
            trust_remote_code=True,
        )

        # add pad token for pretrained and chat models for batched label scoring
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        # add eos token for chat models
        # tokenizer.add_special_tokens({"eos_token": "</s>"})

        # Resize token embeddings
        model.config.pad_token_id = tokenizer.pad_token_id
        # model.config.eos_token_id = tokenizer.eos_token_id
        model.resize_token_embeddings(len(tokenizer))

        self.change_model_device(model)
        self.model = model
        self.tokenizer = tokenizer

    def get_scores_for_labels(self, input, labels):
        # concat labels to the corrposnded input text
        input_with_answers = [i + label for label in labels for i in input]

        # get labels tokens ids
        labels_tokens = self.tokenizer(labels, add_special_tokens=False)["input_ids"]

        # get the last token id of each label
        labels_tokens = [label[-1] for label in labels_tokens]

        # Get encodings for each input text to avoid padding
        input_enc = self.tokenizer.batch_encode_plus(
            input_with_answers,
            return_tensors="pt",
            # add_special_tokens=True,
            # truncation=True,
            padding="longest",
            return_token_type_ids=False,
        )

        for k, v in input_enc.items():
            input_enc[k] = v.to(self.model.device)

        # Get model output logits
        model_output = self.model(**input_enc)

        # Compute the log probabilities associated with each of the labels
        labels_log_probs = F.log_softmax(model_output.logits, dim=-1)

        # Get the ids of the token before the last token before padding (to see the probablity of the last token given the one before the last token)
        before_padding_ids = (
            input_enc["input_ids"].ne(self.tokenizer.pad_token_id).sum(-1) - 2
        )

        # Collect labels scores from the -2 token in labels_log_probs (the one that predict the last token)
        # and collect for each line the id in labels_tokens
        labels_scores = labels_log_probs[:, before_padding_ids, labels_tokens]

        # Need just the diagonal of the matrix, as it the prob of the label for each line
        labels_scores = torch.diag(labels_scores)

        return labels_scores

    def get_generated_prediction(self, prompt):
        # tokenize input
        input_tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]

        if torch.cuda.is_available():
            input_tokenized = input_tokenized.to(self.model.device)

        generation_config = GenerationConfig(
            max_new_tokens=200,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        outputs = self.model.generate(
            input_tokenized,
            # max_new_tokens=200,
            # return_dict_in_generate=True,
            # output_scores=True,
            # do_sample=False,
            generation_config=generation_config,
        )

        # truncate the logits to the tokenizer vocab size
        outputs_scores_truncated = []
        for score in outputs.scores:
            outputs_scores_truncated.append(score[:, : self.tokenizer.vocab_size + 1])

        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs_scores_truncated, normalize_logits=True
        )

        # get the generated tokens
        input_length = input_tokenized.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]

        # decode the generated tokens
        generated_tokens_decoded = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        # print
        # decode_special_tokens = self.tokenizer.batch_decode(
        #    generated_tokens, skip_special_tokens=False
        # )
        # print(f"len(generated_tokens[0]): {len(generated_tokens[0])}")
        # print(f"len(generated_tokens_decoded[0]): {len(generated_tokens_decoded[0])}")
        # print(f"generated_tokens: {generated_tokens}")
        # print(f"generated_tokens_decoded: {generated_tokens_decoded}")
        # print(f"decode skip_special_tokens=False: {decode_special_tokens}")

        # assert len(generated_tokens_decoded[0]) == len(generated_tokens[0])

        logits = [
            (self.tokenizer.decode(id.item()), p.item())
            for id, p in zip(generated_tokens[0], transition_scores[0])
        ]

        return generated_tokens_decoded[0].strip().strip("."), logits
