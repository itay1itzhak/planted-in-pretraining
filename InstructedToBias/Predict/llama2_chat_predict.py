from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import transformers
import torch
import numpy as np
import torch.nn.functional as F


def run(model, tokenizer):
    input_text = "Lu lu lu, I've got some apples. Lu lu lu, you've got some too."
    input_tokenized = tokenizer(
        input_text,
        return_tensors="pt",
    )

    # generate with greedy decoding
    outputs = model.generate(
        **input_tokenized,
        max_new_tokens=5,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=False,
    )
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )

    input_length = input_tokenized.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        # | token | token string | logits | probability
        print(
            f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.4f} | {np.exp(score.numpy()):.2%}"
        )

    # decode the generated tokens
    generated_tokens_decoded = tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True
    )

    logits = [
        (tokenizer.decode(id.item()), p.item())
        for id, p in zip(generated_tokens[0], transition_scores[0])
    ]

    print(generated_tokens_decoded)


model_name = "meta-llama/Llama-2-7b-chat-hf"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_name,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)
run(model, tokenizer)
print("stop")
