from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import transformers
import torch
import numpy as np
import torch.nn.functional as F

model_name = "meta-llama/Llama-2-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"pad_token": "<pad>"})

# Resize token embeddings
model = LlamaForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))


# input_text = "Lu lu lu, I've got some apples. Lu lu lu, you've got some too."
input_text = "Appels are"
input_tokenized = tokenizer(
    input_text,
    return_tensors="pt",
)

################################################## for pretrained model


def get_scores_for_labels(model, tokenizer, input, labels):
    # concat labels to the corrposnded input text
    input_with_answers = [i + label for label in labels for i in input]

    # get labels tokens ids
    labels_tokens = tokenizer(labels, add_special_tokens=False)["input_ids"]

    # get the last token id of each label
    labels_tokens = [label[-1] for label in labels_tokens]

    # Get encodings for each input text to avoid padding
    input_enc = tokenizer.batch_encode_plus(
        input_with_answers,
        return_tensors="pt",
        # add_special_tokens=True,
        # truncation=True,
        padding="longest",
    )

    for k, v in input_enc.items():
        input_enc[k] = v.to(model.device)

    # Get model output logits
    model_output = model(**input_enc)

    # Compute the log probabilities associated with each of the labels
    labels_log_probs = F.log_softmax(model_output.logits, dim=-1)

    # Get the ids of the token before the last token before padding (to see the probablity of the last token given the one before the last token)
    before_padding_ids = input_enc["input_ids"].ne(tokenizer.pad_token_id).sum(-1) - 2

    # Collect labels scores from the -2 token in labels_log_probs (the one that predict the last token)
    # and collect for each line the id in labels_tokens
    labels_scores = labels_log_probs[:, before_padding_ids, labels_tokens]

    # Need just the diagonal of the matrix, as it the prob of the label for each line
    labels_scores = torch.diag(labels_scores)

    return labels_scores.detach()


labels = [
    " tasty",
    # " red",
    " cats",
    " the conclusion is valid",
    " the conclusion is are",
    " red. I have an apple. The conclusion is that it is yellow. The conclusion is valid",
    " red. I have an apple. The conclusion is that it is yellow. The conclusion is invalid",
    # " the conclusion bat invalid",
    # " the argument is valid",
    # " the argument is invalid",
    # "apples",
]

lable_scores = get_scores_for_labels(model, tokenizer, [input_text], labels)
# print score per label text
for label, score in zip(labels, lable_scores):
    print(f"| {label:8s} | {score.numpy():.4f} | {np.exp(score.numpy()):.2%}")

print("")
