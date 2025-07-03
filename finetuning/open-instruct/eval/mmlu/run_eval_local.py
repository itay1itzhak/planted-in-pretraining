import argparse
import os
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import time
from eval.mmlu.categories import subcategories, categories
from eval.utils import (
    get_next_word_predictions,
    load_hf_tokenizer,
    load_hf_lm,
    dynamic_import_function,
)  


from datasets import load_dataset

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = len(df.iloc[idx, 2])  # df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, 2][j])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(choices[df.iloc[idx, 3]])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        # added to be fair to all models
        prompt += format_example(train_df, i)

    return prompt


@torch.no_grad()
def eval_hf_model(
    # args,
    subject,
    model,
    tokenizer,
    dev_df,
    test_df,
    use_chat_format,
    chat_formatting_function,
    ntrain,
    batch_size=1,
):
    prompts = []
    chat_formatting_function = (
        dynamic_import_function(chat_formatting_function) if use_chat_format else None
    )
    for i in range(0, test_df.shape[0]):
        k = ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        if use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "The answer is:"
            else:
                prompt += " The answer is:"
        # added to be fair to all models
        else:
            if "T5" not in model.config.architectures[0]:
                if prompt[-1] in ["\n", " "]:
                    prompt += "The answer is:"
                else:
                    prompt += " The answer is:"

        if i == 0:
            print(prompt)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, add_special_tokens=False
        ).input_ids
        # make sure every prompt is less than 2048 tokens
        while len(tokenized_prompt) > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

            if use_chat_format:
                messages = [{"role": "user", "content": prompt}]
                prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
                if prompt[-1] in ["\n", " "]:
                    prompt += "The answer is:"
                else:
                    prompt += " The answer is:"
            # added to be fair to all models
            else:
                if prompt[-1] in ["\n", " "]:
                    prompt += "The answer is:"
                else:
                    prompt += " The answer is:"

            tokenized_prompt = tokenizer(
                prompt, truncation=False, add_special_tokens=False
            ).input_ids
        prompts.append(prompt)

    # get the answer for all examples
    # adding a prefix space here, as that's expected from the prompt
    # TODO: should raise a warning if this returns more than one token
    answer_choice_ids = [
        tokenizer.encode(" " + answer_choice, add_special_tokens=False)[-1]
        for answer_choice in choices
    ]
    pred_indices, all_probs = get_next_word_predictions(
        model,
        tokenizer,
        prompts,
        candidate_token_ids=answer_choice_ids,
        return_token_predictions=False,
        batch_size=batch_size,
    )

    # get the metrics
    cors = []
    groud_truths = test_df.iloc[:, -1].values
    # convert {0,1,2,3} to {'A', 'B', 'C', 'D'}
    groud_truths = [choices[x] for x in groud_truths]
    for i in range(len(pred_indices)):
        prediction = choices[pred_indices[i]]
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs


# def eval_openai_chat_engine(args, subject, engine, dev_df, test_df, batch_size=1):

#     import tiktoken

#     gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
#     answer_choice_ids = [
#         gpt_tokenizer.encode(" " + x)[0] for x in choices
#     ]  # be careful, the tokenizer will tokenize " A" and "A" differently.

#     prompts = []
#     for i in range(0, test_df.shape[0]):
#         k = args.ntrain
#         prompt_end = format_example(test_df, i, include_answer=False)
#         train_prompt = gen_prompt(dev_df, subject, k)
#         prompt = train_prompt + prompt_end
#         prompts.append(prompt)

#     instances = [{"id": prompt, "prompt": prompt} for _, prompt in enumerate(prompts)]
#     results = query_openai_chat_model(
#         engine=args.openai_engine,
#         instances=instances,
#         batch_size=args.eval_batch_size if args.eval_batch_size else 10,
#         output_path=os.path.join(args.save_dir, f"{subject}_openai_results.jsonl"),
#         logit_bias={token_id: 100 for token_id in answer_choice_ids},
#         max_tokens=1,
#     )

#     # get the metrics
#     cors = []
#     groud_truths = test_df.iloc[:, -1].values
#     for i in range(len(test_df)):
#         prediction = results[i]["output"].strip()
#         ground_truth = groud_truths[i]
#         cors.append(prediction == ground_truth)

#     acc = np.mean(cors)
#     cors = np.array(cors)

#     all_probs = np.array(
#         [[0.25, 0.25, 0.25, 0.25] for _ in range(len(test_df))]
#     )  # dummy probs, just don't want to dig into the openai probs

#     print("Average accuracy {:.3f} - {}".format(acc, subject))
#     return cors, acc, all_probs


def load_model_and_tokenizer(
    model_name_or_path,
    tokenizer_name_or_path,
    use_slow_tokenizer=False,
    load_in_8bit=False,
    gptq=False,
    cache_dir=None,
):
    if model_name_or_path:
        print("Loading model and tokenizer...")
        print(f"model_name_or_path: {model_name_or_path}")
        tokenizer = load_hf_tokenizer(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            use_fast_tokenizer=not use_slow_tokenizer,
        )
        model = load_hf_lm(
            model_name_or_path=model_name_or_path,
            load_in_8bit=load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=gptq,
            cache_dir=cache_dir,
        )
        from transformers import GPTNeoXForCausalLM, OPTForCausalLM

        if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
            tokenizer.model_max_length = model.config.max_position_embeddings
            print(
                "Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(
                    model.config.max_position_embeddings
                )
            )

    return model, tokenizer


def predict_mmlu(
    model,
    tokenizer,
    model_name_or_path,
    use_chat_format=False,
    chat_formatting_function="eval.templates.create_prompt_with_tulu_chat_format",
    data_dir="huggingface/datasets",
    wanted_subjects=None,
    ntrain=0,
    n_instances=None,
    eval_batch_size=1,
    save_dir=f"output/mmlu",
    disable_tqdm=False,
):
    # subjects = sorted(
    #     [
    #         f.split("_test.csv")[0]
    #         for f in os.listdir(os.path.join(args.data_dir, "test"))
    #         if "_test.csv" in f
    #     ]
    # )
    # get subject list from huggingface dataset
    dataset = load_dataset(
        "cais/mmlu", "all", cache_dir=data_dir
    )  
    subjects = np.unique(dataset["test"]["subject"])

    if wanted_subjects:
        assert all(
            subj in subjects for subj in wanted_subjects
        ), f"Some of the subjects you specified are not valid: {subjects}"
        subjects = wanted_subjects

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    # convert to pandas dataframe
    dev_df = pd.DataFrame(dataset["dev"])  # [:ntrain]
    test_df = pd.DataFrame(dataset["test"])

    for sub_i, subject in enumerate(
        tqdm(subjects, desc=f"Evaluating subjects: ", disable=disable_tqdm)
    ):
        # dev_df = pd.read_csv(
        #     os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        # )[: args.ntrain]
        # test_df = pd.read_csv(
        #     os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        # )

        # keep only subject
        dev_df_subject = dev_df[dev_df["subject"] == subject][:ntrain]
        test_df_subject = test_df[test_df["subject"] == subject]

        if n_instances and n_instances < test_df_subject.shape[0]:
            test_df_subject = test_df_subject.sample(n_instances, random_state=42)

        if model_name_or_path:
            cors, acc, probs = eval_hf_model(
                subject=subject,
                model=model,
                tokenizer=tokenizer,
                dev_df=dev_df_subject,
                test_df=test_df_subject,
                use_chat_format=use_chat_format,
                chat_formatting_function=chat_formatting_function,
                ntrain=ntrain,
                batch_size=eval_batch_size,
            )
        # else:
        #     cors, acc, probs = eval_openai_chat_engine(
        #         args, subject, args.openai_engine, dev_df, test_df, args.eval_batch_size
        #     )

        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df_subject["correct"] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df_subject["choice{}_probs".format(choice)] = probs[:, j]
        test_df_subject.to_csv(
            os.path.join(save_dir, "{}.csv".format(subject)),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))
    print(f"Results saved in {save_dir}")

    # save results
    with open(
        os.path.join(save_dir, f"metrics_{model_name_or_path.split('/')[-1]}.json"), "w"
    ) as f:
        json.dump(
            {
                "average_acc": weighted_acc,
                "subcat_acc": {
                    subcat: np.mean(np.concatenate(subcat_cors[subcat]))
                    for subcat in subcat_cors
                },
                "cat_acc": {
                    cat: np.mean(np.concatenate(cat_cors[cat])) for cat in cat_cors
                },
            },
            f,
        )

    return weighted_acc


def main(args):

    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_name_or_path,
        args.tokenizer_name_or_path,
        args.use_slow_tokenizer,
        args.load_in_8bit,
        args.gptq,
        args.cache_dir,
    )

    predict_mmlu(
        model,
        tokenizer,
        args.model_name_or_path,
        use_chat_format=args.use_chat_format,
        chat_formatting_function=args.chat_formatting_function,
        data_dir=args.data_dir,
        wanted_subjects=args.subjects,
        ntrain=args.ntrain,
        n_instances=args.n_instances,
        eval_batch_size=args.eval_batch_size,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
    )
    parser.add_argument("--save_dir", type=str, default="results/mmlu/llama-7B/")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer.",
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="if specified, we will use the OpenAI API to generate the predictions.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="which subjects to evaluate. If not specified, all the 57 subjects will be evaluated.",
    )
    parser.add_argument(
        "--n_instances",
        type=int,
        help="if specified, a maximum of n_instances per subject will be used for the evaluation.",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=1, help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.",
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts.",
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The cache directory to save the model and tokenizer.",
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (
        args.openai_engine is None
    ), "Either model_name_or_path or openai_engine should be specified."
    main(args)
