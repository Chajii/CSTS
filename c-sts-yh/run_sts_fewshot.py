import argparse
import csv
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, AutoTokenizer,
                          default_data_collator, set_seed)
from vllm import LLM, SamplingParams

from utils.fewshot.generate_in_context_dataset import make_dataset, get_prompt, convert_text, add_context
from utils.fewshot.openai_utils import (OPENAI_MODELS, authenticate, get_gpt_prediction)
from utils.fewshot.progress_logger import ProgressLogger
import pandas as pd 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt": AutoModelForCausalLM,
    "llama": AutoModelForCausalLM,
    "t5": AutoModelForSeq2SeqLM,
}

NO_SKIP_MODULES = {
    "gpt": ["GPTJBlock"],
    "llama": ["LlamaDecoderLayer"],
    "t5": ["T5Block"],
}

DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "tf32": torch.float32,  # tf32 set by cuda.backend
}

DATASET_SENT1_KEY = "sentence1"
DATASET_SENT2_KEY = "sentence2"
DATASET_CONDITION_LEY = "condition"
DATASET_ANSWER_LEY = "label"
DATASET_INPUT_KEY = "text"


def extract_float(s):
    match = re.search(r"(\d+\.\d+|\d+)", s)
    if match:
        return float(match.group(1))
    return s


def process_preds(raw_preds, non_numeric, min_similarity, max_similarity):
    preds = list(map(extract_float, raw_preds))
    non_numeric += sum(1 for p in preds if type(p) is not float)
    preds = [
        p if type(p) is float
        else torch.empty(1).uniform_(min_similarity, max_similarity).item()
        for p in preds
    ]
    return preds, non_numeric


def log_example(ix, text, raw_pred, label):
    example_str = "Example %d:\n\t%s\n\tPRED=%s\n\tLABEL=%s" % (
        ix,
        text.replace("\n", "\n\t"),
        raw_pred,
        label,
    )
    logger.info(example_str)


def log_examples(ix, max_cix, example_texts, raw_preds, labels, batch_size):
    for cix in range(min(len(raw_preds), max_cix)):
        log_example(
            ix * batch_size + cix,
            example_texts[cix],
            raw_preds[cix],
            labels[cix],
        )


def openai_model_eval(model, dataset, min_similarity, max_similarity):
    all_preds, all_labels, examples = [], [], []
    non_numeric = 0

    for ix, example in ProgressLogger.wrap_iter("eval", dataset, len(dataset), return_ix=True):
        raw_pred = get_gpt_prediction(model, example[DATASET_INPUT_KEY])
        pred = extract_float(raw_pred)
        if type(pred) is not float:
            non_numeric += 1
            pred = torch.empty(1).uniform_(min_similarity, max_similarity).item()
        label = float(example["label"])
        all_preds.append(pred)
        all_labels.append(label)
        examples.append({
            "id": ix,
            "example": example[DATASET_INPUT_KEY],
            "raw_pred": raw_pred,
            "pred": pred,
            "label": label,
        })
        if ix < 3:
            log_example(ix, example[DATASET_INPUT_KEY], raw_pred, label)

    return all_preds, all_labels, examples, non_numeric


def non_openai_model_eval(
        model,
        tokenizer,
        tokenizer_type,
        dataset,
        dataloader_num_workers,
        batch_size,
        min_similarity,
        max_similarity,
        **kwargs,
):
    non_numeric = 0
    all_preds, all_labels, all_examples = [], [], []

    # Note, multistep_reasoning_zero_shot_cot is not implemented in this case
    if not kwargs["dataset_on_the_fly"]:
        gen_kwargs = {"max_new_tokens": 20}

        def tokenizer_func(examples):
            return tokenizer(
                examples[DATASET_INPUT_KEY],
                padding="longest",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=tokenizer_type == "t5",
                max_length=4096 if tokenizer_type == "llama" else None,
            )

        dataset = dataset.map(tokenizer_func, batched=True, batch_size=batch_size)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=default_data_collator,
            num_workers=dataloader_num_workers,
            shuffle=False,
        )

        with torch.no_grad():
            for ix, example in ProgressLogger.wrap_iter("eval", dataloader, len(dataloader), return_ix=True):
                inputs = {k: v.to(model.device) for k, v in example.items() if k in ["input_ids", "attention_mask"]}
                output = model.generate(**inputs, **gen_kwargs)
                if tokenizer_type == "gpt" or tokenizer_type == "llama":
                    output = output[:, inputs["input_ids"].shape[-1]:]  # extract only new outputs

                raw_preds = tokenizer.batch_decode(output, skip_special_tokens=True)
                preds, non_numeric = process_preds(raw_preds, non_numeric, min_similarity, max_similarity)
                labels = example["labels"].tolist()
                raw_example_texts = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=False)

                if ix == 0:  # log first batch
                    log_examples(ix, batch_size, raw_example_texts, raw_preds, labels, batch_size)

                all_preds.extend(preds)
                all_labels.extend(labels)
                all_examples.extend([{
                    "id": cix + ix * batch_size,
                    "example": example_text,
                    "raw_pred": raw_pred,
                    "pred": pred,
                    "label": label,
                } for cix, example_text, raw_pred, pred, label in zip(range(len(preds)), raw_example_texts, raw_preds, preds, labels)])

    # True for dataset_on_the_fly
    else:
        if not (tokenizer_type == "gpt" or tokenizer_type == "llama"):
            raise ValueError

        if kwargs["k_shot"] != 0:
            raise ValueError  # Need to impl if k_shot != 0

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=dataloader_num_workers,
            shuffle=False,
        )

        multistep_reasoning_zero_shot_cot = False
        if kwargs['multistep_reasoning_zero_shot_cot']:
            multistep_reasoning_zero_shot_cot = True
            zero_shot_prompt1 = "Let's think step by step. "
            zero_shot_prompt2 = "Therefore, the answer (arabic numerals) is: "
        elif kwargs['multistep_reasoning_zero_shot_cot2']:
            multistep_reasoning_zero_shot_cot = True
            zero_shot_prompt1 = "Let's think in terms of the context of the sentence. "
            zero_shot_prompt2 = "Therefore, the answer (arabic numerals) is: "
        elif kwargs['multistep_reasoning_zero_shot_cot3']:
            multistep_reasoning_zero_shot_cot = True
            zero_shot_prompt1 = "Let's think in terms of the context of the sentence. "
            zero_shot_prompt2 = f"Therefore, the answer (arabic numeral between {'0' if kwargs['is_sts'] else '1'} and 5) is: "

        multistep_reasoning_csts_cot = False
        if kwargs['multistep_reasoning_csts_cot']:
            multistep_reasoning_csts_cot = True
            zero_shot_csts_cot_template1 = "INPUT : Please extract the part that corresponds to this condition \"{__2__}\" within this sentence \"{__1__}\" (only within a given sentence). \n OUTPUT :"
            # "INPUT : Please extract the part that corresponds to this condition \"{__2__}\" within this sentence \"{__1__}\". OUTPUT : "
            # INPUT : Please extract the part that corresponds to this condition "The activity of object" within this sentence "A bus in a parking lot with a double decker bus to the side.". OUTPUT :
            # "INPUT : Imagine this sentence \"{__1__}\" with this condition \"{__2__}\" in one line, if possible. OUTPUT> "
            # Please extract the part that corresponds to this condition from this sentence
            # "Please extract the part that corresponds to this condition within the sentence."
            zero_shot_csts_cot_template2 = (
                "Meaning of Sentence 1 in respect to condition: {__1__}.\n"
                "Meaning of Sentence 2 in respect to condition: {__2__}.\n"
                f"Therefore, the similarity score (arabic numeral between {'0' if kwargs['is_sts'] else '1'} and 5) of two sentences is: "
            )

        with torch.no_grad():
            for ix, batch_example in ProgressLogger.wrap_iter("eval", dataloader, len(dataloader), return_ix=True):
                for i in range(len(batch_example['sentence1'])):
                    def inference(input_str):
                        if kwargs['vllm']:
                            sampling_params = SamplingParams()
                            output = model.generate([input_str, ], sampling_params)
                            return output[0].outputs[0].text

                        else:
                            def tokenizer_func(examples):
                                return tokenizer(
                                    examples,
                                    padding="longest",
                                    truncation=True,
                                    return_tensors="pt",
                                    add_special_tokens=False,
                                    max_length=4096 if tokenizer_type == "llama" else None,
                                )

                            inputs = {k: v.to(model.device) for k, v in tokenizer_func(input_str).items() if k in ["input_ids", "attention_mask"]}

                            gen_kwargs = {'max_new_tokens': 200}
                            outputs = model.generate(**inputs, **gen_kwargs)  # prompt + answer
                            outputs = outputs[:, inputs["input_ids"].shape[-1]:]  # extract only new outputs
                            outputs_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0] # 다시 문자열로 바꿈!
                            return outputs_str

                    example = {
                        "sentence1": batch_example["sentence1"][i],
                        "sentence2": batch_example["sentence2"][i],
                        "label": batch_example["label"][i],
                        "condition": batch_example["condition"][i] if not kwargs["is_sts"] else None,
                    }

                    if multistep_reasoning_zero_shot_cot:
                        # Let's think step by step
                        prompt = get_prompt(kwargs["prompt_name"], kwargs["is_sts"]) 
                        # 'On a scale between 1 and 5, how similar are the following two sentences with respect to the condition provided? '
                        'Respond only with a score between 1 and 5.'
                        example["original_text"] = convert_text(example, is_sts=kwargs["is_sts"]) + zero_shot_prompt1
                        example["input_text"] = add_context(example, [], prompt)
                        output_text = inference(example["input_text"]) # prompt + sentence -> return : answer <- 얼마나 유사한지,
                        # output_text 결과 : 

                        # Therefore the answer is
                        example["input_text"] += output_text + ' ' + zero_shot_prompt2
                        #  Sentence 1:  Three football players tackle another player who is holding a ball . Sentence 2:  A group of football players in red and white uniforms . Condition:  What the players are doing. Output: ","5. The two sentences are very similar with respect to the condition provided, as they both describe actions performed by football players. In both sentences, the players are engaged in an activity (tackling and playing football), and the condition is satisfied. Therefore, I would score the similarity between the two sentences as a 5.",Three football players tackle another player who is holding a ball .,A group of football players in red and white uniforms .,What the players are doing.,5,5.0
                        output_text = inference(example["input_text"])

                    elif multistep_reasoning_csts_cot:
                        def fstr(__template__, __1__=None, __2__=None, __3__=None, __4__=None):
                            return eval(f'f"""{__template__}"""')

                        input1 = fstr(zero_shot_csts_cot_template1, batch_example["sentence1"][i], batch_example["condition"][i])
                        output1_before = inference(input1)  # "INPUT : Please extract the part that corresponds to this condition \"{__2__}\" within this sentence \"{__1__}\". OUTPUT : "
                       
                        output1 = re.split("\n|\.", output1_before) # OUTPUT : 뒤에 모델이 생성하는 것 -> ex : football players

                        input2 = fstr(zero_shot_csts_cot_template1, batch_example["sentence2"][i], batch_example["condition"][i])
                        output2_before = inference(input2)
            
                        output2 = re.split("\n|\.", output2_before) # OUTPUT : 뒤에 모델이 생성하는 것 -> ex : baseball players


                        prompt = get_prompt(kwargs["prompt_name"], kwargs["is_sts"]) # 점수 물어봄
                        example["original_text"] = convert_text(example, input_label="INPUT> ", output_label="OUTPUT> ", is_sts=kwargs["is_sts"])
                        input_text = add_context(example, [], prompt)
                        input_text += fstr(zero_shot_csts_cot_template2, output1, output2)
                        example["input_text"] = input_text

                        output_text = inference(example["input_text"])  # 점수 가 나옴
      

                        raw_data = {'condition' : [batch_example["condition"][i]], 'sentence1': [batch_example["sentence1"][i]], 'sentence1_condition': [output1], 'sentence2' : [batch_example["sentence2"][i]], 'sentence2_condition': [output2], 'sentence1_2_predict': [output_text]}
                        pd.DataFrame(raw_data).to_csv('output_check_output.csv', mode='a')

                    else:
                        prompt = get_prompt(kwargs["prompt_name"], kwargs["is_sts"])
                        example["original_text"] = convert_text(example, is_sts=kwargs["is_sts"])
                        example["input_text"] = add_context(example, [], prompt)
                        output_text = inference(example["input_text"])

                    preds, non_numeric = process_preds([output_text, ], non_numeric, min_similarity, max_similarity)

                    instance = {
                        "id": ix * batch_size + i,
                        "example": example["input_text"],
                        "raw_pred": output_text,
                        "pred": preds[0],
                        "label": example["label"].item(),
                    }

                    logger.info(instance)

                    all_preds.append(instance["pred"])
                    all_labels.append(instance["label"])
                    all_examples.append(instance)

    return all_preds, all_labels, all_examples, non_numeric


def process_results(
        prefix,
        eval_time,
        samples,
        non_numeric,
        all_preds,
        all_labels,
        min_similarity,
        max_similarity,
):
    scaled_preds = np.array(all_preds)
    invalid_preds = sum(
        1 for p in scaled_preds if not min_similarity <= p <= max_similarity
    )
    scaled_labels = np.array(all_labels)
    results = {
        "pearsonr": pearsonr(scaled_preds, scaled_labels)[0],
        "spearmanr": spearmanr(scaled_preds, scaled_labels)[0],
        "runtime": eval_time,
        "samples": samples,
        "samples_per_second": samples / eval_time,
        "non_numeric": non_numeric,
        "non_numeric_percent": non_numeric / samples,
        "mse": ((torch.tensor(all_preds) - torch.tensor(all_labels)) ** 2).mean().item(),
        "out_of_range": invalid_preds,
        "out_of_range_percent": invalid_preds / samples,
    }

    return {f"{prefix}_{k}": v for k, v in results.items()}


def get_weights_location(model_name_or_path):
    if not os.path.exists(model_name_or_path):
        return snapshot_download(
            repo_id=model_name_or_path,
            ignore_patterns=["*h5*", "*msgpack*", "*safetensors*", '*tflite*', '*rust_model.ot*'],  # only download pytorch weights
        )
    elif os.path.isdir(model_name_or_path):
        return model_name_or_path
    else:
        return os.path.dirname(model_name_or_path)


def get_index_location(weights_location):
    index_location = os.path.join(weights_location, "pytorch_model.bin.index.json")
    if not os.path.exists(index_location):
        index_location = os.path.join(weights_location, "pytorch_model.bin")
    return index_location


def load_model_weights(model, index_location, dtype, tokenizer_type):
    logger.info("Loading model weights with load_checkpoint_and_dispatch")
    model = load_checkpoint_and_dispatch(
        model,
        index_location,
        device_map="sequential",
        no_split_module_classes=NO_SKIP_MODULES[tokenizer_type],
        dtype=dtype,
        offload_folder="/home/somebodil/workspace/private-projects/Sentence-Representation/c-sts/temp",
    )
    logger.info(f"Loaded model with load_checkpoint_and_dispatch from {index_location}")
    return model


def load_model_and_tokenizer(model_name_or_path, tokenizer_type, api_key, dtype):
    if model_name_or_path in OPENAI_MODELS:
        if not os.path.exists(api_key):
            raise ValueError("api_key must be a file containing your OpenAI API key")

        authenticate(api_key)
        model = model_name_or_path
        tokenizer = None

    else:
        if not torch.cuda.is_available() and dtype != 'fp32':
            logger.info("Using CPU, overriding dtype to fp32")

        dtype = torch.float32 if not torch.cuda.is_available() else DTYPES[dtype]
        model_cls = MODEL_CLASSES[tokenizer_type]
        weights_location = get_weights_location(model_name_or_path)
        config = AutoConfig.from_pretrained(weights_location)
        index_location = get_index_location(weights_location)
        with init_empty_weights():
            logger.info(f"Instantiating model from config")
            model = model_cls.from_config(config)
        model = load_model_weights(model, index_location, dtype, tokenizer_type)
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer_type == "gpt" or tokenizer_type == "llama":
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def save_results(dataset, results, predictions, examples, output_dir, output_file_prefix):
    logger.info(f"{output_file_prefix} results: %s" % json.dumps(results, indent=4))
    logger.info("Writing eval_results to %s" % output_dir)
    with open(Path(output_dir, f"{output_file_prefix}_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    with open(Path(output_dir, f"{output_file_prefix}_predictions.json"), "w") as f:
        json.dump(predictions, f, indent=4)

    # Log examples
    for ix in range(len(examples)):
        if examples[ix]["label"] != dataset['label'][ix]:
            raise ValueError

        examples[ix]["sentence1"] = dataset['sentence1'][ix]
        examples[ix]["sentence2"] = dataset['sentence2'][ix]
        examples[ix]["condition"] = dataset['condition'][ix]

    with open(Path(output_dir, f"{output_file_prefix}_examples.json"), "w") as f:
        json.dump(examples, f, indent=4)

    csv_column = ["id", "example", "raw_pred", "sentence1", "sentence2", "condition", "label", "pred"]
    with open(Path(output_dir, f"{output_file_prefix}_examples.csv"), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_column)
        writer.writeheader()
        writer.writerows(examples)


def run_eval(
        dataset,
        model,
        tokenizer,
        prefix,
        tokenizer_type,
        min_similarity,
        max_similarity,
        dataloader_num_workers,
        batch_size,
        **kwargs,
):
    start_time = time.time()
    if model in OPENAI_MODELS:
        if kwargs["multistep_reasoning"]:
            raise ValueError("Multistep reasoning is not supported for OpenAI models")

        all_preds, all_labels, examples, non_numeric = openai_model_eval(
            model,
            dataset,
            min_similarity,
            max_similarity,
        )

    else:
        all_preds, all_labels, examples, non_numeric = non_openai_model_eval(
            model,
            tokenizer,
            tokenizer_type,
            dataset,
            dataloader_num_workers,
            batch_size,
            min_similarity,
            max_similarity,
            **kwargs,
        )

    eval_time = time.time() - start_time

    predictions = dict(enumerate(all_preds))
    logger.info(f"Example Preds: {all_preds[:3]}")
    logger.info(f"Example Labels: {all_labels[:3]}")

    results = process_results(
        prefix,
        eval_time,
        len(dataset),
        non_numeric,
        all_preds,
        all_labels,
        min_similarity,
        max_similarity,
    )

    return results, predictions, examples


def get_tokenizer_type(model):
    if "t5" in model.lower() or "t0" in model.lower() or "tk-" in model.lower() or "ul2" in model.lower():
        return "t5"
    elif "gpt" in model.lower():
        return "gpt"
    elif "llama" in model.lower():
        return "llama"
    else:
        raise ValueError(f"Unknown tokenizer type {model}")


def main(
        model_name_or_path,
        k_shot,
        prompt_name,
        seed,
        is_sts,
        train_file,
        validation_file,
        test_file,
        output_dir,
        overwrite_output_dir,
        dataloader_num_workers,
        api_key,
        dtype,
        batch_size,
        max_eval_samples,
        **kwargs,
):
    set_seed(seed)


    skip_validation = overwrite_output_dir is False and Path(output_dir, "eval_results.json").exists()
    skip_test = overwrite_output_dir is False and Path(output_dir, "test_results.json").exists()

    if skip_validation:
        logger.info(f"Skipping validation, found eval_results.json in {output_dir}.\n"
                    f"Set overwrite_output_dir=True to override.")
    if skip_test:
        logger.info(f"Skipping test, found test_results.json in {output_dir}.\n"
                    f"Set overwrite_output_dir=True to override.")

    if skip_validation and skip_test:
        return

    if validation_file is None and test_file is None:
        logger.info("No validation or test file provided. Exiting.")
        return

    if model_name_or_path in OPENAI_MODELS:
        assert api_key is not None, "api_key path must be provided for OpenAI models"

    if dtype == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer_type = get_tokenizer_type(model_name_or_path)
    logger.info(f"Using {tokenizer_type} tokenizer")

    if kwargs["vllm"]:
        tensor_parallel_size = 4 if torch.cuda.device_count() == 4 else 2
        model, tokenizer = LLM(model=model_name_or_path, tensor_parallel_size=tensor_parallel_size, dtype="bfloat16"), None

    else:
        model, tokenizer = load_model_and_tokenizer(model_name_or_path, tokenizer_type, api_key, dtype)

    max_similarity = 5.0
    min_similarity = 1.0 if not is_sts else 0.0

    dataset = make_dataset(
        train_file=train_file,
        validation_file=validation_file,
        test_file=test_file,
        tokenizer_type=tokenizer_type,
        k_shot=k_shot,
        prompt_name=prompt_name,
        seed=seed,
        is_sts=is_sts,
        max_eval_samples=max_eval_samples,
        sentence_1_key=DATASET_SENT1_KEY,
        sentence_2_key=DATASET_SENT2_KEY,
        condition_key=DATASET_CONDITION_LEY,
        answer_key=DATASET_ANSWER_LEY,
        dataset_input_label=DATASET_INPUT_KEY,
        dataset_on_the_fly=kwargs["dataset_on_the_fly"],
    )

    train_dataset = dataset["train"]
    eval_dataset, test_dataset = None, None
    if validation_file is not None:
        eval_dataset = dataset["validation"]
    if test_file is not None:
        test_dataset = dataset["test"]

    logger.info(
        "Loaded %d train examples, %d validation examples, %d test examples"
        % (len(train_dataset), len(eval_dataset) if eval_dataset is not None else 0, len(test_dataset) if test_dataset is not None else 0)
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if validation_file is not None:
        logger.info("Evaluating validation dataset")
        eval_results, eval_predictions, eval_examples = run_eval(
            dataset=eval_dataset,
            model=model,
            tokenizer=tokenizer,
            prefix='eval',
            tokenizer_type=tokenizer_type,
            min_similarity=min_similarity,
            max_similarity=max_similarity,
            dataloader_num_workers=dataloader_num_workers,
            batch_size=batch_size,
            prompt_name=prompt_name,  # To kwargs starting from here
            is_sts=is_sts,
            k_shot=k_shot,
            **kwargs,
        )
        save_results(eval_dataset, eval_results, eval_predictions, eval_examples, output_dir, "eval")

    if test_file is not None:
        logger.info("Predicting on test dataset")
        test_results, test_predictions, test_examples = run_eval(
            dataset=test_dataset,
            model=model,
            tokenizer=tokenizer,
            prefix='test',
            tokenizer_type=tokenizer_type,
            min_similarity=min_similarity,
            max_similarity=max_similarity,
            dataloader_num_workers=dataloader_num_workers,
            batch_size=batch_size,
            prompt_name=prompt_name,  # To kwargs starting from here
            is_sts=is_sts,
            k_shot=k_shot,
            **kwargs,
        )
        save_results(test_dataset, test_results, test_predictions, test_examples, output_dir, "test")

    logger.info("Done!")


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def none_or_str(value):
    if value == 'None':
        return None
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='meta-llama/Llama-2-7b-chat-hf',
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        default=0,
        help="Number of examples to use in prompt."
    )
    parser.add_argument(
        "--prompt_name",
        type=str,
        default='short',
        help="Name of prompt to use. See utils/fewshot/generate_in_context_dataset.py for options."
    )
    parser.add_argument(
        "--is_sts",
        type=string_to_bool,
        default=False,
        help="Whether to use STS dataset."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train_file",
        type=str,
        default='data/csts_train.csv',
        help="Path to train file."
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default='data/csts_validation.csv',
        help="Path to validation file. "
             "If not provided, will not run validation."
    )
    parser.add_argument(
        "--test_file",
        type=none_or_str,
        default=None,  # Warning, default is None
        help="Path to test file. "
             "If not provided, will not run test."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='output/0',
        help="Directory to save results"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        type=string_to_bool,
        default=True,
        nargs="?",
        const=True,
        help="Overwrite the content of the output directory"
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--api_key", type=str, help="Path to OpenAI API key")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "bf16", "fp32", "tf32"],
        default='bf16',
        help="Data used for model. "
             "TF32 and BF16 are recommended but only supported for NVIDIA GPUs with Ampere architecture or later."
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument(
        "--dataset_on_the_fly",
        type=string_to_bool,
        default=True
    )
    parser.add_argument(
        "--multistep_reasoning_zero_shot_cot",
        type=string_to_bool,
        default=False
    )
    parser.add_argument(
        "--multistep_reasoning_zero_shot_cot2",
        type=string_to_bool,
        default=False
    )
    parser.add_argument(
        "--multistep_reasoning_zero_shot_cot3",
        type=string_to_bool,
        default=False
    )
    parser.add_argument(
        "--multistep_reasoning_csts_cot",
        type=string_to_bool,
        default=False
    )
    parser.add_argument(
        "--vllm",
        type=string_to_bool,
        default=False
    )
    args = parser.parse_args()

    main(**vars(args))
