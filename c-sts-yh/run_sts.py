import csv
import json
import logging
import os
import random
import sys
from pathlib import Path

import datasets
import numpy as np
import transformers
from datasets import load_dataset, DatasetDict
from scipy.stats import pearsonr, spearmanr
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    PrinterCallback,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from args import (
    DataTrainingArguments,
    ModelArguments,
    TrainingArguments,
)
from utils.progress_logger import LogCallback
from utils.sts.dataset_preprocessing import get_preprocessing_function
from utils.sts.modeling_utils import get_model, DataCollatorWithPadding
from utils.sts.triplet_trainer import TripletTrainer

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    training_args.log_level = "info"
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if model_args.objective in {"triplet", "triplet_mse"}:
        training_args.dataloader_drop_last = True
        training_args.per_device_eval_batch_size = 2

    logger.info("Training/evaluation parameters %s" % training_args)
    last_checkpoint = None

    if (
            os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.warning(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    data_files = {
        "train": data_args.train_file,
        "validation": data_args.validation_file,
    }
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
    elif training_args.do_predict:
        raise ValueError("test_file argument is missing. required for do_predict.")

    for key, name in data_files.items():
        logger.info(f"load a local file for {key}: {name}")

    if data_args.validation_file.endswith(".csv") or data_args.validation_file.endswith(".tsv"):
        # Loading a dataset from local csv files
        if data_args.val_as_test:
            raw_datasets = DatasetDict()
            raw_datasets["train"] = load_dataset("csv", data_files=data_files, split="train[:80%]")
            raw_datasets["validation"] = load_dataset("csv", data_files=data_files, split="train[80%:]")
            raw_datasets["test"] = load_dataset("csv", data_files=data_files, split="validation")

        else:
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )

    else:
        raise ValueError("validation_file should be a csv or a json file.")

    labels = set()
    for key in set(raw_datasets.keys()) - {"test"}:
        labels.update(raw_datasets[key]["label"])

    if data_args.min_similarity is None:
        data_args.min_similarity = min(labels)
        logger.warning(
            f"Setting min_similarity: {data_args.min_similarity}. Override by setting --min_similarity."
        )

    if data_args.max_similarity is None:
        data_args.max_similarity = max(labels)
        logger.warning(
            f"Setting max_similarity: {data_args.max_similarity}. Override by setting --max_similarity."
        )

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=1,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model_cls = get_model(model_args)
    config.update({
        "use_auth_token": model_args.use_auth_token,
        "model_revision": model_args.model_revision,
        "cache_dir": model_args.cache_dir,
        "model_name_or_path": model_args.model_name_or_path,
        "objective": model_args.objective,
        "pooler_type": model_args.pooler_type,
        "transform": model_args.transform,
        "triencoder_head": model_args.triencoder_head,
    })

    model = model_cls(config=config, model_args=model_args, mask_token_id=tokenizer.mask_token_id)
    if model_args.freeze_encoder:
        for param in model.backbone.parameters():
            param.requires_grad = False

    sentence1_key, sentence2_key, condition_key, similarity_key = (
        "sentence1",
        "sentence2",
        "condition",
        "label",
    )

    # Padding strategy]
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            "The max_seq_length passed (%d) is larger than the maximum length for the "
            "model (%d). Using max_seq_length=%d."
            % (
                data_args.max_seq_length,
                tokenizer.model_max_length,
                tokenizer.model_max_length,
            )
        )

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    preprocess_function = get_preprocessing_function(
        tokenizer,
        sentence1_key,
        sentence2_key,
        condition_key,
        similarity_key,
        padding,
        max_seq_length,
        model_args,
        scale=(data_args.min_similarity, data_args.max_similarity)
        if model_args.objective in {"mse", "triplet", "triplet_mse"}
        else None,
        condition_only=data_args.condition_only,
        sentences_only=data_args.sentences_only,
    )

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
            # remove_columns=raw_datasets["validation"].column_names,
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")

        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            input_ids = train_dataset[index]["input_ids"]
            logger.info(f"tokens: {tokenizer.decode(input_ids)}")
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    def compute_metrics(output: EvalPrediction):
        preds = (
            output.predictions[0]
            if isinstance(output.predictions, tuple)
            else output.predictions
        )
        preds = np.squeeze(preds)
        return {
            "mse": ((preds - output.label_ids) ** 2).mean().item(),
            "pearsonr": pearsonr(preds, output.label_ids)[0],
            "spearmanr": spearmanr(preds, output.label_ids)[0],
        }

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            pad_token_type_id=tokenizer.pad_token_type_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Initialize our Trainer
    trainer_cls = (TripletTrainer if model_args.objective in {"triplet", "triplet_mse"} else Trainer)

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(LogCallback)

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    combined = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        combined.update(metrics)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", combined)
        if training_args.do_train:
            metrics = trainer.evaluate(eval_dataset=train_dataset, metric_key_prefix="train")
            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_dataset)
            )
            metrics["train_samples"] = min(max_eval_samples, len(train_dataset))
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", combined)

    if training_args.do_predict:
        if data_args.val_as_test:
            logger.info("*** Predict using Val-set ***")

            results = trainer.predict(predict_dataset, metric_key_prefix="test")
            trainer.log_metrics("test", results.metrics)
            trainer.save_metrics("test", results.metrics)

            # Log examples
            examples = []
            for ix in range(len(results.predictions)):
                examples.append({
                    "sentence1": predict_dataset['sentence1'][ix],
                    "sentence2": predict_dataset['sentence2'][ix],
                    "condition": predict_dataset['condition'][ix],
                    "label": predict_dataset['labels'][ix],
                    "pred": results.predictions[ix].item(),
                })

            # Write as json
            with open(Path(training_args.output_dir, "test_examples.json"), "w") as f:
                json.dump(examples, f, indent=4)

            # Write as csv
            csv_column = ["sentence1", "sentence2", "condition", "label", "pred"]
            with open(Path(training_args.output_dir, "test_examples.csv"), 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_column)
                writer.writeheader()
                writer.writerows(examples)

        else:
            logger.info("*** Predict ***")

            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("labels")

            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = (
                np.squeeze(predictions)
                if model_args.objective in {"mse", "triplet", "triplet_mse"}
                else np.argmax(predictions, axis=1)
            )
            predictions = dict(enumerate(predictions.tolist()))
            output_predict_file = os.path.join(
                training_args.output_dir, f"test_predictions.json"
            )
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w", encoding="utf-8") as outfile:
                    json.dump(predictions, outfile)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "CSTS"}
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == '__main__':
    main()
