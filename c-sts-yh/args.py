from dataclasses import dataclass, field
from typing import Optional, Union

from transformers import (
    TrainingArguments as HFTrainingArguments,
)
from transformers.trainer_utils import IntervalStrategy, SchedulerType


@dataclass
class TrainingArguments(HFTrainingArguments):
    output_dir: str = field(default='output/0')
    overwrite_output_dir: bool = field(default=True)

    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=True, metadata={"help": "Whether to run predictions on the test set."})

    evaluation_strategy: Union[IntervalStrategy, str] = field(default="epoch")
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=1e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.1, metadata={"help": "Weight decay for AdamW if we apply some."})
    max_grad_norm: float = field(default=0.0, metadata={"help": "Max gradient norm."})
    num_train_epochs: float = field(default=3, metadata={"help": "Total number of training epochs to perform."})
    lr_scheduler_type: Union[SchedulerType, str] = field(default="linear")
    warmup_ratio: float = field(default=0.1, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    log_level: Optional[str] = field(default="info")
    disable_tqdm: Optional[bool] = field(default=None)
    save_strategy: Union[IntervalStrategy, str] = field(default="epoch")
    save_total_limit: Optional[int] = field(default=1)
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    data_seed: Optional[int] = field(default=42, metadata={"help": "Random seed to be used with data samplers."})
    fp16: bool = field(default=True)
    log_time_interval: int = field(default=15)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default='eval_spearmanr')


@dataclass
class DataTrainingArguments:
    max_seq_length: int = field(
        default=512,
        metadata={
            'help': 'The maximum total input sequence length after tokenization. '
                    'Sequences longer than this will be truncated, sequences shorter will be padded.'
        },
    )
    overwrite_cache: bool = field(
        default=True,
        metadata={'help': 'Overwrite the cached preprocessed datasets or not.'}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            'help': 'Whether to pad all samples to `max_seq_length`. '
                    'If False, will pad the samples dynamically when batching to the maximum length in the batch.'
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            'help': 'For debugging purposes or quicker training, truncate the number of training examples to this value if set.'
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            'help': 'For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.'
        }
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            'help': 'For debugging purposes or quicker training, truncate the number of prediction examples to this value if set.'
        },
    )
    train_file: Optional[str] = field(
        default='data/csts_train.csv',
        metadata={'help': 'A csv or a json file containing the training data.'}
    )
    validation_file: Optional[str] = field(
        default='data/csts_validation.csv',
        metadata={'help': 'A csv or a json file containing the validation data.'}
    )
    test_file: Optional[str] = field(
        default='data/csts_test.csv',
        metadata={'help': 'A csv or a json file containing the test data.'}
    )
    val_as_test: Optional[bool] = field(
        default=True,
        metadata={'help': 'Split train data for validation and use validation data as test data.'}  # Will also log random 50 example to debug
    )

    # Dataset specific arguments
    max_similarity: Optional[float] = field(default=None, metadata={'help': 'Maximum similarity score.'})
    min_similarity: Optional[float] = field(default=None, metadata={'help': 'Minimum similarity score.'})
    condition_only: Optional[bool] = field(default=False, metadata={'help': 'Only use condition column.'})
    sentences_only: Optional[bool] = field(default=False, metadata={'help': 'Only use sentences column.'})

    def __post_init__(self):
        validation_extension = self.validation_file.split('.')[-1]
        if self.train_file is not None:
            train_extension = self.train_file.split('.')[-1]
            assert train_extension in ['csv', 'json'], '`train_file` should be a csv or a json file.'
            assert train_extension == validation_extension, '`train_file` and `validation_file` should have the same extension.'
        if self.test_file is not None:
            test_extension = self.test_file.split('.')[-1]
            assert test_extension in ['csv', 'json'], '`test_file` should be a csv or a json file.'
            assert test_extension == validation_extension, '`test_file` and `validation_file` should have the same extension.'


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default='princeton-nlp/sup-simcse-roberta-large',
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={'help': 'Pretrained config name or path if not the same as model_name'}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={'help': 'Pretrained tokenizer name or path if not the same as model_name'}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'Where do you want to store the pretrained models downloaded from huggingface.co'},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={'help': 'Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.'},
    )
    model_revision: str = field(
        default='main',
        metadata={'help': 'The specific model version to use (can be a branch name, tag name or commit id).'},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            'help': 'Will use the token generated when running `transformers-cli login` (necessary to use this script with private models).'
        },
    )
    objective: Optional[str] = field(
        default='mse',
        metadata={'help': 'Objective function for training. Options:\
            1) regression: Regression task (uses MSELoss).\
            2) classification: Classification task (uses CrossEntropyLoss).\
            3) triplet: Regression task (uses QuadrupletLoss).\
            4) triplet_mse: Regression task uses QuadrupletLoss with MSE loss.'}
    )
    # What type of modeling
    encoding_type: Optional[str] = field(
        default='bi_encoder',
        metadata={'help': 'What kind of model to choose. Options:\
            1) cross_encoder: Full encoder model.\
            2) bi_encoder: Bi-encoder model.\
            3) tri_encoder: Tri-encoder model.'}
    )
    # Pooler for bi-encoder
    pooler_type: Optional[str] = field(
        default='cls_before_pooler',
        metadata={'help': 'Pooler type: Options:\
            1) cls: Use [CLS] token.\
            2) cls_before_pooler: Use [CLS] token.\
            3) avg: Mean pooling.\
            4) hypernet: Put whole (str, cond) to hn, avg pooling.\
            5) hypernet2: Put whole (str, cond) to hn, avg pooling wo cls.\
            6) hypernet3: Put only (cond) to hn, avg pooling wo cls.'}
    )
    freeze_encoder: Optional[bool] = field(
        default=False,
        metadata={'help': 'Freeze encoder weights.'}
    )
    transform: Optional[bool] = field(
        default=False,
        metadata={'help': 'Use a linear transformation on the encoder output'}
    )
    triencoder_head: Optional[str] = field(
        default='None',  # Should set if you use tri-encoder
        metadata={'help': 'Tri-encoder head type: Options:\
            1) hadamard: Hadamard product.\
            2) transformer: Transformer.'}
    )

    # YYH params

    use_prompt: bool = field(default=False)
    use_prompt2: bool = field(default=False)
    use_prompt3: bool = field(default=False)
    use_prompt4: bool = field(default=False)

    def __post_init__(self):
        self.do_prompt = True if self.use_prompt or self.use_prompt2 or self.use_prompt3 or self.use_prompt4 else False
        self.do_hypernet = True if "hypernet" in self.pooler_type else False
