import logging
import random
from collections import defaultdict

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

prompts = {
    'none': None,
    'short': 'On a scale between 1 and 5, how similar are the following two sentences with respect to the condition provided? '
             'Respond only with a score between 1 and 5.',
    'long': 'Definition: Evaluate the similarity between the two sentences, with respect to the condition. '
            'Assign the pair a score between 1 and 5 as follows: '
            '1 : The two sentences are completely dissimilar with respect to the condition. '
            '2 : The two sentences are dissimilar, but are on a similar topic with respect to the condition. '
            '3 : The two sentences are roughly equivalent, but some important information differs or is missing with respect to the condition. '
            '4 : The two sentences are mostly equivalent, but some unimportant details differ with respect to the condition. '
            '5 : The two sentences are completely equivalent with respect to the condition.',
}

sts_prompts = {
    'none': None,
    'short': 'On a scale between 0 and 5, how similar are the following two sentences? '
             'Respond only with a score between 0 and 5.',
    'long': 'Definition: Evaluate the similarity between them and classify them into classes from 0-5 as follows: '
            '0 : The two sentences are completely dissimilar. '
            '1 : The two sentences are not equivalent, but are on the same topic. '
            '2 : The two sentences are not equivalent, but share some details. '
            '3 : The two sentences are roughly equivalent, but some important information differs/missing. '
            '4 : The two sentences are mostly equivalent, but some unimportant details differ. '
            '5 : The two sentences are completely equivalent, as they mean the same thing.',
}


def get_prompt(prompt_name, is_sts=False):
    if prompt_name is None:
        return None
    else:
        if is_sts:
            return sts_prompts[prompt_name]
        else:
            return prompts[prompt_name]


def convert_text(
        example,
        sep_token=' ',
        input_label='Input: ',
        sentence_1_label='Sentence 1: ',
        sentence_2_label='Sentence 2: ',
        condition_label='Condition: ',
        output_label='Output: ',
        sentence_1_key='sentence1',
        sentence_2_key='sentence2',
        condition_key='condition',
        answer_key='label',  # not used
        is_sts=False,
        skip_output=False,
):
    sent_1 = example[sentence_1_key].strip()
    sent_2 = example[sentence_2_key].strip()

    if is_sts:
        ex_list = [
            input_label,
            sep_token.join([sentence_1_label, sent_1, ]),
            sep_token.join([sentence_2_label, sent_2, ]),
            output_label if not skip_output else ' ',
        ]
    else:
        condition = example[condition_key] if example[condition_key] is not None else ''  # bug, some conditions are None

        ex_list = [
            input_label,
            sep_token.join([sentence_1_label, sent_1, ]),
            sep_token.join([sentence_2_label, sent_2, ]),
            sep_token.join([condition_label, condition, ]),
            output_label if not skip_output else ' ',
        ]

    ex_str = sep_token.join(map(str, ex_list))
    return ex_str


def add_context(
        example,
        context,
        prompt,
        sep_token=' ',  # not used
        answer_label='label',
        label_func=lambda x: f'{float(x)}',
):
    if prompt is not None:
        ex_list = [prompt.strip(' :')]
    else:
        ex_list = []

    for ex in context:
        entry = ex['original_text'] + label_func(ex[answer_label])
        ex_list.extend([entry, ])

    ex_list.append(example['original_text'])  # don't add a label to the last example
    return '\n'.join(ex_list)  # is it ok to use '\n' instead of using sep_token?


def create_in_context_examples(dataset, context_dataset, k, prompt, tokenizer_type, pairs=None):
    contexts = list()
    context_ids = list()
    for ix, entry in enumerate(dataset):
        if pairs is not None:
            random_pairs = random.sample(range(len(pairs)), k=(k + 1) // 2)  # in_context_examples is from train, sampled by id
            context_example_ids = [x for pair in random_pairs for x in pairs[pair]][:k]
        else:
            context_example_ids = random.sample(list(set(range(len(context_dataset))) - {ix}), k=k)
        context_ids.append(context_example_ids)
        context_examples = [context_dataset[idx] for idx in context_example_ids]
        contexts.append(add_context(entry, context_examples, prompt))

    return contexts, context_ids


def get_idx_pairs(
        dataset,
        sentence_1_label,
        sentence_2_label,
        condition_label,
        answer_label,
):
    pairs = defaultdict(list)
    for ix, datum in enumerate(dataset):
        pairs[datum[sentence_1_label] + '<-SEP->' + datum[sentence_2_label]].append(ix)

    # check if all pair exist
    pair_idxs = list(pairs.keys())
    drop_count = 0
    for pair_idx in pair_idxs:
        if len(pairs[pair_idx]) != 2:
            drop_count += len(pairs[pair_idx])
            pairs.pop(pair_idx)
    if drop_count != 0:
        logger.warning(
            'Dropping %d indices for missing pairs. '
            'Dataset has %d pairs total' % (drop_count, len(pair_idxs))
        )

    # negative because we want to sort in descending order (highest similarity first)
    pairs = list(map(lambda x: sorted(pairs[x], key=lambda idx: -dataset[idx][answer_label]), pairs.keys()))

    # check if pairs are really pairs
    for idx1, idx2 in pairs:
        if (dataset[idx1][sentence_1_label] != dataset[idx2][sentence_1_label]) or (dataset[idx1][sentence_2_label] != dataset[idx2][sentence_2_label]):
            raise ValueError('Pairing of indices is incorrect, sentences do not match for pair %d and %d' % (idx1, idx2))
        if dataset[idx1][answer_label] < dataset[idx2][answer_label]:
            raise ValueError('Pairing of indices is incorrect, similarity is not in descending order for pair %d and %d' % (idx1, idx2))

    return pairs


def make_dataset(
        train_file,
        validation_file,
        test_file,
        tokenizer_type,
        k_shot,
        prompt_name,
        seed,
        is_sts,
        max_eval_samples,
        sentence_1_key,
        sentence_2_key,
        condition_key,
        answer_key,
        dataset_input_label,
        dataset_on_the_fly,
):
    data_files = {'train': train_file}
    if validation_file is not None:
        data_files['validation'] = validation_file
    if test_file is not None:
        data_files['test'] = test_file

    config_kwargs = {}
    if is_sts:
        config_kwargs["sep"] = "\t"
        config_kwargs["quoting"] = 3

    raw_datasets = load_dataset('csv', data_files=data_files, keep_in_memory=True, **config_kwargs)

    if dataset_on_the_fly:
        if max_eval_samples != 0:
            raw_datasets['validation'] = raw_datasets['validation'].select(range(min(max_eval_samples, len(raw_datasets['validation']))))
        return raw_datasets

    else:
        raw_datasets = raw_datasets.map(lambda x: {
            'original_text': convert_text(
                x,
                sentence_1_key=sentence_1_key,
                sentence_2_key=sentence_2_key,
                condition_key=condition_key,
                answer_label=answer_key,
                is_sts=is_sts,
            )}, batched=False, keep_in_memory=True)

        prompt = get_prompt(prompt_name, is_sts)

        pairs = None
        if not is_sts:
            pairs = get_idx_pairs(
                raw_datasets['train'],
                sentence_1_label=sentence_1_key,
                sentence_2_label=sentence_2_key,
                condition_label=condition_key,
                answer_label=answer_key,
            )

        def add_text_context_ids_column(dataset, text, context_ids):
            dataset = dataset.add_column('context_ids', context_ids)
            dataset = dataset.add_column(dataset_input_label, text)
            return dataset

        raw_datasets['train'] = add_text_context_ids_column(raw_datasets['train'], *create_in_context_examples(raw_datasets['train'], raw_datasets['train'], k_shot, prompt, tokenizer_type, pairs))

        if validation_file is not None:
            raw_datasets['validation'] = add_text_context_ids_column(raw_datasets['validation'], *create_in_context_examples(raw_datasets['validation'], raw_datasets['train'], k_shot, prompt, tokenizer_type, pairs))
            if max_eval_samples is not None:
                raw_datasets['validation'] = raw_datasets['validation'].select(range(min(max_eval_samples, len(raw_datasets['validation']))))

        if test_file is not None:
            raw_datasets['test'] = add_text_context_ids_column(raw_datasets['test'], *create_in_context_examples(raw_datasets['test'], raw_datasets['train'], k_shot, prompt, tokenizer_type, pairs))

        return raw_datasets
