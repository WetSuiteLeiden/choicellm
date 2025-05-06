import argparse
import csv
import sys
import itertools
import random
import logging
from typing import Generator, Union, Iterable
import string
import json
import os
from openai import OpenAI
from tqdm import tqdm
import functools

from prompttemplate import PromptTemplate
from multichoicemodel import MultipleChoiceModel



def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    argparser.add_argument('--prompt', required=True, type=argparse.FileType('r'), default=None, help='.json file containing the prompt template, few-shot examples, etc. To generate a suitable template, first use the auxiliary command choicellm-template and adapt the template to your needs.')

    argparser.add_argument('--seed', required=False, type=int, default=None, help='Only relevant for --mode comparative; will not affect LLM.')
    argparser.add_argument('--model', required=False, type=str, default="unsloth/Llama-3.2-1B", help='Currently supports base models via transformers, and chat models through OpenAI (specify --openai in that case).')
    argparser.add_argument('--openai', action='store_true', help='Whether to use the OpenAI API; otherwise --model is assumed to be a local model via huggingface transformers.')

    # TODO This option seems to no longer do anything?
    argparser.add_argument('--raw', type=argparse.FileType('w'), default=None, help='[not implemented yet] Optional file to write raw model output to (comparative judgments in case of --mode comparative; GPT output in case of gpt)')

    # if --mode categorical:
    # TODO low-priority implement this, maybe replacing --all_positions
    argparser.add_argument('--n_orders', type=int, help='[not implemented yet] Whether to randomize the order of the categories, and if so, how often; -1 means all orders.', default=None)

    # If --mode comparative:
    argparser.add_argument('--compare_to', required=False, type=argparse.FileType('r'), default=None, help='Only if comparative; file containing the words to compare against. Default is the main file argument itself.')
    argparser.add_argument('--n_comparisons', required=False, type=int, default=100, help='Comparisons per stimulus; only if comparative.')
    argparser.add_argument('--n_choices', required=False, type=int, default=4, help='Choices per comparison; only if comparative.')
    argparser.add_argument('--all_positions', action='store_true', help='Whether to average over all positions; only if comparative.')

    # Backwards comp:
    argparser.add_argument('--mode', choices=['scalar', 'comparative', 'categorical'], default=None, help='[Backwards compatibility only] What kind of prompting: rating on a scale, comparative judgments, or multiple-choice given categories.')
    argparser.add_argument('--labels', nargs='+', required=False, type=str, default=None, help='[Backwards compatibility only] If not given, default 1, 2, 3, 4, 5 for scalar, alphabetic otherwise.')
    argparser.add_argument('--chat', action='store_true', help='[Backwards compatibility only] Whether to prompt the model like a chat/instruct model; otherwise prompt plain text.')

    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO, format='')
    if args.seed is None:
        args.seed = random.randint(0, 99999)
        random.seed(args.seed)
        logging.info(f'{args.seed=}')

    if args.model == argparser.get_default('model'):
        logging.warning(f'WARNING: Using default model {args.model}, which is quite small and may not yield very accurate results; use --model to override it.')
    if not args.openai and 'gpt4' in args.model:
        logging.warning('WARNING: If you meant to use a model available through the OpenAI API, include --openai.')

    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY')) if args.openai else None

    prompt_template = PromptTemplate.from_json(args.prompt, n_choices=args.n_choices, **backwards_compatibility(args))

    do_comparative = prompt_template.mode == 'comparative'
    do_scalar = prompt_template.mode == 'scalar'
    do_categorical = prompt_template.mode == 'categorical'

    if args.openai and not prompt_template.is_chat:
        logging.warning('WARNING: Given --openai, you\'re advised to use a chat-style prompt.')
    if 'instruct' in args.model.lower() and not prompt_template.is_chat:
        logging.warning('WARNING: Given "instruct" model, you\'re advised to use a chat-style prompt.')

    model = MultipleChoiceModel(model_name=args.model, labels=prompt_template.labels_for_logits, prompt_start_for_cache=prompt_template.prompt_start_for_cache, client=client)

    inputs = (l.strip() for l in args.file)

    if do_comparative:
        if args.compare_to:
            compare_to = [l.strip() for l in args.compare_to]
        else:
            compare_to = inputs = list(inputs)
        if len(compare_to) < args.n_comparisons + 1:
            raise ValueError(f'Not enough comparison items for {args.n_choices} × {args.n_comparisons} comparisons per item. '
                             f'Decrease --n_choices or --n_comparisons, or provide a longer list of items to compare'
                             f'to (--compare_to).')
        logging.info(f'Will do {args.n_comparisons * (args.n_choices if args.all_positions else 1)} comparisons per input line.')
        items = iter_items_comparison(inputs, args.n_choices, args.n_comparisons, args.all_positions, prompt_template, compare_to=compare_to)
        fieldnames = ['target_id', 'comparison_id', 'position', 'target', 'result', 'choices', 'proba']
        process_result = get_score_at_position
    elif do_categorical:
        items = iter_items_basic(inputs, prompt_template)
        fieldnames = ['target_id', 'target', 'result', 'proba']
        process_result = functools.partial(get_max_category, category_names=list(prompt_template.categories))
    else:
        items = iter_items_basic(inputs, prompt_template)
        fieldnames = ['target_id', 'target', 'result', 'proba']
        process_result = functools.partial(get_weighted_sum, scale=prompt_template.scale)

    logging.info(f'-------\n{prompt_template}\n-------')

    csv_writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    csv_writer.writeheader()

    for row in items:

        scores = model.get_scores(row['prompt'])

        if do_comparative:  # only mode where postprocess is dependent on row['position']...
            process_result = functools.partial(process_result, position=row['position'])

        result = process_result(scores)

        row['result'] = result
        row['proba'] = ';'.join(str(f) for f in scores)
        del row['prompt']
        csv_writer.writerow(row)


def get_score_at_position(scores: list[float], position: int) -> float:
    return scores[position]


def get_max_category(scores: list[float], category_names: list[str]):
    return category_names[max(range(len(scores)), key=lambda x: scores[x])]


def get_weighted_sum(scores: list[float], scale: list[int | float]) -> float:
    return sum(s * n for s, n in zip(scores, scale))


def iter_items_basic(lines: Iterable[str], prompt_template: Union[str, PromptTemplate]) -> Generator[dict, None, None]:
    for n, line in enumerate(lines):
        prompt = prompt_template.format(line)
        yield {'target_id': n, 'target': line, 'prompt': prompt}


def iter_items_comparison(lines: Iterable[str], n_choices: int, n_comparisons: int, all_positions: bool, prompt_template: Union[str, PromptTemplate], compare_to: list[str]) -> Generator[dict, None, None]:

    n_alternatives = n_choices - 1

    for item_id, item in enumerate(lines):
        all_alternatives = random_sample_not_containing(compare_to, n_comparisons * n_alternatives, item_to_exclude=item)
        for comp_id, alternatives in enumerate(batched(all_alternatives, n_alternatives)):
            positions = range(n_choices) if all_positions else [random.randint(0, n_alternatives)]
            for pos in positions:
                choices = alternatives[:pos] + [item] + alternatives[pos:]
                prompt = prompt_template.format(*choices)
                yield {'target_id': item_id, 'comparison_id': comp_id, 'position': pos, 'target': item, 'choices': ';'.join(choices), 'prompt': prompt}


def random_sample_not_containing(items: list, k: int, item_to_exclude) -> list:
    sample = random.sample(items, k=k + 1)  # one more just in case item is among them
    try:
        sample.remove(item_to_exclude)
    except ValueError:  # if item not found
        sample.pop()
    return sample


def batched(iterable, n, *, strict=False):
    # added to itertools only in 3.12, so included here
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch


def backwards_compatibility(args):
    prompt_kwargs = {}

    if args.mode:  # for backwards compatibility
        logging.warning(f'WARNING: --mode will be deprecated; specify the mode inside the prompt .json file instead, as "mode": "{args.mode}"')
        prompt_kwargs['mode'] = args.mode

    if args.labels:  # for backwards compatibility
        logging.warning(f'WARNING: --labels will be deprecated; specify the scale or labels inside the prompt .json file instead, under "scale" (for scalar), or "categories" (otherwise)')
        if args.mode == 'scalar':
            prompt_kwargs['scale'] = [int(l) for l in args.labels]
        else:
            prompt_kwargs['labels'] = args.labels

    if args.chat:  # for backwards compatibility
        logging.warning(f'WARNING: --chat will be deprecated; specify the prompt type inside the prompt .json file instead, as "chat": true (mind the lowercase, it\'s JSON, not Python)')
        prompt_kwargs['chat'] = True

    return prompt_kwargs


if __name__ == '__main__':
    main()
