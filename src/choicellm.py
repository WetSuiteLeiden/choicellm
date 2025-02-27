import argparse
import csv
import sys
import itertools
import random
import logging
from typing import Generator, Union
import string
import json
import dotenv
import os
from openai import OpenAI
from tqdm import tqdm
import functools

from prompttemplate import PromptTemplate, DEFAULT_PROMPT_INFO
from multichoicemodel import MultipleChoiceModel



def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    argparser.add_argument('--prompt', required=False, type=argparse.FileType('r'), default=None)
    argparser.add_argument('--seed', required=False, type=int, default=None)
    argparser.add_argument('--model', required=False, type=str, default="unsloth/llama-3-70b-bnb-4bit", help='Currently supports base models via transformers, and chat models through OpenAI (specify --openai in that case).')

    argparser.add_argument('--openai', action='store_true', help='Whether to use the openai API; otherwise --model is assumed to be a local model via huggingface transformers.')
    argparser.add_argument('--chat', action='store_true', help='Whether to prompt the model like a chat/instruct model; otherwise prompt plain text.')

    argparser.add_argument('--mode', choices=['scalar', 'comparative', 'categorical'], default='scalar', help='What kind of prompting: rating on a scale, comparative judgments, or multiple-choice given categories.')
    argparser.add_argument('--raw', type=argparse.FileType('w'), default=None, help='Optional file to write raw model output to (comparative judgments in case of --mode comparative; GPT output in case of gpt)')

    argparser.add_argument('--labels', nargs='+', required=False, type=str, default=None, help='If not given, default 1, 2, 3, 4, 5 for scalar, alphabetic otherwise.')

    # if --mode categorical:
    # TODO implement this, albeit low-priority, maybe replacing --all_positions
    argparser.add_argument('--n_orders', type=int, help='[not implemented yet] Whether to randomize the order of the categories, and if so, how often; -1 means all orders.', default=None)

    # If --mode comparative:
    argparser.add_argument('--compare_to', required=False, type=argparse.FileType('r'), default=None, help='Only if --comparative; file containing the words to compare against. Default is the main file argument itself.')
    argparser.add_argument('--n_comparisons', required=False, type=int, default=100, help='Comparisons per stimulus; only if --comparative.')
    argparser.add_argument('--n_choices', required=False, type=int, default=4, help='Choices per comparison; only if --comparative.')
    argparser.add_argument('--all_positions', action='store_true', help='Whether to average over all positions; only if --comparative.')


    # only for backwards comp:
    argparser.add_argument('--comparative', action='store_true', help='[backwards compatibility only] Whether to do comparative judgments instead of absolute/scale.')
    argparser.add_argument('--scale', nargs='+', required=False, type=int, default=None, help='[backwards compatibility only] For --mode scalar')

    args = argparser.parse_args()

    # TODO: Implement multi-label classification? one-versus-rest classification?

    # For backwards compatibility:
    if args.comparative:
        logging.warning('Don\'t use --comparative; kept for backwards compatibility only. Use --mode comparative instead.')
        args.mode = 'comparative'
    if args.scale:
        logging.warning('Don\'t use --scale; kept for backwards compatibility only. Use --labels instead.')
        args.labels = args.scale

    dotenv.load_dotenv()
    logging.basicConfig(level=logging.INFO, format='')
    if args.seed is None:
        args.seed = random.randint(0, 99999)
        random.seed(args.seed)
        logging.info(f'{args.seed=}')

    if args.mode == 'scalar':
        args.labels = [int(l) for l in args.labels] if args.labels else [1, 2, 3, 4, 5]
    else:
        args.labels = args.labels or list(string.ascii_uppercase)

    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY')) if args.openai else None
    if not args.chat and client:
        logging.warning('WARNING: Given --openai, assuming --chat was intended too.')
        args.chat = True
    if not args.chat and 'instruct' in args.model:
        logging.warning('WARNING: Given "instruct" model, assuming --chat was intended too.')
        args.chat = True

    prompt_info = json.load(args.prompt) if args.prompt else DEFAULT_PROMPT_INFO[args.mode]
    input_lines = list(line.strip() for line in args.file)
    labels = (
        args.labels[:args.n_choices] if args.mode == 'comparative'
        else args.labels[:len(prompt_info['categories'])] if args.mode == 'categorical'
        else [str(i) for i in args.labels]
    )

    prompt_template = PromptTemplate(**prompt_info, labels=labels, mode=args.mode, for_chat=args.chat)
    model = MultipleChoiceModel(model_name=args.model, labels=labels, prompt_start_for_cache=prompt_template.prompt_start_for_cache, model_is_chat=args.chat, client=client)

    # TODO: How to chunk/wrap these?
    if args.mode == 'comparative':
        compare_to = list(line.strip() for line in args.compare_to) if args.compare_to is not None else input_lines
        n_comparisons = args.n_choices * args.n_comparisons
        assert len(compare_to) >= n_comparisons + 1, f'Not enough items for {n_comparisons} comparisons! Decrease n_alternatives or n_comparisons.'
        items = iter_items_comparison(input_lines, args.n_choices, args.n_comparisons, args.all_positions, prompt_template, compare_to=compare_to)
        fieldnames = ['target_id', 'comparison_id', 'position', 'target', 'result', 'choices', 'proba']
        process_result = get_score_at_position
    elif args.mode == 'categorical':
        items = iter_items_basic(input_lines, prompt_template)
        fieldnames = ['target_id', 'target', 'result', 'proba']
        process_result = functools.partial(get_max_category, category_names=list(prompt_info['categories'].keys()))
    else:
        items = iter_items_basic(input_lines, prompt_template)
        fieldnames = ['target_id', 'target', 'result', 'proba']
        process_result = functools.partial(get_weighted_sum, scale=args.labels)

    logging.info(f'-------\n{prompt_template}\n-------')

    csv_writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    csv_writer.writeheader()

    for row in items:

        scores = model.get_scores(row['prompt'])

        if args.mode == 'comparative':  # only mode where postprocess is dependent on row['position']...
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


def iter_items_basic(lines: list[str], prompt_template: Union[str, PromptTemplate]) -> Generator[dict, None, None]:

    for n, line in tqdm(enumerate(lines), total=len(lines)):    # TODO Remove progress bars?!
        prompt = prompt_template.format(line)
        yield {'target_id': n, 'target': line, 'prompt': prompt}


def iter_items_comparison(lines: list[str], n_choices: int, n_comparisons: int, all_positions: bool, prompt_template: Union[str, PromptTemplate], compare_to: list[str] = None) -> Generator[dict, None, None]:
    compare_to = compare_to or lines
    total_n_comparisons = len(lines) * n_comparisons * (n_choices if all_positions else 1)
    logging.info(f'Will do {total_n_comparisons} comparisons.')
    n_alternatives = n_choices - 1

    for item_id, item in tqdm(enumerate(lines), total=len(lines)):

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


if __name__ == '__main__':
    main()
