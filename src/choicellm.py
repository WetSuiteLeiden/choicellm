import argparse
import csv
import sys
import itertools
import random
import logging
from typing import Generator, Union, Iterable
import os
from openai import OpenAI
import functools
import io

from prompttemplate import PromptTemplate
from multichoicemodel import MultipleChoiceModel


N_DECIMALS = 6


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                           help='Plaintext file with one word/phrase per line. '
                                'Set the --newlines flag to allow the inclusion of "\\n" to indicate newlines within items. '
                                'Set the --csv flag to allow inputs in .csv format with one column and no header. '
                                'For comparative mode, the csv input can also have n_choices columns.')

    argparser.add_argument('--prompt', required=True, type=argparse.FileType('r'), default=None, help='.json file containing the prompt template, few-shot examples, etc. To generate a suitable template, first use the auxiliary command choicellm-template and adapt the template to your needs.')
    argparser.add_argument('--model', required=False, type=str, default="unsloth/Llama-3.2-1B", help='Currently supports base models via transformers, and chat models through OpenAI (specify --openai in that case).')
    argparser.add_argument('--openai', action='store_true', help='Set this flag for models through the OpenAI API; otherwise --model is assumed to be a local model via huggingface transformers.')

    input_options = argparser.add_argument_group('flags to change input format')
    input_options_excl = input_options.add_mutually_exclusive_group()
    input_options_excl.add_argument('--csv', action='store_true', help='Set this flag if your input file is in .csv format, either a single column to allow newlines etc., or multiple columns (optionally) for comparative mode.')
    input_options_excl.add_argument('--newlines', action='store_true', help='If you want your input file to include newlines, either use full, proper .csv format (and the --csv flag), or use this flag to simply replace occurrences of "\\n" in the input file by proper newlines.')

    # TODO low-priority implement this, maybe replacing --all_positions
    # argparser.add_argument('--n_orders', type=int, help='[not implemented yet] Whether to randomize the order of the categories, and if so, how often; -1 means all orders.', default=None)

    # If --mode comparative:
    comparative_group = argparser.add_argument_group('options for comparative mode (only if comparisons not predetermined in .csv input)')
    comparative_group.add_argument('--compare_to', required=False, type=argparse.FileType('r'), default=None, help='File containing the words to compare against. Default is the main file argument itself. If --csv, then --compare_to must be a csv file as well.')
    comparative_group.add_argument('--compare_deterministic', required=False, action='store_true', help='To make selection of alternatives deterministic; if --compare_to is given (and no overlap with items), this will result in the exact same comparisons per item.')
    comparative_group.add_argument('--n_comparisons', required=False, type=int, default=100, help='Comparisons per stimulus.')
    comparative_group.add_argument('--all_positions', action='store_true', help='Whether to average over all positions; only if comparative.')
    comparative_group.add_argument('--seed', required=False, type=int, default=None, help='Seed to use for shuffling/sampling of comparisons. By default a random seed is picked. If --compare_deterministic, this seed is reused for each item\'s comparisons.')

    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO, format='')
    if args.seed is None:
        args.seed = random.randint(0, 99999)
        random.seed(args.seed)
        logging.info(f'{args.seed=}')

    inputs, is_multicolumn = read_inputs(args.file, args.csv, args.newlines)

    prompt_template = PromptTemplate.from_json(args.prompt)

    if is_multicolumn and prompt_template.mode != 'comparative':
        raise ValueError('Input .csv appears to have multiple columns, which is an option only for \'comparative\' mode.')
    if args.model == argparser.get_default('model'):
        logging.warning(f'WARNING: Using default model {args.model}, which is quite small and may not yield very accurate results; use --model to override it.')
    if not args.openai and 'gpt4' in args.model:
        logging.warning('WARNING: If you meant to use a model available through the OpenAI API, include --openai.')
    if args.openai and not prompt_template.is_chat:
        logging.warning('WARNING: Given --openai, you\'re advised to use a chat-style prompt.')
    if 'instruct' in args.model.lower() and not prompt_template.is_chat:
        logging.warning('WARNING: Given "instruct" model, you\'re advised to use a chat-style prompt.')

    model = MultipleChoiceModel(
        model_name=args.model,
        labels=prompt_template.labels_for_logits,
        prompt_start_for_cache=prompt_template.prompt_start_for_cache,
        openai_client=OpenAI(api_key=os.environ.get('OPENAI_API_KEY')) if args.openai else None
    )

    match prompt_template.mode:
        case 'comparative':
            if is_multicolumn:
                if args.compare_to:
                    logging.warning('WARNING: --compare_to is ignored, because input file is multi-column csv.')
                items = ({'choices': choices, 'prompt': prompt_template.format(*choices)} for choices in inputs)
            else:
                if args.compare_to:
                    compare_to, is_multicolumn2 = read_inputs(args.compare_to, args.csv, args.newlines)
                    if is_multicolumn2:
                        raise ValueError('The --compare_to file appears to contain multiple .csv columns, which is not allowed.')
                    compare_to = list(compare_to)
                else:
                    compare_to = inputs = list(inputs)
                items = iter_items_comparison(inputs, args.n_comparisons, args.all_positions, prompt_template,
                                              compare_to=compare_to, seed_per_item=args.seed if args.compare_deterministic else None)
            add_results_to_dict = add_results_comparative
        case 'categorical':
            items = iter_items_basic(inputs, prompt_template)
            add_results_to_dict = functools.partial(add_results_categorical, category_names=list(prompt_template.categories))
        case 'scalar':
            items = iter_items_basic(inputs, prompt_template)
            add_results_to_dict = functools.partial(add_results_scalar, scale=prompt_template.scale)

    logging.info(f'-------\n{prompt_template}\n-------')

    csv_writer = DictWriterAutoHeader(sys.stdout)
    for item in items:
        probs = model.get_probs(item['prompt'])
        add_results_to_dict(item, probs)
        del item['prompt']
        csv_writer.writerow(item)


def read_inputs(file, is_csv, escaped_newlines) -> tuple[Generator[Union[str, list[str]], None, None], bool]:
    is_multicolumn = False
    if is_csv:
        rows = csv.reader(file)
        first_item = next(rows)
        if len(first_item) > 1:
            is_multicolumn = True
            items = itertools.chain([first_item], rows)
        else:
            items = itertools.chain([first_item[0]], (l[0] for l in rows))
    elif escaped_newlines:
        items = (line.strip().replace('\\n', '\n') for line in file)
    else:
        items = (line.strip() for line in file)
    return items, is_multicolumn


def add_results_scalar(item: dict, probs: list[float], scale: list[int | float]):
    max_index = max(range(len(probs)), key=lambda x: probs[x])
    probs = [round(s, N_DECIMALS) for s in probs]
    item['pred'] = scale[max_index]
    item['prob'] = probs[max_index]
    item['rating'] = round(sum(s * n for s, n in zip(probs, scale)), N_DECIMALS)
    item['probs'] = make_csv_string(probs, delimiter=';')


def add_results_comparative(item: dict, probs: list[float]):
    max_index = max(range(len(probs)), key=lambda x: probs[x])
    probs = [round(s, N_DECIMALS) for s in probs]
    item['pred'] = item['choices'][max_index]
    if 'position' in item:
        item['prob'] = probs[item['position']]
    item['probs'] = make_csv_string(probs, delimiter=';')
    item['choices'] = make_csv_string(item['choices'], delimiter=';')


def add_results_categorical(item: dict, probs: list[float], category_names: list[str]):
    max_index = max(range(len(probs)), key=lambda x: probs[x])
    probs = [round(s, N_DECIMALS) for s in probs]
    item['pred'] = category_names[max_index]
    item['prob'] = probs[max_index]
    item['probs'] = make_csv_string(probs, delimiter=';')


def iter_items_basic(lines: Iterable[str], prompt_template: Union[str, PromptTemplate]) -> Generator[dict, None, None]:
    for n, line in enumerate(lines):
        prompt = prompt_template.format(line)
        yield {'target_id': n, 'target': line, 'prompt': prompt}


def iter_items_comparison(items: Iterable[str], n_comparisons: int, all_positions: bool,
                          prompt_template: Union[str, PromptTemplate], compare_to: list[str],
                          seed_per_item: int) -> Generator[dict, None, None]:

    n_choices = prompt_template.n_choices
    n_alternatives = n_choices - 1

    logging.info(f'Will do {n_comparisons * (n_choices if all_positions else 1)} comparisons per input line.')

    for item_id, item in enumerate(items):

        if seed_per_item is not None:
            random.seed(seed_per_item)

        all_alternatives = random_sample_not_containing(compare_to, n_comparisons * n_alternatives, item_to_exclude=item)

        for comp_id, alternatives in enumerate(batched(all_alternatives, n_alternatives)):
            positions = range(n_choices) if all_positions else [random.randint(0, n_alternatives)]
            for pos in positions:
                choices = alternatives[:pos] + [item] + alternatives[pos:]
                prompt = prompt_template.format(*choices)
                yield {'target_id': item_id, 'comparison_id': comp_id, 'position': pos, 'target': item, 'choices': choices, 'prompt': prompt}


def random_sample_not_containing(items: list, k: int, item_to_exclude) -> list:
    """
    Like random.sample(items, k), but excluding a specific element from the original. Original order not maintained.
    """
    sample = random.sample(items, k=min(k + 1, len(items)))  # one more if possible just in case item is among them
    try:
        sample.remove(item_to_exclude)
    except ValueError:  # if item not found
        if len(sample) > k:
            sample.pop()

    if len(sample) < k:
        raise ValueError(
            f'Not enough comparison items for n_choices × n_comparisons comparisons per item. '
            f'Decrease n_choices or --n_comparisons, or provide a longer list of items to compare '
            f'to (--compare_to).')

    return sample


def batched(iterable, n, *, strict=False) -> Generator[list, None, None]:
    """
    Was added to itertools only in Python 3.12, so included here.
    >>> batched('ABCDEFG', 3)
    ['A', 'B', 'C'], ['D', 'E', 'F'], ['G']
    """
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch


class DictWriterAutoHeader(csv.DictWriter):
    """
    A csv.DictWriter wrapper that lazily sets the fieldnames and writes the header automatically based
    on the first writerow call.
    """

    def __init__(self, f):
        super().__init__(f, fieldnames=[])

    def writerow(self, rowdict):
        if not self.fieldnames:
            self.fieldnames.extend(rowdict.keys())
            self.writeheader()
        super().writerow(rowdict)


def make_csv_string(row: list, delimiter=','):
    output = io.StringIO()
    writer = csv.writer(output, delimiter=delimiter)
    writer.writerow(row)
    csv_string = output.getvalue().strip()
    return csv_string


if __name__ == '__main__':
    main()
