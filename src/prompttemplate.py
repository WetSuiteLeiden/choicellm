import random
from typing import Literal, Union, Optional
import itertools
import copy
import json
import logging
import argparse


# TODO: Drastic refactoring needed
# TODO: Make fields in prompt template optional, i.e., not used if not present;
# TODO: Add optional 'mid point' field to prompt template.

class PromptTemplate:

    """
    Wrapper around plain string prompt templates, and openai-style message lists, both exposing the method `format`.

    Justification:
    - Instantiating prompt templates from prompt info (e.g., .json file) is slightly different in case of scalar, comparative and categorical prompting.
    - I want openai-style 'messages' (list of dicts) to behave the same, on the outside, as a formatable string.
    """

    def __init__(self, mode: Literal['scalar', 'comparative', 'categorical'], is_chat: bool, *args, n_choices: int = None, **kwargs):

        self.mode = mode
        self.is_chat = is_chat
        if n_choices:
            self.n_choices = n_choices  # meh
            kwargs['n_choices'] = n_choices  # meh

        # TODO: Call the following in load_from_json:
        system_prompt, examples, prompt = (
            self.init_for_scalar if mode == 'scalar'
            else self.init_for_comparative if mode == 'comparative'
            else self.init_for_categorical
        )(*args, **kwargs)

        if mode == 'comparative':
            self.labels_for_logits = kwargs['labels'][:kwargs['n_choices']]  # TODO labels_for_logits is the model's business...
        elif mode == 'categorical':
            self.labels_for_logits = kwargs['labels'][:len(kwargs['categories'])]
            self.categories = kwargs['categories']
        else:
            self.scale = kwargs['scale']
            self.labels_for_logits = [str(i) for i in kwargs['scale']]
        if not self.is_chat:  # meh...
            self.labels_for_logits = [' ' + l for l in self.labels_for_logits]

        if is_chat:
            self.prompt_format = [
                {"role": "developer", "content": system_prompt},
                *itertools.chain(*([{"role": "user", "content": example}, {"role": "assistant", "content": response}] for example, response in examples)),
                {"role": "user", "content": prompt}
            ]
            def format(*args, **kwargs):
                messages = copy.deepcopy(self.prompt_format)
                messages[-1]['content'] = messages[-1]['content'].format(*args, **kwargs)
                return messages
            self.prompt_start_for_cache = self.prompt_format[:-1]
            self.format = format
        else:
            prompt_parts = [system_prompt] + [x + ' ' + r for x, r in examples] + [prompt]
            self.prompt_start_for_cache = '\n\n'.join(prompt_parts[:-1])
            self.prompt_format = '\n\n'.join(prompt_parts)
            self.format = self.prompt_format.format

    def format(self, *args, **kwargs) -> Union[str, list[dict]]:
        pass    # to be overwritten by init; is that 'normal'?

    def __str__(self):
        if isinstance(self.prompt_format, str):
            return self.prompt_format
        else:
            return json.dumps(self.prompt_format, indent=2)

    @classmethod
    def from_json(cls, file):

        prompt_info = json.load(file)  # TODO: JSON validation, incl. labels for categorical/comparative must be strings
        prompt_info.pop('_comment', None)

        if "chat" not in prompt_info:
            logging.warning(
                f'WARNING: The prompt .json file does not specify whether to use chat-style prompting; assuming "chat": false')
            prompt_kwargs['chat'] = False

        prompt_info['is_chat'] = prompt_info.pop('chat')

        mode = prompt_info['mode']
        is_chat = prompt_info['mode']

        if mode in ('categorical', 'comparative') and 'labels' not in prompt_info:
            # TODO: In future, allow using categories/items themselves as labels; for now, just ABCD...
            prompt_info['labels'] = list(string.ascii_uppercase)

        return cls(**prompt_info)


    @staticmethod
    def init_for_scalar(system_prompt: str, prompt_format: str, examples: list[dict], scale: list[int | float]) -> tuple[str, list[tuple[str, str]], str]:

        if all(isinstance(n, int) for n in scale):
            rating_as_int = True
        else:
            scale = [float(n) for n in scale]

        labels = [str(number) for number in scale]
        scale_min, scale_max = scale[0], scale[-1]
        system_prompt = system_prompt.format(scale=', '.join(labels), scale_min=scale_min, scale_max=scale_max)
        label_hint = f' (on scale {scale_min}-{scale_max})'

        examples_list = []
        n = 0   # in case no examples
        for n, example in enumerate(examples, start=1):
            rating_scaled = example['target_value'] * (scale_max - scale_min) + scale_min
            rating_scaled = int(rating_scaled) if rating_as_int else float(rating_scaled)
            examples_list.append((
                prompt_format.format(n=n, item=example['item'], label_hint=label_hint),
                str(rating_scaled),
            ))
        prompt = prompt_format.format(n=n+1, item='{}', label_hint=label_hint)

        return system_prompt, examples_list, prompt

    @staticmethod
    def init_for_categorical(system_prompt: str, prompt_format: str, examples: list[dict], categories: dict, labels: Optional[list[str]] = None) -> tuple[str, list[tuple[str, str]], str]:
        if labels:
            categories_full = '\n'.join(f'{l}. {c}: {d}' for l, (c, d) in zip(labels, categories.items()))
        else:
            categories_full = '\n'.join(f'- {c}: {d}' for c, d in categories.items())
        category_names = list(categories.keys())
        system_prompt = system_prompt.format(categories=categories_full)
        examples_list = []
        n = 0

        label_hint = f' (choose from {"/".join(labels)})' if labels else ''

        for n, example in enumerate(examples, start=1):
            examples_list.append((
                prompt_format.format(n=n, item=example['item'], label_hint=label_hint),
                f"{labels[example['target_index']]} ({category_names[example['target_index']]})"
            ))

        prompt = prompt_format.format(n=n+1, item='{}', label_hint=label_hint)

        return system_prompt, examples_list, prompt

    @staticmethod
    def init_for_comparative(system_prompt: str, prompt_format: str, n_choices: int, examples: list[dict], labels: list[str] = None) -> tuple[str, list[tuple[str, str]], str]:

        if labels:
            labels = labels[:n_choices]

        def make_choices_str(choices: list[str], labels: Optional[list[str]] = None) -> str:
            if labels:
                return '\n'.join((f'{label}. {choice}' for label, choice in zip(labels, choices)))
            else:
                return '\n'.join(f'- {choice}' for choice in choices)

        label_hint = f' (choose from {"/".join(labels)})' if labels else ''
        system_prompt = system_prompt
        n = 0
        examples_list = []
        for n, example in enumerate(examples, start=1):

            if example['target_index'] >= n_choices:
                target = example['options'][example['target_index']]
                example['target_index'] = random.randint(0, n_choices-1)
                example['options'][example['target_index']] = target
            example['options'] = example['options'][:n_choices]

            examples_list.append((
                prompt_format.format(n=n, choices=make_choices_str(example['options'], labels), label_hint=label_hint),
                labels[example['target_index']]
            ))
        prompt = prompt_format.format(n=n+1, choices=make_choices_str(['{}'] * n_choices, labels), label_hint=label_hint)
        return system_prompt, examples_list, prompt


TEMPLATE_SCALAR = {
    "mode": "scalar",
    "chat": False,
    "system_prompt": "# Concrete vs. abstract\n\nSome words and phrases are more concrete, some are more abstract. "
                     "We can indicate how concrete a given word or phrase is, as a rating on a scale {scale}, "
                     "with {scale_min} very abstract, and {scale_max} very concrete.",
    "prompt_format": "## Example {n}.\n\nWord/phrase: {item}\n\nConcreteness rating{label_hint}:",
    "scale": [1, 2, 3, 4, 5],
    "examples": [
        {"item": "essentialness", "target_value": 0.0},
        {"item": "frangipane", "target_value": 1.0},
        {"item": "although", "target_value": 0.0},
        {"item": "blackbird", "target_value": 1.0},
        {"item": "bat", "target_value": 1.0},
        {"item": "hope", "target_value": 0.0},
    ],
    "_comment": "In the examples, \"target_value\" is the target rating as a float between [0, 1]. This will be automatically mapped to whichever scale is used. This facilitates trying different scales with the same examples."
}

TEMPLATE_SCALAR_CHAT = TEMPLATE_SCALAR.copy()
TEMPLATE_SCALAR_CHAT.update({
    'chat': True,
    'system_prompt': 'Some words and phrases are more concrete, some are more abstract. You are a helpful assistant, '
                     'who is an expert on rating how *concrete* a given word or phrase is, as a rating on a scale '
                     '{scale}, with {scale_min} very abstract, and {scale_max} very concrete.',
    'prompt_format': '## Question {n}.\n\nWord/phrase: {item}\n\nHow concrete is this {label_hint}?',
})

TEMPLATE_COMPARATIVE = {
    "mode": "comparative",
    "chat": False,
    "system_prompt": "# Concrete vs. abstract\n\nSome words and phrases are more concrete, some are more abstract. "
                     "We can often tell which word or phrase, from a given set, is the _most concrete_ one.",
    "prompt_format": "## Example {n}.\n\n{choices}\n\nThe most concrete is{label_hint}:",  # TODO: Insert "choose from labels" only if choices are not single words?
    "n_choices": 4,
    "labels": ["A", "B", "C", "D"],
    "examples": [
        {"options": ["essentialness", "simulation", "bat", "living"], "target_index": 2},
        {"options": ["blackbird", "high", "cause", "although"], "target_index": 0},
        {"options": ["signature", "frangipane", "hope", "simulation"], "target_index": 1},
    ],
    "_comment": "If \"labels\" is omitted, will use the options themselves as responses (recommended only if the options are single words). Under \"examples\", \"target_index\" is always the integer index of the correct choice in the list, 0-based."
}

TEMPLATE_COMPARATIVE_CHAT = TEMPLATE_COMPARATIVE.copy()
TEMPLATE_COMPARATIVE_CHAT.update({
    'chat': True,
    'system_prompt': 'Some words and phrases are more concrete, some are more abstract. You are a helpful assistant, '
                     'who is an expert on deciding which of several words or phrases is the _most concrete_.',
    'prompt_format': '## Question {n}.\n\n{choices}\n\nWhich of these is the most concrete{label_hint}?',
})

TEMPLATE_CATEGORICAL = {
    "mode": "categorical",
    "chat": False,
    "system_prompt": "# Concrete vs. abstract\n\nSome words and phrases are more concrete, some are more abstract. "
                     "We distinguish the following categories:\n\n{categories}",
    "prompt_format": "## Example {n}.\n\nWord/phrase: {item}\n\nThis word/phrase fits best in category{label_hint}:",
    "labels": ["A", "B", "C"],
    "categories": {
        "concrete": "the word refers to something actual, concrete, empirical",
        "neutral": "the word is neither abstract nor concrete",
        "abstract": "the word refers to something conceptual, intangible, theoretical or vague"
    },
    "examples": [
        {"item": "essentialness", "target_index": 0},
        {"item": "frangipane", "target_index": 2},
        {"item": "although", "target_index": 0},
        {"item": "blackbird", "target_index": 2},
        {"item": "bat", "target_index": 2},
        {"item": "hope", "target_index": 0}
    ],
    "_comment": "If \"labels\" is omitted, will use the options themselves as responses (recommended only if the category names are single words). Under \"examples\", \"target_index\" is always the integer index of the correct choice in the list, 0-based."
}

TEMPLATE_CATEGORICAL_CHAT = TEMPLATE_CATEGORICAL.copy()
TEMPLATE_CATEGORICAL_CHAT.update({
    'chat': True,
    'system_prompt': 'Some words and phrases are more concrete, some are more abstract. You are a helpful assistant, '
                     'who is an expert on categorizing words and phrases into one of three categories:\n\n{categories}',
    'prompt_format': '## Question {n}.\n\n{item}\n\nIn which category does this word/phrase fit best{label_hint}?',
})


def main():

    argparser = argparse.ArgumentParser('Auxiliary command to generate prompt template .json files.')
    argparser.add_argument('--chat', action='store_true', help='Whether to prompt the model like a chat/instruct model; otherwise plain text generation.')
    group = argparser.add_mutually_exclusive_group(required=True)
    group.add_argument('--comparative', action='store_true', help='Generate a prompt template for choicellm --mode comparative.')
    group.add_argument('--scalar', action='store_true', help='Generate a prompt template for choicellm --mode scalar.')
    group.add_argument('--categorical', action='store_true', help='Generate a prompt template for choicellm --mode categorical.')

    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO, format='')

    mode = 'comparative' if args.comparative else 'scalar' if args.scalar else 'categorical'

    template = globals()[f'TEMPLATE_{mode.upper()}{"_CHAT" if args.chat else ""}']
    template_as_json = json.dumps(template, indent=2)

    logging.info(f'Creating a JSON-format prompt template for {mode}, {"chat-style" if args.chat else "plain text generation"} prompting. '
                 f'Save this to a .json file, and modify to suit your needs.')

    print(template_as_json)


if __name__ == '__main__':
    main()
