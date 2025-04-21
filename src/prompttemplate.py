import random
from typing import Literal, Union
import itertools
import copy
import json


DEFAULT_PROMPT_INFO = {
    'scalar': {
        "system_prompt": "# Concrete vs. abstract\n\nSome words and phrases are more concrete, some are more abstract. "
                         "We can indicate how concrete a given word or phrase is, as a rating on a scale {scale}, "
                         "with {scale_min} very abstract, and {scale_max} very concrete:",
        "prompt_format": "## Example {n}.\n\nWord/phrase: {item}\n\nConcreteness rating:",
        # In the examples, "response" is always the integer index of the 'correct' choice on the scale, 0-based. NB: The scale is chosen
        # with the argument --scale when running the script. These examples assume with the default scale 1, 2, 3, 4, 5.
        "examples": [
            {"item": "essentialness", "response": 0},
            {"item": "frangipane", "response": 4},
            {"item": "although", "response": 0},
            {"item": "blackbird", "response": 4},
            {"item": "bat", "response": 4},
            {"item": "hope", "response": 0}
        ]
    },
    'comparative': {
        "system_prompt": "# Concrete vs. abstract\n\nSome words and phrases are more concrete, some are more abstract. "
                         "We can often tell which word or phrase, from a given set, is the _most concrete_ one.",
        "prompt_format": "## Example {n}.\n\n{choices}\n\nThe most concrete is (choose from {labels}):",
        # In the examples, "response" is always the integer index of the 'correct' choice, 0-based.
        "examples": [
            {"choices": ["essentialness", "simulation", "bat", "living"], "response": 2},
            {"choices": ["blackbird", "high", "cause", "although"], "response": 0},
            {"choices": ["signature", "frangipane", "hope", "simulation"], "response": 1}
        ]
    },
    'categorical': {
        "system_prompt": "# Concrete vs. abstract\n\nSome words and phrases are more concrete, some are more abstract. "
                         "We distinguish the following categories: \n\n{categories}",
        "prompt_format": "## Example {n}.\n\nWord/phrase: {item}\n\nThis word/phrase fits best in category:",
        "categories": {
            "concrete": "the word refers to something actual, concrete, empirical",
            "neutral": "the word is neither abstract nor concrete",
            "abstract": "the word refers to something conceptual, intangible, theoretical or vague"
        },
        # In the examples, "response" is always the correct category's integer index in the above dictionary.
        "examples": [
            {"item": "essentialness", "response": 0},
            {"item": "frangipane", "response": 2},
            {"item": "although", "response": 0},
            {"item": "blackbird", "response": 2},
            {"item": "bat", "response": 2},
            {"item": "hope", "response": 0}
        ]
    }
}


class PromptTemplate:

    """
    Wrapper around plain string prompt templates, and openai-style message lists, both exposing the method `format`.

    Justification:
    - Instantiating prompt templates from prompt info (e.g., .json file) is slightly different in case of scalar, comparative and categorical prompting.
    - I want openai-style 'messages' (list of dicts) to behave the same, on the outside, as a formatable string.
    """

    def __init__(self, *args, mode: Literal['scalar', 'comparative', 'categorical'] = 'scalar', for_chat: bool = False, **kwargs):

        system_prompt, examples, prompt = (
            self.init_for_scalar if mode == 'scalar'
            else self.init_for_comparative if mode == 'comparative'
            else self.init_for_categorical
        )(*args, **kwargs)

        if for_chat:
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


    @staticmethod
    def init_for_scalar(
            system_prompt: str,
            prompt_format: str,
            examples: list[dict],
            labels: list[str]
    ) -> tuple[str, list[tuple[str, str]], str]:
        scale = ', '.join(labels)
        system_prompt = system_prompt.format(scale=scale, scale_min=labels[0], scale_max=labels[-1])
        examples_list = []
        n = 0   # in case no examples
        for n, example in enumerate(examples, start=1):
            examples_list.append((
                prompt_format.format(n=n, item=example['item']),
                labels[example['response']]
            ))    # scale_min=labels[0], scale_max=labels[-1]

        prompt = prompt_format.format(n=n+1, item='{}')  # scale_min=labels[0], scale_max=labels[-1]

        return system_prompt, examples_list, prompt

    @staticmethod
    def init_for_categorical(
            system_prompt: str,
            prompt_format: str,
            examples: list[dict],
            categories: dict,
            labels: list[str]
    ) -> tuple[str, list[tuple[str, str]], str]:
        categories_full = '\n'.join(f'{l}. {c}: {d}' for l, (c, d) in zip(labels, categories.items()))
        category_names = list(categories.keys())
        system_prompt = system_prompt.format(categories=categories_full)
        examples_list = []
        n = 0
        for n, example in enumerate(examples, start=1):
            examples_list.append((
                prompt_format.format(n=n, item=example['item']),
                f"{labels[example['response']]} ({category_names[example['response']]})"
            ))

        prompt = prompt_format.format(n=n+1, item='{}')

        return system_prompt, examples_list, prompt

    @staticmethod
    def init_for_comparative(
            system_prompt: str,
            prompt_format: str,
            examples: list[dict],
            labels: list[str]
    ) -> tuple[str, list[tuple[str, str]], str]:

        def make_choices_str(choices: list[str], labels: list[str]) -> str:
            return '\n'.join((f'{label}. {choice}' for label, choice in zip(labels, choices)))

        labels_str = '/'.join(labels)
        system_prompt = system_prompt
        n = 0
        examples_list = []
        for n, example in enumerate(examples, start=1):
            if example['response'] >= len(labels):
                new_response = random.randint(0, len(labels)-1)
                example['choices'][new_response] = example['choices'][example['response']]
                example['response'] = new_response
                example['choices'] = example['choices'][:len(labels)]
            examples_list.append((
                prompt_format.format(n=n, choices=make_choices_str(example['choices'], labels), labels=labels_str),
                labels[example['response']]
            ))
        prompt_format = prompt_format.format(n=n+1, choices=make_choices_str(['{}' for _ in labels], labels), labels=labels_str)
        return system_prompt, examples_list, prompt_format




