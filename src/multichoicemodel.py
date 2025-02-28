import torch
import copy
import logging
import transformers
import functools
from typing import Union
import tiktoken


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultipleChoiceModel:
    """
    Wrapper around huggingface base LLMS and openai chat models for multiple-choice prompting, implementing a `get_scores` method.

    Justification:
    -
    """

    def __init__(self, model_name, labels, model_is_chat, prompt_start_for_cache=None, client=None):

        if client and model_is_chat:    # assuming openai gpt
            tokenizer = tiktoken.encoding_for_model('gpt-4o' if model_name.startswith('o1') else model_name)    # TODO meh
            label_ids = [tokenizer.encode(label)[0] for label in labels]    # TODO verify they are all singleton ids
            logging.info(f'Using label ids: {label_ids}')
            # TODO: For o1 model, logprobs not supported; so consider disabling the logprobs and just getting the output directly?
            self.get_scores = functools.partial(self.get_multiple_choice_prob_openai, model_name=model_name, client=client, labels=labels, label_ids=label_ids)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name, max_length=200).to(DEVICE)  # TODO Check length adequacy
            model.eval()
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
            space_before_labels = False if model_is_chat else len(tokenizer.encode(' ' + labels[0], add_special_tokens=False)) == 1
            label_ids = [tokenizer.encode(' ' + label if space_before_labels else label, add_special_tokens=False)[-1] for label in labels]  # hmmmm that space tho
            logging.info(f'Using label ids (space_before_labels={space_before_labels}): {label_ids}')
            cached_common_start = create_cache(model, tokenizer, prompt_start_for_cache) if prompt_start_for_cache else None    # TODO fix for chat models

            self.get_scores = functools.partial(self.get_multiple_choice_prob, model=model, tokenizer=tokenizer, label_ids=label_ids, cache=cached_common_start, space_before_labels=space_before_labels)

    def get_scores(self, prompt: Union[str, list[str]]) -> list[float]:
        pass    # Hmmm... Overwritten by init.

    @staticmethod
    def get_multiple_choice_prob(prompt: Union[str, list[dict]], model, tokenizer, label_ids, cache=None, space_before_labels=False) -> list[float]:
        past_key_values = copy.deepcopy(cache) if cache else None
        if isinstance(prompt, str):
            if space_before_labels:
                prompt += ' '
            prompt_encoded = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
        else:   # openai-style messages
            prompt_encoded = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(DEVICE)

        model_output = model.generate(prompt_encoded, pad_token_id=tokenizer.eos_token_id, output_logits=True,
                                return_dict_in_generate=True, do_sample=False, max_new_tokens=1, past_key_values=past_key_values)
        logits = model_output.logits[0][:, label_ids]  # first token logits for the labels
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs[0]  # we use singleton batches only

        return probs.tolist()

    @staticmethod
    def get_multiple_choice_prob_openai(prompt: list[dict], client, model_name, labels, label_ids: list[int]) -> list[float]:

        LOGIT_BIAS = 10

        completion = client.chat.completions.create(
            model=model_name,
            messages=prompt,
            logprobs=True,
            top_logprobs=10,
            logit_bias={l: LOGIT_BIAS for l in label_ids},
            max_completion_tokens=10,
        )

        label_logprobs = {}
        for logprob_dict in completion.choices[0].logprobs.content[0].top_logprobs:
            token, logprob = logprob_dict.token, logprob_dict.logprob
            if token in labels:
                label_logprobs[token] = logprob - LOGIT_BIAS

        newline = "\n"
        logging.info(f'{prompt[-1]["content"].replace(newline, "/")} -> {completion.choices[0].message.content}')

        label_logprobs_tensor = torch.tensor([label_logprobs.get(label, -90) - LOGIT_BIAS for label in labels])
        probs = torch.nn.functional.softmax(label_logprobs_tensor, dim=-1)
        return probs.tolist()


def create_cache(model, tokenizer, common_start: str) -> transformers.DynamicCache:
    with torch.no_grad():
        if isinstance(common_start, str):
            inputs = tokenizer.encode(common_start, return_tensors="pt").to(DEVICE)
        else:
            inputs = tokenizer.apply_chat_template(common_start, return_tensors="pt").to(DEVICE)
        cache = transformers.DynamicCache()
        cache = model(inputs, past_key_values=cache).past_key_values
    return cache

