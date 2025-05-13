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

    Its main role is to extract the computed logits to obtain probabilities for the multiple choices.
    """

    def __init__(self, model_name, labels, prompt_start_for_cache=None, openai_client=None):

        if openai_client:
            tokenizer = tiktoken.encoding_for_model(model_name)    # for o1, mind https://github.com/openai/tiktoken/issues/367
            labels = [l.strip() for l in labels]    # Hmmmm strip spaces
            labels_tokenized = [tokenizer.encode(label) for label in labels]
            label_ids = [label_tokenized[0] for label_tokenized in labels_tokenized]
            # TODO: For o1 model, logprobs not supported; so consider disabling the logprobs and just getting the output directly?
            self.get_probs = functools.partial(self.get_multiple_choice_prob_openai, model_name=model_name, client=openai_client, labels=labels, label_ids=label_ids)

            problematic_labels = [label for label, label_tokenized in zip(labels, labels_tokenized) if len(label_tokenized) != 1]
            if problematic_labels:
                raise ValueError(f'Some of the choice labels do not map onto single tokens for your selected model: {", ".join(problematic_labels)}')
            logging.info(f'Using label ids: {label_ids}')

        else:  # assuming local transformers model
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
            labels_tokenized = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
            model.eval()

            cached_common_start = create_cache(model, tokenizer, prompt_start_for_cache) if prompt_start_for_cache else None
            self.get_probs = functools.partial(self.get_multiple_choice_prob,
                                               model=model,
                                               tokenizer=tokenizer,
                                               labels_tokenized=labels_tokenized,
                                               cache=cached_common_start)


    def get_scores(self, prompt: Union[str, list[str]]) -> list[float]:
        pass    # Hmmm... Overwritten by init.

    @staticmethod
    def get_multiple_choice_prob(prompt: Union[str, list[dict]], model, tokenizer, labels_tokenized: list[int], cache=None) -> list[float]:
        """
        Feeds prompt into a local transformers model, and obtains the probabilities of the different multiple-choice
        labels.

        Because the labels can in principle be multi-token, it takes a bit of work to get their probabilities, with iterative
        prompting based on growing prefixes, and multiplying the resulting conditional probabilities.

        When labels are single tokens, this function simply applies the model once and get the logits directly.
        """

        # TODO: Generalize and expose this function, maybe in a more general LLM utils module?

        if isinstance(prompt, str):
            prompt_encoded = tokenizer(prompt, return_tensors='pt').to(DEVICE)
        else:   # openai-style messages
            prompt_encoded = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True, return_dict=True).to(DEVICE)

        choice_probabilities = [1 for _ in labels_tokenized]  # to be iteratively multiplied (conditional probabilities)
        for prefix, label_ids, next_token_ids in get_label_prefixes(tuple(tuple(t) for t in labels_tokenized)):
            prompt_encoded_plus_prefix = torch.cat(
                (prompt_encoded['input_ids'], torch.tensor([prefix], dtype=int).to(DEVICE)),
                dim=-1
            )
            attention_mask_plus_prefix = torch.cat(
                (prompt_encoded['attention_mask'], torch.tensor([[1] * len(prefix)], dtype=int).to(DEVICE)),
                dim=-1
            )
            model_output = model.generate(
                input_ids=prompt_encoded_plus_prefix,
                attention_mask=attention_mask_plus_prefix,  # only to avoid a warning
                pad_token_id=tokenizer.eos_token_id,
                output_logits=True,
                return_dict_in_generate=True,
                do_sample=False,
                max_new_tokens=1,
                past_key_values=copy.deepcopy(cache) if cache else None,
                temperature=None,  # only to avoid a warning
                top_p=None
            )

            next_token_ids_unique = list(set(next_token_ids))
            logits = model_output.logits[0]  # [0] is first (and only) generated token
            logits_to_use = logits[:, next_token_ids_unique]
            probs_to_use = torch.nn.functional.softmax(logits_to_use, dim=-1)[0].tolist()   # [0] is first (and only) batch
            token_id_to_prob = dict(zip(next_token_ids_unique, probs_to_use))

            for label_id, next_token_id in zip(label_ids, next_token_ids):
                choice_probabilities[label_id] *= token_id_to_prob[next_token_id]

        return choice_probabilities


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
            inputs = tokenizer(common_start, return_tensors="pt").to(DEVICE)
        else:
            inputs = tokenizer.apply_chat_template(common_start, return_tensors="pt", return_dict=True).to(DEVICE)

        ## In case multiple beams/sequences, also expand the cache (https://github.com/huggingface/transformers/pull/27576)
        # inputs = inputs.expand(num_beams, *inputs.shape[1:])

        cache = model(input_ids=inputs['input_ids'],
                      attention_mask=inputs['attention_mask'], # only to avoid a warning
                      past_key_values=transformers.DynamicCache(),
                      pad_token_id=tokenizer.eos_token_id).past_key_values
    return cache


@functools.cache
def get_label_prefixes(labels_tokenized: tuple[tuple[int]]) -> list[tuple[tuple[int], list[int], list[int]]]:
    """
    # TODO: Generalize.

    For computing probabilities of multi-token labels (e.g., the two-token string '-1' for negative sentiment),
    beam search isn't quite suitable. So we manually implement beam search, but specifically one beam for each
    possible prefix. To that end, this helper function computes all 'shared prefixes' and their next tokens
    (and index of the corresponding labels).

    Yields 3-tuples for each (shared) prefix and a list of the label_ids that share it, and their respective next_token_ids.

    >>> get_label_prefixes(((2, ), (3, )))
    [((), (0, 1), (2, 3))]
    >>> get_label_prefixes(((1, 2), (1, 3)))
    [((1), (0, 1), (2, 3)), ((), (0, 1), (1, 1)]
    >>> get_label_prefixes(((1, 2, 3), (1, 4, 5), (0, 2), (1, 5)))
    [((1, 2), [0], [3]), ((1, 5), [], []), ((1, 4), [1], [5]), ((1,), [0, 1, 3], [2, 4, 5]), ((0, 2), [], []), ((0,), [2], [2]), ((), [0, 1, 2, 3], [1, 1, 0, 1])]
    """

    max_length = max(len(t) for t in labels_tokenized)
    unique_prefixes = {l[:n] for n in range(max_length) for l in labels_tokenized}
    result = []
    for prefix in unique_prefixes:
        label_ids = []
        next_token_ids = []
        for i, l in enumerate(labels_tokenized):
            if l[:len(prefix)] == prefix and len(l) > len(prefix):
                label_ids.append(i)
                next_token_ids.append(l[len(prefix)])
        result.append((prefix, label_ids, next_token_ids))
    return result