# ChoiceLLM: A Python package for rating/multiple choice prompting 

## Installation

```bash
pip install git+https://github.com/mwestera/choicellm
```

If you want to use models through the OpenAI api, then you need to set the `OPENAI_API_KEY` environment variable to your key, or (alternatively) add a file `.env` in your working directory containing your OpenAI API key, specified like this:

`OPENAI_API_KEY=yoursecretkey123`

Installation makes available the command `choicellm`, and a helper program `choicellm_aggregate`

## Basic usage

The main command `choicellm` takes a plaintext file as argument and some options.

A minimal way of running the command is as follows:

```bash
choicellm items.txt --model "unsloth/llama-3-70b-bnb-4bit" --mode scalar > results.csv
```

This asks the specified language model (from Huggingface) to rate all the items in `items.txt` (one per line) on a scale from 1-5 representing levels of 'concreteness': how concrete (vs. abstract) is a given word or phrase.

Instead of asking the model to rate items on a scale (`--mode scalar`), we can ask it to choose one of several categories: `--mode categorical`. In this case, the default behavior (see below how to specify custom prompts) is for the model to choose, for each item, between 'concrete', 'neutral' and 'abstract'.

```bash
choicellm items.txt --model "unsloth/llama-3-70b-bnb-4bit" --mode categorical > results.csv
```

Finally, we can set `--mode comparative` to ask the model to compare each item to a large number of random other items, which can result in more reliable per-item scores (once aggregated over all comparisons).

```bash
choicellm items.txt --model "unsloth/llama-3-70b-bnb-4bit" --mode comparative > results.csv
```

In this case, the `results.csv` file will not contain a single score per item, but rather a separate row for each comparison it did. The auxiliary command `choicellm_aggregate` may be used to aggregate these comparisons into a single score per item (in this case resulting in values on the scale 1,5):

```bash
choicellm_aggregate results.csv --scale 1,5 > results_aggregated.csv
```

To see some other options with rather minimal explanations, do `choicellm --help`. To illustrate some of those, to prompt OpenAI's `gpt-4o` for concreteness scores in `comparative` mode, you could do (and don't forget to aggregate the results afterwards with `choicellm_aggregate`):

```bash
choicellm items.txt --model "gpt-4o" --openai --labels A B C --n_choices 3 --n_comparisons 50 --mode comparative > results.csv
```

(Though for a chat model like `gpt-4o`, as opposed to a base model like `unsloth/llama-3-70b-bnb-4bit`, it might be better to use a custom prompt, see below.)

It might be useful to save raw model outputs by redirecting the error stream by appending `2> raw_output.log`, though this will (currently) contain some noise from the progress bar.

## Custom prompts

One of the options is `--prompt`, which lets you specify a `.json` file containing system prompt, prompt template and few-shot examples. If you don't specify a `--prompt` file, the program will use a default prompt. The default prompts were designed for getting _concreteness_ judgments from the LLM (just as an example).

The default prompt, and also the elements of a `.json` file for a custom prompt, depends on the mode (`--mode`): it differs slightly for getting scalar judgements, comparisons and multiple-choice categorization. You can find the default prompts in `src/prompttemplate.py`. To create a custom prompt, it is easiest to copy one of these templates into a new `.json` file and then modify it. 

An example with a custom prompt (say, for sentiment analysis) would be as follows -- assuming you have the relevant prompt specification in a file `sentiment.json`: 

```bash
choicellm items.txt --model "unsloth/llama-3-70b-bnb-4bit" --prompt sentiment.json --mode scalar > results.csv
```

You can also specify which scale to use, with `--labels`, as follows. Since this is a three-point scale, your few-shot examples in the `.json` specification of the prompt should use 0 as minimum and 2 as maximum response value.

```bash
choicellm items.txt --model "unsloth/llama-3-70b-bnb-4bit" --prompt sentiment.json --labels 1 2 3 --mode scalar > results.csv
```

Negative scale values (like a scale -1, 0, 1) are currently not supported.

## Which models to use?

While probably not entirely model-agnostic in some unforeseen ways, it should work with most Huggingface models and OpenAI models. Here are some models that work: 

- Reasonable local options that fit in a 48GB GPU: `unsloth/llama-3-70b-bnb-4bit` or (chat model) `unsloth/Llama-3.3-70B-Instruct-bnb-4bit` (in the latter case, include option `--chat`)
- If you want to use the OpenAI API, you can use their models, e.g., `--model gpt-4o --openai`
- For debugging you could use `xiaodongguaAIGC/llama-3-debug` that is like llama3 but tiny and basically random outputs. 

Note that the default prompts were designed for base models, not instruct/chat models. You can specify the option `--chat` to make sure prompts are submitted in the OpenAI chat format (list of messages). When using a chat model, you may want to manually change the prompt formulation a bit, e.g., use the second-person ('you') perspective to instruct the model, and a question mark instead of a colon. These things are not automatically changed.

Here's an example that asks `gpt-4o` to assign to each plaintext headline (say) from `headlines.txt` the most applicable topic (from some predefined list given in the custom prompt `topics.json`, assuming it exists). 

```bash
choicellm headlines.txt --model "gpt-4o" --openai --prompt topics.json --mode categorical > results.csv 2> raw_llm_output.log
```

