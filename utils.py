import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
import random
import os
import copy
import collections 
import scipy.stats as stats
import netgraph
import powerlaw as pwl
import seaborn as sns
import replicate
import anthropic
import torch
from openai import OpenAI
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    import xgrammar as xgr
except ImportError:
    xgr = None

def _get_required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

claude_api_key = os.getenv('ANTHROPIC_API_KEY')
replicate_api_token = os.getenv('REPLICATE_API_KEY')

claude_client = anthropic.Anthropic(api_key=claude_api_key) if claude_api_key else None
replicate_client = replicate.Client(api_token=replicate_api_token) if replicate_api_token else None
openai_client = OpenAI(
    api_key=_get_required_env('OPENAI_API_KEY'),
    organization=os.getenv('OPENAI_ORG'),
)
hf_clients = {}

def set_plot_sizes():

    MEDIUM_SIZE = 24
    SMALL_SIZE = 0.85 * MEDIUM_SIZE
    BIGGER_SIZE = 1.5 * MEDIUM_SIZE

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def _get_huggingface_client(model):
    if model not in hf_clients:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        config = AutoConfig.from_pretrained(model, trust_remote_code=True)

        hf_model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        hf_clients[model] = (tokenizer, hf_model, config)

    return hf_clients[model]


def _require_xgrammar():
    if xgr is None:
        raise RuntimeError(
            "xgrammar is required for JSON schema constrained decoding. "
            "In Colab, run: pip install xgrammar"
        )


def _infer_vocab_size(tokenizer, model, config):
    candidate_values = []

    value = getattr(config, "vocab_size", None)
    if isinstance(value, int) and value > 0:
        candidate_values.append(("config.vocab_size", value))

    value = getattr(tokenizer, "vocab_size", None)
    if isinstance(value, int) and value > 0:
        candidate_values.append(("tokenizer.vocab_size", value))

    try:
        vocab = tokenizer.get_vocab()
        if isinstance(vocab, dict) and len(vocab) > 0:
            candidate_values.append(("len(tokenizer.get_vocab())", len(vocab)))
    except Exception:
        pass

    embedding_layer = model.get_input_embeddings()
    if embedding_layer is not None and hasattr(embedding_layer, "num_embeddings"):
        value = getattr(embedding_layer, "num_embeddings", None)
        if isinstance(value, int) and value > 0:
            candidate_values.append(("model.get_input_embeddings().num_embeddings", value))

    if not candidate_values:
        raise RuntimeError("Could not infer vocab_size from config, tokenizer, or model.")

    source, vocab_size = max(candidate_values, key=lambda item: item[1])
    return source, vocab_size


def is_huggingface_model_supported(model):
    return True


def filter_supported_models(models):
    supported_models = []

    for model in models:
        if is_huggingface_model_supported(model):
            supported_models.append(model)
            continue

        print(
            f"Skipping {model}: it is not supported in the current environment."
        )

    return supported_models


def _is_openai_reasoning_model(model):
    return model.startswith('gpt-5') or model.startswith('o')


def _supports_reasoning_none(model):
    return 'codex' not in model and model.startswith(('gpt-5.1', 'gpt-5.2', 'gpt-5.4'))


def _is_qwen_thinking_model(model):
    return model.startswith('Qwen/') and ('Qwen3' in model or 'QwQ' in model)


def _is_qwen35_model(model):
    return model.startswith('Qwen/') and 'Qwen3.5' in model


def resolve_cot_config(model, cot=False, cot_config=None):
    config = {}

    if _is_openai_reasoning_model(model):
        if model.startswith('gpt-5-pro'):
            effort = 'high'
        elif cot:
            effort = 'medium'
        elif _supports_reasoning_none(model):
            effort = 'none'
        else:
            effort = 'minimal'

        config['openai_reasoning'] = {'effort': effort}

    elif _is_qwen_thinking_model(model):
        config['qwen_enable_thinking'] = bool(cot)

        if _is_qwen35_model(model):
            if cot:
                config.update({
                    'temperature': 1.0,
                    'top_p': 0.95,
                    'top_k': 20,
                    'min_p': 0.0,
                    'repetition_penalty': 1.0,
                    'max_new_tokens': 2000,
                })
            else:
                config.update({
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 20,
                    'min_p': 0.0,
                    'repetition_penalty': 1.0,
                    'max_new_tokens': 1000,
                })
        else:
            if cot:
                config.update({
                    'temperature': 0.6,
                    'top_p': 0.95,
                    'top_k': 20,
                    'min_p': 0.0,
                    'max_new_tokens': 2000,
                })
            else:
                config.update({
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 20,
                    'min_p': 0.0,
                    'max_new_tokens': 1000,
                })

    if cot_config:
        config.update(cot_config)

    return config


def _apply_chat_template(tokenizer, messages, cot_config):
    kwargs = {
        'tokenize': False,
        'add_generation_prompt': True,
    }
    if 'qwen_enable_thinking' in cot_config:
        kwargs['enable_thinking'] = cot_config['qwen_enable_thinking']

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop('enable_thinking', None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _strip_qwen_thinking(text):
    end_tag = '</think>'
    if end_tag in text:
        return text.split(end_tag, 1)[1].strip()
    return text


def _compile_response_grammar(grammar_compiler, response_schema, qwen_thinking_enabled):
    if qwen_thinking_enabled:
        if not hasattr(grammar_compiler, 'compile_structural_tag'):
            raise RuntimeError(
                "Qwen thinking with structured output requires xgrammar.compile_structural_tag. "
                "Upgrade xgrammar or run Qwen with cot=False for schema-constrained output."
            )

        structural_tag = {
            'type': 'structural_tag',
            'format': {
                'type': 'sequence',
                'elements': [
                    {
                        'type': 'tag',
                        'begin': '<think>',
                        'content': {'type': 'any_text'},
                        'end': '</think>',
                    },
                    {
                        'type': 'regex',
                        'pattern': r'\s*',
                    },
                    {
                        'type': 'json_schema',
                        'json_schema': response_schema,
                    },
                ],
            },
        }
        return grammar_compiler.compile_structural_tag(structural_tag)

    return grammar_compiler.compile_json_schema(json.dumps(response_schema))


def _get_huggingface_response(prompt, model, temperature, system_prompt, response_schema=None, cot=False, cot_config=None):
    tokenizer, hf_model, config = _get_huggingface_client(model)
    cot_config = resolve_cot_config(model, cot=cot, cot_config=cot_config)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    input_text = _apply_chat_template(tokenizer, messages, cot_config)
    model_inputs = tokenizer(input_text, return_tensors="pt")
    device = next(hf_model.parameters()).device
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    generation_kwargs = {
        "max_new_tokens": cot_config.get('max_new_tokens', 1000),
        "pad_token_id": tokenizer.pad_token_id,
    }
    qwen_thinking_enabled = cot_config.get('qwen_enable_thinking') is True
    if response_schema is not None:
        _require_xgrammar()
        vocab_size_source, vocab_size = _infer_vocab_size(tokenizer, hf_model, config)
        print(f"[xgrammar] using vocab_size={vocab_size} from {vocab_size_source}")
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
        grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled_grammar = _compile_response_grammar(grammar_compiler, response_schema, qwen_thinking_enabled)
        generation_kwargs["logits_processor"] = [xgr.contrib.hf.LogitsProcessor(compiled_grammar)]

    temperature = cot_config.get('temperature', temperature)
    if temperature in (None, 0):
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = cot_config.get('top_p', 0.95)

    for key in ('top_k', 'min_p', 'repetition_penalty'):
        if key in cot_config:
            generation_kwargs[key] = cot_config[key]

    outputs = hf_model.generate(**model_inputs, **generation_kwargs)
    generated_tokens = outputs[0][model_inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    if qwen_thinking_enabled:
        text = _strip_qwen_thinking(text)
    return text


def get_response(prompt, model, temperature=0.9, system_prompt="You are mimicking a real-life person who wants to make friends.", response_schema=None, cot=False, cot_config=None):
    cot_config = resolve_cot_config(model, cot=cot, cot_config=cot_config)
    if model.startswith('gpt'):
        request_kwargs = {
            "model": model,
            "instructions": system_prompt,
            "input": prompt,
        }
        if 'openai_reasoning' in cot_config:
            request_kwargs["reasoning"] = cot_config['openai_reasoning']
        if temperature is not None and not model.startswith('gpt-5'):
            request_kwargs["temperature"] = temperature

        result = openai_client.responses.create(**request_kwargs)
        return result.output_text
    elif model.startswith('claude'):
        global claude_client
        result = claude_client.messages.create(
            model = model,
            temperature = temperature,
            system = system_prompt,
            max_tokens = 1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ])
        
        return result.content[0].text
    elif model.startswith('Qwen/'):
        return _get_huggingface_response(prompt, model, temperature, system_prompt, response_schema=response_schema, cot=cot, cot_config=cot_config)
    else:
        global replicate_client
        replicate_input = {
            'prompt' : prompt,
        }
        if temperature is not None:
            replicate_input['temperature'] = temperature

        result = replicate_client.run(model, replicate_input)

        return ''.join(result)

def summarize_reasons(filename, model, outfile, title, n_samples=20, n_categories=5, n_resamples=5, degrees=False, categories=None):
    random.seed(1)
    np.random.seed(1)

    suffix = os.path.splitext(filename)[0]

    with open(filename) as f:
        lines = f.read().splitlines()

    data = []

    for line in lines:

        data.append(json.loads(line))

    reason_list = collections.defaultdict(list)

    all_reasons = []

    for d in data:        
        for result in d["reasons"]:
            if result and 'reason' in result.keys():
                reason_list[d['temperature']].append(result['reason'])
                all_reasons.append(result['reason'])

    if categories is None:
        categorization_prompt = f"""
        # Task

        You are given a list of reasons and your task to find {n_categories} categories that best describe the reasons.

        # Input

        The input is a list of reasons. The list is given below after chevrons:
        <REASONS>
        {json.dumps(random.sample(all_reasons, len(reason_list) * n_samples))}
        </REASONS>

        # Output

        The output should be given in JSON format with the following structure:

        [
            {{
                "category" : category,
                "description" : short description of the category
            }}, ...
        ]

        # Notes
        * The names of the categories must be descriptive and mutually exclusive.

        ```json
        """

        for _ in range(10):
            try:
                ans = get_response(categorization_prompt, temperature=0, system_prompt="You are a helpful assistant", model=model)
                categories = json.loads(ans.split('```')[0])
                print(categories)
                break

            except Exception as e:
                print(e)
        
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    category_list = [c['category'] for c in categories]

    records = []

    for i, (k, v) in enumerate(reason_list.items()):
        print('Temperature', k)
        if len(v) <= n_samples:
            n_resamples = 1

        for r in range(n_resamples):
            prompt = f"""
            # Task
            You are given a list of reasons and your task is to classify them into categories.

            # Input
            The input is a list of reasons. The list is given below after chevrons:
            <REASONS>
            {json.dumps(random.sample(v, n_samples), indent=4)}
            </REASONS>

            ## Categories
            The names of the categories are given below after chevrons:
            <CATEGORIES>
            {json.dumps(categories, indent=4)}
            </CATEGORIES>

            Each reason must be assigned to exactly one of the categories.
            
            # Output
            The output should be given as a list of JSON objects with the following structure:

            [
                {{
                        "reason" : reason,
                        "category" : category name
                }}, ...
            ]

            ```json
            """

            for _ in range(10):
                try:
                    ans = get_response(prompt, temperature=0, system_prompt="You are a helpful assistant", model=model)

                    try:
                        result = json.loads(ans.split('```')[0])
                    except:
                        result = json.loads(ans.split('```json')[1].split('```')[0])

                    assert(isinstance(result, list))

                    reason_types = collections.defaultdict(float)

                    total = 0

                    for reason in result:
                        if reason['category'] in category_list:
                            reason_types[reason['category']] += 1
                            total += 1

                    break
                except Exception as e:
                    print(e)

            for key, val in reason_types.items():
                records.append({
                    'Temperature' : k,
                    'Category' : key,
                    'Frequency' : val,
                    'Resample' : r
                })


    df = pd.DataFrame.from_records(records)

    fig.suptitle(title, fontsize=MEDIUM_SIZE)

    sns.barplot(data=df, x='Category', y='Frequency', hue='Temperature', ax=ax, palette='Set2')

    plt.legend(fontsize=0.75*SMALL_SIZE, title='Temperature')

    plt.xticks(rotation=0, fontsize=SMALL_SIZE)

    fig.tight_layout()

    fig.savefig(outfile, dpi=300, bbox_inches='tight')
