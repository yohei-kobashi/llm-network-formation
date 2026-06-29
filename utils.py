import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
import random
import os
import copy
import collections
import gc
import math
import re
import hashlib
import scipy
import scipy.stats as stats
import netgraph
import powerlaw as pwl
import seaborn as sns
import replicate
import anthropic
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    vllm_import_error = None
except Exception as e:
    LLM = None
    SamplingParams = None
    vllm_import_error = e

try:
    from vllm.sampling_params import StructuredOutputsParams
    structured_outputs_import_error = None
except Exception as e:
    StructuredOutputsParams = None
    structured_outputs_import_error = e

def _get_required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

claude_api_key = os.getenv('ANTHROPIC_API_KEY')
replicate_api_token = os.getenv('REPLICATE_API_KEY')

claude_client = anthropic.Anthropic(api_key=claude_api_key) if claude_api_key else None
replicate_client = replicate.Client(api_token=replicate_api_token) if replicate_api_token else None
openai_client = None
vllm_client = {
    'model': None,
    'llm': None,
    'tokenizer': None,
}
vllm_unavailable_models = set()
vllm_schema_warning_models = set()
transformers_schema_warning_models = set()
transformers_unavailable_models = set()
hf_clients = {}

SHARED_BASELINE_MODEL = 'Qwen/Qwen3.5-4B'
SHARED_MODEL_NAMES = [
    'gpt-5-nano',
    'Qwen/Qwen3.5-4B',
    'Qwen/Qwen3.5-0.8B',
]
SHARED_DEFAULT_TEMPERATURES = [None]
SHARED_DEFAULT_COT_CONFIG = {'max_new_tokens': 16384, 'qwen_enable_thinking': True}
SHARED_COT_RETRY_MAX_NEW_TOKENS = 32768


def get_shared_experiment_settings():
    return {
        'BASELINE_MODEL': SHARED_BASELINE_MODEL,
        'MODEL_NAMES': list(SHARED_MODEL_NAMES),
        'DEFAULT_TEMPERATURES': list(SHARED_DEFAULT_TEMPERATURES),
        'DEFAULT_COT_CONFIG': dict(SHARED_DEFAULT_COT_CONFIG) if SHARED_DEFAULT_COT_CONFIG is not None else None,
        'COT_RETRY_MAX_NEW_TOKENS': SHARED_COT_RETRY_MAX_NEW_TOKENS,
    }


def _get_openai_client():
    global openai_client
    if openai_client is None:
        openai_client = OpenAI(
            api_key=_get_required_env('OPENAI_API_KEY'),
            organization=os.getenv('OPENAI_ORG'),
        )
    return openai_client

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


def _require_vllm():
    if LLM is None or SamplingParams is None:
        detail = f" Import error: {vllm_import_error}" if vllm_import_error else ""
        raise RuntimeError(
            "vLLM is required for Qwen local inference with structured output. "
            "In Colab, run the repository setup cell and restart the runtime if vLLM was just installed."
            f"{detail}"
        )


def _get_vllm_client(model):
    _require_vllm()

    if model in vllm_unavailable_models:
        raise RuntimeError(f"vLLM is unavailable for {model}; using Transformers fallback.")

    if vllm_client['model'] != model:
        vllm_client['model'] = None
        vllm_client['llm'] = None
        vllm_client['tokenizer'] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        llm_kwargs = {
            'model': model,
            'trust_remote_code': True,
            'dtype': 'auto',
        }
        if os.getenv('VLLM_GPU_MEMORY_UTILIZATION'):
            llm_kwargs['gpu_memory_utilization'] = float(os.getenv('VLLM_GPU_MEMORY_UTILIZATION'))
        if os.getenv('VLLM_MAX_MODEL_LEN'):
            llm_kwargs['max_model_len'] = int(os.getenv('VLLM_MAX_MODEL_LEN'))
        if os.getenv('VLLM_TENSOR_PARALLEL_SIZE'):
            llm_kwargs['tensor_parallel_size'] = int(os.getenv('VLLM_TENSOR_PARALLEL_SIZE'))

        try:
            llm = LLM(**llm_kwargs)
        except Exception:
            vllm_unavailable_models.add(model)
            raise
        vllm_client['model'] = model
        vllm_client['llm'] = llm
        vllm_client['tokenizer'] = llm.get_tokenizer()

    return vllm_client['llm'], vllm_client['tokenizer']


def _get_transformers_client(model):
    if model in transformers_unavailable_models:
        raise RuntimeError(f"Transformers fallback is unavailable for {model}.")

    if model not in hf_clients:
        print(f"[transformers fallback] Loading {model}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        hf_model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        hf_model.eval()
        hf_clients[model] = (tokenizer, hf_model)
        print(f"[transformers fallback] Loaded {model}.", flush=True)

    return hf_clients[model]


def _try_transformers_response(prompt, model, temperature, system_prompt, response_schema=None, cot=False, cot_config=None):
    try:
        return _get_transformers_response(
            prompt,
            model,
            temperature,
            system_prompt,
            response_schema=response_schema,
            cot=cot,
            cot_config=cot_config,
        )
    except Exception as e:
        transformers_unavailable_models.add(model)
        print(f"[transformers fallback] {model} is unavailable: {str(e).splitlines()[0]}")
        return None


def _try_transformers_responses(prompts, model, temperature, system_prompt, response_schemas=None, cot=False, cot_config=None):
    try:
        return _get_transformers_responses(
            prompts,
            model,
            temperature,
            system_prompt,
            response_schemas=response_schemas,
            cot=cot,
            cot_config=cot_config,
        )
    except Exception as e:
        transformers_unavailable_models.add(model)
        print(f"[transformers fallback] {model} is unavailable: {str(e).splitlines()[0]}")
        return [None] * len(prompts)


def is_huggingface_model_supported(model):
    if model.startswith('Qwen/') and LLM is None:
        detail = f" Import error: {vllm_import_error}" if vllm_import_error else ""
        print(f"vLLM is not importable in this runtime for {model}; using Transformers fallback.{detail}")
        return True

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
    final_marker = 'FINAL_JSON:'
    if final_marker in text:
        return text.rsplit(final_marker, 1)[1].strip()
    end_tag = '</think>'
    if end_tag in text:
        text = text.split(end_tag, 1)[1].strip()
        if final_marker in text:
            return text.rsplit(final_marker, 1)[1].strip()
        return text
    return text


def find_first_json_object_end(text, require_name=True):
    """Return the character offset after the first parseable JSON object."""
    if text is None:
        return None

    decoder = json.JSONDecoder()
    candidates = []
    final_marker = 'FINAL_JSON:'
    if final_marker in text:
        marker_start = text.rfind(final_marker)
        marker_text = text[marker_start + len(final_marker):]
        for match_offset, char in enumerate(marker_text):
            if char == '{':
                candidates.append(marker_start + len(final_marker) + match_offset)

    candidates.extend(i for i, char in enumerate(text) if char == '{')
    seen = set()
    for start in candidates:
        if start in seen:
            continue
        seen.add(start)
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and (not require_name or 'name' in value):
            return start + end
    return None


def count_text_tokens(text, tokenizer=None):
    if text is None:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return len(tokenizer.encode(text))
    return max(1, math.ceil(len(text) / 4))


def summarize_generation_budget_calibration(
    outputs,
    tokenizer=None,
    parse_successes=None,
    percentile=0.90,
    margin=1.5,
    buckets=(8192, 16384, 32768, 65536),
    minimum_budget=8192,
):
    records = []
    final_positions = []
    if parse_successes is None:
        parse_successes = [None] * len(outputs)

    for i, (text, parse_success) in enumerate(zip(outputs, parse_successes)):
        text = '' if text is None else str(text)
        json_end_char = find_first_json_object_end(text)
        inferred_success = json_end_char is not None
        is_success = inferred_success if parse_success is None else bool(parse_success)
        generated_tokens = count_text_tokens(text, tokenizer=tokenizer)
        final_json_position_tokens = (
            count_text_tokens(text[:json_end_char], tokenizer=tokenizer)
            if json_end_char is not None
            else None
        )

        record = {
            'index': i,
            'parse_success': is_success,
            'generated_tokens': generated_tokens,
            'final_json_position_tokens': final_json_position_tokens,
            'json_found': inferred_success,
            'output_chars': len(text),
        }
        records.append(record)
        if is_success and final_json_position_tokens is not None:
            final_positions.append(final_json_position_tokens)

    if final_positions:
        threshold = float(np.quantile(final_positions, percentile)) * margin
        selected_budget = max(minimum_budget, int(math.ceil(threshold)))
        for bucket in sorted(buckets):
            if selected_budget <= bucket:
                selected_budget = bucket
                break
        else:
            selected_budget = max(buckets)
    else:
        threshold = None
        selected_budget = max(buckets)

    summary = {
        'selected_max_new_tokens': selected_budget,
        'percentile': percentile,
        'margin': margin,
        'threshold_tokens': threshold,
        'num_outputs': len(records),
        'num_parse_success': sum(1 for record in records if record['parse_success']),
        'num_json_found': sum(1 for record in records if record['json_found']),
        'final_json_position_tokens': final_positions,
        'records': records,
    }
    return summary


def _build_vllm_sampling_params(temperature, cot_config, response_schema=None):
    sampling_kwargs = {
        'max_tokens': cot_config.get('max_new_tokens', 1000),
    }

    if temperature is None and 'temperature' in cot_config:
        temperature = cot_config['temperature']

    if temperature in (None, 0):
        sampling_kwargs['temperature'] = 0.0
    else:
        sampling_kwargs['temperature'] = temperature

    for key in ('top_p', 'top_k', 'min_p', 'repetition_penalty'):
        if key in cot_config:
            sampling_kwargs[key] = cot_config[key]

    if response_schema is not None:
        if StructuredOutputsParams is None:
            model_key = cot_config.get('model_name', 'unknown')
            if model_key not in vllm_schema_warning_models:
                detail = f" Import error: {structured_outputs_import_error}" if structured_outputs_import_error else ""
                print(f"[vLLM] StructuredOutputsParams is unavailable; response_schema will not be enforced.{detail}")
                vllm_schema_warning_models.add(model_key)
        else:
            sampling_kwargs['structured_outputs'] = StructuredOutputsParams(json=response_schema)

    return SamplingParams(**sampling_kwargs)


def _get_vllm_response(prompt, model, temperature, system_prompt, response_schema=None, cot=False, cot_config=None):
    llm, tokenizer = _get_vllm_client(model)
    cot_config = resolve_cot_config(model, cot=cot, cot_config=cot_config)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    input_text = _apply_chat_template(tokenizer, messages, cot_config)
    qwen_thinking_enabled = cot_config.get('qwen_enable_thinking') is True
    sampling_params = _build_vllm_sampling_params(
        temperature,
        cot_config,
        response_schema=response_schema,
    )

    outputs = llm.generate([input_text], sampling_params=sampling_params, use_tqdm=False)
    text = outputs[0].outputs[0].text
    if qwen_thinking_enabled and cot_config.get('strip_qwen_thinking', True):
        text = _strip_qwen_thinking(text)
    return text


def _get_transformers_response(prompt, model, temperature, system_prompt, response_schema=None, cot=False, cot_config=None):
    return _get_transformers_responses(
        [prompt],
        model,
        temperature,
        system_prompt,
        response_schemas=[response_schema] if response_schema is not None else None,
        cot=cot,
        cot_config=cot_config,
    )[0]


def _get_vllm_responses(prompts, model, temperature, system_prompt, response_schemas=None, cot=False, cot_config=None):
    llm, tokenizer = _get_vllm_client(model)
    cot_config = resolve_cot_config(model, cot=cot, cot_config=cot_config)
    if response_schemas is None:
        response_schemas = [None] * len(prompts)
    elif len(response_schemas) != len(prompts):
        raise ValueError("response_schemas must be None or the same length as prompts.")

    input_texts = []
    sampling_params = []
    for prompt, response_schema in zip(prompts, response_schemas):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        input_texts.append(_apply_chat_template(tokenizer, messages, cot_config))
        sampling_params.append(
            _build_vllm_sampling_params(
                temperature,
                cot_config,
                response_schema=response_schema,
            )
        )

    outputs = llm.generate(input_texts, sampling_params=sampling_params, use_tqdm=False)
    texts = [output.outputs[0].text for output in outputs]
    if cot_config.get('qwen_enable_thinking') is True and cot_config.get('strip_qwen_thinking', True):
        texts = [_strip_qwen_thinking(text) for text in texts]
    return texts


def _get_transformers_responses(prompts, model, temperature, system_prompt, response_schemas=None, cot=False, cot_config=None):
    tokenizer, hf_model = _get_transformers_client(model)
    cot_config = resolve_cot_config(model, cot=cot, cot_config=cot_config)
    print(
        f"[transformers fallback] Generating {len(prompts)} response(s) for {model} "
        f"with max_new_tokens={cot_config.get('max_new_tokens', 1000)}...",
        flush=True,
    )
    input_texts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        input_texts.append(_apply_chat_template(tokenizer, messages, cot_config))

    model_inputs = tokenizer(input_texts, return_tensors="pt", padding=True)
    device = next(hf_model.parameters()).device
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    generation_kwargs = {
        "max_new_tokens": cot_config.get('max_new_tokens', 1000),
        "pad_token_id": tokenizer.pad_token_id,
    }

    if temperature is None and 'temperature' in cot_config:
        temperature = cot_config['temperature']

    if temperature in (None, 0):
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature

    for key in ('top_p', 'top_k', 'min_p', 'repetition_penalty'):
        if key in cot_config:
            generation_kwargs[key] = cot_config[key]

    if (
        response_schemas
        and any(schema is not None for schema in response_schemas)
        and model not in transformers_schema_warning_models
    ):
        print(f"[transformers fallback] response_schema is not enforced for {model}; relying on prompt instructions.")
        transformers_schema_warning_models.add(model)

    with torch.inference_mode():
        outputs = hf_model.generate(**model_inputs, **generation_kwargs)
    print(f"[transformers fallback] Finished generation for {model}.", flush=True)

    input_length = model_inputs["input_ids"].shape[-1]
    texts = []
    for output in outputs:
        generated_tokens = output[input_length:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        if cot_config.get('qwen_enable_thinking') is True and cot_config.get('strip_qwen_thinking', True):
            text = _strip_qwen_thinking(text)
        texts.append(text)

    return texts


def get_response(prompt, model, temperature=0.9, system_prompt="You are mimicking a real-life person who wants to make friends.", response_schema=None, cot=False, cot_config=None):
    cot_config = resolve_cot_config(model, cot=cot, cot_config=cot_config)
    if model.startswith('gpt'):
        client = _get_openai_client()
        request_kwargs = {
            "model": model,
            "instructions": system_prompt,
            "input": prompt,
        }
        if 'openai_reasoning' in cot_config:
            request_kwargs["reasoning"] = cot_config['openai_reasoning']
        if temperature is not None and not model.startswith('gpt-5'):
            request_kwargs["temperature"] = temperature

        result = client.responses.create(**request_kwargs)
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
        if model in vllm_unavailable_models:
            return _try_transformers_response(prompt, model, temperature, system_prompt, response_schema=response_schema, cot=cot, cot_config=cot_config)
        try:
            return _get_vllm_response(prompt, model, temperature, system_prompt, response_schema=response_schema, cot=cot, cot_config=cot_config)
        except Exception as e:
            vllm_unavailable_models.add(model)
            print(f"[vLLM fallback] {model} failed to load or generate with vLLM: {str(e).splitlines()[0]}")
            print(f"[vLLM fallback] Retrying {model} with Transformers.")
            return _try_transformers_response(prompt, model, temperature, system_prompt, response_schema=response_schema, cot=cot, cot_config=cot_config)
    else:
        global replicate_client
        replicate_input = {
            'prompt' : prompt,
        }
        if temperature is not None:
            replicate_input['temperature'] = temperature

        result = replicate_client.run(model, replicate_input)

        return ''.join(result)


def get_responses(prompts, model, temperature=0.9, system_prompt="You are mimicking a real-life person who wants to make friends.", response_schemas=None, cot=False, cot_config=None):
    if model.startswith('Qwen/'):
        if model in vllm_unavailable_models:
            return _try_transformers_responses(prompts, model, temperature, system_prompt, response_schemas=response_schemas, cot=cot, cot_config=cot_config)
        try:
            return _get_vllm_responses(prompts, model, temperature, system_prompt, response_schemas=response_schemas, cot=cot, cot_config=cot_config)
        except Exception as e:
            vllm_unavailable_models.add(model)
            print(f"[vLLM fallback] {model} failed to load or generate with vLLM: {str(e).splitlines()[0]}")
            print(f"[vLLM fallback] Retrying {model} with Transformers.")
            return _try_transformers_responses(prompts, model, temperature, system_prompt, response_schemas=response_schemas, cot=cot, cot_config=cot_config)

    if response_schemas is None:
        response_schemas = [None] * len(prompts)
    elif len(response_schemas) != len(prompts):
        raise ValueError("response_schemas must be None or the same length as prompts.")

    return [
        get_response(prompt, model, temperature=temperature, system_prompt=system_prompt, response_schema=response_schema, cot=cot, cot_config=cot_config)
        for prompt, response_schema in zip(prompts, response_schemas)
    ]

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

# --- Principle 2 network-formation utilities ---
PRINCIPLE2_MEDIUM_SIZE = 26
PRINCIPLE2_SMALL_SIZE = 0.85 * PRINCIPLE2_MEDIUM_SIZE
PRINCIPLE2_BIGGER_SIZE = 1.5 * PRINCIPLE2_MEDIUM_SIZE
MEDIUM_SIZE = PRINCIPLE2_MEDIUM_SIZE
SMALL_SIZE = PRINCIPLE2_SMALL_SIZE
BIGGER_SIZE = PRINCIPLE2_BIGGER_SIZE
IGNORE_EXISTING_OUTPUTS = False
COT_RETRY_MAX_NEW_TOKENS = 32768
RESET_OUTFILES = set()


def set_principle2_runtime_options(ignore_existing_outputs=False, cot_retry_max_new_tokens=32768, medium_size=26):
    global IGNORE_EXISTING_OUTPUTS, COT_RETRY_MAX_NEW_TOKENS
    global MEDIUM_SIZE, SMALL_SIZE, BIGGER_SIZE
    IGNORE_EXISTING_OUTPUTS = bool(ignore_existing_outputs)
    COT_RETRY_MAX_NEW_TOKENS = int(cot_retry_max_new_tokens)
    MEDIUM_SIZE = medium_size
    SMALL_SIZE = 0.85 * MEDIUM_SIZE
    BIGGER_SIZE = 1.5 * MEDIUM_SIZE
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=0.7 * SMALL_SIZE)
    plt.rc('ytick', labelsize=0.7 * SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)


def build_response_schema(candidate_names):
    schema_name_enum = list(candidate_names)
    schema_name_type = "integer" if all(isinstance(name, int) for name in candidate_names) else "string"

    return {
        "type": "object",
        "properties": {
            "name": {
                "type": schema_name_type,
                "enum": schema_name_enum,
            },
            "reason": {
                "type": "string",
            },
        },
        "required": ["name", "reason"],
        "additionalProperties": False,
    }


def build_response_list_schema(candidate_names, max_items=4):
    return {
        "type": "array",
        "items": build_response_schema(candidate_names),
        "minItems": 1,
        "maxItems": max_items,
    }


def build_acceptance_response_schema():
    return {
        "type": "object",
        "properties": {
            "accept": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["accept", "reason"],
        "additionalProperties": False,
    }


def _clean_model_json_text(text):
    text = text.strip()
    if 'FINAL_JSON:' in text:
        text = text.rsplit('FINAL_JSON:', 1)[1].strip()
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    return text


def _balanced_json_fragment(text, opener, closer):
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def first_json_object(text):
    text = _clean_model_json_text(text)
    fenced = re.search(r"(?is)```(?:json)?\s*(\{.*?\})\s*```", text)
    direct_candidates = [text]
    if fenced:
        direct_candidates.insert(0, fenced.group(1).strip())

    for candidate in direct_candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            pass

    fragment = _balanced_json_fragment(text, "{", "}")
    if fragment is None:
        return None
    try:
        return json.loads(fragment)
    except Exception:
        return None


def first_json_array(text):
    text = _clean_model_json_text(text)
    fenced = re.search(r"(?is)```(?:json)?\s*(\[.*?\])\s*```", text)
    direct_candidates = [text]
    if fenced:
        direct_candidates.insert(0, fenced.group(1).strip())

    for candidate in direct_candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            pass

    fragment = _balanced_json_fragment(text, "[", "]")
    if fragment is None:
        return None
    try:
        return json.loads(fragment)
    except Exception:
        return None


def normalize_name(value, candidate_names):
    if value in candidate_names:
        return value

    candidate_names_by_str = {str(name): name for name in candidate_names}

    if isinstance(value, int):
        return candidate_names_by_str.get(str(value))

    if not isinstance(value, str):
        return None

    stripped = value.strip()
    if stripped in candidate_names:
        return stripped
    if stripped in candidate_names_by_str:
        return candidate_names_by_str[stripped]
    if stripped.lower().startswith("person "):
        return candidate_names_by_str.get(stripped.split(" ", 1)[1].strip())
    if stripped.lower().startswith("candidate "):
        return candidate_names_by_str.get(stripped.split(" ", 1)[1].strip())

    return None

def principle2_network_growth(n0, temperature=None, model='gpt-5-mini', environment=None, role='friends', method='llm', num_common_neighbors=True, cot=False, cot_config=None, er=False):

    if method == 'sbm':
        G = nx.stochastic_block_model([n0 // 2, n0 // 2], [[0.5, 0.1], [0.1, 0.5]])

        return [G], []

    else:
        if er:
            G = nx.erdos_renyi_graph(n0, 0.1, seed=0)
        else:
            G = nx.stochastic_block_model([n0 // 2, n0 // 2], [[0.5, 0.1], [0.1, 0.5]], seed=0)


    Gs = [G.copy()]
    results = []

    for t in G.nodes():

        if method == 'llm':
            result = principle2_select_neighbor(G, t, temperature, num_common_neighbors=num_common_neighbors, model=model, environment=environment, role=role, cot=cot, cot_config=cot_config)
            if result:
                v = result['name']
                G.add_edge(t, v)
                results.append(result)
        elif method == 'random':
            v = random.choice(list(set(G.nodes() - set(G.neighbors(t)))))
            G.add_edge(t, v)
            results.append({'name' : v, 'common_friends' : list(set(G.neighbors(v)) & set(G.neighbors(t))), 'reason' : 'random'})
        elif method == 'winner':
            v = None,
            max_common_friends = 0
            for u in G.nodes():
                if u not in G.neighbors(t) and u != t and len(set(G.neighbors(u)) & set(G.neighbors(t))) > max_common_friends:
                    v = u
                    max_common_friends = len(set(G.neighbors(u)) & set(G.neighbors(t)))

            G.add_edge(t, v)
            results.append({'name' : v, 'common_friends' : list(set(G.neighbors(v)) & set(G.neighbors(t))), 'reason' : 'winner'})

        Gs.append(G.copy())

    return Gs, results

def principle2_build_neighbor_request(G, t, environment, role, num_common_neighbors, cot, model, cot_config=None):
    candidate_profiles = []
    for v in G.nodes():
        if v != t and v not in G.neighbors(t):
            if num_common_neighbors:
                candidate_profiles.append({'name' : v, 'common_friends' : len(set(G.neighbors(v)) & set(G.neighbors(t)))})
            else:
                candidate_profiles.append({'name' : v, 'friends' : list(G.neighbors(v))})

    if cot:
        output_format = f"""
    {{
        "reason" : reason for selecting the person,
        "name" : name of the person you selected
    }}
        """
    else:
        output_format = f"""
    {{
        "name" : name of the person you selected,
        "reason" : reason for selecting the person
    }}
        """

    candidate_names = [candidate['name'] for candidate in candidate_profiles]
    response_schema = build_response_schema(candidate_names)
    use_structured_output = not (model.startswith('Qwen/') and cot)
    allowed_names_json = json.dumps(candidate_names, ensure_ascii=False)

    prompt = f"""
    # Task
    {f'You are in a {environment}.' if environment else ''}Your task is to select a person to be {role} with.

    # Input
    The input is a list of dictionaries.
    Your profile is given below after chevrons:
    <PROFILE>
    {json.dumps({'name' : t, 'friends' : list(G.neighbors(t))}, separators=(',', ':'))}
    </PROFILE>

    The list of candidate profiles is given below after chevrons:
    <PROFILES>
    {json.dumps(candidate_profiles, separators=(',', ':'))}
    </PROFILES>

    # Output
    The output should be given in JSON format with the following structure

    {output_format}

    # Notes
    * Return exactly one JSON object.
    * The "reason" value must be at most 10 words.
    * Do not list node IDs or friend IDs in "reason".
    * If your chat template enables thinking, keep reasoning in the thinking section.
    * After any thinking, write a line starting with FINAL_JSON: followed by exactly one JSON object.
    * Do not explain your reasoning outside the JSON object in the final answer.
    * Do not write markdown fences.
    * Do not write any text before or after the JSON object.
    * The value of "name" must be exactly one of these values: {allowed_names_json}
    * Do not rename the person.
    * Do not output labels such as "person 0", "Person 0", or "candidate 0".
    * The final answer must be exactly one JSON object and must not contain text after the JSON object.
    """

    return {
        'prompt': prompt,
        'response_schema': response_schema if use_structured_output else None,
        'candidate_names': set(candidate_names),
    }


def principle2_parse_neighbor_response(ans, request):
    result = first_json_object(ans)
    recovered = False
    if not isinstance(result, dict) or 'name' not in result:
        result = recover_name_from_malformed_response(ans, request['candidate_names'])
        recovered = True
    normalized_name = normalize_name(result['name'], request['candidate_names'])
    if normalized_name is None:
        raise ValueError(f"Invalid candidate name: {result['name']}")
    result['name'] = normalized_name
    if recovered:
        result.setdefault('reason', 'Recovered from malformed JSON response')
        result['parse_recovered'] = True
    return result


def recover_name_from_malformed_response(ans, candidate_names):
    if ans is None:
        raise ValueError('Could not parse a valid JSON object with a name field.')

    text = str(ans)
    patterns = [
        r'"name"\s*:\s*(-?\d+)',
        r"'name'\s*:\s*(-?\d+)",
        r'"name"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"',
        r"'name'\s*:\s*'([^']*)'",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        value = match.group(1)
        if re.fullmatch(r'-?\d+', value):
            value = int(value)
        normalized_name = normalize_name(value, candidate_names)
        if normalized_name is not None:
            print(f'Recovered candidate name from malformed JSON response: {normalized_name}')
            return {'name': normalized_name, 'reason': 'Recovered from malformed JSON response'}

    raise ValueError('Could not parse a valid JSON object with a name field.')


def print_llm_parse_error(e, ans, context='', max_chars=4000):
    prefix = f'LLM response parse error ({context})' if context else 'LLM response parse error'
    print(f'{prefix}: {type(e).__name__}: {e}')
    if ans is None:
        print('LLM raw output: <unavailable>')
        return

    text = str(ans)
    truncated = len(text) > max_chars
    preview = text[:max_chars]
    print('LLM raw output repr:', repr(preview))
    if truncated:
        print(f'LLM raw output truncated at {max_chars} of {len(text)} characters.')


def retry_cot_config(cot_config, attempt):
    if not cot_config:
        return cot_config
    config = dict(cot_config)
    if attempt <= 0:
        return config
    base_tokens = int(config.get('max_new_tokens', 1000))
    retry_max_new_tokens = int(globals().get('COT_RETRY_MAX_NEW_TOKENS', 32768))
    config['max_new_tokens'] = max(base_tokens, retry_max_new_tokens)
    if attempt >= 2:
        config['qwen_enable_thinking'] = False
    return config


def principle2_select_neighbor(G, t, temperature, model, environment, role, num_common_neighbors, cot, cot_config=None):
    request = principle2_build_neighbor_request(G, t, environment, role, num_common_neighbors, cot, model, cot_config)
    for i in range(3):
        ans = None
        try:
            attempt_cot_config = retry_cot_config(cot_config, i) if cot else cot_config
            if cot and attempt_cot_config and attempt_cot_config != cot_config:
                print(f'Retrying with max_new_tokens={attempt_cot_config["max_new_tokens"]}')
            ans = get_response(request['prompt'], temperature=temperature, system_prompt="You are a helpful assistant", model=model, response_schema=request['response_schema'], cot=cot, cot_config=attempt_cot_config)
            result = principle2_parse_neighbor_response(ans, request)
            if i > 0:
                result['retry_attempt'] = i + 1
                if attempt_cot_config and 'max_new_tokens' in attempt_cot_config:
                    result['retry_max_new_tokens'] = attempt_cot_config['max_new_tokens']
            print('NEW EDGE', result)
            return result
        except Exception as e:
            print_llm_parse_error(e, ans, context=f'principle2_select_neighbor attempt={i + 1}, node={t}, model={model}')

def principle2_initialize_growth_state(n, temperature, experiment_record):
    if experiment_record['er']:
        G = nx.erdos_renyi_graph(n, 0.1, seed=0)
    else:
        G = nx.stochastic_block_model([n // 2, n // 2], [[0.5, 0.1], [0.1, 0.5]], seed=0)

    return {
        'n': n,
        'temperature': temperature,
        'temperature_label': 'default' if temperature is None else temperature,
        'model': experiment_record['model'],
        'environment': experiment_record['environment'],
        'role': experiment_record['role'],
        'num_common_neighbors': experiment_record['num_common_neighbors'],
        'cot': experiment_record['cot'],
        'cot_config': experiment_record.get('cot_config'),
        'er': experiment_record['er'],
        'outfile': experiment_record['outfile'],
        'metadata': experiment_record['metadata'],
        'G': G,
        'node_order': list(G.nodes()),
        't_idx': 0,
        'graphs': [G.copy()],
        'reasons': [],
    }


RESET_OUTFILES = set()


def principle2_checkpoint_filename(outfile, n, simulation, temperature_label):
    stem, ext = os.path.splitext(outfile)
    safe_temperature = str(temperature_label).replace(os.sep, '_')
    return f'{stem}.n{n}.sim{simulation}.temp{safe_temperature}.checkpoint.json'


def principle2_serialize_growth_state(state):
    temp = {
        'n' : state['n'],
        'temperature' : state['temperature_label'],
        'simulation' : state['simulation'],
        'graphs' : [nx.to_dict_of_lists(G) for G in state['graphs']],
        'reasons' : state['reasons'],
        'model' : state['model'],
        'environment' : state['environment'] if state['environment'] is not None else 'Baseline',
        'role' : state['role'],
        'num_common_neighbors' : state['num_common_neighbors'],
        'cot' : state['cot'],
        'er' : state['er'],
    }
    if state['metadata']:
        temp.update(state['metadata'])
    return temp


def principle2_save_growth_checkpoint(state):
    checkpoint = principle2_serialize_growth_state(state)
    checkpoint['t_idx'] = state['t_idx']
    checkpoint['node_order'] = state['node_order']
    checkpoint_path = principle2_checkpoint_filename(state['outfile'], state['n'], state['simulation'], state['temperature_label'])
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f)
        f.flush()


def principle2_remove_growth_checkpoint(state):
    checkpoint_path = principle2_checkpoint_filename(state['outfile'], state['n'], state['simulation'], state['temperature_label'])
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


def principle2_load_growth_checkpoint(experiment_record, n, simulation, temperature, temperature_label):
    checkpoint_path = principle2_checkpoint_filename(experiment_record['outfile'], n, simulation, temperature_label)
    if not os.path.exists(checkpoint_path):
        return None
    with open(checkpoint_path) as f:
        checkpoint = json.load(f)
    state = principle2_initialize_growth_state(n, temperature, experiment_record)
    state['simulation'] = simulation
    state['temperature_label'] = temperature_label
    state['node_order'] = checkpoint.get('node_order', state['node_order'])
    state['t_idx'] = checkpoint['t_idx']
    state['graphs'] = []
    for graph in checkpoint['graphs']:
        G = nx.Graph()
        for k, values in graph.items():
            k = int(k)
            G.add_node(k)
            for value in values:
                G.add_edge(k, value)
        state['graphs'].append(G)
    state['G'] = state['graphs'][-1].copy()
    state['reasons'] = checkpoint.get('reasons', [])
    print(f'Resuming checkpoint for n={n}, i={simulation}, temperature={temperature_label}, t_idx={state["t_idx"]}')
    return state


def principle2_maybe_reset_outfile(outfile):
    if not IGNORE_EXISTING_OUTPUTS:
        return
    if outfile in RESET_OUTFILES:
        return
    if os.path.exists(outfile):
        os.remove(outfile)
        print(f'Removed existing output file {outfile}')
    RESET_OUTFILES.add(outfile)


def principle2_pending_growth_states_for_experiment(experiment_record):
    parameters = experiment_record['parameters']
    temperatures = experiment_record['temperatures']
    outfile = experiment_record['outfile']
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    principle2_maybe_reset_outfile(outfile)

    saved_scenarios = set()
    if os.path.exists(outfile) and not IGNORE_EXISTING_OUTPUTS:
        with open(outfile) as f:
            lines = f.read().splitlines()
        for line in lines:
            scenario = json.loads(line)
            saved_scenarios.add((scenario['n'], scenario['simulation'], scenario['temperature']))

    print(f'Loaded {len(saved_scenarios)} completed simulations from {outfile}')
    states = []
    for n in range(parameters['n_min'], parameters['n_max'] + 1, parameters['n_step']):
        for i in range(parameters['num_simulations']):
            for temperature in temperatures:
                temperature_label = 'default' if temperature is None else temperature
                if (n, i, temperature_label) in saved_scenarios:
                    print(f'Skipping simulation for n={n}, i={i}, temperature={temperature_label}')
                    continue
                checkpoint_state = principle2_load_growth_checkpoint(experiment_record, n, i, temperature, temperature_label)
                if checkpoint_state is not None:
                    states.append(checkpoint_state)
                    continue
                print(f'Queueing simulation for n={n}, i={i}, temperature={temperature_label}, outfile={outfile}')
                state = principle2_initialize_growth_state(n, temperature, experiment_record)
                state['simulation'] = i
                states.append(state)
    return states


def principle2_advance_growth_state(state, result):
    t = state['node_order'][state['t_idx']]
    if result:
        state['G'].add_edge(t, result['name'])
        state['reasons'].append(result)
    state['graphs'].append(state['G'].copy())
    state['t_idx'] += 1
    principle2_save_growth_checkpoint(state)


def principle2_write_growth_state(state):
    temp = principle2_serialize_growth_state(state)
    with open(state['outfile'], 'a+') as f:
        f.write(json.dumps(temp) + '\n')
        f.flush()
    principle2_remove_growth_checkpoint(state)


def principle2_run_network_formation_experiments_batch(experiment_records):
    if not experiment_records:
        return

    model = experiment_records[0]['model']
    cot = experiment_records[0]['cot']
    cot_config = experiment_records[0].get('cot_config')
    active_states = []
    for experiment_record in experiment_records:
        active_states.extend(principle2_pending_growth_states_for_experiment(experiment_record))

    if not active_states:
        print(f'All batched simulations already completed for {model}. Skipping inference.')
        return

    print(f'Running {len(active_states)} batched simulations for {model}, cot={cot}')
    while active_states:
        requests_by_temperature = collections.defaultdict(list)
        for state in active_states:
            t = state['node_order'][state['t_idx']]
            print(f'Adding edge for node {t} in {state["metadata"]["experiment_name"]}, simulation={state["simulation"]}')
            request = principle2_build_neighbor_request(
                state['G'],
                t,
                state['environment'],
                state['role'],
                state['num_common_neighbors'],
                state['cot'],
                state['model'],
                state.get('cot_config'),
            )
            requests_by_temperature[state['temperature']].append((state, request))

        results_by_state_id = {}
        for temperature, batch_items in requests_by_temperature.items():
            remaining = list(batch_items)
            for attempt in range(3):
                if not remaining:
                    break

                prompts = [request['prompt'] for _, request in remaining]
                response_schemas = [request['response_schema'] for _, request in remaining]
                attempt_cot_config = retry_cot_config(cot_config, attempt) if cot else cot_config
                if cot and attempt_cot_config and attempt_cot_config != cot_config:
                    print(f'Retrying batch with max_new_tokens={attempt_cot_config["max_new_tokens"]}')
                answers = get_responses(
                    prompts,
                    model,
                    temperature=temperature,
                    system_prompt="You are a helpful assistant",
                    response_schemas=response_schemas,
                    cot=cot,
                    cot_config=attempt_cot_config,
                )
                if all(ans is None for ans in answers):
                    print(f'No local answers available for {model}; marking this batch step as missing.')
                    break

                next_remaining = []
                for (state, request), ans in zip(remaining, answers):
                    try:
                        result = principle2_parse_neighbor_response(ans, request)
                        if attempt > 0:
                            result['retry_attempt'] = attempt + 1
                            if attempt_cot_config and 'max_new_tokens' in attempt_cot_config:
                                result['retry_max_new_tokens'] = attempt_cot_config['max_new_tokens']
                        print('NEW EDGE', result)
                        results_by_state_id[id(state)] = result
                    except Exception as e:
                        print_llm_parse_error(
                            e,
                            ans,
                            context=(
                                f'batch attempt={attempt + 1}, '
                                f'experiment={state["metadata"]["experiment_name"]}, '
                                f'simulation={state["simulation"]}, '
                                f'node={state["node_order"][state["t_idx"]]}, '
                                f'model={state["model"]}, '
                                f'temperature={state["temperature_label"]}'
                            ),
                        )
                        next_remaining.append((state, request))
                remaining = next_remaining

            for state, _ in remaining:
                results_by_state_id[id(state)] = None

        next_active_states = []
        for state in active_states:
            principle2_advance_growth_state(state, results_by_state_id.get(id(state)))
            if state['t_idx'] >= len(state['node_order']):
                principle2_write_growth_state(state)
            else:
                next_active_states.append(state)
        active_states = next_active_states


def principle2_run_network_formation_experiment(n_min, n_max, n_step, num_simulations, outfile, temperatures=None, method='llm', model='gpt-5-mini', environment=None, role='friends', num_common_neighbors=True, cot=False, cot_config=None, er=False, metadata=None):
    """ Run the network formation experiment."""
    if temperatures is None:
        temperatures = [None]

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    principle2_maybe_reset_outfile(outfile)

    saved_scenarios = set()

    if os.path.exists(outfile) and not IGNORE_EXISTING_OUTPUTS:
        with open(outfile) as f:
            lines = f.read().splitlines()

        for line in lines:
            scenario = json.loads(line)
            saved_scenarios.add((scenario['n'], scenario['simulation'], scenario['temperature']))

    expected_scenarios = {
        (n, i, 'default' if temperature is None else temperature)
        for n in range(n_min, n_max + 1, n_step)
        for i in range(num_simulations)
        for temperature in temperatures
    }

    if not IGNORE_EXISTING_OUTPUTS and expected_scenarios.issubset(saved_scenarios):
        print(f'All simulations already completed for {outfile}. Skipping inference.')
        return

    print(f'Loaded {len(saved_scenarios)} completed simulations from {outfile}')

    f = open(outfile, 'a+')

    for n in range(n_min, n_max + 1, n_step):
        for i in range(num_simulations):
            for temperature in temperatures:
                temperature_label = 'default' if temperature is None else temperature
                if (n, i, temperature_label) in saved_scenarios:
                    print(f'Skipping simulation for n={n}, i={i}, temperature={temperature_label}')
                    continue
                print(f'Running simulation for n={n}, i={i}, temperature={temperature_label}')

                experiment_record = {
                    'model': model,
                    'outfile': outfile,
                    'environment': environment,
                    'role': role,
                    'num_common_neighbors': num_common_neighbors,
                    'cot': cot,
                    'cot_config': cot_config,
                    'er': er,
                    'metadata': metadata or {},
                }
                state = principle2_load_growth_checkpoint(experiment_record, n, i, temperature, temperature_label)
                if state is None:
                    state = principle2_initialize_growth_state(n, temperature, experiment_record)
                    state['simulation'] = i

                while state['t_idx'] < len(state['node_order']):
                    t = state['node_order'][state['t_idx']]
                    print(f'Adding edge for node {t}, simulation={state["simulation"]}')
                    result = principle2_select_neighbor(
                        state['G'],
                        t,
                        state['temperature'],
                        state['model'],
                        state['environment'],
                        state['role'],
                        state['num_common_neighbors'],
                        state['cot'],
                        state.get('cot_config'),
                    )
                    principle2_advance_growth_state(state, result)

                principle2_write_growth_state(state)

                if method != 'llm':
                    break

    f.close()

def principle2_draw_graph(G, ax, G0=None, use_netgraph=True, er=False):
    if er:
        group_1 = [n for n in G.nodes()]
        group_2 = []
    else:
        group_1 = [n for n in G.nodes() if n < len(G.nodes()) // 2]
        group_2 = [n for n in G.nodes() if n >= len(G.nodes()) // 2]

    if not G0:
        G0_edges = set()
    else:
        G0_edges = set(G0.edges())
    G_edges = set(G.edges()) - G0_edges

    G_group_1 = (set(nx.subgraph(G, group_1).edges()) & G_edges) - G0_edges
    G0_group_1 = (set(nx.subgraph(G0, group_1).edges()))
    G_group_2 = (set(nx.subgraph(G, group_2).edges()) & G_edges) - G0_edges
    G0_group_2 = set(nx.subgraph(G0, group_2).edges())
    G_between = G_edges - set(nx.subgraph(G, group_1).edges()) - set(nx.subgraph(G, group_2).edges()) - G0_edges
    G0_between = G0_edges - set(nx.subgraph(G0, group_1).edges()) - set(nx.subgraph(G0, group_2).edges())
    pos = nx.spring_layout(G)

    if not use_netgraph:

        node_color = ['#c0392b' if n in group_1 else '#2980b9' for n in G.nodes()]

        if not G0:
            nx.draw(G, pos, ax=ax, node_size=10, width=0.5, node_color=node_color, alpha=0.7, edge_color='#34495e')
        else:

            nx.draw_networkx_edges(G, pos, edgelist=G0_edges, width=0.5, alpha=0.5, edge_color='#34495e', ax=ax)
            nx.draw_networkx_edges(G, pos, edgelist=G_between, width=1.0, alpha=1, edge_color='#f1c40f', ax=ax)
            nx.draw_networkx_edges(G, pos, edgelist=G_group_1, width=2, alpha=1, edge_color='#c0392b', ax=ax)
            nx.draw_networkx_edges(G, pos, edgelist=G0_group_1, width=1.0, alpha=0.5, edge_color='#e74c3c', ax=ax)

            nx.draw_networkx_edges(G, pos, edgelist=G_group_2, width=2, alpha=1, edge_color='#2980b9', ax=ax)
            nx.draw_networkx_edges(G, pos, edgelist=G0_group_2, width=1.0, alpha=0.5, edge_color='#3498db', ax=ax)

            nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), node_size=10, node_color=node_color, alpha=0.7, ax=ax)
    else:
        if er:
            node2community = {i: 0 for i in G.nodes()}
        else:
            node2community = {i: 0 if i < len(G.nodes()) // 2 else 1 for i in G.nodes()}

        node_color = {i : '#c0392b' if node2community[i] == 0 else  '#2980b9' for i in G.nodes()}

        edge_color = {}
        edge_width = {}
        edge_alpha = {}
        for (u, v) in G.edges():
            if (u, v) in G_group_1:
                edge_color[u, v] = '#c0392b'
            elif (u, v) in G_group_2:
                edge_color[u, v] = '#2980b9'
            elif (u, v) in G0_group_1:
                edge_color[u, v] = '#e74c3c'
            elif (u, v) in G0_group_2:
                edge_color[u, v] = '#3498db'
            elif (u, v) in G_between:
                edge_color[u, v] = '#f1c40f'
            else:
                edge_color[u, v] = '#bdc3c7'

            if (u, v) in G_group_1 or (u, v) in G_group_2 or (u, v) in G_between:
                edge_width[u, v] = 2
                edge_alpha[u, v] = 1
            else:
                edge_width[u, v] = 1
                edge_alpha[u, v] = 0.5

        # netgraph.Graph(G, node_layout='community', node_color=node_color, node_layout_kwargs=dict(node_to_community=node2community), node_size=2.5, edge_color=edge_color, edge_layout='bundled', edge_layout_kwargs=dict(k=2000), ax=ax)
        netgraph.Graph(G, node_layout=pos, node_color=node_color, node_layout_kwargs=dict(node_to_community=node2community), node_size=2.5, edge_color=edge_color, edge_width=edge_width, edge_alpha=edge_alpha, ax=ax)


    ax.set_axis_off()

def principle2_prob_edge_within_community(G, G0, er=False):
    if er:
        group_1 = [n for n in G.nodes()]
        group_2 = []
    else:
        group_1 = [n for n in G.nodes() if n < len(G.nodes()) // 2]
        group_2 = [n for n in G.nodes() if n >= len(G.nodes()) // 2]

    G0_edges = set(G0.edges())

    G_edges = set(G.edges()) - G0_edges

    G_between = G_edges - set(nx.subgraph(G, group_1).edges()) - set(nx.subgraph(G, group_2).edges()) - G0_edges

    try:
        return 1 - len(G_between) / (1e-1 + len(G_edges))
    except:
        return 0

def principle2_analyze_experiments(filename, num_common_neighbors=True, er=False, sfx=''):
    os.makedirs('figures/principle_2', exist_ok=True)

    suffix = os.path.split(os.path.splitext(filename)[0])[-1]

    with open(filename) as f:
        lines = f.read().splitlines()

    data = []

    for line in lines:
        data.append(json.loads(line))

    transitivities = collections.defaultdict(list)
    algebraic_connectivities = collections.defaultdict(list)
    probabilities_of_edge_within_community = collections.defaultdict(list)
    # partition_qualitys = collections.defaultdict(list)

    final_graphs = collections.defaultdict(list)

    for d in data:
        Gs = []
        for i, graph in enumerate(d['graphs']):
            G = nx.Graph()

            for k, v in graph.items():
                k = int(k)
                G.add_node(k)
                for n in v:
                    G.add_edge(k, n)

            G.remove_edges_from(nx.selfloop_edges(G))

            if i > 0:
                print('new edge', set(G.edges()) - set(Gs[0].edges()))

            Gs.append(G)

        # fig, ax = plt.subplots(1, 4, figsize=(20, 5))

        # fig.suptitle(f'Temperature = {d["temperature"]}')

        # for i, t in enumerate([0, len(Gs) // 2, len(Gs) - 1]):
        #     G = Gs[t]
        #     ax[i].set_title(f'$t = {t}$')
        #     principle2_draw_graph(G, ax=ax[i], G0=Gs[0])

            # print(d['reasons'])

        if er:
            group_1 = [n for n in G.nodes()]
            group_2 = []
        else:
            group_1 = [n for n in G.nodes() if n < len(G.nodes()) // 2]
            group_2 = [n for n in G.nodes() if n >= len(G.nodes()) // 2]


        final_graphs[d['n'], d['temperature']].append((Gs[-1], Gs[0]))

        initial_transitivity = nx.transitivity(Gs[0])

        transitivity = [nx.transitivity(G) - initial_transitivity for G in Gs]

        algebraic_connectivity = [nx.algebraic_connectivity(G) for G in Gs]

        probability_of_edge_within_community = [principle2_prob_edge_within_community(G, Gs[0], er=er) for G in Gs[1:]]

        # partition_quality = [nx.community.partition_quality(G, communities)[0] for G in Gs]

        # ax[-1].set_title('Metrics')
        # ax[-1].plot(transitivity, label='Marginal Transitivity', color='#c0392b')

        # ax_y = ax[-1].twinx()

        # ax_y.plot(algebraic_connectivity, label='Algebraic Connectivity', color='#2980b9')
        # ax[-1].set_xlabel('t')
        # ax[-1].set_ylabel('Transitivity', color='#c0392b')
        # ax_y.set_ylabel('Algebraic Connectivity', color='#2980b9')

        transitivities[d['n'], d['temperature']].append(transitivity)
        algebraic_connectivities[d['n'], d['temperature']].append(algebraic_connectivity)
        probabilities_of_edge_within_community[d['n'], d['temperature']].append(probability_of_edge_within_community)
        # partition_qualitys[d['n'], d['temperature']].append(partition_quality)

        # fig.tight_layout()
        # fig.savefig(f'figures/principle_2/{suffix}_{d["n"]}_{d["simulation"]}_{d["temperature"]}{"_neighbors" if not num_common_neighbors else ""}.pdf')

    palette = ['#e67e22', '#f1c40f', '#7f8c8d', '#c0392b', '#2980b9', '#34495e']


    # fig, ax = plt.subplots(4, len(transitivities), figsize=(5 * len(transitivities), 10), squeeze=False, sharey='row')

    if er:
        fig_final, ax_final = plt.subplots(1, len(final_graphs) + 1, figsize=(5 * (1 + len(final_graphs)), 5), squeeze=False)

        ax_final[0, -1].spines[['right', 'top']].set_visible(False)
    else:

        fig_final, ax_final = plt.subplots(1, len(final_graphs) + 2, figsize=(5 * (2 + len(final_graphs)), 5), squeeze=False, gridspec_kw={'width_ratios': [1] * len(final_graphs) + [0.5, 0.5]})

        ax_final[0, -1].spines[['right', 'top']].set_visible(False)
        ax_final[0, -2].spines[['right', 'top']].set_visible(False)



    for i, (k, v) in enumerate(sorted(final_graphs.items())):
        G, G0 = v[0]
        principle2_draw_graph(G, ax=ax_final[0, i], G0=G0, er=er)

        ax_final[0, i].set_title(f'Temperature = {k[1]}')


    if er:
        ax_final[0, -1].set_ylabel('Marginal Transitivity')
        ax_final[0, -1].set_xticks([])

    else:
        ax_final[0, -2].set_ylabel('Marginal Transitivity')
        ax_final[0, -2].set_xticks([])


        ax_final[0, -1].set_ylabel('Pr. Edge w Community')
        ax_final[0, -1].set_xticks([])

    for i, (k, c) in enumerate(zip(sorted(transitivities.keys()), palette)):
        v = transitivities[k]
        v = np.array(v)

        mean = v.mean(axis=0)
        std = v.std(axis=0)

        ci = 1.96 * std / np.sqrt(len(v))

        if er:
            ax_final[0, -1].bar(i, mean[-1], color=palette[i], alpha=0.6, label='Temp = ' + str(k[1]))
            ax_final[0, -1].errorbar(i, mean[-1], yerr=ci[-1], color='black', alpha=1)
        else:
            ax_final[0, -2].bar(i, mean[-1], color=palette[i], alpha=0.6, label='Temp = ' + str(k[1]))
            ax_final[0, -2].errorbar(i, mean[-1], yerr=ci[-1], color='black', alpha=1)


    if not er:
        for i, (k, c) in enumerate(zip(sorted(probabilities_of_edge_within_community.keys()), palette)):
            v = probabilities_of_edge_within_community[k]
            v = np.array(v)

            mean = v.mean(axis=0)
            std = v.std(axis=0)

            ci = 1.96 * std / np.sqrt(len(v))


            ax_final[0, -1].bar(i, mean[-1], color=palette[i], alpha=0.5, label='Temp = ' + str(k[1]))
            ax_final[0, -1].errorbar(i, mean[-1], yerr=ci[-1], color='black', alpha=0.5)



    # Null models
    transitivities_null = { 'random' : collections.defaultdict(list), 'winner' : collections.defaultdict(list), 'sbm' : collections.defaultdict(list) }
    algebraic_connectivities_null = { 'random' : collections.defaultdict(list), 'winner' : collections.defaultdict(list), 'sbm' : collections.defaultdict(list) }
    probabilities_of_edge_within_community_null = { 'random' : collections.defaultdict(list), 'winner' : collections.defaultdict(list), 'sbm' : collections.defaultdict(list) }

    for d in data:
        for method in ['random']:
            null_temperature = None if d['temperature'] == 'default' else d['temperature']
            Gs, _ = principle2_network_growth(d['n'], null_temperature, method=method, model='gpt-5-mini', environment=None, role='friends', num_common_neighbors=num_common_neighbors, cot=False, er=er)

            initial_transitivity = nx.transitivity(Gs[0])

            transitivity = [nx.transitivity(G) - initial_transitivity for G in Gs]

            transitivities_null[method][d['n'], d['temperature']].append(transitivity)


            algebraic_connectivity = [nx.algebraic_connectivity(G) for G in Gs]

            algebraic_connectivities_null[method][d['n'], d['temperature']].append(algebraic_connectivity)

            if er:
                group_1 = [n for n in Gs[0].nodes()]
                group_2 = []
            else:
                group_1 = [n for n in G.nodes() if n < len(G.nodes()) // 2]
                group_2 = [n for n in G.nodes() if n >= len(G.nodes()) // 2]

            communities = [group_1, group_2]

            probability_of_edge_within_community = [principle2_prob_edge_within_community(G, Gs[0], er=er) for G in Gs[1:]]

            probabilities_of_edge_within_community_null[method][d['n'], d['temperature']].append(probability_of_edge_within_community)



    for j, method in enumerate(['random']):
        for i, (k, v) in enumerate(transitivities_null[method].items()):
            v = np.array(v)

            mean = v.mean(axis=0)
            std = v.std(axis=0)

            ci = 1.96 * std / np.sqrt(len(v))

            if i == 0:

                if method == 'random':
                    transitivity_temp = mean.mean()
                    print('Transitivity null: ', transitivity_temp)


            if i == 0:
                if er:
                    ax_final[0, -1].bar(j + 3, mean[-1], color=palette[j+3], alpha=0.6, label=method.capitalize())
                    ax_final[0, -1].errorbar(j + 3, mean[-1], yerr=ci[-1], color='black', alpha=1)
                else:
                    ax_final[0, -2].bar(j + 3, mean[-1], color=palette[j+3], alpha=0.6, label=method.capitalize())
                    ax_final[0, -2].errorbar(j + 3, mean[-1], yerr=ci[-1], color='black', alpha=1)


            print('Transitivity T-test', k, method, scipy.stats.ttest_ind([x[-1] for x in transitivities[k]], [x[-1] for x in transitivities_null[method][k]], equal_var=False))

        for i, (k, v) in enumerate(algebraic_connectivities_null[method].items()):
            v = np.array(v)

            mean = v.mean(axis=0)
            std = v.std(axis=0)

            ci = 1.96 * std / np.sqrt(len(v))

            # print('Algebraic Connectivity T-test', k, method, scipy.stats.ttest_ind([x[-1] for x in algebraic_connectivities[k]], [x[-1] for x in algebraic_connectivities_null[method][k]], equal_var=False))

        for i, (k, v) in enumerate(probabilities_of_edge_within_community_null[method].items()):


            v = np.array(v)

            mean = v.mean(axis=0)
            std = v.std(axis=0)

            ci = 1.96 * std / np.sqrt(len(v))

            # ax[1, i].plot(mean, color='#c0392b' if method == 'random' else '#34495e', linestyle='--' if method == 'random' else ':', label=method.capitalize())
            # ax[1, i].fill_between(np.arange(len(mean)), mean - ci, mean + ci, alpha=0.2, color='#c0392b' if method == 'random' else '#34495e')
            if i == 0:
                # ax_combined[0, 2].plot(mean, color='#c0392b' if method == 'random' else '#34495e', linestyle='--' if method == 'random' else ':')
                # ax_combined[0, 2].fill_between(np.arange(len(mean)), mean - ci, mean + ci, alpha=0.2, color='#c0392b' if method == 'random' else '#34495e', hatch='||')

                if method == 'random':
                    probability_temp = mean.mean()
                    print('Probability null: ', probability_temp)

            if i == 0:
                if not er:
                    ax_final[0, -1].bar(j + 3, mean[-1], color=palette[j+3], alpha=0.6, label=method.capitalize())
                    ax_final[0, -1].errorbar(j + 3, mean[-1], yerr=ci[-1], color='black', alpha=1)


            print('Probability of edge within community T-test', k, method, scipy.stats.ttest_ind([x[-1] for x in probabilities_of_edge_within_community[k]], [x[-1] for x in probabilities_of_edge_within_community_null[method][k]], equal_var=False))

        # for i, (k, v) in enumerate(probabilities_of_edge_within_community_null[method].items()):
        #     v = np.array(v)

        #     mean = v.mean(axis=0)
        #     std = v.std(axis=0)

        #     ci = 1.96 * std / np.sqrt(len(v))

        #     ax[1, i].plot(mean, color='#c0392b' if method == 'random' else '#34495e', linestyle='--' if method == 'random' else ':', label=method.capitalize())
        #     ax[1, i].fill_between(np.arange(len(mean)), mean - ci, mean + ci, alpha=0.2, color='#c0392b' if method == 'random' else '#34495e')
        #     if i == 0:
        #         ax_combined[0, 3].plot(mean, color='#c0392b' if method == 'random' else '#34495e', linestyle='--' if method == 'random' else ':')
        #         ax_combined[0, 3].fill_between(np.arange(len(mean)), mean - ci, mean + ci, alpha=0.2, color='#c0392b' if method == 'random' else '#34495e', hatch='||')

        #     if i == 0:
        #         ax_final[0, -2].bar(j + 3, mean[-1], color=palette[j+3], alpha=0.6, label=method.capitalize())
        #         ax_final[0, -2].errorbar(j + 3, mean[-1], yerr=ci[-1], color='black', alpha=1)

        #     print('Partition Quality T-test', k, method, scipy.stats.ttest_ind([x[-1] for x in partition_qualitys[k]], [x[-1] for x in partition_qualitys_null[method][k]], equal_var=False))

    ax_final[0, -1].legend(bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)

    # ax[0, 0].legend(loc='upper left')
    # ax[1, 0].legend(loc='upper left')

    # ax_combined[0, 0].legend(loc='upper left')
    # ax_combined[0, 1].legend(loc='upper left')

    # fig_combined.tight_layout()

    # fig.tight_layout()

    # fig.savefig(f'figures/principle_2/{suffix}_overall{"_neighbors" if not num_common_neighbors else ""}.pdf')

    # fig_combined.savefig(f'figures/principle_2/{suffix}_overall_combined{"_neighbors" if not num_common_neighbors else ""}.pdf')


    fig_final.tight_layout()

    fig_final.savefig(f'figures/principle_2/{suffix}_final{"_neighbors" if not num_common_neighbors else ""}{sfx}.pdf', bbox_inches='tight')

    return transitivity_temp, probability_temp

def principle2_get_table(filenames, sfx='', environments=True, transitivity_null=-1, probability_null=-1, er=False, baseline_model_name='Qwen/Qwen3.5-4B'):
    os.makedirs('figures', exist_ok=True)
    os.makedirs('tables', exist_ok=True)

    records = []

    num_graphs = 0

    for filename in filenames:
        print(filename)
        suffix = os.path.split(os.path.splitext(filename)[0])[-1]
        suffix = suffix.split('+')

        if len(suffix) == 3:
            model = suffix[-2]
            environment = suffix[-1]
        elif len(suffix) == 2:
            model = suffix[-1]
            environment = 'Baseline'
        else:
            model = suffix[-1]
            environment = 'Baseline'

        with open(filename) as f:
            lines = f.read().splitlines()

        data = []

        for line in lines:
            data.append(json.loads(line))

        for d in data:
            if 'model' in d:
                model = str(d['model']).replace('/', '-')
                if d.get('cot') and not model.endswith('_cot'):
                    model = f'{model}_cot'
                environment = d.get('environment', 'Baseline')
                if environment is None:
                    environment = 'Baseline'
                if d.get('cot') and environment != 'Baseline' and not str(environment).endswith('_cot'):
                    environment = f'{environment}_cot'

            Gs = []

            for i, graph in enumerate(d['graphs']):
                G = nx.Graph()

                for k, v in graph.items():
                    k = int(k)
                    G.add_node(k)
                    for n in v:
                        G.add_edge(k, n)

                if i == 0:
                    top_common_neighbors = np.zeros(len(G.nodes()))
                    total = 0
                else:
                    new_edge =  set(G.edges()) - set(Gs[-1].edges())
                    if len(new_edge) == 0:
                        continue

                    new_edge = new_edge.pop()

                    u, v = new_edge

                    common_neighbors_u = [len(set(G.neighbors(u)) & set(G.neighbors(n))) for n in G.nodes() if n != u]

                    # find what position the new edge is in the sorted list of common neighbors
                    try:
                        pos = np.argsort(common_neighbors_u)[::-1].tolist().index(v)
                        top_common_neighbors[pos] += 1
                        total += 1
                    except:
                        pass

                G.remove_edges_from(nx.selfloop_edges(G))

                Gs.append(G)

            top_common_neighbors /= total

            top_common_neighbors = np.cumsum(top_common_neighbors)
            top_common_neighbors = np.insert(top_common_neighbors, 0, 0)


            initial_transitivity = nx.transitivity(Gs[0])
            final_transitivity = nx.transitivity(Gs[-1])
            marginal_transitivity = final_transitivity - initial_transitivity
            final_probability_of_edge_within_community = principle2_prob_edge_within_community(Gs[-1], Gs[0], er=er)

            record = {
                'Model' : model,
                'Environment' : environment,
                'Temperature' : d['temperature'],
                'Marginal Transitivity' : marginal_transitivity,
                # 'Algebraic Connectivity' : final_algebraic_connectivity,
                'Prob. of Edge within Community' : final_probability_of_edge_within_community,
                'Probability of Connecting to Top-$k$' : top_common_neighbors,
                'Top-$k$' : np.arange(0, len(top_common_neighbors)) / len(top_common_neighbors)
            }

            records.append(record)



    df = pd.DataFrame(records)

    rename_models = {
        'gpt-5-nano' : 'GPT-5 Nano',
        'gpt-5-mini' : 'GPT-5 Mini',
        'Qwen-Qwen3.5-4B' : 'Qwen 3.5 4B',
        'Qwen-Qwen3.5-2B' : 'Qwen 3.5 2B',
        'Qwen-Qwen3.5-0.8B' : 'Qwen 3.5 0.8B',
        'gpt-5-nano_cot' : 'GPT-5 Nano (CoT)',
        'gpt-5-mini_cot' : 'GPT-5 Mini (CoT)',
        'Qwen-Qwen3.5-4B_cot' : 'Qwen 3.5 4B (CoT)',
        'Qwen-Qwen3.5-2B_cot' : 'Qwen 3.5 2B (CoT)',
        'Qwen-Qwen3.5-0.8B_cot' : 'Qwen 3.5 0.8B (CoT)'
    }

    rename_env = {
        'school' : 'School',
        'work' : 'Work',
        'community' : 'Community',
        'school_cot' : 'School (CoT)',
        'work_cot' : 'Work (CoT)',
        'community_cot' : 'Community (CoT)',
    }


    ncols = 2 + int(environments)
    if sfx == '_cot':
        condition_label = 'CoT'
    elif sfx == '_er':
        condition_label = 'ER initialization'
    else:
        condition_label = 'Non-CoT SBM'

    df['Model'] = df['Model'].apply(lambda x: rename_models.get(x, x))
    df['Environment'] = df['Environment'].apply(lambda x: rename_env.get(x, x))

    baseline_model_key = baseline_model_name.replace('/', '-')
    baseline_model = rename_models.get(baseline_model_key, baseline_model_key)
    default_temperature = df[df['Temperature'].notna()]['Temperature'].iloc[0]

    df_model = df.query('Environment == "Baseline" and Temperature == @default_temperature')
    df_environment = df.query('Model == @baseline_model')

    df_temperature = df.query('Model == @baseline_model and Environment == "Baseline"')

    if er:
        fig, ax = plt.subplots(1, ncols, figsize=(5 * ncols, 5), squeeze=False)
    else:
        fig, ax = plt.subplots(2, ncols, figsize=(5 * ncols, 10))
    fig.suptitle(condition_label, fontsize=SMALL_SIZE)

    sc_model = sns.barplot(data=df_model, y='Marginal Transitivity', x='Model', ax=ax[0, 0], palette=['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9'])

    sc_temperature = sns.barplot(data=df_temperature, y='Marginal Transitivity', x='Temperature', ax=ax[0, 1], palette=['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9'])

    ax[0, 0].set_ylabel('$D$', fontsize=MEDIUM_SIZE)




    if not er:
        ax[0, 0].set_xticks([])
        ax[0, 1].set_xticks([])

        ax[0, 0].set_xlabel('')
        ax[0, 1].set_xlabel('')

        ax[0, 1].get_yaxis().set_visible(False)

    if er:
        ax[0, 0].set_ylim(0, 0.25)
        ax[0, 1].set_ylim(0, 0.25)
    else:
        ax[0, 0].set_ylim(0, 0.1)
        ax[0, 1].set_ylim(0, 0.1)



    if environments:
        sc_environment = sns.barplot(data=df_environment, y='Marginal Transitivity', x='Environment', ax=ax[0, ncols-1], palette=['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9'])
        ax[0, 2].get_yaxis().set_visible(False)
        ax[0, 2].set_xlabel('')
        if er:
            ax[0, 2].set_ylim(0, 0.25)
        else:
            ax[0, 2].set_ylim(0, 0.1)
            ax[0, 2].set_xticks([])


    if not er:
        sc_model = sns.barplot(data=df_model, y='Prob. of Edge within Community', x='Model', ax=ax[1, 0], palette=['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9'])
        sc_temperature = sns.barplot(data=df_temperature, y='Prob. of Edge within Community', x='Temperature', ax=ax[1, 1], palette=['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9'])

    if transitivity_null != -1:
        ax[0, 0].axhline(y=transitivity_null, color='black', linestyle='--')
        ax[0, 1].axhline(y=transitivity_null, color='black', linestyle='--')
        ax[0, 2].axhline(y=transitivity_null, color='black', linestyle='--')

    if probability_null != -1 and not er:
        ax[1, 0].axhline(y=probability_null, color='black', linestyle='--')
        ax[1, 1].axhline(y=probability_null, color='black', linestyle='--')
        ax[1, 2].axhline(y=probability_null, color='black', linestyle='--')

    sc_model.set_xticklabels(sc_model.get_xticklabels(), rotation=90)
    sc_temperature.set_xticklabels(sc_temperature.get_xticklabels(), rotation=90)

    if er and environments:
        sc_environment.set_xticklabels(sc_environment.get_xticklabels(), rotation=90)

    ax[0, 0].set_title('Model')
    ax[0, 1].set_title('Temperature')

    if not er:
        ax[1, 0].set_ylabel('$\\hat p$', fontsize=MEDIUM_SIZE)
        ax[1, 1].set_ylabel('')

        ax[1, 0].set_xlabel('')
        ax[1, 1].set_xlabel('')


        ax[1, 1].get_yaxis().set_visible(False)

        ax[1, 0].set_ylim(0, 1)
        ax[1, 1].set_ylim(0, 1)

        ax[1, 0].xaxis.label.set_size(MEDIUM_SIZE)
        ax[1, 1].xaxis.label.set_size(MEDIUM_SIZE)

    if environments and not er:
        sc_environment = sns.barplot(data=df_environment, y='Prob. of Edge within Community', x='Environment', ax=ax[1, 2], palette=['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9'])
        sc_environment.set_xticklabels(sc_environment.get_xticklabels(), rotation=90)
        ax[1, 2].set_ylim(0, 1)
        ax[1, 2].get_yaxis().set_visible(False)
        ax[1, 2].set_xlabel('')
        ax[1, 2].set_ylabel('')
        ax[1, 2].spines[['right', 'top']].set_visible(False)
        ax[1, 2].xaxis.label.set_size(MEDIUM_SIZE)


    ax[0, 0].spines[['right', 'top']].set_visible(False)
    ax[0, 1].spines[['right', 'top']].set_visible(False)
    ax[0, 2].spines[['right', 'top']].set_visible(False)
    ax[0, 2].set_title('Environment')


    ax[0, 1].set_ylabel('')
    ax[0, 2].set_ylabel('')
    ax[0, 0].set_xlabel('')
    ax[0, 1].set_xlabel('')
    ax[0, 2].set_xlabel('')


    if not er:
        ax[1, 0].spines[['right', 'top']].set_visible(False)
        ax[1, 1].spines[['right', 'top']].set_visible(False)

    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i, j].tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)
            ax[i, j].yaxis.label.set_size(MEDIUM_SIZE)
            ax[i, j].xaxis.label.set_size(MEDIUM_SIZE)
            ax[i, j].title.set_size(MEDIUM_SIZE)

    fig.savefig(f'figures/triadic_closure{sfx}.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    palette=['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9']

    fig.suptitle(f'{condition_label}: Probability of Connecting to Top-$k$ Common Neighbors', fontsize=SMALL_SIZE)

    breakpoints_arr = [('top', np.array([0.1, 0.2, 0.3, 0.4, 0.5])), ('all', np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]))]

    for label, breakpoints in breakpoints_arr:
        breakpoint_max = np.max(breakpoints)
        breakpoint_min = np.min(breakpoints)

        for i in range(len(ax)):
            ax[i].plot([0, 100 * breakpoint_max], [0, breakpoint_max], color='black', linestyle='--')
            ax[i].set_xlim(100 * breakpoint_min, 100 * breakpoint_max)

        for i, model in enumerate(df_model['Model'].unique()):
            temp = df_model[df_model['Model'] == model]

            n = len(temp['Top-$k$'].values[0])

            indices = np.array([int(x * n) for x in breakpoints])

            color = palette[i]
            linewidth = 1

            if model == baseline_model and df_model['Environment'].values[i] == 'Baseline' and df_model['Temperature'].values[i] == default_temperature:
                color = '#34495e'
                linewidth = 3

            ax[0].plot(100 * temp['Top-$k$'].values[0][indices], temp['Probability of Connecting to Top-$k$'].values.mean(0)[indices], label=model, color=color, linewidth=linewidth, marker='x')



        ax[0].set_title('Model')
        ax[0].set_xlabel('Top-$k$ (%)')
        # ax[0].set_xscale('log')
        # ax[0].set_yscale('log')
        ax[0].set_ylabel('')

        ax[0].legend(fontsize=0.7*SMALL_SIZE, loc='upper left')

        for i, temperature in enumerate(df_temperature['Temperature'].unique()):
            temp = df_temperature[df_temperature['Temperature'] == temperature]

            n = len(temp['Top-$k$'].values[0])

            indices = np.array([int(x * n) for x in breakpoints])

            color = palette[i]
            linewidth = 1

            if temperature == default_temperature and df_temperature['Model'].values[i] == baseline_model and df_temperature['Environment'].values[i] == 'Baseline':
                color = '#34495e'
                linewidth = 3


            ax[1].plot(100 * temp['Top-$k$'].values[0][indices], temp['Probability of Connecting to Top-$k$'].values.mean(0)[indices], label=f'{temperature}', color=color, linewidth=linewidth, marker='x')


        ax[1].set_title('Temperature')
        ax[1].set_xlabel('Top-$k$ (%)')
        # ax[1].set_xscale('log')
        # ax[1].set_yscale('log')
        ax[1].set_ylabel('')

        ax[1].legend(fontsize=0.7*SMALL_SIZE, loc='upper left')

        if environments:

            for i, environment in enumerate(df_environment['Environment'].unique()):
                temp = df_environment[df_environment['Environment'] == environment]

                n = len(temp['Top-$k$'].values[0])

                indices = np.array([int(x * n) for x in breakpoints])

                color = palette[i]
                linewidth = 1

                if environment == 'Baseline' and df_environment['Model'].values[i] == baseline_model and df_environment['Temperature'].values[i] == default_temperature:
                    color = '#34495e'
                    linewidth = 3
                    ax[2].plot(100 * temp['Top-$k$'].values[0][indices], df_model[df_model['Model'] == baseline_model]['Probability of Connecting to Top-$k$'].values.mean(0)[indices], label=baseline_model, color=color, linewidth=linewidth, marker='x')

                else:
                    ax[2].plot(100 * temp['Top-$k$'].values[0][indices], temp['Probability of Connecting to Top-$k$'].values.mean(0)[indices], label=environment, color=color, linewidth=linewidth, marker='x')


            ax[2].set_title('Environment')
            ax[2].set_xlabel('Top-$k$ (%)')

            ax[2].set_ylabel('')
            ax[2].legend(fontsize=0.7*SMALL_SIZE, loc='upper left')
            ax[2].set_ylim(0, 1)
            ax[2].get_yaxis().set_visible(False)
            ax[2].spines[['right', 'top']].set_visible(False)


        ax[0].set_ylim(0, 1)
        ax[1].set_ylim(0, 1)

        # hide y axis numbers
        ax[1].get_yaxis().set_visible(False)

        ax[0].spines[['right', 'top']].set_visible(False)
        ax[1].spines[['right', 'top']].set_visible(False)

        fig.tight_layout()


        fig.savefig(f'figures/top_kcommon{label}_{sfx}.pdf', bbox_inches='tight')



def principle2_experiment_outfile(experiment, output_dir):
    return str(os.path.join(os.fspath(output_dir), f"principle_2_{experiment['name']}.jsonl"))


def principle2_build_experiment_record(experiment, output_dir, default_temperatures):
    environment_role = experiment.get('environment')
    if environment_role is None:
        environment = None
        role = 'friends'
    else:
        environment, role = environment_role

    model = experiment['model']
    num_common_neighbors = experiment.get('num_common_neighbors', False)
    cot = experiment.get('COT', False)
    er = experiment.get('er', False)
    return {
        'experiment': experiment,
        'name': experiment['name'],
        'model': model,
        'outfile': experiment.get('outfile', principle2_experiment_outfile(experiment, output_dir)),
        'parameters': experiment['parameters'],
        'temperatures': experiment.get('temperatures', default_temperatures),
        'environment': environment,
        'role': role,
        'method': experiment.get('method', 'llm'),
        'num_common_neighbors': num_common_neighbors,
        'cot': cot,
        'cot_config': experiment.get('cot_config'),
        'er': er,
        'summary_group': experiment.get('summary_group', 'sbm'),
        'metadata': {
            'experiment_name': experiment['name'],
            'summary_group': experiment.get('summary_group', 'sbm'),
            'model': model,
            'environment': environment if environment is not None else 'Baseline',
            'role': role,
            'num_common_neighbors': num_common_neighbors,
            'cot': cot,
            'er': er,
        },
    }


def principle2_build_cot_calibration_requests(experiment, output_dir, default_temperatures, sample_size=20, seed=0):
    record = principle2_build_experiment_record(experiment, output_dir, default_temperatures)
    n = record['parameters']['n_max']
    temperature = record['temperatures'][0]
    state = principle2_initialize_growth_state(n, temperature, record)
    rng = random.Random(seed)
    nodes = list(state['node_order'])
    rng.shuffle(nodes)
    nodes = nodes[:sample_size]

    requests = []
    for node in nodes:
        request = principle2_build_neighbor_request(
            state['G'],
            node,
            state['environment'],
            state['role'],
            state['num_common_neighbors'],
            True,
            state['model'],
            state.get('cot_config'),
        )
        requests.append((node, request))
    return record, requests


def principle2_run_cot_budget_calibration(
    experiments,
    output_dir,
    default_temperatures,
    default_cot_config,
    run_experiments=True,
    calibrate=True,
    calibration_sample_size=20,
    calibration_max_new_tokens=65536,
    calibration_percentile=0.90,
    calibration_margin=1.5,
    retry_token_buckets=(8192, 16384, 32768, 65536),
    calibration_seed=0,
    calibration_filename='principle_2_cot_budget_calibration.json',
):
    if not calibrate or not run_experiments:
        print('CoT retry budget calibration skipped; using COT_RETRY_MAX_NEW_TOKENS =', COT_RETRY_MAX_NEW_TOKENS)
        return None

    output_dir = os.fspath(output_dir)
    calibration_file = os.path.join(output_dir, calibration_filename)
    cot_experiments = [
        experiment for experiment in experiments
        if experiment.get('run', True)
        and experiment.get('COT', False)
        and experiment['model'].startswith('Qwen/')
    ]
    if not cot_experiments:
        print('No Qwen CoT experiments found; keeping COT_RETRY_MAX_NEW_TOKENS =', COT_RETRY_MAX_NEW_TOKENS)
        return None

    if os.path.exists(calibration_file) and not IGNORE_EXISTING_OUTPUTS:
        with open(calibration_file) as f:
            summary = json.load(f)
        set_principle2_runtime_options(
            ignore_existing_outputs=IGNORE_EXISTING_OUTPUTS,
            cot_retry_max_new_tokens=int(summary['selected_max_new_tokens']),
            medium_size=MEDIUM_SIZE,
        )
        print('Loaded CoT retry budget calibration:', COT_RETRY_MAX_NEW_TOKENS)
        return summary

    experiment = cot_experiments[0]
    record, request_items = principle2_build_cot_calibration_requests(
        experiment,
        output_dir,
        default_temperatures,
        sample_size=calibration_sample_size,
        seed=calibration_seed,
    )
    prompts = [request['prompt'] for _, request in request_items]
    response_schemas = [request['response_schema'] for _, request in request_items]
    calibration_cot_config = dict(record.get('cot_config') or default_cot_config)
    calibration_cot_config.update({
        'max_new_tokens': calibration_max_new_tokens,
        'qwen_enable_thinking': True,
        'strip_qwen_thinking': False,
    })

    print(
        f'Running {len(prompts)} CoT budget calibration generations for {record["model"]} '
        f'with max_new_tokens={calibration_max_new_tokens}'
    )
    outputs = get_responses(
        prompts,
        record['model'],
        temperature=record['temperatures'][0],
        system_prompt='You are a helpful assistant',
        response_schemas=response_schemas,
        cot=True,
        cot_config=calibration_cot_config,
    )

    parse_successes = []
    for (_, request), output in zip(request_items, outputs):
        try:
            principle2_parse_neighbor_response(output, request)
            parse_successes.append(True)
        except Exception:
            parse_successes.append(False)

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(record['model'], trust_remote_code=True)
    except Exception as e:
        print(f'Could not load tokenizer for calibration token counts; using character estimate: {e}')

    summary = summarize_generation_budget_calibration(
        outputs,
        tokenizer=tokenizer,
        parse_successes=parse_successes,
        percentile=calibration_percentile,
        margin=calibration_margin,
        buckets=retry_token_buckets,
        minimum_budget=min(retry_token_buckets),
    )
    summary.update({
        'model': record['model'],
        'experiment_name': record['name'],
        'sample_size': len(outputs),
        'calibration_max_new_tokens': calibration_max_new_tokens,
        'sampled_nodes': [node for node, _ in request_items],
    })

    set_principle2_runtime_options(
        ignore_existing_outputs=IGNORE_EXISTING_OUTPUTS,
        cot_retry_max_new_tokens=int(summary['selected_max_new_tokens']),
        medium_size=MEDIUM_SIZE,
    )
    with open(calibration_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print('Selected COT_RETRY_MAX_NEW_TOKENS =', COT_RETRY_MAX_NEW_TOKENS)
    print('Calibration summary saved to', calibration_file)
    return summary


def principle2_run_configured_experiments(experiments, output_dir, default_temperatures, run_experiments=True, run_analysis=True):
    supported_models = set(filter_supported_models(sorted({experiment['model'] for experiment in experiments})))
    outfiles_by_group = collections.defaultdict(list)
    analysis_nulls = {}
    experiment_records = []
    experiments_to_analyze = []

    for experiment in experiments:
        if not experiment.get('run', True):
            continue

        model = experiment['model']
        if model not in supported_models:
            print(f'Skipping {experiment["name"]}: {model} is not supported in this environment.')
            continue

        record = principle2_build_experiment_record(experiment, output_dir, default_temperatures)
        experiment_records.append(record)

        if experiment.get('include_in_summary', True):
            outfiles_by_group[record['summary_group']].append(record['outfile'])

        if experiment.get('analyze_detail', False):
            experiments_to_analyze.append(record)

    if run_experiments:
        batch_groups = collections.defaultdict(list)
        for record in experiment_records:
            if record['model'].startswith('Qwen/') and record['method'] == 'llm' and record['experiment'].get('batch', True):
                batch_key = (
                    record['model'],
                    record['cot'],
                    json.dumps(record.get('cot_config'), sort_keys=True, default=str),
                )
                batch_groups[batch_key].append(record)
                continue

            principle2_run_network_formation_experiment(
                outfile=record['outfile'],
                method=record['method'],
                model=record['model'],
                environment=record['environment'],
                role=record['role'],
                num_common_neighbors=record['num_common_neighbors'],
                cot=record['cot'],
                cot_config=record.get('cot_config'),
                temperatures=record['temperatures'],
                er=record['er'],
                metadata=record['metadata'],
                **record['parameters'],
            )

        for records in batch_groups.values():
            principle2_run_network_formation_experiments_batch(records)

    if run_analysis:
        for record in experiments_to_analyze:
            group = record['summary_group']
            analysis_nulls[group] = principle2_analyze_experiments(
                record['outfile'],
                num_common_neighbors=record['num_common_neighbors'],
                er=record['er'],
                sfx=f'_{record["name"]}',
            )

        if outfiles_by_group.get('sbm'):
            transitivity_null, probability_null = analysis_nulls.get('sbm', (-1, -1))
            principle2_get_table(outfiles_by_group['sbm'], transitivity_null=transitivity_null, probability_null=probability_null)
        if outfiles_by_group.get('er'):
            transitivity_null, probability_null = analysis_nulls.get('er', (-1, -1))
            principle2_get_table(outfiles_by_group['er'], transitivity_null=transitivity_null, probability_null=probability_null, er=True, sfx='_er')
        if outfiles_by_group.get('cot'):
            principle2_get_table(outfiles_by_group['cot'], sfx='_cot')

    return {
        'supported_models': supported_models,
        'outfiles_by_group': outfiles_by_group,
        'analysis_nulls': analysis_nulls,
        'experiment_records': experiment_records,
        'experiments_to_analyze': experiments_to_analyze,
    }

# --- End Principle 2 network-formation utilities ---

# --- Principle 1 preferential-attachment utilities ---
PRINCIPLE1_BASELINE_MODEL = 'Qwen/Qwen3.5-0.8B'
PRINCIPLE1_MEDIUM_SIZE = 24


def set_principle1_runtime_options(baseline_model='Qwen/Qwen3.5-0.8B', medium_size=24):
    global PRINCIPLE1_BASELINE_MODEL
    global MEDIUM_SIZE, SMALL_SIZE, BIGGER_SIZE
    PRINCIPLE1_BASELINE_MODEL = baseline_model
    MEDIUM_SIZE = medium_size
    SMALL_SIZE = 0.85 * MEDIUM_SIZE
    BIGGER_SIZE = 1.5 * MEDIUM_SIZE
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)


def principle1_draw_graph(G, ax, G0=None, use_netgraph=True, nodecolor='#d35400'):
    if not G0:
        G0_edges = set()
    else:
        G0_edges = set(G0.edges())
    G_edges = set(G.edges()) - G0_edges
    if not use_netgraph:
        pos = nx.spring_layout(G)

        if not G0:
            nx.draw(G, pos, ax=ax, node_size=10, width=1.5, node_color='#d35400', alpha=0.7, edge_color='#34495e')
        else:


            nx.draw_networkx_edges(G, pos, edgelist=G0_edges, width=1.5, alpha=0.5, edge_color='#34495e', ax=ax)
            nx.draw_networkx_edges(G, pos, edgelist=G_edges, width=1.5, alpha=1, edge_color='#e67e22', ax=ax)

            nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), node_size=10, node_color=nodecolor, alpha=0.7, ax=ax)
    else:
        edge_color = {(u, v) : '#34495e' if (u, v) in G0_edges else '#e67e22'  for (u, v) in G.edges()}

        netgraph.Graph(G, node_layout='spring', node_color=nodecolor, node_size=1.0, edge_color=edge_color, ax=ax)

    ax.set_axis_off()

def principle1_initialize_candidate_state(G, degrees):
    candidates = []
    candidate_idx = {}

    for v in G.nodes():
        if degrees:
            candidate = {'name': v, 'number_of_friends': G.degree(v)}
        else:
            candidate = {'name': v, 'friends': list(G.neighbors(v))}

        candidate_idx[v] = len(candidates)
        candidates.append(candidate)

    return candidates, candidate_idx

def principle1_update_candidate_state(candidates, candidate_idx, new_node, selected_node, degrees):
    if selected_node is None:
        if degrees:
            candidate = {'name': new_node, 'number_of_friends': 0}
        else:
            candidate = {'name': new_node, 'friends': []}
    else:
        if degrees:
            candidates[candidate_idx[selected_node]]['number_of_friends'] += 1
            candidate = {'name': new_node, 'number_of_friends': 1}
        else:
            candidates[candidate_idx[selected_node]]['friends'].append(new_node)
            candidate = {'name': new_node, 'friends': [selected_node]}

    candidate_idx[new_node] = len(candidates)
    candidates.append(candidate)

def principle1_build_prompt_candidates(candidates, hash_and_shuffle):
    if not hash_and_shuffle:
        return candidates, None

    hash2idx = {}
    idx2hash = {}

    for candidate in candidates:
        name = candidate['name']
        h = str(hashlib.sha256(str(name).encode()).hexdigest())
        hash2idx[h] = str(name)
        idx2hash[str(name)] = h

    prompt_candidates = []

    for candidate in candidates:
        if 'number_of_friends' in candidate:
            prompt_candidates.append({'name': idx2hash[str(candidate['name'])], 'number_of_friends': candidate['number_of_friends']})
        else:
            prompt_candidates.append({'name': idx2hash[str(candidate['name'])], 'friends': [idx2hash[str(n)] for n in candidate['friends']]})

    return prompt_candidates, hash2idx

def principle1_network_growth(T, n0, temperature=None, model='gpt-5-mini', environment=None, role='friends', cot=False, cot_config=None, hash_and_shuffle=False, degrees=True, method='llm'):
    G = nx.empty_graph(n0)

    # G = nx.erdos_renyi_graph(n0, 0.5)

    candidates, candidate_idx = principle1_initialize_candidate_state(G, degrees)
    edge_history = []
    results = []

    for t in range(n0, n0 + T):
        print(f'Adding node {t}')
        result = None

        if t > 0:
            if method == 'llm':
                result = principle1_select_neighbor(candidates, candidate_idx, temperature, model=model, environment=environment, role=role, cot=cot, cot_config=cot_config, hash_and_shuffle=hash_and_shuffle)
            elif method == 'ba':
                result = {'name' : random.choice(list(G.nodes()), weights=[G.degree(n) for n in G.nodes()])}

        G.add_node(t)

        selected_node = None
        if t > 0 and result:
            selected_node = result['name']
            G.add_edge(t, selected_node)

        edge_history.append((t, selected_node))
        results.append(result)
        principle1_update_candidate_state(candidates, candidate_idx, t, selected_node, degrees)

    return edge_history, results

def principle1_select_neighbor(candidates, candidate_idx, temperature, model, environment, role, cot, cot_config, hash_and_shuffle):
    prompt_candidates, hash2idx = principle1_build_prompt_candidates(candidates, hash_and_shuffle)
    candidate_names = [candidate['name'] for candidate in prompt_candidates]
    response_schema = build_response_schema(candidate_names)
    use_structured_output = not (model.startswith('Qwen/') and cot)
    allowed_names_json = json.dumps(candidate_names, ensure_ascii=False)

    # if len(prompt_candidates) > 200:
    #     prompt_candidates = random.sample(prompt_candidates, 200)

    if cot:
        output_format = f"""
    {{
        "reason" : reason for selecting the person,
        "name" : name of the person you selected
    }}
        """
    else:
        output_format = f"""
    {{
        "name" : name of the person you selected,
        "reason" : reason for selecting the person
    }}
        """

    preferential_attachment_prompt = f"""
    # Task
    {f'You are in a {environment}.' if environment else ''}Your task is to select a person to be {role} with.

    # Input
    The input is a list of dictionaries.

    The profiles are given below after chevrons:

    <PROFILES>
    {json.dumps(prompt_candidates, separators=(',', ':'))}
    </PROFILES>

    # Output
    The output should be given in JSON format with the following structure

    {output_format}

    # Notes

    * Return exactly one JSON object.
    * Do not explain your reasoning outside the JSON object.
    * Do not write markdown fences.
    * Do not write any text before or after the JSON object.
    * The value of "name" must be exactly one of these values: {allowed_names_json}
    * Do not rename the person.
    * Do not output labels such as "person 0", "Person 0", or "candidate 0".
    * The final answer must be exactly one JSON object and must not contain text after the JSON object.
    """

    request = {
        'prompt': preferential_attachment_prompt,
        'response_schema': response_schema if use_structured_output else None,
        'candidate_names': candidate_names,
        'hash2idx': hash2idx,
        'candidate_idx': candidate_idx,
        'hash_and_shuffle': hash_and_shuffle,
    }

    for i in range(10):
        ans = None
        try:
            ans = get_response(preferential_attachment_prompt, model, temperature=temperature, response_schema=response_schema if use_structured_output else None, cot=cot, cot_config=cot_config)
            result = principle1_parse_neighbor_response(ans, request)
            print('NEW EDGE', result)
            return result
        except Exception as e:
            print_llm_parse_error(e, ans, context=f'principle1_select_neighbor attempt={i + 1}, model={model}')

def principle1_build_neighbor_request(candidates, candidate_idx, environment, role, cot, hash_and_shuffle, model):
    prompt_candidates, hash2idx = principle1_build_prompt_candidates(candidates, hash_and_shuffle)
    candidate_names = [candidate['name'] for candidate in prompt_candidates]
    response_schema = build_response_schema(candidate_names)
    use_structured_output = not (model.startswith('Qwen/') and cot)
    allowed_names_json = json.dumps(candidate_names, ensure_ascii=False)

    if cot:
        output_format = f"""
    {{
        "reason" : reason for selecting the person,
        "name" : name of the person you selected
    }}
        """
    else:
        output_format = f"""
    {{
        "name" : name of the person you selected,
        "reason" : reason for selecting the person
    }}
        """

    prompt = f"""
    # Task
    {f'You are in a {environment}.' if environment else ''}Your task is to select a person to be {role} with.

    # Input
    The input is a list of dictionaries.

    The profiles are given below after chevrons:

    <PROFILES>
    {json.dumps(prompt_candidates, separators=(',', ':'))}
    </PROFILES>

    # Output
    The output should be given in JSON format with the following structure

    {output_format}

    # Notes

    * Return exactly one JSON object.
    * Do not explain your reasoning outside the JSON object.
    * Do not write markdown fences.
    * Do not write any text before or after the JSON object.
    * The value of "name" must be exactly one of these values: {allowed_names_json}
    * Do not rename the person.
    * Do not output labels such as "person 0", "Person 0", or "candidate 0".
    * The final answer must be exactly one JSON object and must not contain text after the JSON object.
    """

    return {
        'prompt': prompt,
        'response_schema': response_schema if use_structured_output else None,
        'candidate_names': candidate_names,
        'hash2idx': hash2idx,
        'candidate_idx': candidate_idx,
        'hash_and_shuffle': hash_and_shuffle,
    }

def principle1_parse_neighbor_response(ans, request):
    candidate_names = request['candidate_names']
    result = first_json_object(ans)
    recovered = False
    if not isinstance(result, dict) or 'name' not in result:
        result = recover_name_from_malformed_response(ans, set(candidate_names))
        recovered = True

    normalized_name = normalize_name(result['name'], set(candidate_names))
    if normalized_name is None:
        raise ValueError(f"Invalid candidate name: {result['name']}")
    result['name'] = normalized_name
    if recovered:
        result.setdefault('reason', 'Recovered from malformed JSON response')
        result['parse_recovered'] = True

    if not request['hash_and_shuffle'] and result['name'] in request['candidate_idx']:
        return result
    if request['hash_and_shuffle'] and result['name'] in request['hash2idx']:
        result['name'] = int(request['hash2idx'][result['name']]) if request['hash2idx'][result['name']].isdigit() else request['hash2idx'][result['name']]
        return result

    raise ValueError(f"Invalid candidate name: {result['name']}")

def principle1_initialize_growth_state(n, n0, temperature, experiment_record):
    G = nx.empty_graph(n0)
    candidates, candidate_idx = principle1_initialize_candidate_state(G, experiment_record['degrees'])
    return {
        **experiment_record,
        'n': n,
        'n0': n0,
        'temperature': temperature,
        'temperature_label': 'default' if temperature is None else temperature,
        'G': G,
        'candidates': candidates,
        'candidate_idx': candidate_idx,
        'edge_history': [],
        'reasons': [],
        't': n0,
        'end_t': n0 + n,
    }

def principle1_advance_growth_state(state, result):
    t = state['t']
    G = state['G']
    G.add_node(t)

    selected_node = None
    if result:
        selected_node = result['name']
        G.add_edge(t, selected_node)

    state['edge_history'].append((t, selected_node))
    state['reasons'].append(result)
    principle1_update_candidate_state(state['candidates'], state['candidate_idx'], t, selected_node, state['degrees'])
    state['t'] += 1

def principle1_write_growth_state(state):
    temp = {
        'n' : state['n'],
        'n0' : state['n0'],
        'temperature' : state['temperature_label'],
        'simulation' : state['simulation'],
        'edge_history' : state['edge_history'],
        'reasons' : state['reasons'],
        'model' : state['model'],
        'environment' : state['environment'] if state['environment'] is not None else 'Baseline',
        'role' : state['role'],
        'degrees_experiment' : state['degrees'],
        'cot' : state['cot'],
    }
    if state.get('metadata'):
        temp.update(state['metadata'])

    with open(state['outfile'], 'a+') as f:
        f.write(json.dumps(temp) + '\n')
        f.flush()

def principle1_pending_growth_states_for_experiment(experiment_record):
    parameters = experiment_record['parameters']
    temperatures = experiment_record['temperatures']
    outfile = experiment_record['outfile']
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    saved_scenarios = set()
    if os.path.exists(outfile):
        with open(outfile) as f:
            lines = f.read().splitlines()
        for line in lines:
            scenario = json.loads(line)
            saved_scenarios.add((scenario['n'], scenario['simulation'], scenario['temperature']))

    print(f'Loaded {len(saved_scenarios)} completed simulations from {outfile}')
    states = []
    n_min = parameters['n_min']
    n_max = parameters['n_max']
    n_step = parameters['n_step']
    num_simulations = parameters['num_simulations']
    n0 = 1
    for n in range(n_min, n_max + 1, n_step):
        for i in range(num_simulations):
            for temperature in temperatures:
                temperature_label = 'default' if temperature is None else temperature
                if (n, i, temperature_label) in saved_scenarios:
                    print(f'Skipping simulation for n={n}, i={i}, temperature={temperature_label}')
                    continue
                print(f'Queueing simulation for n={n}, i={i}, temperature={temperature_label}, outfile={outfile}')
                state = principle1_initialize_growth_state(n, n0, temperature, experiment_record)
                state['simulation'] = i
                states.append(state)
    return states

def principle1_run_network_formation_experiments_batch(experiment_records):
    if not experiment_records:
        return

    model = experiment_records[0]['model']
    cot = experiment_records[0]['cot']
    cot_config = experiment_records[0].get('cot_config')
    active_states = []
    for experiment_record in experiment_records:
        active_states.extend(principle1_pending_growth_states_for_experiment(experiment_record))

    if not active_states:
        print(f'All batched simulations already completed for {model}. Skipping inference.')
        return

    print(f'Running {len(active_states)} batched simulations for {model}, cot={cot}')
    while active_states:
        requests_by_temperature = collections.defaultdict(list)
        for state in active_states:
            print(f'Adding node {state["t"]} for {state["metadata"]["experiment_name"]}, simulation={state["simulation"]}')
            request = principle1_build_neighbor_request(
                state['candidates'],
                state['candidate_idx'],
                state['environment'],
                state['role'],
                state['cot'],
                state['hash_and_shuffle'],
                state['model'],
            )
            requests_by_temperature[state['temperature']].append((state, request))

        results_by_state_id = {}
        for temperature, batch_items in requests_by_temperature.items():
            remaining = list(batch_items)
            for attempt in range(10):
                if not remaining:
                    break

                prompts = [request['prompt'] for _, request in remaining]
                response_schemas = [request['response_schema'] for _, request in remaining]
                answers = get_responses(
                    prompts,
                    model,
                    temperature=temperature,
                    response_schemas=response_schemas,
                    cot=cot,
                    cot_config=cot_config,
                )

                next_remaining = []
                for (state, request), ans in zip(remaining, answers):
                    try:
                        result = principle1_parse_neighbor_response(ans, request)
                        print('NEW EDGE', result)
                        results_by_state_id[id(state)] = result
                    except Exception as e:
                        print_llm_parse_error(
                            e,
                            ans,
                            context=(
                                f'batch attempt={attempt + 1}, '
                                f'experiment={state["metadata"]["experiment_name"]}, '
                                f'simulation={state["simulation"]}, '
                                f'node={state["t"]}, '
                                f'model={state["model"]}, '
                                f'temperature={state["temperature_label"]}'
                            ),
                        )
                        next_remaining.append((state, request))
                remaining = next_remaining

            for state, _ in remaining:
                results_by_state_id[id(state)] = None

        next_active_states = []
        for state in active_states:
            principle1_advance_growth_state(state, results_by_state_id.get(id(state)))
            if state['t'] >= state['end_t']:
                principle1_write_growth_state(state)
            else:
                next_active_states.append(state)
        active_states = next_active_states

def principle1_run_network_formation_experiment(n_min, n_max, n_step, num_simulations, outfile, temperatures=None, environment=None, role='friends', degrees=True, model='gpt-5-mini', cot=False, cot_config=None, hash_and_shuffle=False, metadata=None):

    if temperatures is None:
        temperatures = [None]

    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    saved_scenarios = set()

    if os.path.exists(outfile):
        with open(outfile) as f:
            lines = f.read().splitlines()

        for line in lines:
            scenario = json.loads(line)
            saved_scenarios.add((scenario['n'], scenario['simulation'], scenario['temperature']))

    expected_scenarios = {
        (n, i, 'default' if temperature is None else temperature)
        for n in range(n_min, n_max + 1, n_step)
        for i in range(num_simulations)
        for temperature in temperatures
    }

    if expected_scenarios.issubset(saved_scenarios):
        print(f'All simulations already completed for {outfile}. Skipping inference.')
        return

    print(f'Loaded {len(saved_scenarios)} completed simulations from {outfile}')

    f = open(outfile, 'a+')

    for n in range(n_min, n_max + 1, n_step):
        for i in range(num_simulations):
            for temperature in temperatures:
                temperature_label = 'default' if temperature is None else temperature
                if (n, i, temperature_label) in saved_scenarios:
                    print(f'Skipping simulation for n={n}, i={i}, temperature={temperature_label}')
                    continue
                else:
                    print(f'Running simulation for n={n}, i={i}, temperature={temperature_label}')
                    n0 = 1
                    edge_history, reasons = principle1_network_growth(n, n0, temperature=temperature, degrees=degrees, model=model, environment=environment, role=role, cot=cot, cot_config=cot_config, hash_and_shuffle=hash_and_shuffle)

                    temp = {
                        'n' : n,
                        'n0' : n0,
                        'temperature' : temperature_label,
                        'simulation' : i,
                        'edge_history' : edge_history,
                        'reasons' : reasons,
                        'model' : model,
                        'environment' : environment if environment is not None else 'Baseline',
                        'role' : role,
                        'degrees_experiment' : degrees,
                        'cot' : cot,
                    }
                    if metadata:
                        temp.update(metadata)

                    f.write(json.dumps(temp) + '\n')
                    f.flush()

    f.close()

def principle1_reconstruct_graphs(d):
    if 'graphs' in d:
        Gs = []
        for graph in d['graphs']:
            G = nx.Graph()

            for k, v in graph.items():
                k = int(k)
                G.add_node(k)
                for n in v:
                    G.add_edge(k, n)

            G.remove_nodes_from(list(nx.isolates(G)))
            Gs.append(G)

        return Gs

    G = nx.empty_graph(d['n0'])
    Gs = []

    for t, selected_node in d['edge_history']:
        G.add_node(t)
        if selected_node is not None:
            G.add_edge(t, selected_node)

        H = G.copy()
        H.remove_nodes_from(list(nx.isolates(H)))
        Gs.append(H)

    return Gs

def principle1_analyze_experiments(filename, dgr=True):
    os.makedirs('figures/principle_1', exist_ok=True)

    suffix = os.path.split(os.path.splitext(filename)[0])[-1]

    palette = ['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9']

    with open(filename) as f:
        lines = f.read().splitlines()

    data = []

    for line in lines:
        data.append(json.loads(line))

    degree_freqs = collections.defaultdict(list)
    dergee_freqs_barabasi_albert = collections.defaultdict(list)

    wasserstein_distances = collections.defaultdict(list)
    ks_statistics = collections.defaultdict(list)
    gammas = collections.defaultdict(list)
    gammas_barabasi_albert = collections.defaultdict(list)
    sigmas = collections.defaultdict(list)
    sigmas_barabasi_albert = collections.defaultdict(list)
    ks_powerlaw = collections.defaultdict(list)
    confidence_ks_intervals = collections.defaultdict(list)
    pwl_fits = collections.defaultdict(list)
    pwl_fits_barabasi_albert = collections.defaultdict(list)

    final_graphs = collections.defaultdict(list)

    for d in data:
        Gs = principle1_reconstruct_graphs(d)

        final_graphs[d['n'], d['temperature']].append((Gs[-1].copy(), Gs[0].copy()))

        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        fig_barabasi_albert, ax_barabasi_albert = plt.subplots(1, 2, figsize=(10, 5))

        # fig.suptitle(f'Graph created based on Principle 1 with $n = {d["n"]}$, $n_0 = {d["n0"]}$, temperature = {d["temperature"]}')

        G_barabasi_albert = nx.barabasi_albert_graph(n=d['n'], m=1, seed=1)

        for i, t in enumerate([len(Gs) // 3, 2 * len(Gs) // 3, len(Gs) - 1]):
            G = Gs[t]
            ax[i].set_title(f'$t = {t}$')
            if len(Gs[0]) > 2:
                principle1_draw_graph(G, ax=ax[i], G0=Gs[0])
            else:
                principle1_draw_graph(G, ax=ax[i])

        principle1_draw_graph(G_barabasi_albert, ax=ax_barabasi_albert[0], nodecolor='#3498db')


        degrees = [G.degree(n) for n in G.nodes()]
        degrees_barabasi_albert = [G_barabasi_albert.degree(n) for n in G_barabasi_albert.nodes()]

        powerlaw_fit = pwl.Fit(degrees, discrete=True)

        print(f'Temperature {d["temperature"]}: xmin: {powerlaw_fit.xmin}, alpha: {powerlaw_fit.alpha}, sigma: {powerlaw_fit.sigma}')

        powerlaw_fit_barabasi_albert = pwl.Fit(degrees_barabasi_albert, discrete=True)

        wasserstein_distances[d['n'], d['temperature']].append(stats.wasserstein_distance(degrees, degrees_barabasi_albert))
        gammas[d['n'], d['temperature']].append(powerlaw_fit.alpha)
        sigmas[d['n'], d['temperature']].append(powerlaw_fit.sigma)

        gammas_barabasi_albert[d['n'], d['temperature']].append(powerlaw_fit_barabasi_albert.alpha)
        sigmas_barabasi_albert[d['n'], d['temperature']].append(powerlaw_fit_barabasi_albert.sigma)

        ks_stat_powerlaw = getattr(powerlaw_fit.power_law, 'D', None)
        if ks_stat_powerlaw is None:
            ks_stat_powerlaw = powerlaw_fit.power_law.KS(degrees)
        ks_powerlaw[d['n'], d['temperature']].append(ks_stat_powerlaw)

        ax[-1].set_title('Degree distribution')
        ax[-1].spines[['right', 'top']].set_visible(False)

        ax_barabasi_albert[-1].set_title('Degree distribution')

        powerlaw_fit.plot_ccdf(linewidth=3, ax=ax[-1], color='#e74c3c', label='LLM (Empirical)')
        powerlaw_fit_barabasi_albert.plot_ccdf(linewidth=3, ax=ax[-1], color='#3498db', label='BA (Empirical)')

        powerlaw_fit_barabasi_albert.plot_ccdf(linewidth=3, ax=ax_barabasi_albert[-1], color='#3498db', label='BA (Empirical)')

        powerlaw_fit.power_law.plot_ccdf(ax=ax[-1], color='#e74c3c', linestyle='--', label='LLM (Power law fit)')
        powerlaw_fit_barabasi_albert.power_law.plot_ccdf(ax=ax[-1], color='#3498db', linestyle='--', label='BA (Power law fit)')

        powerlaw_fit_barabasi_albert.power_law.plot_ccdf(ax=ax_barabasi_albert[-1], color='#3498db', linestyle='--', label='BA (Power law fit)')

        print(f'BA powerlaw fit gamma: {powerlaw_fit_barabasi_albert.power_law.alpha:.2f} +- {powerlaw_fit_barabasi_albert.power_law.sigma:.2f}')

        pwl_fits[d['n'], d['temperature']].append(powerlaw_fit)
        pwl_fits_barabasi_albert[d['n'], d['temperature']].append(powerlaw_fit_barabasi_albert)

        print(f'Temperature: {d["temperature"]}, KS Test with BA (empirical): {stats.ks_2samp(degrees, degrees_barabasi_albert)}')
        print()

        # Exports to perform bootstrap hypothesis test in R using the poweRlaw package
        df = pd.DataFrame(degrees)
        df.to_csv(f'degrees{"_neighbors" if not dgr else ""}_{d["n"]}_{d["simulation"]}_{d["temperature"]}.txt', header=None, index=False)

        ax[-1].legend()
        ax[-1].set_xlabel('Degree')
        ax[-1].set_ylabel('CCDF')
        ax[-1].spines[['right', 'top']].set_visible(False)


        ax_barabasi_albert[-1].legend()
        ax_barabasi_albert[-1].set_xlabel('Degree')
        ax_barabasi_albert[-1].set_ylabel('CCDF')

        fig.tight_layout()

        fig.suptitle(f'Temperature = {d["temperature"]}', y=1.05)
        fig_barabasi_albert.suptitle('BA Model', y=1.05)

        fig.savefig(f'figures/principle_1/{suffix}_{d["n"]}_{d["simulation"]}_{d["temperature"]}{"_neighbors" if not dgr else ""}.pdf', bbox_inches='tight')
        fig_barabasi_albert.savefig(f'figures/principle_1/{suffix}_{d["n"]}_{d["simulation"]}_{d["temperature"]}_barabasi_albert{"_neighbors" if not dgr else ""}.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1, 1 + len(final_graphs), figsize=(5 * (1 + len(final_graphs)), 5), squeeze=False, gridspec_kw={'width_ratios': [1] * (1 + len(final_graphs))})
    # fig.suptitle(f'Graphs created based on Principle 1 with $n = {d["n"]}$, $n_0={d["n0"]}$')

    for i, k in enumerate(sorted(final_graphs.keys())):
        G, G0 = final_graphs[k][0]

        if len(G0) > 2:
            principle1_draw_graph(G, ax[0, i], G0=G0, nodecolor=palette[i])
        else:
            principle1_draw_graph(G, ax[0, i], nodecolor=palette[i])
        ax[0, i].set_title(f'Temperature = {k[-1]}')


    for i, k in enumerate(sorted(pwl_fits.keys())):
        powerlaw_fit = pwl_fits[k][0]
        powerlaw_fit.plot_ccdf(linewidth=3, ax=ax[0, -1], color=palette[i], label=str(k[-1]))
        powerlaw_fit.power_law.plot_ccdf(ax=ax[0, -1], color=palette[i], linestyle='--')


    for i, k in enumerate(sorted(pwl_fits.keys())):

        powerlaw_fit_barabasi_albert = pwl_fits_barabasi_albert[k][0]

        if i == 0:
            powerlaw_fit_barabasi_albert.plot_ccdf(linewidth=3, ax=ax[0, -1], color='#7f8c8d', label='BA')
            powerlaw_fit_barabasi_albert.power_law.plot_ccdf(ax=ax[0, -1], color='#7f8c8d', linestyle='--')


    ax[0, -1].legend()
    ax[0, -1].set_xlabel('Degree')
    ax[0, -1].set_ylabel('Complementary CDF')
    ax[0, -1].spines[['right', 'top']].set_visible(False)

    fig.tight_layout()

    fig.savefig(f'figures/principle_1/{suffix}_final_graphs{"_neighbors" if not dgr else ""}.pdf')

def principle1_analyze_experiments_multiple_llms(filenames, sfx=''):
    os.makedirs('figures', exist_ok=True)
    os.makedirs('tables', exist_ok=True)

    records = []

    for filename in filenames:
        suffix = os.path.split(os.path.splitext(filename)[0])[-1]
        suffix = suffix.split('+')

        with open(filename) as f:
            lines = f.read().splitlines()

        data = []

        for line in lines:
            data.append(json.loads(line))

        for d in data:
            Gs = principle1_reconstruct_graphs(d)

            G_barabasi_albert = nx.barabasi_albert_graph(n=d['n'], m=1, seed=1)

            G = Gs[-1]

            degrees = np.array([G.degree(n) for n in G.nodes()])
            degrees_barabasi_albert = np.array([G_barabasi_albert.degree(n) for n in G_barabasi_albert.nodes()])


            powerlaw_fit = pwl.Fit(degrees, discrete=True)

            wasserstein_distance = stats.wasserstein_distance(degrees, degrees_barabasi_albert)
            gamma = powerlaw_fit.alpha
            sigma = powerlaw_fit.sigma

            if 'model' in d:
                model = str(d['model']).replace('/', '-')
                if d.get('cot') and not model.endswith('_cot'):
                    model = f'{model}_cot'
                environment = d.get('environment', 'Baseline')
                if environment is None:
                    environment = 'Baseline'
                if d.get('cot') and environment != 'Baseline' and not str(environment).endswith('_cot'):
                    environment = f'{environment}_cot'
            elif len(suffix) == 3:
                model = suffix[-2]
                environment = suffix[-1]
            elif len(suffix) == 2:
                model = suffix[-1]
                environment = 'Baseline'
            else:
                environment = 'Baseline'

            ks_stats = stats.ks_2samp(degrees, degrees_barabasi_albert)

            if ks_stats[1] < 0.001:
                stars = '***'
            elif ks_stats[1] < 0.01:
                stars = '**'
            elif ks_stats[1] < 0.05:
                stars = '*'
            else:
                stars = 'p = {:.3f}'.format(ks_stats[1])


            # top k plot info

            degrees = np.sort(degrees)[::-1]

            # insert 0 at the beginning
            degrees = np.insert(degrees, 0, 0)
            degrees_cumsum = np.cumsum(degrees)
            probability_of_topk_yaxis = degrees_cumsum / degrees_cumsum[-1]

            probability_of_topk_xaxis = np.arange(len(degrees)) / len(degrees)

            record = {
                'Model' : model,
                'Environment' : environment,
                'Temperature' : d['temperature'],
                '$\\hat \\gamma$' : gamma,
                '$\\sigma$' : sigma,
                'KS Test' : f'{ks_stats[0]:.3f} ({stars})',
                'Probability of Connecting to Top-$k$' : probability_of_topk_yaxis,
                'Top-$k$' : probability_of_topk_xaxis
            }

            records.append(record)

    G_barabasi_albert = nx.barabasi_albert_graph(n=d['n'], m=1, seed=1)
    degrees_barabasi_albert = [G_barabasi_albert.degree(n) for n in G_barabasi_albert.nodes()]
    powerlaw_fit = pwl.Fit(degrees_barabasi_albert, discrete=True)
    gamma = powerlaw_fit.alpha
    sigma = powerlaw_fit.sigma

    record = {
        'Model' : 'Barabasi-Albert',
        'Environment' : 'Baseline',
        'Temperature' : None,
        '$\\hat \\gamma$' : gamma,
        '$\\sigma$' : sigma,
        'KS Test' : None
    }

    records.append(record)

    df = pd.DataFrame.from_records(records)


    df.to_csv('tables/exponents.csv', index=False)
    df.to_latex('tables/exponents.tex', index=False, escape=False, float_format="%.3f")

    rename_models = {
        'gpt-5-nano' : 'GPT-5 Nano',
        'gpt-5-mini' : 'GPT-5 Mini',
        'Qwen-Qwen3.5-4B' : 'Qwen 3.5 4B',
        'Qwen-Qwen3.5-2B' : 'Qwen 3.5 2B',
        'Qwen-Qwen3.5-0.8B' : 'Qwen 3.5 0.8B',
        'gpt-5-nano_cot' : 'GPT-5 Nano',
        'gpt-5-mini_cot' : 'GPT-5 Mini',
        'Qwen-Qwen3.5-4B_cot' : 'Qwen 3.5 4B',
        'Qwen-Qwen3.5-2B_cot' : 'Qwen 3.5 2B',
        'Qwen-Qwen3.5-0.8B_cot' : 'Qwen 3.5 0.8B',
    }

    rename_environment = {
        'school' : 'School',
        'community' : 'Community',
        'work' : 'Work',
        'school_cot' : 'School',
        'community_cot' : 'Community',
        'work_cot' : 'Work'
    }


    df['Model'] = df['Model'].apply(lambda x: rename_models.get(x, x))
    df['Environment'] = df['Environment'].apply(lambda x: rename_environment.get(x, x))


    baseline_model_key = PRINCIPLE1_BASELINE_MODEL.replace('/', '-')
    baseline_model = rename_models.get(baseline_model_key, baseline_model_key)
    default_temperature = df[df['Temperature'].notna()]['Temperature'].iloc[0]

    df_model = df.query('Environment == "Baseline" and Temperature == @default_temperature')
    df_model = df_model[df_model['Model'] != 'Barabasi-Albert']

    df_environment = df.query('Model == @baseline_model and Temperature == @default_temperature')
    df_temperature = df.query('Model == @baseline_model and Environment == "Baseline"')

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    sc_model = sns.barplot(data=df_model, y='$\\hat \\gamma$', x='Model', ax=ax[0], palette=['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9'])
    sc_temperature = sns.barplot(data=df_temperature, y='$\\hat \\gamma$', x='Temperature', ax=ax[1], palette=['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9'])
    sc_environment = sns.barplot(data=df_environment, y='$\\hat \\gamma$', x='Environment', ax=ax[2], palette=['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9'])




    sc_model.set_xticklabels(sc_model.get_xticklabels(), rotation=90)
    sc_temperature.set_xticklabels(sc_temperature.get_xticklabels(), rotation=90)
    sc_environment.set_xticklabels(sc_environment.get_xticklabels(), rotation=90)

    ax[0].errorbar(df_model['Model'], df_model['$\\hat \\gamma$'], yerr=df_model['$\\sigma$'], color='black', linestyle='', capsize=10, alpha=0.5)
    ax[1].errorbar(df_temperature['Temperature'].astype(str), df_temperature['$\\hat \\gamma$'], yerr=df_temperature['$\\sigma$'], color='black', linestyle='', capsize=10, alpha=0.5)
    ax[2].errorbar(df_environment['Environment'], df_environment['$\\hat \\gamma$'], yerr=df_environment['$\\sigma$'], color='black', linestyle='', capsize=10, alpha=0.5)



    gamma_ba = df[df['Model'] == 'Barabasi-Albert']['$\\hat \\gamma$'].values[0]
    sigma_ba = df[df['Model'] == 'Barabasi-Albert']['$\\sigma$'].values[0]

    # draw baraasi-albert line
    ax[0].axhline(y=gamma_ba, color='#c0392b', linestyle='--', label='BA (Sample)', linewidth=3)
    ax[1].axhline(y=gamma_ba, color='#c0392b', linestyle='--', label='BA (Sample)', linewidth=3)
    ax[2].axhline(y=gamma_ba, color='#c0392b', linestyle='--', label='BA (Sample)', linewidth=3)

    ax[0].axhline(y=3, color='#34495e', linestyle=':', label='BA (Theoretical)', linewidth=3)
    ax[1].axhline(y=3, color='#34495e', linestyle=':', label='BA (Theoretical)', linewidth=3)
    ax[2].axhline(y=3, color='#34495e', linestyle=':', label='BA (Theoretical)', linewidth=3)

    ax[0].legend(fontsize=0.7*SMALL_SIZE)

    ax[0].set_ylim(1, 5)
    ax[1].set_ylim(1, 5)
    ax[2].set_ylim(1, 5)

    ax[1].set_ylabel('')
    ax[2].set_ylabel('')

    ax[0].set_xlabel('')
    ax[1].set_xlabel('')
    ax[2].set_xlabel('')

    ax[0].set_title('Model')
    ax[1].set_title('Temperature')
    ax[2].set_title('Environment')


    ax[1].get_yaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)

    ax[0].spines[['right', 'top']].set_visible(False)
    ax[1].spines[['right', 'top']].set_visible(False)
    ax[2].spines[['right', 'top']].set_visible(False)

    # fig.tight_layout()

    fig.savefig(f'figures/exponents{sfx}.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # plot probability of connecting to top-k

    palette=['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9']



    breakpoints_arr = [('top', np.array([0.01, 0.015, 0.02, 0.025])), ('all', np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]))]

    for label, breakpoints in breakpoints_arr:

        breakpoint_max = np.max(breakpoints)
        breakpoint_min = np.min(breakpoints)

        fig.suptitle('Probability of Connecting to Top-$k$ Degrees', fontsize=SMALL_SIZE)

        for i in range(len(ax)):
            ax[i].plot([0, 100 * breakpoint_max], [0, breakpoint_max], color='black', linestyle='--')
            ax[i].set_xlim(100 * breakpoint_min, 100 * breakpoint_max)

        for i, model in enumerate(df_model['Model']):
            n = len(df_model['Top-$k$'].values[i])
            indices = np.array([int(x * n) for x in breakpoints])
            jitter = np.random.uniform(0, 0.1) * np.ones(len(indices))

            color = palette[i]
            linewidth = 1

            if model == baseline_model and df_model['Environment'].values[i] == 'Baseline' and df_model['Temperature'].values[i] == default_temperature:
                color = '#34495e'
                linewidth = 3

            ax[0].plot(100 * df_model['Top-$k$'].values[i][indices] + jitter, df_model['Probability of Connecting to Top-$k$'].values[i][indices] + jitter, label=model, color=color, linewidth=linewidth, marker='x')

        ax[0].set_title('Model')
        ax[0].set_xlabel('Top-$k$ (%)')
        # ax[0].set_xscale('log')
        # ax[0].set_yscale('log')
        ax[0].set_ylabel('')

        ax[0].legend(fontsize=0.7*SMALL_SIZE, ncol=2)

        for i, temperature in enumerate(df_temperature['Temperature']):
            n = len(df_model['Top-$k$'].values[i])
            indices = np.array([int(x * n) for x in breakpoints])

            color = palette[i]

            color = palette[i]
            linewidth = 1

            if temperature == default_temperature and df_temperature['Model'].values[i] == baseline_model and df_temperature['Environment'].values[i] == 'Baseline':
                color = '#34495e'
                linewidth = 3

            ax[1].plot(100 * df_temperature['Top-$k$'].values[i][indices], df_temperature['Probability of Connecting to Top-$k$'].values[i][indices], label=f'{temperature}', color=color, linewidth=linewidth, marker='x')

        ax[1].set_title('Temperature')
        ax[1].set_xlabel('Top-$k$ (%)')
        # ax[1].set_xscale('log')
        # ax[1].set_yscale('log')
        ax[1].set_ylabel('')

        ax[1].legend(fontsize=0.7*SMALL_SIZE)

        for i, environment in enumerate(df_environment['Environment']):
            n = len(df_model['Top-$k$'].values[i])
            indices = np.array([int(x * n) for x in breakpoints])

            color = palette[i]
            linewidth = 1

            if environment == 'Baseline' and df_environment['Model'].values[i] == baseline_model and df_environment['Temperature'].values[i] == default_temperature:
                color = '#34495e'
                linewidth = 3

            ax[2].plot(100 * df_environment['Top-$k$'].values[i][indices], df_environment['Probability of Connecting to Top-$k$'].values[i][indices], label=environment, color=color, linewidth=linewidth, marker='x')

        ax[2].set_title('Environment')
        ax[2].set_xlabel('Top-$k$ (%)')
        # ax[2].set_xscale('log')
        # ax[2].set_yscale('log')
        ax[2].set_ylabel('')

        ax[2].legend(fontsize=0.7*SMALL_SIZE, ncols=2)

        ax[0].set_ylim(0, 1)
        ax[1].set_ylim(0, 1)
        ax[2].set_ylim(0, 1)

        # hide y axis numbers
        ax[1].get_yaxis().set_visible(False)
        ax[2].get_yaxis().set_visible(False)

        ax[0].spines[['right', 'top']].set_visible(False)
        ax[1].spines[['right', 'top']].set_visible(False)
        ax[2].spines[['right', 'top']].set_visible(False)

        fig.tight_layout()

        fig.savefig(f'figures/probabilitytopk_{label}_{sfx}.pdf', bbox_inches='tight')


def principle1_experiment_outfile(experiment, output_dir):
    return str(os.path.join(os.fspath(output_dir), f"principle_1_{experiment['name']}.jsonl"))


def principle1_build_experiment_record(experiment, output_dir, default_temperatures):
    environment_role = experiment.get('environment')
    if environment_role is None:
        environment = None
        role = 'friends'
    else:
        environment, role = environment_role

    model = experiment['model']
    return {
        'experiment': experiment,
        'name': experiment['name'],
        'model': model,
        'outfile': experiment.get('outfile', principle1_experiment_outfile(experiment, output_dir)),
        'parameters': experiment['parameters'],
        'temperatures': experiment.get('temperatures', default_temperatures),
        'environment': environment,
        'role': role,
        'degrees': experiment.get('degrees_experiment', False),
        'cot': experiment.get('COT', False),
        'cot_config': experiment.get('cot_config'),
        'hash_and_shuffle': experiment.get('hash_and_shuffle', False),
        'metadata': {
            'experiment_name': experiment['name'],
            'model': model,
            'environment': environment if environment is not None else 'Baseline',
            'role': role,
            'degrees_experiment': experiment.get('degrees_experiment', False),
            'cot': experiment.get('COT', False),
        },
    }


def principle1_run_configured_experiments(experiments, output_dir, default_temperatures, run_experiments=True, run_analysis=True):
    supported_models = set(filter_supported_models(sorted({experiment['model'] for experiment in experiments})))
    non_cot_outfiles = []
    cot_outfiles = []
    experiments_to_analyze = []
    experiment_records = []

    for experiment in experiments:
        if not experiment.get('run', True):
            continue

        model = experiment['model']
        if model not in supported_models:
            print(f'Skipping {experiment["name"]}: {model} is not supported in this environment.')
            continue

        record = principle1_build_experiment_record(experiment, output_dir, default_temperatures)
        experiment_records.append(record)

        if experiment.get('include_in_exponent_summary', True):
            if experiment.get('COT'):
                cot_outfiles.append(record['outfile'])
            else:
                non_cot_outfiles.append(record['outfile'])

        if experiment.get('analyze_detail', False):
            experiments_to_analyze.append(record)

    if run_experiments:
        batch_groups = collections.defaultdict(list)
        for record in experiment_records:
            if record['model'].startswith('Qwen/') and record['experiment'].get('batch', True):
                batch_key = (
                    record['model'],
                    record['cot'],
                    json.dumps(record.get('cot_config'), sort_keys=True, default=str),
                )
                batch_groups[batch_key].append(record)
                continue

            principle1_run_network_formation_experiment(
                **record['parameters'],
                outfile=record['outfile'],
                temperatures=record['temperatures'],
                environment=record['environment'],
                role=record['role'],
                degrees=record['degrees'],
                model=record['model'],
                cot=record['cot'],
                cot_config=record.get('cot_config'),
                hash_and_shuffle=record['hash_and_shuffle'],
                metadata=record['metadata'],
            )

        for records in batch_groups.values():
            principle1_run_network_formation_experiments_batch(records)

    if run_analysis:
        for record in experiments_to_analyze:
            principle1_analyze_experiments(record['outfile'], dgr=not record['degrees'])

        if non_cot_outfiles:
            principle1_analyze_experiments_multiple_llms(non_cot_outfiles)
        if cot_outfiles:
            principle1_analyze_experiments_multiple_llms(cot_outfiles, sfx='_cot')

    return {
        'supported_models': supported_models,
        'non_cot_outfiles': non_cot_outfiles,
        'cot_outfiles': cot_outfiles,
        'experiments_to_analyze': experiments_to_analyze,
        'experiment_records': experiment_records,
    }

# --- End Principle 1 preferential-attachment utilities ---
