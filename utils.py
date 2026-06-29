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
