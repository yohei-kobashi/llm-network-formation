import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import glob
import json
import random
import os
import copy
import collections
import itertools
import gc
import math
import re
import hashlib
import ast
import scipy
import scipy.stats as stats
import netgraph
import powerlaw as pwl
import seaborn as sns
import replicate
import anthropic
import torch
import statsmodels.api as sm
import dataloader
import link_prediction
import dcm
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
from statsmodels.iolib.summary2 import summary_col
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


class ModelUnavailableError(RuntimeError):
    """Raised when a local model can run in neither vLLM nor the Transformers
    fallback. Experiments must skip the model instead of retrying: every
    request would return None, and the run would write records whose graphs
    have no edges (which then poison resume/skip logic and analysis)."""


def _raise_if_model_unusable(model):
    if model in vllm_unavailable_models and model in transformers_unavailable_models:
        detail = f" vLLM import error: {vllm_import_error}." if vllm_import_error else ""
        raise ModelUnavailableError(
            f"{model} cannot run in this runtime: vLLM failed to import/load and the "
            f"Transformers fallback also failed.{detail} In Colab, re-run the setup cell "
            f"and let it restart the runtime, then run the notebook again."
        )

SHARED_BASELINE_MODEL = 'Qwen/Qwen3.5-4B'
SHARED_MODEL_NAMES = [
    'gpt-5-nano',
    'Qwen/Qwen3.5-4B',
    'Qwen/Qwen3.5-0.8B',
]
SHARED_DEFAULT_TEMPERATURES = [1.0]
SHARED_DEFAULT_COT_CONFIG = {'max_new_tokens': 8192, 'qwen_enable_thinking': True}
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


def set_cot_retry_max_new_tokens(value):
    """Update the module-wide CoT retry token budget used by retry_cot_config."""
    global COT_RETRY_MAX_NEW_TOKENS
    COT_RETRY_MAX_NEW_TOKENS = int(value)


def run_cot_budget_calibration(
    experiments,
    output_dir,
    default_temperatures,
    default_cot_config,
    build_calibration_requests,
    parse_response,
    calibration_filename,
    apply_selected_budget=set_cot_retry_max_new_tokens,
    run_experiments=True,
    calibrate=True,
    calibration_sample_size=20,
    calibration_max_new_tokens=65536,
    calibration_percentile=0.90,
    calibration_margin=1.5,
    retry_token_buckets=(8192, 16384, 32768, 65536),
    calibration_seed=0,
):
    """Estimate the CoT retry token budget shared across the principle notebooks.

    ``build_calibration_requests`` must return ``(record, request_items)`` where
    ``request_items`` is a list of ``(label, request)`` tuples and each request has
    ``prompt`` and ``response_schema`` keys. ``parse_response`` is the principle's
    response parser, used to score how many sampled generations parse cleanly.
    """
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
        apply_selected_budget(int(summary['selected_max_new_tokens']))
        print('Loaded CoT retry budget calibration:', COT_RETRY_MAX_NEW_TOKENS)
        return summary

    experiment = cot_experiments[0]
    record, request_items = build_calibration_requests(
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
            parse_response(output, request)
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
        'sampled_nodes': [label for label, _ in request_items],
    })

    apply_selected_budget(int(summary['selected_max_new_tokens']))
    with open(calibration_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print('Selected COT_RETRY_MAX_NEW_TOKENS =', COT_RETRY_MAX_NEW_TOKENS)
    print('Calibration summary saved to', calibration_file)
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
        _raise_if_model_unusable(model)
        if model in vllm_unavailable_models:
            answer = _try_transformers_response(prompt, model, temperature, system_prompt, response_schema=response_schema, cot=cot, cot_config=cot_config)
            _raise_if_model_unusable(model)
            return answer
        try:
            return _get_vllm_response(prompt, model, temperature, system_prompt, response_schema=response_schema, cot=cot, cot_config=cot_config)
        except Exception as e:
            vllm_unavailable_models.add(model)
            print(f"[vLLM fallback] {model} failed to load or generate with vLLM: {str(e).splitlines()[0]}")
            print(f"[vLLM fallback] Retrying {model} with Transformers.")
            answer = _try_transformers_response(prompt, model, temperature, system_prompt, response_schema=response_schema, cot=cot, cot_config=cot_config)
            _raise_if_model_unusable(model)
            return answer
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
        _raise_if_model_unusable(model)
        if model in vllm_unavailable_models:
            answers = _try_transformers_responses(prompts, model, temperature, system_prompt, response_schemas=response_schemas, cot=cot, cot_config=cot_config)
            _raise_if_model_unusable(model)
            return answers
        try:
            return _get_vllm_responses(prompts, model, temperature, system_prompt, response_schemas=response_schemas, cot=cot, cot_config=cot_config)
        except Exception as e:
            vllm_unavailable_models.add(model)
            print(f"[vLLM fallback] {model} failed to load or generate with vLLM: {str(e).splitlines()[0]}")
            print(f"[vLLM fallback] Retrying {model} with Transformers.")
            answers = _try_transformers_responses(prompts, model, temperature, system_prompt, response_schemas=response_schemas, cot=cot, cot_config=cot_config)
            _raise_if_model_unusable(model)
            return answers

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


def set_reset_outputs(value=True):
    """Toggle 'redo from scratch' mode.

    When enabled, the first time each output file is encountered during a run it
    is deleted and regenerated from zero (overwriting the copy on Google Drive)
    instead of resuming from the saved simulations. Call this once before the run
    cell, e.g. set_reset_outputs(RESET_OUTPUTS)."""
    global IGNORE_EXISTING_OUTPUTS
    IGNORE_EXISTING_OUTPUTS = bool(value)
    # Re-arm per-file removal so a fresh run with reset on deletes again.
    RESET_OUTFILES.clear()


def maybe_reset_outfile(outfile):
    """Delete an existing output file once per run when reset mode is on, so the
    experiment regenerates it from scratch instead of appending/skipping."""
    if not IGNORE_EXISTING_OUTPUTS:
        return
    if outfile in RESET_OUTFILES:
        return
    if os.path.exists(outfile):
        os.remove(outfile)
        print(f'Reset: removed existing output file {outfile}')
    RESET_OUTFILES.add(outfile)


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
    if text is None:
        raise ValueError('no response text (generation failed or returned None)')
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
    # On the first failure, immediately disable thinking instead of escalating
    # max_new_tokens. Re-running with a larger token budget is slow; turning off
    # thinking keeps the output short so the base budget stays sufficient.
    config['qwen_enable_thinking'] = False
    return config


def principle2_select_neighbor(G, t, temperature, model, environment, role, num_common_neighbors, cot, cot_config=None):
    request = principle2_build_neighbor_request(G, t, environment, role, num_common_neighbors, cot, model, cot_config)
    for i in range(3):
        ans = None
        try:
            attempt_cot_config = retry_cot_config(cot_config, i) if cot else cot_config
            if cot and attempt_cot_config and attempt_cot_config != cot_config:
                print(f'Retrying with qwen_enable_thinking={attempt_cot_config.get("qwen_enable_thinking")}, max_new_tokens={attempt_cot_config.get("max_new_tokens")}')
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

def principle2_build_summary_records(filenames, er=False):
    """Read the per-simulation jsonl outfiles and compute the summary metrics
    (Marginal Transitivity, Prob. of Edge within Community, Top-$k$ curves) for
    each record. Shared by the plotting table and the markdown report so both
    stay in sync."""
    records = []

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

    return records


def principle2_get_table(filenames, sfx='', environments=True, transitivity_null=-1, probability_null=-1, er=False, baseline_model_name='Qwen/Qwen3.5-4B'):
    os.makedirs('figures', exist_ok=True)
    os.makedirs('tables', exist_ok=True)

    records = principle2_build_summary_records(filenames, er=er)

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
    temperatures_present = df[df['Temperature'].notna()]['Temperature']
    if temperatures_present.empty:
        print('Skipping multi-LLM plots: no records with a temperature value.')
        return
    default_temperature = temperatures_present.iloc[0]

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

    palette=['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9']

    breakpoints_arr = [('top', np.array([0.1, 0.2, 0.3, 0.4, 0.5])), ('all', np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]))]

    for label, breakpoints in breakpoints_arr:
        # A fresh figure per breakpoint range: reusing the same axes across
        # iterations re-plots every line, duplicating the legend entries.
        fig, ax = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
        fig.suptitle(f'{condition_label}: Probability of Connecting to Top-$k$ Common Neighbors', fontsize=SMALL_SIZE)

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



PRINCIPLE2_RENAME_MODELS = {
    'gpt-5-nano': 'GPT-5 Nano',
    'gpt-5-mini': 'GPT-5 Mini',
    'Qwen-Qwen3.5-4B': 'Qwen 3.5 4B',
    'Qwen-Qwen3.5-2B': 'Qwen 3.5 2B',
    'Qwen-Qwen3.5-0.8B': 'Qwen 3.5 0.8B',
    'gpt-5-nano_cot': 'GPT-5 Nano (CoT)',
    'gpt-5-mini_cot': 'GPT-5 Mini (CoT)',
    'Qwen-Qwen3.5-4B_cot': 'Qwen 3.5 4B (CoT)',
    'Qwen-Qwen3.5-2B_cot': 'Qwen 3.5 2B (CoT)',
    'Qwen-Qwen3.5-0.8B_cot': 'Qwen 3.5 0.8B (CoT)',
}

PRINCIPLE2_RENAME_ENV = {
    'school': 'School',
    'work': 'Work',
    'community': 'Community',
    'school_cot': 'School (CoT)',
    'work_cot': 'Work (CoT)',
    'community_cot': 'Community (CoT)',
}

PRINCIPLE2_GROUP_LABELS = {
    'sbm': 'Non-CoT SBM initialization',
    'er': 'ER initialization',
    'cot': 'CoT',
}


def _principle2_markdown_table(headers, rows):
    """Render a markdown table without depending on the optional `tabulate`
    package. Floats are shown with 4 decimals, everything else via ``str``."""
    def fmt(x):
        if isinstance(x, bool):
            return str(x)
        if isinstance(x, float):
            if not np.isfinite(x):
                return '—'
            return f'{x:.4f}'
        return str(x)

    lines = [
        '| ' + ' | '.join(str(h) for h in headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(fmt(x) for x in row) + ' |')
    return '\n'.join(lines)


def principle2_write_markdown_report(
    run_results,
    output_dir,
    analysis_nulls=None,
    top_k_breakpoints=(0.1, 0.2, 0.3, 0.4, 0.5),
    filename='principle_2_results.md',
    title='Principle 2: Triadic Closure — Results',
    timestamp=None,
):
    """Summarize the configured-experiment results as a markdown file and write
    it to ``output_dir`` (e.g. the Google Drive output directory).

    For each summary group (SBM / ER / CoT) the report lists, per
    Model / Environment / Temperature, the mean Marginal Transitivity, the mean
    Prob. of Edge within Community, and the mean Probability of Connecting to
    Top-$k$ Common Neighbors at the requested breakpoints, alongside the null
    baselines produced by the analysis pass."""
    import datetime

    outfiles_by_group = run_results.get('outfiles_by_group', {})
    if analysis_nulls is None:
        analysis_nulls = run_results.get('analysis_nulls', {})
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    sections = [f'# {title}', f'_Generated: {timestamp}_']

    for group in ('sbm', 'er', 'cot'):
        filenames = outfiles_by_group.get(group)
        if not filenames:
            continue

        er = group == 'er'
        records = principle2_build_summary_records(filenames, er=er)
        if not records:
            continue

        df = pd.DataFrame(records)
        df['Model'] = df['Model'].apply(lambda x: PRINCIPLE2_RENAME_MODELS.get(x, x))
        df['Environment'] = df['Environment'].apply(lambda x: PRINCIPLE2_RENAME_ENV.get(x, x))

        sections.append(f'## {PRINCIPLE2_GROUP_LABELS.get(group, group)}')

        nulls = analysis_nulls.get(group)
        if nulls and tuple(nulls) != (-1, -1):
            transitivity_null, probability_null = nulls
            sections.append(
                f'- Transitivity null baseline: **{transitivity_null:.4f}**\n'
                f'- Prob. of edge within community null baseline: **{probability_null:.4f}**'
            )

        grouped = df.groupby(['Model', 'Environment', 'Temperature'], dropna=False)

        headers = [
            'Model', 'Environment', 'Temperature',
            'Marginal Transitivity', 'Prob. of Edge within Community', 'N sims',
        ]
        rows = []
        topk_rows = []
        for key, sub in grouped:
            model, environment, temperature = key
            temp_str = f'{temperature:g}' if isinstance(temperature, (int, float)) else str(temperature)
            rows.append([
                model,
                environment,
                temp_str,
                float(sub['Marginal Transitivity'].mean()),
                float(sub['Prob. of Edge within Community'].mean()),
                len(sub),
            ])

            per_row = []
            for curve in sub['Probability of Connecting to Top-$k$']:
                curve = np.asarray(curve, dtype=float)
                n = curve.shape[0]
                if n == 0:
                    continue
                per_row.append([curve[min(int(b * n), n - 1)] for b in top_k_breakpoints])
            if per_row:
                mean_vals = np.mean(per_row, axis=0)
                topk_rows.append([model, environment, temp_str] + [float(v) for v in mean_vals])

        sections.append(_principle2_markdown_table(headers, rows))

        if top_k_breakpoints and topk_rows:
            topk_headers = ['Model', 'Environment', 'Temperature'] + [
                f'Top-{int(round(b * 100))}%' for b in top_k_breakpoints
            ]
            sections.append('**Probability of Connecting to Top-$k$ Common Neighbors**')
            sections.append(_principle2_markdown_table(topk_headers, topk_rows))

    if len(sections) <= 2:
        sections.append('_No analyzable results were found for any summary group._')

    content = '\n\n'.join(sections) + '\n'
    output_path = os.path.join(os.fspath(output_dir), filename)
    with open(output_path, 'w') as f:
        f.write(content)
    print(f'Wrote markdown report to {output_path}')
    return output_path


# --- Shared markdown-report helpers (used by principle 1/2/3/5 reports) ---
MARKDOWN_RENAME_MODELS = PRINCIPLE2_RENAME_MODELS
MARKDOWN_RENAME_ENV = PRINCIPLE2_RENAME_ENV


def _markdown_model_env(d, fallback_model='', fallback_env='Baseline'):
    """Derive the display Model / Environment for one jsonl record, matching the
    logic used inside the `principleN_get_table` plotting helpers."""
    model = str(d.get('model', fallback_model)).replace('/', '-')
    if d.get('cot') and not model.endswith('_cot'):
        model = f'{model}_cot'
    environment = d.get('environment', fallback_env)
    if environment is None:
        environment = 'Baseline'
    if d.get('cot') and environment != 'Baseline' and not str(environment).endswith('_cot'):
        environment = f'{environment}_cot'
    return MARKDOWN_RENAME_MODELS.get(model, model), MARKDOWN_RENAME_ENV.get(environment, environment)


def _aggregate_markdown_rows(records, metric_keys, group_keys=('Model', 'Environment', 'Temperature')):
    """Group records by ``group_keys`` and average each metric across simulations,
    returning table rows plus a trailing ``N sims`` count."""
    groups = collections.OrderedDict()
    for r in records:
        key = tuple(r.get(k) for k in group_keys)
        groups.setdefault(key, []).append(r)

    rows = []
    for key, subs in groups.items():
        row = []
        for gk, val in zip(group_keys, key):
            if gk == 'Temperature' and isinstance(val, (int, float)):
                row.append(f'{val:g}')
            else:
                row.append(val)
        for mk in metric_keys:
            vals = [
                s[mk] for s in subs
                if isinstance(s.get(mk), (int, float)) and np.isfinite(s[mk])
            ]
            row.append(float(np.mean(vals)) if vals else float('nan'))
        row.append(len(subs))
        rows.append(row)
    return rows


def _topk_markdown_table(records, breakpoints, curve_key='_topk',
                         heading='**Probability of Connecting to Top-$k$**'):
    """Average each group's Top-$k$ cumulative curve at the requested breakpoints."""
    groups = collections.OrderedDict()
    for r in records:
        key = (r['Model'], r['Environment'], r['Temperature'])
        groups.setdefault(key, []).append(r)

    rows = []
    for (model, environment, temperature), subs in groups.items():
        temp_str = f'{temperature:g}' if isinstance(temperature, (int, float)) else str(temperature)
        per_row = []
        for s in subs:
            curve = np.asarray(s[curve_key], dtype=float)
            n = curve.shape[0]
            if n == 0:
                continue
            per_row.append([curve[min(int(b * n), n - 1)] for b in breakpoints])
        if not per_row:
            continue
        mean_vals = np.mean(per_row, axis=0)
        rows.append([model, environment, temp_str] + [float(v) for v in mean_vals])

    if not rows:
        return None
    headers = ['Model', 'Environment', 'Temperature'] + [
        f'Top-{int(round(b * 100))}%' for b in breakpoints
    ]
    return heading + '\n\n' + _principle2_markdown_table(headers, rows)


def _write_markdown_report_file(output_dir, filename, title, sections, timestamp=None):
    import datetime
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    body = [f'# {title}', f'_Generated: {timestamp}_']
    body.extend(sections)
    if len(body) <= 2:
        body.append('_No analyzable results were found._')
    content = '\n\n'.join(body) + '\n'
    output_path = os.path.join(os.fspath(output_dir), filename)
    with open(output_path, 'w') as f:
        f.write(content)
    print(f'Wrote markdown report to {output_path}')
    return output_path


def _dataframe_to_markdown(df, float_fmt='{:.4g}'):
    """Render a pandas DataFrame as a markdown table without depending on the
    optional `tabulate` package."""
    headers = [str(c) for c in df.columns]
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
    ]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(float_fmt.format(value) if np.isfinite(value) else '—')
            elif value is None:
                cells.append('—')
            else:
                cells.append(str(value).replace('\n', ' '))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def principle1_write_markdown_report(
    run_results,
    output_dir,
    top_k_breakpoints=(0.1, 0.2, 0.3, 0.4, 0.5),
    filename='principle_1_results.md',
    title='Principle 1: Preferential Attachment — Results',
    timestamp=None,
):
    """Summarize the Principle 1 (preferential attachment) results as markdown and
    write them to ``output_dir``. Per Model / Environment / Temperature it reports
    the mean fitted power-law exponent, its standard error, the KS statistic
    against a Barabasi-Albert reference, and the Top-$k$ degree-share curve."""
    groups = [
        ('Non-CoT', run_results.get('non_cot_outfiles') or []),
        ('CoT', run_results.get('cot_outfiles') or []),
    ]

    metric_keys = ['Power-law $\\hat\\gamma$', 'Std. err. $\\sigma$', 'KS stat (vs BA)']
    sections = []

    for label, filenames in groups:
        filenames = [f for f in filenames if os.path.exists(f)]
        if not filenames:
            continue

        records = []
        for filename_ in filenames:
            with open(filename_) as f:
                lines = f.read().splitlines()
            for line in lines:
                d = json.loads(line)
                Gs = principle1_reconstruct_graphs(d)
                G = Gs[-1]
                degrees = np.array([G.degree(n) for n in G.nodes()])
                if len(degrees) == 0:
                    continue

                fit = pwl.Fit(degrees, discrete=True)
                G_ba = nx.barabasi_albert_graph(n=d['n'], m=1, seed=1)
                degrees_ba = np.array([G_ba.degree(n) for n in G_ba.nodes()])
                ks = stats.ks_2samp(degrees, degrees_ba)

                sorted_degrees = np.sort(degrees)[::-1]
                sorted_degrees = np.insert(sorted_degrees, 0, 0)
                cumsum = np.cumsum(sorted_degrees)
                topk = cumsum / cumsum[-1] if cumsum[-1] else cumsum

                model, environment = _markdown_model_env(d)
                records.append({
                    'Model': model,
                    'Environment': environment,
                    'Temperature': d['temperature'],
                    'Power-law $\\hat\\gamma$': float(fit.alpha),
                    'Std. err. $\\sigma$': float(fit.sigma),
                    'KS stat (vs BA)': float(ks[0]),
                    '_topk': topk,
                })

        if not records:
            continue

        sections.append(f'## {label}')
        rows = _aggregate_markdown_rows(records, metric_keys)
        headers = ['Model', 'Environment', 'Temperature'] + metric_keys + ['N sims']
        sections.append(_principle2_markdown_table(headers, rows))

        if top_k_breakpoints:
            topk_table = _topk_markdown_table(
                records, top_k_breakpoints,
                heading='**Probability of Connecting to Top-$k$ (degree share)**',
            )
            if topk_table:
                sections.append(topk_table)

    return _write_markdown_report_file(output_dir, filename, title, sections, timestamp)


def principle3_write_markdown_report(
    run_results,
    output_dir,
    filename='principle_3_results.md',
    title='Principle 3: Homophily — Results',
    timestamp=None,
):
    """Summarize the Principle 3 (homophily) results as markdown and write them to
    ``output_dir``. Per Model / Environment / Temperature it reports the mean
    attribute assortativities, Louvain modularity and community structure, plus a
    Random-graph baseline for comparison."""
    outfiles_by_group = run_results.get('outfiles_by_group', {})
    attribute_columns = [
        ('Location', 'location'),
        ('Favorite Color', 'favorite color'),
        ('Hobby', 'hobby'),
        ('Lucky Number', 'lucky number'),
    ]

    # (group key, heading, profiles filename, attribute display names, has communities, mutual acceptance)
    group_specs = [
        ('profiles', 'Homophily (profiles)', PRINCIPLE3_DEFAULT_PROFILES_FILENAME,
         ['Location', 'Favorite Color', 'Hobby'], True, False),
        ('lucky_number', 'Lucky Number', PRINCIPLE3_LUCKY_NUMBER_PROFILES_FILENAME,
         ['Location', 'Hobby', 'Favorite Color', 'Lucky Number'], False, False),
        ('mutual_acceptance', 'Mutual Acceptance', PRINCIPLE3_DEFAULT_PROFILES_FILENAME,
         ['Location', 'Favorite Color', 'Hobby'], True, True),
        ('cot', 'CoT', PRINCIPLE3_DEFAULT_PROFILES_FILENAME,
         ['Location', 'Favorite Color', 'Hobby'], True, False),
    ]

    def assortativities(G):
        out = {}
        for display, attr in attribute_columns:
            try:
                out[display] = float(nx.attribute_assortativity_coefficient(G, attr))
            except Exception:
                out[display] = None
        return out

    sections = []
    for group_key, heading, profiles_filename, attr_names, communities, mutual_acceptance in group_specs:
        filenames = [f for f in (outfiles_by_group.get(group_key) or []) if os.path.exists(f)]
        if not filenames or not os.path.exists(profiles_filename):
            continue

        with open(profiles_filename) as f:
            profiles = [json.loads(line) for line in f.read().splitlines()]
        profiles_dict = {str(profile['name']): profile for profile in profiles}
        profiles_dict_int = {int(profile['name']): profile for profile in profiles}

        records = []
        for filename_ in filenames:
            with open(filename_) as f:
                lines = f.read().splitlines()
            for line in lines:
                d = json.loads(line)
                G = principle3_graph_from_stored(d['graphs'][-1], profiles_dict)
                G.remove_edges_from(nx.selfloop_edges(G))
                G.remove_nodes_from(list(nx.isolates(G)))

                model, environment = _markdown_model_env(d)
                record = {'Model': model, 'Environment': environment, 'Temperature': d['temperature']}
                record.update(assortativities(G))

                if communities:
                    try:
                        louvain = nx.algorithms.community.louvain_communities(G, weight='similarity')
                        record['Modularity'] = float(
                            nx.algorithms.community.modularity(G, louvain, weight='similarity'))
                        record['Number of Communities'] = len(louvain)
                        record['Average Community Size'] = float(np.mean([len(c) for c in louvain]))
                    except Exception:
                        pass
                if mutual_acceptance:
                    record['Mutual Acceptance Probability'] = d.get('mutual_acceptance_probability', 100)

                records.append(record)

                # Random-graph baseline for the same size/temperature.
                try:
                    G_random = principle3_network_growth(
                        d['n'], d['temperature'], method='random',
                        model='', environment='', role='')[0][-1]
                    nx.set_node_attributes(G_random, profiles_dict_int)
                    random_record = {
                        'Model': 'Random', 'Environment': 'Baseline', 'Temperature': d['temperature']}
                    random_record.update(assortativities(G_random))
                    records.append(random_record)
                except Exception:
                    pass

        if not records:
            continue

        metric_keys = list(attr_names)
        if communities:
            metric_keys += ['Modularity', 'Number of Communities', 'Average Community Size']
        if mutual_acceptance:
            metric_keys += ['Mutual Acceptance Probability']

        sections.append(f'## {heading}')
        rows = _aggregate_markdown_rows(records, metric_keys)
        headers = ['Model', 'Environment', 'Temperature'] + metric_keys + ['N sims']
        sections.append(_principle2_markdown_table(headers, rows))

    return _write_markdown_report_file(output_dir, filename, title, sections, timestamp)


def principle5_write_markdown_report(
    run_results,
    output_dir,
    filename='principle_5_results.md',
    title='Principle 5: Small-World Phenomenon — Results',
    timestamp=None,
):
    """Summarize the Principle 5 (small-world) results as markdown and write them to
    ``output_dir``. Per Model / Environment / Temperature / (n, k, beta) it reports
    the mean average shortest-path length ``L`` and clustering coefficient ``C``,
    alongside a Watts-Strogatz baseline for each (n, k, beta)."""
    outfiles_by_group = run_results.get('outfiles_by_group', {})
    group_labels = {'default': 'Default', 'cot': 'CoT'}
    group_keys = ('Model', 'Environment', 'Temperature', 'n', 'k', 'beta')
    metric_keys = ['$L$', '$C$']

    def reconstruct(graph):
        G = nx.Graph()
        for node, neighbors in graph.items():
            node = int(node)
            G.add_node(node)
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        return G

    sections = []
    for group_key in ('default', 'cot'):
        filenames = [f for f in (outfiles_by_group.get(group_key) or []) if os.path.exists(f)]
        if not filenames:
            continue

        records = []
        watts_strogatz_seen = set()
        for filename_ in filenames:
            with open(filename_) as f:
                lines = f.read().splitlines()
            for line in lines:
                d = json.loads(line)
                G = reconstruct(d['graphs'][-1])
                try:
                    length = principle5_average_shortest_path_length_lcc(G)
                    clustering = nx.average_clustering(G)
                except Exception:
                    continue

                model, environment = _markdown_model_env(d)
                records.append({
                    'Model': model, 'Environment': environment, 'Temperature': d['temperature'],
                    'n': d['n'], 'k': d['k'], 'beta': d['beta'],
                    '$L$': float(length), '$C$': float(clustering),
                })
                watts_strogatz_seen.add((d['n'], d['k'], d['beta']))

        # Watts-Strogatz reference for each (n, k, beta).
        for n, k, beta in sorted(watts_strogatz_seen):
            try:
                Gs, _ = principle5_network_growth(
                    n, k, beta, 0, method='W-S', model='', environment='', role='')
                length = principle5_average_shortest_path_length_lcc(Gs[-1])
                clustering = nx.average_clustering(Gs[-1])
            except Exception:
                continue
            records.append({
                'Model': 'Watts-Strogatz', 'Environment': 'Baseline', 'Temperature': '—',
                'n': n, 'k': k, 'beta': beta,
                '$L$': float(length), '$C$': float(clustering),
            })

        if not records:
            continue

        sections.append(f'## {group_labels.get(group_key, group_key)}')
        rows = _aggregate_markdown_rows(records, metric_keys, group_keys=group_keys)
        headers = list(group_keys) + metric_keys + ['N sims']
        sections.append(_principle2_markdown_table(headers, rows))

    return _write_markdown_report_file(output_dir, filename, title, sections, timestamp)


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
        for record in experiment_records:
            maybe_reset_outfile(record['outfile'])
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

            try:
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
            except ModelUnavailableError as e:
                print(f'Skipping experiment {record["metadata"]["experiment_name"]}: {e}')

        for records in batch_groups.values():
            try:
                principle2_run_network_formation_experiments_batch(records)
            except ModelUnavailableError as e:
                names = ', '.join(r['metadata']['experiment_name'] for r in records)
                print(f'Skipping batched experiments ({names}): {e}')

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
    # netgraph's spring layout asserts a non-empty edge list, so fall back to a
    # plain networkx node/edge drawing when the snapshot has no edges yet.
    if not use_netgraph or G.number_of_edges() == 0:
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

def principle1_record_has_any_edge(record):
    """A saved record with zero successful edges (every LLM call failed) is
    garbage: treat it as NOT completed so the simulation is re-run on resume,
    instead of being skipped forever."""
    history = record.get('edge_history')
    if history is not None:
        return any(selected is not None for _, selected in history)
    graphs = record.get('graphs')
    if graphs:
        last = graphs[-1]
        if isinstance(last, dict):
            return any(neighbors for neighbors in last.values())
    return False

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
            if not principle1_record_has_any_edge(scenario):
                print(f'Ignoring saved simulation with no edges (n={scenario["n"]}, simulation={scenario["simulation"]}, temperature={scenario["temperature"]}); it will be re-run.')
                continue
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

                attempt_cot_config = retry_cot_config(cot_config, attempt) if cot else cot_config
                if cot and attempt_cot_config != cot_config:
                    print(f'Retry attempt {attempt + 1}: max_new_tokens={attempt_cot_config.get("max_new_tokens")}, qwen_enable_thinking={attempt_cot_config.get("qwen_enable_thinking")}')
                prompts = [request['prompt'] for _, request in remaining]
                response_schemas = [request['response_schema'] for _, request in remaining]
                answers = get_responses(
                    prompts,
                    model,
                    temperature=temperature,
                    response_schemas=response_schemas,
                    cot=cot,
                    cot_config=attempt_cot_config,
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
            if not principle1_record_has_any_edge(scenario):
                print(f'Ignoring saved simulation with no edges (n={scenario["n"]}, simulation={scenario["simulation"]}, temperature={scenario["temperature"]}); it will be re-run.')
                continue
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

def _safe_power_law_subfit(fit):
    """Return ``fit.power_law`` or ``None`` if the degree distribution is too
    degenerate to fit. A near-star graph (e.g. an LLM that almost always picks
    node 0) yields ``xmin=nan``; accessing ``fit.power_law`` then triggers a
    lazy fit that raises ``ValueError: No data points in defined range of the
    distribution.`` Callers fall back to nan / skip the fitted-line plot."""
    try:
        power_law = fit.power_law
        # Force the lazy fit so a degenerate-distribution failure surfaces here.
        _ = power_law.alpha
        return power_law
    except Exception:
        return None

def _safe_power_law_fit(degrees, **kwargs):
    """Build a ``powerlaw.Fit`` or return ``None`` when ``degrees`` is empty or
    too small to fit. ``pwl.Fit([])`` raises ``ValueError: zero-size array to
    reduction operation minimum which has no identity`` because it computes
    ``np.min(data)`` on an empty array; this happens when the final graph for a
    simulation has no non-isolated nodes (e.g. parsing failed and no edges were
    ever formed). Callers fall back to nan / skip the fit and plots."""
    if degrees is None or len(degrees) == 0:
        return None
    try:
        return pwl.Fit(degrees, **kwargs)
    except Exception:
        return None

def principle1_analyze_experiments(filename, dgr=True):
    os.makedirs('figures/principle_1', exist_ok=True)

    suffix = os.path.split(os.path.splitext(filename)[0])[-1]

    palette = ['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9']

    with open(filename) as f:
        lines = f.read().splitlines()

    data = []

    for line in lines:
        data.append(json.loads(line))

    if not data:
        print(f'Skipping analysis for {filename}: file has no records.')
        return

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

        if not Gs or Gs[-1].number_of_nodes() == 0:
            print(f'Skipping record (n={d["n"]}, simulation={d["simulation"]}, temperature={d["temperature"]}): final graph has no edges (all LLM calls failed).')
            continue

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

        powerlaw_fit = _safe_power_law_fit(degrees, discrete=True)
        powerlaw_fit_barabasi_albert = pwl.Fit(degrees_barabasi_albert, discrete=True)

        if powerlaw_fit is None:
            print(f'  Skipping power-law analysis for n={d["n"]}, temperature={d["temperature"]}: '
                  f'final graph has no non-isolated nodes (|degrees|={len(degrees)}).')
            wasserstein_distances[d['n'], d['temperature']].append(float('nan'))
            gammas[d['n'], d['temperature']].append(float('nan'))
            sigmas[d['n'], d['temperature']].append(float('nan'))
            gammas_barabasi_albert[d['n'], d['temperature']].append(powerlaw_fit_barabasi_albert.alpha)
            sigmas_barabasi_albert[d['n'], d['temperature']].append(powerlaw_fit_barabasi_albert.sigma)
            ks_powerlaw[d['n'], d['temperature']].append(float('nan'))
            pwl_fits[d['n'], d['temperature']].append(None)
            pwl_fits_barabasi_albert[d['n'], d['temperature']].append(powerlaw_fit_barabasi_albert)
            plt.close(fig)
            plt.close(fig_barabasi_albert)
            continue

        print(f'Temperature {d["temperature"]}: xmin: {powerlaw_fit.xmin}, alpha: {powerlaw_fit.alpha}, sigma: {powerlaw_fit.sigma}')

        wasserstein_distances[d['n'], d['temperature']].append(stats.wasserstein_distance(degrees, degrees_barabasi_albert))
        gammas[d['n'], d['temperature']].append(powerlaw_fit.alpha)
        sigmas[d['n'], d['temperature']].append(powerlaw_fit.sigma)

        gammas_barabasi_albert[d['n'], d['temperature']].append(powerlaw_fit_barabasi_albert.alpha)
        sigmas_barabasi_albert[d['n'], d['temperature']].append(powerlaw_fit_barabasi_albert.sigma)

        llm_power_law = _safe_power_law_subfit(powerlaw_fit)
        if llm_power_law is None:
            print(f'  Skipping power-law fit for n={d["n"]}, temperature={d["temperature"]}: degree distribution is degenerate (likely a star graph).')
            ks_stat_powerlaw = float('nan')
        else:
            ks_stat_powerlaw = getattr(llm_power_law, 'D', None)
            if ks_stat_powerlaw is None:
                ks_stat_powerlaw = llm_power_law.KS(degrees)
        ks_powerlaw[d['n'], d['temperature']].append(ks_stat_powerlaw)

        ax[-1].set_title('Degree distribution')
        ax[-1].spines[['right', 'top']].set_visible(False)

        ax_barabasi_albert[-1].set_title('Degree distribution')

        powerlaw_fit.plot_ccdf(linewidth=3, ax=ax[-1], color='#e74c3c', label='LLM (Empirical)')
        powerlaw_fit_barabasi_albert.plot_ccdf(linewidth=3, ax=ax[-1], color='#3498db', label='BA (Empirical)')

        powerlaw_fit_barabasi_albert.plot_ccdf(linewidth=3, ax=ax_barabasi_albert[-1], color='#3498db', label='BA (Empirical)')

        if llm_power_law is not None:
            llm_power_law.plot_ccdf(ax=ax[-1], color='#e74c3c', linestyle='--', label='LLM (Power law fit)')
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
        if powerlaw_fit is None:
            continue
        powerlaw_fit.plot_ccdf(linewidth=3, ax=ax[0, -1], color=palette[i], label=str(k[-1]))
        summary_power_law = _safe_power_law_subfit(powerlaw_fit)
        if summary_power_law is not None:
            summary_power_law.plot_ccdf(ax=ax[0, -1], color=palette[i], linestyle='--')


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

            if len(degrees) == 0:
                print(f'  Skipping {filename} record (n={d["n"]}, temperature={d["temperature"]}): '
                      f'final graph has no non-isolated nodes.')
                continue

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

    if not records:
        print(f'Skipping multi-LLM analysis{sfx or ""}: no usable records (all simulations were empty or missing).')
        return

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
        'gpt-5-nano_cot' : 'GPT-5 Nano-CoT',
        'gpt-5-mini_cot' : 'GPT-5 Mini-CoT',
        'Qwen-Qwen3.5-4B_cot' : 'Qwen 3.5 4B-CoT',
        'Qwen-Qwen3.5-2B_cot' : 'Qwen 3.5 2B-CoT',
        'Qwen-Qwen3.5-0.8B_cot' : 'Qwen 3.5 0.8B-CoT',
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
    temperatures_present = df[df['Temperature'].notna()]['Temperature']
    if temperatures_present.empty:
        print(f'Skipping multi-LLM plots{sfx or ""}: no records with a temperature value (only the Barabasi-Albert reference remained).')
        return
    default_temperature = temperatures_present.iloc[0]

    df_model = df.query('Environment == "Baseline" and Temperature == @default_temperature')
    df_model = df_model[df_model['Model'] != 'Barabasi-Albert']

    # Model comparison panel: base models plus the baseline model's CoT
    # variant as its own bar, in a fixed left-to-right order.
    model_order = ['GPT-5 Nano', 'Qwen 3.5 0.8B', 'Qwen 3.5 4B', f'{baseline_model}-CoT']
    df_model = df_model.copy()
    df_model['__order'] = df_model['Model'].apply(
        lambda m: model_order.index(m) if m in model_order else len(model_order))
    df_model = df_model.sort_values('__order').drop(columns='__order').reset_index(drop=True)

    df_environment = df.query('Model == @baseline_model and Temperature == @default_temperature')

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    sc_model = sns.barplot(data=df_model, y='$\\hat \\gamma$', x='Model', ax=ax[0], palette=['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9'])
    sc_environment = sns.barplot(data=df_environment, y='$\\hat \\gamma$', x='Environment', ax=ax[1], palette=['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9'])

    sc_model.set_xticklabels(sc_model.get_xticklabels(), rotation=90)
    sc_environment.set_xticklabels(sc_environment.get_xticklabels(), rotation=90)

    ax[0].errorbar(df_model['Model'], df_model['$\\hat \\gamma$'], yerr=df_model['$\\sigma$'], color='black', linestyle='', capsize=10, alpha=0.5)
    ax[1].errorbar(df_environment['Environment'], df_environment['$\\hat \\gamma$'], yerr=df_environment['$\\sigma$'], color='black', linestyle='', capsize=10, alpha=0.5)



    gamma_ba = df[df['Model'] == 'Barabasi-Albert']['$\\hat \\gamma$'].values[0]
    sigma_ba = df[df['Model'] == 'Barabasi-Albert']['$\\sigma$'].values[0]

    # draw baraasi-albert line
    ax[0].axhline(y=gamma_ba, color='#c0392b', linestyle='--', label='BA (Sample)', linewidth=3)
    ax[1].axhline(y=gamma_ba, color='#c0392b', linestyle='--', label='BA (Sample)', linewidth=3)

    ax[0].axhline(y=3, color='#34495e', linestyle=':', label='BA (Theoretical)', linewidth=3)
    ax[1].axhline(y=3, color='#34495e', linestyle=':', label='BA (Theoretical)', linewidth=3)

    ax[0].legend(fontsize=0.7*SMALL_SIZE)

    ax[0].set_ylim(1, 5)
    ax[1].set_ylim(1, 5)

    ax[1].set_ylabel('')

    ax[0].set_xlabel('')
    ax[1].set_xlabel('')

    ax[0].set_title('Model')
    ax[1].set_title('Environment')


    ax[1].get_yaxis().set_visible(False)

    ax[0].spines[['right', 'top']].set_visible(False)
    ax[1].spines[['right', 'top']].set_visible(False)

    # fig.tight_layout()

    fig.savefig(f'figures/exponents{sfx}.pdf', bbox_inches='tight')

    # plot probability of connecting to top-k

    palette=['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9']



    breakpoints_arr = [('top', np.array([0.01, 0.015, 0.02, 0.025])), ('all', np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]))]

    for label, breakpoints in breakpoints_arr:

        # A fresh figure per breakpoint range: reusing the same axes across
        # iterations re-plots every line, duplicating the legend entries.
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

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

        for i, environment in enumerate(df_environment['Environment']):
            n = len(df_environment['Top-$k$'].values[i])
            indices = np.array([int(x * n) for x in breakpoints])

            color = palette[i]
            linewidth = 1

            if environment == 'Baseline' and df_environment['Model'].values[i] == baseline_model and df_environment['Temperature'].values[i] == default_temperature:
                color = '#34495e'
                linewidth = 3

            ax[1].plot(100 * df_environment['Top-$k$'].values[i][indices], df_environment['Probability of Connecting to Top-$k$'].values[i][indices], label=environment, color=color, linewidth=linewidth, marker='x')

        ax[1].set_title('Environment')
        ax[1].set_xlabel('Top-$k$ (%)')
        # ax[1].set_xscale('log')
        # ax[1].set_yscale('log')
        ax[1].set_ylabel('')

        ax[1].legend(fontsize=0.7*SMALL_SIZE, ncols=2)

        ax[0].set_ylim(0, 1)
        ax[1].set_ylim(0, 1)

        # hide y axis numbers
        ax[1].get_yaxis().set_visible(False)

        ax[0].spines[['right', 'top']].set_visible(False)
        ax[1].spines[['right', 'top']].set_visible(False)

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


def principle1_build_cot_calibration_requests(experiment, output_dir, default_temperatures, sample_size=20, seed=0):
    record = principle1_build_experiment_record(experiment, output_dir, default_temperatures)
    n = record['parameters']['n_max']
    n0 = 1
    temperature = record['temperatures'][0]
    state = principle1_initialize_growth_state(n, n0, temperature, record)
    rng = random.Random(seed)

    collected = []
    while state['t'] < state['end_t']:
        request = principle1_build_neighbor_request(
            state['candidates'],
            state['candidate_idx'],
            state['environment'],
            state['role'],
            True,
            state['hash_and_shuffle'],
            state['model'],
        )
        collected.append((state['t'], request))

        # Advance the graph deterministically (without LLM calls) so later requests
        # reflect realistic candidate-set sizes.
        candidate_names = request['candidate_names']
        result = None
        if candidate_names:
            chosen = rng.choice(candidate_names)
            if request['hash_and_shuffle'] and request.get('hash2idx'):
                mapped = request['hash2idx'][chosen]
                chosen = int(mapped) if str(mapped).isdigit() else mapped
            result = {'name': chosen, 'reason': 'calibration'}
        principle1_advance_growth_state(state, result)

    rng.shuffle(collected)
    return record, collected[:sample_size]


def principle1_run_cot_budget_calibration(
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
    calibration_filename='principle_1_cot_budget_calibration.json',
):
    return run_cot_budget_calibration(
        experiments,
        output_dir,
        default_temperatures,
        default_cot_config,
        build_calibration_requests=principle1_build_cot_calibration_requests,
        parse_response=principle1_parse_neighbor_response,
        calibration_filename=calibration_filename,
        run_experiments=run_experiments,
        calibrate=calibrate,
        calibration_sample_size=calibration_sample_size,
        calibration_max_new_tokens=calibration_max_new_tokens,
        calibration_percentile=calibration_percentile,
        calibration_margin=calibration_margin,
        retry_token_buckets=retry_token_buckets,
        calibration_seed=calibration_seed,
    )


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
        for record in experiment_records:
            maybe_reset_outfile(record['outfile'])
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

            try:
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
            except ModelUnavailableError as e:
                print(f'Skipping experiment {record["metadata"]["experiment_name"]}: {e}')

        for records in batch_groups.values():
            try:
                principle1_run_network_formation_experiments_batch(records)
            except ModelUnavailableError as e:
                names = ', '.join(r['metadata']['experiment_name'] for r in records)
                print(f'Skipping batched experiments ({names}): {e}')

    if run_analysis:
        for record in experiments_to_analyze:
            if not os.path.exists(record['outfile']):
                print(f'Skipping analysis for {record["outfile"]}: file does not exist (experiment skipped or not run).')
                continue
            principle1_analyze_experiments(record['outfile'], dgr=not record['degrees'])

        non_cot_existing = [f for f in non_cot_outfiles if os.path.exists(f)]
        cot_existing = [f for f in cot_outfiles if os.path.exists(f)]
        for missing in sorted((set(non_cot_outfiles) | set(cot_outfiles)) - (set(non_cot_existing) | set(cot_existing))):
            print(f'Skipping multi-LLM analysis input {missing}: file does not exist.')
        # Single unified comparison: CoT runs are merged in so the baseline
        # model's CoT variant shows up as its own bar in the Model panel
        # (labelled '... -CoT'); CoT records are excluded from the Environment
        # panel because that panel filters on the non-CoT baseline label.
        comparison_existing = non_cot_existing + cot_existing
        if comparison_existing:
            principle1_analyze_experiments_multiple_llms(comparison_existing)

    return {
        'supported_models': supported_models,
        'non_cot_outfiles': non_cot_outfiles,
        'cot_outfiles': cot_outfiles,
        'experiments_to_analyze': experiments_to_analyze,
        'experiment_records': experiment_records,
    }

# --- End Principle 1 preferential-attachment utilities ---

# --- Principle 3 profile-homophily utilities ---
PRINCIPLE3_DEFAULT_PROFILES_FILENAME = 'profiles.jsonl'
PRINCIPLE3_LUCKY_NUMBER_PROFILES_FILENAME = 'profiles_with_lucky_number.jsonl'


def set_principle3_runtime_options(default_profiles_filename='profiles.jsonl', lucky_number_profiles_filename='profiles_with_lucky_number.jsonl', medium_size=26):
    global PRINCIPLE3_DEFAULT_PROFILES_FILENAME, PRINCIPLE3_LUCKY_NUMBER_PROFILES_FILENAME
    global MEDIUM_SIZE, SMALL_SIZE, BIGGER_SIZE
    PRINCIPLE3_DEFAULT_PROFILES_FILENAME = default_profiles_filename
    PRINCIPLE3_LUCKY_NUMBER_PROFILES_FILENAME = lucky_number_profiles_filename
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


def principle3_generate_individuals(n, filename=None, seed=None, lucky_number=False):
    if filename is None:
        filename = PRINCIPLE3_DEFAULT_PROFILES_FILENAME

    rng = random.Random(seed) if seed is not None else random

    profiles = []

    hobbies = ['reading', 'writing', 'cooking']
    colors = ['red', 'orange', 'yellow', 'green']

    locations = ['New York City', 'Boston', 'Washington DC']

    lucky_numbers = [1, 2, 3, 4, 5, 6, 7]

    for i in range(n):
        profile = {
            'name' : i,
            'hobby' : rng.choice(hobbies),
            'favorite color' : rng.choice(colors),
            'location' : rng.choice(locations)
        }

        if lucky_number:
            profile['lucky number'] = rng.choice(lucky_numbers)

        profiles.append(profile)

    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(filename, 'w+') as f:
        [f.write(json.dumps(profile) + '\n') for profile in profiles]

    return filename


def principle3_ensure_profiles(profiles_filename, n, lucky_number=False, seed=0):
    """Generate the profiles file if it is missing (or has fewer than ``n``
    profiles), so a fresh Google Drive output directory does not crash the
    experiments. Existing files are left untouched to keep results reproducible."""
    if os.path.exists(profiles_filename):
        with open(profiles_filename) as f:
            existing = [line for line in f.read().splitlines() if line.strip()]
        if len(existing) >= n:
            return profiles_filename

    principle3_generate_individuals(n, filename=profiles_filename, seed=seed, lucky_number=lucky_number)
    print(f'Generated {n} profiles{" with lucky number" if lucky_number else ""} -> {profiles_filename}')
    return profiles_filename

def principle3_network_growth(n0, temperature=None, model='gpt-5-mini', environment=None, role='friends', method='llm', cot=False, cot_config=None, profiles_filename=None, mutual_acceptance=False):
    if profiles_filename is None:
        profiles_filename = PRINCIPLE3_DEFAULT_PROFILES_FILENAME

    with open(profiles_filename) as f:
        profiles = f.read().splitlines()
        profiles = [json.loads(profile) for profile in profiles]

    profiles = profiles[:n0]

    G = nx.Graph()
    G.add_nodes_from(range(n0))

    Gs = []
    results = []
    total_requests = 0
    num_accepted_requests = 0

    for t in range(n0):

        if method == 'llm':
            result = principle3_select_neighbor(G, t, profiles, temperature, model=model, environment=environment, role=role, cot=cot, cot_config=cot_config, mutual_acceptance=mutual_acceptance)

            if result:
                for r in result:
                    v = r['name']
                    if r.get('accepted', False):
                        G.add_edge(t, v, similarity=r['similarity'])
                        num_accepted_requests += 1
                    total_requests += 1
                results.append(result)
        elif method in ['random', 'homophilous', 'heterophilous']:
            if method == 'random':
                new_nodes = random.sample(list(set(G.nodes()) - set([t])), 4)
            elif method == 'homophilous':
                new_nodes = list(sorted([v for v in G.nodes() if v != t], key=lambda v: len(set(principle3_profile_set(profiles[t])) & principle3_profile_set(profiles[v])), reverse=True))[:4]
            elif method == 'heterophilous':
                new_nodes = list(sorted([v for v in G.nodes() if v != t], key=lambda v: len(set(principle3_profile_set(profiles[t])) & principle3_profile_set(profiles[v]))))[:4]

            for v in new_nodes:
                intersection = list(set(principle3_profile_set(profiles[t])) & principle3_profile_set(profiles[v]))
                union = list(set(principle3_profile_set(profiles[t])) | principle3_profile_set(profiles[v]))
                similarity = len(intersection)
                G.add_edge(t, v, intersection=intersection, union=union, similarity=similarity)
                num_accepted_requests += 1
                total_requests += 1

            results.append([{'name' : v, 'intersection' : intersection, 'union' : union, 'similarity' : similarity} for v in new_nodes])

        Gs.append(G.copy())

    return Gs, results, num_accepted_requests / total_requests * 100

def principle3_profile_set(p):
    temp = []
    for k, v in p.items():
        if k == 'name':
            continue
        else:
            temp.append(v)

    return set(temp)


def principle3_graph_from_stored(graph, profiles_dict=None):
    """Reconstruct a stored principle 3 graph, tolerating both the dict-of-dicts
    ({node: {neighbor: {attrs}}}) and the older dict-of-lists ({node: [neighbors]})
    serializations. Node ids are normalized to strings to match the profile keys.

    When ``profiles_dict`` is given, node attributes are set from it and every edge
    is guaranteed a ``similarity`` attribute (recomputed from the profiles when the
    stored graph did not carry edge attributes)."""
    G = nx.Graph()
    for node, neighbors in graph.items():
        node = str(node)
        G.add_node(node)
        if isinstance(neighbors, dict):
            for neighbor, attrs in neighbors.items():
                G.add_edge(node, str(neighbor), **(attrs if isinstance(attrs, dict) else {}))
        else:
            for neighbor in neighbors:
                G.add_edge(node, str(neighbor))

    if profiles_dict is not None:
        nx.set_node_attributes(G, profiles_dict)
        for u, v, data in G.edges(data=True):
            if 'similarity' not in data:
                profile_u = profiles_dict.get(u)
                profile_v = profiles_dict.get(v)
                if profile_u is not None and profile_v is not None:
                    data['similarity'] = len(
                        principle3_profile_set(profile_u) & principle3_profile_set(profile_v))

    return G


def principle3_build_selection_request(G, t, profiles, environment, role, cot, model):
    candidate_profiles = []
    for v in G.nodes():
        if v != t:
            candidate_profiles.append(profiles[v])

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
    response_schema = build_response_list_schema(candidate_names, max_items=4)
    use_structured_output = not (model.startswith('Qwen/') and cot)
    allowed_names_json = json.dumps(candidate_names, ensure_ascii=False)

    prompt = f"""
    # Task
    {f'You are in a {environment}.' if environment else ''}Your task is to select a person to be {role} with.

    # Profile
    Your profile is given below after chevrons:
    <PROFILE>
    {json.dumps(profiles[t], separators=(',', ':'))}
    </PROFILE>

    # Candidate Profiles
    The cadidate profiles to be friends with are given below after chevrons:

    <PROFILES>
    {json.dumps(candidate_profiles, separators=(',', ':'))}
    </PROFILES>

    # Output
    The output should be given a list of JSON objects with the following structure

    [
        {output_format}, ...
    ]

    # Notes
    * Return exactly one JSON array.
    * The output must be a list of JSON objects ranked in the order of preference.
    * You can make at most 4 selections.
    * Do not explain your reasoning outside the JSON array.
    * Do not write markdown fences.
    * Do not write any text before or after the JSON array.
    * Each value of "name" must be exactly one of these values: {allowed_names_json}
    * Do not rename the person.
    * Do not output labels such as "person 0", "Person 0", or "candidate 0".
    """

    return {
        'prompt': prompt,
        'response_schema': response_schema if use_structured_output else None,
        'candidate_names': set(candidate_names),
        't': t,
        'profiles': profiles,
    }

def principle3_parse_selection_response(ans, request):
    results = first_json_array(ans)
    if not isinstance(results, list):
        raise ValueError('Could not parse a valid JSON array.')

    filtered_results = []
    seen_names = set()
    t = request['t']
    profiles = request['profiles']
    for result in results[:4]:
        if not isinstance(result, dict) or 'name' not in result:
            continue
        normalized_name = normalize_name(result['name'], request['candidate_names'])
        if normalized_name is None or normalized_name in seen_names:
            continue
        seen_names.add(normalized_name)
        result['name'] = normalized_name
        v = result['name']
        result['intersection'] = list(principle3_profile_set(profiles[t]) & principle3_profile_set(profiles[v]))
        result['union'] = list(principle3_profile_set(profiles[t]) | principle3_profile_set(profiles[v]))
        result['similarity'] = len(result['intersection'])
        filtered_results.append(result)

    if not filtered_results:
        raise ValueError('No valid candidate names in response.')
    return filtered_results

def principle3_build_acceptance_request(t, r, profiles, environment, role, cot, model):
    acceptance_response_schema = build_acceptance_response_schema()
    use_acceptance_structured_output = not (model.startswith('Qwen/') and cot)
    prompt = f"""
            # Task
            {f'You are in a {environment}.' if environment else ''}. You receive a request to be {role} with a person.

            # Profile
            Your profile is given below after chevrons:
            <PROFILE>
            {json.dumps(profiles[t], separators=(',', ':'))}
            </PROFILE>

            # Candidate Profile
            The candidate profile to be friends with is given below after chevrons:
            <PROFILE>
            {json.dumps(profiles[r['name']], separators=(',', ':'))}
            </PROFILE>

            # Output
            The output should be a JSON object with the following structure

            {{
                "accept" : true/false,
                "reason" : reason for accepting or rejecting the request
            }}

            # Notes
            * Return exactly one JSON object.
            * You can only accept or reject the request.
            * Do not write markdown fences.
            * Do not write any text before or after the JSON object.
            """
    return {
        'prompt': prompt,
        'response_schema': acceptance_response_schema if use_acceptance_structured_output else None,
    }

def principle3_parse_acceptance_response(ans):
    results = first_json_object(ans)
    if not isinstance(results, dict) or 'accept' not in results:
        raise ValueError('Could not parse a valid JSON object with an accept field.')
    return bool(results['accept']), results.get('reason', '')

def principle3_select_neighbor(G, t, profiles, temperature, model, environment, role, cot, cot_config=None, mutual_acceptance=False):
    request = principle3_build_selection_request(G, t, profiles, environment, role, cot, model)
    filtered_results = []
    for i in range(10):
        try:
            ans = get_response(request['prompt'], model, temperature, response_schema=request['response_schema'], cot=cot, cot_config=cot_config)
            filtered_results = principle3_parse_selection_response(ans, request)
            break
        except Exception as e:
            print(e)

    for r in filtered_results:
        if mutual_acceptance:
            acceptance_request = principle3_build_acceptance_request(t, r, profiles, environment, role, cot, model)

            for i in range(10):
                try:
                    ans = get_response(acceptance_request['prompt'], model, temperature, response_schema=acceptance_request['response_schema'], cot=cot, cot_config=cot_config)
                    r['accepted'], r['acceptance_reason'] = principle3_parse_acceptance_response(ans)
                    break
                except Exception as e:
                    print(e)
            else:
                r['accepted'] = False
                r['acceptance_reason'] = 'Could not parse mutual acceptance response'
        else:
            r['accepted'] = True
            r['acceptance_reason'] = 'Mutual acceptance is not enabled'

    print(f'Node: {t}, Results: {filtered_results}')

    return filtered_results

def principle3_load_profiles(profiles_filename, n):
    with open(profiles_filename) as f:
        profiles = f.read().splitlines()
        profiles = [json.loads(profile) for profile in profiles]
    return profiles[:n]

def principle3_initialize_growth_state(n, temperature, experiment_record):
    profiles = principle3_load_profiles(experiment_record['profiles_filename'], n)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    return {
        'n': n,
        'temperature': temperature,
        'temperature_label': 'default' if temperature is None else temperature,
        'model': experiment_record['model'],
        'environment': experiment_record['environment'],
        'role': experiment_record['role'],
        'cot': experiment_record['cot'],
        'cot_config': experiment_record.get('cot_config'),
        'profiles_filename': experiment_record['profiles_filename'],
        'mutual_acceptance': experiment_record['mutual_acceptance'],
        'outfile': experiment_record['outfile'],
        'metadata': experiment_record['metadata'],
        'G': G,
        'profiles': profiles,
        't': 0,
        'graphs': [],
        'reasons': [],
        'total_requests': 0,
        'num_accepted_requests': 0,
    }

def principle3_pending_growth_states_for_experiment(experiment_record):
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
    for n in range(parameters['n_min'], parameters['n_max'] + 1, parameters['n_step']):
        for i in range(parameters['num_simulations']):
            for temperature in temperatures:
                temperature_label = 'default' if temperature is None else temperature
                if (n, i, temperature_label) in saved_scenarios:
                    print(f'Skipping simulation for n={n}, i={i}, temperature={temperature_label}')
                    continue
                print(f'Queueing simulation for n={n}, i={i}, temperature={temperature_label}, outfile={outfile}')
                state = principle3_initialize_growth_state(n, temperature, experiment_record)
                state['simulation'] = i
                states.append(state)
    return states

def principle3_apply_selection_results(state, filtered_results):
    t = state['t']
    if filtered_results:
        for r in filtered_results:
            v = r['name']
            if r.get('accepted', False):
                state['G'].add_edge(t, v, similarity=r['similarity'])
                state['num_accepted_requests'] += 1
            state['total_requests'] += 1
        state['reasons'].append(filtered_results)

    state['graphs'].append(state['G'].copy())
    state['t'] += 1

def principle3_write_growth_state(state):
    mutual_acceptance_probability = 0
    if state['total_requests']:
        mutual_acceptance_probability = state['num_accepted_requests'] / state['total_requests'] * 100

    temp = {
        'n' : state['n'],
        'temperature' : state['temperature_label'],
        'simulation' : state['simulation'],
        'graphs' : [nx.to_dict_of_dicts(G) for G in state['graphs']],
        'reasons' : state['reasons'],
        'mutual_acceptance_probability' : mutual_acceptance_probability,
        'model' : state['model'],
        'environment' : state['environment'] if state['environment'] is not None else 'Baseline',
        'role' : state['role'],
        'cot' : state['cot'],
        'profiles_filename' : state['profiles_filename'],
        'mutual_acceptance' : state['mutual_acceptance'],
    }
    if state['metadata']:
        temp.update(state['metadata'])
    with open(state['outfile'], 'a+') as f:
        f.write(json.dumps(temp) + '\n')
        f.flush()

def principle3_add_default_acceptance(filtered_results):
    for r in filtered_results:
        r['accepted'] = True
        r['acceptance_reason'] = 'Mutual acceptance is not enabled'

def principle3_run_network_formation_experiments_batch(experiment_records):
    if not experiment_records:
        return

    model = experiment_records[0]['model']
    cot = experiment_records[0]['cot']
    cot_config = experiment_records[0].get('cot_config')
    active_states = []
    for experiment_record in experiment_records:
        active_states.extend(principle3_pending_growth_states_for_experiment(experiment_record))

    if not active_states:
        print(f'All batched simulations already completed for {model}. Skipping inference.')
        return

    print(f'Running {len(active_states)} batched simulations for {model}, cot={cot}')
    while active_states:
        requests_by_temperature = collections.defaultdict(list)
        for state in active_states:
            print(f'Selecting neighbors for node {state["t"]} in {state["metadata"]["experiment_name"]}, simulation={state["simulation"]}')
            request = principle3_build_selection_request(
                state['G'],
                state['t'],
                state['profiles'],
                state['environment'],
                state['role'],
                state['cot'],
                state['model'],
            )
            requests_by_temperature[state['temperature']].append((state, request))

        selections_by_state_id = {}
        for temperature, batch_items in requests_by_temperature.items():
            remaining = list(batch_items)
            for attempt in range(10):
                if not remaining:
                    break

                attempt_cot_config = retry_cot_config(cot_config, attempt) if cot else cot_config
                if cot and attempt_cot_config != cot_config:
                    print(f'Retry attempt {attempt + 1}: max_new_tokens={attempt_cot_config.get("max_new_tokens")}, qwen_enable_thinking={attempt_cot_config.get("qwen_enable_thinking")}')
                prompts = [request['prompt'] for _, request in remaining]
                response_schemas = [request['response_schema'] for _, request in remaining]
                answers = get_responses(
                    prompts,
                    model,
                    temperature=temperature,
                    response_schemas=response_schemas,
                    cot=cot,
                    cot_config=attempt_cot_config,
                )

                next_remaining = []
                for (state, request), ans in zip(remaining, answers):
                    try:
                        filtered_results = principle3_parse_selection_response(ans, request)
                        selections_by_state_id[id(state)] = filtered_results
                    except Exception as e:
                        print(e)
                        next_remaining.append((state, request))
                remaining = next_remaining

            for state, _ in remaining:
                selections_by_state_id[id(state)] = []

        acceptance_items_by_temperature = collections.defaultdict(list)
        for state in active_states:
            filtered_results = selections_by_state_id.get(id(state), [])
            if state['mutual_acceptance']:
                for r in filtered_results:
                    request = principle3_build_acceptance_request(state['t'], r, state['profiles'], state['environment'], state['role'], state['cot'], state['model'])
                    acceptance_items_by_temperature[state['temperature']].append((state, r, request))
            else:
                principle3_add_default_acceptance(filtered_results)

        for temperature, batch_items in acceptance_items_by_temperature.items():
            remaining = list(batch_items)
            for attempt in range(10):
                if not remaining:
                    break

                attempt_cot_config = retry_cot_config(cot_config, attempt) if cot else cot_config
                if cot and attempt_cot_config != cot_config:
                    print(f'Retry attempt {attempt + 1}: max_new_tokens={attempt_cot_config.get("max_new_tokens")}, qwen_enable_thinking={attempt_cot_config.get("qwen_enable_thinking")}')
                prompts = [request['prompt'] for _, _, request in remaining]
                response_schemas = [request['response_schema'] for _, _, request in remaining]
                answers = get_responses(
                    prompts,
                    model,
                    temperature=temperature,
                    response_schemas=response_schemas,
                    cot=cot,
                    cot_config=attempt_cot_config,
                )

                next_remaining = []
                for (state, r, request), ans in zip(remaining, answers):
                    try:
                        r['accepted'], r['acceptance_reason'] = principle3_parse_acceptance_response(ans)
                    except Exception as e:
                        print(e)
                        next_remaining.append((state, r, request))
                remaining = next_remaining

            for _, r, _ in remaining:
                r['accepted'] = False
                r['acceptance_reason'] = 'Could not parse mutual acceptance response'

        next_active_states = []
        for state in active_states:
            filtered_results = selections_by_state_id.get(id(state), [])
            print(f'Node: {state["t"]}, Results: {filtered_results}')
            principle3_apply_selection_results(state, filtered_results)
            if state['t'] >= state['n']:
                principle3_write_growth_state(state)
            else:
                next_active_states.append(state)
        active_states = next_active_states

def principle3_run_network_formation_experiment(n_min, n_max, n_step, num_simulations, outfile, temperatures=None, method='llm', model='gpt-5-mini', environment=None, role='friends', cot=False, cot_config=None, profiles_filename=None, mutual_acceptance=False, metadata=None):
    if profiles_filename is None:
        profiles_filename = PRINCIPLE3_DEFAULT_PROFILES_FILENAME

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

                    Gs, reasons, mutual_acceptance_probability = principle3_network_growth(n, temperature=temperature, method=method, model=model, environment=environment, role=role, cot=cot, cot_config=cot_config, profiles_filename=profiles_filename, mutual_acceptance=mutual_acceptance)

                    temp = {
                        'n' : n,
                        'temperature' : temperature_label,
                        'simulation' : i,
                        'graphs' : [nx.to_dict_of_dicts(G) for G in Gs],
                        'reasons' : reasons,
                        'mutual_acceptance_probability' : mutual_acceptance_probability,
                        'model' : model,
                        'environment' : environment if environment is not None else 'Baseline',
                        'role' : role,
                        'cot' : cot,
                        'profiles_filename' : profiles_filename,
                        'mutual_acceptance' : mutual_acceptance,
                    }
                    if metadata:
                        temp.update(metadata)

                    f.write(json.dumps(temp) + '\n')
                    f.flush()

                if method != 'llm':
                    break

    f.close()

def principle3_draw_graph(G, ax, communities=None, palette=None, use_netgraph=True, ego_network=False):

    if not use_netgraph:
        pos = nx.spring_layout(G, weight='similarity')

        if communities:
            for i, community in enumerate(communities):
                nx.draw_networkx_nodes(G, pos, nodelist=list(community), node_size=20, node_color=palette[i], ax=ax)
        else:
            nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), node_size=20, node_color='#d35400', ax=ax)

        edge_widths = [1 + G.edges[u, v]['similarity'] for (u, v) in G.edges()]

        nx.draw_networkx_edges(G, pos, edgelist=list(G.edges()), width=edge_widths, edge_color='#34495e', alpha=0.5, ax=ax)
    elif use_netgraph and not ego_network:
        node2community = {node: i for i, community in enumerate(communities) for node in community}
        node_color = {node : palette[node2community[node]] for node in G.nodes()}
        edge_widths = {(u, v) : (1 + G.edges[u, v]['similarity']) * 0.5 for (u, v) in G.edges()}

        # netgraph.Graph(G, node_layout='community', node_color=node_color, node_layout_kwargs=dict(node_to_community=node2community), node_size=2.5, edge_width=edge_widths, edge_color='#34495e', edge_layout='bundled', edge_layout_kwargs=dict(k=2000), ax=ax)
        netgraph.Graph(G, node_layout='community', node_color=node_color, node_layout_kwargs=dict(node_to_community=node2community), node_size=2.5, edge_width=edge_widths, edge_color='#34495e', ax=ax)

    elif use_netgraph and ego_network:
        ego_net = nx.ego_graph(G, list(G.nodes())[0], radius=1)

        node2community = {node: i for i, community in enumerate(communities) for node in community}

        node_color = {node : palette[node2community[node]] for node in ego_net.nodes()}
        node_size = {}

        for node in ego_net.nodes():
            if node == list(G.nodes())[0]:
                node_size[node] = 5
            else:
                node_size[node] = 2.5

        edge_widths = {(u, v) : (1 + ego_net.edges[u, v]['similarity']) * 0.5 for (u, v) in ego_net.edges()}
        netgraph.Graph(ego_net, node_layout='community', node_color=node_color, node_layout_kwargs=dict(node_to_community=node2community), node_size=node_size, edge_width=edge_widths, edge_color='#34495e', ax=ax)

    ax.set_axis_off()

def principle3_analyze_experiments(filename):
    os.makedirs('figures/principle_3', exist_ok=True)

    palette = ['#d35400', '#34495e', '#2980b9', '#e67e22', '#f1c40f', '#7f8c8d', '#27ae60', '#16a085', '#bdc3c7', '#1abc9c', '#2ecc71', '#3498db', '#9b59b6', '#8e44ad', '#ecf0f1']

    with open(filename) as f:
        lines = f.read().splitlines()

    data = []


    for line in lines:
        data.append(json.loads(line))

    edge_similarity_distributions = { 'homophilous' : collections.defaultdict(list), 'random' : collections.defaultdict(list), 'heterophilous' : collections.defaultdict(list), 'llm' : collections.defaultdict(list) }
    wasserstein_distance = { 'homophilous' : collections.defaultdict(list), 'random' : collections.defaultdict(list), 'heterophilous' : collections.defaultdict(list) }
    louvain_communities = { 'homophilous' : collections.defaultdict(list), 'random' : collections.defaultdict(list), 'heterophilous' : collections.defaultdict(list), 'llm' : collections.defaultdict(list) }
    louvain_modularity = { 'homophilous' : collections.defaultdict(list), 'random' : collections.defaultdict(list), 'heterophilous' : collections.defaultdict(list), 'llm' : collections.defaultdict(list) }
    location_assortativities = { 'homophilous' : collections.defaultdict(list), 'random' : collections.defaultdict(list), 'heterophilous' : collections.defaultdict(list), 'llm' : collections.defaultdict(list)}
    favorite_color_assortativities = { 'homophilous' : collections.defaultdict(list), 'random' : collections.defaultdict(list), 'heterophilous' : collections.defaultdict(list), 'llm' : collections.defaultdict(list)}
    hobby_assortativities = { 'homophilous' : collections.defaultdict(list), 'random' : collections.defaultdict(list), 'heterophilous' : collections.defaultdict(list), 'llm' : collections.defaultdict(list)}

    final_graphs = collections.defaultdict(list)

    with open(PRINCIPLE3_DEFAULT_PROFILES_FILENAME) as f:
        profiles = f.read().splitlines()
        profiles = [json.loads(profile) for profile in profiles]

    profiles_dict = {str(profile['name']) : profile for profile in profiles}

    for d in data:
        Gs = []
        for graph in d['graphs']:
            G = principle3_graph_from_stored(graph, profiles_dict)

            G.remove_edges_from(nx.selfloop_edges(G))
            # G.remove_nodes_from(list(nx.isolates(G)))

            Gs.append(G)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        louvain_communities['llm'][d['n'], d['temperature']].append(nx.algorithms.community.louvain_communities(Gs[-1], weight='similarity'))
        louvain_modularity['llm'][d['n'], d['temperature']].append(nx.algorithms.community.modularity(Gs[-1], louvain_communities['llm'][d['n'], d['temperature']][-1], weight='similarity'))


        final_graphs[d['n'], d['temperature']].append((Gs[-1], louvain_communities['llm'][d['n'], d['temperature']][-1]))


        G_homophilous = principle3_network_growth(d['n'], d['temperature'], method='homophilous')[0][-1]
        G_heterophilous = principle3_network_growth(d['n'], d['temperature'], method='heterophilous')[0][-1]
        G_random = principle3_network_growth(d['n'], d['temperature'], method='random')[0][-1]

        nx.set_node_attributes(G_homophilous, {int(profile['name']) : profile for profile in profiles})
        nx.set_node_attributes(G_heterophilous, {int(profile['name']) : profile for profile in profiles})
        nx.set_node_attributes(G_random, {int(profile['name']) : profile for profile in profiles})

        ax[0].set_title(f'Temperature = {d["temperature"]}')

        principle3_draw_graph(Gs[-1], ax=ax[0], communities=louvain_communities['llm'][d['n'], d['temperature']][-1], palette=palette)

        edge_similarity_distribution = [G.edges[u, v]['similarity'] for (u, v) in Gs[-1].edges()]
        edge_similarity_distribution_homophilous = [G_homophilous.edges[u, v]['similarity'] for (u, v) in G_homophilous.edges()]
        edge_similarity_distribution_heterophilous = [G_heterophilous.edges[u, v]['similarity'] for (u, v) in G_heterophilous.edges()]
        edge_similarity_distribution_random = [G_random.edges[u, v]['similarity'] for (u, v) in G_random.edges()]

        wasserstein_distance['homophilous'][d['n'], d['temperature']].append(stats.wasserstein_distance(edge_similarity_distribution, edge_similarity_distribution_homophilous))
        wasserstein_distance['heterophilous'][d['n'], d['temperature']].append(stats.wasserstein_distance(edge_similarity_distribution, edge_similarity_distribution_heterophilous))
        wasserstein_distance['random'][d['n'], d['temperature']].append(stats.wasserstein_distance(edge_similarity_distribution, edge_similarity_distribution_random))

        print(f'Temperature: {d["temperature"]} T-test Test Homophilous: {stats.ttest_ind(edge_similarity_distribution, edge_similarity_distribution_homophilous, equal_var=False, alternative="less")}')
        print(f'Temperature: {d["temperature"]} T-test Test Random: {stats.ttest_ind(edge_similarity_distribution, edge_similarity_distribution_random, equal_var=False, alternative="two-sided")}')


        edge_similarity_distributions['homophilous'][d['n'], d['temperature']].extend(edge_similarity_distribution_homophilous)
        edge_similarity_distributions['heterophilous'][d['n'], d['temperature']].extend(edge_similarity_distribution_heterophilous)
        edge_similarity_distributions['random'][d['n'], d['temperature']].extend(edge_similarity_distribution_random)
        edge_similarity_distributions['llm'][d['n'], d['temperature']].extend(edge_similarity_distribution)

        louvain_communities['homophilous'][d['n'], d['temperature']].append(nx.algorithms.community.louvain_communities(G_homophilous, weight='similarity'))
        louvain_communities['random'][d['n'], d['temperature']].append(nx.algorithms.community.louvain_communities(G_random, weight='similarity'))

        louvain_modularity['homophilous'][d['n'], d['temperature']].append(nx.algorithms.community.modularity(G_homophilous, louvain_communities['homophilous'][d['n'], d['temperature']][-1], weight='similarity'))
        louvain_modularity['random'][d['n'], d['temperature']].append(nx.algorithms.community.modularity(G_random, louvain_communities['random'][d['n'], d['temperature']][-1], weight='similarity'))

        location_assortativities['homophilous'][d['n'], d['temperature']].append(nx.attribute_assortativity_coefficient(G_homophilous, 'location'))
        location_assortativities['heterophilous'][d['n'], d['temperature']].append(nx.attribute_assortativity_coefficient(G_heterophilous, 'location'))
        location_assortativities['random'][d['n'], d['temperature']].append(nx.attribute_assortativity_coefficient(G_random, 'location'))
        location_assortativities['llm'][d['n'], d['temperature']].append(nx.attribute_assortativity_coefficient(Gs[-1], 'location'))

        favorite_color_assortativities['homophilous'][d['n'], d['temperature']].append(nx.attribute_assortativity_coefficient(G_homophilous, 'favorite color'))
        favorite_color_assortativities['heterophilous'][d['n'], d['temperature']].append(nx.attribute_assortativity_coefficient(G_heterophilous, 'favorite color'))
        favorite_color_assortativities['random'][d['n'], d['temperature']].append(nx.attribute_assortativity_coefficient(G_random, 'favorite color'))
        favorite_color_assortativities['llm'][d['n'], d['temperature']].append(nx.attribute_assortativity_coefficient(Gs[-1], 'favorite color'))

        hobby_assortativities['homophilous'][d['n'], d['temperature']].append(nx.attribute_assortativity_coefficient(G_homophilous, 'hobby'))
        hobby_assortativities['heterophilous'][d['n'], d['temperature']].append(nx.attribute_assortativity_coefficient(G_heterophilous, 'hobby'))
        hobby_assortativities['random'][d['n'], d['temperature']].append(nx.attribute_assortativity_coefficient(G_random, 'hobby'))
        hobby_assortativities['llm'][d['n'], d['temperature']].append(nx.attribute_assortativity_coefficient(Gs[-1], 'hobby'))


        sns.histplot(edge_similarity_distribution, ax=ax[1], label='LLM', color='#d35400', binwidth=0.45, discrete=True, stat='density')
        ax[1].xaxis.set_major_locator(mticker.MultipleLocator(1))

        ax[1].set_xlabel('Number of Common Attributes')
        ax[1].set_ylabel('Probability of Edge Creation')
        ax[1].set_ylim(0, 0.75)
        ax[1].legend()

        fig.tight_layout()
        fig.savefig(f'figures/principle_3/principle_3_profiles_{d["n"]}_{d["simulation"]}_{d["temperature"]}.png', dpi=300)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))


    ax[0].set_xlabel('Temperature')
    ax[0].set_ylabel('Wasserstein Distance')


    wasserstein_distance_means = { 'homophilous' : [], 'heterophilous' : [], 'random' : [] }
    wasserstein_distance_stds = { 'homophilous' : [],  'heterophilous' : [], 'random' : [] }

    for method, color in zip(['homophilous', 'heterophilous', 'random'], palette[:3]):
        for i, k in enumerate(sorted(wasserstein_distance[method].keys())):
            v = np.array(wasserstein_distance[method][k])
            mean = v.mean()
            std = v.std()

            wasserstein_distance_means[method].append(mean)
            wasserstein_distance_stds[method].append(std)

    for method, color in zip(['homophilous', 'random'], palette[:3]):

        ax[0].plot(np.arange(len(wasserstein_distance[method])), wasserstein_distance_means[method], label=method.capitalize(), marker='o', color=color)
        ax[0].fill_between(np.arange(len(wasserstein_distance[method])), np.array(wasserstein_distance_means[method]) - 1.96 * np.array(wasserstein_distance_stds[method]) / np.sqrt(len(wasserstein_distance[method])), np.array(wasserstein_distance_means[method]) + 1.96 * np.array(wasserstein_distance_stds[method]) / np.sqrt(len(wasserstein_distance[method])), alpha=0.2, color=color)

    ax[0].set_xticks(np.arange(len(wasserstein_distance['homophilous'])))
    ax[0].set_xticklabels([f'{k[1]}' for k in sorted(wasserstein_distance['homophilous'].keys())])
    ax[0].legend()
    ax[0].set_ylabel(f'Wasserstein Distance')


    objs = []

    for i, k in enumerate(sorted(edge_similarity_distributions['llm'].keys())):
        objs.append(pd.DataFrame.from_dict({'Number of Common Attributes' : edge_similarity_distributions['llm'][k], 'Method' : f'{k[1]}'}))

        if i == len(edge_similarity_distributions['llm'].keys()) - 1:
            for method in ['homophilous', 'random']:
                objs.append(pd.DataFrame.from_dict({'Number of Common Attributes' : edge_similarity_distributions[method][k], 'Method' : method.capitalize()}))


    df = pd.concat(axis=0, ignore_index=True, objs=objs)

    sns.histplot(
        data=df, x='Number of Common Attributes', hue='Method', multiple='dodge', palette=palette,
        bins=range(4), ax=ax[1], discrete=True, shrink=0.8, stat='probability', common_norm=False
    )

    sns.move_legend(ax[1], bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)

    ax[1].set_ylabel('Probability of Edge Creation')

    fig.tight_layout()

    fig.savefig('figures/principle_3/principle_3_profiles_overall.pdf')

    fig, ax = plt.subplots(1, len(final_graphs), figsize=(5 * (len(final_graphs)), 5), squeeze=False)
    ax = ax[0]

    for i, (k, v) in enumerate(sorted(final_graphs.items())):
        G = v[0][0]
        communities = v[0][1]
        ax[i].set_title(f"Temperature = {k[-1]}", fontsize=MEDIUM_SIZE)
        principle3_draw_graph(G, ax=ax[i], communities=communities, palette=palette)


    # sns.histplot(
    #     data=df, x='Number of Common Attributes', hue='Method', multiple='dodge',
    #     bins=range(4), ax=ax[-1], discrete=True, shrink=0.8, stat='probability', common_norm=False, palette=palette
    # )

    # sns.barplot(
    #     data=df, y='Number of Common Attributes', x='Method', ax=ax[-1], palette=palette
    # )

    # plt.xticks(fontsize=SMALL_SIZE, rotation=90)


    # sns.move_legend(ax[-1], bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)

    # ax[-1].set_ylabel('# Common Attributes')

    fig.tight_layout()

    fig.savefig('figures/principle_3/principle_3_profiles_final_graphs.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].set_ylabel('Louvain Modularity')
    ax[1].set_ylabel('Num. of Communities')
    ax[2].set_ylabel('Community Size')

    ax[0].spines[['right', 'top']].set_visible(False)
    ax[1].spines[['right', 'top']].set_visible(False)
    ax[2].spines[['right', 'top']].set_visible(False)

    j = 0

    for method in ['llm']:
        for i, k in enumerate(sorted(louvain_modularity[method].keys())):
            if method != 'llm' and i != 0:
                continue

            modularities = np.array(louvain_modularity[method][k])
            number_of_communities = np.array([len(c) for c in louvain_communities[method][k]])
            community_sizes = np.array([len(v) for c in louvain_communities[method][k] for v in c])


            if method == 'llm':
                label = f'{k[-1]}'
            else:
                label = method.capitalize()

            mean_modularity = modularities.mean()
            ci_modularity = 1.96 * modularities.std() / np.sqrt(len(modularities))

            mean_number_of_communities = number_of_communities.mean()
            ci_number_of_communities = 1.96 * number_of_communities.std() / np.sqrt(len(number_of_communities))

            mean_community_sizes = community_sizes.mean()
            ci_community_sizes = 1.96 * community_sizes.std() / np.sqrt(len(community_sizes))

            # ax[0].bar(i, mean_modularity, yerr=ci_modularity, label=label, color=palette[j])
            # ax[1].bar(i, mean_number_of_communities, yerr=ci_number_of_communities, label=label, color=palette[j])
            # ax[2].bar(i, mean_community_sizes, yerr=ci_community_sizes, label=label, color=palette[j])


            ax[0].errorbar(k[-1], mean_modularity, yerr=ci_modularity, fmt='o', label=label, color=palette[j], capsize=5)
            ax[1].errorbar(k[-1], mean_number_of_communities, yerr=ci_number_of_communities, fmt='o', label=label, color=palette[j], capsize=5)
            ax[2].errorbar(k[-1], mean_community_sizes, yerr=ci_community_sizes, fmt='o', label=label, color=palette[j], capsize=5)




            # sns.distplot(modularities, hist=False, ax=ax[0], label=label, color=palette[j])
            # sns.distplot(number_of_communities, hist=False, ax=ax[1], label=label, color=palette[j])
            # sns.distplot(community_sizes, hist=False, ax=ax[2], label=label, color=palette[j])



            j += 1

            for r, k_prime in enumerate(sorted(louvain_modularity[method].keys())):
                if k_prime[-1] < k[-1]:

                    modularities_prime = np.array(louvain_modularity[method][k_prime])
                    number_of_communities_prime = np.array([len(c) for c in louvain_communities[method][k_prime]])
                    community_sizes_prime = np.array([len(v) for c in louvain_communities[method][k_prime] for v in c])

                    print(f'T-test Temperatures: {k[-1]}, {k_prime[-1]} Modularity: {stats.ttest_ind(modularities, modularities_prime, equal_var=False, alternative="greater")}')
                    print(f'T-test Temperatures: {k[-1]}, {k_prime[-1]} Number of Communities: {stats.ttest_ind(number_of_communities, number_of_communities_prime, equal_var=False, alternative="greater")}')
                    print(f'T-test Temperatures: {k[-1]}, {k_prime[-1]} Community Sizes: {stats.ttest_ind(community_sizes, community_sizes_prime, equal_var=False, alternative="less")}')


    # ax[0].legend(loc='upper right')

    # fig.supxlabel('Temperature', fontsize=MEDIUM_SIZE)

    fig.tight_layout()

    fig.savefig('figures/principle_3/principle_3_profiles_louvain_modularity.pdf', bbox_inches='tight')

    assortativities_records = []

    for method in ['llm', 'random']:
        for i, k in enumerate(sorted(location_assortativities[method].keys())):
            if method == 'llm':
                label = f'{k[-1]}'
            else:
                label = method.capitalize()[:4]

            for v in location_assortativities[method][k]:
                assortativities_records.append({
                    'Method' : label,
                    'Location' : v,
                })

            for v in favorite_color_assortativities[method][k]:
                assortativities_records.append({
                    'Method' : label,
                    'Color' : v,
                })

            for v in hobby_assortativities[method][k]:
                assortativities_records.append({
                    'Method' : label,
                    'Hobby' : v,
                })


    df = pd.DataFrame.from_records(assortativities_records)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)


    fig.suptitle('Attribute Assortativities', fontsize=MEDIUM_SIZE)

    num_tests = 3

    alpha = 0.1
    errorbar_ci = 100 - alpha / num_tests

    sns.barplot(data=df, x='Method', y='Location', ax=ax[0], palette=palette, errorbar=('ci', errorbar_ci))
    sns.barplot(data=df, x='Method', y='Color', ax=ax[1], palette=palette, errorbar=('ci', errorbar_ci))
    sns.barplot(data=df, x='Method', y='Hobby', ax=ax[2], palette=palette, errorbar=('ci', errorbar_ci))




    for i in range(3):
        for tick in ax[i].get_xticklabels():
            tick.set_fontsize(SMALL_SIZE)


        ax[i].set_xlabel('')
        ax[i].spines[['right', 'top']].set_visible(False)


    fig.tight_layout()

    fig.savefig('figures/principle_3/principle_3_assortativities.pdf', bbox_inches='tight')

def principle3_get_table(filenames, sfx='', attributes=['Location', 'Favorite Color', 'Hobby'], environments=True, profiles_filename=None, mutual_acceptance=False, communities=True):
    if profiles_filename is None:
        profiles_filename = PRINCIPLE3_DEFAULT_PROFILES_FILENAME

    os.makedirs('figures', exist_ok=True)
    os.makedirs('tables', exist_ok=True)

    records_assortativities = []
    records_communities = []

    num_tests = len(attributes)

    for filename in filenames:
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

        with open(profiles_filename) as f:
            profiles = f.read().splitlines()
            profiles = [json.loads(profile) for profile in profiles]

        profiles_dict = {str(profile['name']) : profile for profile in profiles}

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
            for graph in d['graphs']:
                G = principle3_graph_from_stored(graph, profiles_dict)

                G.remove_edges_from(nx.selfloop_edges(G))
                G.remove_nodes_from(list(nx.isolates(G)))

                Gs.append(G)

            # import pdb; pdb.set_trace()

            louvain_communities = nx.algorithms.community.louvain_communities(Gs[-1], weight='similarity')

            louvain_modularity = nx.algorithms.community.modularity(Gs[-1], louvain_communities, weight='similarity')
            louvain_num_communities = len(louvain_communities)
            louvain_community_average_size = np.mean([len(c) for c in louvain_communities])

            location_assortativity = nx.attribute_assortativity_coefficient(Gs[-1], 'location')
            favorite_color_assortativity = nx.attribute_assortativity_coefficient(Gs[-1], 'favorite color')
            hobby_assortativity = nx.attribute_assortativity_coefficient(Gs[-1], 'hobby')
            lucky_number_assortativity = nx.attribute_assortativity_coefficient(Gs[-1], 'lucky number')

            mutual_acceptance_probability = d.get('mutual_acceptance_probability', 100)

            G_random = principle3_network_growth(d['n'], d['temperature'], method='random', model='', environment='', role='')[0][-1]

            nx.set_node_attributes(G_random, {int(profile['name']) : profile for profile in profiles})

            location_assortativity_random = nx.attribute_assortativity_coefficient(G_random, 'location')
            favorite_color_assortativity_random = nx.attribute_assortativity_coefficient(G_random, 'favorite color')
            hobby_assortativity_random = nx.attribute_assortativity_coefficient(G_random, 'hobby')
            lucky_number_assortativity_random = nx.attribute_assortativity_coefficient(G_random, 'lucky number')



            record = {
                'Temperature' : d['temperature'],
                'Model' : model,
                'Environment' : environment,
                'Location' : location_assortativity,
                'Favorite Color' : favorite_color_assortativity,
                'Hobby' : hobby_assortativity,
                'Lucky Number' : lucky_number_assortativity,
                'Modularity' : louvain_modularity,
                'Mutual Acceptance Probability' : mutual_acceptance_probability,
            }

            record_random = {
                'Temperature' : d['temperature'],
                'Model' : 'Random',
                'Environment' : 'Baseline',
                'Location' : location_assortativity_random,
                'Favorite Color' : favorite_color_assortativity_random,
                'Lucky Number' : lucky_number_assortativity_random,
                'Hobby' : hobby_assortativity_random,
                'Mutual Acceptance Probability' : 100,
            }

            records_assortativities.append(record)
            records_assortativities.append(record_random)

            record_communities = {
                'Temperature' : d['temperature'],
                'Model' : model,
                'Environment' : environment,
                'Modularity' : louvain_modularity,
                'Number of Communities' : louvain_num_communities,
                'Average Community Size' : louvain_community_average_size
            }

            records_communities.append(record_communities)


    df = pd.DataFrame(records_assortativities)

    # average over simulations
    df_groupped = df.groupby(['Model', 'Environment', 'Temperature']).mean().reset_index()

    # do t-test with random
    for temperature in df['Temperature'].unique():
        for metric in attributes:
            for model in df['Model'].unique():
                if model == 'Random':
                    continue
                data = df[(df['Temperature'] == temperature) & (df['Model'] == model)][metric].values
                data_random = df[(df['Temperature'] == temperature) & (df['Model'] == 'Random')][metric].values

                t, p = stats.ttest_ind(data, data_random, equal_var=False)

                # Get stars
                if p < 0.001:
                    p = '***'
                elif p < 0.01:
                    p = '**'
                elif p < 0.05:
                    p = '*'
                else:
                    p = ''

                if p != '':
                    df_groupped.loc[(df_groupped['Temperature'] == temperature) & (df_groupped['Model'] == model), metric] = f'{df_groupped.loc[(df_groupped["Temperature"] == temperature) & (df_groupped["Model"] == model), metric].values[0]:.3f} ({p})'

    df_groupped = df_groupped[df_groupped['Model'] != 'Random']

    df_groupped.sort_values(['Model', 'Environment', 'Temperature'], inplace=True)

    df_groupped.to_csv(f'tables/principle_3_assortativities{sfx}.csv', index=False)
    df_groupped.to_latex(f'tables/principle_3_assortativities{sfx}.tex', index=False, escape=False, float_format="%.2f")

    df_communities = pd.DataFrame(records_communities)

    df_communities_groupped = df_communities.groupby(['Model', 'Environment', 'Temperature']).mean().reset_index()

    # do t-test of Louvain modularity with 0
    for temperature in df_communities['Temperature'].unique():
        for model in df_communities['Model'].unique():
            if model == 'Random':
                continue
            data = df_communities[(df_communities['Temperature'] == temperature) & (df_communities['Model'] == model)]['Modularity'].values

            t, p = stats.ttest_1samp(data, 0, alternative='greater')

            # Get stars
            if p < 0.001:
                p = '***'
            elif p < 0.01:
                p = '**'
            elif p < 0.05:
                p = '*'
            else:
                p = ''

            if p != '':
                df_communities_groupped.loc[(df_communities_groupped['Temperature'] == temperature) & (df_communities_groupped['Model'] == model), 'Modularity'] = f'{df_communities_groupped.loc[(df_communities_groupped["Temperature"] == temperature) & (df_communities_groupped["Model"] == model), "Modularity"].values[0]:.3f} ({p})'

    df_communities_groupped = df_communities_groupped[df_communities_groupped['Model'] != 'Random']

    df_communities_groupped.sort_values(['Model', 'Environment', 'Temperature'], inplace=True)

    df_communities_groupped.to_csv(f'tables/principle_3_communities{sfx}.csv', index=False)
    df_communities_groupped.to_latex(f'tables/principle_3_communities{sfx}.tex', index=False, escape=False, float_format="%.2f")

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

    rename_env = {
        'school' : 'Sch',
        'work' : 'Wrk',
        'community' : 'Cmn',
        'school_cot' : 'Sch',
        'work_cot' : 'Wrk',
        'community_cot' : 'Cmn',
    }

    df['Model'] = df['Model'].apply(lambda x: rename_models.get(x, x))

    df_baseline = df[df['Environment'] == 'Baseline']
    df_baseline = df_baseline[df_baseline['Model'] != 'Random']


    df_nonbaseline = df[df['Environment'] != 'Baseline']
    df_nonbaseline['Environment'] = df_nonbaseline['Environment'].apply(lambda x: rename_env.get(x, x))


    fig, ax = plt.subplots(1 + int(environments), len(attributes) + int(mutual_acceptance) + int(communities), figsize=(5 * (len(attributes) + int(mutual_acceptance) + int(communities)), 5 * (1 + int(environments))), squeeze=False)


    # df_random = df[df['Model'] == 'Random']
    alpha = 0.1

    ci_erorrorbar = 100 - alpha / num_tests


    for i, y in enumerate(attributes):
        print('attribute', y)
        sns.barplot(data=df_baseline, y=y, x="Temperature", hue="Model", palette=['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9'], ax=ax[0, i], orient='v', errorbar=('ci', ci_erorrorbar))

        if environments:
            sns.barplot(data=df_nonbaseline, y=y, x="Environment", color='#2980b9', ax=ax[1, i], orient='v', errorbar=('ci', ci_erorrorbar))

    df_communities_baseline = df_communities[df_communities['Environment'] == 'Baseline']
    df_communities_baseline = df_communities_baseline[df_communities_baseline['Model'] != 'Random']
    df_communities_baseline['Model'] = df_communities_baseline['Model'].apply(lambda x: rename_models.get(x, x))

    if communities:
        sns.barplot(data=df_communities_baseline, y='Modularity', x='Temperature', hue='Model', palette=['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9'], ax=ax[0, len(attributes)])
        ax[0, len(attributes)].set_ylabel('Modularity')
        ax[0 ,len(attributes)].set_title('')

    if environments:
        df_communities_nonbaseline = df_communities[df_communities['Environment'] != 'Baseline']
        df_communities_nonbaseline['Environment'] = df_communities_nonbaseline['Environment'].apply(lambda x: rename_env.get(x, x))
        df_communities_nonbaseline['Model'] = df_communities_nonbaseline['Model'].apply(lambda x: rename_models.get(x, x))


        if communities:
            sns.barplot(data=df_communities_nonbaseline, y='Modularity', x='Environment', color='#2980b9', ax=ax[1, len(attributes)])
            ax[1, len(attributes)].set_ylabel('Modularity')
            ax[1, len(attributes)].set_title('')

    if mutual_acceptance:
        sns.barplot(data=df_baseline, y='Mutual Acceptance Probability', x='Temperature', hue='Model', palette=['#e67e22', '#f1c40f', '#3498db', '#7f8c8d', '#c0392b', '#34495e', '#2980b9'], ax=ax[0, len(attributes) + int(communities)])

        if environments:
            sns.barplot(data=df_nonbaseline, y='Mutual Acceptance Probability', x='Environment', color='#2980b9', ax=ax[1, len(attributes) + int(communities)])

    # put legend of ax[0, 0] on top of figure
    # ax[0, 0].legend(loc='upper right', bbox_to_anchor=(1, 1))

    for i in range(int(communities) + len(attributes) + int(mutual_acceptance)):
        for j in range(1 + int(environments)):

            ax[j, i].set_ylabel('')
            ax[j, i].set_xlabel('')
            ax[j, i].spines[['right', 'top']].set_visible(False)
            ax[j, i].set_ylim(0, 1)
            # ax[j, i].axhline(y=0, color='black', linestyle='--')

    for i, attribute in enumerate(attributes):
        ax[0, i].set_title(attribute)


    for i in range(int(communities) + len(attributes) + int(mutual_acceptance)):
        for j in range(1 + int(environments)):
            ax[j, i].legend().set_visible(False)

    # use one of the legends of top row to create a legend for the whole figure
    handles, labels = ax[0, 0].get_legend_handles_labels()

    ax[0, 0].set_ylabel('Assortativity')
    if environments:
        ax[1, 0].set_ylabel('Assortativity')


    if communities:
        ax[0, len(attributes)].set_ylabel('Modularity')
        if environments:
            ax[1, len(attributes)].set_ylabel('Modularity')

    if mutual_acceptance:
        ax[0, len(attributes) + int(communities)].set_ylabel('Mutual Acceptance Prob')


    # put figure legend on top of figure above titles
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)

    fig.tight_layout()

    fig.savefig(f'figures/assortativities{sfx}.pdf', bbox_inches='tight')


def principle3_experiment_outfile(experiment, output_dir):
    return str(os.path.join(os.fspath(output_dir), f"principle_3_{experiment['name']}.jsonl"))


def principle3_build_experiment_record(experiment, output_dir, default_temperatures):
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
        'outfile': experiment.get('outfile', principle3_experiment_outfile(experiment, output_dir)),
        'parameters': experiment['parameters'],
        'temperatures': experiment.get('temperatures', default_temperatures),
        'environment': environment,
        'role': role,
        'method': experiment.get('method', 'llm'),
        'cot': experiment.get('COT', False),
        'cot_config': experiment.get('cot_config'),
        'profiles_filename': experiment.get('profiles_filename', PRINCIPLE3_DEFAULT_PROFILES_FILENAME),
        'mutual_acceptance': experiment.get('mutual_acceptance', False),
        'summary_group': experiment.get('summary_group', 'profiles'),
        'metadata': {
            'experiment_name': experiment['name'],
            'summary_group': experiment.get('summary_group', 'profiles'),
            'model': model,
            'environment': environment if environment is not None else 'Baseline',
            'role': role,
            'cot': experiment.get('COT', False),
            'profiles_filename': experiment.get('profiles_filename', PRINCIPLE3_DEFAULT_PROFILES_FILENAME),
            'mutual_acceptance': experiment.get('mutual_acceptance', False),
        },
    }


def principle3_build_cot_calibration_requests(experiment, output_dir, default_temperatures, sample_size=20, seed=0):
    record = principle3_build_experiment_record(experiment, output_dir, default_temperatures)
    n = record['parameters']['n_max']
    temperature = record['temperatures'][0]
    state = principle3_initialize_growth_state(n, temperature, record)
    rng = random.Random(seed)
    nodes = list(state['G'].nodes())
    rng.shuffle(nodes)
    nodes = nodes[:sample_size]

    requests = []
    for t in nodes:
        request = principle3_build_selection_request(
            state['G'],
            t,
            state['profiles'],
            state['environment'],
            state['role'],
            True,
            state['model'],
        )
        requests.append((t, request))
    return record, requests


def principle3_run_cot_budget_calibration(
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
    calibration_filename='principle_3_cot_budget_calibration.json',
):
    return run_cot_budget_calibration(
        experiments,
        output_dir,
        default_temperatures,
        default_cot_config,
        build_calibration_requests=principle3_build_cot_calibration_requests,
        parse_response=principle3_parse_selection_response,
        calibration_filename=calibration_filename,
        run_experiments=run_experiments,
        calibrate=calibrate,
        calibration_sample_size=calibration_sample_size,
        calibration_max_new_tokens=calibration_max_new_tokens,
        calibration_percentile=calibration_percentile,
        calibration_margin=calibration_margin,
        retry_token_buckets=retry_token_buckets,
        calibration_seed=calibration_seed,
    )


def principle3_run_configured_experiments(experiments, output_dir, default_temperatures, run_experiments=True, run_analysis=True):
    supported_models = set(filter_supported_models(sorted({experiment['model'] for experiment in experiments})))
    outfiles_by_group = collections.defaultdict(list)
    experiment_records = []
    experiments_to_analyze = []

    for experiment in experiments:
        if not experiment.get('run', True):
            continue

        model = experiment['model']
        if model not in supported_models:
            print(f'Skipping {experiment["name"]}: {model} is not supported in this environment.')
            continue

        record = principle3_build_experiment_record(experiment, output_dir, default_temperatures)
        experiment_records.append(record)

        if experiment.get('include_in_summary', True):
            outfiles_by_group[record['summary_group']].append(record['outfile'])

        if experiment.get('analyze_detail', False):
            experiments_to_analyze.append(record)

    # Generate any missing profiles files (e.g. on a fresh Google Drive output
    # directory) so both the experiments and the analysis can read them.
    if experiment_records:
        profiles_max_n = max(
            (record['parameters'].get('n_max', 50) for record in experiment_records),
            default=50,
        )
        lucky_basename = os.path.basename(PRINCIPLE3_LUCKY_NUMBER_PROFILES_FILENAME)
        for profiles_filename in sorted({record['profiles_filename'] for record in experiment_records}):
            principle3_ensure_profiles(
                profiles_filename,
                profiles_max_n,
                lucky_number=os.path.basename(profiles_filename) == lucky_basename,
            )

    if run_experiments:
        for record in experiment_records:
            maybe_reset_outfile(record['outfile'])
        batch_groups = collections.defaultdict(list)
        for record in experiment_records:
            if record['model'].startswith('Qwen/') and record['method'] == 'llm' and record['experiment'].get('batch', True):
                batch_key = (
                    record['model'],
                    record['cot'],
                    record['mutual_acceptance'],
                    record['profiles_filename'],
                    json.dumps(record.get('cot_config'), sort_keys=True, default=str),
                )
                batch_groups[batch_key].append(record)
                continue

            try:
                principle3_run_network_formation_experiment(
                    outfile=record['outfile'],
                    method=record['method'],
                    model=record['model'],
                    environment=record['environment'],
                    role=record['role'],
                    cot=record['cot'],
                    cot_config=record.get('cot_config'),
                    profiles_filename=record['profiles_filename'],
                    mutual_acceptance=record['mutual_acceptance'],
                    temperatures=record['temperatures'],
                    metadata=record['metadata'],
                    **record['parameters'],
                )
            except ModelUnavailableError as e:
                print(f'Skipping experiment {record["metadata"]["experiment_name"]}: {e}')

        for records in batch_groups.values():
            try:
                principle3_run_network_formation_experiments_batch(records)
            except ModelUnavailableError as e:
                names = ', '.join(r['metadata']['experiment_name'] for r in records)
                print(f'Skipping batched experiments ({names}): {e}')

    if run_analysis:
        for record in experiments_to_analyze:
            principle3_analyze_experiments(record['outfile'])

        if outfiles_by_group.get('profiles'):
            principle3_get_table(outfiles_by_group['profiles'], environments=True, mutual_acceptance=False, communities=True)
        if outfiles_by_group.get('lucky_number'):
            principle3_get_table(
                outfiles_by_group['lucky_number'],
                sfx='_lucky_number',
                attributes=['Location', 'Hobby', 'Favorite Color', 'Lucky Number'],
                environments=False,
                profiles_filename=PRINCIPLE3_LUCKY_NUMBER_PROFILES_FILENAME,
                mutual_acceptance=False,
                communities=False,
            )
        if outfiles_by_group.get('mutual_acceptance'):
            principle3_get_table(
                outfiles_by_group['mutual_acceptance'],
                sfx='_mutual_acceptance',
                environments=False,
                mutual_acceptance=True,
                communities=True,
            )
        if outfiles_by_group.get('cot'):
            principle3_get_table(outfiles_by_group['cot'], sfx='_cot', environments=False, mutual_acceptance=False, communities=True)

    return {
        'supported_models': supported_models,
        'outfiles_by_group': outfiles_by_group,
        'experiment_records': experiment_records,
        'experiments_to_analyze': experiments_to_analyze,
    }

# --- End Principle 3 profile-homophily utilities ---

# --- Principle 5 small-world utilities ---

def set_principle5_runtime_options(medium_size=28):
    global MEDIUM_SIZE, SMALL_SIZE, BIGGER_SIZE
    MEDIUM_SIZE = medium_size
    SMALL_SIZE = 0.85 * MEDIUM_SIZE
    BIGGER_SIZE = 1.5 * MEDIUM_SIZE
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)
    plt.rc('ytick', labelsize=MEDIUM_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

def principle5_get_stars(p, num_tests=1, parenthesis=True):
    if p < 0.001 / num_tests:
        stars = '***'
    elif p < 0.01 / num_tests:
        stars = '**'
    elif p < 0.05 / num_tests:
        stars = '*'
    else:
        stars = ''

    if parenthesis and stars != '':
        return f'({stars})'
    else:
        return stars

def principle5_barrat_weight_clustering_coefficient(G):

    triangles = 0
    triples = 0

    for node in G.nodes():
        for neighbor in G.neighbors(node):
            for neighbor2 in G.neighbors(neighbor):
                if neighbor2 in G.neighbors(node):
                    triangles += 1

                triples += 1

    return triangles / triples

def principle5_fit_beta_ws(G, k, method='binary_search', tol=0.01, max_iter=100):
    if method == 'closed_form':
        C = principle5_barrat_weight_clustering_coefficient(G)
        C0 = 3 * (k - 1) / (2 * (2 * k - 1))
        return 1 - (C / C0)**(1 / 3)

    elif method == 'binary_search':

        beta_max = 1
        beta_min = 0.01

        n = len(G)

        C = nx.average_clustering(G)

        i = 0

        while True:
            beta = (beta_max + beta_min) / 2

            Gs_WS, _ = principle5_network_growth(n, k, beta, 0, method='W-S', model=None, environment=None, role=None)
            G_WS = Gs_WS[-1]

            C_WS = nx.average_clustering(G_WS)

            if abs(C_WS - C) < tol:
                return beta
            elif C_WS < C:
                beta_max = beta
            else:
                beta_min = beta

            i += 1

            if i >= max_iter:
                return principle5_fit_beta_ws(G, k, method='closed_form')

        return beta

def principle5_network_growth(n, k, beta, temperature=None, environment=None, role='friends', model='gpt-5-mini', cot=False, cot_config=None, method='llm'):
    G = nx.Graph()

    # Create ring network
    for i in range(n):
        G.add_node(i)

    for i in G.nodes():
        G.add_edge(i, (i + 1) % n)

    for i in G.nodes():
        for j in G.nodes():
            if 0 < abs(i - j) % (n - 1 - k / 2) <= k / 2:
                G.add_edge(i, j)

    Gs = []
    results = []

    for i in G.nodes():
        neighbors = list(G.neighbors(i))
        for j in neighbors:
            if 0 < (j - i) % n <= k / 2:
                if np.random.uniform() <= beta:
                        while True:
                            if method == 'W-S':
                                v = random.choice(list(set(G.nodes())))
                            elif method == 'llm':
                                result = principle5_select_neighbor(G, i, temperature, model, environment, role, cot=cot, cot_config=cot_config)
                                if not result:
                                    break
                                v = result['name']

                            if v != i and v not in G.neighbors(i):
                                if method == 'llm':
                                    results.append(result)
                                G.add_edge(i, v)
                                G.remove_edge(i, j)
                                break


        Gs.append(G.copy())

    return Gs, results

def principle5_build_neighbor_request(G, t, environment, role, cot, model):
    features = []
    for v in G.nodes():
        if v != t and v not in G.neighbors(t):
            features.append({'name' : v, 'neighbors' : list(G.neighbors(v))})

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

    candidate_names = [feature['name'] for feature in features]
    response_schema = build_response_schema(candidate_names)
    use_structured_output = not (model.startswith('Qwen/') and cot)
    allowed_names_json = json.dumps(candidate_names, ensure_ascii=False)

    prompt = f"""
    # Task
    {f'You are in a {environment}.' if environment else ''}Your task is to select a person to be {role} with.

    # Input
    The input is a list of dictionaries. Each dictionary has two keys: 'name', 'neighbors'.
    'name' is the name of the person, and 'neighbors' is a list of the person's friends.
    The data is given below after chevrons:
    <DEGREES>
    {json.dumps(features, separators=(',', ':'))}
    </DEGREES>

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
        'candidate_names': set(candidate_names),
    }

def principle5_parse_neighbor_response(ans, request):
    result = first_json_object(ans)
    if not isinstance(result, dict) or 'name' not in result:
        raise ValueError('Could not parse a valid JSON object with a name field.')
    normalized_name = normalize_name(result['name'], request['candidate_names'])
    if normalized_name is None:
        raise ValueError(f"Invalid candidate name: {result['name']}")
    result['name'] = normalized_name
    return result

def principle5_select_neighbor(G, t, temperature, model, environment, role, cot, cot_config=None):
    request = principle5_build_neighbor_request(G, t, environment, role, cot, model)
    for i in range(10):
        try:
            ans = get_response(request['prompt'], model, temperature=temperature, response_schema=request['response_schema'], cot=cot, cot_config=cot_config)
            result = principle5_parse_neighbor_response(ans, request)
            print('NEW EDGE', result)
            return result
        except Exception as e:
            print(e)

def principle5_initialize_ws_graph(n, k):
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)

    for i in G.nodes():
        G.add_edge(i, (i + 1) % n)

    for i in G.nodes():
        for j in G.nodes():
            if 0 < abs(i - j) % (n - 1 - k / 2) <= k / 2:
                G.add_edge(i, j)

    return G

def principle5_initialize_growth_state(n, k, beta, temperature, experiment_record):
    return {
        'n': n,
        'k': k,
        'beta': beta,
        'temperature': temperature,
        'temperature_label': 'default' if temperature is None else temperature,
        'model': experiment_record['model'],
        'environment': experiment_record['environment'],
        'role': experiment_record['role'],
        'cot': experiment_record['cot'],
        'cot_config': experiment_record.get('cot_config'),
        'outfile': experiment_record['outfile'],
        'metadata': experiment_record['metadata'],
        'G': principle5_initialize_ws_graph(n, k),
        'node_order': list(range(n)),
        'node_idx': 0,
        'neighbors': None,
        'neighbor_idx': 0,
        'pending_i': None,
        'pending_j': None,
        'graphs': [],
        'reasons': [],
    }

def principle5_pending_growth_states_for_experiment(experiment_record):
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
            saved_scenarios.add((scenario['n'], scenario['simulation'], scenario['k'], scenario['beta'], scenario['temperature']))

    print(f'Loaded {len(saved_scenarios)} completed simulations from {outfile}')
    states = []
    k = parameters['k']
    beta = experiment_record['beta']
    for n in range(parameters['n_min'], parameters['n_max'] + 1, parameters['n_step']):
        for i in range(parameters['num_simulations']):
            for temperature in temperatures:
                temperature_label = 'default' if temperature is None else temperature
                if (n, i, k, beta, temperature_label) in saved_scenarios:
                    print(f'Skipping simulation for n={n}, i={i}, k={k}, beta={beta}, temperature={temperature_label}')
                    continue
                print(f'Queueing simulation for n={n}, i={i}, k={k}, beta={beta}, temperature={temperature_label}, outfile={outfile}')
                state = principle5_initialize_growth_state(n, k, beta, temperature, experiment_record)
                state['simulation'] = i
                states.append(state)
    return states

def principle5_advance_state_to_next_request(state):
    while state['node_idx'] < len(state['node_order']):
        i = state['node_order'][state['node_idx']]
        if state['neighbors'] is None:
            state['neighbors'] = list(state['G'].neighbors(i))
            state['neighbor_idx'] = 0

        while state['neighbor_idx'] < len(state['neighbors']):
            j = state['neighbors'][state['neighbor_idx']]
            state['neighbor_idx'] += 1
            if 0 < (j - i) % state['n'] <= state['k'] / 2:
                if np.random.uniform() <= state['beta']:
                    state['pending_i'] = i
                    state['pending_j'] = j
                    return True

        state['graphs'].append(state['G'].copy())
        state['node_idx'] += 1
        state['neighbors'] = None
        state['neighbor_idx'] = 0

    return False

def principle5_apply_rewire_result(state, result):
    i = state['pending_i']
    j = state['pending_j']
    if result:
        v = result['name']
        if v != i and v not in state['G'].neighbors(i):
            state['reasons'].append(result)
            state['G'].add_edge(i, v)
            state['G'].remove_edge(i, j)
    state['pending_i'] = None
    state['pending_j'] = None

def principle5_write_growth_state(state):
    temp = {
        'n' : state['n'],
        'k' : state['k'],
        'beta' : state['beta'],
        'temperature' : state['temperature_label'],
        'simulation' : state['simulation'],
        'graphs' : [nx.to_dict_of_lists(G) for G in state['graphs']],
        'reasons' : state['reasons'],
        'model' : state['model'],
        'environment' : state['environment'] if state['environment'] is not None else 'Baseline',
        'role' : state['role'],
        'cot' : state['cot'],
    }
    if state['metadata']:
        temp.update(state['metadata'])
    with open(state['outfile'], 'a+') as f:
        f.write(json.dumps(temp) + '\n')
        f.flush()

def principle5_run_network_formation_experiments_batch(experiment_records):
    if not experiment_records:
        return

    model = experiment_records[0]['model']
    cot = experiment_records[0]['cot']
    cot_config = experiment_records[0].get('cot_config')
    active_states = []
    for experiment_record in experiment_records:
        active_states.extend(principle5_pending_growth_states_for_experiment(experiment_record))

    if not active_states:
        print(f'All batched simulations already completed for {model}. Skipping inference.')
        return

    print(f'Running {len(active_states)} batched simulations for {model}, cot={cot}')
    while active_states:
        ready_states = []
        completed_states = []
        for state in active_states:
            if principle5_advance_state_to_next_request(state):
                ready_states.append(state)
            else:
                completed_states.append(state)

        for state in completed_states:
            principle5_write_growth_state(state)

        if not ready_states:
            break

        requests_by_temperature = collections.defaultdict(list)
        for state in ready_states:
            print(f'Rewiring node {state["pending_i"]} in {state["metadata"]["experiment_name"]}, simulation={state["simulation"]}')
            request = principle5_build_neighbor_request(
                state['G'],
                state['pending_i'],
                state['environment'],
                state['role'],
                state['cot'],
                state['model'],
            )
            requests_by_temperature[state['temperature']].append((state, request))

        results_by_state_id = {}
        for temperature, batch_items in requests_by_temperature.items():
            remaining = list(batch_items)
            for attempt in range(10):
                if not remaining:
                    break

                attempt_cot_config = retry_cot_config(cot_config, attempt) if cot else cot_config
                if cot and attempt_cot_config != cot_config:
                    print(f'Retry attempt {attempt + 1}: max_new_tokens={attempt_cot_config.get("max_new_tokens")}, qwen_enable_thinking={attempt_cot_config.get("qwen_enable_thinking")}')
                prompts = [request['prompt'] for _, request in remaining]
                response_schemas = [request['response_schema'] for _, request in remaining]
                answers = get_responses(
                    prompts,
                    model,
                    temperature=temperature,
                    response_schemas=response_schemas,
                    cot=cot,
                    cot_config=attempt_cot_config,
                )

                next_remaining = []
                for (state, request), ans in zip(remaining, answers):
                    try:
                        result = principle5_parse_neighbor_response(ans, request)
                        print('NEW EDGE', result)
                        results_by_state_id[id(state)] = result
                    except Exception as e:
                        print(e)
                        next_remaining.append((state, request))
                remaining = next_remaining

            for state, _ in remaining:
                results_by_state_id[id(state)] = None

        active_states = []
        for state in ready_states:
            principle5_apply_rewire_result(state, results_by_state_id.get(id(state)))
            active_states.append(state)

def principle5_run_network_formation_experiment(n_min, n_max, n_step, k, beta, num_simulations, outfile, temperatures=None, environment=None, role='friends', method='llm', model='gpt-5-mini', cot=False, cot_config=None, metadata=None):
    if temperatures is None:
        temperatures = [None]

    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    saved_scenarios = set()

    if os.path.exists(outfile):
        with open(outfile) as f:
            lines = f.read().splitlines()

        for line in lines:
            scenario = json.loads(line)
            saved_scenarios.add((scenario['n'], scenario['simulation'], scenario['k'], scenario['beta'], scenario['temperature']))

    expected_scenarios = {
        (n, i, k, beta, 'default' if temperature is None else temperature)
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
                if (n, i, k, beta, temperature_label) in saved_scenarios:
                    print(f'Skipping simulation for n={n}, i={i}, k={k}, beta={beta}, temperature={temperature_label}')
                    continue
                else:
                    print(f'Running simulation for n={n}, i={i}, k={k}, beta={beta}, temperature={temperature_label}')

                    Gs, reasons = principle5_network_growth(n, k, beta, temperature=temperature, method=method, model=model, environment=environment, role=role, cot=cot, cot_config=cot_config)

                    temp = {
                        'n' : n,
                        'k' : k,
                        'beta' : beta,
                        'temperature' : temperature_label,
                        'simulation' : i,
                        'graphs' : [nx.to_dict_of_lists(G) for G in Gs],
                        'reasons' : reasons,
                        'model' : model,
                        'environment' : environment if environment is not None else 'Baseline',
                        'role' : role,
                        'cot' : cot,
                    }
                    if metadata:
                        temp.update(metadata)

                    f.write(json.dumps(temp) + '\n')
                    f.flush()

                if method != 'llm':
                    break

    f.close()

def principle5_draw_graph(G, ax, G0=None, use_netgraph=True):
    if not use_netgraph:
        pos = nx.circular_layout(G)
        nx.draw(G, pos, ax=ax, node_size=10, width=1.5, node_color='#d35400', alpha=0.7, edge_color='#2c3e50')
    else:
        netgraph.Graph(G, ax=ax, node_size=2.5, edge_width=1, node_color='#d35400', edge_color='#2c3e50', node_layout='circular', edge_layout='bundled', edge_layout_kwargs=dict(k=2000))
    ax.set_axis_off()

def principle5_average_shortest_path_length_lcc(G):
    """Average shortest path length on the largest connected component.

    LLM rewiring (and generated W-S null models) can disconnect the graph, for
    which nx.average_shortest_path_length raises 'Graph is not connected.'. Fall
    back to the largest connected component, matching how combined_lcc is used
    in the combined-model analysis. Returns nan for an empty graph and 0.0 for a
    single-node component (no paths)."""
    if G.number_of_nodes() == 0:
        return float('nan')
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc)
    if G.number_of_nodes() < 2:
        return 0.0
    return nx.average_shortest_path_length(G)

def principle5_analyze_experiments(filename, suffix='', fit_beta_method='binary_search'):
    os.makedirs('figures/principle_5', exist_ok=True)

    with open(filename) as f:
        lines = f.read().splitlines()

    data = []

    for line in lines:
        data.append(json.loads(line))


    average_shortest_path_lengths = collections.defaultdict(list)
    average_clustering_coefficients = collections.defaultdict(list)
    graphs = {}
    hat_betas = []
    temperatures = set()

    for d in data:
        Gs = []
        for graph in d['graphs']:
            G = nx.Graph()

            for k, v in graph.items():
                k = int(k)
                G.add_node(k)
                for n in v:
                    G.add_edge(k, n)

            Gs.append(G)

        hat_beta = principle5_fit_beta_ws(Gs[-1], d['k'], method=fit_beta_method)

        average_shortest_path_len = [principle5_average_shortest_path_length_lcc(G) for G in Gs]
        average_clustering_coefficient = [nx.average_clustering(G) for G in Gs]
        average_shortest_path_lengths[d['n'], d['k'], d['beta'], d['temperature']].append(average_shortest_path_len)
        average_clustering_coefficients[d['n'], d['k'], d['beta'], d['temperature']].append(average_clustering_coefficient)

        Gs_WS_estimated, _ = principle5_network_growth(d['n'], d['k'], hat_beta, d['temperature'], method='W-S', model=None, environment=None, role=None)

        G_WS_estimated = Gs_WS_estimated[-1]

        average_shortest_path_len_WS_estimated = principle5_average_shortest_path_length_lcc(G_WS_estimated)
        average_clustering_coefficient_WS_estimated = nx.average_clustering(G_WS_estimated)

        hat_beta_record = {
            'n' : d['n'],
            'k' : d['k'],
            'beta' : d['beta'],
            'hat_beta' : hat_beta,
            'Temperature' : d['temperature'],
            'Simulation' : d['simulation'],
            'Difference in $L$' : average_shortest_path_len_WS_estimated - average_shortest_path_len[-1],
            'Difference in $C$' : average_clustering_coefficient_WS_estimated - average_clustering_coefficient[-1],
            'Avg. Shortest Path Length' : average_shortest_path_len[-1],
            'Avg. Clustering Coefficient' : average_clustering_coefficient[-1],
            'Avg. Shortest Path Length WS' : average_shortest_path_len_WS_estimated,
            'Avg. Clustering Coefficient WS' : average_clustering_coefficient_WS_estimated
        }

        hat_betas.append(hat_beta_record)

        temperatures.add(d['temperature'])

        graphs[d['n'], d['k'], d['beta'], d['temperature'], d['simulation']] = Gs[-1].copy()

    fig_final, ax_final = plt.subplots(1, 2 + len(average_shortest_path_lengths), figsize=(5 * (2 + len(average_shortest_path_lengths)), 5), squeeze=False, gridspec_kw={'width_ratios' : [1] * len(average_shortest_path_lengths) + [0.5, 0.5]})

    ax_final[0, -1].spines[['right', 'top']].set_visible(False)
    ax_final[0, -2].spines[['right', 'top']].set_visible(False)

    i = 0

    for key in sorted(graphs.keys()):
        if key[-1] == 0:
            G = graphs[key]
            principle5_draw_graph(G, ax_final[0, i])
            ax_final[0, i].set_title(f'Temperature = {key[3]}')

            i += 1

    ax_final[0, -1].set_ylabel('$L$')
    ax_final[0, -2].set_ylabel('$C$')


    palette = ['#2980b9', '#f1c40f', '#7f8c8d', '#d35400', '#34495e', '#e67e22',]

    fig, ax = plt.subplots(2, len(average_shortest_path_lengths), figsize=(5 * len(average_shortest_path_lengths), 10), squeeze=False)
    fig_combined, ax_combined = plt.subplots(1, 2, figsize=(10, 5), squeeze=False)

    ax_combined[0, -1].spines[['right', 'top']].set_visible(False)
    ax_combined[0, -2].spines[['right', 'top']].set_visible(False)

    ax_combined[0, 0].set_ylabel('$L$')
    ax_combined[0, 1].set_ylabel('$C$')
    ax_combined[0, 0].set_xlabel('t')
    ax_combined[0, 1].set_xlabel('t')


    for i, (k, c) in enumerate(zip(sorted(average_shortest_path_lengths.keys()), palette)):
        v = average_shortest_path_lengths[k]
        v = np.array(v)

        mean = v.mean(axis=0)
        std = v.std(axis=0)

        ci = 1.96 * std / np.sqrt(len(v))

        ax[0, i].plot(mean, color='#34495e', label='LLM')
        ax[0, i].fill_between(np.arange(len(mean)), mean - ci, mean + ci, alpha=0.2, color='#34495e')

        ax[0, i].set_title(f'Temperature = {k[3]}')

        ax[0, i].set_xlabel('t')
        ax[0, i].set_ylabel('$L$')

        ax[0, i].set_xlim(0, len(mean) - 1)

        ax_combined[0, 0].plot(mean, color=c, label=str(k[3]))
        ax_combined[0, 0].fill_between(np.arange(len(mean)), mean - ci, mean + ci, alpha=0.2, color=c)

        ax_final[0, -1].bar(i, mean[-1], yerr=2*ci[-1], color=c, alpha=0.5, label=str(k[3]))

        for k_prime in sorted(average_shortest_path_lengths.keys()):
            if k[-1] > k_prime[-1]:
                print(f'T-test for Temperatures {k[-1]} and {k_prime[-1]} for $L$: {scipy.stats.ttest_ind([x[-1] for x in average_shortest_path_lengths[k]], [x[-1] for x in average_shortest_path_lengths[k_prime]], equal_var=False, alternative="less")}')


    for i, (k, c) in enumerate(zip(sorted(average_clustering_coefficients.keys()), palette)):
        v = average_clustering_coefficients[k]
        v = np.array(v)

        mean = v.mean(axis=0)
        std = v.std(axis=0)

        ci = 1.96 * std / np.sqrt(len(v))

        ax[1, i].plot(mean, color='#34495e', label='LLM')
        ax[1, i].fill_between(np.arange(len(mean)), mean - ci, mean + ci, alpha=0.2, color='#34495e')

        ax[1, i].set_ylabel('$C$')

        ax[1, i].set_xlabel('t')

        ax[1, i].set_xlim(0, len(mean) - 1)

        ax_combined[0, 1].plot(mean, color=c, label=str(k[3]))
        ax_combined[0, 1].fill_between(np.arange(len(mean)), mean - ci, mean + ci, alpha=0.2, color=c)

        ax_final[0, -2].bar(i, mean[-1], yerr=2*ci[-1], color=c, alpha=0.5, label=str(k[3]))

        for k_prime in sorted(average_clustering_coefficients.keys()):
            if k[-1] > k_prime[-1]:
                print(f'T-test for Temperatures {k[-1]} and {k_prime[-1]} for $C$: {scipy.stats.ttest_ind([x[-1] for x in average_clustering_coefficients[k]], [x[-1] for x in average_clustering_coefficients[k_prime]], equal_var=False, alternative="less")}')

    # Null models
    average_shortest_path_lengths_null = { 'W-S' : collections.defaultdict(list), 'random' : collections.defaultdict(list) }
    average_clustering_coefficients_null = { 'W-S' : collections.defaultdict(list), 'random' : collections.defaultdict(list) }

    for d in data:
        for method in ['W-S']:
            if method == 'random':
                Gs, _ = principle5_network_growth(d['n'], d['k'], 1, d['temperature'], method='W-S', model=None, environment=None, role=None)
            else:
                Gs, _ = principle5_network_growth(d['n'], d['k'], d['beta'], d['temperature'], method=method, model=None, environment=None, role=None)
            average_shortest_path_lengths_null[method][d['n'], d['k'], d['beta'], d['temperature']].append([principle5_average_shortest_path_length_lcc(G) for G in Gs])
            average_clustering_coefficients_null[method][d['n'], d['k'], d['beta'], d['temperature']].append([nx.average_clustering(G) for G in Gs])

    for method in ['W-S']:
        for i, (k, v) in enumerate(average_shortest_path_lengths_null[method].items()):
            v = np.array(v)

            mean = v.mean(axis=0)
            std = v.std(axis=0)

            ci = 1.96 * std / np.sqrt(len(v))

            if method == 'W-S':
                ax[0, i].plot(mean, color='#d35400', linestyle='--', label=method)
            elif method == 'random':
                ax[0, i].plot(mean, color='#d35400', linestyle=':', label=method)

            ax[0, i].fill_between(np.arange(len(mean)), mean - ci, mean + ci, alpha=0.2, color='#d35400', hatch='||')

            if i == 0:
                ax_combined[0, 0].plot(mean, color='#d35400', linestyle='--', label=method)
                ax_combined[0, 0].fill_between(np.arange(len(mean)), mean - ci, mean + ci, alpha=0.2, color='#d35400', hatch='||')

                ax_final[0, -1].bar(len(average_shortest_path_lengths), mean[-1], yerr=2*ci[-1], color='#d35400', alpha=0.5, label=method)

            for k_prime in sorted(average_shortest_path_lengths.keys()):
                if k == k_prime:
                    print(f'T-test for Temperature {k[-1]} and W-S for $L$ (two-sided): {scipy.stats.ttest_ind([x[-1] for x in average_shortest_path_lengths_null[method]], [x[-1] for x in average_shortest_path_lengths[k_prime]], equal_var=False, alternative="two-sided")}')


    for method in ['W-S']:
        for i, (k, v) in enumerate(average_clustering_coefficients_null[method].items()):
            v = np.array(v)

            mean = v.mean(axis=0)
            std = v.std(axis=0)

            ci = 1.96 * std / np.sqrt(len(v))

            if method == 'W-S':
                ax[1, i].plot(mean, color='#d35400', linestyle='--', label=method)
            elif method == 'random':
                ax[1, i].plot(mean, color='#d35400', linestyle=':', label=method)

            ax[1, i].fill_between(np.arange(len(mean)), mean - ci, mean + ci, alpha=0.2, color='#d35400', hatch='||')

            if i == 0:
                ax_combined[0, 1].plot(mean, color='#d35400', linestyle='--', label=method)
                ax_combined[0, 1].fill_between(np.arange(len(mean)), mean - ci, mean + ci, alpha=0.2, color='#d35400', hatch='||')

                ax_final[0, -2].bar(len(average_shortest_path_lengths), mean[-1], yerr=2*ci[-1], color='#d35400', alpha=0.5, label=method)

            for k_prime in sorted(average_clustering_coefficients.keys()):
                if k == k_prime:
                    print(f'T-test for Temperature {k[-1]} and W-S for $C$ (greater): {scipy.stats.ttest_ind([x[-1] for x in average_clustering_coefficients_null[method]], [x[-1] for x in average_clustering_coefficients[k_prime]], equal_var=False, alternative="greater")}')
    ax_final[0, -1].set_xticks([])
    ax_final[0, -2].set_xticks([])

    ax_final[0, -1].legend(bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)


    for i in range(len(average_shortest_path_lengths)):
        ax[0, i].legend(loc='upper left')
        ax[1, i].legend(loc='upper left')

        ax[0, i].set_ylim(2, 3)

    ax_combined[0, 0].legend(loc='upper right')

    fig_combined.tight_layout()

    fig.tight_layout()

    fig.savefig(f'figures/principle_5/principle_5_overall{f"_{suffix}" if suffix else ""}.pdf')

    fig_combined.savefig(f'figures/principle_5/principle_5_overall_combined{f"_{suffix}" if suffix else ""}.pdf')

    fig_final.tight_layout()

    fig_final.savefig(f'figures/principle_5/principle_5_final_graphs{f"_{suffix}" if suffix else ""}.pdf')

    hat_betas = pd.DataFrame.from_records(hat_betas)

    fig_estimated, ax_estimated = plt.subplots(1, 2, figsize=(10, 5), squeeze=False)

    sns.boxplot(data=hat_betas, x='Temperature', y='hat_beta', ax=ax_estimated[0, 0], palette=palette)
    ax_estimated[0, 0].set_ylabel('Estimated $\\hat{\\beta}$')

    sns.violinplot(data=hat_betas, x='Temperature', y='Difference in $L$', ax=ax_estimated[0, 1], palette=palette)

    for j, temperature in enumerate(temperatures):
        df_temp = hat_betas.query('Temperature == @temperature')
        t, p = scipy.stats.ttest_ind(df_temp['Avg. Shortest Path Length'], df_temp['Avg. Shortest Path Length WS'], equal_var=False, alternative="two-sided")
        print(f'T-test for average path length vs WS with estimated beta for temperature {temperature}: t = {t}, p = {p}')
        ax_estimated[0, 1].text(j, 0.9, f'P = {p:.2f}', ha='center', fontsize=0.65*SMALL_SIZE)
        ax_estimated[0, 1].set_ylim(-0.6, 0.85)

    ax_estimated[0, 0].spines[['right', 'top']].set_visible(False)
    ax_estimated[0, 1].spines[['right', 'top']].set_visible(False)


    fig_estimated.tight_layout()

    fig_estimated.savefig(f'figures/principle_5/principle_5_estimated_beta{f"_{suffix}" if suffix else ""}.png')

def principle5_plot_multiple_networks_small_world(filename, outfile):

    with open(filename) as f:
        lines = f.read().splitlines()

    data = []

    for line in lines:
        data.append(json.loads(line))

    records = []

    temperatures = set()
    seen = set()

    for d in data:
        Gs = []
        for graph in d['graphs']:
            G = nx.Graph()

            for k, v in graph.items():
                k = int(k)
                G.add_node(k)
                for n in v:
                    G.add_edge(k, n)

            Gs.append(G)

        try:
            average_shortest_path_len = principle5_average_shortest_path_length_lcc(Gs[-1])
            average_clustering_coefficient = nx.average_clustering(Gs[-1])

            record = {
                'n' : d['n'],
                'log(n)' : np.log(d['n']),
                '1/log(n)' : 1 / np.log(d['n']),
                'k' : d['k'],
                'beta' : d['beta'],
                'temperature' : d['temperature'],
                'simulation' : d['simulation'],
                '$L$' : average_shortest_path_len,
                '$C$' : average_clustering_coefficient
            }


            temperatures.add(d['temperature'])
            records.append(record)


            seen.add((d['n'], d['k'], d['beta']))
        except:
            pass

    for n, k, beta in seen:
        try:
            Gs, _ = principle5_network_growth(n, k, beta, 0, method='W-S')

            average_shortest_path_len = principle5_average_shortest_path_length_lcc(Gs[-1])
            average_clustering_coefficient = nx.average_clustering(Gs[-1])

            record = {
                'n' : n,
                'log(n)' : np.log(n),
                '1/log(n)' : 1 / np.log(n),
                'k' : k,
                'beta' : beta,
                'temperature' : 'W-S',
                'simulation' : 0,
                '$L$' : average_shortest_path_len,
                '$C$' : average_clustering_coefficient
            }

            records.append(record)
        except:
            pass

    df = pd.DataFrame.from_records(records)


    fig, ax = plt.subplots(1, 2, figsize=(10, 5), squeeze=False)

    palette = ['#2980b9', '#f1c40f', '#7f8c8d', '#d35400', '#34495e', '#e67e22',]


    for i, temperature in enumerate(temperatures):
        regress_result = scipy.stats.linregress(df.query('temperature == @temperature')['log(n)'], df.query('temperature == @temperature')['$L$'])
        stars = '***' if regress_result.pvalue < 0.001 else '**' if regress_result.pvalue < 0.01 else '*' if regress_result.pvalue < 0.05 else ''

        sns.regplot(data=df.query('temperature == @temperature'), x='log(n)', y='$L$', ax=ax[0, 0], label=f'{temperature}, $a$ = {regress_result.slope:.2f} ({stars})', color=palette[i])


    regress_result = scipy.stats.linregress(df.query(f'temperature == "W-S"')['log(n)'], df.query(f'temperature == "W-S"')['$L$'])
    stars = '***' if regress_result.pvalue < 0.001 else '**' if regress_result.pvalue < 0.01 else '*' if regress_result.pvalue < 0.05 else ''
    sns.regplot(data=df.query(f'temperature == "W-S"'), x='log(n)', y='$L$', ax=ax[0, 0], label=f'W-S, $a$ = {regress_result.slope:.2f} ({stars})', color=palette[len(temperatures)])

    # ax[0, 0].legend(fontsize=0.75*SMALL_SIZE, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1)

    ax[0, 0].legend(fontsize=0.5*SMALL_SIZE, loc='lower right')
    ax[0, 0].spines[['right', 'top']].set_visible(False)

    for i, temperature in enumerate(temperatures):
        regress_result = scipy.stats.linregress(df.query('temperature == @temperature')['1/log(n)'], df.query('temperature == @temperature')['$C$'])
        stars = '(***)' if regress_result.pvalue < 0.001 else '(**)' if regress_result.pvalue < 0.01 else '(*)' if regress_result.pvalue < 0.05 else ''
        sns.regplot(data=df.query('temperature == @temperature'), x='1/log(n)', y='$C$', ax=ax[0, 1], label=f'{temperature}, $a$ = {regress_result.slope:.2f} {stars}', color=palette[i])

    regress_result = scipy.stats.linregress(df.query(f'temperature == "W-S"')['1/log(n)'], df.query(f'temperature == "W-S"')['$C$'])
    stars = '(***)' if regress_result.pvalue < 0.001 else '(**)' if regress_result.pvalue < 0.01 else '(*)' if regress_result.pvalue < 0.05 else ''
    sns.regplot(data=df.query(f'temperature == "W-S"'), x='1/log(n)', y='$C$', ax=ax[0, 1], label=f'W-S, $a$ = {regress_result.slope:.2f} {stars}', color=palette[len(temperatures)])

    # ax[0, 1].legend(fontsize=0.75*SMALL_SIZE, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1)

    ax[0, 1].legend(fontsize=0.5*SMALL_SIZE, loc='lower right')

    ax[0, 1].spines[['right', 'top']].set_visible(False)

    fig.tight_layout()

    fig.savefig(outfile, dpi=300, bbox_inches='tight')

def principle5_get_table(filenames, sfx=''):
    os.makedirs('figures/principle_5', exist_ok=True)
    os.makedirs('tables', exist_ok=True)

    records_coefficients = []
    sns.set_palette(['#2980b9', '#f1c40f', '#7f8c8d', '#d35400', '#34495e', '#e67e22',])

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

    rename_env = {
        'school' : 'School',
        'work' : 'Work',
        'community' : 'Community',
        'school_cot' : 'School',
        'work_cot' : 'Work',
        'community_cot' : 'Community',
    }

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), squeeze=False)

    for filename in filenames:
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

        records = []

        temperatures = set()
        seen = set()

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
            for graph in d['graphs']:
                G = nx.Graph()

                for k, v in graph.items():
                    k = int(k)
                    G.add_node(k)
                    for n in v:
                        G.add_edge(k, n)

                Gs.append(G)

            try:
                average_shortest_path_len = principle5_average_shortest_path_length_lcc(Gs[-1])
                average_clustering_coefficient = nx.average_clustering(Gs[-1])

                record = {
                    'n' : d['n'],
                    'log(n)' : np.log(d['n']),
                    '1/log(n)' : 1 / np.log(d['n']),
                    'k' : d['k'],
                    'beta' : d['beta'],
                    'temperature' : d['temperature'],
                    'simulation' : d['simulation'],
                    '$L$' : average_shortest_path_len,
                    '$C$' : average_clustering_coefficient,
                    'Model' : model,
                    'Environment' : environment
                }


                temperatures.add(d['temperature'])
                records.append(record)


                seen.add((d['n'], d['k'], d['beta']))
            except:
                pass

        for n, k, beta in seen:
            Gs, _ = principle5_network_growth(n, k, beta, 0, method='W-S', model='', environment='', role='')

            average_shortest_path_len = principle5_average_shortest_path_length_lcc(Gs[-1])
            average_clustering_coefficient = nx.average_clustering(Gs[-1])

            record = {
                'n' : n,
                'log(n)' : np.log(n),
                '1/log(n)' : 1 / np.log(n),
                'k' : k,
                'beta' : beta,
                'temperature' : 'default',
                'simulation' : 0,
                '$L$' : average_shortest_path_len,
                '$C$' : average_clustering_coefficient,
                'Model' : 'W-S',
                'Environment' : 'Baseline'
            }

            records.append(record)

        df = pd.DataFrame.from_records(records)

        df['Model'] = df['Model'].apply(lambda x: rename_models.get(x, x))
        df['Environment'] = df['Environment'].apply(lambda x: rename_env.get(x, x))


        for temperature in df['temperature'].unique():
            for model in df['Model'].unique():
                for environment in df['Environment'].unique():
                    if model == 'W-S':
                        continue
                    query = 'temperature == @temperature & Model == @model & Environment == @environment'

                    if not df.query(query).empty:
                        regress_result_L = scipy.stats.linregress(df.query(query, inplace=False)['log(n)'], df.query(query, inplace=False)['$L$'])
                        stars_L = principle5_get_stars(regress_result_L.pvalue, parenthesis=False, num_tests=2)
                        sns.regplot(data=df.query(query), x='log(n)', y='$L$', ax=ax[0, 0], label=f'{model} {f"({environment})" if environment != "Baseline" else ""} $a$ = {regress_result_L.slope:.2f} ({stars_L})')

                        regress_result_C = scipy.stats.linregress(df.query(query, inplace=False)['1/log(n)'], df.query(query, inplace=False)['$C$'])
                        stars_C = principle5_get_stars(regress_result_C.pvalue, parenthesis=False, num_tests=2)

                        records_coefficients.append({
                            'Model' : model,
                            'Environment' : environment,
                            'Temperature' : temperature,
                            'Regression Coefficient ($L \\sim \\log (n)$)' : f'{regress_result_L.slope:.2f} ({stars_L})',
                            'Regression Coefficient ($C \\sim 1 / \\log (n)$)' : f'{regress_result_C.slope:.2f} ({stars_C})'
                        })


    query = f'Model == "W-S"'

    regress_result_L_WS = scipy.stats.linregress(df.query(query, inplace=False)['log(n)'], df.query(query, inplace=False)['$L$'])
    stars_L_WS = principle5_get_stars(regress_result_L_WS.pvalue, parenthesis=False, num_tests=2)

    regress_result_C_WS = scipy.stats.linregress(df.query(query, inplace=False)['1/log(n)'], df.query(query, inplace=False)['$C$'])
    stars_C_WS = principle5_get_stars(regress_result_C_WS.pvalue, parenthesis=False, num_tests=2)

    records_coefficients.append({
        'Model' : 'W-S',
        'Environment' : '',
        'Temperature' : None,
        'Regression Coefficient ($L \\sim \\log (n)$)' : f'{regress_result_L_WS.slope:.2f} ({stars_L_WS})',
        'Regression Coefficient ($C \\sim 1 / \\log (n)$)' : f'{regress_result_C_WS.slope:.2f} ({stars_C_WS})'
    })

    sns.regplot(data=df.query(query), x='log(n)', y='$L$', ax=ax[0, 0], label=f'W-S, $a$ = {regress_result_L_WS.slope:.2f} ({stars_L_WS})')

    ax[0, 0].set_ylabel('$L$')
    ax[0, 0].set_xlabel('$\\log (n)$')

    # Move legend to the right outside of the plot
    ax[0, 0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    sns.despine()

    fig.savefig(f'figures/principle_5/principle_5_multiple{sfx}.pdf', bbox_inches='tight')

    df_coefficients = pd.DataFrame.from_records(records_coefficients)

    df_coefficients.sort_values(['Model', 'Environment', 'Temperature'], inplace=True)

    df_coefficients.to_csv(f'tables/principle_5_multiple{sfx}.csv', index=False)
    df_coefficients.to_latex(f'tables/principle_5_multiple{sfx}.tex', index=False, escape=False, float_format="%.2f")

def principle5_experiment_outfile(experiment, output_dir):
    return str(os.path.join(os.fspath(output_dir), f"principle_5_{experiment['name']}.jsonl"))


def principle5_build_experiment_record(experiment, output_dir, default_temperatures):
    environment_role = experiment.get('environment')
    if environment_role is None:
        environment = None
        role = 'friends'
    else:
        environment, role = environment_role

    model = experiment['model']
    params = dict(experiment['parameters'])
    params['beta'] = experiment['beta']
    cot = experiment.get('COT', False)
    return {
        'experiment': experiment,
        'name': experiment['name'],
        'model': model,
        'outfile': experiment.get('outfile', principle5_experiment_outfile(experiment, output_dir)),
        'parameters': params,
        'temperatures': experiment.get('temperatures', default_temperatures),
        'environment': environment,
        'role': role,
        'method': experiment.get('method', 'llm'),
        'beta': experiment['beta'],
        'cot': cot,
        'cot_config': experiment.get('cot_config'),
        'summary_group': experiment.get('summary_group', 'default'),
        'metadata': {
            'experiment_name': experiment['name'],
            'summary_group': experiment.get('summary_group', 'default'),
            'model': model,
            'environment': environment if environment is not None else 'Baseline',
            'role': role,
            'cot': cot,
        },
    }


def principle5_build_cot_calibration_requests(experiment, output_dir, default_temperatures, sample_size=20, seed=0):
    record = principle5_build_experiment_record(experiment, output_dir, default_temperatures)
    n = record['parameters']['n_max']
    k = record['parameters']['k']
    beta = record['parameters']['beta']
    temperature = record['temperatures'][0]
    state = principle5_initialize_growth_state(n, k, beta, temperature, record)
    rng = random.Random(seed)
    nodes = list(state['G'].nodes())
    rng.shuffle(nodes)
    nodes = nodes[:sample_size]

    requests = []
    for t in nodes:
        request = principle5_build_neighbor_request(
            state['G'],
            t,
            state['environment'],
            state['role'],
            True,
            state['model'],
        )
        requests.append((t, request))
    return record, requests


def principle5_run_cot_budget_calibration(
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
    calibration_filename='principle_5_cot_budget_calibration.json',
):
    return run_cot_budget_calibration(
        experiments,
        output_dir,
        default_temperatures,
        default_cot_config,
        build_calibration_requests=principle5_build_cot_calibration_requests,
        parse_response=principle5_parse_neighbor_response,
        calibration_filename=calibration_filename,
        run_experiments=run_experiments,
        calibrate=calibrate,
        calibration_sample_size=calibration_sample_size,
        calibration_max_new_tokens=calibration_max_new_tokens,
        calibration_percentile=calibration_percentile,
        calibration_margin=calibration_margin,
        retry_token_buckets=retry_token_buckets,
        calibration_seed=calibration_seed,
    )


def principle5_run_configured_experiments(experiments, output_dir, default_temperatures, run_experiments=True, run_analysis=True):
    supported_models = set(filter_supported_models(sorted({experiment['model'] for experiment in experiments})))
    outfiles_by_group = collections.defaultdict(list)
    experiment_records = []
    experiments_to_analyze = []

    for experiment in experiments:
        if not experiment.get('run', True):
            continue

        model = experiment['model']
        if model not in supported_models:
            print(f'Skipping {experiment["name"]}: {model} is not supported in this environment.')
            continue

        record = principle5_build_experiment_record(experiment, output_dir, default_temperatures)
        experiment_records.append(record)

        if experiment.get('include_in_summary', True):
            outfiles_by_group[record['summary_group']].append(record['outfile'])

        if experiment.get('analyze_detail', False):
            experiments_to_analyze.append(record)

    if run_experiments:
        for record in experiment_records:
            maybe_reset_outfile(record['outfile'])
        batch_groups = collections.defaultdict(list)
        for record in experiment_records:
            if record['model'].startswith('Qwen/') and record['method'] == 'llm' and record['experiment'].get('batch', True):
                batch_key = (
                    record['model'],
                    record['cot'],
                    record['beta'],
                    json.dumps(record.get('cot_config'), sort_keys=True, default=str),
                )
                batch_groups[batch_key].append(record)
                continue

            try:
                principle5_run_network_formation_experiment(
                    outfile=record['outfile'],
                    temperatures=record['temperatures'],
                    environment=record['environment'],
                    role=record['role'],
                    method=record['method'],
                    model=record['model'],
                    cot=record['cot'],
                    cot_config=record.get('cot_config'),
                    metadata=record['metadata'],
                    **record['parameters'],
                )
            except ModelUnavailableError as e:
                print(f'Skipping experiment {record["metadata"]["experiment_name"]}: {e}')

        for records in batch_groups.values():
            try:
                principle5_run_network_formation_experiments_batch(records)
            except ModelUnavailableError as e:
                names = ', '.join(r['metadata']['experiment_name'] for r in records)
                print(f'Skipping batched experiments ({names}): {e}')

    if run_analysis:
        for record in experiments_to_analyze:
            principle5_analyze_experiments(record['outfile'], suffix=record['name'])

        if outfiles_by_group.get('default'):
            principle5_get_table(outfiles_by_group['default'])
        if outfiles_by_group.get('cot'):
            principle5_get_table(outfiles_by_group['cot'], sfx='_cot')

    return {
        'supported_models': supported_models,
        'outfiles_by_group': outfiles_by_group,
        'experiment_records': experiment_records,
        'experiments_to_analyze': experiments_to_analyze,
    }

# --- End Principle 5 small-world utilities ---

# --- Combined real-world network utilities ---

def set_combined_model_runtime_options(font_scale=1.2):
    sns.set_theme(font_scale=font_scale)

COMBINED_RENAME_MODELS = {
    'gpt-5-nano' : 'GPT-5 Nano',
    'gpt-5-mini' : 'GPT-5 Mini',
    'Qwen-Qwen3.5-4B' : 'Qwen 3.5 4B',
    'Qwen-Qwen3.5-2B' : 'Qwen 3.5 2B',
    'Qwen-Qwen3.5-0.8B' : 'Qwen 3.5 0.8B',
    'gpt-5-nano+link_prediction' : 'GPT-5 Nano',
    'gpt-5-mini+link_prediction' : 'GPT-5 Mini',
    'Qwen-Qwen3.5-4B+link_prediction' : 'Qwen 3.5 4B',
    'Qwen-Qwen3.5-2B+link_prediction' : 'Qwen 3.5 2B',
    'Qwen-Qwen3.5-0.8B+link_prediction' : 'Qwen 3.5 0.8B',
    'Qwen-Qwen3.5-4B-nothinking' : 'Qwen 3.5 4B (no thinking)',
    'Qwen-Qwen3.5-4B-thinking' : 'Qwen 3.5 4B (thinking)',
    'Qwen-Qwen3.5-4B-nothinking+link_prediction' : 'Qwen 3.5 4B (no thinking)',
    'Qwen-Qwen3.5-4B-thinking+link_prediction' : 'Qwen 3.5 4B (thinking)',
}

def _combined_checkpoint_path(outfile, ego, simulation, temperature_label, num_samples, num_choices):
    stem = os.path.splitext(outfile)[0]

    def safe(value):
        return str(value).replace('/', '-').replace('.', 'p')

    return f'{stem}.ckpt.ego{safe(ego)}.sim{simulation}.temp{safe(temperature_label)}.ns{num_samples}.nc{num_choices}.checkpoint.json'


def _combined_serialize_rng():
    py_state = random.getstate()
    np_state = np.random.get_state()
    return {
        'py': [py_state[0], list(py_state[1]), py_state[2]],
        'np': [np_state[0], [int(x) for x in np_state[1]], int(np_state[2]), int(np_state[3]), float(np_state[4])],
    }


def _combined_restore_rng(state):
    if not state:
        return
    py_state = state.get('py')
    if py_state:
        random.setstate((py_state[0], tuple(py_state[1]), py_state[2]))
    np_state = state.get('np')
    if np_state:
        np.random.set_state((np_state[0], np.array(np_state[1], dtype=np.uint32), np_state[2], np_state[3], np_state[4]))


def _combined_save_checkpoint(path, next_index, results, candidates):
    """Atomically persist mid-simulation progress so an interrupted run can resume."""
    payload = {
        'next_index': next_index,
        'results': results,
        'candidates': candidates,
        'rng': _combined_serialize_rng(),
    }
    tmp = f'{path}.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f)
        f.flush()
    os.replace(tmp, path)


def _combined_load_checkpoint(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f'Ignoring unreadable checkpoint {path}: {e}')
        return None


def _combined_remove_checkpoint(path):
    if path and os.path.exists(path):
        os.remove(path)


def _combined_remove_all_checkpoints(outfile):
    stem = os.path.splitext(outfile)[0]
    for path in glob.glob(f'{stem}.ckpt.*.checkpoint.json'):
        os.remove(path)


def combined_network_growth(G0, temperature=None, name='', num_choices=1, method='llm', num_samples=-1, num_nodes_samples=-1, model='gpt-5-mini', sampling_strategy='random', cot=False, cot_config=None, checkpoint_path=None, checkpoint_every=25):
    # Set seed
    random.seed(0)
    np.random.seed(0)

    # Copy the ground truth graph
    G = G0.copy()

    Gs = [G.copy()]

    profiles = nx.get_node_attributes(G, 'features')

    if sampling_strategy == 'link_prediction':
        sk_model = link_prediction.train_link_predictor(G, profiles=profiles, name=name)
    else:
        sk_model = None

    # Edges to drop
    dropped_edges = []

    if num_nodes_samples > 0 and num_nodes_samples < len(G):
        print(f'Sampling {num_nodes_samples} nodes from {len(G)} nodes')
        print(G.nodes())

        nodes = random.sample(list(G.nodes()), num_nodes_samples)
    else:
        nodes = list(G.nodes())


    # Drop one neighbor for each node
    for v in nodes:
        dropped_v_edges = []
        for _ in range(num_choices):
            if len(list(G.neighbors(v))) > 0:

                while True:
                    u = random.choice(list(G.neighbors(v)))
                    if (v, u) not in dropped_edges:
                        dropped_v_edges.append((v, u))
                        G.remove_edge(v, u)
                        break

        dropped_edges.append(dropped_v_edges)

    Gs = [G.copy()]
    results = []
    candidates = []

    # Resume from a checkpoint if one exists: the deterministic setup above
    # reproduces the same node order, dropped edges and link predictor, so we
    # only need to replay the recorded edges, restore the RNG state, and continue
    # from the next unprocessed node.
    start_index = 0
    if checkpoint_path and method == 'llm':
        checkpoint = _combined_load_checkpoint(checkpoint_path)
        if checkpoint is not None:
            start_index = checkpoint['next_index']
            results = checkpoint['results']
            candidates = checkpoint['candidates']
            for saved_result in results:
                for r in saved_result:
                    G.add_edge(*r['edge'], similarity=r['similarity'])
            _combined_restore_rng(checkpoint.get('rng'))
            print(f'Resuming {os.path.basename(checkpoint_path)} from node {start_index}/{len(nodes)}')

    for i in range(start_index, len(nodes)):
        t = nodes[i]

        if method == 'llm':
            print('{}/{}'.format(i + 1, len(nodes)))
            result, candidate = combined_select_neighbor(G, t, profiles, temperature, num_choices=len(dropped_edges[i]), dropped_nodes=[u for (_, u) in dropped_edges[i]], num_samples=num_samples, model=model, sampling_strategy=sampling_strategy, sk_model=sk_model, cot=cot, cot_config=cot_config)

            if result:
                for r in result:
                    v = r['name']
                    r['edge'] = (t, v)
                    G.add_edge(t, v, similarity=r['similarity'])
                results.append(result)

            candidates.append(candidate)
        if method == 'ground_truth':
            if num_samples > num_choices:
                choice_set = random.sample([v for v in G.nodes() if v != t], num_samples - num_choices)
            else:
                choice_set = [v for v in G.nodes() if v != t]

            new_nodes = [e[1] for e in dropped_edges[i]]

            choice_set = choice_set + new_nodes


            result = []

            for v in new_nodes:

                profiles[t]['neighbors'] = list(G.neighbors(t))
                profiles[v]['neighbors'] = list(G.neighbors(v))
                profiles[t]['degree'] = len(profiles[t]['neighbors'])
                profiles[v]['degree'] = len(profiles[v]['neighbors'])

                similarity = combined_measure_similarity(profiles[t], profiles[v])
                G.add_edge(t, v, similarity=similarity, weight=similarity['common_attributes'])

                result.append({'name' : v, 'similarity' : similarity, 'reason' : method, 'dropped' : True})

            candidate = []

            for v in choice_set:
                profiles[t]['neighbors'] = list(G.neighbors(t))
                profiles[v]['neighbors'] = list(G.neighbors(v))
                profiles[t]['degree'] = len(profiles[t]['neighbors'])
                profiles[v]['degree'] = len(profiles[v]['neighbors'])

                similarity = combined_measure_similarity(profiles[t], profiles[v])
                candidate.append({'name' : v, 'similarity' : similarity, 'reason' : method})

            candidates.append(candidate)
            results.append(result)

            print(f'Node: {t}, Links: {result}, Candidates: {candidate}')

        Gs.append(G.copy())

        # Periodically persist progress so a disconnect only loses the last few
        # nodes instead of the whole ego.
        if checkpoint_path and method == 'llm' and (i + 1) % checkpoint_every == 0 and (i + 1) < len(nodes):
            _combined_save_checkpoint(checkpoint_path, i + 1, results, candidates)

    return Gs, results, candidates

def combined_fit_dcm(results):

    similarities = [r['similarity'] for result in results for r in result]
    similarities_df = pd.DataFrame.from_records(similarities)
    similarities_df = sm.add_constant(similarities_df)

    outcomes = np.array([r['edge'][1] for result in results for r in result])

    print(similarities_df)

    mnl_model = sm.MNLogit(outcomes, similarities_df)
    mnl_results = mnl_model.fit()

    print(mnl_results.summary())

    return mnl_results

def combined_measure_similarity(profile1, profile2):

    similarity = {
        'common_attributes' : 0,
        'common_neighbors' : len(set(profile1['neighbors']) & set(profile2['neighbors'])),
        'degree' : profile2['degree'],
    }

    for k in profile1.keys():
        if k != 'name' and k != 'neighbors' and k in profile2.keys():
            if isinstance(profile1[k], list):
                similarity['common_attributes'] += len(set(profile1[k]) & set(profile2[k]))
            elif profile1[k] == profile2[k]:
                similarity['common_attributes'] += 1

    return similarity

def combined_build_selection_request(G, t, profiles, num_choices=1, num_samples=-1, dropped_nodes=[], model='gpt-5-mini', sampling_strategy='random', sk_model=None):
    if num_samples > 0:
        if sampling_strategy == 'random':
            choice_set = random.sample([v for v in G.nodes() if v != t and v not in G.neighbors(t)], max(0, num_samples - len(dropped_nodes))) + dropped_nodes
        elif sampling_strategy == 'pagerank':
            pagerank_scores = nx.pagerank(G)
            temp_nodes = [v for v in G.nodes() if v != t and v not in G.neighbors(t)]
            # pagerank scores to numpy
            pagerank_scores = np.array([pagerank_scores[v] for v in temp_nodes])
            choice_set = np.random.choice(temp_nodes, size=min(num_samples - len(dropped_nodes), len(temp_nodes)), replace=False, p=pagerank_scores/np.sum(pagerank_scores)).tolist() + dropped_nodes
        elif sampling_strategy == 'degree':
            temp_nodes = [v for v in G.nodes() if v != t and v not in G.neighbors(t)]
            degree_scores = np.array([G.degree(v) for v in temp_nodes])
            choice_set = np.random.choice(temp_nodes, size=min(num_samples - len(dropped_nodes), len(temp_nodes)), replace=False, p=degree_scores/np.sum(degree_scores)).tolist() + dropped_nodes
        elif sampling_strategy == 'link_prediction':
            choice_set = link_prediction.recommend_friends(sk_model, G, profiles, t, k=num_samples - len(dropped_nodes)) + dropped_nodes
    else:
        choice_set = [v for v in G.nodes() if v != t and v not in G.neighbors(t)]

    candidate_profiles = []

    for v in choice_set + [t]:
        profiles[v]['neighbors'] = list(G.neighbors(v))
        profiles[v]['degree'] = len(profiles[v]['neighbors'])
        profiles[v]['name'] = v
        candidate_profiles.append(profiles[v])

    random.shuffle(candidate_profiles)

    prompt = f"""
    # Task
    Your task is to select a set of people to be friends with.

    # Profile
    Your profile is given below after chevrons:
    <PROFILE>
    {json.dumps(profiles[t])}
    </PROFILE>

    # Candidate Profiles
    The cadidate profiles to be friends with are given below after chevrons:

    <PROFILES>
    {json.dumps(candidate_profiles)}
    </PROFILES>

    # Output
    The output should be given a list of JSON objects with the following structure

    [
        {{
            "name" : name of the person you selected,
            "reason" : reason for selecting the person
        }}, ...
    ]

    # Notes
    * The output must be a list of JSON objects ranked in the order of preference.
    * You can make at most {num_choices} selection{'s' if num_choices > 1 else ''}.
    * If your chat template enables thinking, keep reasoning in the thinking section.
    * Your output must be contained within the json markdown cue.

    ```json
    """

    candidate_names = [candidate_profile['name'] for candidate_profile in candidate_profiles]
    # Constrain Qwen's structured-output decoding to the offered candidates, the
    # same way principle_2 does. Skipped when no selection is expected.
    response_schema = build_response_list_schema(candidate_names, max_items=num_choices) if num_choices > 0 else None

    return {'prompt': prompt, 'candidate_profiles': candidate_profiles, 'candidate_names': candidate_names, 'num_choices': num_choices, 'response_schema': response_schema}


def combined_parse_selection_response(ans):
    for parser in (
        lambda a: json.loads(a.split('```')[0]),
        lambda a: json.loads(a.split('```json')[1].split('```')[0]),
    ):
        try:
            results = parser(ans)
            if isinstance(results, list):
                return results
        except Exception:
            pass

    results = first_json_array(ans)
    if not isinstance(results, list):
        raise ValueError('Could not parse a JSON array from the response.')
    return results


def combined_select_neighbor(G, t, profiles, temperature=None, num_choices=1, num_samples=-1, dropped_nodes=[], model='gpt-5-mini', sampling_strategy='random', sk_model=None, cot=False, cot_config=None):
    request = combined_build_selection_request(G, t, profiles, num_choices=num_choices, num_samples=num_samples, dropped_nodes=dropped_nodes, model=model, sampling_strategy=sampling_strategy, sk_model=sk_model)
    candidate_profiles = request['candidate_profiles']

    # Mirror principle_2's Qwen vLLM inference: structured-output decoding except
    # for Qwen chain-of-thought, up to 3 attempts, disabling thinking on retry.
    use_structured_output = not (model.startswith('Qwen/') and cot)
    response_schema = request['response_schema'] if use_structured_output else None

    for attempt in range(3):
        ans = None
        try:
            attempt_cot_config = retry_cot_config(cot_config, attempt) if cot else cot_config
            if cot and attempt_cot_config and attempt_cot_config != cot_config:
                print(f'Retrying with qwen_enable_thinking={attempt_cot_config.get("qwen_enable_thinking")}, max_new_tokens={attempt_cot_config.get("max_new_tokens")}')
            ans = get_response(request['prompt'], model, temperature=temperature, system_prompt="You are a helpful assistant", response_schema=response_schema, cot=cot, cot_config=attempt_cot_config)
            results = combined_parse_selection_response(ans)

            filtered_results = []
            for result in results:
                v = result['name']
                if v in G.nodes():
                    result['similarity'] = combined_measure_similarity(profiles[t], profiles[v])
                    result['dropped'] = v in dropped_nodes
                    filtered_results.append(result)

            # Strict acceptance like principle_2: if a selection was requested but no
            # valid candidate came back (e.g. a truncated CoT output), treat it as a
            # parse failure so the retry fires instead of silently recording nothing.
            if num_choices > 0 and not filtered_results:
                raise ValueError(f'No valid candidate selected among {request["candidate_names"]!r}')

            if attempt > 0:
                for result in filtered_results:
                    result['retry_attempt'] = attempt + 1
                    if attempt_cot_config and 'max_new_tokens' in attempt_cot_config:
                        result['retry_max_new_tokens'] = attempt_cot_config['max_new_tokens']

            print(f'Node: {t}, Links: {filtered_results}')

            candidates = []

            for candidate_profile in candidate_profiles:
                similarity = combined_measure_similarity(profiles[t], candidate_profile)
                candidates.append({'name' : candidate_profile['name'], 'similarity' : similarity})

            return filtered_results, candidates
        except Exception as e:
            print_llm_parse_error(e, ans, context=f'combined_select_neighbor attempt={attempt + 1}, node={t}, model={model}')

    return [], []

def combined_dataset_available(name, datasets_dir='datasets'):
    """Return True if the local data files for a dataset are present. andorra and
    mobiled are proprietary and not shipped with the repo, so runs over them are
    skipped when their files are missing instead of crashing."""
    if name == 'andorra':
        required = [os.path.join(datasets_dir, 'andorra', 'andorra.txt')]
    elif name == 'mobiled':
        required = [os.path.join(datasets_dir, 'mobiled', 'mobiled.txt')]
    else:
        required = [os.path.join(datasets_dir, 'facebook100', f'{name}.mat')]
    return all(os.path.exists(path) for path in required)


def combined_run_network_formation_experiment(name, num_simulations, outfile, temperatures=None, method='llm', num_choices=1, num_samples=-1, num_nodes_samples=-1, model='gpt-5-mini', dataloader_fn=None, sampling_strategy='random', cot=False, cot_config=None):
    try:
        networks = dataloader_fn()
    except FileNotFoundError as e:
        print(f'Skipping {name}: dataset files not found ({e}).')
        return

    if temperatures is None:
        temperatures = [None]

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    maybe_reset_outfile(outfile)
    if IGNORE_EXISTING_OUTPUTS:
        # Reset mode regenerates from zero, so discard any stale resume checkpoints.
        _combined_remove_all_checkpoints(outfile)

    saved_scenarios = set()

    if os.path.exists(outfile):
        with open(outfile) as f:
            lines = f.read().splitlines()

        for line in lines:
            scenario = json.loads(line)
            saved_scenarios.add((scenario['name'], scenario['ego'], scenario['simulation'], scenario['temperature'], scenario['num_samples'], scenario['num_choices']))

    expected_scenarios = {
        (name, ego, i, 'default' if temperature is None else temperature, num_samples, num_choices)
        for ego in networks
        for i in range(num_simulations)
        for temperature in temperatures
    }

    if expected_scenarios.issubset(saved_scenarios):
        print(f'All simulations already completed for {outfile}. Skipping inference.')
        return

    print(f'Loaded {len(saved_scenarios)} completed simulations from {outfile}')

    f = open(outfile, 'a+')

    for ego, G0 in networks.items():
        for i in range(num_simulations):
            for temperature in temperatures:
                temperature_label = 'default' if temperature is None else temperature
                checkpoint_path = _combined_checkpoint_path(outfile, ego, i, temperature_label, num_samples, num_choices)
                if (name, ego, i, temperature_label, num_samples, num_choices) in saved_scenarios:
                    print(f'Skipping simulation for name={name}, ego={ego}, i={i}, temperature={temperature_label}, num_choices={num_choices}, num_samples={num_samples}, method={method}')
                    _combined_remove_checkpoint(checkpoint_path)
                    continue
                else:
                    print(f'Running simulation for name={name}, ego={ego}, i={i}, temperature={temperature_label}, num_choices={num_choices}, num_samples={num_samples}, method={method}')

                    Gs, results, candidates = combined_network_growth(G0, temperature=temperature, method=method, name=name, num_choices=num_choices, num_samples=num_samples, num_nodes_samples=num_nodes_samples, model=model, sampling_strategy=sampling_strategy, cot=cot, cot_config=cot_config, checkpoint_path=checkpoint_path)

                    temp = {
                        'name' : name,
                        'ego' : ego,
                        'temperature' : temperature_label,
                        'simulation' : i,
                        'num_choices' : num_choices,
                        'num_samples' : num_samples,
                        'graphs' : [nx.to_dict_of_dicts(G) for G in [Gs[0], Gs[-1]]],
                        'results' : results,
                        'candidates' : candidates,
                        'model' : model,
                        'sampling_strategy' : sampling_strategy,
                        'cot' : cot
                    }

                    f.write(json.dumps(temp) + '\n')
                    f.flush()
                    # The scenario is now recorded in the outfile; drop its checkpoint.
                    _combined_remove_checkpoint(checkpoint_path)

                if method != 'llm':
                    break

    f.close()

def combined_build_cot_calibration_requests(experiment, output_dir, default_temperatures, sample_size=20, seed=0):
    dataloader_fn = experiment['dataloader_fn']
    model = experiment['model']
    num_samples = experiment.get('num_samples', -1)
    num_choices = experiment.get('num_choices', 1)
    num_nodes_samples = experiment.get('num_nodes_samples', -1)
    sampling_strategy = experiment.get('sampling_strategy', 'random')
    name = experiment.get('name', 'combined')
    temperature = experiment.get('temperatures', default_temperatures)[0]

    random.seed(seed)
    np.random.seed(seed)

    networks = dataloader_fn()
    ego, G0 = next(iter(networks.items()))
    G = G0.copy()
    profiles = nx.get_node_attributes(G, 'features')

    if sampling_strategy == 'link_prediction':
        sk_model = link_prediction.train_link_predictor(G, profiles=profiles, name=name)
    else:
        sk_model = None

    if num_nodes_samples > 0 and num_nodes_samples < len(G):
        nodes = random.sample(list(G.nodes()), num_nodes_samples)
    else:
        nodes = list(G.nodes())

    # Mirror combined_network_growth: drop one neighbor per choice for each node.
    dropped_edges = []
    for v in nodes:
        dropped_v_edges = []
        for _ in range(num_choices):
            if len(list(G.neighbors(v))) > 0:
                while True:
                    u = random.choice(list(G.neighbors(v)))
                    if (v, u) not in dropped_edges:
                        dropped_v_edges.append((v, u))
                        G.remove_edge(v, u)
                        break
        dropped_edges.append(dropped_v_edges)

    rng = random.Random(seed)
    indices = list(range(len(nodes)))
    rng.shuffle(indices)
    indices = indices[:sample_size]

    requests = []
    for i in indices:
        t = nodes[i]
        request = combined_build_selection_request(
            G,
            t,
            profiles,
            num_choices=max(1, len(dropped_edges[i])),
            dropped_nodes=[u for (_, u) in dropped_edges[i]],
            num_samples=num_samples,
            model=model,
            sampling_strategy=sampling_strategy,
            sk_model=sk_model,
        )
        requests.append((t, request))

    record = {
        'model': model,
        'name': name,
        'temperatures': [temperature],
        'cot_config': experiment.get('cot_config'),
    }
    return record, requests


def combined_run_cot_budget_calibration(
    dataloader_fn,
    output_dir,
    model,
    default_temperatures,
    default_cot_config,
    name='combined',
    num_samples=-1,
    num_choices=1,
    num_nodes_samples=-1,
    sampling_strategy='random',
    cot_config=None,
    run_experiments=True,
    calibrate=True,
    calibration_sample_size=20,
    calibration_max_new_tokens=65536,
    calibration_percentile=0.90,
    calibration_margin=1.5,
    retry_token_buckets=(8192, 16384, 32768, 65536),
    calibration_seed=0,
    calibration_filename='combined_model_cot_budget_calibration.json',
):
    experiments = [{
        'name': name,
        'model': model,
        'COT': True,
        'dataloader_fn': dataloader_fn,
        'temperatures': list(default_temperatures),
        'cot_config': cot_config or default_cot_config,
        'num_samples': num_samples,
        'num_choices': num_choices,
        'num_nodes_samples': num_nodes_samples,
        'sampling_strategy': sampling_strategy,
    }]
    return run_cot_budget_calibration(
        experiments,
        output_dir,
        default_temperatures,
        default_cot_config,
        build_calibration_requests=combined_build_cot_calibration_requests,
        parse_response=lambda ans, request: combined_parse_selection_response(ans),
        calibration_filename=calibration_filename,
        run_experiments=run_experiments,
        calibrate=calibrate,
        calibration_sample_size=calibration_sample_size,
        calibration_max_new_tokens=calibration_max_new_tokens,
        calibration_percentile=calibration_percentile,
        calibration_margin=calibration_margin,
        retry_token_buckets=retry_token_buckets,
        calibration_seed=calibration_seed,
    )


def combined_draw_graph(G, ax, communities=None, palette=None):

    pos = nx.spring_layout(G)

    netgraph.Graph(G, node_layout=pos, node_color='#d35400', node_size=2.5, edge_color='#34495e', edge_width=1, ax=ax)

    ax.set_axis_off()

def combined_generate_regression_table(filename, outfile, bias=True, log_transform=True, exclude_log=[]):

    palette = ['#d35400', '#34495e', '#2980b9', '#e67e22', '#f1c40f', '#7f8c8d', '#27ae60', '#16a085', '#bdc3c7', '#1abc9c', '#2ecc71', '#3498db', '#9b59b6', '#8e44ad', '#ecf0f1']

    with open(filename) as f:
        lines = f.read().splitlines()

    data = []

    for line in lines:
        data.append(json.loads(line))

    feature_names = ['degree', 'common_attributes', 'common_neighbors']

    regression_table_df = []

    names = set([d['name'] for d in data])

    for d in data:

        log_likelihoods = {}

        for num_features in range(len(feature_names) + 1):
            for feature_combination in itertools.combinations(feature_names, num_features):
                feature_combination = list(feature_combination)
                theta, standard_errors, log_likelihood, _, probabilities, ame, sdame, _ = dcm.fit_discrete_choice_model((d['results'], d['candidates']), feature_names=feature_combination, bias=bias, log_transform=log_transform, exclude_log=exclude_log, calculate_p_values=True, calculate_average_marginal_effects=True, input_type='results_candidates')


                temp = {
                    'Name' : d["name"],
                    'Ego' : d["ego"],
                    'Temperature' : d["temperature"],
                    'Simulation' : d["simulation"],
                    'Number of Choices' : d["num_choices"],
                    'Number of Samples' : d["num_samples"],
                    'Independent Variable' : feature_combination,
                    'Coefficients' : theta[:-1].tolist(),
                    'Standard Errors' : standard_errors[:-1].tolist(),
                    'Log Likelihood' : log_likelihood,
                    'Probabilities' : probabilities.tolist() if probabilities is not None else None,
                    'AME' : ame.tolist() if ame is not None else None,
                    'SE AME' : sdame.tolist() if sdame is not None else None,
                }

                log_likelihoods[tuple(sorted(feature_combination))] = log_likelihood
                p_values = np.array([1 - stats.chi2.cdf(2 * (log_likelihood - log_likelihoods[tuple(sorted(feature_combination[:i] + feature_combination[i + 1:]))]), 1) for i in range(len(feature_combination))])

                print(f'features: {feature_combination}, theta: {theta}, standard_errors: {standard_errors}, p-values: {p_values}, log_likelihood: {log_likelihood}, AME: {ame}, AME (SE): {sdame}')

                temp['P-values'] = p_values.tolist()



                regression_table_df.append(temp)

    regression_table_df = pd.DataFrame.from_records(regression_table_df)

    regression_table_df.to_excel(outfile)

def combined_compare_models(filenames1, filenames2, bias=True, log_transform=True, exclude_log=[], heatmap=True, suptitle='', supxlabel='', supylabel='', outfile='figures/comparison_between_models.png'):

    palette = ['#d35400', '#34495e', '#2980b9', '#e67e22', '#f1c40f', '#7f8c8d', '#27ae60', '#16a085', '#bdc3c7', '#1abc9c', '#2ecc71', '#3498db', '#9b59b6', '#8e44ad', '#ecf0f1']

    feature_names = ['degree', 'common_attributes', 'common_neighbors']

    records_between_models = []
    records_effects = []

    for filename1, filename2 in zip(filenames1, filenames2):

        basename1 = filename1.split('+')
        basename2 = filename2.split('+')

        if len(basename1) == 3:
            model1 = basename1[-2] + '+' + basename1[-1]
        elif len(basename1) == 2:
            model1 = basename1[-1]

        if len(basename2) == 3:
            model2 = basename2[-2] + '+' + basename2[-1]
        elif len(basename2) == 2:
            model2 = basename2[-1]

        # remove file extension from model1
        model1 = model1.replace('.jsonl', '')
        model2 = model2.replace('.jsonl', '')


        with open(filename1) as f:
            lines1 = f.read().splitlines()

        with open(filename2) as f:
            lines2 = f.read().splitlines()

        data1 = []

        for line in lines1:
            data1.append(json.loads(line))

        data2 = []

        for line in lines2:
            data2.append(json.loads(line))


        for d1, d2 in zip(data1, data2):
            if d1['name'] != d2['name']:
                print(f'Skipping {d1["name"]} and {d2["name"]} as they are not the same scenario')
                continue
            else:
                print(f'Comparing {model1} and {model2} on {d1["name"]} at temperature {d1["temperature"]}')

            distance_mean, distance_std, theta_spearman, theta1, theta2, sd1, sd2, ame1, ame2, sdame1, sdame2, p_values_ame1, p_values_ame2 = dcm.combined_compare_models((d1['results'], d1['candidates']), (d2['results'], d2['candidates']), on='Alternative Set', method='tv', bias=bias, feature_names=feature_names, log_transform=log_transform, exclude_log=exclude_log, calculate_p_values=True, calculate_average_marginal_effects=True, input_type='results_candidates')

            records_between_models.append({
                'Name' : d1['name'].capitalize(),
                'Model1': COMBINED_RENAME_MODELS.get(model1, model1),
                'Model2': COMBINED_RENAME_MODELS.get(model2, model2),
                'TV Distance': distance_mean,
                'TV Distance Std': distance_std,
                'Effect Spearman Correlation': theta_spearman,
                'Theta1': theta1,
                'Theta2': theta2,
                'StandardError1': sd1,
                'StandardError2': sd2,
                'AME1' : ame1,
                'AME2' : ame2,
                'StandardErrorAME1': sdame1,
                'StandardErrorAME2': sdame2,
                'P-values AME1': p_values_ame1,
                'P-values AME2': p_values_ame2
            })

            for j, feature in enumerate(feature_names):

                stars_1 = '***' if p_values_ame1[j] < 0.001 else '**' if p_values_ame1[j] < 0.01 else '*' if p_values_ame1[j] < 0.05 else ''
                stars_2 = '***' if p_values_ame2[j] < 0.001 else '**' if p_values_ame2[j] < 0.01 else '*' if p_values_ame2[j] < 0.05 else ''

                ame1_formatted = f"{ame1[j]:.2f}{stars_1} ({sdame1[j]:.2f})"
                ame2_formatted = f"{ame2[j]:.2f}{stars_2} ({sdame2[j]:.2f})"

                records_effects.append({
                    'Name' : d1['name'].capitalize(),
                    'Model': COMBINED_RENAME_MODELS.get(model1, model1),
                    'Label' : supxlabel,
                    'Feature': feature,
                    'AME': ame1_formatted,
                })

                records_effects.append({
                    'Name' : d2['name'].capitalize(),
                    'Model': COMBINED_RENAME_MODELS.get(model2, model2),
                    'Label' : supylabel,
                    'Feature': feature,
                    'AME': ame2_formatted
                })

    records_between_models_df = pd.DataFrame.from_records(records_between_models)
    records_effects_df = pd.DataFrame.from_records(records_effects)
    records_effects_df.drop_duplicates(subset=['Name', 'Model', 'Label', 'Feature'], inplace=True)

    records_between_models_df.to_excel('tables/comparison_between_models.xlsx', index=False)

    names = set(records_between_models_df['Name'].tolist())

    records_effects_df.to_excel('tables/effects_between_models.xlsx', index=False)

    if heatmap:
        fig, ax = plt.subplots(2, len(names), figsize=(3*len(names), 6), squeeze=False)

        for i, name in enumerate(names):

            ax[0, i].set_title(name.capitalize())

            sns.heatmap(records_between_models_df.query(f'Name == "{name}"').pivot(index='Model1', columns='Model2', values='Effect Spearman Correlation'),
                annot=True, fmt='.2f', ax=ax[0, i],
                cbar=(i == len(names) - 1),
                # cbar_kws={'label': 'Spearman Correlation'},
                vmin=-1, vmax=1)

            ax[0, i].set_xlabel('')
            ax[0, i].set_ylabel('')


            sns.heatmap(records_between_models_df.query(f'Name == "{name}"').pivot(index='Model1', columns='Model2', values='TV Distance'),
                annot=True, fmt='.2f', ax=ax[1, i],
                cbar=(i == len(names) - 1),
                # cbar_kws={'label': 'TV Distance'},
                vmin=0, vmax=1)

            ax[1, i].set_xlabel('')
            ax[1, i].set_ylabel('')

            sns.despine(ax=ax[0, i])
            sns.despine(ax=ax[1, i])

        ax[0, 0].set_ylabel('Spearman Correlation')
        ax[1, 0].set_ylabel('TV Distance')

        for i in range(ax.shape[0]-1):
            for j in range(ax.shape[1]):
                ax[i, j].set_xticklabels([])

        for j in range(1, ax.shape[1]):
            for i in range(ax.shape[0]):
                ax[i, j].set_yticklabels([])


    else:

        fig, ax = plt.subplots(1, 2, figsize=(12, 3), squeeze=False)

        sns.barplot(x='Model2', y='Effect Spearman Correlation', hue='Name', data=records_between_models_df, ax=ax[0, 0], palette=palette)
        ax[0, 0].set_xlabel('')
        ax[0, 0].set_ylabel('Spearman Correlation')
        # remove legend
        ax[0, 0].legend_.remove()

        sns.barplot(x='Model2', y='TV Distance', hue='Name', data=records_between_models_df, ax=ax[0, 1], palette=palette)
        ax[0, 1].set_xlabel('')
        ax[0, 1].set_ylabel('TV Distance')

        # move legend to the right
        ax[0, 1].legend_.remove()
        ax[0, 1].legend(loc='upper left', bbox_to_anchor=(1, 1), title='Name')

        sns.despine(ax=ax[0, 0])
        sns.despine(ax=ax[0, 1])

        ax[0, 1].set_ylim(0, 1)
        ax[0, 0].set_ylim(-1, 1)

    fig.suptitle(suptitle)
    fig.supxlabel(supxlabel)
    fig.supylabel(supylabel)

    fig.subplots_adjust(top=0.85)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches='tight')

def combined_build_regression_table_df(filenames, full=False):
    """Build the formatted regression-coefficient DataFrames (main + AME) shared by
    the LaTeX and markdown regression-table writers."""
    if isinstance(filenames, str):
        filenames = [filenames]

    frames = [pd.read_excel(filename) for filename in filenames if os.path.exists(filename)]
    if not frames:
        return None, None

    regression_table_df = pd.concat(frames)

    regression_table_df = regression_table_df.query('`Independent Variable` != "[]"')

    table_rows_df = []
    table_rows_ame_df = []

    ego_row = True

    temperature2idx = {}


    for i, row in regression_table_df.iterrows():
        temp = {}
        temp_ame = {}
        if row['Ego'] == -1:
            ego_row = False
        else:
            temp['Ego'] = row['Ego']
            temp_ame['Ego'] = row['Ego']
        temp['Temperature'] = str(row['Temperature'])
        temp_ame['Temperature'] = str(row['Temperature'])

        if row['Temperature'] not in temperature2idx:
            temperature2idx[row['Temperature']] = len(temperature2idx)

        independent_variables = ast.literal_eval(row['Independent Variable'])

        if (not full and len(independent_variables) ==  3) or full:

            p_values = ast.literal_eval(row['P-values'])
            coefficients = ast.literal_eval(row['Coefficients'])
            standard_errors = ast.literal_eval(row['Standard Errors'])
            ame = ast.literal_eval(row['AME']) if 'AME' in row else None
            sdame = ast.literal_eval(row['SE AME']) if 'SE AME' in row else None

            for j, feat_name in enumerate(independent_variables):
                stars = '***' if float(p_values[j]) < 0.001 else '**' if float(p_values[j]) < 0.01 else '*' if float(p_values[j]) < 0.05 else ''
                temp[f'{feat_name.replace("_", " ").capitalize()}'] = f"{float(coefficients[j]):.2f}{stars} ({float(standard_errors[j]):.1g})"
                temp_ame[f'{feat_name.replace("_", " ").capitalize()}'] = f"{float(ame[j]):.2f} ({float(sdame[j]):.1g})"

            temp['Log Likelihood'] = f"{row['Log Likelihood']:,.2f}"
            temp['AIC'] = f'{2 * (len(independent_variables) + 1) - 2 * row["Log Likelihood"]:,.2f}'

            table_rows_df.append(temp)
            table_rows_ame_df.append(temp_ame)

    table_rows_df = pd.DataFrame.from_records(table_rows_df, columns=['Ego'] if ego_row else [] +  ['Temperature', 'Degree', 'Common attributes', 'Common neighbors', 'Log Likelihood', 'AIC'])
    table_rows_ame_df = pd.DataFrame.from_records(table_rows_ame_df, columns=['Ego'] if ego_row else [] +  ['Temperature', 'Degree', 'Common attributes', 'Common neighbors'])
    table_rows_df = table_rows_df.fillna(' ')
    table_rows_ame_df = table_rows_ame_df.fillna(' ')

    return table_rows_df, table_rows_ame_df


def combined_pretty_print_regression_table(filenames, outfile, full=False):
    table_rows_df, table_rows_ame_df = combined_build_regression_table_df(filenames, full=full)
    if table_rows_df is None:
        print(f'combined_pretty_print_regression_table: no input tables found for {outfile}.')
        return

    table_rows_df.to_latex(outfile, index=False, escape=True, column_format='lcccccc')

    table_rows_ame_df.to_latex(outfile.replace('.tex', '_ame.tex'), index=False, escape=True, column_format='lcccccc')

def combined_modularity_change(filenames, subgraph=False):

    for filename in filenames:

        with open(filename) as f:
            lines = f.read().splitlines()

        for line in lines:
            temp = json.loads(line)

            G0 = nx.from_dict_of_dicts(temp['graphs'][0])
            G1 = nx.from_dict_of_dicts(temp['graphs'][-1])

            if subgraph:
                H = nx.difference(G1, G0)
                H.remove_nodes_from(list(nx.isolates(H)))
                G0 = nx.subgraph(G0, H.nodes())
                G1 = nx.subgraph(G1, H.nodes())

            modularities0 = []
            modularities1 = []

            for seed in range(10):
                communities0 = nx.community.louvain_communities(G0, seed=seed)
                modularities0.append(nx.community.modularity(G0, communities0))

                communities1 = nx.community.louvain_communities(G1, seed=seed)
                modularities1.append(nx.community.modularity(G1, communities1))


            t, p = stats.ttest_ind(modularities0, modularities1, equal_var=False, alternative='less')

            print(f'{temp["name"]}, {temp["temperature"]}, T-test: {t}, p-value: {p}')

def combined_lcc(G):
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    return G.subgraph(Gcc[0])

def combined_small_worldness(filenames, name, dataloader_fn, subgraph=False):

    networks = dataloader_fn()

    G_initial = networks[list(networks.keys())[0]]

    # LCC subgraph
    G_initial = combined_lcc(G_initial)

    average_shortest_path_length_initial = nx.average_shortest_path_length(G_initial)
    clustering_coefficient_initial = nx.average_clustering(G_initial)

    for filename in filenames:

            with open(filename) as f:
                lines = f.read().splitlines()

            for line in lines:
                temp = json.loads(line)

                G0 = nx.from_dict_of_dicts(temp['graphs'][0])
                G1 = nx.from_dict_of_dicts(temp['graphs'][-1])
                G1 = combined_lcc(G1)

                average_shortest_path_length = nx.average_shortest_path_length(G1)
                clustering_coefficient = nx.average_clustering(G1)

                average_shortest_path_length_initial_change = (average_shortest_path_length - average_shortest_path_length_initial) / average_shortest_path_length_initial * 100
                clustering_coefficient_initial_change = (clustering_coefficient - clustering_coefficient_initial) / clustering_coefficient_initial * 100

                print(f'{temp["name"]}, {temp["temperature"]}, Average Shortest Path Length Change: {average_shortest_path_length_initial_change}, Clustering Coefficient Change: {clustering_coefficient_initial_change}')

def combined_build_graph_statistics_change_df(filenames, subgraph=False):
    """Compute the per-graph statistics-change records shared by the LaTeX and
    markdown writers."""
    records = []

    for filename in filenames:

            basename1 = filename.split('+')

            if len(basename1) == 3:
                model = basename1[-2] + '+' + basename1[-1]
            elif len(basename1) == 2:
                model = basename1[-1]


            # remove file extension from model
            model = model.replace('.jsonl', '')

            with open(filename) as f:
                lines = f.read().splitlines()

            for line in lines:
                temp = json.loads(line)

                name = temp['name']


                G0 = nx.from_dict_of_dicts(temp['graphs'][0])
                G1 = nx.from_dict_of_dicts(temp['graphs'][-1])

                print(f'Analyzing {name} (n = {len(G0)}, m = {len(G0.edges())}) at temperature {temp["temperature"]} using model {model}')

                degrees_G0 = np.array([d for _, d in G0.degree()])
                degrees_G1 = np.array([d for _, d in G1.degree()])

                # 2-sample KS test
                ks_statistic, p_value = stats.ks_2samp(degrees_G0, degrees_G1)

                # get sizes of connected components
                cc_sizes_G0 = [len(c) for c in nx.connected_components(G0)]
                cc_sizes_G1 = [len(c) for c in nx.connected_components(G1)]

                # 2-sample KS test for connected components sizes
                ks_statistic_cc, p_value_cc = stats.ks_2samp(cc_sizes_G0, cc_sizes_G1)

                # distribution of singular values
                if len(G0) < 10000:
                    svd_G0 = np.linalg.svd(nx.to_numpy_array(G0), compute_uv=False)
                    svd_G1 = np.linalg.svd(nx.to_numpy_array(G1), compute_uv=False)
                else:
                    svd_G0, _ = scipy.sparse.linalg.eigsh(nx.to_scipy_sparse_array(G0), k=10)
                    svd_G1, _ = scipy.sparse.linalg.eigsh(nx.to_scipy_sparse_array(G1), k=10)

                ks_statistic_svd, p_value_svd = stats.ks_2samp(svd_G0, svd_G1)

                # distributions of local clustering coefficients
                clustering_coeffs_G0 = np.array(list(nx.clustering(G0).values()))
                clustering_coeffs_G1 = np.array(list(nx.clustering(G1).values()))

                ks_statistic_clustering, p_value_clustering = stats.ks_2samp(clustering_coeffs_G0, clustering_coeffs_G1)

                m0 = G0.number_of_edges()
                m1 = G1.number_of_edges()

                number_of_new_edges_added = abs(m1 - m0) / m0 * 100

                print(f'Name: {temp["name"]}, Temperature: {temp["temperature"]}')

                records.append({
                    'Name' : temp['name'],
                    'Model' : COMBINED_RENAME_MODELS.get(model, model),
                    'Temp' : temp['temperature'],
                    'Degree Distribution (KS)' : ks_statistic,
                    'Degree Distribution (P-value)' : p_value,
                    'Sizes of CCs (KS)' : ks_statistic_cc,
                    'Sizes of CCs (P-value)' : p_value_cc,
                    'Adjacency Spectrum (KS)' : ks_statistic_svd,
                    'Adjacency Spectrum (P-value)' : p_value_svd,
                    'Local Clustering Coefficient (KS)' : ks_statistic_clustering,
                    'Local Clustering Coefficient (P-value)' : p_value_clustering,
                    'Number of New Edges Added (%)' : number_of_new_edges_added,
                })

    return pd.DataFrame.from_records(records)


def combined_graph_statistics_change(filenames, outfile, subgraph=False):
    records_df = combined_build_graph_statistics_change_df(filenames, subgraph=subgraph)

    with open(outfile, 'w') as f:
        f.write(records_df.to_latex(index=False, escape=True, column_format='lccccccccccc', float_format='%.1g'))


def combined_write_markdown_report(
    output_dir,
    regression_sections=None,
    graph_stats_sections=None,
    filename='combined_model_results.md',
    title='Real-world Networks (Combined Model) — Results',
    full=False,
    timestamp=None,
):
    """Consolidate the combined-model results into a single markdown file written to
    ``output_dir``.

    ``regression_sections`` and ``graph_stats_sections`` are each a list of dicts:
      - regression: ``{'label': str, 'xlsx_files': [...], 'full': bool (optional)}``
        -> a discrete-choice regression-coefficient table (Degree, Common
        attributes, Common neighbors, Log Likelihood, AIC).
      - graph stats: ``{'label': str, 'jsonl_files': [...], 'subgraph': bool}``
        -> a table of how graph statistics change from the initial to the final
        network (KS statistics for degree / component sizes / spectrum / clustering
        and the percentage of new edges added).

    Missing input files are skipped, so the report reflects whatever has actually
    been generated in this session."""
    sections = []

    for spec in (regression_sections or []):
        xlsx_files = [f for f in spec.get('xlsx_files', []) if os.path.exists(f)]
        if not xlsx_files:
            continue
        table_rows_df, _ = combined_build_regression_table_df(xlsx_files, full=spec.get('full', full))
        if table_rows_df is None or table_rows_df.empty:
            continue
        sections.append(f'## Regression — {spec["label"]}')
        sections.append(_dataframe_to_markdown(table_rows_df))

    for spec in (graph_stats_sections or []):
        jsonl_files = [f for f in spec.get('jsonl_files', []) if os.path.exists(f)]
        if not jsonl_files:
            continue
        records_df = combined_build_graph_statistics_change_df(jsonl_files, subgraph=spec.get('subgraph', False))
        if records_df is None or records_df.empty:
            continue
        sections.append(f'## Graph Statistics Change — {spec["label"]}')
        sections.append(_dataframe_to_markdown(records_df))

    return _write_markdown_report_file(output_dir, filename, title, sections, timestamp)


def combined_measure_relative_increase(filenames):

    for filename in filenames:

        with open(filename) as f:
            lines = f.read().splitlines()

        data = []

        for line in lines:
            data.append(json.loads(line))

        for d in data:
            total, count = 0, 0
            for results in d["results"]:
                for result in results:
                    count += int(result['dropped'])
                    total += 1

            accuracy = count / total * 100
            random_guess = 100 / d["num_choices"]
            relative_increase = (accuracy - random_guess) / random_guess * 100

            print(f'{d["name"]}, {d["temperature"]}, {d["simulation"]}, Relative Increase in Accuracy % = {relative_increase}')


# --- End combined real-world network utilities ---
