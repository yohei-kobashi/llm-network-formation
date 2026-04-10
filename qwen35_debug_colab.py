import json
import re
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    import xgrammar as xgr
except ImportError:
    xgr = None


MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
]

SYSTEM_PROMPT = "You are mimicking a real-life person who wants to make friends."
MAX_NEW_TOKENS = 256
TRIALS = 3

_HF_CACHE = {}


def require_xgrammar():
    if xgr is None:
        raise RuntimeError(
            "xgrammar is required for JSON schema constrained decoding. "
            "In Colab, run: pip install xgrammar"
        )


def get_hf_client(model_id: str):
    if model_id not in _HF_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        _HF_CACHE[model_id] = (tokenizer, model, config)

    return _HF_CACHE[model_id]


def infer_vocab_size(tokenizer, model, config):
    candidate_values = []

    for attr in ("vocab_size",):
        value = getattr(config, attr, None)
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
    return source, vocab_size, candidate_values


def build_candidates(name_mode: str = "int"):
    if name_mode == "int":
        return [
            {"name": 0, "friends": [1, 2]},
            {"name": 1, "friends": [0]},
            {"name": 2, "friends": [0]},
            {"name": 3, "friends": []},
        ]

    if name_mode == "string":
        return [
            {"name": "0", "friends": ["1", "2"]},
            {"name": "1", "friends": ["0"]},
            {"name": "2", "friends": ["0"]},
            {"name": "3", "friends": []},
        ]

    raise ValueError(f"Unsupported name_mode: {name_mode}")


def build_prompt(candidates, environment=None, role="neighbors", cot=False):
    allowed_names = [candidate["name"] for candidate in candidates]
    allowed_names_json = json.dumps(allowed_names, ensure_ascii=False)

    if cot:
        output_format = """
{
    "reason": "reason for selecting the person",
    "name": "name of the person you selected"
}
"""
    else:
        output_format = """
{
    "name": "name of the person you selected",
    "reason": "reason for selecting the person"
}
"""

    return f"""
# Task
{"You are in a " + environment + "." if environment else ""}Your task is to select a person to be {role} with.

# Input
The input is a list of dictionaries.

The profiles are given below after chevrons:

<PROFILES>
{json.dumps(candidates, separators=(",", ":"))}
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
* Your first character must be {{ and your last character must be }}.
""".strip()


def render_chat_input(tokenizer, prompt: str, use_system_prompt: bool):
    messages = []
    if use_system_prompt:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


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


def generate_raw_response(
    model_id: str,
    prompt: str,
    use_system_prompt: bool,
    response_schema,
    temperature=None,
):
    require_xgrammar()
    tokenizer, model, config = get_hf_client(model_id)
    rendered_input = render_chat_input(tokenizer, prompt, use_system_prompt)
    model_inputs = tokenizer(rendered_input, return_tensors="pt")
    device = next(model.parameters()).device
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    vocab_size_source, vocab_size, vocab_size_candidates = infer_vocab_size(tokenizer, model, config)
    print(f"[xgrammar] using vocab_size={vocab_size} from {vocab_size_source}")
    print(f"[xgrammar] vocab size candidates: {vocab_size_candidates}")

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = grammar_compiler.compile_json_schema(json.dumps(response_schema))
    xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)

    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "pad_token_id": tokenizer.pad_token_id,
        "logits_processor": [xgr_logits_processor],
    }
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    if temperature in (None, 0):
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = 0.95

    with torch.no_grad():
        outputs = model.generate(**model_inputs, **generation_kwargs)

    generated_tokens = outputs[0][model_inputs["input_ids"].shape[-1]:]
    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return rendered_input, raw_text


def first_json_object(text: str):
    text = text.strip()

    direct_candidates = [text]
    if "```json" in text:
        direct_candidates.append(text.split("```json", 1)[1].split("```", 1)[0].strip())
    if "```" in text:
        direct_candidates.append(text.split("```", 1)[0].strip())

    for candidate in direct_candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            pass

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                fragment = text[start:i + 1]
                try:
                    return json.loads(fragment)
                except Exception:
                    return None

    return None


def normalize_name(value: Any, candidate_names):
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

    patterns = [
        r"(?i)^person\s+(-?\d+)$",
        r"(?i)^candidate\s+(-?\d+)$",
        r"(?i)^id[_\s-]?(-?\d+)$",
    ]
    for pattern in patterns:
        match = re.fullmatch(pattern, stripped)
        if match:
            normalized = match.group(1)
            if normalized in candidate_names_by_str:
                return candidate_names_by_str[normalized]

    return None


def run_case(model_id: str, name_mode: str, use_system_prompt: bool):
    candidates = build_candidates(name_mode=name_mode)
    candidate_names = {candidate["name"] for candidate in candidates}
    prompt = build_prompt(candidates)
    response_schema = build_response_schema(sorted(candidate_names, key=str))

    rendered_input, raw_text = generate_raw_response(
        model_id=model_id,
        prompt=prompt,
        use_system_prompt=use_system_prompt,
        response_schema=response_schema,
        temperature=None,
    )

    parsed = first_json_object(raw_text)
    result = {
        "parse_ok": parsed is not None,
        "raw_text": raw_text,
        "parsed": parsed,
        "candidate_names": sorted(candidate_names, key=str),
        "response_schema": response_schema,
        "accepted_by_original_logic": False,
        "accepted_after_normalization": False,
        "name_type": None,
        "normalized_name": None,
        "normalized_name_type": None,
    }

    if parsed is not None and isinstance(parsed, dict) and "name" in parsed:
        result["name_type"] = type(parsed["name"]).__name__
        result["accepted_by_original_logic"] = parsed["name"] in candidate_names
        normalized_name = normalize_name(parsed["name"], candidate_names)
        result["normalized_name"] = normalized_name
        if normalized_name is not None:
            result["normalized_name_type"] = type(normalized_name).__name__
            result["accepted_after_normalization"] = normalized_name in candidate_names

    return rendered_input, result


def print_case_summary(model_id: str, name_mode: str, use_system_prompt: bool, trial: int, rendered_input: str, result: dict):
    title = f"{model_id} | names={name_mode} | system={use_system_prompt} | trial={trial}"
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    print("candidate_names:", result["candidate_names"])
    print("response_schema:", json.dumps(result["response_schema"], ensure_ascii=False))
    print("parse_ok:", result["parse_ok"])
    print("name_type:", result["name_type"])
    print("normalized_name:", result["normalized_name"])
    print("normalized_name_type:", result["normalized_name_type"])
    print("accepted_by_original_logic:", result["accepted_by_original_logic"])
    print("accepted_after_normalization:", result["accepted_after_normalization"])
    print("rendered_input_preview:", rendered_input[:500].replace("\n", "\\n"))
    print("raw_text:")
    print(result["raw_text"])
    print("parsed:")
    print(result["parsed"])
    print()


def main():
    print("This script uses JSON schema constrained decoding via xgrammar.")
    print("If it is not installed in Colab, run: pip install xgrammar")
    print()

    cases = [
        ("int", True),
        ("int", False),
        ("string", True),
        ("string", False),
    ]

    for model_id in MODELS:
        for name_mode, use_system_prompt in cases:
            for trial in range(1, TRIALS + 1):
                rendered_input, result = run_case(
                    model_id=model_id,
                    name_mode=name_mode,
                    use_system_prompt=use_system_prompt,
                )
                print_case_summary(
                    model_id=model_id,
                    name_mode=name_mode,
                    use_system_prompt=use_system_prompt,
                    trial=trial,
                    rendered_input=rendered_input,
                    result=result,
                )


if __name__ == "__main__":
    main()
