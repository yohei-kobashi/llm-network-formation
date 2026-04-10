import json
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
]

SYSTEM_PROMPT = "You are mimicking a real-life person who wants to make friends."
MAX_NEW_TOKENS = 256
TRIALS = 3

_HF_CACHE = {}


def get_hf_client(model_id: str):
    if model_id not in _HF_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        _HF_CACHE[model_id] = (tokenizer, model)

    return _HF_CACHE[model_id]


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

* The name of the person you selected must be one of the names in the input.
* Your output must be JSON only.

```json
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


def generate_raw_response(
    model_id: str,
    prompt: str,
    use_system_prompt: bool,
    temperature=None,
):
    tokenizer, model = get_hf_client(model_id)
    rendered_input = render_chat_input(tokenizer, prompt, use_system_prompt)
    model_inputs = tokenizer(rendered_input, return_tensors="pt")
    device = next(model.parameters()).device
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "pad_token_id": tokenizer.pad_token_id,
    }
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


def normalize_name(value: Any):
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def run_case(model_id: str, name_mode: str, use_system_prompt: bool):
    candidates = build_candidates(name_mode=name_mode)
    candidate_names = {candidate["name"] for candidate in candidates}
    prompt = build_prompt(candidates)

    rendered_input, raw_text = generate_raw_response(
        model_id=model_id,
        prompt=prompt,
        use_system_prompt=use_system_prompt,
        temperature=None,
    )

    parsed = first_json_object(raw_text)
    result = {
        "parse_ok": parsed is not None,
        "raw_text": raw_text,
        "parsed": parsed,
        "candidate_names": sorted(candidate_names, key=str),
        "accepted_by_original_logic": False,
        "accepted_after_numeric_coercion": False,
        "name_type": None,
        "normalized_name_type": None,
    }

    if parsed is not None and isinstance(parsed, dict) and "name" in parsed:
        result["name_type"] = type(parsed["name"]).__name__
        result["accepted_by_original_logic"] = parsed["name"] in candidate_names
        normalized_name = normalize_name(parsed["name"])
        result["normalized_name_type"] = type(normalized_name).__name__
        result["accepted_after_numeric_coercion"] = normalized_name in candidate_names

    return rendered_input, result


def print_case_summary(model_id: str, name_mode: str, use_system_prompt: bool, trial: int, rendered_input: str, result: dict):
    title = f"{model_id} | names={name_mode} | system={use_system_prompt} | trial={trial}"
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    print("candidate_names:", result["candidate_names"])
    print("parse_ok:", result["parse_ok"])
    print("name_type:", result["name_type"])
    print("normalized_name_type:", result["normalized_name_type"])
    print("accepted_by_original_logic:", result["accepted_by_original_logic"])
    print("accepted_after_numeric_coercion:", result["accepted_after_numeric_coercion"])
    print("rendered_input_preview:", rendered_input[:500].replace("\n", "\\n"))
    print("raw_text:")
    print(result["raw_text"])
    print("parsed:")
    print(result["parsed"])
    print()


def main():
    print("If Qwen fails only for integer candidate names but succeeds for string candidate names,")
    print("the likely cause is that principle_1.ipynb rejects stringified numeric ids.")
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
