"""Run the frozen Qwen base model on the development prompts and save raw JSONL."""

import argparse
import json
from pathlib import Path

from render_prompts import DEFAULT_MANIFEST, load_development, render_prompt


MODEL_ID = "Qwen/Qwen3-0.6B-Base"
MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
MAX_NEW_TOKENS = 64


def load_qwen(device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    return model.eval(), tokenizer


def prompt_token_ids(tokenizer, prompt):
    return tokenizer.encode(prompt, add_special_tokens=False, truncation=False)


def greedy_answer(model, tokenizer, prompt, device):
    import torch
    from transformers import GenerationConfig

    prompt_ids = prompt_token_ids(tokenizer, prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    config = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    with torch.inference_mode():
        sequences = model.generate(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids), generation_config=config,
        )
    continuation = sequences[0, len(prompt_ids):].tolist()
    if continuation and continuation[-1] == tokenizer.eos_token_id:
        continuation.pop()
    return tokenizer.decode(continuation, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    checksum, cases = load_development(args.manifest)
    if len(cases) != 20:
        raise ValueError(f"expected 20 frozen development cases, found {len(cases)}")
    print(json.dumps({
        "model": MODEL_ID, "revision": MODEL_REVISION, "manifest_sha256": checksum,
        "device": args.device, "max_new_tokens": MAX_NEW_TOKENS, "case_count": len(cases),
    }), flush=True)
    model, tokenizer = load_qwen(args.device)
    records = [
        {"case_id": case.case_id,
         "answer": greedy_answer(model, tokenizer, render_prompt(case.task), args.device)}
        for case in cases
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes("".join(json.dumps(record) + "\n" for record in records).encode("utf-8"))


if __name__ == "__main__":
    main()
