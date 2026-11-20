"""Run the frozen Qwen base model on the development prompts and save raw JSONL."""

import argparse
import json
from pathlib import Path

from render_prompts import DEFAULT_MANIFEST, load_development, render_prompt


MODEL_ID = "Qwen/Qwen3-0.6B-Base"
MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
MAX_NEW_TOKENS = 64
GENERATION_PROTOCOL = "plain-greedy-v1"


def load_qwen(device, *, fp32=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    dtype = torch.float16 if device == "cuda" and not fp32 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    return model.eval(), tokenizer


def prompt_token_ids(tokenizer, prompt):
    return tokenizer.encode(prompt, add_special_tokens=False, truncation=False)


def greedy_answer(model, tokenizer, prompt, device):
    return greedy_continuation(model, tokenizer, prompt, device)[0]


def greedy_continuation(model, tokenizer, prompt, device):
    """Return raw answer text and the generated token IDs before EOS removal."""
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
    eos = tokenizer.eos_token_id
    ended = bool(continuation) and continuation[-1] == eos
    reason = "eos" if ended else "token_cap" if len(continuation) == MAX_NEW_TOKENS else "unknown"
    answer = tokenizer.decode(
        continuation[:-1] if ended else continuation,
        skip_special_tokens=False, clean_up_tokenization_spaces=False,
    )
    return answer, {"new_token_ids": continuation, "eos_token_id": eos, "stop_reason": reason}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--generation-metadata", type=Path, help="optional token-level JSON sidecar")
    args = parser.parse_args()
    if args.generation_metadata and args.generation_metadata.resolve() == args.output.resolve():
        parser.error("generation metadata and predictions need separate output paths")

    checksum, cases = load_development(args.manifest)
    if len(cases) != 20:
        raise ValueError(f"expected 20 frozen development cases, found {len(cases)}")
    settings = {
        "protocol": GENERATION_PROTOCOL,
        "model": MODEL_ID, "revision": MODEL_REVISION, "manifest_sha256": checksum,
        "device": args.device, "max_new_tokens": MAX_NEW_TOKENS, "case_count": len(cases),
    }
    print(json.dumps(settings), flush=True)
    model, tokenizer = load_qwen(args.device)
    records = []
    evidence = []
    for case in cases:
        prompt = render_prompt(case.task)
        if args.generation_metadata:
            answer, metadata = greedy_continuation(model, tokenizer, prompt, args.device)
            evidence.append({"case_id": case.case_id, **metadata})
        else:
            answer = greedy_answer(model, tokenizer, prompt, args.device)
        records.append({"case_id": case.case_id, "answer": answer})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes("".join(json.dumps(record) + "\n" for record in records).encode("utf-8"))
    if args.generation_metadata:
        args.generation_metadata.parent.mkdir(parents=True, exist_ok=True)
        args.generation_metadata.write_bytes(
            (json.dumps({**settings, "cases": evidence}, indent=2) + "\n").encode("utf-8")
        )


if __name__ == "__main__":
    main()
