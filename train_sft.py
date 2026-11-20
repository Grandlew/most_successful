"""Run a bounded response-only SFT check on four cases from the train split."""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from baseline_qwen import MAX_NEW_TOKENS, MODEL_ID, MODEL_REVISION, greedy_answer, load_qwen, prompt_token_ids
from render_prompts import DEFAULT_MANIFEST, load_cases, render_prompt
from verify import InvalidAnswer, evaluate_revision, verify_answer


IGNORE_INDEX = -100
TINY_SET_SIZE = 4
SEED = 42
MAX_SEQUENCE_TOKENS = 512
STEP_CAP = 100
CHECK_INTERVAL = 10
LEARNING_RATE = 5e-4


@dataclass(frozen=True)
class TrainingExample:
    input_ids: list[int]
    labels: list[int]
    answer_start: int


def select_tiny_cases(manifest):
    checksum, train = load_cases(manifest, "train")
    if len(train) < TINY_SET_SIZE:
        raise ValueError(f"train must have at least {TINY_SET_SIZE} cases")
    return checksum, sorted(train, key=lambda case: case.case_id)[:TINY_SET_SIZE]


def encode_example(tokenizer, prompt, target, max_length=MAX_SEQUENCE_TOKENS):
    # Generate from this exact prompt prefix. Joint retokenization can merge
    # characters across the boundary and move the first supervised token.
    prefix = prompt_token_ids(tokenizer, prompt)
    response = tokenizer.encode(target, add_special_tokens=False, truncation=False)
    eos = tokenizer.eos_token_id
    if not prefix or not response or eos is None or eos in response:
        raise ValueError("expected a nonempty prompt and response followed by one EOS token")
    if tokenizer.decode(response, skip_special_tokens=False, clean_up_tokenization_spaces=False) != target:
        raise ValueError("response tokens do not round-trip to the JSON target")
    target_ids = [*response, eos]
    ids = prefix + target_ids
    if len(ids) > max_length:
        raise ValueError(f"sequence has {len(ids)} tokens, exceeding {max_length}; target would be truncated")
    # Qwen's causal-LM loss shifts these labels once. Keep the first answer
    # label at its own token index so the last prompt position predicts it.
    return TrainingExample(ids, [IGNORE_INDEX] * len(prefix) + target_ids, len(prefix))


def collate_examples(examples, pad_token_id):
    if pad_token_id is None:
        raise ValueError("tokenizer requires a pad token ID")
    width = max(len(example.input_ids) for example in examples)
    batch = {"input_ids": [], "labels": [], "attention_mask": []}
    for example in examples:
        size = len(example.input_ids)
        pad = width - size
        batch["input_ids"].append(example.input_ids + [pad_token_id] * pad)
        batch["labels"].append(example.labels + [IGNORE_INDEX] * pad)
        batch["attention_mask"].append([1] * size + [0] * pad)
    return batch


def _greedy_check(model, tokenizer, tasks):
    model.eval()
    results = []
    for case_id, prompt, revision in tasks:
        answer = greedy_answer(model, tokenizer, prompt, "cuda")
        try:
            correct = verify_answer(revision, answer).correct
        except InvalidAnswer:
            status = "malformed"
        else:
            status = "correct" if correct else "incorrect"
        results.append({"case_id": case_id, "answer": answer, "status": status})
    return results


def run_tiny_sft(manifest, max_steps):
    import torch
    from peft import LoraConfig, get_peft_model

    if not 1 <= max_steps <= STEP_CAP:
        raise ValueError(f"max_steps must be between 1 and {STEP_CAP}")
    if not torch.cuda.is_available():
        raise RuntimeError("the tiny-set run requires a CUDA GPU")

    checksum, cases = select_tiny_cases(manifest)
    settings = {
        "manifest_sha256": checksum, "case_ids": [case.case_id for case in cases],
        "model": MODEL_ID, "revision": MODEL_REVISION, "seed": SEED,
        "gpu": torch.cuda.get_device_name(), "torch_version": str(torch.__version__),
        "precision": "FP16 autocast with FP32 LoRA weights and dynamic loss scaling",
        "max_steps": max_steps, "max_new_tokens": MAX_NEW_TOKENS,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS, "learning_rate": LEARNING_RATE,
    }
    print(json.dumps({"event": "tiny_sft_start", **settings}), flush=True)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    base, tokenizer = load_qwen("cuda")
    model = get_peft_model(base, LoraConfig(
        task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear",
    ))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.dtype != torch.float32 for parameter in trainable):
        raise RuntimeError("expected FP32 trainable LoRA parameters for gradient scaling")

    tasks = [(case.case_id, render_prompt(case.task), evaluate_revision(case.task)) for case in cases]
    examples = [
        encode_example(tokenizer, prompt, json.dumps(revision.new, sort_keys=True, separators=(",", ":")))
        for _, prompt, revision in tasks
    ]
    boundaries = [
        {"case_id": case.case_id, "answer_start": example.answer_start,
         "first_answer_token": example.labels[example.answer_start],
         "target_tokens_including_eos": len(example.input_ids) - example.answer_start}
        for case, example in zip(cases, examples)
    ]
    batch = {key: torch.tensor(value, dtype=torch.long, device="cuda")
             for key, value in collate_examples(examples, tokenizer.pad_token_id).items()}
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    losses = []
    skipped_steps = 0
    results = []
    exact = 0
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model(**batch, use_cache=False).loss
        value = loss.item()
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite loss at step {step}: {value}")
        losses.append(value)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped_steps += int(scaler.get_scale() < previous_scale)

        if step % CHECK_INTERVAL == 0 or step == max_steps:
            results = _greedy_check(model, tokenizer, tasks)
            exact = sum(row["status"] == "correct" for row in results)
            print(json.dumps({"event": "tiny_sft_check", "step": step, "loss": value,
                              "exact": exact, "total": len(tasks)}), flush=True)
            if exact == len(tasks):
                break

    torch.cuda.synchronize()
    return {
        **settings, "boundaries": boundaries, "losses": losses, "steps": len(losses),
        "skipped_steps": skipped_steps, "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.perf_counter() - started,
        "greedy_joint_exact": {"numerator": exact, "denominator": len(tasks)},
        "passed": exact == len(tasks), "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=STEP_CAP)
    args = parser.parse_args()
    report = run_tiny_sft(args.manifest, args.max_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"event": "tiny_sft_complete", "output": str(args.output),
                      "passed": report["passed"], "elapsed_seconds": report["elapsed_seconds"],
                      "peak_allocated_gpu_bytes": report["peak_allocated_gpu_bytes"],
                      "greedy_joint_exact": report["greedy_joint_exact"]}), flush=True)
    if not report["passed"]:
        parser.exit(1, "tiny-set check reached the step cap without 4/4 strict exact matches\n")


if __name__ == "__main__":
    main()
