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
FULL_LEARNING_RATE = 2e-5
MODES = ("lora", "full")


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


def _prepare_model(mode, device):
    if mode == "full":
        model, tokenizer = load_qwen(device, fp32=True)
        if getattr(model, "peft_config", None) is not None:
            raise RuntimeError("full mode must use an unwrapped model without adapters")
        model.requires_grad_(True)
        return model, tokenizer
    if mode != "lora":
        raise ValueError(f"unknown training mode: {mode}")

    from peft import LoraConfig, get_peft_model

    base, tokenizer = load_qwen(device)
    return get_peft_model(base, LoraConfig(
        task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear",
    )), tokenizer


def _make_optimizer(model, learning_rate):
    import torch

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.dtype != torch.float32 for parameter in trainable):
        raise RuntimeError("expected FP32 trainable parameters for gradient scaling")
    return torch.optim.AdamW(trainable, lr=learning_rate, foreach=False), trainable


def _weighted_microbatches(examples, microbatch_size):
    chunks = [examples[start:start + microbatch_size]
              for start in range(0, len(examples), microbatch_size)]
    # The model returns mean loss over the non-masked labels after its shift.
    counts = [sum(label != IGNORE_INDEX for example in chunk for label in example.labels[1:])
              for chunk in chunks]
    total = sum(counts)
    return [(chunk, count / total) for chunk, count in zip(chunks, counts)]


def run_tiny_sft(manifest, max_steps, mode="lora"):
    import torch

    if not 1 <= max_steps <= STEP_CAP:
        raise ValueError(f"max_steps must be between 1 and {STEP_CAP}")
    if mode not in MODES:
        raise ValueError(f"unknown training mode: {mode}")
    if not torch.cuda.is_available():
        raise RuntimeError("the tiny-set run requires a CUDA GPU")

    checksum, cases = select_tiny_cases(manifest)
    full = mode == "full"
    microbatch_size = 1 if full else TINY_SET_SIZE
    learning_rate = FULL_LEARNING_RATE if full else LEARNING_RATE
    max_attempts = 2 * max_steps
    settings = {
        "manifest_sha256": checksum, "case_ids": [case.case_id for case in cases],
        "model": MODEL_ID, "revision": MODEL_REVISION, "seed": SEED,
        "gpu": torch.cuda.get_device_name(), "torch_version": str(torch.__version__),
        "mode": mode, "precision": "FP16 autocast with FP32 trainable weights and dynamic loss scaling",
        "max_steps": max_steps, "max_attempts": max_attempts, "max_new_tokens": MAX_NEW_TOKENS,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS, "learning_rate": learning_rate,
        "microbatch_size": microbatch_size, "effective_batch_size": len(cases),
        "check_interval": CHECK_INTERVAL, "optimizer": "AdamW",
    }
    print(json.dumps({"event": "tiny_sft_start", **settings}), flush=True)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    boundaries = []
    parameter_counts = None
    weight_check = None
    losses = []
    skipped_steps = 0
    updates = attempts = 0
    history = []
    outcome = "attempt_cap"
    error = None

    try:
        model, tokenizer = _prepare_model(mode, "cuda")
        optimizer, trainable = _make_optimizer(model, learning_rate)
        parameter_counts = {"total": sum(parameter.numel() for parameter in model.parameters()),
                            "trainable": sum(parameter.numel() for parameter in trainable)}
        if full and parameter_counts["total"] != parameter_counts["trainable"]:
            raise RuntimeError("full mode requires every parameter to be trainable")
        print(json.dumps({"event": "tiny_sft_parameters", **parameter_counts,
                          "adapters_attached": not full}), flush=True)
        if full:
            name, weight = min(
                ((name, parameter) for name, parameter in model.named_parameters() if parameter.ndim >= 2),
                key=lambda pair: pair[1].numel(),
            )
            original_weight = weight.detach().clone()
            weight_check = {"parameter": name, "changed_after_first_update": None}

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
        batches = [
            ({key: torch.tensor(value, dtype=torch.long, device="cuda")
              for key, value in collate_examples(chunk, tokenizer.pad_token_id).items()}, fraction)
            for chunk, fraction in _weighted_microbatches(examples, microbatch_size)
        ]
        scaler = torch.amp.GradScaler("cuda", init_scale=1024)

        def check_score():
            results = _greedy_check(model, tokenizer, tasks)
            exact = sum(row["status"] == "correct" for row in results)
            history.append({"optimizer_updates": updates, "attempts": attempts,
                            "joint_exact": _exact_count(exact), "cases": results})
            print(json.dumps({"event": "tiny_sft_check", "optimizer_updates": updates,
                              "attempts": attempts, "loss": losses[-1] if losses else None,
                              "exact": exact, "total": len(tasks)}), flush=True)
            return exact == len(tasks)

        initial_pass = check_score()
        if initial_pass and not full:
            outcome = "passed"
        while outcome != "passed" and updates < max_steps and attempts < max_attempts:
            attempts += 1
            model.train()
            value = 0.0
            for batch, fraction in batches:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(**batch, use_cache=False).loss * fraction
                contribution = loss.item()
                if not math.isfinite(contribution):
                    raise RuntimeError(f"non-finite loss at attempt {attempts}: {contribution}")
                value += contribution
                scaler.scale(loss).backward()
            losses.append(value)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() < previous_scale:
                skipped_steps += 1
                continue

            updates += 1
            if full and updates == 1:
                weight_check["changed_after_first_update"] = not torch.equal(original_weight, weight.detach())
                del original_weight
                print(json.dumps({"event": "tiny_sft_weight_check", **weight_check}), flush=True)
                if not weight_check["changed_after_first_update"]:
                    raise RuntimeError("base-model weight did not change after the first optimizer update")

            if updates % CHECK_INTERVAL == 0 or updates == max_steps or (initial_pass and updates == 1):
                if check_score():
                    outcome = "passed"

        if history[-1]["optimizer_updates"] != updates and check_score():
            outcome = "passed"
        if outcome != "passed" and updates == max_steps:
            outcome = "update_cap"
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as failure:
        outcome, error = "out_of_memory", str(failure)

    best = max((row["joint_exact"]["numerator"] for row in history), default=None)
    latest = history[-1] if history else None

    return {
        **settings, "boundaries": boundaries, "losses": losses, "steps": updates,
        "optimizer_updates": updates, "attempts": attempts, "parameter_counts": parameter_counts,
        "base_weight_check": weight_check, "score_history": history,
        "skipped_steps": skipped_steps, "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.perf_counter() - started,
        "greedy_joint_exact": latest["joint_exact"] if latest else None,
        "best_greedy_joint_exact": _exact_count(best) if best is not None else None,
        "passed": outcome == "passed", "outcome": outcome, "error": error,
        "cases": latest["cases"] if latest else [],
    }


def _exact_count(exact):
    return {"numerator": exact, "denominator": TINY_SET_SIZE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=STEP_CAP)
    parser.add_argument("--mode", choices=MODES, default="lora")
    args = parser.parse_args()
    report = run_tiny_sft(args.manifest, args.max_steps, mode=args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"event": "tiny_sft_complete", "output": str(args.output),
                      "passed": report["passed"], "elapsed_seconds": report["elapsed_seconds"],
                      "peak_allocated_gpu_bytes": report["peak_allocated_gpu_bytes"],
                      "outcome": report["outcome"], "error": report["error"],
                      "best_greedy_joint_exact": report["best_greedy_joint_exact"]}), flush=True)
    if not report["passed"]:
        parser.exit(1, f"tiny-set check did not pass ({report['outcome']})\n")


if __name__ == "__main__":
    main()
