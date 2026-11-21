"""Compare matched full-parameter Qwen SFT on one and two GPUs.

The fixed acceptance bounds are abs(error) <= atol + rtol * abs(reference)
for scalar token-mean loss, and RMS(error) <= atol + rtol * RMS(reference)
for each gradient and parameter-delta tensor. Max elementwise absolute errors
and relative L2 errors are recorded as diagnostics. Initial weights must match
byte for byte. Any skipped FP16 update fails the comparison.
"""

import argparse
import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

from baseline_qwen import MODEL_ID, MODEL_REVISION
from render_prompts import DEFAULT_MANIFEST, render_prompt
from train_sft import (
    FULL_LEARNING_RATE, IGNORE_INDEX, SEED, _make_optimizer, _prepare_model,
    collate_examples, encode_example, select_tiny_cases,
)
from verify import evaluate_revision


PARITY_UPDATES = 2
WARMUP_UPDATES = 2
TIMED_UPDATES = 3
INITIAL_SCALE = 1024
MAX_GRAD_NORM = 1.0
TRACE_PARAMETER = "model.layers.11.self_attn.k_norm.weight"
TOLERANCES = {
    "loss": {"atol": 0.02, "rtol": 0.01},
    "first_gradient": {"atol": 1e-6, "rtol": 0.025},
    "parameter_delta": {"atol": 2e-7, "rtol": 0.08},
}


def global_batches(case_ids, updates):
    """Rotate the same four ordered train IDs within each effective batch."""
    if len(case_ids) != 4 or len(set(case_ids)) != 4:
        raise ValueError("expected four distinct train IDs")
    return [case_ids[step % 4:] + case_ids[:step % 4] for step in range(updates)]


def supervised_count(labels):
    return sum(label != IGNORE_INDEX for label in labels[1:])


def ddp_loss_weight(local_tokens, global_tokens, world_size):
    """Scale a microbatch mean before DDP averages gradients over ranks."""
    if not 0 < local_tokens <= global_tokens or world_size < 1:
        raise ValueError("invalid supervised token counts or world size")
    return world_size * local_tokens / global_tokens


def _prepare_batches(case_ids, examples, world):
    counts = {key: supervised_count(example.labels) for key, example in examples.items()}
    global_count = sum(counts[key] for key in case_ids)
    return {
        "case_ids": case_ids,
        "rank_case_ids": [case_ids[rank::world] for rank in range(world)],
        "rank_supervised_tokens": [sum(counts[key] for key in case_ids[rank::world])
                                   for rank in range(world)],
        "global_supervised_tokens": global_count,
        "global_non_padding_tokens": sum(len(examples[key].input_ids) for key in case_ids),
    }


def _train_update(model, optimizer, trainable, scaler, microbatches, world, gradient_hook=None,
                  *, fp16=True, trace=None):
    import torch
    import torch.distributed as dist

    optimizer.zero_grad(set_to_none=True)
    device = trainable[0].device
    global_count = torch.tensor(sum(count for _, count in microbatches), device=device)
    if world == 2:
        dist.all_reduce(global_count)
    total = global_count.item()
    local_sum = torch.zeros((), dtype=torch.float64, device=device)
    for index, (batch, count) in enumerate(microbatches):
        sync = model.no_sync() if world == 2 and index < len(microbatches) - 1 else nullcontext()
        with sync:
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                loss = model(**batch, use_cache=False).loss
            local_sum += loss.detach().double() * count
            scaler.scale(loss * ddp_loss_weight(count, total, world)).backward()
    scaler.unscale_(optimizer)
    if gradient_hook is not None:
        gradient_hook()
    if trace is not None:
        trace.pre_clip()
    norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
    before = scaler.get_scale()
    if trace is not None:
        trace.post_clip(norm, before)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if world == 2:
        dist.all_reduce(local_sum)
    loss_value = local_sum.item() / total
    after = scaler.get_scale()
    skipped = after < before
    if trace is not None:
        trace.post_step(optimizer, after, skipped)
    return {"token_mean_loss": loss_value if math.isfinite(loss_value) else None,
            "skipped": skipped}


def _cpu_tensors(model, gradients=False):
    return {name: (parameter.grad if gradients else parameter).detach().cpu()
            for name, parameter in model.named_parameters()}


def _initial_hash(tensors):
    digest = hashlib.sha256()
    for name, tensor in tensors.items():
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(memoryview(tensor.numpy()).cast("B"))
    return digest.hexdigest()


def _distances(actual, reference):
    import torch

    difference = actual - reference
    absolute = difference.abs().max().item()
    rms = torch.linalg.vector_norm(difference).item() / math.sqrt(difference.numel())
    scale = torch.linalg.vector_norm(reference).item() / math.sqrt(reference.numel())
    if not all(map(math.isfinite, (absolute, rms, scale))):
        return {"max_abs": None, "rms_abs": None, "reference_rms": None, "relative_l2": None}
    return {"max_abs": absolute, "rms_abs": rms,
            "reference_rms": scale, "relative_l2": rms / scale if scale else (0.0 if not rms else None)}


def _errors(actual, reference, limit):
    distances = _distances(actual, reference)
    scale = distances["reference_rms"]
    bound = limit["atol"] + limit["rtol"] * scale if scale is not None else None
    rms = distances["rms_abs"]
    return {**distances, "bound_rms": bound, "tolerance_ratio": rms / bound if bound is not None else None,
            "passed": bound is not None and rms <= bound}


def _parameter_deltas(current, previous, initial):
    return {"update_delta": current - previous, "cumulative_delta": current - initial}


class ParameterTrace:
    """Capture one small base tensor on rank zero without changing the optimizer."""

    def __init__(self, parameter, initial):
        self.parameter = parameter
        self.initial = initial
        self.previous = initial
        self.steps = []

    def pre_clip(self):
        self.steps.append({"gradient_pre_clip": self.parameter.grad.detach().to("cpu", copy=True)})

    def post_clip(self, norm, scale):
        step = self.steps[-1]
        step.update(
            gradient_post_clip=self.parameter.grad.detach().to("cpu", copy=True),
            global_pre_clip_norm=norm.item(),
            clip_factor=(MAX_GRAD_NORM / (norm + 1e-6)).clamp(max=1.0).item(),
            scaler_before=scale,
        )

    def post_step(self, optimizer, scale, skipped):
        step = self.steps[-1]
        state = optimizer.state.get(self.parameter, {})
        current = self.parameter.detach().to("cpu", copy=True)
        step.update(_parameter_deltas(current, self.previous, self.initial))
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            step[key] = value.detach().to("cpu", copy=True) if value is not None else None
        step.update(scaler_after=scale, skipped=skipped)
        self.previous = current


TRACE_TENSORS = ("gradient_pre_clip", "gradient_post_clip", "exp_avg", "exp_avg_sq",
                 "update_delta", "cumulative_delta")
TRACE_SCALARS = ("global_pre_clip_norm", "clip_factor", "scaler_before", "scaler_after")


def _compare_trace(actual, reference):
    rows = []
    for step, (a, b) in enumerate(zip(actual, reference, strict=True), 1):
        tensors = {key: _distances(a[key], b[key]) if a[key] is not None and b[key] is not None else None
                   for key in TRACE_TENSORS}
        cumulative = _errors(a["cumulative_delta"], b["cumulative_delta"], TOLERANCES["parameter_delta"])
        scalars = {key: {"one_gpu": b[key], "two_gpu": a[key], "absolute_error": abs(a[key] - b[key])}
                   if math.isfinite(a[key]) and math.isfinite(b[key]) else None for key in TRACE_SCALARS}
        rows.append({"step": step, "tensors": tensors, "scalars": scalars,
                     "cumulative_delta_vs_original_bound": cumulative,
                     "skipped": {"one_gpu": b["skipped"], "two_gpu": a["skipped"]}})

    first = next((row["step"] for row in rows
                  if row["cumulative_delta_vs_original_bound"]["tolerance_ratio"] is not None
                  and row["cumulative_delta_vs_original_bound"]["tolerance_ratio"] > 1), None)
    final_difference = actual[-1]["cumulative_delta"] - reference[-1]["cumulative_delta"]
    index = int(final_difference.abs().argmax())

    def coordinate(step, key):
        value = step[key]
        if value is not None:
            scalar = value.flatten()[index].item()
            return scalar if math.isfinite(scalar) else None
        return None

    return {"parameter": TRACE_PARAMETER, "steps": rows, "first_step_over_bound": first,
            "worst_final_coordinate": {
                "flat_index": index,
                "steps": [{key: {"one_gpu": coordinate(b, key), "two_gpu": coordinate(a, key)}
                           for key in TRACE_TENSORS} for a, b in zip(actual, reference, strict=True)],
            }}


def _tensor_report(actual, reference, limit):
    if actual.keys() != reference.keys():
        raise ValueError("reference and comparison parameter names differ")
    rows = {name: _errors(value, reference[name], limit) for name, value in actual.items()}
    worst = max(rows, key=lambda name: math.inf if rows[name]["tolerance_ratio"] is None
                else rows[name]["tolerance_ratio"])
    return {"passed": all(row["passed"] for row in rows.values()),
            "worst_tensor": {"name": worst, **rows[worst]}, "tensors": rows}


def _loss_report(actual, reference):
    limit = TOLERANCES["loss"]
    rows = []
    for a, b in zip(actual, reference, strict=True):
        error = abs(a - b) if a is not None and b is not None else None
        rows.append({"one_gpu": b, "two_gpu": a, "absolute_error": error,
                     "relative_error": error / abs(b) if error is not None and b else None,
                     "passed": error is not None and error <= limit["atol"] + limit["rtol"] * abs(b)})
    return {"passed": all(row["passed"] for row in rows), "updates": rows}


def _write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _agree(update, world, device):
    """Report every rank's skip flag even if a runtime bug makes them differ."""
    import torch
    import torch.distributed as dist

    local = torch.tensor(int(update["skipped"]), device=device)
    if world == 1:
        return [bool(local.item())]
    gathered = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(gathered, local)
    return [bool(value.item()) for value in gathered]


def _parity(args, model, update, metadata, rank, world):
    import torch

    reference_dir = args.reference_dir
    initial = _cpu_tensors(model) if rank == 0 else None
    before = _initial_hash(initial) if rank == 0 else None
    reference = _read(reference_dir / "reference.json") if rank == 0 and world == 2 else None
    if reference and reference["settings"] != metadata["settings"]:
        raise ValueError("one- and two-GPU parity settings differ")
    comparisons = {}
    trace = None
    if rank == 0 and args.trace_dir is not None:
        parameter = model.get_parameter(TRACE_PARAMETER)
        trace = ParameterTrace(parameter, initial[TRACE_PARAMETER])

    def capture(kind, values):
        path = reference_dir / f"{kind}.pt"
        if world == 1:
            reference_dir.mkdir(parents=True, exist_ok=True)
            torch.save(values, path)
        else:
            expected = torch.load(path, mmap=True, weights_only=True)
            comparisons[kind] = _tensor_report(values, expected, TOLERANCES[kind])

    losses, skipped = [], []
    for step in range(PARITY_UPDATES):
        hook = None
        if step == 0 and rank == 0:
            hook = lambda: capture("first_gradient", _cpu_tensors(model, gradients=True))
        result = update(step, hook, trace)
        losses.append(result["token_mean_loss"])
        skipped.append(result["skipped_by_rank"])
    if rank != 0:
        return True

    deltas = {name: parameter.detach().cpu() - initial[name]
              for name, parameter in model.named_parameters()}
    capture("parameter_delta", deltas)
    if reference:
        comparisons["loss"] = _loss_report(losses, reference["token_mean_losses"])
        comparisons["initial_state"] = {"passed": before == reference["initial_state_sha256"],
                                        "one_gpu_sha256": reference["initial_state_sha256"],
                                        "two_gpu_sha256": before}
    passed = not any(map(any, skipped)) and all(loss is not None for loss in losses)
    if reference:
        passed = passed and reference["passed"] and all(row["passed"] for row in comparisons.values())
    report = {**metadata, "passed": passed, "initial_state_sha256": before,
              "token_mean_losses": losses, "skipped_by_update_and_rank": skipped,
              "skipped_updates_per_rank": [sum(row[index] for row in skipped) for index in range(world)],
              "comparisons": comparisons}
    if trace is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        trace_name = f"{args.precision}_trace"
        torch.save({"settings": metadata["settings"], "initial_state_sha256": before,
                    "steps": trace.steps}, args.trace_dir / f"{trace_name}_{world}gpu.pt")
        if world == 2:
            expected_trace = torch.load(args.trace_dir / f"{trace_name}_1gpu.pt", weights_only=True)
            if (expected_trace["settings"] != reference["settings"]
                    or expected_trace["initial_state_sha256"] != reference["initial_state_sha256"]):
                raise ValueError("diagnostic trace does not match the run's one-GPU reference")
            diagnostic = _compare_trace(trace.steps, expected_trace["steps"])
            _write(args.trace_dir / f"{trace_name}_comparison.json", {
                "settings": metadata["settings"], "parity_passed": passed, **diagnostic,
            })
    if world == 1:
        _write(reference_dir / "reference.json", report)
    _write(args.output, report)
    print(json.dumps({"event": "parity_complete", "world_size": world, "passed": passed,
                      "losses": losses, "skipped_by_update_and_rank": skipped,
                      "comparison_passed": {name: row["passed"] for name, row in comparisons.items()}}), flush=True)
    return passed


def _profile(args, update, metadata, rank, world, device):
    import torch
    import torch.distributed as dist

    for step in range(WARMUP_UPDATES):
        warmup = update(step)
        if any(warmup["skipped_by_rank"]) or warmup["token_mean_loss"] is None:
            raise RuntimeError("profile warm-up update failed")
    torch.cuda.synchronize(device)
    if world == 2:
        dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    results = [update(step) for step in range(WARMUP_UPDATES, WARMUP_UPDATES + TIMED_UPDATES)]
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    local = {"rank": rank, "elapsed_seconds": seconds,
             "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
             "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    ranks = [local] if world == 1 else [None] * world
    if world == 2:
        dist.all_gather_object(ranks, local)
    passed = all(not any(row["skipped_by_rank"]) and row["token_mean_loss"] is not None for row in results)
    if rank == 0:
        seconds = max(row["elapsed_seconds"] for row in ranks)
        timed = metadata["batches"][WARMUP_UPDATES:]
        nonpadding = sum(row["global_non_padding_tokens"] for row in timed)
        supervised = sum(row["global_supervised_tokens"] for row in timed)
        _write(args.output, {**metadata, "passed": passed, "warmup_updates": WARMUP_UPDATES,
                             "timed_updates": TIMED_UPDATES, "elapsed_seconds": seconds,
                             "global_non_padding_tokens": nonpadding, "global_supervised_tokens": supervised,
                             "global_non_padding_tokens_per_second": nonpadding / seconds,
                             "global_supervised_tokens_per_second": supervised / seconds,
                             "per_rank": ranks, "updates": results})
        print(json.dumps({"event": "profile_complete", "world_size": world, "passed": passed,
                          "elapsed_seconds": seconds, "non_padding_tokens_per_second": nonpadding / seconds,
                          "supervised_tokens_per_second": supervised / seconds, "per_rank": ranks}), flush=True)
    return passed


def _require_profile_gate(gate, settings):
    if gate["world_size"] != 2 or not gate["passed"] or gate["settings"] != settings:
        raise ValueError("profile requires a passing two-GPU parity report for the same settings")


def run(args):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world not in (1, 2) or torch.cuda.device_count() != world:
        raise ValueError("expose exactly one CUDA device per process and launch one or two local ranks")
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    if world == 2:
        dist.init_process_group("nccl", timeout=timedelta(minutes=4))
    try:
        torch.manual_seed(SEED)
        base, tokenizer = _prepare_model("full", device)
        for module in base.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
            if hasattr(module, "attention_dropout"):
                module.attention_dropout = 0.0
        base.train()
        optimizer, trainable = _make_optimizer(base, FULL_LEARNING_RATE)
        model = DistributedDataParallel(base, device_ids=[rank], broadcast_buffers=False,
                                        gradient_as_bucket_view=True) if world == 2 else base
        fp16 = args.precision == "fp16"
        scaler = torch.amp.GradScaler("cuda", init_scale=INITIAL_SCALE, enabled=fp16)
        checksum, cases = select_tiny_cases(args.manifest)
        examples = {case.case_id: encode_example(
            tokenizer, render_prompt(case.task),
            json.dumps(evaluate_revision(case.task).new, sort_keys=True, separators=(",", ":")),
        ) for case in cases}
        ids = list(examples)
        tokens = {key: [example.input_ids, example.labels] for key, example in examples.items()}
        encoded_checksum = hashlib.sha256(json.dumps(tokens, sort_keys=True).encode("utf-8")).hexdigest()
        settings = {"model": MODEL_ID, "revision": MODEL_REVISION, "manifest_sha256": checksum,
                    "encoded_examples_sha256": encoded_checksum,
                    "seed": SEED, "precision": "FP32 weights, FP16 autocast and dynamic loss scaling",
                    "parity_updates": PARITY_UPDATES, "optimizer": "AdamW", "learning_rate": FULL_LEARNING_RATE,
                    "adamw_betas": list(optimizer.defaults["betas"]), "adamw_eps": optimizer.defaults["eps"],
                    "adamw_weight_decay": optimizer.defaults["weight_decay"], "max_grad_norm": MAX_GRAD_NORM,
                    "initial_loss_scale": INITIAL_SCALE, "dropout": 0.0, "microbatch_size": 1,
                    "parity_global_batches": global_batches(ids, PARITY_UPDATES), "tolerances": TOLERANCES}
        if not fp16:
            settings.update(precision="FP32 weights and compute, no loss scaling", initial_loss_scale=1)
        if args.phase == "profile":
            _require_profile_gate(_read(args.parity_report), settings)
        count = PARITY_UPDATES if args.phase == "parity" else WARMUP_UPDATES + TIMED_UPDATES
        batches = [_prepare_batches(order, examples, world)
                   for order in global_batches(ids, count)]
        if world == 2 and not any(len(set(batch["rank_supervised_tokens"])) > 1 for batch in batches):
            raise ValueError("the workload must test unequal rank-local supervised token counts")
        encoded = {key: ({field: torch.tensor(value, dtype=torch.long, device=device)
                          for field, value in collate_examples([example], tokenizer.pad_token_id).items()},
                         supervised_count(example.labels)) for key, example in examples.items()}
        metadata = {"phase": args.phase, "world_size": world, "gpu": torch.cuda.get_device_name(device),
                    "torch_version": str(torch.__version__), "settings": settings, "batches": batches}
        if rank == 0:
            print(json.dumps({"event": "start", **metadata}), flush=True)

        def update(step, hook=None, trace=None):
            local = [encoded[key] for key in batches[step]["rank_case_ids"][rank]]
            result = _train_update(model, optimizer, trainable, scaler, local, world, hook,
                                   fp16=fp16, trace=trace)
            result["skipped_by_rank"] = _agree(result, world, device)
            return result

        if args.phase == "parity":
            return _parity(args, base, update, metadata, rank, world)
        return _profile(args, update, metadata, rank, world, device)
    finally:
        if world == 2:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("parity", "profile"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--parity-report", type=Path)
    parser.add_argument("--trace-dir", type=Path, help=f"capture both updates for {TRACE_PARAMETER}")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()
    if args.phase == "parity" and args.reference_dir is None:
        parser.error("parity requires --reference-dir")
    if args.phase == "profile" and args.parity_report is None:
        parser.error("profile requires --parity-report")
    if args.phase == "profile" and args.trace_dir is not None:
        parser.error("the parameter trace is only available in the parity phase")
    if args.precision == "fp32" and (args.phase != "parity" or args.trace_dir is None):
        parser.error("FP32 is only available as an opt-in parity diagnostic with --trace-dir")
    if not run(args):
        parser.exit(1, "Day 6 check failed; see the JSON report\n")


if __name__ == "__main__":
    main()
