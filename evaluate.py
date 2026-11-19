"""Score the frozen development split from JSONL predictions or a CPU baseline.

Each prediction line has a case_id and an answer containing the raw model
response as a JSON string. Raw text lets the Day 1 verifier detect duplicate
answer keys and other malformed JSON. Counts include every selected case.
"""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from render_prompts import DEFAULT_MANIFEST, DevelopmentCase, load_development, unique_fields
from verify import AnswerCheck, InvalidAnswer, evaluate_revision, verify_answer


BASELINES = ("copy-original", "oracle-revised")


def read_predictions(path: Path) -> Iterable[tuple[str, object]]:
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line, object_pairs_hook=unique_fields)
            except ValueError as error:
                raise ValueError(f"invalid prediction record on line {number}: {error}") from error
            if (not isinstance(record, dict) or record.keys() != {"case_id", "answer"}
                    or not isinstance(record["case_id"], str)):
                raise ValueError(f"prediction line {number} requires a string case_id and an answer")
            yield record["case_id"], record["answer"]


def baseline_predictions(cases: list[DevelopmentCase], baseline: str) -> Iterable[tuple[str, str]]:
    if baseline not in BASELINES:
        raise ValueError(f"unknown baseline: {baseline}")
    for case in cases:
        revision = evaluate_revision(case.task)
        values = revision.old if baseline == "copy-original" else revision.new
        yield case.case_id, json.dumps(values, sort_keys=True)


def _metric(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _score_case(case: DevelopmentCase, predictions: dict[str, object]) -> dict[str, object]:
    revision = evaluate_revision(case.task)
    changed_total = len(revision.changed)
    stable_total = len(revision.new) - changed_total
    check = AnswerCheck(frozenset(), frozenset(), frozenset())
    error_message = None
    if case.case_id not in predictions:
        status = "missing"
    else:
        try:
            answer = predictions[case.case_id]
            if not isinstance(answer, str):
                raise InvalidAnswer("answer must be JSON text")
            check = verify_answer(revision, answer)
        except InvalidAnswer as error:
            status = "malformed"
            error_message = str(error)
        else:
            status = "correct" if check.correct else "incorrect"

    valid = status in ("correct", "incorrect")
    return {
        "case_id": case.case_id,
        "shape_group": case.shape_group,
        "depth": case.depth,
        "status": status,
        "joint_exact": status == "correct",
        "changed_correct": changed_total - len(check.stale) - len(check.incorrect) if valid else 0,
        "changed_total": changed_total,
        "stable_preserved": stable_total - len(check.unnecessary) if valid else 0,
        "stable_total": stable_total,
        "stale_outputs": sorted(check.stale),
        "unnecessary_changes": sorted(check.unnecessary),
        "other_incorrect_outputs": sorted(check.incorrect),
        "error": error_message,
    }


def evaluate(cases: list[DevelopmentCase], predictions: Iterable[tuple[str, object]]) -> dict:
    known = {case.case_id for case in cases}
    if len(known) != len(cases):
        raise ValueError("duplicate evaluation case IDs")
    by_id = {}
    for case_id, answer in predictions:
        if case_id not in known:
            raise ValueError(f"unknown prediction case ID: {case_id}")
        if case_id in by_id:
            raise ValueError(f"duplicate prediction case ID: {case_id}")
        by_id[case_id] = answer

    results = [_score_case(case, by_id) for case in cases]
    changed = sum(row["changed_total"] for row in results)
    stable = sum(row["stable_total"] for row in results)
    metrics = {
        "joint_exact": _metric(sum(row["joint_exact"] for row in results), len(results)),
        "changed_accuracy": _metric(sum(row["changed_correct"] for row in results), changed),
        "stable_preservation": _metric(sum(row["stable_preserved"] for row in results), stable),
        "stale_values": _metric(sum(len(row["stale_outputs"]) for row in results), changed),
        "unnecessary_changes": _metric(sum(len(row["unnecessary_changes"]) for row in results), stable),
        "malformed_answers": _metric(sum(row["status"] == "malformed" for row in results), len(results)),
        "missing_predictions": _metric(sum(row["status"] == "missing" for row in results), len(results)),
    }
    return {"metrics": metrics, "cases": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the frozen development cases.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path, help="JSONL with case_id and raw answer fields")
    source.add_argument("--baseline", choices=BASELINES)
    args = parser.parse_args()
    try:
        checksum, cases = load_development(args.manifest)
        if args.baseline:
            predictions = baseline_predictions(cases, args.baseline)
        else:
            predictions = read_predictions(args.predictions)
        report = {
            "manifest_sha256": checksum,
            "prediction_source": args.baseline or "predictions",
            **evaluate(cases, predictions),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        parser.exit(1, f"evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
