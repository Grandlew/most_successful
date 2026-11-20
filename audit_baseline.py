"""Audit frozen development predictions without changing the strict baseline score."""

import argparse
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from baseline_qwen import GENERATION_PROTOCOL, MAX_NEW_TOKENS, MODEL_ID, MODEL_REVISION
from evaluate import evaluate, read_predictions
from render_prompts import DEFAULT_MANIFEST, load_development, unique_fields
from verify import InvalidAnswer, evaluate_revision, verify_answer


def _container_end(text, start):
    expected = []
    quoted = escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "{[":
            expected.append("}" if char == "{" else "]")
        elif char in "}]":
            if char != expected.pop():
                return index + 1, "invalid_json"
            if not expected:
                return index + 1, None
    return len(text), "incomplete_json"


def _reject_constant(token):
    raise ValueError(f"not a JSON constant: {token}")


def _locate_object(answer):
    """Return one unchanged top-level object and an extraction diagnostic."""
    if not isinstance(answer, str) or not answer.strip():
        return None, "no_usable_answer"
    if answer.lstrip().startswith('"'):
        try:
            if isinstance(json.loads(answer), str):
                return None, "non_object_json"
        except (ValueError, RecursionError):
            pass
    objects = []
    problems = []
    index = 0
    while index < len(answer):
        if answer[index] not in "{[":
            index += 1
            continue
        end, problem = _container_end(answer, index)
        candidate = answer[index:end]
        index = end
        if problem:
            problems.append(problem)
            continue
        try:
            value = json.loads(candidate, parse_int=str, parse_constant=_reject_constant)
        except (ValueError, RecursionError):
            problems.append("invalid_json")
        else:
            if isinstance(value, dict):
                objects.append(candidate)
            else:
                problems.append("non_object_json")
    if len(objects) > 1 or (objects and problems):
        return None, "ambiguous_json"
    if not objects:
        if "incomplete_json" in problems:
            return None, "incomplete_json"
        return None, problems[0] if problems else "no_usable_answer"
    return objects[0], "plain" if objects[0] == answer.strip() else "extra_text_or_fence"


def extract_json_object(answer):
    """Extract a single unambiguous object without rewriting its JSON text."""
    return _locate_object(answer)[0]


def _group_values(pairs):
    values = {}
    for key, value in pairs:
        values.setdefault(key, []).append(value)
    return values


def _classify(case, answer, strict, metadata):
    candidate, extraction = _locate_object(answer)
    exact = False
    wrong_values = {}
    identifiable_keys = []
    verification_error = None
    category = extraction
    if candidate is not None:
        revision = evaluate_revision(case.task)
        # Reuse the oracle verifier for scoring. Group occurrences only for
        # diagnosis so repeated keys cannot overwrite an earlier occurrence.
        try:
            check = verify_answer(revision, candidate)
        except InvalidAnswer as error:
            verification_error = str(error)
            if verification_error.startswith("duplicate JSON key"):
                category = "duplicate_keys"
            elif verification_error.startswith("wrong output keys"):
                category = "wrong_keys"
            elif verification_error.endswith("must be a JSON integer"):
                category = "wrong_types"
            else:
                category = "invalid_json"
        else:
            exact = check.correct
            category = extraction if extraction != "plain" else ("correct" if exact else "wrong_values")

        values = json.loads(candidate, object_pairs_hook=_group_values,
                            parse_int=lambda token: int(Decimal(token)))
        for key, expected in revision.new.items():
            found = values.get(key, [])
            if len(found) == 1 and type(found[0]) is int:
                identifiable_keys.append(key)
                if found[0] != expected:
                    wrong_values[key] = {"observed": found[0], "expected": expected}

    reason = metadata["stop_reason"] if metadata else None
    if metadata and metadata["new_token_ids"] == [metadata["eos_token_id"]] and answer == "":
        disposition = "Matching replay generated EOS as the first token. Retain the empty reply as a model failure."
    elif category == "no_usable_answer":
        disposition = "No usable object was emitted. Retain the strict result. The stopping cause is uncertain."
    elif category == "ambiguous_json":
        disposition = "Competing JSON spans make the reply ambiguous. Retain the strict result."
    elif extraction == "extra_text_or_fence":
        disposition = "Retain the strict result and score the unchanged embedded object for diagnostics only."
    else:
        disposition = "Retain the original reply and strict result."
    return {
        "case_id": case.case_id, "answer": answer, "category": category,
        "complete_answer_object": candidate is not None, "extraction": extraction,
        "extracted_object": candidate, "strict_status": strict["status"],
        "strict_exact": strict["joint_exact"], "strict_error": strict["error"],
        "extraction_exact": exact, "extraction_verifier_error": verification_error,
        "identifiable_integer_keys": sorted(identifiable_keys), "wrong_values": wrong_values,
        "original_generation_metadata_available": False, "matching_replay_generation": metadata,
        "truncation_confirmed": None if reason in (None, "unknown") else reason == "token_cap" and extraction == "incomplete_json",
        "uncertain": category in {"ambiguous_json", "invalid_json"} or (
            category in {"no_usable_answer", "incomplete_json"} and reason in (None, "unknown")
        ),
        "disposition": disposition,
    }


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_fields)


def _replay_metadata(path, checksum, replay, original):
    if dict(replay) != dict(original) or len(replay) != len(original):
        raise ValueError("replay predictions do not match the original replies by case ID")
    evidence = _read_json(path)
    settings = {"manifest_sha256": checksum, "model": MODEL_ID, "revision": MODEL_REVISION,
                "protocol": GENERATION_PROTOCOL, "max_new_tokens": MAX_NEW_TOKENS, "device": "cpu"}
    if any(evidence.get(key) != value for key, value in settings.items()):
        raise ValueError("generation metadata does not match the frozen CPU protocol")
    rows = {row["case_id"]: row for row in evidence["cases"]}
    if rows.keys() != dict(original).keys() or len(rows) != len(evidence["cases"]):
        raise ValueError("generation metadata must cover every original case exactly once")
    for row in rows.values():
        ids = row["new_token_ids"]
        eos = row["eos_token_id"]
        if type(eos) is not int or len(ids) > MAX_NEW_TOKENS or any(type(token) is not int for token in ids):
            raise ValueError("invalid generation metadata token IDs")
        stop_reason = "eos" if ids and ids[-1] == eos else "token_cap" if len(ids) == MAX_NEW_TOKENS else "unknown"
        if row["stop_reason"] != stop_reason:
            raise ValueError("generation metadata stop reason disagrees with token IDs")
    return rows


def _count(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator}


def audit(manifest, predictions, saved_report, replay_predictions=None, generation_metadata=None):
    checksum, cases = load_development(manifest)
    original = list(read_predictions(predictions))
    strict = evaluate(cases, original)
    if len(cases) != 20 or len(original) != 20:
        raise ValueError("audit requires exactly the 20 frozen development IDs")
    reproduced = {"manifest_sha256": checksum, "prediction_source": "predictions", **strict}
    if _read_json(saved_report) != reproduced:
        raise ValueError("saved evaluation report disagrees with the reproduced strict score")

    evidence = {}
    if (replay_predictions is None) != (generation_metadata is None):
        raise ValueError("provide both replay predictions and generation metadata, or neither")
    if replay_predictions is not None:
        replay = list(read_predictions(replay_predictions))
        evaluate(cases, replay)
        evidence = _replay_metadata(generation_metadata, checksum, replay, original)
    by_id = dict(original)
    classified = [
        _classify(case, by_id[case.case_id], result, evidence.get(case.case_id))
        for case, result in zip(cases, strict["cases"])
    ]
    inputs = {"predictions": predictions, "saved_report": saved_report}
    if evidence:
        inputs.update(replay_predictions=replay_predictions, generation_metadata=generation_metadata)
    return {
        "audit_version": 1, "generation_protocol": GENERATION_PROTOCOL,
        "manifest_sha256": checksum,
        "artifact_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()},
        "strict_metrics": strict["metrics"],
        "extraction_joint_exact": _count(sum(row["extraction_exact"] for row in classified), len(cases)),
        "extraction_coverage": _count(sum(row["complete_answer_object"] for row in classified), len(cases)),
        "reply_counts": dict(Counter(row["category"] for row in classified)),
        "wrong_value_count": sum(len(row["wrong_values"]) for row in classified),
        "confirmed_truncation_count": sum(row["truncation_confirmed"] is True for row in classified),
        "uncertain_reply_count": sum(row["uncertain"] for row in classified),
        "matching_replay_first_token_eos_count": sum(
            row["new_token_ids"] == [row["eos_token_id"]] for row in evidence.values()
        ),
        "cases": classified,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--saved-report", type=Path, required=True)
    parser.add_argument("--replay-predictions", type=Path)
    parser.add_argument("--generation-metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = (args.manifest, args.predictions, args.saved_report, args.replay_predictions, args.generation_metadata)
    if any(path is not None and path.resolve() == args.output.resolve() for path in sources):
        parser.error("audit output must not overwrite a source artifact")
    try:
        report = audit(*sources)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"baseline audit failed: {error}\n")


if __name__ == "__main__":
    main()
