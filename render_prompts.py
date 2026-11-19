"""Render the common revision prompt for the frozen development split as JSONL."""

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from generate import generate_experiment_cases
from schema import RevisionTask
from split_dataset import MANIFEST_VERSION, manifest_checksum, output_depth, shape_group
from verify import evaluate_revision


DEFAULT_MANIFEST = Path(__file__).resolve().parent / "data" / "day03_manifest.json"


@dataclass(frozen=True)
class DevelopmentCase:
    case_id: str
    shape_group: str
    depth: int
    task: RevisionTask


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


def load_development(path: Path) -> tuple[str, list[DevelopmentCase]]:
    """Verify the manifest and join only its development rows to regenerated tasks."""
    manifest = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_fields)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    checksum = manifest.pop("sha256", None)
    if checksum != manifest_checksum(manifest):
        raise ValueError("manifest checksum mismatch")
    if manifest.get("version") != MANIFEST_VERSION or type(manifest.get("seed")) is not int:
        raise ValueError("unsupported manifest version or invalid seed")

    rows = manifest["cases"]
    counts = Counter(row["case_id"] for row in rows)
    selected = {row["case_id"]: row for row in rows if row["split"] == "development"}
    if not selected:
        raise ValueError("manifest has no development cases")
    for case_id in selected:
        if counts[case_id] != 1:
            raise ValueError(f"duplicate development case ID in manifest: {case_id}")

    joined = {}
    for task, record in generate_experiment_cases(manifest["seed"]):
        case_id = record["case_id"]
        if case_id not in selected:
            continue
        if case_id in joined:
            raise ValueError(f"duplicate regenerated development case ID: {case_id}")
        row = selected[case_id]
        if row["shape_group"] != shape_group(task) or row["depth"] != output_depth(task):
            raise ValueError(f"regenerated task disagrees with manifest metadata: {case_id}")
        joined[case_id] = DevelopmentCase(case_id, row["shape_group"], row["depth"], task)
    missing = selected.keys() - joined.keys()
    if missing:
        raise ValueError(f"missing regenerated development case IDs: {', '.join(sorted(missing))}")
    return checksum, [joined[case_id] for case_id in selected]


def render_prompt(task: RevisionTask) -> str:
    old = evaluate_revision(task).old
    graph = {
        "inputs": {node.name: node.value for node in task.inputs},
        "derived": [asdict(node) for node in task.evaluation_order],
    }
    return "\n".join((
        "Recompute the requested outputs after applying the single premise edit.",
        "Original graph",
        json.dumps(graph, sort_keys=True),
        "Original requested values",
        json.dumps(old, sort_keys=True),
        "Premise edit (replace the named input with this value)",
        json.dumps(asdict(task.edit), sort_keys=True),
        "Requested output keys",
        json.dumps(sorted(task.outputs)),
        "Return only a JSON object with exactly the requested keys and integer values.",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        _, cases = load_development(args.manifest)
        rows = [json.dumps({"case_id": case.case_id, "prompt": render_prompt(case.task)})
                for case in cases]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        parser.exit(1, f"prompt rendering failed: {error}\n")


if __name__ == "__main__":
    main()
