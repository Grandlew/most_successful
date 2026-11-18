"""Write the split manifest before prompt rendering.

The sha256 field hashes the manifest payload with that field omitted. The
payload is UTF-8 JSON with sorted keys, compact separators, and no trailing LF.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path

from generate import generate_experiment_cases
from schema import RevisionTask


MANIFEST_VERSION = 1
TRAIN_DEPTH_LIMIT = 3
SPLITS = ("train", "development", "topology_holdout", "depth_extrapolation")


def _node_depths(task: RevisionTask) -> dict[str, int]:
    depths = dict.fromkeys((node.name for node in task.inputs), 0)
    for node in task.evaluation_order:
        depths[node.name] = 1 + max(depths[node.left], depths[node.right])
    return depths


def output_depth(task: RevisionTask) -> int:
    """Return the longest path in edges from an input to a requested output."""
    depths = _node_depths(task)
    return max(depths[output] for output in task.outputs)


def shape_group(task: RevisionTask) -> str:
    """Hash an exact canonical wiring for the small DAGs in the catalog.

    Enumerate orderings within invariant depth and out-degree classes. Encode
    each vertex once, with sorted incoming vertex indices including repeated
    operands. Shared dependencies therefore keep a single vertex identity.
    """
    incoming = {node.name: (node.left, node.right) for node in task.derived}
    outgoing = Counter(operand for pair in incoming.values() for operand in pair)
    classes = defaultdict(list)
    for name, depth in _node_depths(task).items():
        classes[depth, outgoing[name]].append(name)

    def wiring(order):
        indices = {name: index for index, name in enumerate(order)}
        return tuple(tuple(sorted(indices[source] for source in incoming.get(name, ())))
                     for name in order)

    canon = min(
        wiring(tuple(name for block in blocks for name in block))
        for blocks in product(*(permutations(classes[key]) for key in sorted(classes)))
    )
    return hashlib.sha256(json.dumps(canon, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Case:
    case_id: str
    group: str
    depth: int
    operations: frozenset[str]


def _assign_groups(groups: dict[str, list[Case]], seed: int) -> dict[str, list[str]]:
    # Rank groups independently of sampling and generator iteration order.
    ranked = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).digest())
    bounds = {group: (min(c.depth for c in cases), max(c.depth for c in cases))
              for group, cases in groups.items()}
    deep = [group for group in ranked if bounds[group][0] > TRAIN_DEPTH_LIMIT]
    familiar = [group for group in ranked if group not in deep]
    if len(familiar) < 4 or not deep:
        raise ValueError("the catalog needs at least four familiar groups and one deep group")

    # Train covers both ends of the familiar range. This also keeps a group
    # together when its output selections span multiple depths.
    shallowest = min(familiar, key=lambda group: bounds[group][0])
    deepest = max(familiar, key=lambda group: bounds[group][1])
    train = [shallowest, deepest] if shallowest != deepest else [
        shallowest, next(group for group in familiar if group != shallowest)
    ]
    remaining = [group for group in familiar if group not in train]
    return dict(zip(SPLITS, (train, remaining[::2], remaining[1::2], deep)))


def _validate_splits(splits: dict[str, list[Case]]) -> None:
    seen_cases: set[str] = set()
    seen_groups: set[str] = set()
    for name, cases in splits.items():
        ids = {case.case_id for case in cases}
        groups = {case.group for case in cases}
        if not cases or len(ids) != len(cases) or seen_cases & ids or seen_groups & groups:
            raise ValueError(f"empty split or overlapping cases or groups in {name}")
        seen_cases.update(ids)
        seen_groups.update(groups)

    train = splits["train"]
    min_depth = min(case.depth for case in train)
    max_depth = max(case.depth for case in train)
    operations = set().union(*(case.operations for case in train))
    topology = splits["topology_holdout"]
    extrapolation = splits["depth_extrapolation"]
    if any(not min_depth <= case.depth <= max_depth for case in topology):
        raise ValueError("topology holdout falls outside the train depth range")
    if any(case.depth <= max_depth for case in extrapolation):
        raise ValueError("depth extrapolation must exceed maximum train depth in every case")
    if any(case.operations - operations for case in (*topology, *extrapolation)):
        raise ValueError("holdouts contain an operation absent from train")


def build_manifest(seed: int) -> dict[str, object]:
    groups = defaultdict(list)
    for task, record in generate_experiment_cases(seed):
        group = shape_group(task)
        groups[group].append(Case(
            record["case_id"], group, output_depth(task),
            frozenset(node.operation for node in task.derived),
        ))

    assignment = _assign_groups(groups, seed)
    splits = {name: [case for group in assigned for case in groups[group]]
              for name, assigned in assignment.items()}
    _validate_splits(splits)
    entries = [
        {"case_id": case.case_id, "shape_group": case.group, "split": name, "depth": case.depth}
        for name, cases in splits.items()
        for case in sorted(cases, key=lambda case: case.case_id)
    ]
    summary = {
        name: {
            "case_count": len(cases),
            "shape_group_count": len(assignment[name]),
            "min_depth": min(case.depth for case in cases),
            "max_depth": max(case.depth for case in cases),
        }
        for name, cases in splits.items()
    }
    payload = {"version": MANIFEST_VERSION, "seed": seed, "cases": entries, "summary": summary}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Day 3 split manifest.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = build_manifest(args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"split generation failed: {error}\n")


if __name__ == "__main__":
    main()
