"""Generate the Day 2 audit JSONL with `python generate.py --seed 42 --output PATH`.

Inputs and replacement values are sampled from [-9, 9] excluding zero.
Every Day 2 computed node stays in [-171, 171]. Its longest chain is bounded by
(9 + 9) * 9 + 9, and the diamond joins two branches bounded by 18 each.
A replacement always differs from the original input. Each case gets at most
100 attempts to find a task that has not already been emitted.
"""

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from schema import DerivedNode, InputEdit, InputNode, RevisionTask
from verify import evaluate_revision


SAMPLE_VALUES = tuple(range(-9, 0)) + tuple(range(1, 10))
MAX_ATTEMPTS = 100
CASES_PER_CATEGORY = 5


@dataclass(frozen=True)
class AuditTemplate:
    inputs: tuple[str, ...]
    derived: tuple[DerivedNode, ...]
    outputs: tuple[str, ...]
    edit: str = "x"


_INPUTS = ("x", "k", "m")
_CHAIN = (
    DerivedNode("a", "+", "x", "k"),
    DerivedNode("b", "*", "a", "m"),
    DerivedNode("c", "-", "b", "k"),
)
_BRANCHES = (
    DerivedNode("a", "+", "x", "k"),
    DerivedNode("b", "-", "x", "m"),
)
_DISCONNECTED = (
    *_CHAIN[:2],
    DerivedNode("stable", "-", "k", "m"),
    DerivedNode("off", "+", "unused", "k"),
)
_TEMPLATES = {
    "chain": AuditTemplate(_INPUTS, _CHAIN, ("c",)),
    "diamond": AuditTemplate(
        _INPUTS, (*_BRANCHES, DerivedNode("total", "+", "a", "b")), ("total",)
    ),
    "cancellation": AuditTemplate(
        _INPUTS, (*_BRANCHES, DerivedNode("gap", "-", "a", "b")), ("a", "b", "gap")
    ),
    "irrelevant_edit": AuditTemplate(
        (*_INPUTS, "unused"), _DISCONNECTED, ("b", "stable"), edit="unused"
    ),
    "mixed_changed_stable": AuditTemplate((*_INPUTS, "unused"), _DISCONNECTED, ("b", "stable")),
    "three_or_more_changed": AuditTemplate(_INPUTS, _CHAIN, ("a", "b", "c")),
}

_DIAMOND = _TEMPLATES["diamond"].derived
_DEEP_CHAIN = (
    *_CHAIN,
    DerivedNode("d", "+", "c", "m"),
    DerivedNode("e", "-", "d", "x"),
    DerivedNode("f", "+", "e", "k"),
)
# Each entry has two output selections over the same wiring.
_EXPERIMENT_CATALOG = (
    AuditTemplate(_INPUTS, _CHAIN, ("a",)),
    AuditTemplate(_INPUTS, _CHAIN, ("c",)),
    AuditTemplate(_INPUTS, _DIAMOND, ("total",)),
    AuditTemplate(_INPUTS, _DIAMOND, ("a", "total")),
    AuditTemplate((*_INPUTS, "unused"), _DISCONNECTED, ("b",)),
    _TEMPLATES["mixed_changed_stable"],
    AuditTemplate(_INPUTS, (*_CHAIN[:2], DerivedNode("c", "-", "a", "k")), ("b",)),
    AuditTemplate(_INPUTS, (*_CHAIN[:2], DerivedNode("c", "-", "a", "k")), ("b", "c")),
    AuditTemplate(_INPUTS, (*_CHAIN[:2], DerivedNode("c", "+", "b", "b")), ("b",)),
    AuditTemplate(_INPUTS, (*_CHAIN[:2], DerivedNode("c", "+", "b", "b")), ("c",)),
    AuditTemplate(_INPUTS, (*_CHAIN[:2], DerivedNode("c", "-", "b", "a")), ("c",)),
    AuditTemplate(_INPUTS, (*_CHAIN[:2], DerivedNode("c", "-", "b", "a")), ("b", "c")),
    AuditTemplate(_INPUTS, _DEEP_CHAIN, ("d",)),
    AuditTemplate(_INPUTS, _DEEP_CHAIN, ("f",)),
    AuditTemplate(_INPUTS, (*_DEEP_CHAIN[:4], DerivedNode("e", "-", "d", "b")), ("d",)),
    AuditTemplate(_INPUTS, (*_DEEP_CHAIN[:4], DerivedNode("e", "-", "d", "b")), ("e",)),
)


def _sample_task(template: AuditTemplate, rng: random.Random) -> RevisionTask:
    values = {name: rng.choice(SAMPLE_VALUES) for name in template.inputs}
    replacements = [value for value in SAMPLE_VALUES if value != values[template.edit]]
    return RevisionTask(
        inputs=tuple(InputNode(name, value) for name, value in values.items()),
        derived=template.derived,
        outputs=template.outputs,
        edit=InputEdit(template.edit, rng.choice(replacements)),
    )


def _json(record: object) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _digest(encoded: str) -> str:
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _task_record(task: RevisionTask) -> dict[str, object]:
    return {
        "inputs": [asdict(node) for node in sorted(task.inputs, key=lambda node: node.name)],
        "derived": [asdict(node) for node in sorted(task.derived, key=lambda node: node.name)],
        "outputs": sorted(task.outputs),
        "edit": asdict(task.edit),
    }


def graph_family_id(task: RevisionTask) -> str:
    """Hash the named nodes, operations, and ordered operands of the graph."""
    graph = {
        "inputs": sorted(node.name for node in task.inputs),
        "derived": sorted(
            (node.name, node.operation, node.left, node.right) for node in task.derived
        ),
    }
    return _digest(_json(graph))


def _labeled_cases(
    seed: int, templates: tuple[tuple[str, AuditTemplate], ...]
) -> Iterator[tuple[str, RevisionTask, dict[str, object]]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    for category, template in templates:
        for index in range(1, CASES_PER_CATEGORY + 1):
            for _ in range(MAX_ATTEMPTS):
                task = _sample_task(template, rng)
                task_record = _task_record(task)
                key = _json(task_record)
                if key not in seen:
                    break
            else:
                raise RuntimeError(
                    f"{category} case {index}: sampled only duplicate tasks "
                    f"in {MAX_ATTEMPTS} attempts"
                )

            seen.add(key)
            revision = evaluate_revision(task)
            yield category, task, {
                "case_id": _digest(key),
                "graph_family_id": graph_family_id(task),
                "task": task_record,
                "old": revision.old,
                "new": revision.new,
                "changed": sorted(revision.changed),
            }


def generate_cases(seed: int) -> list[dict[str, object]]:
    """Generate five distinct tasks per audit category, labeled by the Day 1 oracle."""
    return [
        {**record, "audit_category": category}
        for category, _, record in _labeled_cases(seed, tuple(_TEMPLATES.items()))
    ]


def generate_experiment_cases(seed: int) -> Iterator[tuple[RevisionTask, dict[str, object]]]:
    """Sample labeled cases from the experiment catalog independently of the audit."""
    templates = tuple(
        (f"experiment template {index}", template)
        for index, template in enumerate(_EXPERIMENT_CATALOG, start=1)
    )
    for _, task, record in _labeled_cases(seed, templates):
        yield task, record


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 30 labeled Day 2 audit cases.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        records = generate_cases(args.seed)
        jsonl = "".join(_json(record) + "\n" for record in records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(jsonl.encode("utf-8"))
    except (RuntimeError, OSError) as error:
        parser.exit(1, f"generation failed: {error}\n")


if __name__ == "__main__":
    main()
