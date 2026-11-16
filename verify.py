"""Evaluate arithmetic revisions and diagnose candidate JSON answers."""

import json
from dataclasses import dataclass
from decimal import Decimal
from operator import add, mul, sub

from schema import RevisionTask


_OPERATIONS = {"+": add, "-": sub, "*": mul}


@dataclass(frozen=True)
class Revision:
    old: dict[str, int]
    new: dict[str, int]
    changed: frozenset[str]


@dataclass(frozen=True)
class AnswerCheck:
    stale: frozenset[str]
    unnecessary: frozenset[str]
    incorrect: frozenset[str]

    @property
    def correct(self) -> bool:
        return not (self.stale or self.unnecessary or self.incorrect)


class InvalidAnswer(ValueError):
    """The candidate does not have the required JSON answer format."""


def _evaluate(task: RevisionTask, inputs: dict[str, int]) -> dict[str, int]:
    values = inputs.copy()
    for node in task.evaluation_order:
        values[node.name] = _OPERATIONS[node.operation](values[node.left], values[node.right])
    return {name: values[name] for name in task.outputs}


def evaluate_revision(task: RevisionTask) -> Revision:
    """Return requested values before and after the edit, and the outputs that changed."""
    inputs = {node.name: node.value for node in task.inputs}
    old = _evaluate(task, inputs)
    new = _evaluate(task, {**inputs, task.edit.name: task.edit.value})
    return Revision(old, new, frozenset(name for name in old if old[name] != new[name]))


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidAnswer(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise InvalidAnswer(f"invalid JSON constant: {token}")


def _parse_integer(token: str) -> int:
    try:
        return int(token)
    except ValueError:
        # JSON has already validated the token. Arithmetic can exceed int's text digit limit.
        return int(Decimal(token))


def verify_answer(revision: Revision, answer: str) -> AnswerCheck:
    """Raise InvalidAnswer for format errors or classify each wrong output once."""
    if not isinstance(answer, str):
        raise InvalidAnswer("answer must be JSON text")
    try:
        candidate = json.loads(
            answer,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_int=_parse_integer,
        )
    except InvalidAnswer:
        raise
    except (ValueError, RecursionError) as error:
        raise InvalidAnswer(f"invalid JSON: {error}") from error

    if not isinstance(candidate, dict):
        raise InvalidAnswer("answer must be a JSON object")

    missing = revision.new.keys() - candidate.keys()
    extra = candidate.keys() - revision.new.keys()
    if missing or extra:
        raise InvalidAnswer(f"wrong output keys: missing={sorted(missing)}, extra={sorted(extra)}")

    stale: set[str] = set()
    unnecessary: set[str] = set()
    incorrect: set[str] = set()
    for name, expected in revision.new.items():
        value = candidate[name]
        if type(value) is not int:
            raise InvalidAnswer(f"output {name!r} must be a JSON integer")
        if value == expected:
            continue
        if name not in revision.changed:
            unnecessary.add(name)
        elif value == revision.old[name]:
            stale.add(name)
        else:
            incorrect.add(name)

    return AnswerCheck(frozenset(stale), frozenset(unnecessary), frozenset(incorrect))
