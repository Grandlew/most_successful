"""Validated arithmetic revision tasks with named inputs and binary operations."""

from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from typing import Literal


Operation = Literal["+", "-", "*"]


@dataclass(frozen=True)
class InputNode:
    name: str
    value: int


@dataclass(frozen=True)
class DerivedNode:
    name: str
    operation: Operation
    left: str
    right: str


@dataclass(frozen=True)
class InputEdit:
    name: str
    value: int


@dataclass(frozen=True)
class RevisionTask:
    inputs: tuple[InputNode, ...]
    derived: tuple[DerivedNode, ...]
    outputs: tuple[str, ...]
    edit: InputEdit
    evaluation_order: tuple[DerivedNode, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Snapshot the declarations so validation remains valid if callers pass lists.
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "derived", tuple(self.derived))
        object.__setattr__(self, "outputs", tuple(self.outputs))

        names: set[str] = set()
        for node in (*self.inputs, *self.derived):
            if not isinstance(node.name, str) or not node.name:
                raise ValueError("node names must be nonempty strings")
            if node.name in names:
                raise ValueError(f"duplicate node name: {node.name!r}")
            names.add(node.name)

        for node in self.inputs:
            if type(node.value) is not int:
                raise ValueError(f"input {node.name!r} must have an integer value")

        if self.edit.name not in names:
            raise ValueError(f"unknown edit target: {self.edit.name!r}")
        derived_by_name = {node.name: node for node in self.derived}
        if self.edit.name in derived_by_name:
            raise ValueError(f"cannot edit derived node: {self.edit.name!r}")
        if type(self.edit.value) is not int:
            raise ValueError("edit value must be an integer")

        if not self.outputs:
            raise ValueError("a revision task requires at least one requested output")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("duplicate requested output")
        for output in self.outputs:
            if output not in names:
                raise ValueError(f"unknown requested output: {output!r}")

        dependencies: dict[str, tuple[str, ...]] = {node.name: () for node in self.inputs}
        for node in self.derived:
            if node.operation not in ("+", "-", "*"):
                raise ValueError(f"unsupported operation on {node.name!r}: {node.operation!r}")
            operands = (node.left, node.right)
            for operand in operands:
                if operand not in names:
                    raise ValueError(f"unknown operand {operand!r} on {node.name!r}")
            dependencies[node.name] = operands

        try:
            order = tuple(TopologicalSorter(dependencies).static_order())
        except CycleError as error:
            raise ValueError(f"cycle in dependencies: {error.args[1]}") from error

        evaluation_order = tuple(
            derived_by_name[name] for name in order if name in derived_by_name
        )
        object.__setattr__(self, "evaluation_order", evaluation_order)
