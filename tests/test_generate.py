import json
import operator
import subprocess
import sys
import unittest
from collections import Counter
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from generate import MAX_ATTEMPTS, generate_cases
from schema import InputEdit, InputNode, RevisionTask


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COUNTS = {
    "chain": 5,
    "diamond": 5,
    "cancellation": 5,
    "irrelevant_edit": 5,
    "mixed_changed_stable": 5,
    "three_or_more_changed": 5,
}


def operands(node):
    return node["left"], node["right"]


def reference_values(inputs, derived):
    """Walk from each node toward its inputs to check labels and intermediate values."""
    operations = {"+": operator.add, "-": operator.sub, "*": operator.mul}

    @lru_cache(maxsize=None)
    def value(name):
        if name in inputs:
            return inputs[name]
        node = derived[name]
        left, right = operands(node)
        return operations[node["operation"]](value(left), value(right))

    return {name: value(name) for name in (*inputs, *derived)}


class GenerateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = generate_cases(42)

    def test_cli_is_byte_reproducible_and_matches_audit(self):
        with TemporaryDirectory() as directory:
            outputs = [Path(directory) / name / "audit.jsonl" for name in ("first", "second")]
            for path in outputs:
                result = subprocess.run(
                    [sys.executable, "generate.py", "--seed", "42", "--output", str(path)],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            actual = outputs[0].read_bytes()
            self.assertEqual(actual, outputs[1].read_bytes())
            # Git may check out the committed LF file with CRLF on Windows.
            audit = (ROOT / "data" / "day02_audit.jsonl").read_text(encoding="utf-8")
            self.assertEqual(actual, audit.encode("utf-8"))
            self.assertEqual([json.loads(line) for line in actual.splitlines()], self.cases)

    def test_unique_tasks_and_exact_category_counts(self):
        self.assertEqual(Counter(case["audit_category"] for case in self.cases), EXPECTED_COUNTS)
        self.assertEqual(len({case["case_id"] for case in self.cases}), 30)
        task_keys = {json.dumps(case["task"], sort_keys=True) for case in self.cases}
        self.assertEqual(len(task_keys), 30)

    def test_each_case_meets_its_assigned_category(self):
        for case in self.cases:
            category = case["audit_category"]
            task = case["task"]
            edited = task["edit"]["name"]
            derived = {node["name"]: node for node in task["derived"]}
            outputs = set(task["outputs"])
            changed = set(case["changed"])
            stable = outputs - changed

            def ancestors(name):
                if name not in derived:
                    return set()
                left, right = operands(derived[name])
                return {left, right} | ancestors(left) | ancestors(right)

            descendants = {name for name in outputs if edited in ancestors(name)}
            with self.subTest(category=category, case_id=case["case_id"]):
                if category == "chain":
                    (end,) = task["outputs"]
                    path = {end} | ancestors(end)
                    self.assertEqual(path.intersection(derived), set(derived))
                    self.assertGreaterEqual(len(derived), 2)
                    self.assertIn(end, descendants)
                    for node in derived.values():
                        on_path = [
                            name for name in operands(node)
                            if name == edited or edited in ancestors(name)
                        ]
                        self.assertEqual(len(on_path), 1)
                    self.assertEqual(changed, {end})
                elif category == "diamond":
                    (join,) = task["outputs"]
                    left, right = operands(derived[join])
                    self.assertNotEqual(left, right)
                    self.assertNotIn(left, ancestors(right))
                    self.assertNotIn(right, ancestors(left))
                    self.assertIn(edited, ancestors(left) & ancestors(right))
                    self.assertEqual(changed, {join})
                elif category == "cancellation":
                    self.assertTrue(descendants & stable)
                    self.assertTrue(descendants & changed)
                elif category == "irrelevant_edit":
                    self.assertNotIn(edited, outputs)
                    self.assertFalse(descendants)
                    self.assertFalse(changed)
                elif category == "mixed_changed_stable":
                    self.assertTrue(changed)
                    self.assertTrue(stable)
                elif category == "three_or_more_changed":
                    self.assertGreaterEqual(len(changed), 3)
                else:
                    self.fail(f"unknown audit category {category!r}")

    def test_labels_and_bounds_with_independent_arithmetic(self):
        for case in self.cases:
            task = case["task"]
            inputs = {node["name"]: node["value"] for node in task["inputs"]}
            derived = {node["name"]: node for node in task["derived"]}
            edited = task["edit"]["name"]
            replacement = task["edit"]["value"]
            with self.subTest(case_id=case["case_id"]):
                for value in (*inputs.values(), replacement):
                    self.assertIs(type(value), int)
                    self.assertLessEqual(1, abs(value))
                    self.assertLessEqual(abs(value), 9)
                self.assertNotEqual(inputs[edited], replacement)
                old = reference_values(inputs, derived)
                new = reference_values({**inputs, edited: replacement}, derived)
                largest = max(abs(value) for value in (*old.values(), *new.values()))
                self.assertLessEqual(largest, 171)
                outputs = task["outputs"]
                self.assertEqual(case["old"], {name: old[name] for name in outputs})
                self.assertEqual(case["new"], {name: new[name] for name in outputs})
                expected_changes = sorted(name for name in outputs if old[name] != new[name])
                self.assertEqual(case["changed"], expected_changes)

    def test_targeted_seed_42_labels(self):
        cancellation = next(case for case in self.cases if case["audit_category"] == "cancellation")
        # Edit x from 9 to -7 with k=-6 and m=4. The gap is k+m=-2.
        self.assertEqual(cancellation["old"], {"a": 3, "b": 5, "gap": -2})
        self.assertEqual(cancellation["new"], {"a": -13, "b": -11, "gap": -2})
        self.assertEqual(cancellation["changed"], ["a", "b"])

        irrelevant = next(
            case for case in self.cases if case["audit_category"] == "irrelevant_edit"
        )
        # The edit is on unused. Requested b=(3+3)*(-3) and stable=3-(-3).
        self.assertEqual(irrelevant["old"], {"b": -18, "stable": 6})
        self.assertEqual(irrelevant["new"], {"b": -18, "stable": 6})
        self.assertEqual(irrelevant["changed"], [])

    def test_family_ids_are_shared_by_graphs_across_seeds_and_categories(self):
        other_seed = generate_cases(43)
        self.assertNotEqual(
            [case["case_id"] for case in self.cases], [case["case_id"] for case in other_seed]
        )
        families = {category: set() for category in EXPECTED_COUNTS}
        for case in (*self.cases, *other_seed):
            families[case["audit_category"]].add(case["graph_family_id"])
        self.assertTrue(all(len(ids) == 1 for ids in families.values()))
        self.assertEqual(families["chain"], families["three_or_more_changed"])
        self.assertEqual(families["irrelevant_edit"], families["mixed_changed_stable"])
        self.assertEqual(len(set.union(*families.values())), 4)

    def test_duplicate_sampling_stops_and_reports_failure(self):
        task = RevisionTask((InputNode("x", 1),), (), ("x",), InputEdit("x", 2))
        with patch("generate._sample_task", return_value=task) as sample:
            with self.assertRaisesRegex(RuntimeError, rf"chain case 2: .*{MAX_ATTEMPTS} attempts"):
                generate_cases(42)
        self.assertEqual(sample.call_count, MAX_ATTEMPTS + 1)


if __name__ == "__main__":
    unittest.main()
