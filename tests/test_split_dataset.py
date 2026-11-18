import hashlib
import json
import subprocess
import sys
import unittest
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from generate import generate_cases, generate_experiment_cases
from schema import DerivedNode, InputEdit, InputNode, RevisionTask
from split_dataset import build_manifest, output_depth, shape_group


ROOT = Path(__file__).resolve().parents[1]


def load_task(record):
    return RevisionTask(
        tuple(InputNode(**node) for node in record["inputs"]),
        tuple(DerivedNode(**node) for node in record["derived"]),
        tuple(record["outputs"]),
        InputEdit(**record["edit"]),
    )


class ShapeTests(unittest.TestCase):
    def test_day_two_addition_and_cancellation_diamonds_share_a_group(self):
        diamonds = {
            category: [load_task(case["task"]) for case in generate_cases(42)
                       if case["audit_category"] == category]
            for category in ("diamond", "cancellation")
        }
        self.assertEqual(diamonds["diamond"][0].derived[-1].operation, "+")
        self.assertEqual(diamonds["cancellation"][0].derived[-1].operation, "-")
        self.assertEqual(len({shape_group(task) for tasks in diamonds.values() for task in tasks}), 1)

    def test_group_ignores_renaming_order_values_operators_edit_and_outputs(self):
        original = RevisionTask(
            (InputNode("x", 2), InputNode("k", 3)),
            (DerivedNode("a", "+", "x", "k"), DerivedNode("b", "*", "a", "x")),
            ("b",), InputEdit("x", 4),
        )
        variant = RevisionTask(
            (InputNode("y", 8), InputNode("z", 7)),
            (DerivedNode("p", "-", "z", "q"), DerivedNode("q", "*", "y", "z")),
            ("q",), InputEdit("y", 9),
        )
        self.assertEqual(shape_group(original), shape_group(variant))
        self.assertEqual(output_depth(original), 2)
        self.assertEqual(output_depth(variant), 1)
        self.assertEqual(output_depth(replace(variant, outputs=("z",))), 0)

    def test_group_preserves_shared_dependencies(self):
        shared = RevisionTask(
            (InputNode("x", 2), InputNode("y", 3), InputNode("z", 4)),
            (DerivedNode("a", "+", "x", "z"), DerivedNode("b", "+", "x", "z"),
             DerivedNode("c", "+", "a", "b")),
            ("c",), InputEdit("x", 5),
        )
        separate = replace(shared, derived=(
            shared.derived[0], DerivedNode("b", "+", "y", "z"), shared.derived[2],
        ))
        self.assertEqual(output_depth(shared), output_depth(separate))
        self.assertNotEqual(shape_group(shared), shape_group(separate))


class ManifestTests(unittest.TestCase):
    def test_membership_depth_ranges_and_operations(self):
        for seed in (42, 43):
            with self.subTest(seed=seed):
                manifest = build_manifest(seed)
                tasks = {record["case_id"]: task for task, record in generate_experiment_cases(seed)}
                rows = manifest["cases"]
                self.assertEqual(Counter(row["case_id"] for row in rows), dict.fromkeys(tasks, 1))
                self.assertEqual(len(rows), 80)
                groups = defaultdict(set)
                depths = defaultdict(set)
                operations = defaultdict(set)
                group_depths = defaultdict(set)
                for row in rows:
                    task = tasks[row["case_id"]]
                    group, split, depth = row["shape_group"], row["split"], row["depth"]
                    self.assertEqual(group, shape_group(task))
                    self.assertEqual(depth, output_depth(task))
                    groups[group].add(split)
                    depths[split].add(depth)
                    group_depths[group].add(depth)
                    operations[split].update(node.operation for node in task.derived)

                self.assertEqual(len(groups), 8)
                self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
                self.assertTrue(any(len(values) > 1 for values in group_depths.values()))
                self.assertEqual(set(depths), {"train", "development", "topology_holdout", "depth_extrapolation"})
                self.assertEqual(Counter(row["split"] for row in rows), dict.fromkeys(depths, 20))
                self.assertEqual((min(depths["train"]), max(depths["train"])), (1, 3))
                self.assertTrue(all(1 <= depth <= 3 for depth in depths["topology_holdout"]))
                self.assertTrue(all(depth > 3 for depth in depths["depth_extrapolation"]))
                self.assertEqual(operations["train"], {"+", "-", "*"})
                self.assertLessEqual(operations["topology_holdout"], operations["train"])
                self.assertLessEqual(operations["depth_extrapolation"], operations["train"])

    def test_cli_regenerates_identical_manifest_and_checksum(self):
        with TemporaryDirectory() as directory:
            paths = [Path(directory) / filename for filename in ("first.json", "second.json")]
            for path in paths:
                result = subprocess.run(
                    [sys.executable, "split_dataset.py", "--seed", "42", "--output", str(path)],
                    cwd=ROOT, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            output = paths[0].read_bytes()
            self.assertEqual(output, paths[1].read_bytes())
        checked_in = (ROOT / "data" / "day03_manifest.json").read_text(encoding="utf-8")
        self.assertEqual(output, checked_in.encode("utf-8"))
        payload = json.loads(output)
        checksum = payload.pop("sha256")
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["seed"], 42)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(checksum, hashlib.sha256(canonical).hexdigest())


if __name__ == "__main__":
    unittest.main()
