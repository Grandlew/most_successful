import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evaluate import baseline_predictions, evaluate, read_predictions
from render_prompts import DEFAULT_MANIFEST, DevelopmentCase, load_development, render_prompt
from schema import DerivedNode, InputEdit, InputNode, RevisionTask
from split_dataset import manifest_checksum
from verify import evaluate_revision


def cancellation_task():
    return RevisionTask(
        inputs=(InputNode("x", 4), InputNode("three", 3), InputNode("one", 1)),
        derived=(
            DerivedNode("a", "+", "x", "three"),
            DerivedNode("b", "-", "x", "one"),
            DerivedNode("gap", "-", "a", "b"),
        ),
        outputs=("a", "b", "gap"),
        edit=InputEdit("x", 9),
    )


class EvaluateTests(unittest.TestCase):
    def assert_metric(self, report, name, numerator, denominator):
        self.assertEqual(report["metrics"][name], {
            "numerator": numerator,
            "denominator": denominator,
            "rate": numerator / denominator if denominator else None,
        })

    def test_cancellation_scores_all_five_responses(self):
        task = cancellation_task()
        revision = evaluate_revision(task)
        self.assertEqual(revision.old, {"a": 7, "b": 3, "gap": 4})
        self.assertEqual(revision.new, {"a": 12, "b": 8, "gap": 4})
        cases = [DevelopmentCase(str(index), "diamond", 2, task) for index in range(5)]
        report = evaluate(cases, [
            ("0", '{"a":12,"b":8,"gap":4}'),
            ("1", '{"a":7,"b":8,"gap":4}'),
            ("2", '{"a":12,"b":8,"gap":5}'),
            ("3", '{"a":12,"b":8}'),
        ])
        self.assert_metric(report, "joint_exact", 1, 5)
        self.assert_metric(report, "changed_accuracy", 5, 10)
        self.assert_metric(report, "stable_preservation", 2, 5)
        self.assert_metric(report, "stale_values", 1, 10)
        self.assert_metric(report, "unnecessary_changes", 1, 5)
        self.assert_metric(report, "malformed_answers", 1, 5)
        self.assert_metric(report, "missing_predictions", 1, 5)
        rows = report["cases"]
        self.assertEqual([row["status"] for row in rows],
                         ["correct", "incorrect", "incorrect", "malformed", "missing"])
        self.assertEqual(rows[1]["stale_outputs"], ["a"])
        self.assertEqual(rows[2]["unnecessary_changes"], ["gap"])
        self.assertEqual({(row["shape_group"], row["depth"]) for row in rows}, {("diamond", 2)})

    def test_duplicate_and_unknown_predictions_are_rejected(self):
        case = DevelopmentCase("known", "diamond", 2, cancellation_task())
        with self.assertRaisesRegex(ValueError, "duplicate prediction case ID: known"):
            evaluate([case], [("known", "{}"), ("known", "{}")])
        with self.assertRaisesRegex(ValueError, "unknown prediction case ID: unknown"):
            evaluate([case], [("unknown", "{}")])

    def test_zero_eligible_outputs_return_null_rate(self):
        task = replace(cancellation_task(), outputs=("gap",))
        case = DevelopmentCase("stable", "diamond", 2, task)
        report = evaluate([case], [("stable", '{"gap":4}')])
        self.assert_metric(report, "changed_accuracy", 0, 0)
        self.assert_metric(report, "stale_values", 0, 0)
        self.assert_metric(report, "stable_preservation", 1, 1)

    def test_prediction_reader_preserves_raw_duplicate_answer_keys(self):
        case = DevelopmentCase("known", "diamond", 2, cancellation_task())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            record = {"case_id": "known", "answer": '{"a":12,"a":12,"b":8,"gap":4}'}
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            report = evaluate([case], read_predictions(path))
        self.assert_metric(report, "malformed_answers", 1, 1)
        self.assertIn("duplicate JSON key", report["cases"][0]["error"])

    def test_frozen_development_baselines(self):
        _, cases = load_development(DEFAULT_MANIFEST)
        for baseline, joint, changed in (("copy-original", 0, 0), ("oracle-revised", 20, 29)):
            with self.subTest(baseline=baseline):
                report = evaluate(cases, baseline_predictions(cases, baseline))
                self.assert_metric(report, "joint_exact", joint, 20)
                self.assert_metric(report, "changed_accuracy", changed, 29)
                self.assert_metric(report, "stable_preservation", 1, 1)


class RenderTests(unittest.TestCase):
    def test_prompt_contains_only_allowed_fields(self):
        lines = render_prompt(cancellation_task()).splitlines()
        self.assertEqual(lines[::2], [
            "Recompute the requested outputs after applying the single premise edit.",
            json.dumps({
                "inputs": {"x": 4, "three": 3, "one": 1},
                "derived": [
                    {"name": "a", "operation": "+", "left": "x", "right": "three"},
                    {"name": "b", "operation": "-", "left": "x", "right": "one"},
                    {"name": "gap", "operation": "-", "left": "a", "right": "b"},
                ],
            }, sort_keys=True),
            '{"a": 7, "b": 3, "gap": 4}',
            '{"name": "x", "value": 9}',
            '["a", "b", "gap"]',
        ])
        self.assertEqual(lines[1::2], [
            "Original graph", "Original requested values",
            "Premise edit (replace the named input with this value)", "Requested output keys",
            "Return only a JSON object with exactly the requested keys and integer values.",
        ])

    def test_manifest_rejects_tampering_and_missing_or_duplicate_selected_ids(self):
        original = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        selected = next(row for row in original["cases"] if row["split"] == "development")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({**original, "seed": 43}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest checksum mismatch"):
                load_development(path)

            payload = {key: value for key, value in original.items() if key != "sha256"}
            mismatched = [dict(row) for row in payload["cases"]]
            next(row for row in mismatched if row == selected)["case_id"] = "unavailable"
            for rows, error in (
                (mismatched, "missing regenerated development case IDs: unavailable"),
                ([*payload["cases"], {**selected, "split": "train"}], "duplicate development case ID"),
            ):
                with self.subTest(error=error):
                    changed = {**payload, "cases": rows}
                    manifest = {**changed, "sha256": manifest_checksum(changed)}
                    path.write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        load_development(path)

    def test_join_rejects_duplicate_regenerated_selected_id(self):
        _, cases = load_development(DEFAULT_MANIFEST)
        case = cases[0]
        repeated = [(case.task, {"case_id": case.case_id})] * 2
        with patch("render_prompts.generate_experiment_cases", return_value=iter(repeated)):
            with self.assertRaisesRegex(ValueError, "duplicate regenerated development case ID"):
                load_development(DEFAULT_MANIFEST)


if __name__ == "__main__":
    unittest.main()
