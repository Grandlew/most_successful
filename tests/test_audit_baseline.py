import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from audit_baseline import _classify, audit, extract_json_object
from baseline_qwen import MAX_NEW_TOKENS, greedy_continuation
from evaluate import evaluate
from render_prompts import DEFAULT_MANIFEST, DevelopmentCase, load_development
from test_revision import cancellation_task
from test_sft import HAS_MODEL_DEPS, ToyTokenizer
from verify import InvalidAnswer, evaluate_revision, verify_answer


class ExtractionTests(unittest.TestCase):
    def test_plain_fenced_and_prose_preserve_exact_object_text(self):
        object_text = '{\n  "gap":4, "a": 12,"b":8\n}'
        for answer in (object_text, f"```json\n{object_text}\n```", f"Here is the result: {object_text}. Done."):
            with self.subTest(answer=answer):
                self.assertEqual(extract_json_object(answer), object_text)
                self.assertTrue(verify_answer(evaluate_revision(cancellation_task()), object_text).correct)

    def test_duplicate_keys_are_preserved_and_rejected_by_verifier(self):
        object_text = '{"a":12,"a":12,"b":8,"gap":4}'
        self.assertEqual(extract_json_object(f"Answer: {object_text}"), object_text)
        with self.assertRaisesRegex(InvalidAnswer, "duplicate JSON key"):
            verify_answer(evaluate_revision(cancellation_task()), object_text)

    def test_ambiguous_truncated_and_nonobject_replies_are_rejected(self):
        for answer in ('{"a":1} or {"a":2}', '{"a":1} then {"a":',
                       '```json\n{"a":12,', '[{"a":1}]', '"{}"', 'No answer.'):
            with self.subTest(answer=answer):
                self.assertIsNone(extract_json_object(answer))

    def test_strings_with_braces_and_nested_objects_keep_outer_span(self):
        text = json.dumps({"note": 'literal \\"} {', "nested": {"x": 1}})
        self.assertEqual(extract_json_object(f"Response {text}"), text)

    def test_classifications_record_identifiable_wrong_values_and_metadata_uncertainty(self):
        case = DevelopmentCase("fixture", "diamond", 2, cancellation_task())
        fixtures = (
            ('{"a":7,"b":8}', "wrong_keys"),
            ('{"a":7,"b":"8","gap":4}', "wrong_types"),
            ('{"a":7,"b":8,"b":8,"gap":4}', "duplicate_keys"),
            ('{"a":7,"b":8,"gap":4}', "wrong_values"),
        )
        for answer, category in fixtures:
            with self.subTest(category=category):
                strict = evaluate([case], [(case.case_id, answer)])["cases"][0]
                row = _classify(case, answer, strict, None)
                self.assertEqual(row["category"], category)
                self.assertEqual(row["wrong_values"], {"a": {"observed": 7, "expected": 12}})
                self.assertIsNone(row["truncation_confirmed"])
        answer = '{"a":12,'
        strict = evaluate([case], [(case.case_id, answer)])["cases"][0]
        unknown = _classify(case, answer, strict, None)
        self.assertTrue(unknown["uncertain"])
        self.assertIsNone(unknown["truncation_confirmed"])
        cap = {"stop_reason": "token_cap", "new_token_ids": [1] * MAX_NEW_TOKENS, "eos_token_id": 0}
        self.assertTrue(_classify(case, answer, strict, cap)["truncation_confirmed"])

    def test_audit_reproduces_strict_score_and_reports_separate_diagnostic_score(self):
        checksum, cases = load_development(DEFAULT_MANIFEST)
        predictions = [(case.case_id, "") for case in cases]
        predictions[0] = (cases[0].case_id, "```json\n" + json.dumps(evaluate_revision(cases[0].task).new) + "\n```")
        with TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            saved = Path(directory) / "score.json"
            raw.write_text("".join(json.dumps({"case_id": key, "answer": answer}) + "\n"
                                   for key, answer in predictions), encoding="utf-8")
            saved.write_text(json.dumps({"manifest_sha256": checksum, "prediction_source": "predictions",
                                         **evaluate(cases, predictions)}), encoding="utf-8")
            report = audit(DEFAULT_MANIFEST, raw, saved)
            self.assertEqual(report["strict_metrics"]["joint_exact"]["numerator"], 0)
            self.assertEqual(report["extraction_joint_exact"], {"numerator": 1, "denominator": 20})
            self.assertEqual(report["extraction_coverage"], {"numerator": 1, "denominator": 20})
            self.assertEqual(report["reply_counts"], {"extra_text_or_fence": 1, "no_usable_answer": 19})
            self.assertEqual(report["uncertain_reply_count"], 19)
            raw.write_text("\n".join(raw.read_text(encoding="utf-8").splitlines()[:-1]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly the 20 frozen development IDs"):
                audit(DEFAULT_MANIFEST, raw, saved)


class GenerationEvidenceTests(unittest.TestCase):
    @unittest.skipUnless(HAS_MODEL_DEPS, "CPU generation test requires torch and transformers")
    def test_first_token_eos_and_token_cap_have_distinct_evidence(self):
        import torch

        class FakeModel:
            def __init__(self, suffix):
                self.suffix = suffix

            def generate(self, *, input_ids, **kwargs):
                return torch.cat((input_ids, torch.tensor([self.suffix])), dim=1)

        for suffix, reason, text in (([0], "eos", ""), ([ord("x")] * MAX_NEW_TOKENS, "token_cap", "x" * MAX_NEW_TOKENS)):
            with self.subTest(reason=reason):
                answer, evidence = greedy_continuation(FakeModel(suffix), ToyTokenizer(), "prompt", "cpu")
                self.assertEqual(answer, text)
                self.assertEqual(evidence, {"stop_reason": reason, "eos_token_id": 0, "new_token_ids": suffix})


if __name__ == "__main__":
    unittest.main()
