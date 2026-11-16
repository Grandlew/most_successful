import unittest
from dataclasses import replace

from schema import DerivedNode, InputEdit, InputNode, RevisionTask
from verify import AnswerCheck, InvalidAnswer, Revision, evaluate_revision, verify_answer


def cancellation_task() -> RevisionTask:
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


class RevisionTests(unittest.TestCase):
    def test_chain_and_original_inputs(self):
        task = RevisionTask(
            inputs=(InputNode("x", 2), InputNode("k", 3)),
            derived=(
                DerivedNode("sum", "+", "x", "k"),
                DerivedNode("product", "*", "sum", "k"),
                DerivedNode("result", "-", "k", "product"),
            ),
            outputs=("x", "result"),
            edit=InputEdit("x", -5),
        )
        expected = Revision(
            {"x": 2, "result": -12}, {"x": -5, "result": 9}, frozenset({"x", "result"})
        )
        self.assertEqual(evaluate_revision(task), expected)
        self.assertEqual(task.inputs, (InputNode("x", 2), InputNode("k", 3)))
        self.assertEqual(evaluate_revision(task), expected)

    def test_diamond_and_declaration_order(self):
        task = RevisionTask(
            inputs=(InputNode("x", 2), InputNode("k", 3)),
            derived=(
                DerivedNode("left", "+", "x", "k"),
                DerivedNode("right", "*", "x", "k"),
                DerivedNode("total", "+", "left", "right"),
            ),
            outputs=("total", "right"),
            edit=InputEdit("x", 5),
        )
        expected = Revision(
            {"total": 11, "right": 6},
            {"total": 23, "right": 15},
            frozenset({"total", "right"}),
        )
        for derived in (task.derived, task.derived[::-1]):
            with self.subTest(derived=derived):
                self.assertEqual(evaluate_revision(replace(task, derived=derived)), expected)

    def test_cancellation_and_revision_diagnostics(self):
        revision = evaluate_revision(cancellation_task())
        self.assertEqual(revision.old, {"a": 7, "b": 3, "gap": 4})
        self.assertEqual(revision.new, {"a": 12, "b": 8, "gap": 4})
        self.assertEqual(revision.changed, {"a", "b"})

        self.assertTrue(verify_answer(revision, '{"gap": 4, "b": 8, "a": 12}').correct)
        wrong = verify_answer(revision, '{"a": 7, "b": 6, "gap": 5}')
        self.assertFalse(wrong.correct)
        self.assertEqual(
            wrong,
            AnswerCheck(stale=frozenset({"a"}), unnecessary=frozenset({"gap"}), incorrect=frozenset({"b"})),
        )

    def test_irrelevant_edit(self):
        task = RevisionTask(
            inputs=(InputNode("x", 2), InputNode("unused", 3)),
            derived=(DerivedNode("square", "*", "x", "x"),),
            outputs=("square", "x"),
            edit=InputEdit("unused", 7),
        )
        values = {"square": 4, "x": 2}
        self.assertEqual(evaluate_revision(task), Revision(values, values, frozenset()))

    def test_rejects_empty_outputs(self):
        with self.assertRaisesRegex(ValueError, "at least one requested output"):
            replace(cancellation_task(), outputs=())

    def test_invalid_graphs(self):
        task = replace(cancellation_task(), outputs=("a",))
        cases = (
            ("duplicate node", {"inputs": (*task.inputs, InputNode("x", 1))}),
            ("duplicate node", {"derived": (*task.derived, task.derived[0])}),
            ("duplicate node", {"derived": (*task.derived, DerivedNode("x", "+", "a", "b"))}),
            ("unknown operand", {"derived": (DerivedNode("a", "+", "x", "missing"),)}),
            ("unknown requested output", {"outputs": ("missing",)}),
            ("unknown edit target", {"edit": InputEdit("missing", 9)}),
            ("duplicate requested output", {"outputs": ("a", "a")}),
            ("unsupported operation", {"derived": (DerivedNode("a", "/", "x", "one"),)}),
            ("cannot edit derived node", {"edit": InputEdit("a", 9)}),
            ("cycle", {"derived": (DerivedNode("a", "+", "a", "x"),)}),
            ("cycle", {"derived": (
                DerivedNode("a", "+", "b", "x"), DerivedNode("b", "-", "a", "x"),
            )}),
            ("must have an integer value", {"inputs": (InputNode("x", True),)}),
            ("edit value must be an integer", {"edit": InputEdit("x", 9.0)}),
        )
        for message, changes in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                replace(task, **changes)

    def test_rejects_invalid_answers_even_when_true_equals_one_in_python(self):
        task = RevisionTask((InputNode("x", 0),), (), ("x",), InputEdit("x", 1))
        revision = evaluate_revision(task)
        self.assertTrue(verify_answer(revision, '{"x": 1}').correct)
        cases = (
            ('{"x": 1,}', "invalid JSON"),
            ('{"x": 1} {}', "invalid JSON"),
            ('{"x": 1, "x": 1}', "duplicate JSON key"),
            ('{"x": 1, "\\u0078": 1}', "duplicate JSON key"),
            ('{}', "missing"),
            ('{"x": 1, "y": 2}', "extra"),
            ('{"x": true}', "JSON integer"),
            ('{"x": 1.0}', "JSON integer"),
            ('{"x": 1e0}', "JSON integer"),
            ('{"x": "1"}', "JSON integer"),
            ('{"x": null}', "JSON integer"),
            ('{"x": NaN}', "invalid JSON constant"),
            ('[1]', "JSON object"),
        )
        for answer, message in cases:
            with self.subTest(answer=answer), self.assertRaisesRegex(InvalidAnswer, message):
                verify_answer(revision, answer)

    def test_accepts_json_integers_larger_than_the_default_conversion_limit(self):
        task = RevisionTask((InputNode("x", 1),), (), ("x",), InputEdit("x", 10**5000))
        answer = '{"x": 1' + "0" * 5000 + "}"
        self.assertTrue(verify_answer(evaluate_revision(task), answer).correct)


if __name__ == "__main__":
    unittest.main()
