import importlib.util
import unittest

from ddp_parity import (
    TRACE_SCALARS, TRACE_TENSORS, TOLERANCES, ParameterTrace, _compare_trace, _loss_report,
    _prepare_batches, _require_profile_gate, _tensor_report, ddp_loss_weight, global_batches,
)
from train_sft import TrainingExample


class BatchTests(unittest.TestCase):
    def test_failed_or_unmatched_parity_report_blocks_profiling(self):
        settings = {"precision": "fp16", "tolerances": TOLERANCES}
        good = {"world_size": 2, "passed": True, "settings": settings}
        _require_profile_gate(good, settings)
        for invalid in ({**good, "passed": False}, {**good, "world_size": 1},
                        {**good, "settings": {**settings, "precision": "fp32"}}):
            with self.subTest(report=invalid), self.assertRaisesRegex(ValueError, "passing two-GPU parity"):
                _require_profile_gate(invalid, settings)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "parameter trace requires torch")
    def test_trace_uses_previous_weight_for_update_delta_and_initial_for_cumulative(self):
        import torch

        weight = torch.nn.Parameter(torch.tensor([1.0, 3.0]))
        initial = weight.detach().clone()
        trace = ParameterTrace(weight, initial)
        optimizer = torch.optim.AdamW([weight], lr=0.02)
        after = []
        for gradient in ([3.0, 4.0], [-4.0, 3.0]):
            weight.grad = torch.tensor(gradient)
            trace.pre_clip()
            norm = torch.nn.utils.clip_grad_norm_([weight], 1.0)
            trace.post_clip(norm, 1024)
            optimizer.step()
            trace.post_step(optimizer, 1024, False)
            after.append(weight.detach().clone())

        first, second = trace.steps
        torch.testing.assert_close(first["gradient_pre_clip"], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(first["gradient_post_clip"], torch.tensor([0.6, 0.8]))
        self.assertAlmostEqual(first["global_pre_clip_norm"], 5.0)
        self.assertAlmostEqual(first["clip_factor"], 0.2)
        torch.testing.assert_close(first["exp_avg"], torch.tensor([0.06, 0.08]))
        torch.testing.assert_close(first["exp_avg_sq"], torch.tensor([0.00036, 0.00064]))
        torch.testing.assert_close(first["update_delta"], after[0] - initial)
        torch.testing.assert_close(second["update_delta"], after[1] - after[0])
        torch.testing.assert_close(second["cumulative_delta"], after[1] - initial)
        self.assertFalse(torch.equal(second["update_delta"], second["cumulative_delta"]))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "parameter comparison requires torch")
    def test_trace_identifies_first_cumulative_tolerance_crossing(self):
        import torch

        def row(update, cumulative):
            return {**{key: torch.tensor([0.0]) for key in TRACE_TENSORS},
                    **{key: 1.0 for key in TRACE_SCALARS}, "skipped": False,
                    "update_delta": torch.tensor([update]), "cumulative_delta": torch.tensor([cumulative])}

        reference = [row(1e-5, 1e-5), row(1e-5, 2e-5)]
        actual = [row(1.05e-5, 1.05e-5), row(1.25e-5, 2.3e-5)]
        report = _compare_trace(actual, reference)
        self.assertEqual(report["first_step_over_bound"], 2)
        self.assertTrue(report["steps"][0]["cumulative_delta_vs_original_bound"]["passed"])
        self.assertFalse(report["steps"][1]["cumulative_delta_vs_original_bound"]["passed"])
        self.assertNotIn("passed", report["steps"][1]["tensors"]["update_delta"])

    def test_order_partition_and_token_counts_match_across_world_sizes(self):
        examples = {
            key: TrainingExample([17, *range(count)], [-100, *range(count)], 1)
            for key, count in zip("abcd", (2, 4, 3, 5))
        }
        self.assertEqual(global_batches(list(examples), 2), [list("abcd"), list("bcda")])
        one = _prepare_batches(list(examples), examples, 1)
        two = _prepare_batches(list(examples), examples, 2)
        self.assertEqual(one["rank_supervised_tokens"], [14])
        self.assertEqual(two["rank_supervised_tokens"], [5, 9])
        self.assertEqual(two["rank_case_ids"], [list("ac"), list("bd")])
        for field in ("case_ids", "global_supervised_tokens", "global_non_padding_tokens"):
            self.assertEqual(one[field], two[field])
        self.assertEqual(two["global_non_padding_tokens"], 18)

    def test_rank_averaging_with_unequal_counts_and_local_losses(self):
        first = ((1, 2.0), (2, 3.0))
        second = ((4, 7.0), (2, 5.0))
        # Rank-local means are 8/3 and 38/6. An unweighted rank average
        # would give 4.5. The true global token mean is 46/9.
        scaled = [sum(mean * ddp_loss_weight(count, 9, 2) for count, mean in batches)
                  for batches in (first, second)]
        self.assertAlmostEqual(sum(scaled) / 2, 46 / 9)
        self.assertNotAlmostEqual(sum(scaled) / 2, 4.5)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "autograd comparison requires torch")
    def test_accumulated_rank_gradients_match_true_global_gradient(self):
        import torch

        ranks = (((1.0,), (2.0, 4.0)), ((3.0, 5.0, 6.0, 7.0), (8.0, 9.0)))
        count = sum(len(batch) for rank in ranks for batch in rank)
        weight = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        all_tokens = torch.tensor([token for rank in ranks for batch in rank for token in batch], dtype=weight.dtype)
        (weight * all_tokens - 1).square().mean().backward()
        expected = weight.grad.clone()
        gradients = []
        for rank in ranks:
            weight.grad = None
            for batch in rank:
                loss = (weight * torch.tensor(batch, dtype=weight.dtype) - 1).square().mean()
                (loss * ddp_loss_weight(len(batch), count, 2)).backward()
            gradients.append(weight.grad.clone())
        torch.testing.assert_close(torch.stack(gradients).mean(), expected, atol=1e-5, rtol=1e-7)

    def test_loss_tolerance_fails_nonfinite_or_excessive_error(self):
        self.assertTrue(_loss_report([1.0, 1.01], [1.0, 1.0])["passed"])
        self.assertFalse(_loss_report([None, 1.5], [1.0, 1.0])["passed"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "tensor comparison requires torch")
    def test_tensor_tolerance_reports_errors_and_worst_tensor(self):
        import torch

        tensors = {"same": torch.ones(4), "bad": torch.ones(4)}
        other = {**tensors, "bad": torch.zeros(4)}
        report = _tensor_report(other, tensors, TOLERANCES["first_gradient"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["worst_tensor"]["name"], "bad")
        self.assertEqual(report["worst_tensor"]["max_abs"], 1.0)
        self.assertEqual(report["worst_tensor"]["relative_l2"], 1.0)


if __name__ == "__main__":
    unittest.main()
