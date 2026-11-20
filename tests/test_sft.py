import importlib.util
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import baseline_qwen
from evaluate import evaluate, read_predictions
from render_prompts import DEFAULT_MANIFEST, load_development, render_prompt
from train_sft import (
    IGNORE_INDEX, _make_optimizer, _prepare_model, _weighted_microbatches,
    collate_examples, encode_example, select_tiny_cases,
)
from verify import evaluate_revision


HAS_MODEL_DEPS = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))


class ToyTokenizer:
    eos_token_id = pad_token_id = 0

    def encode(self, text, *, add_special_tokens, truncation):
        assert not add_special_tokens and not truncation
        return [ord(char) for char in text]

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert not skip_special_tokens and not clean_up_tokenization_spaces
        return "".join(chr(token) for token in ids)


class SFTTests(unittest.TestCase):
    def test_lora_mode_wraps_the_original_model_and_unknown_modes_are_rejected(self):
        peft = SimpleNamespace(LoraConfig=Mock(return_value="config"),
                               get_peft_model=Mock(return_value="wrapped"))
        with (patch.dict("sys.modules", {"peft": peft}),
              patch("train_sft.load_qwen", return_value=("base", "tokenizer")) as load):
            self.assertEqual(_prepare_model("lora", "cpu"), ("wrapped", "tokenizer"))
            load.assert_called_once_with("cpu")
            peft.get_peft_model.assert_called_once_with("base", "config")
            with self.assertRaisesRegex(ValueError, "unknown training mode"):
                _prepare_model("unknown", "cpu")

    @unittest.skipUnless(HAS_MODEL_DEPS, "CPU optimizer test requires torch")
    def test_full_mode_optimizes_every_base_parameter_without_wrapping(self):
        import torch

        base = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
        base.requires_grad_(False)
        with patch("train_sft.load_qwen", return_value=(base, "tokenizer")) as load:
            model, tokenizer = _prepare_model("full", "cpu")
            load.assert_called_once_with("cpu", fp32=True)
            self.assertIs(model, base)
            self.assertEqual(tokenizer, "tokenizer")
            self.assertFalse(hasattr(model, "peft_config"))
            optimizer, trainable = _make_optimizer(model, 0.01)
            self.assertEqual({id(p) for p in optimizer.param_groups[0]["params"]},
                             {id(p) for p in base.parameters()})
            self.assertTrue(all(p.requires_grad and p.dtype == torch.float32 for p in trainable))
            before = base[-1].bias.detach().clone()
            base(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, base[-1].bias))

            base.peft_config = {"adapter": "present"}
            with self.assertRaisesRegex(RuntimeError, "without adapters"):
                _prepare_model("full", "cpu")

    def test_microbatch_means_are_weighted_by_supervised_tokens_including_eos(self):
        tokenizer = ToyTokenizer()
        short = encode_example(tokenizer, "Long prompt.", '{"a":1}')
        long = encode_example(tokenizer, "Edit.", '{"a":-42}')
        self.assertEqual(_weighted_microbatches([short, long], 1),
                         [([short], 8 / 18), ([long], 10 / 18)])
        self.assertEqual(_weighted_microbatches([short, long], 2), [([short, long], 1)])

    def test_four_ids_are_deterministic_and_train_only(self):
        checksum, cases = select_tiny_cases(DEFAULT_MANIFEST)
        manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(checksum, manifest["sha256"])
        expected = sorted(row["case_id"] for row in manifest["cases"] if row["split"] == "train")[:4]
        self.assertEqual(len(set(expected)), 4)
        self.assertEqual([case.case_id for case in cases], expected)

    def test_first_response_and_eos_are_labeled_even_when_pad_is_eos(self):
        examples = [encode_example(ToyTokenizer(), prompt, answer)
                    for prompt, answer in (("Edit.", '{"a":1}'), ("Longer edit.", '{"a":-22}'))]
        batch = collate_examples(examples, pad_token_id=0)
        for index, example in enumerate(examples):
            boundary = example.answer_start
            size = len(example.input_ids)
            labels = batch["labels"][index]
            self.assertEqual(labels[:boundary], [IGNORE_INDEX] * boundary)
            self.assertEqual(labels[boundary], ord("{"))
            self.assertEqual(labels[boundary:size], example.input_ids[boundary:])
            self.assertEqual(labels[size - 1], 0)
            self.assertTrue(all(label == IGNORE_INDEX for label in labels[size:]))
            self.assertEqual(batch["attention_mask"][index], [1] * size + [0] * (len(labels) - size))
        self.assertEqual(examples[0].answer_start, 5)
        self.assertGreater(len(examples[1].input_ids), len(examples[0].input_ids))
        self.assertEqual(batch["input_ids"][0][-1], 0)
        self.assertEqual(batch["labels"][0][-1], IGNORE_INDEX)

    def test_rejects_truncation_even_if_only_eos_would_be_lost(self):
        tokenizer = ToyTokenizer()
        prompt, answer = "Edit.", '{"a":1}'
        length = len(prompt + answer)
        with self.assertRaisesRegex(ValueError, "target would be truncated"):
            encode_example(tokenizer, prompt, answer, max_length=length)
        self.assertEqual(len(encode_example(tokenizer, prompt, answer, max_length=length + 1).labels), length + 1)

    @unittest.skipUnless(HAS_MODEL_DEPS, "CPU model test requires torch and transformers")
    def test_qwen_loss_shifts_once_across_first_response_token(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM

        examples = [encode_example(ToyTokenizer(), prompt, answer)
                    for prompt, answer in (("Edit.", '{"a":1}'), ("A longer edit.", '{"b":-22}'))]
        batch = {key: torch.tensor(value) for key, value in collate_examples(examples, 0).items()}
        config = Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                             head_dim=8, pad_token_id=0, eos_token_id=0)
        torch.manual_seed(42)
        model = Qwen3ForCausalLM(config).eval()
        with torch.no_grad():
            output = model(**batch, use_cache=False)
            # Each target, including the first token and EOS, is scored by
            # logits at the preceding position in the unshifted batch.
            logits = torch.cat([output.logits[i, example.answer_start - 1:len(example.input_ids) - 1]
                                for i, example in enumerate(examples)])
            targets = torch.tensor([token for example in examples
                                    for token in example.input_ids[example.answer_start:]])
            expected = torch.nn.functional.cross_entropy(logits, targets)
        torch.testing.assert_close(output.loss, expected)

    @unittest.skipUnless(os.getenv("QWEN_TEST_TOKENIZER"), "set QWEN_TEST_TOKENIZER to the pinned tokenizer directory")
    def test_pinned_tokenizer_boundaries_on_selected_examples(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(os.environ["QWEN_TEST_TOKENIZER"], local_files_only=True)
        self.assertEqual(tokenizer.pad_token_id, tokenizer.eos_token_id)
        _, cases = select_tiny_cases(DEFAULT_MANIFEST)
        lengths = set()
        for case in cases:
            prompt = render_prompt(case.task)
            target = json.dumps(evaluate_revision(case.task).new, sort_keys=True, separators=(",", ":"))
            example = encode_example(tokenizer, prompt, target)
            boundary = example.answer_start
            self.assertEqual(example.input_ids[:boundary], tokenizer.encode(prompt, add_special_tokens=False))
            self.assertEqual(example.labels[boundary], tokenizer.encode(target, add_special_tokens=False)[0])
            self.assertEqual(tokenizer.decode(example.input_ids[:-1], clean_up_tokenization_spaces=False), prompt + target)
            self.assertEqual(example.labels[-1], tokenizer.eos_token_id)
            lengths.add(len(example.input_ids))
        self.assertGreater(len(lengths), 1)


class BaselineTests(unittest.TestCase):
    @unittest.skipUnless(HAS_MODEL_DEPS, "CPU generation test requires torch and transformers")
    def test_greedy_generation_decodes_only_raw_new_tokens(self):
        import torch

        class FakeModel:
            def generate(self, *, input_ids, attention_mask, generation_config):
                self.config = generation_config
                return torch.cat((input_ids, torch.tensor([[10, 120, 32, 0]])), dim=1)

        model = FakeModel()
        answer = baseline_qwen.greedy_answer(model, ToyTokenizer(), "prompt", "cpu")
        self.assertEqual(answer, "\nx ")
        self.assertFalse(model.config.do_sample)
        self.assertEqual(model.config.num_beams, 1)
        self.assertEqual(model.config.max_new_tokens, baseline_qwen.MAX_NEW_TOKENS)

    def test_baseline_emits_all_development_ids_and_preserves_malformed_answers(self):
        _, cases = load_development(DEFAULT_MANIFEST)
        raw = "\n```json\n{}\n```\n"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            with (patch("sys.argv", ["baseline_qwen.py", "--output", str(path)]),
                  patch("baseline_qwen.load_qwen", return_value=(None, None)),
                  patch("baseline_qwen.greedy_answer", return_value=raw) as greedy,
                  patch("builtins.print")):
                baseline_qwen.main()
            predictions = list(read_predictions(path))
        self.assertEqual(predictions, [(case.case_id, raw) for case in cases])
        self.assertEqual([call.args[2] for call in greedy.call_args_list],
                         [render_prompt(case.task) for case in cases])
        metrics = evaluate(cases, predictions)["metrics"]
        self.assertEqual(metrics["malformed_answers"]["numerator"], 20)
        self.assertEqual(metrics["missing_predictions"]["numerator"], 0)


if __name__ == "__main__":
    unittest.main()
