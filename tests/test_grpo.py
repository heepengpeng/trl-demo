"""Regression tests for the GRPO training entrypoint."""

import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


class GrpoEntrypointTest(unittest.TestCase):
    def test_import_does_not_start_training(self):
        os.environ.pop("HF_ENDPOINT", None)
        sys.modules.pop("grpo", None)

        module = importlib.import_module("grpo")

        self.assertTrue(hasattr(module, "main"))

    def test_uses_local_gsm8k_dataset_and_qwen_model(self):
        sys.modules.pop("grpo", None)
        module = importlib.import_module("grpo")
        expected_model_path = str(
            Path(module.__file__).resolve().parent
            / "models"
            / "Qwen"
            / "Qwen2___5-0___5B-Instruct"
        )

        self.assertEqual(module.DATASET_NAME, "openai/gsm8k")
        self.assertEqual(
            module.DATASET_PATH,
            str(Path(module.__file__).resolve().parent / "datasets" / "gsm8k" / "train"),
        )
        self.assertEqual(module.MODEL_PATH, expected_model_path)

    def test_load_train_dataset_uses_local_gsm8k_path(self):
        sys.modules.pop("grpo", None)
        module = importlib.import_module("grpo")
        raw_dataset = mock.Mock()
        formatted_dataset = mock.Mock()
        raw_dataset.map.return_value = formatted_dataset

        with mock.patch.object(module, "load_from_disk", return_value=raw_dataset) as loader:
            dataset = module.load_train_dataset()

        self.assertIs(dataset, formatted_dataset)
        loader.assert_called_once_with(module.DATASET_PATH)
        raw_dataset.map.assert_called_once_with(module.format_gsm8k_example)

    def test_formats_gsm8k_example_for_grpo_rewards(self):
        sys.modules.pop("grpo", None)
        module = importlib.import_module("grpo")
        example = {
            "question": "Natalia sold clips to 48 friends in April.",
            "answer": "48 / 2 = 24\n#### 24",
        }

        formatted = module.format_gsm8k_example(example)

        self.assertEqual(
            formatted,
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "Natalia sold clips to 48 friends in April.",
                    }
                ],
                "solution": "48 / 2 = 24\n#### 24",
            },
        )

    def test_main_trains_with_qwen_accuracy_reward_and_gsm8k(self):
        sys.modules.pop("grpo", None)
        module = importlib.import_module("grpo")
        fake_dataset = object()
        trainer = mock.Mock()
        training_args = object()

        with mock.patch.object(module, "load_train_dataset", return_value=fake_dataset):
            with mock.patch.object(module, "build_training_args", return_value=training_args):
                with mock.patch.object(module, "GRPOTrainer", return_value=trainer) as trainer_cls:
                    module.main()

        trainer_cls.assert_called_once_with(
            model=module.MODEL_PATH,
            args=training_args,
            reward_funcs=[module.gsm8k_numeric_reward, module.accuracy_reward],
            train_dataset=fake_dataset,
        )
        trainer.train.assert_called_once_with()

    def test_build_training_args_uses_fast_defaults(self):
        sys.modules.pop("grpo", None)
        module = importlib.import_module("grpo")
        training_args = object()

        with mock.patch.object(module, "GRPOConfig", return_value=training_args) as config_cls:
            result = module.build_training_args()

        self.assertIs(result, training_args)
        config_cls.assert_called_once_with(
            output_dir="grpo-checkpoints",
            max_steps=1000,
            per_device_train_batch_size=8,
            num_generations=8,
            max_completion_length=256,
            bf16=True,
        )

    def test_gsm8k_numeric_reward_matches_last_number(self):
        sys.modules.pop("grpo", None)
        module = importlib.import_module("grpo")

        rewards = module.gsm8k_numeric_reward(
            completions=[
                "Natalia sold 72 clips altogether.",
                [{"role": "assistant", "content": "The answer is 71."}],
                "I cannot solve it.",
            ],
            solution=[
                "Natalia sold 48+24 = 72 clips.\n#### 72",
                "Natalia sold 48+24 = 72 clips.\n#### 72",
                "Natalia sold 48+24 = 72 clips.\n#### 72",
            ],
        )

        self.assertEqual(rewards, [1.0, 0.0, 0.0])

    def test_accuracy_reward_accepts_string_completions(self):
        sys.modules.pop("grpo", None)
        module = importlib.import_module("grpo")

        with mock.patch.object(module, "trl_accuracy_reward", return_value=[1.0]) as reward:
            result = module.accuracy_reward(
                completions=["The answer is \\boxed{72}."],
                solution=["#### 72"],
            )

        self.assertEqual(result, [1.0])
        reward.assert_called_once_with(
            completions=[
                [{"role": "assistant", "content": "The answer is \\boxed{72}."}]
            ],
            solution=["#### 72"],
        )


class DownloadDatasetTest(unittest.TestCase):
    def test_downloads_raw_gsm8k_train_split_to_project_dataset_path(self):
        sys.modules.pop("download_dataset", None)
        module = importlib.import_module("download_dataset")
        dataset = mock.Mock()

        with mock.patch.object(module, "load_dataset", return_value=dataset) as loader:
            module.main()

        loader.assert_called_once_with("openai/gsm8k", "main", split="train")
        dataset.map.assert_not_called()
        dataset.save_to_disk.assert_called_once_with(str(module.DATASET_PATH))


class GrpoOnePointFiveBEntrypointTest(unittest.TestCase):
    def test_uses_local_one_point_five_b_model(self):
        sys.modules.pop("grpo_1_5b", None)
        module = importlib.import_module("grpo_1_5b")
        expected_model_path = str(
            Path(module.__file__).resolve().parent
            / "models"
            / "Qwen"
            / "Qwen2___5-1___5B-Instruct"
        )

        self.assertEqual(module.MODEL_PATH, expected_model_path)

    def test_build_training_args_matches_one_point_five_b_plan(self):
        sys.modules.pop("grpo_1_5b", None)
        module = importlib.import_module("grpo_1_5b")
        training_args = object()

        with mock.patch.object(module, "GRPOConfig", return_value=training_args) as config_cls:
            result = module.build_training_args()

        self.assertIs(result, training_args)
        config_cls.assert_called_once_with(
            output_dir="grpo-gsm8k",
            per_device_train_batch_size=4,
            num_generations=8,
            max_completion_length=512,
            logging_steps=10,
            num_train_epochs=1,
        )

    def test_main_uses_accuracy_reward_only(self):
        sys.modules.pop("grpo_1_5b", None)
        module = importlib.import_module("grpo_1_5b")
        fake_dataset = object()
        trainer = mock.Mock()
        training_args = object()

        with mock.patch.object(module, "load_train_dataset", return_value=fake_dataset):
            with mock.patch.object(module, "build_training_args", return_value=training_args):
                with mock.patch.object(module, "GRPOTrainer", return_value=trainer) as trainer_cls:
                    module.main()

        trainer_cls.assert_called_once_with(
            model=module.MODEL_PATH,
            reward_funcs=module.accuracy_reward,
            train_dataset=fake_dataset,
            args=training_args,
        )
        trainer.train.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
