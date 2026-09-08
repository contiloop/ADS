from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config_loader import compose_config
from train_stage import _eval_cmd, parse_args


def make_args(target: str, *variables: str) -> list[str]:
    result = subprocess.run(
        ["make", "-n", target, *variables], cwd=ROOT,
        text=True, capture_output=True, check=True,
    )
    tokens = shlex.split(result.stdout.replace("\\\n", " "))
    script = "src/train_stage.py" if target == "train-stage" else "src/train.py"
    return tokens[tokens.index(script) + 1:]


def stage_config(*variables: str):
    with patch.object(sys, "argv", ["train_stage.py", *make_args("train-stage", *variables)]):
        args = parse_args()
    cfg = compose_config(ROOT / args.config, overrides=args.override)
    cmd = _eval_cmd(cfg=cfg, args=args, subset_idx=0)
    overrides = [cmd[i + 1] for i, value in enumerate(cmd) if value == "--override"]
    eval_cfg = compose_config(ROOT / args.eval_config, overrides=overrides)
    return args, cfg, eval_cfg


class MakeTrainStageDefaultsTest(TestCase):
    def test_bare_target_selects_gemma_full_and_four_gpus(self):
        args, cfg, evaluation = stage_config()
        self.assertEqual(cfg["model"]["name_or_path"], "google/gemma-4-E2B-it")
        self.assertEqual(cfg["training"]["tuning_mode"], "full")
        self.assertEqual(args.sft_nproc_per_node, 4)
        self.assertEqual(cfg["inference"]["num_gpus"], 4)
        self.assertEqual(cfg["qe"]["selection"]["num_gpus"], 4)
        self.assertEqual(evaluation["eval"]["generation"]["num_gpus"], 4)
        self.assertEqual(evaluation["model"], cfg["model"])
        self.assertIn("gemma4", cfg["data"]["prepared_download"]["local_dir"])

    def test_partial_override_retains_stage_defaults(self):
        args, cfg, evaluation = stage_config("TRAIN_OVERRIDES=logging.wandb.enabled=false")
        self.assertFalse(cfg["logging"]["wandb"]["enabled"])
        self.assertEqual(cfg["model"]["family"], "gemma4")
        self.assertEqual(cfg["training"]["tuning_mode"], "full")
        self.assertEqual(args.sft_nproc_per_node, 4)
        self.assertEqual(evaluation["eval"]["generation"]["num_gpus"], 4)

    def test_qwen_full_example_selects_matching_model_and_data(self):
        args, cfg, evaluation = stage_config(
            "SFT_NPROC_PER_NODE=4",
            "TRAIN_OVERRIDES=model=qwen35_4b_it training=full inference.num_gpus=4 qe.selection.num_gpus=4",
            "EVAL_OVERRIDES=eval.generation.num_gpus=4",
        )
        self.assertEqual(cfg["model"]["family"], "qwen3.5")
        self.assertEqual(cfg["training"]["tuning_mode"], "full")
        self.assertEqual(args.sft_nproc_per_node, 4)
        self.assertIn("qwen35", cfg["data"]["prepared_download"]["local_dir"])
        self.assertEqual(evaluation["model"], cfg["model"])
        self.assertEqual(evaluation["eval"]["generation"]["num_gpus"], 4)

    def test_explicit_model_tuning_and_gpu_settings_win(self):
        args, cfg, evaluation = stage_config(
            "SFT_NPROC_PER_NODE=2",
            "TRAIN_OVERRIDES=model=qwen35_4b_it training=lora inference.num_gpus=2 qe.selection.num_gpus=1 eval.generation.num_gpus=2",
            "EVAL_OVERRIDES=eval.generation.num_gpus=1",
        )
        self.assertEqual(cfg["model"]["family"], "qwen3.5")
        self.assertEqual(cfg["training"]["tuning_mode"], "lora")
        self.assertEqual(args.sft_nproc_per_node, 2)
        self.assertEqual(cfg["inference"]["num_gpus"], 2)
        self.assertEqual(cfg["qe"]["selection"]["num_gpus"], 1)
        self.assertEqual(cfg["eval"]["generation"]["num_gpus"], 2)
        self.assertEqual(evaluation["eval"]["generation"]["num_gpus"], 1)

    def test_train_target_keeps_existing_defaults(self):
        args = make_args("train")
        self.assertEqual(args[args.index("--sft-nproc-per-node") + 1], "1")
        self.assertNotIn("--override", args)
