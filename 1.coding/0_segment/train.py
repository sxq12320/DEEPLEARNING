"""One-click training entry for the segmentation model."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine.trainer import Trainer


DEFAULT_CFG = {
	"image_dir": "",
	"mask_dir": "",
	"label_dir": "",
	"label_type": "mask",
	"imgsz": 128,
	"epochs": 20,
	"batch": 8,
	"lr": 1e-3,
	"workers": 0,
	"synthetic_length": 32,
	"augment": True,
	"cpu": False,
	"project": "0_segment/checkpoints/results",
	"name": "test_1",
	"seed": 22,
}


def parse_args(argv=None):
	"""Parse CLI arguments for training."""
	parser = argparse.ArgumentParser(description="Segmentation training (one-click)")
	parser.add_argument("--cfg", type=str, default="", help="Path to a JSON config file")
	parser.add_argument("--save-cfg", type=str, default="", help="Save merged config to JSON")
	parser.add_argument("--print-cfg", action="store_true", help="Print merged config")

	parser.add_argument("--image-dir", type=str, default=None, help="Training image directory")
	parser.add_argument("--mask-dir", type=str, default=None, help="Training mask directory")
	parser.add_argument("--label-dir", type=str, default=None, help="Training label directory (alias)")
	parser.add_argument("--label-type", type=str, default=None, choices=["mask", "txt", "json", "npy"])
	parser.add_argument("--imgsz", type=int, default=None, help="Image size (square)")
	parser.add_argument("--epochs", type=int, default=None)
	parser.add_argument("--batch", type=int, default=None)
	parser.add_argument("--lr", type=float, default=None)
	parser.add_argument("--workers", type=int, default=None)
	parser.add_argument("--synthetic-length", type=int, default=None)
	parser.add_argument("--augment", dest="augment", action="store_true", help="Enable augmentation")
	parser.add_argument("--no-augment", dest="augment", action="store_false", help="Disable augmentation")
	parser.set_defaults(augment=None)
	parser.add_argument("--cpu", action="store_true", default=None, help="Force CPU mode")
	parser.add_argument("--project", type=str, default=None)
	parser.add_argument("--name", type=str, default=None)
	parser.add_argument("--seed", type=int, default=None)

	return parser.parse_args(argv)


def load_cfg(path):
	"""Load a JSON config file."""
	if not path:
		return {}
	cfg_path = Path(path)
	if not cfg_path.is_file():
		raise FileNotFoundError(f"Config file not found: {cfg_path}")
	with cfg_path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, dict):
		raise ValueError("Config file must be a JSON object")
	return data


def merge_cfg(cli_args):
	"""Merge defaults, file config, and CLI overrides."""
	cfg = DEFAULT_CFG.copy()
	file_cfg = load_cfg(cli_args.cfg)

	unknown = [k for k in file_cfg.keys() if k not in cfg]
	if unknown:
		print(f"Warning: unknown keys in config file will be ignored: {unknown}")

	for k, v in file_cfg.items():
		if k in cfg:
			cfg[k] = v

	for k in cfg.keys():
		v = getattr(cli_args, k, None)
		if v is not None:
			cfg[k] = v

	return cfg


def save_cfg(cfg, path):
	"""Save merged config to JSON."""
	out_path = Path(path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w", encoding="utf-8") as f:
		json.dump(cfg, f, indent=2, ensure_ascii=True)


def set_seed(seed):
	"""Set random seed for reproducibility."""
	if seed is None:
		return
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def main(argv=None):
	cli_args = parse_args(argv)
	cfg = merge_cfg(cli_args)

	if cli_args.print_cfg:
		print(json.dumps(cfg, indent=2, ensure_ascii=True))

	if cli_args.save_cfg:
		save_cfg(cfg, cli_args.save_cfg)

	set_seed(cfg.get("seed"))
	args = argparse.Namespace(**cfg)
	trainer = Trainer(args)
	trainer.train()


if __name__ == "__main__":
	main()


