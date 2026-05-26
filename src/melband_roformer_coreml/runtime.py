from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import coremltools as ct
import torch
import yaml
from ml_collections import ConfigDict
from torch import nn


def parse_compute_units(name: str) -> ct.ComputeUnit:
    return getattr(ct.ComputeUnit, name)


def load_config(path: Path) -> ConfigDict:
    with path.open("r", encoding="utf-8") as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    config.model.flash_attn = False
    return config


def compute_frames(config: ConfigDict) -> int:
    chunk_size = int(config.inference.chunk_size)
    n_fft = int(config.model.stft_n_fft)
    hop_length = int(config.model.stft_hop_length)
    return (chunk_size + 2 * (n_fft // 2) - n_fft) // hop_length + 1


def load_model(repo_dir: Path, config: ConfigDict, checkpoint_path: Path) -> nn.Module:
    if not repo_dir.exists():
        raise FileNotFoundError(f"Missing source model repository: {repo_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    repo_dir_str = str(repo_dir)
    if repo_dir_str not in sys.path:
        sys.path.insert(0, repo_dir_str)

    from utils import get_model_from_config  # pylint: disable=import-outside-toplevel

    model = get_model_from_config("mel_band_roformer", config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict: dict[str, Any]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(checkpoint)!r}")

    stripped_state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(stripped_state_dict, strict=False)
    if missing or unexpected:
        print(f"load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("missing keys:", missing[:10])
        if unexpected:
            print("unexpected keys:", unexpected[:10])

    model.eval()
    model.cpu()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
