from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch

from .memory import PeakMemoryMonitor
from .paths import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_COREML_DIR,
    DEFAULT_EXTERNAL_REPO_DIR,
    MASK_CORE_METADATA_NAME,
    MASK_CORE_MODEL_NAME,
    WAVEFORM_MODEL_NAME,
)
from .runtime import compute_frames, load_config, load_model, parse_compute_units
from .wrappers import MaskCoreWrapper, FixedWaveformToWaveformWrapper, replace_rotary_embeddings


def summarize_error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    diff = np.asarray(reference) - np.asarray(candidate)
    return {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "max_abs_err": float(np.max(np.abs(diff))),
        "mean_abs_err": float(np.mean(np.abs(diff))),
    }


def verify_maskcore(args: argparse.Namespace) -> dict[str, Any]:
    repo_dir = Path(args.repo_dir).resolve()
    config = load_config(Path(args.config_path).resolve())
    model = load_model(repo_dir, config, Path(args.checkpoint_path).resolve())
    frames = compute_frames(config)
    replace_rotary_embeddings(model, frames)
    wrapper = MaskCoreWrapper(model, frames=frames).eval()

    coreml_dir = Path(args.coreml_dir).resolve()
    metadata_path = coreml_dir / MASK_CORE_METADATA_NAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shape = metadata["mask_core"]["input_shape"]

    torch.manual_seed(args.seed)
    example = torch.randn(*shape)
    with torch.no_grad():
        torch_output = wrapper(example).cpu().numpy()

    mlmodel_path = coreml_dir / MASK_CORE_MODEL_NAME
    with PeakMemoryMonitor() as monitor:
        mlmodel = ct.models.MLModel(str(mlmodel_path), compute_units=parse_compute_units(args.compute_units))
        prediction = mlmodel.predict({"packed_stft": example.cpu().numpy().astype(np.float32)})
    coreml_output = prediction["packed_masks"]
    result = summarize_error(torch_output, coreml_output)
    result["compute_units"] = args.compute_units
    result.update(monitor.result("coreml"))
    return result


def verify_full(args: argparse.Namespace) -> dict[str, Any]:
    repo_dir = Path(args.repo_dir).resolve()
    config = load_config(Path(args.config_path).resolve())
    model = load_model(repo_dir, config, Path(args.checkpoint_path).resolve()).eval()
    frames = compute_frames(config)
    replace_rotary_embeddings(model, frames)
    wrapper = FixedWaveformToWaveformWrapper(model, config, frames=frames).eval()

    torch.manual_seed(args.seed)
    example = torch.randn(1, model.audio_channels, int(config.inference.chunk_size))
    with torch.no_grad():
        torch_output = wrapper(example).cpu().numpy()

    mlmodel_path = Path(args.coreml_dir).resolve() / WAVEFORM_MODEL_NAME
    with PeakMemoryMonitor() as monitor:
        mlmodel = ct.models.MLModel(str(mlmodel_path), compute_units=parse_compute_units(args.compute_units))
        prediction = mlmodel.predict({"audio": example.cpu().numpy().astype(np.float32)})
    coreml_output = prediction["vocals"]
    result = summarize_error(torch_output, coreml_output)
    result["compute_units"] = args.compute_units
    result.update(monitor.result("coreml"))
    return result


def run_verify(args: argparse.Namespace) -> None:
    if args.mode == "full":
        result = verify_full(args)
    else:
        result = verify_maskcore(args)
    if args.result_path:
        result_path = Path(args.result_path).resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
