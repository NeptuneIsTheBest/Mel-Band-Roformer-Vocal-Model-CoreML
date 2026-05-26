#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import psutil
import torch
import yaml
from ml_collections import ConfigDict

from convert_melbandroformer_coreml import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_REPO_DIR,
    FixedWaveformToWaveformWrapper,
    MaskCoreWrapper,
    compute_frames,
    load_model,
    replace_rotary_embeddings,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Mel-Band-RoFormer CoreML outputs against PyTorch.")
    parser.add_argument("--repo-dir", default=str(DEFAULT_REPO_DIR))
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--mode", choices=["full", "maskcore"], default="maskcore")
    parser.add_argument("--result-path", default="")
    parser.add_argument(
        "--compute-units",
        choices=["ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY"],
        default="CPU_ONLY",
        help="Core ML compute units to use when loading the mlpackage.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


class PeakMemoryMonitor:
    def __init__(self, interval: float = 0.05) -> None:
        self.process = psutil.Process()
        self.interval = interval
        self.start_rss_mb = self._rss_mb()
        self.peak_rss_mb = self.start_rss_mb
        self.end_rss_mb = self.start_rss_mb
        self._running = False
        self._thread: threading.Thread | None = None

    def _rss_mb(self) -> float:
        return self.process.memory_info().rss / 1024 / 1024

    def _sample(self) -> None:
        while self._running:
            self.peak_rss_mb = max(self.peak_rss_mb, self._rss_mb())
            time.sleep(self.interval)

    def __enter__(self) -> "PeakMemoryMonitor":
        self.start_rss_mb = self._rss_mb()
        self.peak_rss_mb = self.start_rss_mb
        self._running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.end_rss_mb = self._rss_mb()
        self.peak_rss_mb = max(self.peak_rss_mb, self.end_rss_mb)

    def result(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_rss_start_mb": self.start_rss_mb,
            f"{prefix}_rss_peak_mb": self.peak_rss_mb,
            f"{prefix}_rss_end_mb": self.end_rss_mb,
        }


def parse_compute_units(name: str) -> ct.ComputeUnit:
    return getattr(ct.ComputeUnit, name)


def load_config(path: Path) -> ConfigDict:
    with path.open("r", encoding="utf-8") as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    config.model.flash_attn = False
    return config


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

    metadata_path = Path(args.output_dir).resolve() / "MelBandRoformerVocal_iOS18_maskcore_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shape = metadata["mask_core"]["input_shape"]

    torch.manual_seed(args.seed)
    example = torch.randn(*shape)
    with torch.no_grad():
        torch_output = wrapper(example).cpu().numpy()

    mlmodel_path = Path(args.output_dir).resolve() / "MelBandRoformerVocal_iOS18_maskcore.mlpackage"
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

    mlmodel_path = Path(args.output_dir).resolve() / "MelBandRoformerVocal_macOS_waveform.mlpackage"
    with PeakMemoryMonitor() as monitor:
        mlmodel = ct.models.MLModel(str(mlmodel_path), compute_units=parse_compute_units(args.compute_units))
        prediction = mlmodel.predict({"audio": example.cpu().numpy().astype(np.float32)})
    coreml_output = prediction["vocals"]
    result = summarize_error(torch_output, coreml_output)
    result["compute_units"] = args.compute_units
    result.update(monitor.result("coreml"))
    return result


def main() -> None:
    args = parse_args()
    if args.mode == "full":
        result = verify_full(args)
    else:
        result = verify_maskcore(args)
    if args.result_path:
        result_path = Path(args.result_path).resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.exit(main())
