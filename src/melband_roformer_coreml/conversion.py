from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
from ml_collections import ConfigDict
from torch import nn

from .paths import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_COREML_DIR,
    DEFAULT_EXTERNAL_REPO_DIR,
    DEFAULT_LOG_DIR,
    MASK_CORE_METADATA_NAME,
    MASK_CORE_MODEL_NAME,
    WAVEFORM_METADATA_NAME,
    WAVEFORM_MODEL_NAME,
)
from .runtime import compute_frames, load_config, load_model
from .wrappers import MaskCoreWrapper, FixedWaveformToWaveformWrapper, replace_rotary_embeddings


def convert_to_coreml(
    traced: torch.jit.ScriptModule,
    inputs: list[ct.TensorType],
    output_names: list[str],
    output_path: Path,
    compute_precision: Any = ct.precision.FLOAT16,
    use_sliced_sdpa: bool = False,
) -> None:
    target = getattr(ct.target, "iOS26", getattr(ct.target, "iOS18"))
    pass_pipeline = None
    if use_sliced_sdpa:
        pass_pipeline = ct.PassPipeline.DEFAULT
        pass_pipeline.append_pass("common::scaled_dot_product_attention_sliced_q")
        pass_pipeline.set_options(
            "common::scaled_dot_product_attention_sliced_q",
            {"min_seq_length": 128, "seq_length_divider": 32},
        )
    mlmodel = ct.convert(
        traced,
        source="pytorch",
        inputs=inputs,
        outputs=[ct.TensorType(name=name) for name in output_names],
        minimum_deployment_target=target,
        convert_to="mlprogram",
        compute_precision=compute_precision,
        pass_pipeline=pass_pipeline,
    )
    mlmodel.save(str(output_path))


def write_error(path: Path, exc: BaseException) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), encoding="utf-8")


def write_mask_core_metadata(path: Path, config: ConfigDict, model: nn.Module, example_width: int) -> None:
    metadata = {
        "sample_rate": int(config.model.sample_rate),
        "chunk_size": int(config.inference.chunk_size),
        "num_overlap": int(config.inference.num_overlap),
        "audio_channels": int(model.audio_channels),
        "stft": {
            "n_fft": int(config.model.stft_n_fft),
            "hop_length": int(config.model.stft_hop_length),
            "win_length": int(config.model.stft_win_length),
            "normalized": bool(config.model.stft_normalized),
            "center": True,
        },
        "mask_core": {
            "input_name": "packed_stft",
            "output_name": "packed_masks",
            "input_shape": [1, 801, int(example_width)],
            "output_shape": [1, 801, int(example_width)],
            "packing": "STFT -> view_as_real -> stereo folded into frequency -> index freq_indices -> fold complex into last dimension.",
            "freq_indices": model.freq_indices.cpu().numpy().astype(np.int64).tolist(),
            "num_bands_per_freq": model.num_bands_per_freq.cpu().numpy().astype(np.int64).tolist(),
            "num_freqs_per_band": model.num_freqs_per_band.cpu().numpy().astype(np.int64).tolist(),
        },
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_waveform_metadata(path: Path, config: ConfigDict, model: nn.Module) -> None:
    metadata = {
        "sample_rate": int(config.model.sample_rate),
        "chunk_size": int(config.inference.chunk_size),
        "num_overlap": int(config.inference.num_overlap),
        "input_name": "audio",
        "output_name": "vocals",
        "input_shape": [1, int(model.audio_channels), int(config.inference.chunk_size)],
        "output_shape": [1, int(model.audio_channels), int(config.inference.chunk_size)],
        "stft": {
            "n_fft": int(config.model.stft_n_fft),
            "hop_length": int(config.model.stft_hop_length),
            "win_length": int(config.model.stft_win_length),
            "normalized": bool(config.model.stft_normalized),
            "center": True,
            "pad_mode": "reflect",
        },
        "coreml": {
            "format": "mlprogram",
            "minimum_deployment_target": "latest available in coremltools target enum (iOS26 with coremltools 9.0)",
            "compute_precision": "FLOAT32",
            "attention": "fused scaled_dot_product_attention",
        },
        "coreml_boundary": "waveform-to-waveform fixed 8 second stereo chunk",
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def trace_waveform(model: nn.Module, config: ConfigDict, seed: int) -> tuple[torch.jit.ScriptModule, torch.Tensor]:
    torch.manual_seed(seed)
    chunk_size = int(config.inference.chunk_size)
    example = torch.randn(1, model.audio_channels, chunk_size)
    wrapper = FixedWaveformToWaveformWrapper(model, config, frames=compute_frames(config)).eval()
    with torch.no_grad():
        _ = wrapper(example)
        traced = torch.jit.trace(wrapper, example, strict=False, check_trace=False)
    return traced, example


def trace_mask_core(model: nn.Module, config: ConfigDict, seed: int) -> tuple[torch.jit.ScriptModule, torch.Tensor]:
    torch.manual_seed(seed)
    frames = compute_frames(config)
    input_width = int(sum(model.band_split.dim_inputs))
    example = torch.randn(1, frames, input_width)
    wrapper = MaskCoreWrapper(model, frames=frames).eval()
    with torch.no_grad():
        _ = wrapper(example)
        traced = torch.jit.trace(wrapper, example, strict=False, check_trace=False)
    return traced, example


def run_convert(args: argparse.Namespace) -> None:
    repo_dir = Path(args.repo_dir).resolve()
    config_path = Path(args.config_path).resolve()
    checkpoint_path = Path(args.checkpoint_path).resolve()
    coreml_dir = Path(args.coreml_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    coreml_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    torch.set_grad_enabled(False)

    config = load_config(config_path)
    model = load_model(repo_dir, config, checkpoint_path)
    frames = compute_frames(config)
    replace_rotary_embeddings(model, frames)

    full_output = coreml_dir / WAVEFORM_MODEL_NAME
    mask_output = coreml_dir / MASK_CORE_MODEL_NAME
    full_metadata_output = coreml_dir / WAVEFORM_METADATA_NAME
    metadata_output = coreml_dir / MASK_CORE_METADATA_NAME

    full_succeeded = False
    if not args.skip_full:
        print("Tracing waveform-to-waveform model...")
        try:
            traced_full, audio_example = trace_waveform(model, config, args.seed)
            print("Converting waveform-to-waveform model to CoreML...")
            convert_to_coreml(
                traced_full,
                [ct.TensorType(name="audio", shape=audio_example.shape)],
                ["vocals"],
                full_output,
                compute_precision=ct.precision.FLOAT32,
                use_sliced_sdpa=args.slice_sdpa,
            )
            write_waveform_metadata(full_metadata_output, config, model)
            print(f"wrote={full_output}")
            print(f"wrote={full_metadata_output}")
            full_succeeded = True
        except Exception as exc:  # noqa: BLE001
            write_error(log_dir / "full_conversion_error.txt", exc)
            print(f"Full conversion failed. See {log_dir / 'full_conversion_error.txt'}")

    if args.force_maskcore or not full_succeeded:
        print("Tracing mask-core fallback model...")
        traced_mask, packed_example = trace_mask_core(model, config, args.seed)
        print("Converting mask-core fallback model to CoreML...")
        convert_to_coreml(
            traced_mask,
            [ct.TensorType(name="packed_stft", shape=packed_example.shape)],
            ["packed_masks"],
            mask_output,
        )
        write_mask_core_metadata(metadata_output, config, model, packed_example.shape[-1])
        print(f"wrote={mask_output}")
        print(f"wrote={metadata_output}")
