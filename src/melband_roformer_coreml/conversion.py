from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

import coremltools as ct
import torch
from ml_collections import ConfigDict
from torch import nn

from .paths import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_COREML_DIR,
    DEFAULT_EXTERNAL_REPO_DIR,
    DEFAULT_LOG_DIR,
    WAVEFORM_METADATA_NAME,
    WAVEFORM_MODEL_NAME,
)
from .runtime import compute_frames, load_config, load_model
from .wrappers import FixedWaveformToWaveformWrapper, replace_rotary_embeddings


COREML_TARGET = getattr(ct.target, "iOS26", getattr(ct.target, "iOS18"))
DEFAULT_SLICE_SDPA = True
DEFAULT_SDPA_MIN_SEQ_LENGTH = 128
DEFAULT_SDPA_SEQ_LENGTH_DIVIDER = 32

COMPUTE_PRECISIONS: dict[str, Any] = {
    "FLOAT16": ct.precision.FLOAT16,
    "FLOAT32": ct.precision.FLOAT32,
}


def parse_compute_precision(name: str) -> Any:
    try:
        return COMPUTE_PRECISIONS[name]
    except KeyError as exc:
        choices = ", ".join(COMPUTE_PRECISIONS)
        raise ValueError(f"Unsupported compute precision {name!r}; expected one of: {choices}") from exc


def convert_to_coreml(
    traced: torch.jit.ScriptModule,
    inputs: list[ct.TensorType],
    output_names: list[str],
    output_path: Path,
    compute_precision: Any = ct.precision.FLOAT16,
    use_sliced_sdpa: bool = DEFAULT_SLICE_SDPA,
    sdpa_min_seq_length: int = DEFAULT_SDPA_MIN_SEQ_LENGTH,
    sdpa_seq_length_divider: int = DEFAULT_SDPA_SEQ_LENGTH_DIVIDER,
) -> None:
    pass_pipeline = None
    if use_sliced_sdpa:
        pass_pipeline = ct.PassPipeline.DEFAULT
        pass_pipeline.append_pass("common::scaled_dot_product_attention_sliced_q")
        pass_pipeline.set_options(
            "common::scaled_dot_product_attention_sliced_q",
            {
                "min_seq_length": sdpa_min_seq_length,
                "seq_length_divider": sdpa_seq_length_divider,
            },
        )
    mlmodel = ct.convert(
        traced,
        source="pytorch",
        inputs=inputs,
        outputs=[ct.TensorType(name=name) for name in output_names],
        minimum_deployment_target=COREML_TARGET,
        convert_to="mlprogram",
        compute_precision=compute_precision,
        pass_pipeline=pass_pipeline,
    )
    mlmodel.save(str(output_path))


def write_error(path: Path, exc: BaseException) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), encoding="utf-8")


def write_waveform_metadata(
    path: Path,
    config: ConfigDict,
    model: nn.Module,
    *,
    compute_precision_name: str,
    use_sliced_sdpa: bool,
    sdpa_min_seq_length: int,
    sdpa_seq_length_divider: int,
) -> None:
    attention = (
        "sliced scaled_dot_product_attention over Q"
        if use_sliced_sdpa
        else "fused scaled_dot_product_attention"
    )
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
            "minimum_deployment_target": COREML_TARGET.name,
            "compute_precision": compute_precision_name,
            "attention": attention,
            "sliced_sdpa": {
                "enabled": use_sliced_sdpa,
                "min_seq_length": sdpa_min_seq_length,
                "seq_length_divider": sdpa_seq_length_divider,
            },
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


def run_convert(args: argparse.Namespace) -> None:
    repo_dir = Path(args.repo_dir).resolve()
    config_path = Path(args.config_path).resolve()
    checkpoint_path = Path(args.checkpoint_path).resolve()
    coreml_dir = Path(args.coreml_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    coreml_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.sdpa_min_seq_length < 0:
        raise ValueError("--sdpa-min-seq-length must be >= 0")
    if args.sdpa_seq_length_divider < 1:
        raise ValueError("--sdpa-seq-length-divider must be >= 1")
    compute_precision = parse_compute_precision(args.compute_precision)

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    torch.set_grad_enabled(False)

    config = load_config(config_path)
    model = load_model(repo_dir, config, checkpoint_path)
    frames = compute_frames(config)
    replace_rotary_embeddings(model, frames)

    full_output = coreml_dir / WAVEFORM_MODEL_NAME
    full_metadata_output = coreml_dir / WAVEFORM_METADATA_NAME

    print("Tracing waveform-to-waveform model...")
    try:
        traced_full, audio_example = trace_waveform(model, config, args.seed)
        print("Converting waveform-to-waveform model to CoreML...")
        convert_to_coreml(
            traced_full,
            [ct.TensorType(name="audio", shape=audio_example.shape)],
            ["vocals"],
            full_output,
            compute_precision=compute_precision,
            use_sliced_sdpa=args.slice_sdpa,
            sdpa_min_seq_length=args.sdpa_min_seq_length,
            sdpa_seq_length_divider=args.sdpa_seq_length_divider,
        )
        write_waveform_metadata(
            full_metadata_output,
            config,
            model,
            compute_precision_name=args.compute_precision,
            use_sliced_sdpa=args.slice_sdpa,
            sdpa_min_seq_length=args.sdpa_min_seq_length,
            sdpa_seq_length_divider=args.sdpa_seq_length_divider,
        )
        print(f"wrote={full_output}")
        print(f"wrote={full_metadata_output}")
    except Exception as exc:
        error_path = log_dir / "full_conversion_error.txt"
        write_error(error_path, exc)
        print(f"Full conversion failed. See {error_path}")
        raise
