#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from ml_collections import ConfigDict
from torch import nn


ROOT = Path(__file__).resolve().parent
DEFAULT_REPO_DIR = ROOT / "Mel-Band-Roformer-Vocal-Model"
DEFAULT_CONFIG_PATH = DEFAULT_REPO_DIR / "configs" / "config_vocals_mel_band_roformer.yaml"
DEFAULT_CHECKPOINT_PATH = ROOT / "checkpoints" / "MelBandRoformer.ckpt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_LOG_DIR = ROOT / "logs"


class WaveformWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.model(audio)


class FixedRotaryEmbedding(nn.Module):
    def __init__(
        self,
        base_freqs: torch.Tensor,
        *,
        outer_batch: int,
        heads: int,
        seq_len: int,
        dim: int,
    ) -> None:
        super().__init__()
        positions = torch.arange(seq_len, dtype=base_freqs.dtype)
        freqs = torch.einsum("n,f->nf", positions, base_freqs.detach().cpu())
        freqs = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        self.register_buffer("cos", freqs.cos().reshape(1, 1, seq_len, dim), persistent=False)
        self.register_buffer("sin", freqs.sin().reshape(1, 1, seq_len, dim), persistent=False)
        self.outer_batch = outer_batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.half_dim = dim // 2

    def rotate_queries_or_keys(self, t: torch.Tensor) -> torch.Tensor:
        x = t.reshape(self.outer_batch, self.heads, self.seq_len, self.half_dim, 2)
        x1 = x[..., 0]
        x2 = x[..., 1]
        rotated = torch.stack((-x2, x1), dim=-1)
        rotated = rotated.reshape(self.outer_batch, self.heads, self.seq_len, self.dim)
        return (t * self.cos) + (rotated * self.sin)


class FixedAttention(nn.Module):
    def __init__(self, attention: nn.Module, *, outer_batch: int, seq_len: int) -> None:
        super().__init__()
        self.heads = attention.heads
        self.dim_head = attention.to_qkv.out_features // (3 * attention.heads)
        self.inner_dim = self.heads * self.dim_head
        self.outer_batch = outer_batch
        self.seq_len = seq_len
        self.scale = self.dim_head ** -0.5

        self.norm = attention.norm
        self.to_qkv = attention.to_qkv
        self.to_gates = attention.to_gates
        self.to_out = attention.to_out
        self.rotary_embed = attention.rotary_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        qkv = self.to_qkv(x)
        qkv = qkv.reshape(self.outer_batch, self.seq_len, 3, self.heads, self.dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        q = self.rotary_embed.rotate_queries_or_keys(q)
        k = self.rotary_embed.rotate_queries_or_keys(k)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        gates = self.to_gates(x).sigmoid()
        gates = gates.permute(0, 2, 1).reshape(self.outer_batch, self.heads, self.seq_len, 1)
        out = out * gates

        out = out.permute(0, 2, 1, 3).reshape(self.outer_batch, self.seq_len, self.inner_dim)
        return self.to_out(out)


class MaskCoreWrapper(nn.Module):
    def __init__(self, model: nn.Module, frames: int = 801) -> None:
        super().__init__()
        self.band_split = model.band_split
        self.layers = model.layers
        self.mask_estimators = model.mask_estimators
        self.frames = frames
        self.bands = len(model.band_split.dim_inputs)
        self.dim = self.layers[0][0].layers[0][0].to_gates.in_features

    def forward(self, packed_stft: torch.Tensor) -> torch.Tensor:
        x = self.band_split(packed_stft)

        for time_transformer, freq_transformer in self.layers:
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.reshape(self.bands, self.frames, self.dim)
            x = time_transformer(x)

            x = x.reshape(1, self.bands, self.frames, self.dim)
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.reshape(self.frames, self.bands, self.dim)
            x = freq_transformer(x)

            x = x.reshape(1, self.frames, self.bands, self.dim)

        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        return masks[:, 0]


class FixedWaveformToWaveformWrapper(nn.Module):
    def __init__(self, model: nn.Module, config: ConfigDict, frames: int) -> None:
        super().__init__()
        self.mask_core = MaskCoreWrapper(model, frames=frames)

        self.audio_channels = int(model.audio_channels)
        self.chunk_size = int(config.inference.chunk_size)
        self.n_fft = int(config.model.stft_n_fft)
        self.hop_length = int(config.model.stft_hop_length)
        self.frames = int(frames)
        self.freqs = self.n_fft // 2 + 1
        self.merged_freqs = self.freqs * self.audio_channels
        self.selected_bins = int(model.freq_indices.numel())
        self.pad = self.n_fft // 2
        self.padded_length = self.chunk_size + 2 * self.pad
        self.istft_length = (self.frames - 1) * self.hop_length + self.n_fft

        window = torch.hann_window(self.n_fft)
        freq = torch.arange(self.freqs, dtype=torch.float32).reshape(self.freqs, 1)
        time = torch.arange(self.n_fft, dtype=torch.float32).reshape(1, self.n_fft)
        angles = 2 * torch.pi * freq * time / self.n_fft

        self.register_buffer("stft_real_weight", (window * torch.cos(angles)).reshape(self.freqs, 1, self.n_fft))
        self.register_buffer("stft_imag_weight", (-window * torch.sin(angles)).reshape(self.freqs, 1, self.n_fft))

        scale = torch.full((self.freqs, 1), 2.0 / self.n_fft, dtype=torch.float32)
        scale[0, 0] = 1.0 / self.n_fft
        scale[-1, 0] = 1.0 / self.n_fft
        istft_real = window * scale * torch.cos(angles)
        istft_imag = -window * scale * torch.sin(angles)
        istft_weight = torch.cat([istft_real, istft_imag], dim=0).reshape(self.freqs * 2, 1, self.n_fft)
        self.register_buffer("istft_weight", istft_weight)

        envelope = torch.zeros(self.istft_length, dtype=torch.float32)
        window_square = window.square()
        for frame in range(self.frames):
            start = frame * self.hop_length
            envelope[start:start + self.n_fft] += window_square
        envelope = envelope.clamp_min(1e-11).reshape(1, 1, self.istft_length)
        self.register_buffer("istft_envelope", envelope)

        freq_indices = model.freq_indices.cpu().long()
        select_matrix = torch.zeros(self.merged_freqs, self.selected_bins, dtype=torch.float32)
        select_matrix[freq_indices, torch.arange(self.selected_bins)] = 1.0

        denom = torch.repeat_interleave(model.num_bands_per_freq.cpu().float(), self.audio_channels)
        reduce_matrix = select_matrix.transpose(0, 1) / denom.reshape(1, self.merged_freqs).clamp_min(1e-8)
        self.register_buffer("select_matrix", select_matrix)
        self.register_buffer("reduce_matrix", reduce_matrix)

    def _reflect_center_pad(self, audio: torch.Tensor) -> torch.Tensor:
        left = audio[:, :, 1:self.pad + 1].flip(dims=[-1])
        right = audio[:, :, self.chunk_size - self.pad - 1:self.chunk_size - 1].flip(dims=[-1])
        return torch.cat((left, audio, right), dim=-1)

    def _stft(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padded = self._reflect_center_pad(audio)
        channel_audio = padded.reshape(self.audio_channels, 1, self.padded_length)
        real = F.conv1d(channel_audio, self.stft_real_weight, stride=self.hop_length)
        imag = F.conv1d(channel_audio, self.stft_imag_weight, stride=self.hop_length)

        real = real.reshape(1, self.audio_channels, self.freqs, self.frames)
        imag = imag.reshape(1, self.audio_channels, self.freqs, self.frames)
        real = real.permute(0, 2, 1, 3).reshape(1, self.merged_freqs, self.frames)
        imag = imag.permute(0, 2, 1, 3).reshape(1, self.merged_freqs, self.frames)

        real_bt = real.permute(0, 2, 1)
        imag_bt = imag.permute(0, 2, 1)
        selected_real = torch.matmul(real_bt, self.select_matrix)
        selected_imag = torch.matmul(imag_bt, self.select_matrix)
        packed = torch.stack((selected_real, selected_imag), dim=-1).reshape(
            1, self.frames, self.selected_bins * 2
        )
        return packed, real, imag

    def _apply_masks(
        self,
        packed_masks: torch.Tensor,
        stft_real: torch.Tensor,
        stft_imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masks = packed_masks.reshape(1, self.frames, self.selected_bins, 2)
        mask_real = torch.matmul(masks[:, :, :, 0], self.reduce_matrix).permute(0, 2, 1)
        mask_imag = torch.matmul(masks[:, :, :, 1], self.reduce_matrix).permute(0, 2, 1)

        vocals_real = (stft_real * mask_real) - (stft_imag * mask_imag)
        vocals_imag = (stft_real * mask_imag) + (stft_imag * mask_real)
        return vocals_real, vocals_imag

    def _istft(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        real = real.reshape(1, self.freqs, self.audio_channels, self.frames)
        imag = imag.reshape(1, self.freqs, self.audio_channels, self.frames)
        real = real.permute(0, 2, 1, 3).reshape(self.audio_channels, self.freqs, self.frames)
        imag = imag.permute(0, 2, 1, 3).reshape(self.audio_channels, self.freqs, self.frames)

        stft = torch.cat((real, imag), dim=1)
        audio = F.conv_transpose1d(stft, self.istft_weight, stride=self.hop_length)
        audio = audio / self.istft_envelope
        audio = audio[:, :, self.pad:self.pad + self.chunk_size]
        return audio.reshape(1, self.audio_channels, self.chunk_size)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        packed_stft, stft_real, stft_imag = self._stft(audio)
        packed_masks = self.mask_core(packed_stft)
        vocals_real, vocals_imag = self._apply_masks(packed_masks, stft_real, stft_imag)
        return self._istft(vocals_real, vocals_imag)


def compute_frames(config: ConfigDict) -> int:
    chunk_size = int(config.inference.chunk_size)
    n_fft = int(config.model.stft_n_fft)
    hop_length = int(config.model.stft_hop_length)
    return (chunk_size + 2 * (n_fft // 2) - n_fft) // hop_length + 1


def replace_rotary_embeddings(model: nn.Module, frames: int) -> None:
    bands = len(model.band_split.dim_inputs)
    for time_transformer, freq_transformer in model.layers:
        for transformer_layer in time_transformer.layers:
            attention = transformer_layer[0]
            attention.rotary_embed = FixedRotaryEmbedding(
                attention.rotary_embed.freqs,
                outer_batch=bands,
                heads=attention.heads,
                seq_len=frames,
                dim=attention.to_qkv.out_features // (3 * attention.heads),
            )
            transformer_layer[0] = FixedAttention(attention, outer_batch=bands, seq_len=frames)

        for transformer_layer in freq_transformer.layers:
            attention = transformer_layer[0]
            attention.rotary_embed = FixedRotaryEmbedding(
                attention.rotary_embed.freqs,
                outer_batch=frames,
                heads=attention.heads,
                seq_len=bands,
                dim=attention.to_qkv.out_features // (3 * attention.heads),
            )
            transformer_layer[0] = FixedAttention(attention, outer_batch=frames, seq_len=bands)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Mel-Band-RoFormer vocals model to CoreML.")
    parser.add_argument("--repo-dir", default=str(DEFAULT_REPO_DIR))
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--skip-full", action="store_true", help="Skip waveform-to-waveform conversion.")
    parser.add_argument("--force-maskcore", action="store_true", help="Always build the mask-core fallback too.")
    parser.add_argument(
        "--slice-sdpa",
        action="store_true",
        help="Enable Core ML's sliced scaled-dot-product-attention pass for diagnostics.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_config(path: Path) -> ConfigDict:
    with path.open("r", encoding="utf-8") as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    config.model.flash_attn = False
    return config


def load_model(repo_dir: Path, config: ConfigDict, checkpoint_path: Path) -> nn.Module:
    sys.path.insert(0, str(repo_dir))
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

    stripped_state_dict = {}
    for key, value in state_dict.items():
        stripped_state_dict[key.removeprefix("module.")] = value

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


def write_metadata(path: Path, config: ConfigDict, model: nn.Module, example_width: int) -> None:
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


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    config_path = Path(args.config_path).resolve()
    checkpoint_path = Path(args.checkpoint_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    torch.set_grad_enabled(False)

    config = load_config(config_path)
    model = load_model(repo_dir, config, checkpoint_path)
    frames = compute_frames(config)
    replace_rotary_embeddings(model, frames)

    full_output = output_dir / "MelBandRoformerVocal_macOS_waveform.mlpackage"
    mask_output = output_dir / "MelBandRoformerVocal_iOS18_maskcore.mlpackage"
    full_metadata_output = output_dir / "MelBandRoformerVocal_macOS_waveform_metadata.json"
    metadata_output = output_dir / "MelBandRoformerVocal_iOS18_maskcore_metadata.json"

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
            print(f"Wrote {full_output}")
            print(f"Wrote {full_metadata_output}")
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
        write_metadata(metadata_output, config, model, packed_example.shape[-1])
        print(f"Wrote {mask_output}")
        print(f"Wrote {metadata_output}")


if __name__ == "__main__":
    main()
