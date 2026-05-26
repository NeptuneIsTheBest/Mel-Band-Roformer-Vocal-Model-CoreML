from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf

from .memory import PeakMemoryMonitor, autorelease_pool, rss_mb
from .paths import DEFAULT_AUDIO_OUTPUT_DIR, DEFAULT_COREML_DIR, WAVEFORM_MODEL_NAME
from .runtime import parse_compute_units


def make_window(chunk_size: int) -> np.ndarray:
    fade_size = chunk_size // 10
    window = np.ones(chunk_size, dtype=np.float32)
    window[:fade_size] *= np.linspace(0.0, 1.0, fade_size, dtype=np.float32)
    window[-fade_size:] *= np.linspace(1.0, 0.0, fade_size, dtype=np.float32)
    return window


def prepare_mix(input_path: Path) -> tuple[np.ndarray, int, bool]:
    mix, sr = sf.read(input_path, dtype="float32", always_2d=True)
    original_mono = mix.shape[1] == 1
    if original_mono:
        mix = np.repeat(mix, 2, axis=1)
    if mix.shape[1] != 2:
        raise ValueError(f"Expected mono or stereo audio, got {mix.shape[1]} channels")
    if sr != 44100:
        raise ValueError(f"Expected 44100 Hz audio, got {sr} Hz")
    return mix.T.copy(), sr, original_mono


def run_track(
    model: ct.models.MLModel,
    mix: np.ndarray,
    chunk_size: int,
    num_overlap: int,
    max_chunks: int,
) -> np.ndarray:
    step = chunk_size // num_overlap
    fade_size = chunk_size // 10
    border = chunk_size - step
    original_length = mix.shape[1]

    if mix.shape[1] > 2 * border and border > 0:
        mix = np.pad(mix, ((0, 0), (border, border)), mode="reflect")

    total_length = mix.shape[1]
    num_chunks = math.ceil(total_length / step)
    window = make_window(chunk_size)
    result = np.zeros_like(mix, dtype=np.float32)
    counter = np.zeros_like(mix, dtype=np.float32)

    start_time = time.time()
    for chunk_index, offset in enumerate(range(0, total_length, step), start=1):
        part = mix[:, offset:offset + chunk_size]
        valid_length = part.shape[-1]
        if valid_length < chunk_size:
            pad = chunk_size - valid_length
            if valid_length > chunk_size // 2 + 1:
                part = np.pad(part, ((0, 0), (0, pad)), mode="reflect")
            else:
                part = np.pad(part, ((0, 0), (0, pad)), mode="constant")

        with autorelease_pool():
            prediction = model.predict({"audio": part[np.newaxis, :, :].astype(np.float32)})
            vocals = np.asarray(prediction["vocals"], dtype=np.float32)[0].copy()

        chunk_window = window.copy()
        if offset == 0:
            chunk_window[:fade_size] = 1.0
        elif offset + chunk_size >= total_length:
            chunk_window[-fade_size:] = 1.0

        end = offset + valid_length
        result[:, offset:end] += vocals[:, :valid_length] * chunk_window[:valid_length]
        counter[:, offset:end] += chunk_window[:valid_length]
        del prediction, vocals, part
        gc.collect()

        elapsed = time.time() - start_time
        rate = elapsed / chunk_index
        remaining = rate * (num_chunks - chunk_index)
        print(
            f"chunk {chunk_index}/{num_chunks} "
            f"elapsed={elapsed:.1f}s eta={remaining:.1f}s rss={rss_mb():.1f}MB",
            flush=True,
        )
        if max_chunks and chunk_index >= max_chunks:
            break

    estimated = result / np.maximum(counter, 1e-8)
    if mix.shape[1] != original_length:
        estimated = estimated[:, border:-border]
    return estimated


def run_infer(args: argparse.Namespace) -> None:
    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mix, sr, original_mono = prepare_mix(input_path)
    print(f"input={input_path}")
    print(f"sample_rate={sr} channels={mix.shape[0]} frames={mix.shape[1]} duration={mix.shape[1] / sr:.2f}s")
    print(f"model={Path(args.model_path).resolve()}")
    print(f"compute_units={args.compute_units}")

    with PeakMemoryMonitor() as monitor:
        model = ct.models.MLModel(str(Path(args.model_path).resolve()), compute_units=parse_compute_units(args.compute_units))
        vocals = run_track(model, mix, args.chunk_size, args.num_overlap, args.max_chunks)
    print(json.dumps({"event": "coreml_memory", **monitor.result()}, indent=2), flush=True)
    if args.no_write:
        return

    instrumental = mix - vocals

    vocals_out = vocals.T
    instrumental_out = instrumental.T
    if original_mono:
        vocals_out = vocals_out[:, 0]
        instrumental_out = instrumental_out[:, 0]

    stem = input_path.stem
    vocals_path = output_dir / f"{stem}_vocals_coreml.wav"
    instrumental_path = output_dir / f"{stem}_instrumental_coreml.wav"
    sf.write(vocals_path, vocals_out, sr, subtype="FLOAT")
    sf.write(instrumental_path, instrumental_out, sr, subtype="FLOAT")
    print(f"wrote={vocals_path}")
    print(f"wrote={instrumental_path}")
