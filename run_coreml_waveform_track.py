#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import json
import math
import sys
import threading
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import psutil
import soundfile as sf


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "outputs" / "MelBandRoformerVocal_macOS_waveform.mlpackage"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "audio"


class MacOSAutoreleasePool:
    def __init__(self) -> None:
        self._pool: int | None = None
        self._objc: ctypes.CDLL | None = None
        self._msg_send = None
        self._drain = 0

    def __enter__(self) -> "MacOSAutoreleasePool":
        if sys.platform != "darwin":
            return self
        try:
            ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
            objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
            objc.objc_getClass.restype = ctypes.c_void_p
            objc.objc_getClass.argtypes = [ctypes.c_char_p]
            objc.sel_registerName.restype = ctypes.c_void_p
            objc.sel_registerName.argtypes = [ctypes.c_char_p]
            msg_send = objc.objc_msgSend
            msg_send.restype = ctypes.c_void_p
            msg_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

            pool_class = objc.objc_getClass(b"NSAutoreleasePool")
            alloc = objc.sel_registerName(b"alloc")
            init = objc.sel_registerName(b"init")
            drain = objc.sel_registerName(b"drain")
            pool = msg_send(msg_send(pool_class, alloc), init)

            self._objc = objc
            self._msg_send = msg_send
            self._drain = drain
            self._pool = pool
        except Exception:  # noqa: BLE001
            self._pool = None
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._pool is not None and self._msg_send is not None:
            self._msg_send(self._pool, self._drain)
        self._pool = None


def autorelease_pool() -> contextlib.AbstractContextManager[object]:
    if sys.platform == "darwin":
        return MacOSAutoreleasePool()
    return contextlib.nullcontext()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run waveform-to-waveform Mel-Band-RoFormer CoreML inference.")
    parser.add_argument("input_path")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--chunk-size", type=int, default=352800)
    parser.add_argument("--num-overlap", type=int, default=2)
    parser.add_argument(
        "--compute-units",
        choices=["ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY"],
        default="CPU_ONLY",
        help="Core ML compute units to use when loading the mlpackage.",
    )
    parser.add_argument("--max-chunks", type=int, default=0, help="0 means process the full track.")
    parser.add_argument("--no-write", action="store_true", help="Run inference without writing output files.")
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

    def result(self) -> dict[str, float]:
        return {
            "rss_start_mb": self.start_rss_mb,
            "rss_peak_mb": self.peak_rss_mb,
            "rss_end_mb": self.end_rss_mb,
        }


def parse_compute_units(name: str) -> ct.ComputeUnit:
    return getattr(ct.ComputeUnit, name)


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


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


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()
