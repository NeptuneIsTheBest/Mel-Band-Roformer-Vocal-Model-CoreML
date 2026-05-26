# Mel-Band-RoFormer Vocal Model CoreML

Tools for converting Kimberley Jensen's Mel-Band-RoFormer vocal separation model to a macOS CoreML `mlprogram`, verifying it against PyTorch, and running waveform inference.

The converted CoreML model is waveform-to-waveform for one fixed-size chunk:

- Model input: `audio`, `float32`, shape `[1, 2, 352800]`
- Model output: `vocals`, `float32`, shape `[1, 2, 352800]`
- Audio format: mono or stereo input, 44.1 kHz
- Chunk size: 352800 samples, about 8 seconds

This repository does not include the upstream model source checkout, checkpoint, generated CoreML package, or generated audio. Those files are created under ignored local directories.

## Setup

Use Python 3.12. The default Python on some machines may be too new for the PyTorch/CoreML stack.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Prepare Assets

Clone the upstream model repository into `external/` and download the checkpoint into `artifacts/checkpoints/`:

```bash
melband-coreml prepare
```

Default local paths:

```text
external/Mel-Band-Roformer-Vocal-Model/
artifacts/checkpoints/MelBandRoformer.ckpt
```

## Convert

```bash
melband-coreml convert
```

Primary output:

```text
artifacts/coreml/MelBandRoformerVocal_macOS_waveform.mlpackage
```

The package contains a large `weight.bin` file close to 1 GB, so it is intentionally ignored.

## Verify

```bash
melband-coreml verify --mode full --compute-units CPU_ONLY
```

The current conversion has been checked against the fixed PyTorch waveform wrapper with approximately:

```text
max_abs_err  = 3.958121e-09
mean_abs_err = 7.413795e-11
```

## Run Inference

Input audio must be 44.1 kHz mono or stereo. The CoreML model predicts the vocal waveform only. The runner also writes an instrumental stem by subtracting predicted vocals from the input mix.

Create a small reproducible smoke-test file:

```bash
mkdir -p artifacts/smoke
python - <<'PY'
from pathlib import Path

import numpy as np
import soundfile as sf

sr = 44100
seconds = 10
t = np.arange(sr * seconds, dtype=np.float32) / sr
left = 0.2 * np.sin(2 * np.pi * 220 * t)
right = 0.2 * np.sin(2 * np.pi * 330 * t)
audio = np.stack([left, right], axis=1).astype(np.float32)
path = Path("artifacts/smoke/input_44k_stereo.wav")
sf.write(path, audio, sr, subtype="FLOAT")
print(path)
PY
```

Run inference:

```bash
melband-coreml infer artifacts/smoke/input_44k_stereo.wav --compute-units CPU_ONLY
```

Outputs:

```text
artifacts/audio/input_44k_stereo_vocals_coreml.wav
artifacts/audio/input_44k_stereo_instrumental_coreml.wav
```

`CPU_ONLY` is the recommended default on macOS for this package. It avoids extra GPU/ANE memory spikes seen with `ALL` or `CPU_AND_GPU`. The runner wraps each CoreML prediction in a macOS autorelease pool and prints per-chunk RSS while processing long tracks.

## What Changed For CoreML

The original PyTorch `forward(audio)` graph uses dynamic shape conversions, `torch.stft`, `torch.istft`, complex tensors, and scatter-style band averaging. Those are not suitable for a direct full-graph CoreML conversion, so this wrapper expresses the same fixed 8 second waveform path with CoreML-friendly real-valued operations:

1. Reflect center padding and STFT are implemented with fixed `conv1d` DFT kernels.
2. Stereo/frequency packing and band mask averaging are implemented with fixed matrix multiplies.
3. Rotary embeddings are replaced with fixed-shape rotary buffers.
4. Attention is kept as fused `scaled_dot_product_attention`.
5. Complex mask application is expanded into real/imaginary arithmetic.
6. ISTFT overlap-add is implemented with fixed `conv_transpose1d` kernels and a precomputed Hann-window envelope.

Do not enable `--slice-sdpa` unless you are deliberately debugging CoreML attention lowering. The sliced attention pass can expand the graph and cause very high memory usage.
