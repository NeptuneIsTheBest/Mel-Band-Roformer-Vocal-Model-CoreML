# Mel-Band-RoFormer Vocal Model CoreML

Tools for converting Kimberley Jensen's Mel-Band-RoFormer vocal separation model to a macOS CoreML `mlprogram`, verifying it against PyTorch, and running waveform inference.

The converted CoreML model is waveform-to-waveform for one fixed-size chunk:

- Model input: `audio`, `float32`, shape `[1, 2, 352800]`
- Model output: `vocals`, `float32`, shape `[1, 2, 352800]`
- Audio format: mono or stereo input, 44.1 kHz
- Chunk size: 352800 samples, about 8 seconds

## Setup

Use Python 3.12. The default Python on some machines may be too new for the PyTorch/CoreML stack.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

You can download the prebuilt CoreML package from the [v1.0.0 release](https://github.com/NeptuneIsTheBest/Mel-Band-Roformer-Vocal-Model-CoreML/releases/tag/v1.0.0), or follow the steps below to prepare and convert it locally.

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

The default conversion writes an fp16 `mlprogram` and enables CoreML's sliced-Q
scaled-dot-product-attention lowering for long sequence attention. This keeps
the same fixed 8 second input shape while reducing GPU attention workspace
pressure. For diagnostics, the defaults can be changed:

```bash
melband-coreml convert \
  --compute-precision FLOAT16 \
  --sdpa-min-seq-length 128 \
  --sdpa-seq-length-divider 32
```

Use `--no-slice-sdpa` only when comparing against the unsliced fused CoreML
attention lowering.

## Verify

```bash
melband-coreml verify --compute-units CPU_ONLY
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

`CPU_ONLY` remains the safest fallback, especially for older or unsliced packages. Packages converted with the current defaults can also be tested with `CPU_AND_GPU`; sliced SDPA is intended to avoid the large GPU attention workspace from the unsliced model. The runner wraps each CoreML prediction in a macOS autorelease pool and prints per-chunk RSS while processing long tracks.

## What Changed For CoreML

The upstream PyTorch model is written as a flexible waveform graph. Its `forward(audio)` path builds shapes dynamically, calls `torch.stft` and `torch.istft`, carries complex-valued tensors through the mask path, and uses scatter-style logic to average overlapping frequency bands. That is convenient in PyTorch, but it is not a good direct conversion target for CoreML. The CoreML wrapper keeps the same 8 second vocal-separation path, but rewrites the conversion boundary as a fixed-shape, real-valued graph:

- The exported model accepts exactly one stereo chunk, `audio` with shape `[1, 2, 352800]`, and returns `vocals` with the same shape. Longer tracks are handled by the Python runner, which chunks the waveform and overlap-adds the chunk predictions outside the model.
- The STFT front end is implemented manually. The wrapper applies the same reflect center padding, builds Hann-windowed real and imaginary DFT kernels ahead of time, and runs them with fixed-stride `conv1d` instead of relying on `torch.stft`.
- Stereo frequency bins are flattened into a stable `[frequency, channel]` layout. The model's selected mel-band inputs are gathered with a precomputed selection matrix, so the traced graph uses matrix multiplication rather than dynamic indexing.
- The RoFormer core is still the upstream neural network: band splitting, time and frequency transformer blocks, mask estimators, and fused `scaled_dot_product_attention` are preserved. The attention wrapper only fixes the tensor reshapes and replaces rotary embedding generation with precomputed cosine and sine buffers for the known sequence lengths.
- The predicted complex masks are unpacked into real and imaginary components. Complex multiplication is written out explicitly as real arithmetic, which avoids CoreML complex tensor support while preserving the same mask application.
- Band mask averaging is also expressed as a fixed matrix multiply. The reduction matrix includes the per-frequency denominator from the upstream model, so frequencies covered by multiple bands are averaged deterministically.
- The ISTFT back end is implemented with fixed `conv_transpose1d` kernels. The wrapper precomputes the inverse DFT weights and Hann-window overlap envelope, divides by that envelope, and crops away the center padding to return the original 8 second chunk length.

The conversion writes an `mlprogram` model in `FLOAT16` precision by default. For long time-axis attention, the default pass pipeline rewrites CoreML's fused `scaled_dot_product_attention` into statically sliced Q chunks. This preserves the 8 second waveform input while reducing peak GPU attention workspace. Verification compares CoreML output against this fixed PyTorch waveform wrapper, not against a separate post-processing path.
