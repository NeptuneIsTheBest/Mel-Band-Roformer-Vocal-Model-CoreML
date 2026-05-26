# Mel-Band-RoFormer Vocal Model CoreML

Scripts for converting Kimberley Jensen's Mel-Band-RoFormer vocal separation model to a macOS CoreML `mlprogram`.

The converted model is waveform-to-waveform:

- Input: `audio`, `float32`, shape `[1, 2, 352800]`
- Output: `vocals`, `float32`, shape `[1, 2, 352800]`
- Audio format: stereo, 44.1 kHz
- Chunk size: 352800 samples, about 8 seconds

This repository intentionally does not include the checkpoint or generated CoreML weights. Download the checkpoint locally and generate the `.mlpackage` with the scripts below.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model.git Mel-Band-Roformer-Vocal-Model
```

Download the model checkpoint:

```bash
python download_melbandroformer_weights.py
```

The checkpoint is downloaded to:

```text
checkpoints/MelBandRoformer.ckpt
```

## Convert

```bash
python convert_melbandroformer_coreml.py
```

The primary output is:

```text
outputs/MelBandRoformerVocal_macOS_waveform.mlpackage
```

The package contains a large file at:

```text
outputs/MelBandRoformerVocal_macOS_waveform.mlpackage/Data/com.apple.CoreML/weights/weight.bin
```

That file is close to 1 GB, so it is not tracked in this public scripts repository.

## Run Inference

```bash
python run_coreml_waveform_track.py input.flac --compute-units CPU_ONLY
```

Outputs are written to:

```text
outputs/audio/<name>_vocals_coreml.wav
outputs/audio/<name>_instrumental_coreml.wav
```

`CPU_ONLY` is the recommended default on macOS for this package. It avoids the extra GPU/ANE memory spikes seen with `ALL` or `CPU_AND_GPU`. The runner also wraps each CoreML prediction in a macOS autorelease pool and prints per-chunk RSS while processing long tracks.

## Verify

```bash
python verify_melbandroformer_coreml.py --mode full --compute-units CPU_ONLY
```

The current conversion was checked against the fixed PyTorch waveform wrapper with approximately:

```text
max_abs_err  = 3.608875e-09
mean_abs_err = 7.136894e-11
```

## What Changed For CoreML

The original PyTorch `forward(audio)` graph uses dynamic shape conversions, `torch.stft`, `torch.istft`, complex tensors, and scatter-style band averaging. Those are not suitable for a direct full-graph CoreML conversion, so the conversion wrapper expresses the same fixed 8 second waveform path with CoreML-friendly real-valued operations:

1. Reflect center padding and STFT are implemented with fixed `conv1d` DFT kernels.
2. Stereo/frequency packing and band mask averaging are implemented with fixed matrix multiplies.
3. Rotary embeddings are replaced with fixed-shape rotary buffers.
4. Attention is kept as fused `scaled_dot_product_attention`.
5. Complex mask application is expanded into real/imaginary arithmetic.
6. ISTFT overlap-add is implemented with fixed `conv_transpose1d` kernels and a precomputed Hann-window envelope.

Do not enable `--slice-sdpa` unless you are deliberately debugging CoreML attention lowering. The sliced attention pass can expand the graph and cause very high memory usage.
