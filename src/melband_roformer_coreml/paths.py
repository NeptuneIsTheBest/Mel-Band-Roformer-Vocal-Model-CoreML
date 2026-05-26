from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EXTERNAL_REPO_DIR = REPO_ROOT / "external" / "Mel-Band-Roformer-Vocal-Model"
DEFAULT_CONFIG_PATH = DEFAULT_EXTERNAL_REPO_DIR / "configs" / "config_vocals_mel_band_roformer.yaml"

DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DEFAULT_CHECKPOINT_DIR = DEFAULT_ARTIFACTS_DIR / "checkpoints"
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "MelBandRoformer.ckpt"
DEFAULT_COREML_DIR = DEFAULT_ARTIFACTS_DIR / "coreml"
DEFAULT_AUDIO_OUTPUT_DIR = DEFAULT_ARTIFACTS_DIR / "audio"
DEFAULT_LOG_DIR = DEFAULT_ARTIFACTS_DIR / "logs"

WAVEFORM_MODEL_NAME = "MelBandRoformerVocal_macOS_waveform.mlpackage"
WAVEFORM_METADATA_NAME = "MelBandRoformerVocal_macOS_waveform_metadata.json"
