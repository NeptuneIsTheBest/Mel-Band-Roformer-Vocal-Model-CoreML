from __future__ import annotations

import argparse

from . import __version__
from .paths import (
    DEFAULT_AUDIO_OUTPUT_DIR,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_COREML_DIR,
    DEFAULT_EXTERNAL_REPO_DIR,
    DEFAULT_LOG_DIR,
    WAVEFORM_MODEL_NAME,
)


DEFAULT_SOURCE_REPO_URL = "https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model.git"
DEFAULT_HF_REPO_ID = "KimberleyJSN/melbandroformer"
DEFAULT_CHECKPOINT_FILENAME = "MelBandRoformer.ckpt"


def run_prepare_command(args: argparse.Namespace) -> None:
    from .prepare import run_prepare

    run_prepare(args)


def run_convert_command(args: argparse.Namespace) -> None:
    from .conversion import run_convert

    run_convert(args)


def run_verify_command(args: argparse.Namespace) -> None:
    from .verification import run_verify

    run_verify(args)


def run_infer_command(args: argparse.Namespace) -> None:
    from .inference import run_infer

    run_infer(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="melband-coreml",
        description="Prepare, convert, verify, and run the Mel-Band-RoFormer CoreML vocal model.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Clone source model code and download the checkpoint.")
    prepare.add_argument("--source-repo-url", default=DEFAULT_SOURCE_REPO_URL)
    prepare.add_argument("--repo-dir", default=str(DEFAULT_EXTERNAL_REPO_DIR))
    prepare.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID)
    prepare.add_argument("--checkpoint-filename", default=DEFAULT_CHECKPOINT_FILENAME)
    prepare.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    prepare.set_defaults(func=run_prepare_command)

    convert = subparsers.add_parser("convert", help="Convert the PyTorch checkpoint to CoreML.")
    convert.add_argument("--repo-dir", default=str(DEFAULT_EXTERNAL_REPO_DIR))
    convert.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    convert.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    convert.add_argument("--coreml-dir", default=str(DEFAULT_COREML_DIR))
    convert.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    convert.add_argument(
        "--slice-sdpa",
        action="store_true",
        help="Enable Core ML's sliced scaled-dot-product-attention pass for diagnostics.",
    )
    convert.add_argument("--seed", type=int, default=1234)
    convert.set_defaults(func=run_convert_command)

    verify = subparsers.add_parser("verify", help="Compare CoreML outputs with the PyTorch wrapper.")
    verify.add_argument("--repo-dir", default=str(DEFAULT_EXTERNAL_REPO_DIR))
    verify.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    verify.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    verify.add_argument("--coreml-dir", default=str(DEFAULT_COREML_DIR))
    verify.add_argument("--result-path", default="")
    verify.add_argument(
        "--compute-units",
        choices=["ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY"],
        default="CPU_ONLY",
        help="Core ML compute units to use when loading the mlpackage.",
    )
    verify.add_argument("--seed", type=int, default=1234)
    verify.set_defaults(func=run_verify_command)

    infer = subparsers.add_parser("infer", help="Run CoreML waveform inference on an audio file.")
    infer.add_argument("input_path")
    infer.add_argument("--model-path", default=str(DEFAULT_COREML_DIR / WAVEFORM_MODEL_NAME))
    infer.add_argument("--output-dir", default=str(DEFAULT_AUDIO_OUTPUT_DIR))
    infer.add_argument("--chunk-size", type=int, default=352800)
    infer.add_argument("--num-overlap", type=int, default=2)
    infer.add_argument(
        "--compute-units",
        choices=["ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY"],
        default="CPU_ONLY",
        help="Core ML compute units to use when loading the mlpackage.",
    )
    infer.add_argument("--max-chunks", type=int, default=0, help="0 means process the full track.")
    infer.add_argument("--no-write", action="store_true", help="Run inference without writing output files.")
    infer.set_defaults(func=run_infer_command)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
