#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Mel-Band-RoFormer vocal checkpoint.")
    parser.add_argument("--repo-id", default="KimberleyJSN/melbandroformer")
    parser.add_argument("--filename", default="MelBandRoformer.ckpt")
    parser.add_argument("--output-dir", default="checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
    )
    print(Path(path).resolve())


if __name__ == "__main__":
    main()
