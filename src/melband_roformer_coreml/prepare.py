from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download

from .paths import DEFAULT_CHECKPOINT_DIR, DEFAULT_EXTERNAL_REPO_DIR


DEFAULT_SOURCE_REPO_URL = "https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model.git"
DEFAULT_HF_REPO_ID = "KimberleyJSN/melbandroformer"
DEFAULT_CHECKPOINT_FILENAME = "MelBandRoformer.ckpt"


def clone_source_repo(repo_url: str, repo_dir: Path) -> None:
    if repo_dir.exists():
        print(f"source_repo={repo_dir} status=reused")
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True)
    print(f"source_repo={repo_dir} status=cloned")


def download_checkpoint(repo_id: str, filename: str, checkpoint_dir: Path) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=checkpoint_dir,
    )
    resolved = Path(path).resolve()
    print(f"checkpoint={resolved}")
    return resolved


def run_prepare(args: argparse.Namespace) -> None:
    clone_source_repo(args.source_repo_url, Path(args.repo_dir).resolve())
    download_checkpoint(
        repo_id=args.hf_repo_id,
        filename=args.checkpoint_filename,
        checkpoint_dir=Path(args.checkpoint_dir).resolve(),
    )
