"""Reproducibility utilities: deterministic seeds, config/code SHA-256 hashing, git state."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def set_global_seed(seed: int) -> np.random.Generator:
    """Fix Python, OS hash, and NumPy RNG seeds; return a dedicated Generator."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML config safely as a dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file {path}, got {type(data).__name__}")
    return data


def compute_config_hash(config_obj: Any) -> str:
    """Compute canonical SHA-256 hex digest of any JSON-serializable config structure."""
    canonical = json.dumps(config_obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file on disk (normalizing CRLF->LF for text/code files)."""
    p = Path(path)
    raw = p.read_bytes()
    if p.suffix.lower() in {".py", ".yaml", ".yml", ".md", ".json", ".toml", ".tex", ".bib"}:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def compute_tree_sha256(paths: Sequence[str | Path]) -> str:
    """Compute composite SHA-256 seal across a list of files or directories (sorted deterministically)."""
    hasher = hashlib.sha256()
    collected: list[Path] = []
    for item in paths:
        p = Path(item)
        if p.is_file():
            collected.append(p)
        elif p.is_dir():
            for sub in sorted(p.rglob("*")):
                if sub.is_file() and "__pycache__" not in sub.parts and sub.suffix != ".pyc":
                    collected.append(sub)
    for fpath in sorted(collected, key=lambda x: x.as_posix()):
        hasher.update(fpath.name.encode("utf-8"))
        hasher.update(compute_file_sha256(fpath).encode("utf-8"))
    return hasher.hexdigest()


def get_git_commit_hash(repo_root: str | Path | None = None) -> str:
    """Return current git HEAD commit hash (with -dirty suffix if working tree modified), or 'untracked'."""
    cwd = Path(repo_root) if repo_root else Path.cwd()
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if head.returncode != 0:
            return "git_uninitialized"
        commit = head.stdout.strip()[:12]
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        dirty = "-dirty" if status.stdout.strip() else ""
        return f"{commit}{dirty}"
    except Exception:
        return "git_unavailable"
