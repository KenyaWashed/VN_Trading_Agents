"""Automated CI Script: Runs ruff, mypy, pytest (including mandatory leakage tests), and paper audit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"[CI] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=cwd, check=False)
    if res.returncode != 0:
        raise SystemExit(f"[CI FAILED] Command exited with {res.returncode}: {' '.join(cmd)}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    py = sys.executable
    _run([py, "-m", "ruff", "check", "src", "tests", "experiments", "paper", "scripts", "train_colab_kaggle.py"], repo_root)
    _run([py, "-m", "mypy", "src/vn_rsi"], repo_root)
    _run([py, "-m", "pytest", "tests/"], repo_root)
    if (repo_root / "paper" / "generated_provenance.json").exists():
        _run([py, "paper/verify_paper.py"], repo_root)
    print("[CI] ALL CHECKS GREEN.")


if __name__ == "__main__":
    main()
