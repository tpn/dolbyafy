from __future__ import annotations

import subprocess
from importlib import metadata
from pathlib import Path
from typing import Optional

from dolbyafy import __version__


def _package_version() -> str:
    try:
        return metadata.version("dolbyafy")
    except metadata.PackageNotFoundError:
        return __version__


def _git_describe() -> Optional[str]:
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "describe", "--tags", "--dirty", "--always", "--abbrev=8"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def get_version() -> str:
    base_version = _package_version()
    git_version = _git_describe()
    if git_version:
        return f"{base_version} (git:{git_version})"
    return base_version
