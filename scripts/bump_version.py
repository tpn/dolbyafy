from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "src" / "dolbyafy" / "__init__.py"


def _replace(path: Path, pattern: str, replacement: str) -> None:
    data = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, data, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Expected one match in {path}, got {count}.")
    path.write_text(updated, encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/bump_version.py 1.0")
        return 2

    version = sys.argv[1].strip()
    if not re.match(r"^\\d+\\.\\d+\\.\\d+$", version):
        print("Version must look like 1.0.0")
        return 2

    _replace(
        PYPROJECT,
        r'^(version\\s*=\\s*)".*"$',
        rf'\\1"{version}"',
    )
    _replace(
        INIT,
        r'^__version__\\s*=\\s*".*"$',
        rf'__version__ = "{version}"',
    )
    print(f"Bumped version to {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
