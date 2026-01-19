#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: scripts/cut_release.sh v1.0.0"
  exit 2
fi

tag="$1"

if ! [[ "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Tag must look like v1.0.0"
  exit 2
fi

version="$(python - <<'PY'
from pathlib import Path
import tomllib
data = tomllib.loads(Path("pyproject.toml").read_bytes())
print(data["project"]["version"])
PY
)"

if [[ "$tag" != "v$version" ]]; then
  echo "Tag $tag does not match project version $version"
  exit 2
fi

git tag -a "$tag" -m "Release $tag"
echo "Created tag $tag"
echo "Next: git push origin $tag"
