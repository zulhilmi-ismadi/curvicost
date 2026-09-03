#!/usr/bin/env bash
# Re-vendor the study engine into the package. Run after editing code/p16/.
# The vendored copy is READ-ONLY by convention: edit code/p16/, then sync.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
study="$here/../../code/p16"
dest="$here/../src/curvicost/_core"

if [ ! -d "$study" ]; then
  echo "study tree not found at $study -- nothing to sync" >&2
  exit 1
fi

for f in "$study"/*.py; do
  name="$(basename "$f")"
  [ "$name" = "__init__.py" ] && continue   # carries the vendoring notice
  cp "$f" "$dest/$name"
  echo "synced $name"
done
echo "done -- now run: pytest tests/test_core_sync.py tests/test_reproduces_study.py"
