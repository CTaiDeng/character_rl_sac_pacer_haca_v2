#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN=${PYTHON:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Unable to locate Python interpreter '$PYTHON_BIN'." >&2
  exit 1
fi

echo "Using Python interpreter: $PYTHON_BIN"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install --upgrade numpy
"$PYTHON_BIN" -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
