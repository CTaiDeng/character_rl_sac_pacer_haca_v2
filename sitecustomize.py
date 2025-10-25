"""
Global UTF-8 (no BOM) + LF write policy for this repository.

When any Python script is executed from the repository root (so that this
`sitecustomize.py` is on sys.path), we patch builtins.open for text-writing
operations to default to UTF-8 without BOM and LF line endings, unless the
caller explicitly sets `encoding` or `newline`.

We also patch `pathlib.Path.write_text` to normalize newlines to `\n` and use
UTF-8 by default.

To disable temporarily, set environment variable:
  DISABLE_UTF8LF_SITEPATCH=1
"""

import os
import builtins
from pathlib import Path

if os.environ.get("DISABLE_UTF8LF_SITEPATCH") in {"1", "true", "True"}:
    # Opt-out for debugging or special cases
    pass
else:
    _orig_open = builtins.open

    def _patched_open(file, mode="r", buffering=-1, encoding=None, errors=None,
                      newline=None, closefd=True, opener=None):
        try:
            text_mode = "b" not in mode
            write_mode = any(ch in mode for ch in ("w", "a", "x"))
            if text_mode and write_mode:
                if encoding is None:
                    encoding = "utf-8"
                if newline is None:
                    newline = "\n"
        except Exception:
            # In case of odd modes, fall back safely
            pass
        return _orig_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

    builtins.open = _patched_open  # type: ignore

    # Patch Path.write_text to normalize to LF and enforce UTF-8 by default
    _orig_write_text = Path.write_text

    def _patched_write_text(self: Path, data: str, encoding: str | None = None, errors: str | None = None):
        enc = encoding or "utf-8"
        # Normalize newlines to LF
        if isinstance(data, str):
            data = data.replace("\r\n", "\n").replace("\r", "\n")
        with self.open("w", encoding=enc, errors=errors, newline="\n") as f:  # type: ignore[arg-type]
            return f.write(data)

    Path.write_text = _patched_write_text  # type: ignore

