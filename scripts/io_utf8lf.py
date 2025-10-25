# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

"""
Lightweight helpers to write files as UTF-8 (no BOM) + LF.

Typical usage in scripts:
    from io_utf8lf import write_text, open_utf8lf, write_json
    write_text("out.md", content)
    with open_utf8lf("out.txt", "w") as f:
        f.write("...\n")
"""

from __future__ import annotations

import json
import builtins
from pathlib import Path


def ensure_lf(text: str) -> str:
    if not isinstance(text, str):
        return text
    return text.replace("\r\n", "\n").replace("\r", "\n")


def open_utf8lf(path: str | Path, mode: str = "w", **kwargs):
    if "b" in mode:
        return builtins.open(path, mode, **kwargs)
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("newline", "\n")
    return builtins.open(path, mode, **kwargs)


def write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    text = ensure_lf(text)
    with open_utf8lf(path, "w", encoding=encoding) as f:
        f.write(text)


def append_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    text = ensure_lf(text)
    with open_utf8lf(path, "a", encoding=encoding) as f:
        f.write(text)


def write_json(path: str | Path, obj, ensure_ascii: bool = False, indent: int | None = 2) -> None:
    with open_utf8lf(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)
        f.write("\n")  # trailing newline

