#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

"""
将 README.md 规范为 UTF-8（无 BOM）+ LF 行尾。

- 默认处理仓库根的 README.md；也可传入路径参数覆盖。
- 仅重写换行，不改动正文；自动去除 UTF-8 BOM。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def normalize_to_lf(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def read_text_bytes(p: Path) -> bytes:
    return p.read_bytes()


def decode_text(b: bytes) -> str:
    try:
        return b.decode("utf-8-sig")
    except Exception:
        try:
            return b.decode("gbk")
        except Exception:
            return b.decode("utf-8", errors="replace")


def write_lf_utf8(p: Path, text: str) -> None:
    # 始终以 UTF-8（无 BOM）+ LF 写回
    p.write_text(normalize_to_lf(text), encoding="utf-8", newline="\n")


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[1]
    target = Path(argv[0]) if argv else (root / "README.md")
    if not target.exists():
        print(f"[fix_readme_lf] skip: not found {target}")
        return 0
    b = read_text_bytes(target)
    text = decode_text(b)
    write_lf_utf8(target, text)
    print(f"[fix_readme_lf] normalized to UTF-8+LF: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

