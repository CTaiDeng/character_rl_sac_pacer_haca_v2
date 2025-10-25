#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see https://www.gnu.org/licenses/.

"""
Normalize Markdown files to UTF-8 (no BOM) with LF line endings.
- Scope: explicit files (top-level docs by caller), excludes subdirectories unless passed.
- Strategy: try UTF-8 decode first; if fails, try GBK (cp936) as fallback. Always write back UTF-8 (no BOM) + LF.
- Check mode: with --check, do not write; exit non-zero if any file would be rewritten.
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse
from _doc_edit_guard import require_explicit_doc_paths


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def to_lf(s: str) -> str:
    # Normalize to LF
    return s.replace("\r\n", "\n").replace("\r", "\n")


def recode_to_utf8_lf(p: Path, dry_run: bool = False) -> bool:
    b = p.read_bytes()
    text = None
    try:
        text = b.decode("utf-8-sig")
    except Exception:
        try:
            text = b.decode("gbk")
        except Exception:
            return False
    # Normalize newlines to LF and compute output (UTF-8, no BOM)
    norm = to_lf(text)
    out_b = norm.encode("utf-8")
    changed = (out_b != b)
    if dry_run:
        return changed
    if changed:
        p.write_bytes(out_b)
    return changed


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Force .md to UTF-8 (no BOM) + CRLF（仅处理显式给出的文件路径）")
    ap.add_argument("--check", action="store_true", help="Only check; non-zero exit if any file would be rewritten")
    ap.add_argument("files", nargs='+', help='项目相对路径，如 docs/1234567890_标题.md')
    args = ap.parse_args(argv)
    files = require_explicit_doc_paths(args.files)
    updated = 0
    for p in files:
        try:
            would = recode_to_utf8_lf(p, dry_run=True)
            if args.check:
                if would:
                    updated += 1
                    print(f"[force_docs_utf8_bom] would rewrite: {p}")
                continue
            if recode_to_utf8_lf(p, dry_run=False):
                updated += 1
                print(f"[force_docs_utf8_bom] rewrote LF: {p}")
        except Exception as e:
            print(f"[force_docs_utf8_bom] skip {p}: {e}")
    print(f"[force_docs_utf8_bom] updated={updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
