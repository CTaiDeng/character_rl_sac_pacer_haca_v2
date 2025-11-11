#!/usr/bin/env python
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
Check staged or provided source files for required GPL-3.0 SPDX header and copyright line.

Policy (from AGENTS.md):
- Required in all source files (non-Markdown, non-docs/kernel_reference, non-build/cache):
  - SPDX-License-Identifier: GPL-3.0-only (or containing GPL-3.0 in SPDX)
  - Copyright (C) 2025 GaoZheng
This checker blocks commit if any staged file is missing the header.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List


CODE_EXTS = {
    ".py", ".sh", ".bash", ".zsh", ".ps1", ".rb", ".pl", ".r", ".jl",
    ".c", ".h", ".cpp", ".cc", ".hpp", ".cs", ".java", ".go",
    ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".swift", ".kt", ".kts", ".scala", ".rs",
    ".php", ".lua", ".sql", ".html", ".htm", ".xml", ".svg",
}

SKIP_EXTS = {".md", ".mdx", ".markdown", ".rst", ".adoc", ".txt"}

SKIP_DIR_PREFIXES = [
    os.path.join("docs", "kernel_reference") + os.sep,
    ".git" + os.sep,
    "node_modules" + os.sep,
    "dist" + os.sep,
    "build" + os.sep,
    "out" + os.sep,
    "target" + os.sep,
    "bin" + os.sep,
    "obj" + os.sep,
    "__pycache__" + os.sep,
    ".pytest_cache" + os.sep,
    ".mypy_cache" + os.sep,
    ".cache" + os.sep,
    "venv" + os.sep,
    ".venv" + os.sep,
]


def _is_skipped(path: str) -> bool:
    # Normalize to repo-relative path if possible
    p = path.replace("\\", "/")
    # Use os.sep logic for robust prefix checks across platforms
    ps = path
    for pref in SKIP_DIR_PREFIXES:
        if ps.startswith(pref) or p.startswith(pref.replace(os.sep, "/")):
            return True
    return False


def read_head_lines(path: str, max_lines: int = 120) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line)
            return "".join(lines)
    except Exception:
        return ""


def has_required_header(text: str) -> bool:
    t = text
    if not t:
        return False
    # SPDX and GPL presence
    spdx_ok = ("SPDX-License-Identifier" in t and "GPL-3.0" in t)
    gpl_hint = ("GNU General Public License" in t)
    copyright_ok = ("Copyright (C) 2025 GaoZheng" in t)
    return (spdx_ok or gpl_hint) and copyright_ok


def gather_paths_from_stdin(null_separated: bool) -> List[str]:
    data = sys.stdin.buffer.read()
    if not data:
        return []
    if null_separated:
        parts = data.split(b"\x00")
    else:
        parts = data.splitlines()
    out: List[str] = []
    for b in parts:
        if not b:
            continue
        try:
            s = b.decode("utf-8", errors="ignore")
        except Exception:
            continue
        s = s.strip().strip("\r")
        if s:
            out.append(s)
    return out


def filter_source_candidates(paths: Iterable[str]) -> List[str]:
    result: List[str] = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        if _is_skipped(p):
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext in SKIP_EXTS:
            continue
        if ext not in CODE_EXTS:
            continue
        result.append(p)
    return result


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Check required GPL-3.0 headers in source files")
    ap.add_argument("paths", nargs="*", help="Paths to check (if omitted, requires --stdin)")
    ap.add_argument("--stdin", action="store_true", help="Read file list from stdin")
    ap.add_argument("-0", dest="null_sep", action="store_true", help="When reading stdin, use NUL as separator")
    args = ap.parse_args(argv)

    paths: List[str] = []
    if args.stdin:
        paths = gather_paths_from_stdin(args.null_sep)
    paths.extend(args.paths)

    if not paths:
        # Nothing to check
        return 0

    candidates = filter_source_candidates(paths)
    missing: List[str] = []
    for p in candidates:
        head = read_head_lines(p)
        if not has_required_header(head):
            missing.append(p)

    if missing:
        sys.stderr.write("[pre-commit] 缺少 GPL 头部的文件：\n")
        for m in missing:
            sys.stderr.write(f"  - {m}\n")
        sys.stderr.write("\n修复命令：pwsh ./scripts/add_gpl_header.ps1 -Root .\n")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

