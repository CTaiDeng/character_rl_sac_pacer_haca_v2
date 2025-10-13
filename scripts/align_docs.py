#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

"""
文档对齐指令（严格模式）：
- 最高原则：未显式给出项目相对路径时，不对 docs 下文章做任何修改；只更新 README 文末的“文档摘要索引”。
- 当显式给出项目相对路径列表时，仅对这些文件执行规范化与页脚/编码处理，随后更新 README 索引。

用法：
  python scripts/align_docs.py [docs/1234_标题.md docs/5678_标题.md ...]
"""

from __future__ import annotations

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


def _resolve_venv_python() -> str:
    win_path = ROOT / ".venv" / "Scripts" / "python.exe"
    posix_path = ROOT / ".venv" / "bin" / "python"
    if platform.system().lower().startswith("win") and win_path.exists():
        return str(win_path)
    if posix_path.exists() and os.access(posix_path, os.X_OK):
        return str(posix_path)
    return sys.executable


def run(cmd: List[str]) -> int:
    try:
        print("[align_docs] $", " ".join(cmd))
        return subprocess.call(cmd, cwd=str(ROOT))
    except Exception as e:
        print(f"[align_docs] 运行失败: {e}")
        return 1


def main() -> int:
    rc = 0
    py = _resolve_venv_python()

    files = sys.argv[1:]
    if files:
        # 仅处理显式目标文件
        rc |= run([py, str(ROOT / 'scripts' / 'ensure_docs_style_from_date.py'), *files])
        rc |= run([py, str(ROOT / 'scripts' / 'insert_o3_citation_note.py'), *files])
        rc |= run([py, str(ROOT / 'scripts' / 'force_docs_utf8_bom.py'), *files])
        rc |= run([py, str(ROOT / 'scripts' / 'insert_docs_license_footer.py'), *files])

    # README 索引（只读复制，不改动原文档）
    rc |= run([py, str(ROOT / 'scripts' / 'update_readme_index.py')])
    # 仅规范化 README
    rc |= run([py, str(ROOT / 'scripts' / 'md_normalize.py'), 'README.md'])

    if rc == 0:
        print('[align_docs] 文档对齐完成（严格模式）')
    else:
        print('[align_docs] 文档对齐存在问题，请查看上方日志')
    return rc


if __name__ == '__main__':
    sys.exit(main())

