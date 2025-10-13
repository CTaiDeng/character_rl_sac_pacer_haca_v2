#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

"""
为指定目录顶层、文件名匹配 "^<ts>_*.md" 的文档，在文档首个标题下一行写入/更新日期行：
  日期：YYYY-MM-DD

规则
- 仅处理顶层，不递归。
- 目录列表：docs/；my_docs/project_docs/；my_project/gmx_split_20250924_011827/docs/。
- 日期来源：文件名中的秒级时间戳（不从 git 取），对应 UTC→本地时间的 YYYY-MM-DD（使用本地时区）。
- 插入位置：文档首个以 # 开头的标题下方一行；若已有“日期：”行则就地更新。
- 编码：读写均为 UTF-8（BOM），行尾规范化为 LF。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import argparse
from _doc_edit_guard import require_explicit_doc_paths


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRS = [
    ROOT / 'docs',
    ROOT / 'my_docs' / 'project_docs',
    ROOT / 'my_project' / 'gmx_split_20250924_011827' / 'docs',
]

PREFIX_RE = re.compile(r'^(\d+)_.*\.md$', re.IGNORECASE)
TITLE_RE = re.compile(r'^\s*#{1,6}\s+')
DATE_RE = re.compile(r'^\s*日期：')


def read_text(path: Path) -> Tuple[str, str]:
    data = path.read_bytes()
    nl = '\r\n' if b'\r\n' in data else '\n'
    text = data.decode('utf-8-sig', errors='replace')
    return text, nl


def write_text(path: Path, text: str, nl: str) -> None:
    # 统一为 LF 写回（仓库规范），并带 BOM
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    with open(path, 'w', encoding='utf-8-sig', newline='\n') as f:
        f.write(text)


def ensure_date_after_title(text: str, date_str: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    # 找到首个标题行
    title_idx = None
    for i, ln in enumerate(lines):
        if TITLE_RE.match(ln):
            title_idx = i
            break
    if title_idx is None:
        # 无标题，则在文件头插入日期
        return '\n'.join([f'日期：{date_str}', ''] + lines)

    insert_idx = title_idx + 1
    # 跳过紧随其后的空行
    while insert_idx < len(lines) and lines[insert_idx].strip() == '':
        insert_idx += 1

    # 若已有日期行则就地更新
    if insert_idx < len(lines) and DATE_RE.match(lines[insert_idx]):
        lines[insert_idx] = f'日期：{date_str}'
        return '\n'.join(lines)

    # 否则在标题下方插入
    new_lines: List[str] = []
    new_lines.extend(lines[:title_idx+1])
    new_lines.append(f'日期：{date_str}')
    new_lines.append('')
    new_lines.extend(lines[title_idx+1:])
    return '\n'.join(new_lines)


def process_files(files: List[Path]) -> int:
    changed = 0
    for p in files:
        if p.suffix.lower() != '.md':
            continue
        m = PREFIX_RE.match(p.name)
        if not m:
            continue
        try:
            ts = int(m.group(1))
            dt = datetime.fromtimestamp(ts)
            date_str = dt.strftime('%Y-%m-%d')
        except Exception:
            continue
        text, nl = read_text(p)
        new_text = ensure_date_after_title(text, date_str)
        if new_text != text:
            write_text(p, new_text, nl)
            changed += 1
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description='Insert/Update date after title（仅处理显式给出的文件路径）')
    ap.add_argument('files', nargs='+', help='项目相对路径，如 docs/1234567890_标题.md')
    args = ap.parse_args()
    files = require_explicit_doc_paths(args.files)
    total = process_files(files)
    print(f'[insert_doc_date_from_prefix] updated={total}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
