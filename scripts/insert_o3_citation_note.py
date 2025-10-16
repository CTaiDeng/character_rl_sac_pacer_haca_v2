#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

"""
当 docs/*.md 正文出现以下关键词之一时，在“日期：YYYY-MM-DD”下一行插入统一注释：
- O3理论
- O3元数学理论
- 主纤维丛版广义非交换李代数
- PFB-GNLA

注释内容（单行）：
  #### ***注：“O3理论/O3元数学理论/主纤维丛版广义非交换李代数(PFB-GNLA)”相关理论参见： [作者（GaoZheng）网盘分享](https://drive.google.com/drive/folders/1lrgVtvhEq8cNal0Aa0AjeCNQaRA8WERu?usp=sharing) 或 [作者（GaoZheng）开源项目](https://github.com/CTaiDeng/open_meta_mathematical_theory) 或 [作者（GaoZheng）主页](https://mymetamathematics.blogspot.com)，欢迎访问！***

写回编码：UTF-8（BOM）+ LF。
仅处理显式传入的文件列表；不接受目录参数，避免误处理 scripts/README.md 等非知识库文件。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

from _doc_edit_guard import require_explicit_doc_paths
from _docs_config import load_skip_paths, is_under


KEYWORDS = (
    'O3理论',
    'O3元数学理论',
    '主纤维丛版广义非交换李代数',
    'PFB-GNLA',
)

NOTE = '#### ***注：“O3理论/O3元数学理论/主纤维丛版广义非交换李代数(PFB-GNLA)”相关理论参见： [作者（GaoZheng）网盘分享](https://drive.google.com/drive/folders/1lrgVtvhEq8cNal0Aa0AjeCNQaRA8WERu?usp=sharing) 或 [作者（GaoZheng）开源项目](https://github.com/CTaiDeng/open_meta_mathematical_theory) 或 [作者（GaoZheng）主页](https://mymetamathematics.blogspot.com)，欢迎访问！***'

DATE_LINE_RE = re.compile(r'^\s*-\s*日期：\d{4}-\d{2}-\d{2}\s*$', re.M)


def _to_lf(s: str) -> str:
    # 统一换行并清理潜在的 BOM 字符
    return s.replace('\r\n', '\n').replace('\r', '\n').replace('\ufeff', '')


def _should_inject(text: str) -> bool:
    if NOTE in text:
        return False
    return any(k in text for k in KEYWORDS)


def _insert_note(text: str) -> str:
    text_lf = _to_lf(text)
    lines = text_lf.split('\n')
    # 寻找“日期：”行，或回退到首个 H1 标题后一行
    idx_date = next((i for i, ln in enumerate(lines) if DATE_LINE_RE.match(ln.strip())), None)
    if idx_date is not None:
        insert_at = idx_date + 1
    else:
        idx_h1 = next((i for i, ln in enumerate(lines) if ln.lstrip().startswith('# ')), None)
        insert_at = (idx_h1 + 1) if idx_h1 is not None else 0
    lines.insert(insert_at, NOTE)
    return '\n'.join(lines)


def process_file(path: Path) -> bool:
    try:
        raw = path.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        raw = path.read_text(encoding='utf-8')
    if not _should_inject(raw):
        return False
    new_text = _insert_note(raw)
    if new_text == raw:
        return False
    # 写回 UTF-8（BOM）+ LF
    path.write_text(_to_lf(new_text), encoding='utf-8-sig')
    print(f"[insert_o3_note] injected: {path}")
    return True


def main(argv: List[str]) -> int:
    if len(argv) <= 1:
        print('[insert_o3_note] 无文件参数，跳过（需显式传入 docs/*.md 路径）')
        return 0
    files = require_explicit_doc_paths(argv[1:])
    # 跳过只读引用目录（如 docs/kernel_reference）
    repo_root = Path(__file__).resolve().parents[1]
    skip_paths = load_skip_paths(repo_root)
    changed = 0
    for p in files:
        if is_under(p, skip_paths):
            continue
        try:
            if process_file(p):
                changed += 1
        except Exception as e:
            print(f"[insert_o3_note] error: {p}: {e}")
    print(f"[insert_o3_note] done, changed={changed}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
