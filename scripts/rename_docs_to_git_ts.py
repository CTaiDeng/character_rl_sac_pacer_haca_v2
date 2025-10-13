#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

"""
重写文档文件名前缀为“文件创建时间（秒）”，并在秒级冲突时按标题名降序依次向前一秒分配。

范围（仅顶层，不递归）：
- docs/
- my_docs/project_docs/
- my_project/gmx_split_20250924_011827/docs/

规则：
- 仅处理匹配 "^\d+_.*\.md$" 的 Markdown 文件。
- 目标秒时间戳取自文件创建时间（Windows 下 st_ctime；若获取失败回退到 mtime）。
- 一旦创建确定后不再变更；仅当同秒出现多个文件时，组内按标题名（去前缀与扩展名）降序排序，依次分配 ts, ts-1, ts-2, ...。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRS = [
    ROOT / 'docs',
    ROOT / 'my_docs' / 'project_docs',
    ROOT / 'my_project' / 'gmx_split_20250924_011827' / 'docs',
]

NAME_RE = re.compile(r'^(\d+)_([\s\S]+)\.md$', re.IGNORECASE)


def safe_print(*args):
    try:
        print(*args)
    except UnicodeEncodeError:
        try:
            msg = ' '.join(str(a) for a in args)
            sys.stdout.write(msg.encode('ascii', 'ignore').decode('ascii', 'ignore') + '\n')
        except Exception:
            pass


def creation_epoch_seconds(path: Path) -> Optional[int]:
    try:
        return int(path.stat().st_ctime)
    except Exception:
        try:
            return int(path.stat().st_mtime)
        except Exception:
            return None


def allocate_unique_ts(desired_ts: int, used: Set[int]) -> int:
    ts = desired_ts
    while ts in used:
        ts -= 1
    used.add(ts)
    return ts


def plan_new_paths(files: List[Path]) -> Dict[Path, Path]:
    # 收集 (path, title, desired_ts)
    entries: List[Tuple[Path, str, int]] = []
    for p in files:
        m = NAME_RE.match(p.name)
        if not m:
            continue
        title = m.group(2)  # 不含 .md
        ts = creation_epoch_seconds(p)
        if ts is None:
            try:
                ts = int(m.group(1))
            except Exception:
                continue
        entries.append((p, title, ts))

    # 确定性排序：先按标题降序，再按 ts 升序（稳定排序实现“同秒内按标题降序”）
    entries.sort(key=lambda x: x[1], reverse=True)
    entries.sort(key=lambda x: x[2])

    used_ts: Set[int] = set()
    mapping: Dict[Path, Path] = {}
    for p, title, desired in entries:
        final_ts = allocate_unique_ts(desired, used_ts)
        new_name = f"{final_ts}_{title}.md"
        mapping[p] = p.with_name(new_name)
    return mapping


def perform_renames(mapping: Dict[Path, Path]) -> Tuple[int, int]:
    changed = 0
    skipped = 0
    pending: Dict[Path, Path] = {src: dst for src, dst in mapping.items() if src.name != dst.name}
    if not pending:
        return changed, skipped

    while pending:
        progressed = False
        for src, dst in list(pending.items()):
            if not dst.exists():
                safe_print(f"[MOVE] {src.name} -> {dst.name}")
                try:
                    os.replace(src, dst)
                    changed += 1
                except Exception as e:
                    safe_print(f"[ERROR] move failed: {src.name} -> {dst.name}: {e}")
                pending.pop(src, None)
                progressed = True
        if progressed:
            continue

        # break cycles by moving one blocking file to a temp name
        src, dst = next(iter(pending.items()))
        tmp = src.with_name(f"__tmp__{uuid4().hex}__{src.name}")
        safe_print(f"[TMP] {src.name} -> {tmp.name}")
        try:
            os.replace(src, tmp)
        except Exception as e:
            safe_print(f"[ERROR] tmp move failed: {src.name} -> {tmp.name}: {e}")
            pending.pop(src, None)
            skipped += 1
            continue
        pending.pop(src, None)
        pending[tmp] = dst

    return changed, skipped


def main() -> int:
    total_changed = 0
    total_skipped = 0
    for d in TARGET_DIRS:
        if not (d.exists() and d.is_dir()):
            continue
        files = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == '.md'])
        mapping = plan_new_paths(files)
        changed, skipped = perform_renames(mapping)
        total_changed += changed
        total_skipped += skipped
    safe_print(f"[rename_docs_to_git_ts] done: changed={total_changed} skipped={total_skipped}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

