#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

"""
根据文内标题与“日期：YYYY-MM-DD”规范化文首结构，并将文件重命名为
"<秒时间戳>_<标题>.md"。

范围（顶层、非递归）：
- docs/
- my_docs/project_docs/（不进入子目录；kernel_reference 天然排除）
- my_project/gmx_split_20250924_011827/docs/

文首规则：
- 第一行：H1 标题 "# <标题>"
- 空一行
- 三行元信息（相邻无空行）：
  - "- 作者：GaoZheng"
  - "- 日期：YYYY-MM-DD"
  - "- 版本：vX.Y.Z"（缺失则初始化为 v1.0.0，已存在则保留）
- 元信息后空一行
- 其后内容（可包含 O3 注释、摘要等）不受本脚本约束

时间戳策略：
- 文件名中的“秒时间戳”由“文内日期”的本地 00:00:00 换算而来；
- 同日文章按标题名（去前后缀）降序排序，从当日第一秒起逐秒递增分配：base_ts+1, base_ts+2, ...

本脚本只对传入的明确文件列表进行处理（由 _doc_edit_guard 守卫）。
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

from _doc_edit_guard import require_explicit_doc_paths


ROOT = Path(__file__).resolve().parents[1]

NAME_RE = re.compile(r'^(\d+)_([\s\S]+)\.md$', re.IGNORECASE)
TITLE_RE = re.compile(r'^\s*#\s+(.+?)\s*$')
AUTHOR_RE = re.compile(r'^\s*-\s*作者：\s*(.+?)\s*$')
DATE_RE = re.compile(r'^\s*-\s*日期：\s*(\d{4}-\d{2}-\d{2})\s*$')
VERSION_RE = re.compile(r'^\s*-\s*版本：\s*(v\d+\.\d+\.\d+)\s*$', re.IGNORECASE)


def to_lf(s: str) -> str:
    return s.replace('\r\n', '\n').replace('\r', '\n')


def read_text(path: Path) -> Tuple[str, str]:
    b = path.read_bytes()
    nl = '\r\n' if b'\r\n' in b else '\n'
    try:
        txt = b.decode('utf-8-sig')
    except Exception:
        try:
            txt = b.decode('gbk')
        except Exception:
            txt = b.decode('utf-8', errors='replace')
    # 清理潜在的重复 BOM（U+FEFF）
    if txt.startswith('\ufeff'):
        txt = txt.lstrip('\ufeff')
    return txt, nl


def write_text(path: Path, text: str) -> None:
    # 写回为 UTF-8（BOM）+ LF
    data = ('\ufeff' + to_lf(text)).encode('utf-8')
    path.write_bytes(data)


def sanitize_title_for_filename(title: str) -> str:
    # Windows 非法字符替换：\\ / : * ? " < > |
    return re.sub(r'[\\/:*?\"<>|]', '－', title).strip()


@dataclass
class DocInfo:
    path: Path
    title: str
    date_str: str  # YYYY-MM-DD
    base_ts: int   # epoch seconds at local midnight


def parse_title(lines: List[str]) -> Tuple[Optional[str], Optional[int]]:
    for i, ln in enumerate(lines[:50]):
        m = TITLE_RE.match(ln)
        if m:
            return m.group(1), i
    return None, None


def find_meta_indices(lines: List[str], start_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    author_idx = None
    date_idx = None
    version_idx = None
    for i in range(start_idx + 1, min(len(lines), start_idx + 40)):
        if author_idx is None and AUTHOR_RE.match(lines[i]):
            author_idx = i
        if date_idx is None and DATE_RE.match(lines[i]):
            date_idx = i
        if version_idx is None and VERSION_RE.match(lines[i]):
            version_idx = i
        if author_idx is not None and date_idx is not None and version_idx is not None:
            break
    return author_idx, date_idx, version_idx


def ensure_meta_and_collect(path: Path, default_author: str = 'GaoZheng') -> Tuple[Optional[DocInfo], bool]:
    """Return DocInfo and whether content was modified."""
    text, _nl = read_text(path)
    lines = to_lf(text).split('\n')
    title, h1_idx = parse_title(lines)
    changed = False
    if title is None:
        # 未找到 H1 标题，跳过
        return None, False
    # Ensure blank line after H1
    insert_pos = h1_idx + 1
    if insert_pos >= len(lines) or lines[insert_pos].strip() != '':
        lines.insert(insert_pos, '')
        changed = True
    # Recompute after potential insert
    author_idx, date_idx, version_idx = find_meta_indices(lines, h1_idx)
    # Determine desired meta values
    # Author
    author_line = lines[author_idx].strip() if author_idx is not None else f'- 作者：{default_author}'
    # Date
    if date_idx is not None:
        mdate0 = DATE_RE.match(lines[date_idx])
        date_val = mdate0.group(1) if mdate0 else datetime.now().strftime('%Y-%m-%d')
    else:
        m = NAME_RE.match(path.name)
        if m:
            try:
                ts = int(m.group(1))
                dt = datetime.fromtimestamp(ts)
                date_val = dt.strftime('%Y-%m-%d')
            except Exception:
                date_val = datetime.now().strftime('%Y-%m-%d')
        else:
            date_val = datetime.now().strftime('%Y-%m-%d')
    date_line = f'- 日期：{date_val}'
    # Version
    if version_idx is not None:
        version_line = lines[version_idx].strip()
    else:
        version_line = '- 版本：v1.0.0'

    # Normalize meta block placement:
    # Expected positions:
    #   h1_idx
    #   h1_idx+1: ''
    #   h1_idx+2: author
    #   h1_idx+3: date
    #   h1_idx+4: version
    #   h1_idx+5: ''
    expected_block = [author_line, date_line, version_line]
    # Ensure minimum length
    while len(lines) < h1_idx + 6:
        lines.append('')
    # Write block
    if lines[h1_idx + 2].strip() != expected_block[0]:
        lines[h1_idx + 2] = expected_block[0]
        changed = True
    if lines[h1_idx + 3].strip() != expected_block[1]:
        lines[h1_idx + 3] = expected_block[1]
        changed = True
    if lines[h1_idx + 4].strip() != expected_block[2]:
        lines[h1_idx + 4] = expected_block[2]
        changed = True
    # Ensure exactly one blank line after version
    if h1_idx + 5 >= len(lines) or lines[h1_idx + 5].strip() != '':
        lines.insert(h1_idx + 5, '')
        changed = True
    # Remove any extra blank line between date and version
    if lines[h1_idx + 4].strip() == '' and VERSION_RE.match(lines[h1_idx + 5] if h1_idx + 5 < len(lines) else ''):
        # unlikely path, but keep guard
        del lines[h1_idx + 4]
        changed = True

    # Clean up duplicated meta lines that might exist elsewhere near the header
    # Scan a small window after the normalized block and drop duplicates
    end_scan = min(len(lines), h1_idx + 20)
    k = h1_idx + 6
    while k < end_scan:
        ln = lines[k]
        if AUTHOR_RE.match(ln) or DATE_RE.match(ln) or VERSION_RE.match(ln):
            del lines[k]
            end_scan -= 1
            changed = True
            continue
        k += 1

    # Re-scan for date value
    _, date_idx, _ = find_meta_indices(lines, h1_idx)
    mdate = DATE_RE.match(lines[date_idx]) if date_idx is not None else None
    if not mdate:
        return None, False
    date_str = mdate.group(1)

    # Build DocInfo
    base_dt = datetime.strptime(date_str, '%Y-%m-%d')
    # Local midnight epoch seconds
    base_ts = int(base_dt.timestamp())
    info = DocInfo(path=path, title=title, date_str=date_str, base_ts=base_ts)

    # 规范“摘要”标题为二级标题：## 摘要：
    # 仅在文首范围内查找一次，兼容旧样式 ###/####，并补全中文冒号。
    normalized_summary = False
    for i, ln in enumerate(lines[:200]):
        if re.match(r'^\s*#{3,}\s*摘要\s*[:：]?\s*$', ln):
            lines[i] = '## 摘要：'
            changed = True
            normalized_summary = True
            break
        if re.match(r'^\s*##\s*摘要\s*$', ln):
            lines[i] = '## 摘要：'
            changed = True
            normalized_summary = True
            break
        if re.match(r'^\s*##\s*摘要\s*[:：]\s*$', ln):
            if ln.strip() != '## 摘要：':
                lines[i] = '## 摘要：'
                changed = True
            normalized_summary = True
            break

    if changed:
        write_text(path, '\n'.join(lines))
    return info, changed


def allocate_unique_ts(entries: List[DocInfo]) -> Dict[Path, int]:
    # Group by base_ts
    by_ts: Dict[int, List[DocInfo]] = {}
    for e in entries:
        by_ts.setdefault(e.base_ts, []).append(e)
    # Within each group, sort by title descending
    for ts, arr in by_ts.items():
        arr.sort(key=lambda x: x.title, reverse=True)
    # Assign unique ts per directory — from first second, increasing
    assigned: Dict[Path, int] = {}
    used: set[int] = set()
    for ts in sorted(by_ts.keys()):
        cursor = ts + 1  # start from the first second of the day
        for e in by_ts[ts]:
            while cursor in used:
                cursor += 1
            used.add(cursor)
            assigned[e.path] = cursor
            cursor += 1
    return assigned


def plan_renames(dir_entries: List[DocInfo]) -> Dict[Path, Path]:
    if not dir_entries:
        return {}
    assigned = allocate_unique_ts(dir_entries)
    mapping: Dict[Path, Path] = {}
    for e in dir_entries:
        final_ts = assigned[e.path]
        safe_title = sanitize_title_for_filename(e.title)
        new_name = f'{final_ts}_{safe_title}.md'
        dst = e.path.with_name(new_name)
        if dst.name != e.path.name:
            mapping[e.path] = dst
    return mapping


def perform_renames(mapping: Dict[Path, Path]) -> int:
    changed = 0
    # Resolve potential name collisions by staging temp names
    pending = {src: dst for src, dst in mapping.items() if src.name != dst.name}
    used: set[str] = set()
    for dst in pending.values():
        used.add(dst.name)
    while pending:
        progressed = False
        for src, dst in list(pending.items()):
            if not dst.exists():
                try:
                    os.replace(src, dst)
                    changed += 1
                except Exception:
                    # try temp indirection
                    tmp = src.with_name(f'__tmp__{src.name}')
                    try:
                        os.replace(src, tmp)
                        os.replace(tmp, dst)
                        changed += 1
                    except Exception:
                        pass
                pending.pop(src, None)
                progressed = True
        if progressed:
            continue
        # break cycle
        src, dst = next(iter(pending.items()))
        tmp = src.with_name(f'__tmp__{src.name}')
        try:
            os.replace(src, tmp)
            os.replace(tmp, dst)
            changed += 1
        except Exception:
            pending.pop(src, None)
            continue
        pending.pop(src, None)
    return changed


def process_files(files: List[Path]) -> Tuple[int, int]:
    infos: List[DocInfo] = []
    content_updates = 0
    for p in files:
        info, changed = ensure_meta_and_collect(p)
        if info is None:
            continue
        infos.append(info)
        if changed:
            content_updates += 1
    mapping = plan_renames(infos)
    renamed = perform_renames(mapping)
    return content_updates, renamed


def main() -> int:
    ap = argparse.ArgumentParser(description='根据文内日期规范化文首并重命名（需显式传入文件列表）')
    ap.add_argument('files', nargs='+', help='项目相对路径，如 docs/1234567890_标题.md')
    args = ap.parse_args()
    files = require_explicit_doc_paths(args.files)
    u, r = process_files(files)
    print(f'[ensure_docs_style_from_date] updated={u} renamed={r}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
