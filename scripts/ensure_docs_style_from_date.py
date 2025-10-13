#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

"""
Ensure docs article style based on in-document date and title, and rename files to
"<epoch_seconds>_<title>.md" accordingly. Duplicate timestamps resolve by title
descending, assigning ts, ts-1, ts-2, ... within the same directory.

Scope (top-level only, non-recursive):
- docs/
- my_docs/project_docs/ (exclude subdirectories; kernel_reference is naturally excluded)
- my_project/gmx_split_20250924_011827/docs/

Style rules inferred from examples:
- First line is H1: "# <标题>"
- Then a blank line
- Then meta lines:
  - "- 作者：GaoZheng"
  - "- 日期：YYYY-MM-DD"
- Then a blank line
- Then optional O3 citation note block
- Then a blank line
- Then section heading "### 摘要：" and summary

Timestamp rule:
- The filename timestamp MUST be derived from the in-document date (local midnight, seconds).
- If many files share the same date (thus same base ts), sort by title descending
  and assign unique seconds ts, ts-1, ts-2, ...

This tool can fix content (insert missing author/date lines) and perform renames.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from _doc_edit_guard import require_explicit_doc_paths


ROOT = Path(__file__).resolve().parents[1]

NAME_RE = re.compile(r'^(\d+)_([\s\S]+)\.md$', re.IGNORECASE)
TITLE_RE = re.compile(r'^\s*#\s+(.+?)\s*$')
AUTHOR_RE = re.compile(r'^\s*-\s*作者：\s*(.+?)\s*$')
DATE_RE = re.compile(r'^\s*-\s*日期：\s*(\d{4}-\d{2}-\d{2})\s*$')


def to_lf(s: str) -> str:
    return s.replace('\r\n', '\n').replace('\r', '\n')


def read_text(path: Path) -> Tuple[str, str]:
    b = path.read_bytes()
    nl = '\r\n' if b'\r\n' in b else '\n'
    try:
        txt = b.decode('utf-8-sig')
    except Exception:
        txt = b.decode('gbk', errors='replace')
    return txt, nl


def write_text(path: Path, text: str) -> None:
    data = ('\ufeff' + to_lf(text)).encode('utf-8')
    path.write_bytes(data)


def sanitize_title_for_filename(title: str) -> str:
    # Replace characters invalid on Windows filenames: \ / : * ? " < > |
    return re.sub(r'[\\/:*?"<>|]', '·', title).strip()


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


def find_meta_indices(lines: List[str], start_idx: int) -> Tuple[Optional[int], Optional[int]]:
    author_idx = None
    date_idx = None
    for i in range(start_idx + 1, min(len(lines), start_idx + 20)):
        m1 = AUTHOR_RE.match(lines[i])
        m2 = DATE_RE.match(lines[i])
        if m1 and author_idx is None:
            author_idx = i
        if m2 and date_idx is None:
            date_idx = i
        if author_idx is not None and date_idx is not None:
            break
    return author_idx, date_idx


def ensure_meta_and_collect(path: Path, default_author: str = 'GaoZheng') -> Tuple[Optional[DocInfo], bool]:
    """Return DocInfo and whether content was modified."""
    text, _nl = read_text(path)
    lines = to_lf(text).split('\n')
    title, h1_idx = parse_title(lines)
    changed = False
    if title is None:
        # No title — cannot process
        return None, False
    # Ensure blank line after H1
    insert_pos = h1_idx + 1
    if insert_pos >= len(lines) or lines[insert_pos].strip() != '':
        lines.insert(insert_pos, '')
        changed = True
    # Recompute after potential insert
    author_idx, date_idx = find_meta_indices(lines, h1_idx)
    # Ensure author line
    if author_idx is None:
        lines.insert(h1_idx + 2, f'- 作者：{default_author}')
        changed = True
        # Shift indices
        author_idx = h1_idx + 2
        date_idx = None  # force re-find
    # Ensure date line
    if date_idx is None:
        # Try to derive from filename prefix
        m = NAME_RE.match(path.name)
        if m:
            try:
                ts = int(m.group(1))
                dt = datetime.fromtimestamp(ts)
                date_str = dt.strftime('%Y-%m-%d')
            except Exception:
                date_str = datetime.now().strftime('%Y-%m-%d')
        else:
            date_str = datetime.now().strftime('%Y-%m-%d')
        # place after author line (which we ensured exists)
        # Ensure a blank line between meta and rest
        lines.insert(author_idx + 1, f'- 日期：{date_str}')
        lines.insert(author_idx + 2, '')
        changed = True
    else:
        # Ensure a blank line after date line
        if date_idx + 1 >= len(lines) or lines[date_idx + 1].strip() != '':
            lines.insert(date_idx + 1, '')
            changed = True

    # Re-scan for date value
    author_idx, date_idx = find_meta_indices(lines, h1_idx)
    mdate = DATE_RE.match(lines[date_idx]) if date_idx is not None else None
    if not mdate:
        return None, False
    date_str = mdate.group(1)

    # Build DocInfo
    base_dt = datetime.strptime(date_str, '%Y-%m-%d')
    # Local midnight epoch seconds
    base_ts = int(base_dt.timestamp())
    info = DocInfo(path=path, title=title, date_str=date_str, base_ts=base_ts)

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
    # Assign unique ts per directory
    assigned: Dict[Path, int] = {}
    used: set[int] = set()
    for ts in sorted(by_ts.keys()):
        for e in by_ts[ts]:
            cur = ts
            while cur in used:
                cur -= 1
            used.add(cur)
            assigned[e.path] = cur
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
    ap = argparse.ArgumentParser(description='Ensure docs style from in-document date（仅处理显式给出的文件路径）')
    ap.add_argument('files', nargs='+', help='项目相对路径，如 docs/1234567890_标题.md')
    args = ap.parse_args()
    files = require_explicit_doc_paths(args.files)
    u, r = process_files(files)
    print(f'[ensure_docs_style_from_date] updated={u} renamed={r}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
