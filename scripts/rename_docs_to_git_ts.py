#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 docs 目录下以“数字_”开头的 Markdown 文件重命名为：
  <git入库时间戳sec>_<原文件名去掉旧前缀>

- 入库时间：git log --diff-filter=A --follow --format=%at -n 1 <file>
- 仅处理 ^\d+_.*\.md$；其余跳过。
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


def safe_print(*args):
    try:
        print(*args)
    except UnicodeEncodeError:
        try:
            msg = ' '.join(str(a) for a in args)
            sys.stdout.write(msg.encode('ascii', 'ignore').decode('ascii', 'ignore') + '\n')
        except Exception:
            pass


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / 'docs'
NAME_RE = re.compile(r'^(\d+)_([\s\S]+\.md)$', re.IGNORECASE)


def git_added_epoch_seconds(path: Path) -> Optional[int]:
    try:
        out = subprocess.check_output(
            ['git', 'log', '--diff-filter=A', '--follow', '--format=%at', '-n', '1', str(path)],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
        )
        s = out.decode('utf-8', errors='ignore').strip()
        if not s:
            return None
        return int(s.splitlines()[0].strip())
    except Exception:
        return None


def next_available_name(target: Path) -> Path:
    if not target.exists():
        return target
    base = target.stem
    suffix = target.suffix
    parent = target.parent
    for ch in 'abcdefghijklmnopqrstuvwxyz':
        cand = parent / f"{base}{ch}{suffix}"
        if not cand.exists():
            return cand
    i = 2
    while True:
        cand = parent / f"{base}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def main() -> int:
    if not DOCS_DIR.is_dir():
        safe_print(f"[rename_docs_to_git_ts] 未找到目录：{DOCS_DIR}")
        return 1
    files = sorted([p for p in DOCS_DIR.iterdir() if p.is_file() and p.suffix.lower() == '.md'])
    changed = 0
    skipped = 0
    for p in files:
        m = NAME_RE.match(p.name)
        if not m:
            skipped += 1
            continue
        old_prefix, rest = m.group(1), m.group(2)
        ts = git_added_epoch_seconds(p)
        if ts is None:
            safe_print(f"[WARN] 获取入库时间失败，跳过：{p.name}")
            skipped += 1
            continue
        new_name = f"{ts}_{rest}"
        if new_name == p.name:
            continue
        target = p.with_name(new_name)
        target = next_available_name(target)
        safe_print(f"[MOVE] {p.name} -> {target.name}")
        try:
            os.replace(p, target)
            changed += 1
        except Exception as e:
            safe_print(f"[ERROR] 重命名失败：{p.name} -> {target.name}: {e}")
    safe_print(f"[rename_docs_to_git_ts] 重命名完成：{changed} 个，跳过 {skipped} 个。")
    return 0


if __name__ == '__main__':
    sys.exit(main())

