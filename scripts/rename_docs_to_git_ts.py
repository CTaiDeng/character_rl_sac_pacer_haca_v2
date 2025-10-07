#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 docs 目录下形如 <ts>_*.md 的文档前缀重写为该文件在 Git 中的首次入库时间（秒）。

命名规范：
- 前缀（秒级时间戳）必须全局唯一，视作“文章ID”。
- 若同时入库导致冲突（相同秒），则对后续文件按“后退 1 秒”逐次分配：ts-1、ts-2、…
- 仅处理顶层 docs 中满足 ^\d+_.*\.md$ 的文件；不递归，不进入只读引用目录。
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
from uuid import uuid4


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


def load_existing_used_ts(files: List[Path]) -> Set[int]:
    used: Set[int] = set()
    for p in files:
        m = NAME_RE.match(p.name)
        if not m:
            continue
        try:
            used.add(int(m.group(1)))
        except Exception:
            pass
    return used


def allocate_unique_ts(desired_ts: int, used: Set[int]) -> int:
    ts = desired_ts
    while ts in used:
        ts -= 1
    used.add(ts)
    return ts


def plan_new_paths(files: List[Path]) -> Dict[Path, Path]:
    entries: List[Tuple[Path, str, int]] = []  # (path, rest, desired_ts)
    for p in files:
        m = NAME_RE.match(p.name)
        if not m:
            continue
        rest = m.group(2)
        ts = git_added_epoch_seconds(p)
        if ts is None:
            try:
                ts = int(m.group(1))
            except Exception:
                continue
        entries.append((p, rest, ts))

    # deterministic ordering: ts asc, filename asc
    entries.sort(key=lambda x: (x[2], x[0].name))

    # 从空集开始分配，确保“同秒入库”时首个条目保留原秒，其余依次后退
    used_ts: Set[int] = set()
    mapping: Dict[Path, Path] = {}
    for p, rest, desired in entries:
        final_ts = allocate_unique_ts(desired, used_ts)
        new_name = f"{final_ts}_{rest}"
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
                    safe_print(f"[ERROR] 重命名失败: {src.name} -> {dst.name}: {e}")
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
            safe_print(f"[ERROR] 临时重命名失败: {src.name} -> {tmp.name}: {e}")
            pending.pop(src, None)
            skipped += 1
            continue
        pending.pop(src, None)
        pending[tmp] = dst

    return changed, skipped


def main() -> int:
    if not DOCS_DIR.is_dir():
        safe_print(f"[rename_docs_to_git_ts] 未找到目录：{DOCS_DIR}")
        return 1
    files = sorted([p for p in DOCS_DIR.iterdir() if p.is_file() and p.suffix.lower() == '.md'])
    mapping = plan_new_paths(files)
    changed, skipped = perform_renames(mapping)
    safe_print(f"[rename_docs_to_git_ts] 重命名完成：{changed} 个，跳过 {skipped} 个。")
    return 0


if __name__ == '__main__':
    sys.exit(main())
