#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Doc edit guard: 强制要求以“项目相对路径”的方式显式指明目标文件，方可改动 docs 下的文章。

允许的前缀（仅顶层，不递归目录本身；是否递归由调用方决定）：
- docs/
- my_docs/project_docs/
- my_project/gmx_split_20250924_011827/docs/

用法（在各脚本中）：
    from _doc_edit_guard import require_explicit_doc_paths
    files = require_explicit_doc_paths(sys.argv[1:])
    # 返回经规范化、去重、存在性校验后的 Path 列表（相对/绝对输入均可）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_PREFIXES = [
    ROOT / 'docs',
    ROOT / 'my_docs' / 'project_docs',
    ROOT / 'my_project' / 'gmx_split_20250924_011827' / 'docs',
]


def _is_under_allowed(p: Path) -> bool:
    try:
        rp = p.resolve()
    except Exception:
        rp = p
    for pref in ALLOWED_PREFIXES:
        try:
            if pref.resolve() in rp.parents or rp.parent.resolve() == pref.resolve():
                return True
        except Exception:
            # 若 resolve 失败，退化为字符串前缀判断
            if str(p).replace('\\', '/').startswith(str(pref).replace('\\', '/') + '/'):
                return True
    return False


def require_explicit_doc_paths(args: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for a in args:
        if not a:
            continue
        p = (ROOT / a) if not os.path.isabs(a) else Path(a)
        paths.append(p)
    if not paths:
        print('[doc-edit-guard] 拒绝修改：未显式给出项目相对路径（示例：docs/1234567890_标题.md）', file=sys.stderr)
        sys.exit(2)

    normed: List[Path] = []
    for p in paths:
        if p.is_dir():
            print(f'[doc-edit-guard] 拒绝修改：给出了目录而非具体文件：{p}', file=sys.stderr)
            sys.exit(2)
        if not _is_under_allowed(p):
            print(f'[doc-edit-guard] 拒绝修改：不在允许的知识库路径内：{p}', file=sys.stderr)
            sys.exit(2)
        if not p.exists():
            print(f'[doc-edit-guard] 警告：文件不存在（忽略）：{p}', file=sys.stderr)
            continue
        if p.suffix.lower() != '.md':
            print(f'[doc-edit-guard] 拒绝修改：仅允许 Markdown 文档（.md）：{p}', file=sys.stderr)
            sys.exit(2)
        normed.append(p)
    if not normed:
        print('[doc-edit-guard] 无可处理的目标文件（均不存在或被忽略）', file=sys.stderr)
        sys.exit(2)
    # 去重
    uniq = []
    seen = set()
    for p in normed:
        s = str(p.resolve())
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq

