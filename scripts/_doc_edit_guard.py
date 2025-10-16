#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

"""
文档修改守卫：强制要求“项目相对路径”方式显式指定待处理文件，
防止对 docs 下文章的误改或越权修改。

允许的前缀（顶层、非递归判断，可在子目录内匹配）：
- docs/
- my_docs/project_docs/
- my_project/gmx_split_20250924_011827/docs/

在其他脚本中使用：
    from _doc_edit_guard import require_explicit_doc_paths
    files = require_explicit_doc_paths(sys.argv[1:])
    # 返回已规范化且去重后的 Path 列表；若校验失败会直接退出。
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
            if (
                pref.resolve() == rp.resolve()
                or pref.resolve() in rp.parents
                or rp.parent.resolve() == pref.resolve()
            ):
                return True
        except Exception:
            # 当 resolve 失败时，回退为字符串前缀判断
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
        print('[doc-edit-guard] 拒绝修改：未显式给出项目相对路径示例，如 docs/1234567890_标题.md', file=sys.stderr)
        sys.exit(2)

    normed: List[Path] = []
    for p in paths:
        if p.is_dir():
            print(f'[doc-edit-guard] 拒绝修改：传入目录而非具体文件：{p}', file=sys.stderr)
            sys.exit(2)
        if not _is_under_allowed(p):
            print(f'[doc-edit-guard] 拒绝修改：文件不在知识库允许路径内：{p}', file=sys.stderr)
            sys.exit(2)
        # 禁止修改知识库顶层的 LICENSE.md（只读）
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        docs_root = ROOT / 'docs'
        try:
            if rp.name.lower() == 'license.md' and (rp.parent.resolve() == docs_root.resolve()):
                print(f'[doc-edit-guard] 拒绝修改：禁止改动知识库许可文件：{p}', file=sys.stderr)
                sys.exit(2)
        except Exception:
            pass
        if not p.exists():
            print(f'[doc-edit-guard] 警告：文件不存在，跳过：{p}', file=sys.stderr)
            continue
        if p.suffix.lower() != '.md':
            print(f'[doc-edit-guard] 拒绝修改：仅允许 Markdown（.md）：{p}', file=sys.stderr)
            sys.exit(2)
        normed.append(p)
    if not normed:
        print('[doc-edit-guard] 无可处理的目标文件：均不存在或被过滤', file=sys.stderr)
        sys.exit(2)
    # 去重
    uniq: List[Path] = []
    seen: set[str] = set()
    for p in normed:
        try:
            s = str(p.resolve())
        except Exception:
            s = str(p)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq

