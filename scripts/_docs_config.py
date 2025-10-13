#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program. If not, see https://www.gnu.org/licenses/.


"""
通用：读取 docs 处理白名单配置，并提供路径判断工具。

配置文件：scripts/docs_processing_config.json
示例：
{
  "skip_paths": [
    "docs/kernel_reference"
  ]
}
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List


def _norm(path: str) -> str:
    return os.path.abspath(path).replace('\\', '/').rstrip('/')


def load_skip_paths(repo_root: Path) -> List[Path]:
    cfg = repo_root / 'scripts' / 'docs_processing_config.json'
    if not cfg.is_file():
        return []
    try:
        data = json.loads(cfg.read_text(encoding='utf-8'))
    except Exception:
        return []
    items = data.get('skip_paths', []) or []
    out: List[Path] = []
    for it in items:
        if not isinstance(it, str) or not it.strip():
            continue
        p = (repo_root / it.replace('\\', '/')).absolute()
        out.append(p)
    return out


def is_under(path: Path, bases: List[Path]) -> bool:
    p = _norm(str(Path(path)))
    for b in bases:
        bb = _norm(str(Path(b)))
        if p == bb or p.startswith(bb + '/'):
            return True
    return False

