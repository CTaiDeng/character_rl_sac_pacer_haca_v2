#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import argparse
from pathlib import Path
from _doc_edit_guard import require_explicit_doc_paths


def replace_in_file(p: Path, old: str, new: str) -> bool:
    try:
        data = p.read_bytes()
        text = data.decode('utf-8-sig', errors='replace')
    except Exception:
        return False
    if old not in text:
        return False
    text = text.replace(old, new)
    p.write_bytes(('\ufeff' + text.replace('\r\n', '\n').replace('\r', '\n')).encode('utf-8'))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description='Bulk replace copyright lines in explicit docs')
    ap.add_argument('files', nargs='+', help='项目相对路径，例如 docs/1234567890_标题.md')
    ap.add_argument('--old', default='Copyright (C) 2025 GaoZheng')
    ap.add_argument('--new', default='Copyright (C) 2025- GaoZheng')
    args = ap.parse_args()
    files = require_explicit_doc_paths(args.files)
    changed = 0
    for p in files:
        if replace_in_file(p, args.old, args.new):
            print(f'[bulk-replace] updated: {p}')
            changed += 1
    print(f'[bulk-replace] changed={changed} files')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

