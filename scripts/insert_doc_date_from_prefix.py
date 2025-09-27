#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
为 docs 目录中带有“时间戳前缀_<名称>.md”的文档，在标题下方插入/更新日期行：
  日期：YYYY-MM-DD

规则：
- 仅处理文件名匹配 ^\d+_.*\.md$ 的 Markdown；
- 日期来源于文件名前缀（Unix epoch 秒），转换为本地时区日期 YYYY-MM-DD；
- 插入位置为文档的首个标题行（以 # 开头）之后；
- 若紧随标题已有“日期：”行，则就地更新为新日期；
- 读写编码：UTF-8（BOM）。
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, List


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / 'docs'

PREFIX_RE = re.compile(r'^(\d+)_.*\.md$', re.IGNORECASE)
TITLE_RE = re.compile(r'^\s*#{1,6}\s+')
DATE_RE = re.compile(r'^\s*日期：')


def read_text(path: Path) -> Tuple[str, str]:
    data = path.read_bytes()
    nl = '\r\n' if b'\r\n' in data else '\n'
    text = data.decode('utf-8-sig', errors='replace')
    return text, nl


def write_text(path: Path, text: str, nl: str) -> None:
    text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\n', nl)
    # Path.write_text 不支持 newline 参数，改用显式打开
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write(text)


def ensure_date_after_title(text: str, date_str: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    # 找到首个标题行
    title_idx = None
    for i, ln in enumerate(lines):
        if TITLE_RE.match(ln.strip()):
            title_idx = i
            break
    if title_idx is None:
        # 无标题，插入在文件头部
        return '\n'.join([f'日期：{date_str}', ''] + lines)

    insert_idx = title_idx + 1
    # 跳过标题后的空行
    while insert_idx < len(lines) and lines[insert_idx].strip() == '':
        insert_idx += 1

    # 如果已有日期行则更新
    if insert_idx < len(lines) and DATE_RE.match(lines[insert_idx]):
        lines[insert_idx] = f'日期：{date_str}'
        return '\n'.join(lines)

    # 否则在标题后插入“日期：”与空行
    new_lines: List[str] = []
    new_lines.extend(lines[:title_idx+1])
    new_lines.append(f'日期：{date_str}')
    new_lines.append('')
    new_lines.extend(lines[title_idx+1:])
    return '\n'.join(new_lines)


def main() -> int:
    if not DOCS_DIR.is_dir():
        print(f'[insert_doc_date_from_prefix] 未找到目录：{DOCS_DIR}')
        return 1
    changed = 0
    for p in sorted(DOCS_DIR.iterdir()):
        if not (p.is_file() and p.suffix.lower() == '.md'):
            continue
        m = PREFIX_RE.match(p.name)
        if not m:
            continue
        try:
            ts = int(m.group(1))
            dt = datetime.fromtimestamp(ts)
            date_str = dt.strftime('%Y-%m-%d')
        except Exception:
            continue
        text, nl = read_text(p)
        new_text = ensure_date_after_title(text, date_str)
        if new_text != text:
            write_text(p, new_text, nl)
            changed += 1
    print(f'[insert_doc_date_from_prefix] 已更新 {changed} 个文件')
    return 0


if __name__ == '__main__':
    sys.exit(main())
