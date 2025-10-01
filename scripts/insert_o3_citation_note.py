#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在 docs/*.md 中检测关键词并在“日期：YYYY-MM-DD”下一行插入统一注释。
- 关键词任意出现："O3理论"、"O3元数学理论"、"主纤维丛版广义非交换李代数"、"PFB-GNLA"
- 注释文本：
  #### ***注：“O3理论/O3元数学理论/主纤维丛版广义非交换李代数(PFB-GNLA)”相关理论参见： [作者（GaoZheng）网盘分享](https://drive.google.com/drive/folders/1lrgVtvhEq8cNal0Aa0AjeCNQaRA8WERu?usp=sharing) 或 [作者（GaoZheng）主页](https://mymetamathematics.blogspot.com)***
- 仅在未存在该注释时插入；若未找到“日期：”行，则尝试在首个标题行后插入；仍未找到则插入文件首部。
- 读写编码使用 UTF-8（带 BOM），LF 行尾。

用法：
  python scripts/insert_o3_citation_note.py [docs]
"""

from __future__ import annotations
import sys
from pathlib import Path
import re
from _docs_config import load_skip_paths, is_under

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
SKIP_PATHS = load_skip_paths(ROOT)

KEYWORDS = (
    'O3理论',
    'O3元数学理论',
    '主纤维丛版广义非交换李代数',
    'PFB-GNLA',
)
NOTE = (
    '#### ***注：“O3理论/O3元数学理论/主纤维丛版广义非交换李代数(PFB-GNLA)”相关理论参见： '
    '[作者（GaoZheng）网盘分享](https://drive.google.com/drive/folders/1lrgVtvhEq8cNal0Aa0AjeCNQaRA8WERu?usp=sharing) '
    '或 [作者（GaoZheng）主页](https://mymetamathematics.blogspot.com)***'
)

DATE_RE = re.compile(r'^日期：\d{4}-\d{2}-\d{2}\s*$', re.M)


def should_inject(text: str) -> bool:
    if NOTE in text:
        return False
    return any(k in text for k in KEYWORDS)


def split_lines_keep(text: str) -> list[str]:
    # 标准化行为 LF
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text.split('\n')


def insert_note(text: str) -> str:
    lines = split_lines_keep(text)
    # 查找日期行
    idx_date = next((i for i, ln in enumerate(lines) if ln.strip().startswith('日期：')), None)
    if idx_date is not None:
        insert_at = idx_date + 1
    else:
        # 回退：找首个标题
        idx_h1 = next((i for i, ln in enumerate(lines) if ln.lstrip().startswith('# ')), None)
        insert_at = (idx_h1 + 1) if idx_h1 is not None else 0
    lines.insert(insert_at, NOTE)
    return '\n'.join(lines)


def process_file(path: Path) -> bool:
    try:
        raw = path.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        raw = path.read_text(encoding='utf-8')
    if not should_inject(raw):
        return False
    new_text = insert_note(raw)
    if new_text == raw:
        return False
    # 写回 UTF-8（BOM），LF
    path.write_text(new_text.replace('\r\n', '\n').replace('\r', '\n'), encoding='utf-8-sig')
    print(f"[insert_o3_note] injected: {path}")
    return True


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else DOCS
    if not base.exists():
        print(f"[insert_o3_note] skip, not found: {base}")
        return 0
    changed = 0
    for p in base.rglob('*.md'):
        # 仅处理 docs 目录下的小写 .md
        if p.is_file():
            # 按白名单跳过指定路径
            if is_under(p, SKIP_PATHS):
                continue
            try:
                if process_file(p):
                    changed += 1
            except Exception as e:
                print(f"[insert_o3_note] error: {p}: {e}")
    print(f"[insert_o3_note] done, changed={changed}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
