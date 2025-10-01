#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将仓库内 Markdown 文档中的 $\texttt{...}$ 行内片段批量转换为反引号 `...`，
仅在代码围栏外进行替换：
  - 匹配形如 $\texttt{a\_b}$ 的片段；
  - 将内容中的 \_ 还原为 _；
  - 输出为 `a_b`。

用法：
  python scripts/convert_texttt_to_backticks.py            # 全仓 .md
  python scripts/convert_texttt_to_backticks.py a.md b.md  # 指定文件
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple
from _docs_config import load_skip_paths, is_under

FENCE_RE = re.compile(r"^\s*(```|~~~)")
TEXTTT_RE = re.compile(r"\$\s*\\texttt\{([^}]+)\}\s*\$")


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SKIP_PATHS = load_skip_paths(Path(ROOT))


def read_text(path: str) -> Tuple[str, str]:
    with open(path, 'rb') as f:
        data = f.read()
    nl = '\r\n' if b'\r\n' in data else '\n'
    text = data.decode('utf-8-sig', errors='replace')
    return text, nl


def write_text(path: str, text: str, nl: str) -> None:
    text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\n', nl)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write(text)


def split_by_fences(text: str):
    lines = text.splitlines(keepends=True)
    segments = []
    buf = []
    in_code = False
    for ln in lines:
        if FENCE_RE.match(ln):
            if buf:
                segments.append((in_code, ''.join(buf)))
                buf = []
            segments.append((True, ln))
            in_code = not in_code
        else:
            buf.append(ln)
    if buf:
        segments.append((in_code, ''.join(buf)))
    return segments


def convert_segment(seg: str) -> str:
    def _repl(m: re.Match) -> str:
        inner = m.group(1)
        inner = inner.replace(r"\_", "_")
        return f"`{inner}`"

    return TEXTTT_RE.sub(_repl, seg)


def process_file(path: str) -> bool:
    text, nl = read_text(path)
    parts = split_by_fences(text)
    changed = False
    out_parts: List[str] = []
    for in_code, seg in parts:
        if in_code:
            out_parts.append(seg)
        else:
            new_seg = convert_segment(seg)
            if new_seg != seg:
                changed = True
            out_parts.append(new_seg)
    if changed:
        write_text(path, ''.join(out_parts), nl)
    return changed


def find_all_md(root: str) -> List[str]:
    out: List[str] = []
    skip = {'.git', '.idea', 'out', 'dist', 'build', '__pycache__'}
    for dp, dns, fns in os.walk(root):
        # 目录级跳过（内置目录 + 配置白名单）
        def _keep_dir(d: str) -> bool:
            if d in skip:
                return False
            absd = os.path.abspath(os.path.join(dp, d))
            return not is_under(Path(absd), SKIP_PATHS)

        dns[:] = [d for d in dns if _keep_dir(d)]
        for fn in fns:
            if fn.lower().endswith('.md'):
                p = os.path.join(dp, fn)
                if not is_under(Path(p), SKIP_PATHS):
                    out.append(p)
    return out


def main(argv: List[str]) -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    files = [os.path.abspath(p) for p in argv] if argv else find_all_md(root)
    changed = 0
    for p in files:
        if not os.path.isfile(p):
            continue
        # 单文件入口也按白名单跳过
        if is_under(Path(p), SKIP_PATHS):
            continue
        try:
            if process_file(p):
                changed += 1
        except Exception as e:
            print(f"[convert_texttt_to_backticks] 跳过 {p}: {e}")
    print(f"[convert_texttt_to_backticks] 已转换 {changed} 个 Markdown 文件")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
