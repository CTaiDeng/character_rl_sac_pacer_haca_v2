#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Markdown 正规化脚本（数学/代码格式实时审查）

规则：
- 将行间公式 \[ ... \] → $$ ... $$（忽略以双反斜杠开头的 \\[2pt] 等行内可选参数）。
- 将行内公式 \( ... \) → $ ... $。
- 将围栏外的行内代码 `code_like` → $\texttt{code\_like}$（对下划线转义）。
- 保护代码围栏 ``` 与 ~~~ 内部内容不做上述转换。

使用：
  python scripts/md_normalize.py                # 处理仓库内所有 .md
  python scripts/md_normalize.py README.md a/b.md  # 仅处理指定文件

读写编码：UTF-8 (with BOM)
"""

import os
import re
import sys
from typing import List, Tuple


FENCE_RE = re.compile(r"^\s*(```|~~~)")


def split_by_fences(text: str) -> List[Tuple[bool, str]]:
    """将文档按代码围栏切分为片段。(in_code, segment_text)"""
    lines = text.splitlines(keepends=True)
    segments: List[Tuple[bool, str]] = []
    buf: List[str] = []
    in_code = False
    for line in lines:
        if FENCE_RE.match(line):
            # flush current buffer
            if buf:
                segments.append((in_code, ''.join(buf)))
                buf = []
            # fence line itself
            segments.append((True, line))
            in_code = not in_code
        else:
            buf.append(line)
    if buf:
        segments.append((in_code, ''.join(buf)))
    return segments


def convert_math_delimiters(s: str) -> str:
    # \[ ... \] → $$ ... $$ (not preceded by backslash)
    s = re.sub(r"(?<!\\)\\\[(.+?)(?<!\\)\\\]", r"$$\1$$", s, flags=re.DOTALL)
    # \( ... \) → $ ... $ (not preceded by backslash)
    s = re.sub(r"(?<!\\)\\\((.+?)(?<!\\)\\\)", r"$\1$", s, flags=re.DOTALL)
    return s


def escape_for_texttt(code: str) -> str:
    # 仅按规范对下划线转义；保留其余字符
    return code.replace('_', r'\_')


INLINE_CODE_RE = re.compile(r"`([^`\n]+?)`")


def convert_inline_code(s: str) -> str:
    # 按最新规范：行内代码保持反引号，不做数学打字体转换
    return s


def normalize_markdown(text: str) -> str:
    parts = split_by_fences(text)
    normalized: List[str] = []
    in_code = False
    for in_code, seg in parts:
        if in_code:
            # 代码围栏内：原样保留
            normalized.append(seg)
        else:
            x = convert_math_delimiters(seg)
            x = convert_inline_code(x)
            # KaTeX 兼容性修复：\cdotp 在 KaTeX 中可能不可用，用 \cdot 代替
            x = re.sub(r"\\cdotp\b", r"\\cdot", x)
            normalized.append(x)
    return ''.join(normalized)


def read_text(path: str) -> Tuple[str, str]:
    with open(path, 'rb') as f:
        data = f.read()
    # 检测换行风格
    newline = '\r\n' if b'\r\n' in data else '\n'
    text = data.decode('utf-8-sig', errors='replace')
    return text, newline


def write_text(path: str, text: str, newline: str) -> None:
    # 始终写回 LF（与 .gitattributes: *.md text eol=lf 保持一致，避免 CRLF 警告）
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    with open(path, 'w', encoding='utf-8-sig', newline='\n') as f:
        f.write(text)


def find_all_md(root: str) -> List[str]:
    out: List[str] = []
    skip_dirs = {'.git', '.idea', 'out', 'dist', 'build', '__pycache__'}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in filenames:
            if fn.lower().endswith('.md'):
                out.append(os.path.join(dirpath, fn))
    return out


def main(argv: List[str]) -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    files = [os.path.abspath(p) for p in argv] if argv else find_all_md(repo_root)
    changed = 0
    for path in files:
        if not os.path.isfile(path):
            continue
        try:
            original, nl = read_text(path)
            normalized = normalize_markdown(original)
            if normalized != original:
                write_text(path, normalized, nl)
                changed += 1
        except Exception as e:
            print(f"[md_normalize] skip {path}: {e}")
    if changed:
        print(f"[md_normalize] normalized {changed} markdown file(s)")
    else:
        print("[md_normalize] no change")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
