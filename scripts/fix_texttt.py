#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复 README.md 中被误写为 "$\texttt" 变体的问题，例如 "$\texttt" 前出现空格/Tab 导致形如 "$\texttt"→"$\texttt" 或 "$\texttt"→"$\texttt"。

具体规则：
- 将 "$\s+exttt{" 规范为 "$\\texttt{"；
- 将 "$\s+texttt{" 规范为 "$\\texttt{"；

只处理 README.md，读写 UTF-8（带 BOM）。
"""

import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
README = os.path.join(ROOT, 'README.md')


def read_text(path: str):
    with open(path, 'rb') as f:
        data = f.read()
    nl = '\r\n' if b'\r\n' in data else '\n'
    return data.decode('utf-8-sig', errors='replace'), nl


def write_text(path: str, text: str, nl: str):
    text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\n', nl)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write(text)


def main():
    text, nl = read_text(README)
    before = text
    # $ <spaces/tabs> exttt{  ->  $	exttt{
    text = re.sub(r"\$[ \t]+exttt\{", r"$\\texttt{", text)
    # $ <spaces/tabs> texttt{  ->  $	exttt{
    text = re.sub(r"\$[ \t]+texttt\{", r"$\\texttt{", text)
    if text != before:
        write_text(README, text, nl)
        print("[fix_texttt] 已修复 README.md 中 $ texttt 写法为 $\\texttt")
    else:
        print("[fix_texttt] 未发现需要修复的 $ texttt 写法")


if __name__ == '__main__':
    main()

