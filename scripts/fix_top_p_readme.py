#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 README.md 中的 “Top‑p/Top–p/Top‒p” 等非 ASCII 连接号变体统一替换为纯文本 “Top-p”。
同时确保不会引入数学环境（保持纯文字）。
"""

import os

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
    for sym in ['\u2011', '\u2010', '\u2012', '\u2013']:
        text = text.replace(f'Top{sym}p', 'Top-p')
    # 也修正混入空格的形式
    text = text.replace('Top ‑ p', 'Top-p').replace('Top – p', 'Top-p').replace('Top ‒ p', 'Top-p')
    if text != before:
        write_text(README, text, nl)
        print('[fix_top_p_readme] 已将 README 中的 Top‑p 变体统一为 Top-p')
    else:
        print('[fix_top_p_readme] README 中未发现 Top‑p 变体')

if __name__ == '__main__':
    main()

