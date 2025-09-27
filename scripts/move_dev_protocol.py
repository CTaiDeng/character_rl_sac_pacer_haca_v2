#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 README.md 中的“开发协议（Development Protocol）”小节统一迁移为指向 AGENTS.md 的说明。

操作：定位 README.md 中以 `##` 开头且包含“开发协议”的标题，
删除该标题到下一个 `##` 标题（或文件末尾）之间的内容，
替换为：

## 开发协议
本项目的开发协议已统一至 AGENTS.md，请参见该文件的“演示与环境约定”“Markdown 规范”“文档摘要同步规范”等章节。

读写编码：UTF-8（BOM）。
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


def move_section():
    text, nl = read_text(README)
    lines = text.splitlines()
    # 找到“开发协议”标题行
    start = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith('##') and ('开发协议' in ln or 'Development Protocol' in ln):
            start = i
            break
    if start is None:
        print('[move_dev_protocol] 未找到“开发协议”小节，跳过')
        return
    # 找到下一个二级标题或文件末尾
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].lstrip().startswith('## '):
            end = j
            break

    replacement = [
        '## 开发协议',
        '',
        '本项目的开发协议已统一至 AGENTS.md，请参见该文件的“演示与环境约定”“Markdown 规范”“文档摘要同步规范”等章节。',
        '',
    ]

    new_lines = lines[:start] + replacement + lines[end:]
    write_text(README, '\n'.join(new_lines), nl)
    print(f'[move_dev_protocol] 已统一迁移 README 中的开发协议小节（行 {start+1}–{end}）')


if __name__ == '__main__':
    move_section()

