#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
从 docs/*.md 提取“摘要”并在 README.md 文末维护索引。

摘要来源优先级：
1) <!-- SUMMARY-START --> ... <!-- SUMMARY-END -->
2) 文首首个由连续 “> ” 开头的块（常见的 “> 摘要：...” 形式）
3) 若缺失，则在标题后自动插入占位摘要，并使用该占位内容。

索引块由以下标记包裹，便于幂等更新：
  <!-- DOCS-SUMMARY-INDEX:START --> ... <!-- DOCS-SUMMARY-INDEX:END -->

读写编码：UTF-8 (with BOM)
"""

import os
import re
from typing import List, Tuple


START_MARK = "<!-- DOCS-SUMMARY-INDEX:START -->"
END_MARK = "<!-- DOCS-SUMMARY-INDEX:END -->"


def read_text(path: str):
    with open(path, 'rb') as f:
        data = f.read()
    nl = '\r\n' if b'\r\n' in data else '\n'
    return data.decode('utf-8-sig', errors='replace'), nl


def write_text(path: str, text: str, nl: str):
    # README 与 docs 写回统一使用 LF，避免 CRLF/LF 混用导致 Git 警告
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    with open(path, 'w', encoding='utf-8-sig', newline='\n') as f:
        f.write(text)


def ensure_summary_in_doc(path: str) -> Tuple[str, bool]:
    """返回 (摘要文本, 文档是否被更新)"""
    text, nl = read_text(path)

    # 1) 注释包裹
    m = re.search(r"<!--\s*SUMMARY-START\s*-->(.*?)<!--\s*SUMMARY-END\s*-->", text, flags=re.DOTALL)
    if m:
        summary = m.group(1).strip()
        single_line = ' '.join(line.strip() for line in summary.splitlines() if line.strip())
        # 去重前缀“摘要：”
        single_line = re.sub(r'^摘要\s*[:：]\s*', '', single_line)
        return single_line, False

    # 2) 连续区块引用
    lines = text.splitlines()
    start = None
    end = None
    for i, ln in enumerate(lines[:100]):  # 仅扫描前 100 行
        if ln.lstrip().startswith('>'):
            start = i
            j = i
            while j < len(lines) and lines[j].lstrip().startswith('>'):
                j += 1
            end = j
            break
    if start is not None and end is not None:
        block = lines[start:end]
        # 去掉前缀 '>' 和空白
        cleaned = []
        for b in block:
            s = b.lstrip()
            if s.startswith('>'):
                s = s[1:]
            cleaned.append(s.strip())
        single_line = ' '.join([c for c in cleaned if c])
        single_line = re.sub(r'^摘要\s*[:：]\s*', '', single_line)
        return single_line, False

    # 3) 无摘要：在标题后插入占位摘要
    updated_lines = []
    inserted = False
    for idx, ln in enumerate(lines):
        updated_lines.append(ln)
        if not inserted and ln.strip().startswith('#'):
            updated_lines.append('')
            updated_lines.append('> 摘要：TODO：请补充本篇文档摘要（120–300字）。建议概述核心目标、方法、关键结果与适用范围。')
            updated_lines.append('')
            inserted = True
    if not inserted:
        # 若未找到标题，则在文件头部添加
        updated_lines = [
            '# 文档标题缺失',
            '',
            '> 摘要：TODO：请补充本篇文档摘要（120–300字）。建议概述核心目标、方法、关键结果与适用范围。',
            '',
        ] + lines

    new_text = '\n'.join(updated_lines)
    write_text(path, new_text, nl)
    return 'TODO：请补充本篇文档摘要（120–300字）。', True


def collect_docs(root: str) -> List[str]:
    p = os.path.join(root, 'docs')
    if not os.path.isdir(p):
        return []
    files = [os.path.join(p, fn) for fn in os.listdir(p) if fn.lower().endswith('.md')]
    files.sort()
    return files


def build_index_block(items: List[Tuple[str, str]]) -> str:
    lines: List[str] = []
    lines.append('## 文档摘要索引')
    lines.append(START_MARK)
    for rel_path, summary in items:
        # 使用反引号包裹路径，遵循“行内代码保留反引号、不转换为数学字体”的规范
        lines.append(f"- `{rel_path}`")
        lines.append(f"  - 摘要：{summary}")
    lines.append(END_MARK)
    return '\n'.join(lines) + '\n'


def upsert_readme_index(root: str, items: List[Tuple[str, str]]):
    readme = os.path.join(root, 'README.md')
    if not os.path.isfile(readme):
        return
    text, nl = read_text(readme)
    block = build_index_block(items)
    # 移除历史重复的“文档摘要（docs）”块，避免与索引重复
    # 使用行级定位，删除从包含 "docs" 的二级标题开始到下一个二级标题之前的内容
    lines = text.splitlines()
    def _is_h2(ln: str) -> bool:
        return ln.lstrip().startswith('## ')
    legacy_start = None
    for idx, ln in enumerate(lines):
        if _is_h2(ln) and 'docs' in ln:
            legacy_start = idx
            break
    if legacy_start is not None:
        legacy_end = len(lines)
        for j in range(legacy_start + 1, len(lines)):
            if _is_h2(lines[j]):
                legacy_end = j
                break
        del lines[legacy_start:legacy_end]
        text = '\n'.join(lines)

    # 移除已存在的索引块（无论位置），确保最终位于文末
    # 移除已存在的索引块（无论位置）
    header_block_pat = re.compile(
        r'(?ms)^[ \t]*##[ \t]*文档摘要索引\s*\n\s*' + re.escape(START_MARK) + r'.*?' + re.escape(END_MARK) + r'\s*'
    )
    text = header_block_pat.sub('', text)
    plain_block_pat = re.compile(re.escape(START_MARK) + r'.*?' + re.escape(END_MARK), re.DOTALL)
    text = plain_block_pat.sub('', text)

    # 文末追加（保留至少一行空行）
    new_text = text.rstrip() + '\n\n' + block
    write_text(readme, new_text, nl)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    docs = collect_docs(root)
    items: List[Tuple[str, str]] = []
    updated_docs = 0
    for path in docs:
        summary, updated = ensure_summary_in_doc(path)
        rel = os.path.relpath(path, root).replace('\\', '/')
        # 适度截断到 ~300 字符
        summary = summary.strip()
        if len(summary) > 320:
            summary = summary[:317].rstrip() + '…'
        items.append((rel, summary))
        if updated:
            updated_docs += 1
    upsert_readme_index(root, items)
    print(f"[update_readme_index] collected={len(items)} updated_missing_summaries={updated_docs}")


if __name__ == '__main__':
    main()
