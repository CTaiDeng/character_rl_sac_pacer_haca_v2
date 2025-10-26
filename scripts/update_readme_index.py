#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

"""
从 docs/*.md 收集“摘要”，在 README.md 末尾维护统一索引块。

规范与约束：
- 摘要来源优先级：
  1) <!-- SUMMARY-START -->...<!-- SUMMARY-END --> 注释块
  2) 文中 "## 摘要" 标题后的首段正文（跳过空行）
  3) 顶部以 "> 摘要：" 开头的连续引用行（兼容旧文档）
- 只读策略：不修改 docs 原文内容；如缺少摘要，仅在 README 索引中显示占位提示。
- 覆盖范围：仅顶层 docs/*.md；显式排除 docs/LICENSE.md 与 docs/kernel_reference/**。
- 写回策略：所有写回统一 UTF-8（无 BOM）+ LF；严禁 CRLF 或 BOM。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional

# I/O 采用统一 UTF-8+LF 写回
try:
    from io_utf8lf import read_text as _read_text_lf, write_text as _write_text_lf  # type: ignore
except Exception:  # 兜底：sitecustomize 也会强制 UTF-8+LF
    _read_text_lf = None  # type: ignore
    _write_text_lf = None  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
README = ROOT / "README.md"

START_MARK = "<!-- DOCS-SUMMARY-INDEX:START -->"
END_MARK = "<!-- DOCS-SUMMARY-INDEX:END -->"


def _read_text(path: Path) -> str:
    if _read_text_lf is not None:
        return _read_text_lf(path)
    # 兼容 UTF-8-SIG；标准化为 LF
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _write_text(path: Path, text: str) -> None:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if _write_text_lf is not None:
        _write_text_lf(path, normalized)
    else:
        path.write_text(normalized, encoding="utf-8")


def iter_top_level_docs() -> Iterable[Path]:
    if not DOCS.is_dir():
        return []
    for p in sorted(DOCS.glob("*.md")):
        # 排除顶层 LICENSE 与任何非文件
        if not p.is_file():
            continue
        if p.name.lower() == "license.md":
            continue
        yield p


def extract_summary(text: str) -> Optional[str]:
    # 1) 注释块优先
    m = re.search(r"<!--\s*SUMMARY-START\s*-->(.*?)<!--\s*SUMMARY-END\s*-->", text, flags=re.S)
    if m:
        block = m.group(1).strip()
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        return " ".join(lines)

    # 2) 二级标题 “## 摘要” 后首段
    h = re.search(r"^##\s*摘要\s*$", text, flags=re.M)
    if h:
        start = h.end()
        rest = text[start:]
        # 跳过空行
        rest = re.sub(r"^\s*\n", "", rest, count=1, flags=re.M)
        # 取到下一空行或标题前
        m2 = re.search(r"\n\s*\n|\n#", rest)
        para = rest[: m2.start()] if m2 else rest
        lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
        if lines:
            return " ".join(lines)

    # 3) 兼容旧的 > 摘要： 引用段（连续）
    m = re.search(r"^(?:>\s*摘要\s*[:：].*?(?:\n|$))+", text, flags=re.M)
    if m:
        block = m.group(0)
        lines = []
        for ln in block.splitlines():
            s = ln.lstrip()
            if s.startswith(">"): s = s[1:]
            s = s.strip()
            if s:
                # 去除前缀
                s = re.sub(r"^摘要\s*[:：]\s*", "", s)
                lines.append(s)
        if lines:
            return " ".join(lines)

    return None


def build_index_block() -> tuple[str, int]:
    entries: list[str] = []
    count = 0
    for p in iter_top_level_docs():
        count += 1
        text = _read_text(p)
        summary = extract_summary(text)
        if not summary:
            summary = "[缺少摘要：请在文首添加 '## 摘要' 或 SUMMARY 注释块]"
        # README 中使用点击路径
        rel_path = p.as_posix().replace(ROOT.as_posix() + "/", "")
        entries.append(f"- `{rel_path}`\n  - {summary}")

    block = [START_MARK, "", *entries, "", END_MARK]
    return "\n".join(block) + "\n", count


def upsert_index_in_readme(readme_text: str, index_block: str) -> str:
    if START_MARK in readme_text and END_MARK in readme_text:
        pattern = re.compile(re.escape(START_MARK) + r".*?" + re.escape(END_MARK), re.S)
        return pattern.sub(lambda _m: index_block.strip(), readme_text)
    # 若不存在标记，则在末尾追加“## 文档摘要索引”标题与区块
    parts = [readme_text.rstrip(), "", "## 文档摘要索引", "", index_block.strip(), ""]
    return "\n".join(parts)


def main() -> int:
    index_block, n = build_index_block()
    readme = _read_text(README) if README.exists() else ""
    new_text = upsert_index_in_readme(readme, index_block)
    _write_text(README, new_text)
    print(f"collected={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
