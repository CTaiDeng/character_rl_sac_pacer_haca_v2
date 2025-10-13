#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
为 docs 根目录下符合 `^\d+_.*\.md$` 的 Markdown 文档在文末追加统一许可声明页脚；已存在则不重复插入。

- 作用范围：仅限 docs/ 顶层，不递归子目录（例如跳过 docs/kernel_reference/）。
- 编码与行尾：写回 UTF-8（带 BOM）+ LF。
- 版式要求：在分隔线 `---` 前后均保留一个空行（即文档末尾内容，与 `---` 之间有 1 个空行；`---` 与“许可声明”标题之间也有 1 个空行）。
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
import subprocess
from datetime import datetime, timezone


RE_TS_MD = re.compile(r"^\d+_.*\.md$", re.IGNORECASE)
def build_footer_block(year_text: str) -> str:
    """根据年份文本构建页脚块。

    要求：在分隔线 `---` 前后各保留一个空行（调用方负责在块前添加一个空行）。
    """
    return (
        "---\n\n"
        "**许可声明 (License)**\n\n"
        f"Copyright (C) {year_text} GaoZheng\n\n"
        "本文档采用[知识共享-署名-非商业性使用-禁止演绎 4.0 国际许可协议 (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.zh-Hans)进行许可。\n"
    )


def has_footer(text: str) -> bool:
    # 以关键句作为存在性判定，避免重复插入
    return (
        "**许可声明 (License)**" in text
        and "CC BY-NC-ND 4.0" in text
        and "creativecommons.org/licenses/by-nc-nd/4.0" in text
    )


def standardize_existing_footer(text: str, footer_block: str) -> str:
    """
    若已存在许可声明，则将从最后一个许可声明块（含其上方可能存在的分隔线与空行）起至文末替换为规范页脚，
    并确保在分隔线 `---` 前后各 1 个空行。
    """
    # 寻找最后一次出现的“许可声明”标题位置
    anchor = "**许可声明 (License)**"
    idx = text.rfind(anchor)
    if idx == -1:
        return text
    # 从该标题向上寻找最近的分隔线 ---（允许有若干空行/空白）
    head = text[:idx]
    tail = text[idx:]
    # 在 head 末尾回溯查找 '---' 所在行
    # 简单策略：找到最后一次出现的以 --- 开头的行的位置
    last_sep_pos = -1
    pos = 0
    for m in re.finditer(r"^\s*---\s*$", head, re.MULTILINE):
        last_sep_pos = m.start()
    # 计算替换起点：如果找到分隔线，则从分隔线之前的可能空行起截断；否则从标题行起截断
    start = last_sep_pos if last_sep_pos != -1 else idx
    # 向上吞并分隔线前的空行
    while start > 0 and text[start - 1] == "\n":
        start -= 1
    # 保留正文部分，并在其后放置恰好一个空行，再跟规范页脚
    body = text[:start].rstrip("\n")
    return body + "\n\n" + footer_block


def normalize_lf(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def write_utf8_bom(path: Path, text: str) -> None:
    data = normalize_lf(text)
    with open(path, "wb") as f:
        f.write(b"\xef\xbb\xbf")  # UTF-8 BOM
        f.write(data.encode("utf-8"))


def _git_first_add_year(repo: Path, file_path: Path) -> int | None:
    try:
        ts = subprocess.check_output(
            ["git", "log", "--diff-filter=A", "--follow", "--format=%at", "-n", "1", "--", str(file_path)],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="ignore").strip()
        if ts:
            return datetime.fromtimestamp(int(ts), tz=timezone.utc).year
    except Exception:
        return None
    return None


def _git_last_mod_year(repo: Path, file_path: Path) -> int | None:
    try:
        ts = subprocess.check_output(
            ["git", "log", "-1", "--format=%at", "--", str(file_path)],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="ignore").strip()
        if ts:
            return datetime.fromtimestamp(int(ts), tz=timezone.utc).year
    except Exception:
        return None
    return None


def _creation_year_from_name(p: Path) -> int | None:
    m = re.match(r"^(\d+)_", p.name)
    if not m:
        return None
    try:
        sec = int(m.group(1))
        return datetime.fromtimestamp(sec, tz=timezone.utc).year
    except Exception:
        return None


def process_docs(root: Path, dry_run: bool = False) -> int:
    docs = root / "docs"
    if not docs.is_dir():
        return 0
    updated = 0
    for p in sorted(docs.iterdir()):
        if not p.is_file():
            continue
        if not RE_TS_MD.match(p.name):
            continue
        try:
            # 读入（自动忽略 BOM）
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        norm = normalize_lf(text)
        # 动态年份：创立年（文件名时间戳 -> git 首次入库 -> 文件创建时间）与最后修改年（git 最近提交 -> mtime）
        create_year = _creation_year_from_name(p)
        if create_year is None:
            cy = _git_first_add_year(root, p)
            create_year = cy if cy is not None else datetime.fromtimestamp(p.stat().st_ctime, tz=timezone.utc).year
        last_year = _git_last_mod_year(root, p)
        if last_year is None:
            last_year = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).year
        year_text = f"{create_year}-{last_year}" if last_year > create_year else f"{create_year}"
        footer_block = build_footer_block(year_text)

        if has_footer(norm):
            new_text = standardize_existing_footer(norm, footer_block)
        else:
            # 统一在末尾追加；确保分隔线前有一个空行
            base = norm.rstrip("\n")
            new_text = base + "\n\n" + footer_block
        if dry_run:
            print(f"[insert_docs_license_footer] DRY add footer: {p}")
        else:
            write_utf8_bom(p, new_text)
        updated += 1
    return updated


def main() -> int:
    ap = argparse.ArgumentParser(description="Insert/check license footer into docs/*.md with ts prefix (top-level only)")
    ap.add_argument("--dry-run", action="store_true", help="Do not write changes; report only")
    ap.add_argument("--check", action="store_true", help="Exit with non-zero status if any file would be updated")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    # --check implies dry-run
    dry = args.dry_run or args.check
    n = process_docs(root, dry_run=dry)
    if dry:
        print(f"[insert_docs_license_footer] DryRun: would update {n} files")
        if args.check:
            return 1 if n > 0 else 0
    else:
        print(f"[insert_docs_license_footer] Updated {n} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
