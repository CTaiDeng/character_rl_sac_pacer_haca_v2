#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文档对齐指令：标准化 docs 知识库

步骤：
1) 将 docs/<ts>_*.md 的时间戳前缀重写为该文件的 git 入库时间戳（秒）
2) 将该时间戳（转为 YYYY-MM-DD）写入文档主标题下一行（若存在则更新）
3) 重建 README 文末的文档摘要索引
4) 规范化 Markdown（行内/行间数学分隔、保留代码围栏），编码 UTF-8（BOM）

用法：
  python scripts/align_docs.py
"""

import os
import sys
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> int:
    try:
        print("[align_docs] $", " ".join(cmd))
        return subprocess.call(cmd, cwd=str(ROOT))
    except Exception as e:
        print(f"[align_docs] 运行失败: {e}")
        return 1


def main() -> int:
    rc = 0
    rc |= run([sys.executable, str(ROOT / 'scripts' / 'rename_docs_to_git_ts.py')])
    rc |= run([sys.executable, str(ROOT / 'scripts' / 'insert_doc_date_from_prefix.py')])
    rc |= run([sys.executable, str(ROOT / 'scripts' / 'update_readme_index.py')])
    # 规范化 README 与 docs
    rc |= run([sys.executable, str(ROOT / 'scripts' / 'md_normalize.py'), 'README.md'])
    rc |= run([sys.executable, str(ROOT / 'scripts' / 'md_normalize.py'), 'docs'])
    if rc == 0:
        print('[align_docs] 文档对齐完成')
    else:
        print('[align_docs] 文档对齐存在错误，请查看上方输出')
    return rc


if __name__ == '__main__':
    sys.exit(main())

