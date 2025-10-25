#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

"""
UTF-8（无 BOM）+ LF 写回策略（scripts 目录自动加载）

说明：Python 会在启动时自动导入名为 `sitecustomize` 的模块（若能在 sys.path 找到）。
当从 `scripts/` 目录运行脚本或启动 REPL 时，本文件会被自动导入，从而对默认
写文件行为做只读覆盖：
- 对文本写入（w/a/x 且非二进制），若未显式指定 encoding/newline，则强制
  encoding='utf-8' 且 newline='\n'；
- 覆盖 Path.write_text，统一将内容中的 CRLF/CR 归一化为 LF 并以 UTF-8 写出。

如需临时关闭：设置环境变量 `DISABLE_UTF8LF_SITEPATCH=1`。
"""

import os
import builtins
from pathlib import Path

if os.environ.get("DISABLE_UTF8LF_SITEPATCH") in {"1", "true", "True"}:
    pass
else:
    _orig_open = builtins.open

    def _patched_open(file, mode="r", buffering=-1, encoding=None, errors=None,
                      newline=None, closefd=True, opener=None):
        try:
            text_mode = "b" not in mode
            write_mode = any(ch in mode for ch in ("w", "a", "x"))
            if text_mode and write_mode:
                if encoding is None:
                    encoding = "utf-8"
                if newline is None:
                    newline = "\n"
        except Exception:
            pass
        return _orig_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

    builtins.open = _patched_open  # type: ignore

    _orig_write_text = Path.write_text

    def _patched_write_text(self: Path, data: str, encoding: str | None = None, errors: str | None = None):
        enc = encoding or "utf-8"
        if isinstance(data, str):
            data = data.replace("\r\n", "\n").replace("\r", "\n")
        with self.open("w", encoding=enc, errors=errors, newline="\n") as f:  # type: ignore[arg-type]
            return f.write(data)

    Path.write_text = _patched_write_text  # type: ignore

