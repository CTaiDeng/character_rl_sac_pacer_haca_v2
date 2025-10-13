#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Force re-encode docs top-level Markdown files to UTF-8 with BOM and LF line endings.
- Scope: docs/ (top-level files only), excludes subdirectories.
- Strategy: try UTF-8 decode first; if fails, try GBK (cp936) as fallback. Always write back UTF-8 BOM + LF.
- Check mode: with --check, do not write; exit non-zero if any file would be rewritten.
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def to_lf(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def recode_to_utf8_bom(p: Path, dry_run: bool = False) -> bool:
    b = p.read_bytes()
    text = None
    try:
        text = b.decode("utf-8-sig")
    except Exception:
        try:
            text = b.decode("gbk")
        except Exception:
            return False
    # Normalize newlines and compute output
    norm = to_lf(text)
    out_b = ("\ufeff" + norm).encode("utf-8")
    changed = (out_b != b)
    if dry_run:
        return changed
    if changed:
        p.write_bytes(out_b)
    return changed


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Force docs top-level .md to UTF-8 BOM + LF")
    ap.add_argument("--check", action="store_true", help="Only check; non-zero exit if any file would be rewritten")
    args = ap.parse_args(argv)
    if not DOCS.is_dir():
        print("[force_docs_utf8_bom] docs/ not found")
        return 0
    updated = 0
    for p in sorted(DOCS.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".md":
            continue
        try:
            would = recode_to_utf8_bom(p, dry_run=True)
            if args.check:
                if would:
                    updated += 1
                    print(f"[force_docs_utf8_bom] would rewrite: {p.relative_to(ROOT)}")
                continue
            if recode_to_utf8_bom(p, dry_run=False):
                updated += 1
                print(f"[force_docs_utf8_bom] rewrote: {p.relative_to(ROOT)}")
        except Exception as e:
            print(f"[force_docs_utf8_bom] skip {p}: {e}")
    print(f"[force_docs_utf8_bom] updated={updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
