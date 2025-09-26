from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


def _iter_jsonl(path: Path) -> list[object]:
    data: list[object] = []
    try:
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: parse error on line {idx}: {e}") from e
            data.append(obj)
    except UnicodeDecodeError as e:
        raise ValueError(f"{path}: encoding error (expect UTF-8): {e}") from e
    return data


def _candidate_files(target: Path, recursive: bool, pattern: str) -> Iterable[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        return target.rglob(pattern) if recursive else target.glob(pattern)
    return []


def convert_jsonl_to_json(
    inputs: list[Path],
    *,
    recursive: bool = False,
    indent: int = 2,
    ensure_ascii: bool = False,
    force: bool = False,
    pattern: str = "*.jsonl",
) -> int:
    processed = 0
    for target in inputs:
        for path in _candidate_files(target, recursive, pattern):
            if not path.name.lower().endswith(".jsonl"):
                continue
            out_path = path.with_suffix("")
            out_path = out_path.with_name(out_path.name + ".json")
            if out_path.exists() and not force:
                print(f"[SKIP] {out_path} exists (use --force to overwrite)")
                continue
            try:
                payload = _iter_jsonl(path)
            except ValueError as e:
                print(f"[ERROR] {e}")
                return 1
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent),
                encoding="utf-8",
            )
            processed += 1
            print(f"[OK] {path} -> {out_path} (items={len(payload)})")
    if processed == 0:
        print("[WARN] No files processed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert *.jsonl to formatted *.json (JSON array).",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Input files or directories. For directories, use --recursive to walk subfolders.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for *.jsonl under directories.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Pretty-print indent (default: 2).",
    )
    parser.add_argument(
        "--ascii",
        dest="ensure_ascii",
        action="store_true",
        help="Escape non-ASCII characters (default: keep UTF-8).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing *.json outputs.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="Glob pattern when scanning directories (default: *.jsonl).",
    )
    args = parser.parse_args()
    return convert_jsonl_to_json(
        list(args.paths),
        recursive=args.recursive,
        indent=args.indent,
        ensure_ascii=args.ensure_ascii,
        force=args.force,
        pattern=args.pattern,
    )


if __name__ == "__main__":
    sys.exit(main())

