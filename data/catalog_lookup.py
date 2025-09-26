from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache
from typing import Iterable, Tuple

DATA_DIR = Path(__file__).resolve().parent

CATALOG_PATHS = [
    DATA_DIR / "chinese_name_frequency_word.json",
    DATA_DIR / "chinese_frequency_word.json",
]


def _iter_word_entries(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    label = path.name
    if isinstance(data, dict):
        for word, value in data.items():
            entry_id = None
            if isinstance(value, dict):
                entry_id = value.get("id")
            elif isinstance(value, (int, float, str)):
                entry_id = value
            yield str(word), (label, (str(entry_id).strip() if entry_id is not None else None))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                word = str(item.get("word", "")).strip()
                entry_id = item.get("id")
                yield word, (label, (str(entry_id).strip() if entry_id is not None else None))
            else:
                yield str(item), (label, None)


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, list[tuple[str, str | None]]]:
    catalog: dict[str, list[tuple[str, str | None]]] = {}
    for p in CATALOG_PATHS:
        for word, (label, entry_id) in _iter_word_entries(p) or []:
            token = str(word or "").strip()
            if not token:
                continue
            catalog.setdefault(token, []).append((label, entry_id))
    return catalog


def annotate(term: str) -> tuple[str, bool]:
    """Return human-readable annotation like ' (data/..#id; data/..#id2)' and matched flag.

    If a catalog file has no id for the word, display without '#id'. If none matched, display '未命中'.
    """
    if term is None:
        return "", False
    lookup = str(term).strip()
    if not lookup:
        return "", False
    catalog = load_catalog()
    label_order = [
        ("chinese_name_frequency_word.json", "data/chinese_name_frequency_word.json"),
        ("chinese_frequency_word.json", "data/chinese_frequency_word.json"),
    ]
    label_to_id: dict[str, str | None] = {label: None for label, _ in label_order}
    matched = False
    for label, entry_id in catalog.get(lookup, []):
        if label in label_to_id and entry_id is not None and not label_to_id[label]:
            label_to_id[label] = entry_id
            matched = True
    parts: list[str] = []
    for label, display in label_order:
        entry_id = label_to_id[label]
        if entry_id:
            parts.append(f"{display}#{entry_id}")
        else:
            parts.append(f"{display}未命中")
    return (" (" + "; ".join(parts) + ")"), matched


def longest_prefix_hit(text: str, lengths: Iterable[int]) -> tuple[str, bool, str]:
    """Return (segment, matched, annotation) for the longest prefix in catalog."""
    t = str(text or "")
    Ls = sorted({int(L) for L in lengths if int(L) > 0 and int(L) <= len(t)}, reverse=True)
    for L in Ls:
        seg = t[:L]
        if not seg:
            continue
        ann, ok = annotate(seg)
        if ok:
            return seg, True, ann
    # fallback
    return t[:2], False, annotate(t[:2])[0] if len(t) >= 2 else ("", False, "")[2]


def suffix_hit(text: str, lengths: Iterable[int]) -> tuple[str, bool, str]:
    """Return (segment, matched, annotation) for the longest suffix in catalog."""
    t = str(text or "")
    Ls = sorted({int(L) for L in lengths if int(L) > 0 and int(L) <= len(t)}, reverse=True)
    for L in Ls:
        seg = t[-L:]
        if not seg:
            continue
        ann, ok = annotate(seg)
        if ok:
            return seg, True, ann
    # fallback
    tail2 = t[-2:] if len(t) >= 2 else t
    return tail2, False, annotate(tail2)[0] if tail2 else ""


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--lengths", type=str, default=None, help="comma-separated lengths")
    args = parser.parse_args()
    result = {}
    if args.query:
        ann, ok = annotate(args.query)
        result["annotate"] = {"term": args.query, "matched": ok, "annotation": ann}
    if args.prefix:
        lens = [int(x) for x in (args.lengths or "").split(",") if x.strip().isdigit()]
        seg, ok, ann = longest_prefix_hit(args.prefix, lens)
        result["prefix"] = {"text": args.prefix, "segment": seg, "matched": ok, "annotation": ann}
    if args.suffix:
        lens = [int(x) for x in (args.lengths or "").split(",") if x.strip().isdigit()]
        seg, ok, ann = suffix_hit(args.suffix, lens)
        result["suffix"] = {"text": args.suffix, "segment": seg, "matched": ok, "annotation": ann}
    print(json.dumps(result, ensure_ascii=False))

