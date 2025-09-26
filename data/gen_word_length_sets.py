from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

DATA_DIR = Path(__file__).resolve().parent
NAMES_PATH = DATA_DIR / "chinese_name_frequency_word.json"
FREQ_PATH = DATA_DIR / "chinese_frequency_word.json"
OUT_PATH = DATA_DIR / "word_length_sets.json"


def _iter_words(obj) -> Iterable[str]:
    if obj is None:
        return []
    if isinstance(obj, dict):
        return obj.keys()
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and "word" in item:
                yield str(item.get("word", ""))
            else:
                yield str(item)
        return
    # fallback: unsupported structure
    return []


def _length_set(path: Path) -> set[int]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    lengths: set[int] = set()
    for w in _iter_words(data):
        token = str(w or "").strip()
        if not token:
            continue
        lengths.add(len(token))
    return lengths


def build_word_length_sets() -> dict:
    names = _length_set(NAMES_PATH)
    freq = _length_set(FREQ_PATH)
    union = sorted(set(names) | set(freq))
    return {
        "names": {
            "unique_length_count": len(names),
            "lengths": sorted(names),
        },
        "freq": {
            "unique_length_count": len(freq),
            "lengths": sorted(freq),
        },
        "union": {
            "unique_length_count": len(union),
            "lengths": union,
        },
    }


def main() -> int:
    payload = build_word_length_sets()
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {OUT_PATH} with union lengths: {payload['union']['lengths']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

