import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOCS = [
    ROOT / "INPUT_OUTPUT_SCHEME.md",
    ROOT / "NETWORK_TOPOLOGY.md",
    ROOT / "STEP_SCORING.md",
]

def contains(p: Path, needles: list[str]) -> bool:
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    text_lower = text.lower()
    return any(n.lower() in text_lower for n in needles)

def main() -> int:
    ok = True
    # 1) 检查 character_history_extension_limit 是否在文档中体现
    needles_cfg = ["character_history_extension_limit", "历史回溯", "左扩", "至多 16"]
    if not any(contains(d, needles_cfg) for d in DOCS):
        print("[WARN] 文档未提及 character_history_extension_limit 或历史回溯拓扑上限(16)。")
        ok = False

    # 2) 检查 word_length_sets.json 并提示文档是否引用
    wls = ROOT / "data" / "word_length_sets.json"
    if wls.exists():
        try:
            data = json.loads(wls.read_text(encoding="utf-8"))
            union = data.get("union", {})
            lengths = union.get("lengths", [])
            if lengths and not any(contains(d, ["word_length_sets.json", "union.lengths"]) for d in DOCS):
                print("[WARN] 文档未说明使用 data/word_length_sets.json 的 union.lengths 进行后缀命中判定。")
                ok = False
        except Exception as e:
            print(f"[WARN] 读取 word_length_sets.json 失败: {e}")
            ok = False
    else:
        print("[WARN] 缺少 data/word_length_sets.json，无法驱动可变长度后缀命中。")
        ok = False

    if ok:
        print("[OK] 文档-配置同步项通过基本检查。")
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main())

