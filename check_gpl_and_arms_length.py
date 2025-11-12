#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

"""
检查项：
1) 对“未被 .gitignore 忽略”的全部 Python 脚本（.py）进行排查，判断是否使用了 GPL 家族许可证（GPL/AGPL/LGPL）的第三方 Python 包；
2)（可选，仅 --scope src）校验 src 与 data/scripts/tests 之间是否保持“臂长通信”（禁止 src 作为库方式 import 这些目录下的模块，禁止通过 sys.path 注入路径联动）。

用法示例：
  - 文本报告（默认全仓）：
      python3 check_gpl_and_arms_length.py
  - JSON 输出：
      python3 check_gpl_and_arms_length.py --json
  - 指定范围：
      python3 check_gpl_and_arms_length.py --scope repo|src   # 默认 repo（全仓）
  - CI 严格模式（发现 GPL 家族依赖失败）：
      python3 check_gpl_and_arms_length.py --fail-on-gpl
  - 追加臂长约束（仅对 --scope src 生效）：
      python3 check_gpl_and_arms_length.py --scope src --fail-on-armlength

说明：
- 依赖探测基于静态 AST 提取 import 顶层模块名，并用 importlib.metadata 映射到安装分发包以读取许可证与版本。
- 目标文件集合严格以 git 索引和 .gitignore 为准：通过 `git ls-files` + `git ls-files -o --exclude-standard` 枚举未被忽略的 .py 文件（包含已跟踪与未忽略的未跟踪）。
- 该脚本为只读检查，不修改仓库内容。
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import sysconfig
from dataclasses import dataclass
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import importlib.metadata as ilmd  # Python 3.8+
except Exception:  # pragma: no cover
    ilmd = None  # type: ignore


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
SCRIPTS_DIR = ROOT / "scripts"
TESTS_DIR = ROOT / "tests"


# ------------------------ 工具方法 ------------------------

EXCLUDE_DIR_NAMES = {".git", ".venv", "__pycache__", "out"}


def _git_ls_py_files(root: Path, subdir: Optional[Path] = None) -> List[Path]:
    """使用 git（尊重 .gitignore）列出 .py 文件（包含已跟踪 + 未忽略的未跟踪）。

    当 subdir 提供时，仅在该子目录范围内枚举。
    """
    import subprocess
    files: List[Path] = []
    # 计算 pathspec
    pathspec: List[str] = []
    if subdir is not None:
        try:
            rel = os.path.relpath(subdir, root)
        except Exception:
            rel = str(subdir)
        # 使用 "-- <path>" 限定范围，再在 Python 侧过滤 .py 后缀，避免依赖特性通配
        pathspec = ["--", rel]
    try:
        tracked_args = ["git", "ls-files"] + pathspec
        tracked = subprocess.check_output(tracked_args, cwd=str(root)).decode("utf-8", errors="ignore").splitlines()
    except Exception:
        tracked = []
    try:
        untracked_args = ["git", "ls-files", "-o", "--exclude-standard"] + pathspec
        untracked = subprocess.check_output(untracked_args, cwd=str(root)).decode("utf-8", errors="ignore").splitlines()
    except Exception:
        untracked = []
    seen: Set[str] = set()
    for rel in tracked + untracked:
        rel = rel.strip()
        if not rel or not rel.endswith(".py"):
            continue
        if rel in seen:
            continue
        seen.add(rel)
        p = (root / rel).resolve()
        if p.is_file():
            files.append(p)
    return files


def list_py_files(d: Path, recursive: bool = True) -> List[Path]:
    """基于 git（.gitignore）列出 .py 文件；git 不可用时回退到 os.walk（粗略排除目录）。

    - 当 d 位于仓库根内时，使用 `git ls-files` + 未忽略未跟踪模式限定子树范围；
    - 否则仅回退扫描。
    """
    if not d.exists():
        return []
    try:
        # d 在仓库根内：统一走 git（尊重 .gitignore）
        d_res = d.resolve()
        root_res = ROOT.resolve()
        if d_res == root_res:
            return _git_ls_py_files(ROOT, None)
        # d 若为根子目录（例如 src/），仍使用 git 并限定范围
        if str(d_res).startswith(str(root_res)):
            return _git_ls_py_files(ROOT, d)
    except Exception:
        pass

    # 回退扫描（git 不可用）
    if not recursive:
        return [p for p in d.glob("*.py") if p.is_file()]
    out: List[Path] = []
    for root, dirs, files in os.walk(d):
        dirs[:] = [nm for nm in dirs if nm not in EXCLUDE_DIR_NAMES]
        for fn in files:
            if fn.endswith(".py"):
                out.append(Path(root) / fn)
    return out


def extract_top_level_imports(py_path: Path) -> List[Tuple[str, int]]:
    """返回 [(顶层模块名, 行号), ...]。忽略 __future__。"""
    out: List[Tuple[str, int]] = []
    try:
        text = py_path.read_text(encoding="utf-8", errors="ignore")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SyntaxWarning)
            node = ast.parse(text, filename=str(py_path))
    except Exception:
        return out
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for alias in n.names:
                name = (alias.name or "").split(".")[0]
                if name and name != "__future__":
                    out.append((name, n.lineno))
        elif isinstance(n, ast.ImportFrom):
            # from . import x -> relative，忽略；仅关注绝对顶层名
            if n.level and n.level > 0:
                continue
            mod = (n.module or "").split(".")[0]
            if mod and mod != "__future__":
                out.append((mod, n.lineno))
    return out


def detect_sys_path_hacks(py_path: Path) -> List[Tuple[int, str]]:
    """检测 sys.path.append/insert 等语句中包含 data/scripts/tests 的情况。"""
    res: List[Tuple[int, str]] = []
    try:
        text = py_path.read_text(encoding="utf-8", errors="ignore")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SyntaxWarning)
            node = ast.parse(text, filename=str(py_path))
    except Exception:
        return res
    targets = {"data", "scripts", "tests"}
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            try:
                func_repr = ast.unparse(n.func)  # type: ignore[attr-defined]
            except Exception:
                # 简易回退
                func_repr = getattr(getattr(n.func, "attr", None), "__str__", lambda: "")()
            if "sys.path" in func_repr and any(k in func_repr for k in ("append", "insert", "extend")):
                # 查找参数字符串
                for arg in n.args:
                    s = None
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        s = arg.value
                    elif isinstance(arg, ast.JoinedStr):
                        s = "".join([p.value for p in arg.values if isinstance(p, ast.Constant) and isinstance(p.value, str)])
                    if s:
                        lower = s.lower()
                        if any(("/" + t + "/") in lower or (os.sep + t + os.sep) in lower or lower.endswith("/" + t) or lower.endswith(os.sep + t) for t in targets):
                            res.append((n.lineno, s))
    return res


def stdlib_root() -> Optional[str]:
    try:
        return sysconfig.get_paths().get("stdlib")
    except Exception:
        return None


def discover_src_top_packages(src: Path) -> Set[str]:
    pkgs: Set[str] = set()
    if not src.exists():
        return pkgs
    for p in src.iterdir():
        if p.is_dir() and (p / "__init__.py").exists():
            pkgs.add(p.name)
        elif p.is_file() and p.suffix == ".py":
            pkgs.add(p.stem)
    return pkgs


def build_top_level_to_dist_map() -> Dict[str, List[ilmd.Distribution]]:
    """尽力映射顶层模块名 -> 分发包列表。优先使用 packages_distributions；回退读取 top_level.txt。"""
    mapping: Dict[str, List[ilmd.Distribution]] = {}
    if ilmd is None:
        return mapping
    # 优先路径（Python 3.10+ 或 backport 提供）
    try:
        packages_distributions = getattr(ilmd, "packages_distributions", None)
        if packages_distributions is not None:
            for top, dists in packages_distributions().items():  # type: ignore
                for dist_name in dists:
                    try:
                        dist = ilmd.distribution(dist_name)
                    except Exception:
                        continue
                    mapping.setdefault(top, []).append(dist)
            return mapping
    except Exception:
        pass
    # 回退：遍历分发读取 top_level.txt
    try:
        for dist in ilmd.distributions():  # type: ignore
            try:
                tl = dist.read_text("top_level.txt")  # type: ignore[attr-defined]
            except Exception:
                tl = None
            tops: List[str] = []
            if tl:
                tops = [x.strip() for x in tl.splitlines() if x.strip()]
            # 次优回退：根据文件推断一级包名
            if not tops:
                try:
                    files = dist.files or []  # type: ignore[attr-defined]
                except Exception:
                    files = []
                for f in files:
                    parts = str(f).split("/")
                    if len(parts) >= 2 and parts[0] and parts[0][0].isalpha():
                        tops.append(parts[0])
            for t in set(tops):
                mapping.setdefault(t, []).append(dist)
    except Exception:
        pass
    return mapping


def license_family(license_str: str, classifiers: Iterable[str]) -> Tuple[str, str]:
    """返回 (family, raw_license)。family 取值：GPL/AGPL/LGPL/Non-GPL/Unknown。"""
    raw = (license_str or "").strip()
    lowers = (raw.lower(), "\n".join(classifiers).lower())
    joined = "\n".join(lowers)
    if not raw and not joined.strip():
        return ("Unknown", raw)
    if "agpl" in joined or "affero" in joined:
        return ("AGPL", raw or "AGPL (from classifiers)")
    if "lgpl" in joined:
        return ("LGPL", raw or "LGPL (from classifiers)")
    if "gpl" in joined:
        return ("GPL", raw or "GPL (from classifiers)")
    return ("Non-GPL", raw)


@dataclass
class ThirdPartyUse:
    module: str
    file_refs: List[Tuple[str, int]]  # (relative path, line)
    distributions: List[Dict[str, str]]  # [{name, version, license_family, license_raw}]


@dataclass
class ArmsLengthViolation:
    file: str
    line: int
    kind: str  # import/path_hack
    detail: str


def analyze(scope: str = "repo") -> Tuple[List[ThirdPartyUse], List[ArmsLengthViolation]]:
    std_root = stdlib_root() or ""
    toplevel_map = build_top_level_to_dist_map() if ilmd is not None else {}

    # 1) 收集 import
    imports_by_module: Dict[str, List[Tuple[str, int]]] = {}
    violations: List[ArmsLengthViolation] = []

    target_files: List[Path]
    if scope == "src":
        target_files = list_py_files(SRC_DIR)
    else:
        # 全仓：以仓根为起点递归扫描，排除常见目录
        target_files = list_py_files(ROOT)

    for py in target_files:
        # 相对路径
        try:
            rel = str(py.relative_to(ROOT))
        except Exception:
            rel = str(py)
        # imports
        for mod, lineno in extract_top_level_imports(py):
            if scope == "src":
                # 臂长：仅在 src 范围下检查
                if mod in {"data", "scripts", "tests"}:
                    violations.append(ArmsLengthViolation(file=rel, line=lineno, kind="import", detail=f"import {mod}"))
            imports_by_module.setdefault(mod, []).append((rel, lineno))
        # sys.path hacks（仅在 src 范围下检查）
        if scope == "src":
            for lineno, s in detect_sys_path_hacks(py):
                violations.append(ArmsLengthViolation(file=rel, line=lineno, kind="path_hack", detail=s))

    # 2) 过滤内置/标准库与本仓首方模块
    thirdparty: Dict[str, List[Tuple[str, int]]] = {}
    repo_root_s = str(ROOT.resolve()).replace("\\", "/")
    for mod, refs in imports_by_module.items():
        # 不将 data/scripts/tests 计入第三方统计（可能是本仓目录名；在 src 模式由臂长规则单独检查）
        if mod in {"data", "scripts", "tests"}:
            continue

        # 通过 importlib 解析模块来源，区分 标准库/内置、本仓、本地未安装第三方
        try:
            import importlib.util as ilu
            spec = ilu.find_spec(mod)
        except Exception:
            spec = None
        is_stdlib = False
        is_local_repo = False
        if spec is None:
            is_stdlib = False
            is_local_repo = False
        else:
            origin = getattr(spec, "origin", None)
            if origin in {None, "built-in", "frozen"}:
                is_stdlib = True
            else:
                origin_s = str(origin)
                norm = origin_s.replace("\\", "/")
                if std_root and norm.startswith(str(std_root).replace("\\", "/")):
                    is_stdlib = True
                elif repo_root_s and norm.startswith(repo_root_s):
                    is_local_repo = True

        if is_stdlib or is_local_repo:
            continue
        thirdparty[mod] = refs

    # 3) 映射分发与许可证
    uses: List[ThirdPartyUse] = []
    for mod, refs in sorted(thirdparty.items()):
        dists = []
        # 优先：toplevel -> distributions 映射
        cand_dists = []
        if mod in toplevel_map:
            cand_dists = toplevel_map[mod]
        else:
            # 回退：用分发名等于模块名的近似
            if ilmd is not None:
                for dist in ilmd.distributions():  # type: ignore
                    try:
                        name = (dist.metadata.get("Name") or "").strip()
                    except Exception:
                        name = ""
                    if name and name.lower() == mod.lower():
                        cand_dists.append(dist)
        seen_names: Set[str] = set()
        for dist in cand_dists:
            try:
                name = (dist.metadata.get("Name") or "").strip()
                version = (dist.metadata.get("Version") or "").strip()
                lic = (dist.metadata.get("License") or "").strip()
                classifiers = [c for c in dist.metadata.get_all("Classifier") or []]
            except Exception:
                continue
            fam, raw = license_family(lic, classifiers)
            key = f"{name}=={version}"
            if not name or key in seen_names:
                continue
            seen_names.add(key)
            dists.append({
                "name": name,
                "version": version,
                "license_family": fam,
                "license_raw": raw or lic or "",
            })
        uses.append(ThirdPartyUse(module=mod, file_refs=refs, distributions=dists))

    return uses, violations


def main() -> int:
    ap = argparse.ArgumentParser(description="Check GPL-family dependencies (repo-wide) and optional arm's-length (src)")
    ap.add_argument("--json", action="store_true", help="输出 JSON 报告")
    ap.add_argument("--fail-on-gpl", action="store_true", help="发现 GPL/AGPL/LGPL 依赖时退出码为 2")
    ap.add_argument("--fail-on-armlength", action="store_true", help="发现臂长通信违规时退出码为 3（仅 --scope src 有效）")
    ap.add_argument("--scope", choices=["repo", "src"], default="repo", help="检查范围：repo=全仓（默认）、src=仅源码并启用臂长检查")
    args = ap.parse_args()

    uses, violations = analyze(scope=args.scope)
    # 统计：去重 import 模块数量，扫描文件数量
    try:
        scanned_files = len(list_py_files(ROOT)) if args.scope == "repo" else len(list_py_files(SRC_DIR))
    except Exception:
        scanned_files = 0
    unique_imports = len({u.module for u in uses})

    # 汇总统计
    gpl_uses: List[Dict[str, str]] = []
    for u in uses:
        for d in u.distributions:
            if d.get("license_family") in {"GPL", "AGPL", "LGPL"}:
                gpl_uses.append({
                    "module": u.module,
                    "distribution": d["name"],
                    "version": d.get("version", ""),
                    "license_family": d.get("license_family", ""),
                    "license_raw": d.get("license_raw", ""),
                })

    if args.json:
        payload = {
            "summary": {
                "third_party_modules": len(uses),
                "gpl_family_dependencies": len(gpl_uses),
                "armlength_violations": len(violations) if args.scope == "src" else 0,
                "scanned_files": scanned_files,
                "unique_imports": unique_imports,
            },
            "third_party_uses": [
                {
                    "module": u.module,
                    "file_refs": u.file_refs,
                    "distributions": u.distributions,
                }
                for u in uses
            ],
            "gpl_family": gpl_uses,
            "armlength_violations": (
                [
                    {"file": v.file, "line": v.line, "kind": v.kind, "detail": v.detail}
                    for v in violations
                ]
                if args.scope == "src" else []
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        # 彩色样式
        C = {
            "b": "\033[1m",
            "g": "\033[32m",
            "y": "\033[33m",
            "r": "\033[31m",
            "c": "\033[36m",
            "reset": "\033[0m",
        }
        # 1) 第三方许可证检查（本项目样式）
        print(f"{C['b']}{C['c']}[第三方许可证检查]{C['reset']}")
        print(f" {C['c']}-{C['reset']} 扫描脚本数: {scanned_files}")
        print(f" {C['c']}-{C['reset']} 唯一导入模块数: {unique_imports}")
        if gpl_uses:
            print(f" {C['c']}-{C['reset']} GPL 家族依赖：")
            for g in gpl_uses:
                print(f"   {C['r']}!{C['reset']} 分发 {g['distribution']}=={g['version']}（通过 import {g['module']}）| 许可证 {g['license_family']}（{g['license_raw']}）")
        else:
            print(f" {C['c']}-{C['reset']} GPL 家族依赖：未发现（基于本地包元数据）")

        # 2) 臂长边界检查（仅在 src 场景严格；repo 场景提供概览）
        print(f"{C['b']}{C['c']}[臂长边界检查]{C['reset']}")
        # 直接导入 data/scripts/tests
        if args.scope == "src":
            direct_imports = [v for v in violations if v.kind == "import"]
            if not direct_imports:
                print(f" {C['c']}-{C['reset']} 未发现对 data/scripts/tests 的直接导入")
            else:
                print(f" {C['y']}-{C['reset']} 发现直接导入：")
                for v in direct_imports:
                    print(f"   {C['y']}+{C['reset']} {v.file}:{v.line} → {v.detail}")
            path_hacks = [v for v in violations if v.kind == "path_hack"]
            if path_hacks:
                print(f" {C['y']}-{C['reset']} 检测到朝向 data/scripts/tests 的 sys.path 操作：")
                for v in path_hacks:
                    print(f"   {C['y']}+{C['reset']} {v.file}:{v.line} → {v.detail}")
        else:
            print(f" {C['c']}-{C['reset']} 范围=repo；臂长规则仅在 --scope src 生效")

        # 3) 证据：CLI 使用（子进程）
        print(f" {C['c']}-{C['reset']} CLI 使用证据（臂长侧）：")
        # 简易扫描：grep 典型子进程调用与脚本入口使用痕迹
        evidences: List[str] = []
        try:
            import subprocess
            # 查找 subprocess.* 调用
            r = subprocess.check_output(["rg", "-n", "subprocess\\.", "-S"], cwd=str(ROOT))
            for line in r.decode("utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                evidences.append(line)
        except Exception:
            pass
        # 附加：常见 CLI 文案（不执行，只做静态字符串线索）
        for pat in [
            "python scripts/align_docs.py",
            "python scripts/update_readme_index.py",
            "compute_chapter_tfidf.py",
        ]:
            try:
                import subprocess
                r = subprocess.check_output(["rg", "-n", pat], cwd=str(ROOT))
                for line in r.decode("utf-8", errors="ignore").splitlines():
                    evidences.append(line)
            except Exception:
                pass
        if evidences:
            # 取前若干条避免刷屏
            for ev in evidences[:10]:
                print(f"   {C['g']}+{C['reset']} {ev}")
            if len(evidences) > 10:
                print(f"   {C['g']}+{C['reset']} ... 另有 {len(evidences)-10} 条")
        else:
            print(f"   {C['g']}+{C['reset']} 未发现明显的 CLI/子进程使用线索")

        ok_gpl = (len(gpl_uses) == 0)
        ok_boundary = (args.scope != "src") or (len(violations) == 0)
        if ok_gpl and ok_boundary:
            print(f"{C['g']}[OK]{C['reset']} 未发现 GPL 家族依赖；与宿主保持臂长通信（基于当前环境可用的元数据与静态扫描）")
        else:
            if not ok_gpl and ok_boundary:
                print(f"{C['y']}[WARN]{C['reset']} 发现 GPL 家族依赖，请评估兼容性与许可证义务")
            elif ok_gpl and not ok_boundary:
                print(f"{C['y']}[WARN]{C['reset']} 发现与宿主边界的直连或路径注入，请改为 CLI/文件 I/O")
            else:
                print(f"{C['r']}[FAIL]{C['reset']} 同时存在 GPL 家族依赖与臂长边界问题，请立即整改")

    exit_code = 0
    if args.fail_on_gpl and gpl_uses:
        exit_code = max(exit_code, 2)
    if args.fail_on_armlength and args.scope == "src" and violations:
        exit_code = 3 if exit_code == 0 else exit_code
    raise SystemExit(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
