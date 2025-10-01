#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import subprocess
from typing import List, Tuple


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        return e.output.decode("utf-8", errors="replace")


def _repo_root() -> str:
    out = _run(["git", "rev-parse", "--show-toplevel"]).strip()
    return out or os.getcwd()


def _normalize_prefix(p: str) -> str:
    p = (p or "").replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    if p.endswith("/"):
        p = p[:-1]
    return p


def _load_commit_filters(repo_root: str) -> Tuple[List[str], List[str]]:
    cfg_path = os.path.join(repo_root, "scripts", "docs_processing_config.json")
    include: List[str] = []
    exclude: List[str] = []
    try:
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            inc = data.get("commit_msg_include_prefixes") or []
            exc = data.get("commit_msg_exclude_prefixes") or []
            include = [_normalize_prefix(x) for x in inc if isinstance(x, str) and x.strip()]
            exclude = [_normalize_prefix(x) for x in exc if isinstance(x, str) and x.strip()]
            if not exclude:
                for x in (data.get("skip_paths") or []):
                    if isinstance(x, str) and x.strip():
                        exclude.append(_normalize_prefix(x))
    except Exception:
        include, exclude = [], []
    return include, exclude


def _build_pathspecs(include: List[str], exclude: List[str]) -> List[str]:
    pathspecs: List[str] = []
    if include:
        pathspecs.extend(include)
    else:
        pathspecs.append(".")
    for p in exclude:
        if not p:
            continue
        pathspecs.append(f":(exclude){p}")
    return pathspecs


def collect_diff_filtered(max_patch_chars: int = 8000) -> tuple[str, str]:
    """Return (stat, patch) filtered by config include/exclude prefixes.

    Config file: scripts/docs_processing_config.json
    - commit_msg_include_prefixes: ["src", "scripts", ...]  # optional whitelist
    - commit_msg_exclude_prefixes: ["docs/kernel_reference", ...]  # optional blacklist
      If not provided, falls back to `skip_paths` for exclusion.
    """
    _root = _repo_root()
    include, exclude = _load_commit_filters(_root)
    stat_cmd = ["git", "diff", "--staged", "--name-status"]
    patch_cmd = ["git", "diff", "--staged", "--unified=0"]
    pathspecs = _build_pathspecs(include, exclude)
    if pathspecs:
        stat_cmd += ["--"] + pathspecs
        patch_cmd += ["--"] + pathspecs
    stat = _run(stat_cmd).strip()
    patch = _run(patch_cmd).strip()
    if len(patch) > max_patch_chars:
        patch = patch[: max_patch_chars - 1] + "\n��(truncated)"
    return stat, patch

