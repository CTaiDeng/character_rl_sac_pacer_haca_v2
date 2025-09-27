#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对 README.md 中残留英文段落进行定点替换为中文。

匹配规则（逐行，多行模式）：
- 行首包含 "The $\\texttt{data/}$ directory contains sample textual material" 的行。
- 行首包含 "The demo now works" 的行。
- 行首包含 "Actual numbers vary because the demo" 的行。
- 行首包含 "After the log finishes, the script" 的行。

读写编码：UTF‑8（BOM），保留原换行。
"""

import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
README = os.path.join(ROOT, 'README.md')


def read_text(path: str):
    with open(path, 'rb') as f:
        data = f.read()
    nl = '\r\n' if b'\r\n' in data else '\n'
    return data.decode('utf-8-sig', errors='replace'), nl


def write_text(path: str, text: str, nl: str):
    text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\n', nl)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write(text)


def main():
    text, nl = read_text(README)

    patterns = [
        (
            re.compile(r"(?m)^The\s+\$\\texttt\{data/\}\$\s+directory contains sample textual material.*$"),
            (
                "数据目录 $\\texttt{data/}$ 包含用于本项目的示例文本素材，结构与实际文章相仿。"
                "例如，$\\texttt{data/sample\\_article.txt}$ 提供一篇多段落中文示例，围绕状态表示、策略参数化与评估流程（SAC 概念）展开，"
                "并补充离线数据融合、超参数搜索与展望等段落。文本较长，以便验证分片处理与批量载入逻辑。"
                "文件通过 $\\texttt{\"[----------------------------------------------------->\"}$ 分隔段落，便于下游工具将其视作教师模型输出的逐段提示。"
            ),
        ),
        (
            re.compile(r"(?m)^The demo now works.*$"),
            (
                "当前演示基于纯文本输入，可调用 $\\texttt{src.character\\_sac\\_trainer.analyze\\_summary}$ 在“上一轮摘要 + 当前章节”拼接后，"
                "对长度、相似度、覆盖率、新颖度以及词法合规等指标进行分析："
            ),
        ),
        (
            re.compile(r"(?m)^Actual numbers vary because the demo.*$"),
            (
                "由于演示采用随机采样的方式生成动作，具体数值会有所波动，但日志结构应与示例一致。"
                "每一步都会同时报告字符长度与当前输入片段的首/尾预览；在迭代摘要预览中也会直观呈现关键指标。"
                "完成 76 步后，训练器会打印阶段性汇总，包括各损失项与奖励的均值等，便于观察收敛趋势。"
            ),
        ),
        (
            re.compile(r"(?m)^After the log finishes, the script.*$"),
            (
                "日志结束后，脚本会生成 CSV/HTML 报表，将本次训练记录写入 $\\texttt{out/step\\_metrics.csv}$ 与 $\\texttt{out/round\\_metrics.csv}$；"
                "此外会基于这些 CSV 自动生成一份可视化结果页 $\\texttt{out/rewards.html}$，便于直接查看 Step 与 Round 的指标走势和位置统计。"
            ),
        ),
    ]

    changed = 0
    for pat, repl in patterns:
        new_text, n = pat.subn(repl, text)
        if n:
            changed += n
            text = new_text

    if changed:
        write_text(README, text, nl)
        print(f"[patch_readme_translate] 替换 {changed} 处英文为中文")
    else:
        print("[patch_readme_translate] 未发现需替换的英文行")


if __name__ == '__main__':
    main()

