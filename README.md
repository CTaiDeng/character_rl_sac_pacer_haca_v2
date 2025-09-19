# G008

This repository hosts scaffolding for a Soft Actor-Critic (SAC) implementation located in `src/rl_sac`. The current code base provides class and function skeletons that describe the flow of data between replay buffers, policy/value networks, and training routines. A lightweight training demonstration is available to show how the components fit together when driven by synthetic data distilled from the bundled example article.

## 开发协议（Development Protocol）

* 演示脚本将策略网络视为**微型 LLM 头部**，通过 SAC 更新对示例文章进行知识蒸馏。
* `data/sample_article.txt` 使用 `"[----------------------------------------------------->"` 作为段落分割符号，模拟教师模型输出来的分段提示。
* 训练过程中对每个分割执行**迭代摘要**：第 1 个摘要默认为空字符串，将其与第 1 个分割（两个分隔符之间的内容）拼接后得到第 1 次输出；随后把该摘要与第 2 个分割组合生成第 2 次输出，如此迭代，模拟蒸馏时“上一次摘要 + 间隔内容 → 新摘要”的累积推理轨迹。

## Examples

The `data/` directory contains sample textual material that mimics the structure of articles used throughout the project. For instance, `data/sample_article.txt` 提供了一篇多段落的中文示例文章，围绕状态表示、策略参数化以及评估流程等 SAC 概念展开，并补充了离线数据融合、超参数搜索与未来展望等段落。这些文字被刻意写得较长，以便验证分片处理与批量载入逻辑。文件通过 `"[----------------------------------------------------->"` 分隔段落，从而便于下游工具将其视作教师模型输出的逐段提示。

### Loading the sample article

You can load the example document using standard Python file operations. The snippet below demonstrates how to stream the file and split it into paragraphs for further preprocessing:

```python
from pathlib import Path

example_path = Path("data/sample_article.txt")
text = example_path.read_text(encoding="utf-8")
intervals = [
    interval.strip()
    for interval in text.split("[----------------------------------------------------->")
    if interval.strip()
]

for idx, interval in enumerate(intervals, start=1):
    print(f"Interval {idx}: {interval[:60]}...")
```

This workflow mirrors the intended usage within data ingestion pipelines, ensuring that each section of the article can be independently tokenized or transformed before feeding into SAC-related training tasks.

## Demo training run

The repository ships with a `train_demo.py` module under `src/` that wires together the replay buffer, agent, and trainer scaffolding using a toy environment constructed from the sample article statistics and iterative distillation summaries.

### Dependencies

The demo requires Python 3.10+ and the CPU build of [PyTorch](https://pytorch.org/). Optionally create and activate a virtual environment before installing the dependencies and running the script:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install torch
```

### Running the demo

Execute the module from the repository root. Ensure `src/` is available on `PYTHONPATH` (for example by activating the virtual environment above) and run it with `-m`:

```bash
PYTHONPATH=src python -m train_demo --steps 8
# or, thanks to the `src/__init__.py` package initializer:
python -m src.train_demo --steps 8
```

The `--steps` flag controls the number of simulated environment interactions, while `--replay-capacity` adjusts the maximum number of transitions retained in the demo buffer.

### Expected output

The command prints a short training log summarizing the reward, replay buffer size, and placeholder policy loss for each simulated step. Example output:

```
Step 01 | reward=-10.54 buffer=1 policy_loss=nan
Step 02 | reward=-5.21 buffer=2 policy_loss=nan
Step 03 | reward=-4.97 buffer=3 policy_loss=nan
Step 04 | reward=-2.08 buffer=4 policy_loss=2.08
...
```

Actual numbers vary because the demo samples synthetic actions stochastically, but the structure of the log should match the example.

### Saved artifacts

After the log finishes, the script 序列化一个模型快照到 `out/demo_agent_snapshot.json`，其中包含演示代理的占位符状态与运行元数据（如训练步数、经验回放容量）。该代理始终在 CPU 上训练，并记录策略头部的参数数量，同时标注导出的模型体积。为了满足新的存档协议，脚本会在 `out/demo_agent_model.bin` 写出一个精确 199 MB（209,460,851 字节）的二进制模型文件，用以模拟重量级微型 LLM 头部的交付物。所有产物会自动创建父目录 `out/`，便于在多阶段流程中复用或进一步加工演示产出的检查点。
