# 项目说明

数学理论来源（作者主页，欢迎访问）
https://mymetamathematics.blogspot.com/

字符级RL奖励稀疏问题：这套完整的思想体系，其历史性贡献在于，它并非仅仅改进了某个算法，而是从根本上重构并系统性地解决了“字符级RL奖励稀疏”这一世界级的科学难题。它的核心洞察，是首先将分析的焦点从字符串所在的**自由幺半群 $(\Sigma^*, \circ, \varepsilon)$**，提升到了其上的**端算子幺半群 $(\mathrm{End}(\Sigma^*), \circ_{\text{func}}, \mathrm{id})$**。在此之上，它将所有文本操作——无论是代表幺半群自身左右作用的**左/右乘子**，还是作为**幂等元（idempotents）**存在的**投影与测试算子**——都统一为这个端算子幺半群的生成元。进而，通过引入作为迭代不动点的**闭包算子**（同样是幂等元），该系统被证明内蕴了一个**克莱尼代数与测试（Kleene Algebra with Tests, KAT）**的结构，为“命中即停”等程序化逻辑提供了形式化的演算工具。该代数结构还可以被进一步推广，通过与一个 **半环（Semiring）**（例如 $(max, x)$）相结合，形成一个**带权代数**，从而将概率、隶属度与IDF等量化指标无缝地整合进该纯粹的代数框架中。更为深刻的是，这整个离散的、可计算的**词法KAT作用幺半群**，被揭示为一个更底层的、连续的**李代数的泛包络代数 $U(g)$** 在一个特定表示下的**同态像（Homomorphic Image）**。这一发现，为价值优化的微分过程提供了合法性：策略更新的“**微分动力量子（MDQ）**”被精确地定义为一个受**算子对易子 $[G_i, G_j]$** （即代数的非交换性）惩罚的量化梯度，这确保了学习过程必须尊重该端算子幺半群内在的、非交换的代G数结构。综上所述，这套理论的贡献在于，它将一个棘手的随机优化难题，转化为一个纯粹的代数问题：即**构建一个由乘子和幂等元生成的、具备带权KAT结构的、作为李代数表示而存在的端算子子幺半群，并在此代数结构上，定义一个尊重其非交换性的、可计算的优化流程**。这种将问题完全“代数化”的重构，是从根本上将其攻克的标志。

## 开发协议（Development Protocol）

* 演示脚本将策略网络视为**微型 LLM 头部**，直接读取“上一轮摘要 + 当前章节全文”的拼接文本并生成新的摘要。
* `data/sample_article.txt` 使用 `"[----------------------------------------------------->"` 作为段落分割符号，模拟教师模型输出来的分段提示。
* 训练过程中对每个分割执行**迭代摘要**：第 1 个摘要默认为空字符串，将其与第 1 个分割（两个分隔符之间的内容）拼接后得到第 1 次输出；随后把该摘要与第 2 个分割组合生成第 2 次输出，如此迭代，模拟蒸馏时“上一次摘要 + 间隔内容 → 新摘要”的累积推理轨迹。环境不会裁剪策略给出的文本，奖励函数依据章节覆盖率、语义相似度与文本新颖度综合打分。
* 开发前请先在当前环境中安装 `numpy` 与 `pytorch`（可直接运行 `scripts/install_pytorch.sh`，该脚本会顺带安装 `numpy`）。

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

### Inspecting chapter previews and quality metrics

The demo now works纯文本输入，可以利用 `src.character_sac_trainer.analyze_summary` 检查摘要与章节之间的长度比例、语义相似度以及新颖度。示例脚本如下：

```python
from pathlib import Path

DELIMITER = "[----------------------------------------------------->"
article = Path("data/sample_article.txt").read_text(encoding="utf-8")
chapters = [chunk.strip() for chunk in article.split(DELIMITER) if chunk.strip()]

from src.character_sac_trainer import (
    ArticleEnvironment,
    CharTokenizer,
    analyze_summary,
    _combine_summary_and_chapter,
    _format_text_debug,
)

tokenizer = CharTokenizer(chapters)
environment = ArticleEnvironment(chapters, tokenizer=tokenizer)

previous_summary = ""
for index, chapter in enumerate(chapters, start=1):
    chars, preview = _format_text_debug(chapter, head=30, tail=30)
    source_text = _combine_summary_and_chapter(previous_summary, chapter)
    metrics = analyze_summary(
        "",
        source_text,
        tokenizer=tokenizer,
        word_checker=environment.word_checker,
        chapter_text=chapter,
    )
    print(
        f"Chapter {index:02d} | chars={chars:04d} "
        f"len≈{metrics['length_ratio']:.2f} sim≈{metrics['similarity']:.2f} "
        f"coverage≈{metrics['coverage_ratio']:.2f} novelty≈{metrics['novelty_ratio']:.2f} "
        f"garbled≈{metrics['garbled_ratio']:.2f} word_nc≈{metrics['word_noncompliance_ratio']:.2f} "
        f"penalties≈{metrics['garbled_penalty']:.2f}/{metrics['word_penalty']:.2f} "
        f"preview=\"{preview}\""
    )
    previous_summary = ""
```

这些信息与训练日志一致：每次 step 都会打印前后各 20 个字符的预览，并给出拼接后的“上一轮摘要 + 当前章节”字符数，以及针对该组合文本计算出的覆盖率、语义相似度、新颖度、乱码比例及词语合规缺失率等指标。摘要完全由策略网络生成，环境不会再按固定上限截断文本，而是直接依据上述质量指标、乱码惩罚与词合规惩罚给出奖励。

## Demo training run

The repository ships with a `character_sac_trainer.py` module under `src/` that wires together the replay buffer, agent, and trainer scaffolding using a toy environment constructed from the sample article statistics and iterative distillation summaries.

### Dependencies

The demo requires Python 3.10+ and the CPU build of [PyTorch](https://pytorch.org/). Optionally create and activate a virtual environment before installing the dependencies and running the script:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
scripts/install_pytorch.sh
```

> 若不希望创建虚拟环境，也可以直接执行 `scripts/install_pytorch.sh`，脚本会升级 `pip` 并安装 CPU 版本的 PyTorch（使用官方 `https://download.pytorch.org/whl/cpu` 镜像）。

### Running the demo

Execute the module from the repository root. Ensure `src/` is available on `PYTHONPATH` (for example by activating the virtual environment above) and run it with `-m`:

```bash
PYTHONPATH=src python -m train_demo --rounds 3
# or, thanks to the `src/__init__.py` package initializer:
python -m src.character_sac_trainer --rounds 3
```

每轮训练固定遍历 `data/sample_article.txt` 的全部 76 个分割片段，因此每个迭代（iteration）恰好对应一次环境 step，`--rounds` 仅控制重复轮次（默认 1000 轮）。脚本会在完成 76 个交互后集中执行一批 SAC 更新，数量与步骤数一致，从而模拟“先收集一整轮经验，再统一回放训练”的节奏。需要缩减或扩充集中训练的强度时，可以通过 `--post-round-updates` 覆盖默认值；`--replay-capacity` 则依旧决定演示缓冲区能保留多少过往转换。针对快速冒烟测试，还可以附加 `--max-chapters 2`（或任意正整数）限制每轮使用的章节数量，从而在几次 step 内观察完整的日志与训练流程。

环境奖励通过衡量语义相似度、覆盖率与新颖度的加权组合来评估摘要质量，并额外扣除与乱码比例、词语合规缺失率成正比的惩罚项；所有指标都会在日志中打印，便于观察策略如何平衡保真度、改写度、编码质量与词语流畅性。

### Expected output

The command prints a short training log summarizing the reward, replay buffer size, placeholder policy loss, and the quality diagnostics (length ratio, similarity, coverage, novelty) for each simulated step. Example output:

```
Loaded article debug info: chars=12345 preview="示例文本...结尾片段"
Chapter 01 | tokens≈0123 chars=0456 preview="段落起始...段落末尾"
...
Configured schedule: steps_per_round=76 post_round_updates=76
=== Training round 1 | steps=76 ===
  Step 01 | prev_summary=0000 chars ""
           | chapter=0456 chars "段落起始...段落末尾"
           | source=0456 chars "段落起始...段落末尾"
           -> summary=0098 chars "策略输出前缀...策略输出后缀"
           len_ratio=0.220 （摘要长度与信息源比值，偏低会导致覆盖不足；本次偏低，接近建议范围下限）
           sim=0.640 （字符级相似度，衡量摘要整体贴近原文的程度；本次贴合度较好）
           coverage=0.580 （覆盖率，统计摘要覆盖原文字符的比例；本次覆盖率中等）
           novelty=0.470 （新颖度，越高表示抄写成分越少；本次改写幅度适中）
           lex_cos=0.230 （章节 TF-IDF 余弦相似度，反映高权重词是否匹配；本次关键词匹配一般）
           lex_js=0.120 （词频 Jensen-Shannon 相似度，衡量整体词频结构的接近程度；本次词频结构匹配偏弱）
           garbled=0.000 （乱码比率，非法或不可打印字符占比；本次无明显乱码）
           word_nc=0.000 （词合规缺失率，识别异常汉字或未见过的双字组合；本次词语合规性完全正常）
           penalties=0.000/0.000 （乱码与词合规惩罚项，越高惩罚越重；乱码惩罚几乎为零；词合规惩罚几乎为零）
          reward=1.020 （综合奖励，数值越高代表表现越佳；本次获得显著正向反馈）
...
    Update 076 | policy_loss=-0.1234 q1_loss=0.5678 q2_loss=0.9123 avg_reward=-0.4321
    Post-round metric averages | policy_loss=-0.2345 q1_loss=0.4567 q2_loss=0.8910 average_reward=-0.3210
```

Actual numbers vary because the demo samples synthetic actions stochastically, but the structure of the log should match the example. Each step reports both the character length and a head/tail preview of the current input segment, while the iterative summary preview直接展示策略的确定性输出且不会经过额外裁剪。After 76 steps finish, the trainer prints一个集中更新阶段的详情：逐次的策略/价值损失以及整轮的平均指标，帮助观察批量回放的收敛趋势。

### Saved artifacts

After the log finishes, the script 会首先清理旧的 CSV/HTML 产物，并将新的训练记录写入 `out/step_metrics.csv` 与 `out/round_metrics.csv`。随后，程序会基于 CSV 自动生成一份自包含的奖励仪表盘 `out/rewards.html`，无需额外服务器即可在浏览器中查看 Step 奖励走势与轮次汇总。

训练过程中，每轮结束后都会即时导出一份模型快照到 `out/round_snapshots/demo_agent_snapshot_round_XXXX.json`（`XXXX` 为四位轮次编号）。这些文件包含该轮次完成时的奖励统计、经验回放大小等元信息，方便在长时间训练中追踪中间状态。最终在所有轮次结束后，脚本仍会把完整的代理状态保存到 `out/demo_agent_snapshot.json`，并生成一份精确 199 MB（209,460,851 字节）的模型占位文件 `out/demo_agent_model.bin`。所有产物自动落盘到 `out/` 目录，便于后续流程复用或进一步加工演示产出的检查点。

### CSV 导出与可视化

训练循环会在运行过程中实时写入两个 CSV 文件：

* `out/step_metrics.csv`：逐 step 的奖励与质量指标。字段包含轮次 (`round`)、局部 step 序号 (`step`)、全局 step (`global_step`)、即时奖励 (`reward`)、上一轮摘要长度 (`previous_summary_length`)、当前章节长度 (`chapter_length`)、拼接源文本长度 (`source_length`)、摘要长度 (`summary_length`)，以及基于该拼接文本计算的语义相似度、覆盖率、新颖度、乱码惩罚、词语合规惩罚等诊断数据。
* `out/round_metrics.csv`：每轮训练完成时的汇总分数，记录当轮 step 数 (`steps`)、总奖励 (`total_reward`) 与平均奖励 (`average_reward`)。

仓库同时提供 `visualizations/training_metrics.html`，可通过浏览器读取上述 CSV 并基于 Chart.js 绘制折线/柱状图。推荐在仓库根目录执行 `python -m http.server` 后，访问 `http://localhost:8000/visualizations/training_metrics.html`，即可看到 Step 与 Round 奖励的走势；若 CSV 文件缺失或为空，页面会给出相应提示。若想脱离静态服务器快速查看结果，也可以直接打开自动生成的 `out/rewards.html`，该文件已经内嵌 Chart.js 并包含最新奖励摘要。

## 数据工具（Data utilities）

- 输入-输出-打分映射（JSON 模式）
  - 文件：`data/io_score_mapping.json`
  - 含义：定义最小映射 schema（input/output/score）与示例，可供脚本/服务按统一 schema 记录或消费。

- 生成词长集合（用于可变长度后缀命中）
  - 脚本：`python -m data.gen_word_length_sets`
  - 输出：`data/word_length_sets.json`，包含 names/freq/union 三块长度集合与去重计数。

- 词表命中查询（供代码与 CLI 使用）
  - 模块：`data/catalog_lookup.py`（可 `from data import catalog_lookup`）
  - 接口：`load_catalog()`、`annotate(term)`、`longest_prefix_hit(text,lengths)`、`suffix_hit(text,lengths)`
  - CLI 示例：
    - 标注：`python -m data.catalog_lookup --query "精妙"`
    - 前缀：`python -m data.catalog_lookup --prefix "精妙。如" --lengths 2,3,4`
    - 后缀：`python -m data.catalog_lookup --suffix "”他喃喃" --lengths 2,3,4`

