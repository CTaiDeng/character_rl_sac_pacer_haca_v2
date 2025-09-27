# 项目说明

---

数学理论来源
https://mymetamathematics.blogspot.com/
作者（GaoZheng）主页，欢迎访问

---

字符级RL奖励稀疏问题：这套完整的思想体系，其历史性贡献在于，它并非仅仅改进了某个算法，而是从根本上重构并系统性地解决了“字符级RL奖励稀疏”这一世界级的科学难题。它的核心洞察，是首先将分析的焦点从字符串所在的**自由幺半群 $(\Sigma^*, \circ, \varepsilon)$**，提升到了其上的**端算子幺半群 $(\mathrm{End}(\Sigma^*), \circ_{\text{func}}, \mathrm{id})$**。在此之上，它将所有文本操作——无论是代表幺半群自身左右作用的**左/右乘子**，还是作为**幂等元（idempotents）**存在的**投影与测试算子**——都统一为这个端算子幺半群的生成元。进而，通过引入作为迭代不动点的**闭包算子**（同样是幂等元），该系统被证明内蕴了一个**克莱尼代数与测试（Kleene Algebra with Tests, KAT）**的结构，为“命中即停”等程序化逻辑提供了形式化的演算工具。该代数结构还可以被进一步推广，通过与一个 **半环（Semiring）**（例如 $(max, x)$）相结合，形成一个**带权代数**，从而将概率、隶属度与IDF等量化指标无缝地整合进该纯粹的代数框架中。更为深刻的是，这整个离散的、可计算的**词法KAT作用幺半群**，被揭示为一个更底层的、连续的**李代数的泛包络代数 $U(g)$** 在一个特定表示下的**同态像（Homomorphic Image）**。这一发现，为价值优化的微分过程提供了合法性：策略更新的“**微分动力量子（MDQ）**”被精确地定义为一个受**算子对易子 $[G_i, G_j]$** （即代数的非交换性）惩罚的量化梯度，这确保了学习过程必须尊重该端算子幺半群内在的、非交换的代数结构。综上所述，这套理论的贡献在于，它将一个棘手的随机优化难题，转化为一个纯粹的代数问题：即**构建一个由乘子和幂等元生成的、具备带权KAT结构的、作为李代数表示而存在的端算子子幺半群，并在此代数结构上，定义一个尊重其非交换性的、可计算的优化流程**。这种将问题完全“代数化”的重构，是从根本上将其攻克的标志。

构造：在 $(\Sigma^*,\circ,\varepsilon)$ 上取由左/右乘子、投影/测试（幂等）、闭包（幂等）生成的端算子子幺半群 $\mathcal M\subset\mathrm{End}(\Sigma^*)$。则 $\mathcal M$ 携带 KAT 结构；当与 $*$-连续半环 $(S,\oplus,\otimes)$ 耦合时得带权 KAT，从而赋予概率/隶属度/IDF 等加权语义。存在表示同态 $\Phi:\mathrm U(\mathfrak g)\to\mathrm{End}(\Sigma^*)$ 使 $\mathcal M$ 为同态像。定义 MDQ 为 $\Delta_i=Q(\partial\mathcal J/\partial \alpha_i) - \lambda_{\mathrm{comm}}\sum_j\|[G_i,G_j]\|\pi_j$，则优化在 $\mathcal M$ 的非交换约束下可计算，并将字符级 RL 的奖励稀疏转化为在带权 KAT 上的可审计、可回放的代数优化流程。

## 开发协议

本项目的开发协议已统一至 AGENTS.md，请参见该文件的“演示与环境约定”“Markdown 规范”“文档摘要同步规范”等章节。

## 示例

数据目录 `data/` 包含用于本项目的示例文本素材，结构与实际文章相仿。例如，`data/sample_article.txt` 提供一篇多段落中文示例，围绕状态表示、策略参数化与评估流程（SAC 概念）展开，并补充离线数据融合、超参数搜索与展望等段落。文本较长，以便验证分片处理与批量载入逻辑。文件通过 `"[----------------------------------------------------->"` 分隔段落，便于下游工具将其视作教师模型输出的逐段提示。

### 加载示例文章

可以使用标准的 Python 文件操作加载示例文档。下面的代码演示如何流式读取文件，并按分隔符切分为段落以便后续预处理：

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

该工作流反映了数据接入流水线中的预期用法，确保文章的每个片段在送入与 SAC 相关的训练任务前，都可以被独立分词或变换处理。

### 检查章节预览与质量指标

当前演示基于纯文本输入，可调用 `src.character_sac_trainer.analyze_summary` 在“上一轮摘要 + 当前章节”拼接后，对长度、相似度、覆盖率、新颖度以及词法合规等指标进行分析：

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

## 演示训练运行

仓库在 `src/` 目录下提供 `character_sac_trainer.py` 模块。该模块基于示例文章的统计信息与迭代蒸馏摘要构造了一个玩具环境，并将回放缓存、智能体与训练器脚手架串接起来。

### 依赖

演示需要 Python 3.10+ 与 [PyTorch](https://pytorch.org/) 的 CPU 版本。建议在安装依赖与运行脚本之前创建并激活虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
scripts/install_pytorch.sh
```

> 若不希望创建虚拟环境，也可以直接执行 `scripts/install_pytorch.sh`，脚本会升级 `pip` 并安装 CPU 版本的 PyTorch（使用官方 `https://download.pytorch.org/whl/cpu` 镜像）。

### 运行演示

请在仓库根目录执行模块。确保 `src/` 已包含在 `PYTHONPATH` 中（例如激活上面的虚拟环境），并使用 `-m` 方式运行：

```bash
PYTHONPATH=src python -m train_demo --rounds 3
# or, thanks to the `src/__init__.py` package initializer:
python -m src.character_sac_trainer --rounds 3
```

每轮训练固定遍历 `data/sample_article.txt` 的全部 76 个分割片段，因此每个迭代（iteration）恰好对应一次环境 step，`--rounds` 仅控制重复轮次（默认 1000 轮）。脚本会在完成 76 个交互后集中执行一批 SAC 更新，数量与步骤数一致，从而模拟“先收集一整轮经验，再统一回放训练”的节奏。需要缩减或扩充集中训练的强度时，可以通过 `--post-round-updates` 覆盖默认值；`--replay-capacity` 则依旧决定演示缓冲区能保留多少过往转换。针对快速冒烟测试，还可以附加 `--max-chapters 2`（或任意正整数）限制每轮使用的章节数量，从而在几次 step 内观察完整的日志与训练流程。

环境奖励通过衡量语义相似度、覆盖率与新颖度的加权组合来评估摘要质量，并额外扣除与乱码比例、词语合规缺失率成正比的惩罚项；所有指标都会在日志中打印，便于观察策略如何平衡保真度、改写度、编码质量与词语流畅性。

### 预期输出

该命令会打印精简的训练日志，汇总每个模拟 step 的奖励、回放缓冲区大小、占位的策略损失，以及质量诊断指标（长度比、相似度、覆盖率、新颖度）。示例输出：

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

由于演示采用随机采样的方式生成动作，具体数值会有所波动，但日志结构应与示例一致。每一步都会同时报告字符长度与当前输入片段的首/尾预览；在迭代摘要预览中也会直观呈现关键指标。完成 76 步后，训练器会打印阶段性汇总，包括各损失项与奖励的均值等，便于观察收敛趋势。

### 产物保存

日志结束后，脚本会生成 CSV/HTML 报表，将本次训练记录写入 `out/step_metrics.csv` 与 `out/round_metrics.csv`；此外会基于这些 CSV 自动生成一份可视化结果页 `out/rewards.html`，便于直接查看 Step 与 Round 的指标走势和位置统计。

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

## 文档摘要索引
<!-- DOCS-SUMMARY-INDEX:START -->
- $\texttt{docs/1758365560\_将阅读理解形式化为“认知资本”的交易与增值过程：基于传统数学的严格论证.md}$
  - 摘要：本文围绕：首先明确问题背景与约束，给出可验证的形式化定义与工程接口；随后分解系统/模型/数据/指标的关键设计，并给出可复现的实现与对齐路径；最后总结风险与边界条件，给出落地建议与扩展路线。
- $\texttt{docs/1758532267\_字符粒度策略环境 V2：无泄漏 POMDP + 离散最大熵 SAC（期望备份·Top‑p）.md}$
  - 摘要：本文面向字符级 POMDP 场景，系统化整理离散动作 SAC 的实现细节：策略/价值网络结构、温度/熵目标的自适应、Top-p 采样与合规 Mask 的协同，以及 CQL/BC/DAgger/EMA 等稳定训练技巧。结合生产日志与指标，给出从冷启动到稳态的调参与收敛路径，并讨论长序列与约束采样下的可观测性折中。
- $\texttt{docs/1758828679\_字符模式 SAC 的工程实现与数学化描述v1.0.0.md}$
  - 摘要：版本 v1.0.0 聚焦最小可用字符级 SAC：定义观测/动作/奖励与回放结构，给出策略与双 Q 网络的参数化与损失，提供训练循环与指标记录的标准模板。强调能跑通、易复现与可度量，为后续版本的稳态与性能优化打下基线。
- $\texttt{docs/1758831598\_字符模式 SAC 的工程实现与数学化描述v2.0.0.md}$
  - 摘要：版本 v2.0.0 在 v1 基线之上引入候选采样改进、奖励拆分与度量细化、目标网络与软更新策略，并完善日志与可视化管线。通过更稳定的超参与数据流，显著提升训练收敛性与可观测性，适配更长上下文与更严格的合规约束。
- $\texttt{docs/1758837092\_《字符模式 SAC 的工程实现与数学化描述》对中文知识蒸馏的意义.md}$
  - 摘要：本文围绕：首先明确问题背景与约束，给出可验证的形式化定义与工程接口；随后分解系统/模型/数据/指标的关键设计，并给出可复现的实现与对齐路径；最后总结风险与边界条件，给出落地建议与扩展路线。
- $\texttt{docs/1758838190\_可变词数×注意力长度（Flex-Attn）方案：架构说明与落地路线图.md}$
  - 摘要：阐述可变成本注意力（Flex-Attn）的动机、设计与实现：在合规约束与预算限制下，按需分配注意力计算资源。文中拆解组件与调用关系、关键超参与时间/显存开销，并给出与历史/状态缓存结合的工程实践与调优建议。
- $\texttt{docs/1758863280\_中文知识蒸馏基座的企业级价值评估：质量×成本×治理×扩展性.md}$
  - 摘要：从价值偏置与微分耦合出发，连接 PFB-GNLA/KAT/GRL 的理论与工程：刻画可解释的约束梯度与通信惩罚，分析对收敛路径与泛化边界的影响。结合可复现实验，给出可检验的结论与实用建议。
- $\texttt{docs/1758863280\_医疗问答端到端示例：Flex-Attn 生成“奥司他韦”专业定义.md}$
  - 摘要：阐述可变成本注意力（Flex-Attn）的动机、设计与实现：在合规约束与预算限制下，按需分配注意力计算资源。文中拆解组件与调用关系、关键超参与时间/显存开销，并给出与历史/状态缓存结合的工程实践与调优建议。
- $\texttt{docs/1758863280\_零训练表驱动 Flex-Attn：可计算词法 + 有限状态索引的快速落地.md}$
  - 摘要：阐述可变成本注意力（Flex-Attn）的动机、设计与实现：在合规约束与预算限制下，按需分配注意力计算资源。文中拆解组件与调用关系、关键超参与时间/显存开销，并给出与历史/状态缓存结合的工程实践与调优建议。
- $\texttt{docs/1758865953\_词法KAT作用幺半群.md}$
  - 摘要：介绍 Kleene Algebra with Tests（KAT）与相关闭包/半环结构在本项目中的角色：用以建模可验证控制流、停机点与合规模式。提供从数学结构到工程接口的映射规范，支撑规则检查、代价累积与策略约束的统一表达。
- $\texttt{docs/1758867891\_价值偏好向量（微分动力量子）的构造：PFB-GNLA 退化下的词法KAT作用幺半群 × GRL路径积分.md}$
  - 摘要：介绍 Kleene Algebra with Tests（KAT）与相关闭包/半环结构在本项目中的角色：用以建模可验证控制流、停机点与合规模式。提供从数学结构到工程接口的映射规范，支撑规则检查、代价累积与策略约束的统一表达。
- $\texttt{docs/1758867891\_词法KAT作用幺半群的幂子幺半群谱系（规范与工程用法）.md}$
  - 摘要：介绍 Kleene Algebra with Tests（KAT）与相关闭包/半环结构在本项目中的角色：用以建模可验证控制流、停机点与合规模式。提供从数学结构到工程接口的映射规范，支撑规则检查、代价累积与策略约束的统一表达。
- $\texttt{docs/1758870348\_“微分动力量子（MDQ）”在离散化LLM的工程化落地：最小单元、线性积累、热插拔与统一版本治理.md}$
  - 摘要：提出 MDQ 机制稳定离散 LLM/策略管道：支持小单元交互与统一版本控制，缓解长序列采样的非平稳与暴露偏差。结合指令设计与记忆扩展策略，给出训练/推理一体化的实现路线与评估指标。
- $\texttt{docs/1758870348\_基于传统数学语言的形式化：PFB-GNLA 退化 × 词法KAT作用幺半群 × GRL路径积分中的“价值偏好向量与微分动力量子”.md}$
  - 摘要：介绍 Kleene Algebra with Tests（KAT）与相关闭包/半环结构在本项目中的角色：用以建模可验证控制流、停机点与合规模式。提供从数学结构到工程接口的映射规范，支撑规则检查、代价累积与策略约束的统一表达。
- $\texttt{docs/1758899963\_神经网络等价解耦与“三层分治”（MDQ 网络 × 索引泛函 × OOV 内存库）落地方案.md}$
  - 摘要：提出 MDQ 机制稳定离散 LLM/策略管道：支持小单元交互与统一版本控制，缓解长序列采样的非平稳与暴露偏差。结合指令设计与记忆扩展策略，给出训练/推理一体化的实现路线与评估指标。
- $\texttt{docs/1758899963\_这套理论是否“巧妙”：结论与十条硬核巧思.md}$
  - 摘要：本文围绕：首先明确问题背景与约束，给出可验证的形式化定义与工程接口；随后分解系统/模型/数据/指标的关键设计，并给出可复现的实现与对齐路径；最后总结风险与边界条件，给出落地建议与扩展路线。
- $\texttt{docs/1758907821\_这套理论对“字符级RL奖励稀疏”世界级难题的实质性贡献（企业口径，长文版）.md}$
  - 摘要：本文围绕：首先明确问题背景与约束，给出可验证的形式化定义与工程接口；随后分解系统/模型/数据/指标的关键设计，并给出可复现的实现与对齐路径；最后总结风险与边界条件，给出落地建议与扩展路线。
- $\texttt{docs/1758911505\_这套理论对“字符级RL奖励稀疏”世界级难题的实质性贡献.md}$
  - 摘要：本文围绕：首先明确问题背景与约束，给出可验证的形式化定义与工程接口；随后分解系统/模型/数据/指标的关键设计，并给出可复现的实现与对齐路径；最后总结风险与边界条件，给出落地建议与扩展路线。
- $\texttt{docs/1758952640\_字符模式 SAC 的工程实现与数学化描述v3.0.0.md}$
  - 摘要：在 v2.0.0 基于“长度集合 U 的可变后缀命中”基础上，v3.0.0 将“向前拓扑命中”从单一词扩展为“拓扑词包命中”（可配置的一组词/短语，支持非交换的专有词组），并形式化为“拓扑词包算子”；同时将“向后拓扑”从单字符扩展为“迭代多字符预测”，定义为“多字符迭代算子”。这两类算子以统一接口接入合规模块与奖励记录，兼容 v1/v2 的行为，并通过配置文件灵活开关与调参，便于在产线场景下做可审计、可回放的策略治理。
- $\texttt{docs/1758967896\_字符模式 SAC 的工程实现与数学化描述v3.0.1.md}$
  - 摘要：在 v3.0.0 基于“拓扑词包（向前）+ 多字符迭代（向后）”的框架上，v3.0.1 进一步强调“尾缀的可词包性”：不仅向前拓扑在 $s=\chi_t\oplus q$ 的尾部可匹配词包，向后的“迭代尾缀”也允许直接对“后缀词包”命中，从而以统一的“词包语义”覆盖前后两个方向。本文给出后缀词包的形式化定义、与多字符迭代的融合伪代码、配置与日志扩展，以及回滚与评审要点，确保升级在可观测性、稳定性与合规治理下落地。
<!-- DOCS-SUMMARY-INDEX:END -->
