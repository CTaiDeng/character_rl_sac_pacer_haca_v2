# 迭代粒度设计说明

本文档基于《docs/1755455209_将阅读理解形式化为“认知资本”的交易与增值过程：基于传统数学的严格论证.md》，梳理本仓库中“阅读迭代-认知资本”评估体系的实现方案，并说明新增“段落粒度”训练模式与配置管理的设计决策。

## 背景与目标

论文将阅读理解过程抽象为「认知资本」在信息资产之间的交易与增值：

- **资产拆分**：文本被划分为结构化片段，逐步暴露给智能体；
- **资本状态**：每一步通过操作（ACQUIRE / VERIFY / LINK / …）更新知识库；
- **价值评估**：资本价值由覆盖率、多样性、验证度等指标加权计算，兼顾成本与预算惩罚。

项目最初以“章节”为最小迭代单位：每一轮遍历一个章节，章节内部只有 1 个 step。为了更细致地模拟论文中“持续交易、细颗粒度增值”的场景，需要在不破坏原有章节级流程的前提下，引入“段落粒度”的迭代机制：

1. **保持章节作为预算与评估的基本周期**（一轮 = 一章）；
2. **细化 step 为段落**，使资本在同一章节内多次增值；
3. **保持旧逻辑兼容**，可随配置切换章节模式 / 段落模式；
4. **提供明确的配置项枚举**，便于在 config 模板中说明可选方案。

## 结构设计

### 配置管理

- 新增 $\texttt{iteration\_granularity}$ 配置字段，枚举值 $\texttt{"chapter"}$ 与 $\texttt{"paragraph"}$；
- 补充 $\texttt{iteration\_granularity\_options}$（枚举提示）、$\texttt{paragraph\_split\_min\_length}$、$\texttt{paragraph\_merge\_strategy}$ 等字段，允许控制段落划分的最小长度及合并策略；
- $\texttt{\_load\_training\_config}$ 解析上述字段，并在启动时通过 $\texttt{\_announce\_training\_config}$ 明确打印配置路径与 JSON 内容；
- 引入 $\texttt{\_initialize\_run\_paths()}$，每次训练输出写入 $\texttt{out/train\_{timestamp}}$ 目录，保证多次实验互不覆盖，同时在 LOG 中标记本次运行目录。

### 文本粒度与环境

- 章节粒度沿用现有流程：$\texttt{load\_article\_features()}$ 产出章节列表，$\texttt{ArticleEnvironment}$ 在单步内消费整章；
- 段落粒度新增 $\texttt{\_split\_into\_paragraphs()}$，支持按照空行划分段落并基于最小长度合并短段；
- $\texttt{build\_demo\_components()}$ 根据配置生成：
  - $\texttt{environment\_segments}$：章节模式下为章节列表，段落模式下为整本书按段落展开的序列；
  - $\texttt{per\_round\_intervals}$：段落模式下为 $\texttt{List[List[str]]}$，每个子列表对应一个章节的段落序列；
  - $\texttt{observations\_for\_seed}$：段落模式下为段落级 $\texttt{TextObservation}$，以章节编号标识 $\texttt{CHxx}$；
- $\texttt{ArticleEnvironment}$ 新增 $\texttt{configure(chapters)}$ 方法，可在不重建对象的情况下切换当前可见的片段序列，并重新初始化词合规器与资本估值器。

### 训练调度

- $\texttt{DemoTrainer}$ 扩展 $\texttt{per\_round\_intervals}$ 参数，若存在则在每轮开始前调用 $\texttt{environment.configure()}$ 设置当前章节的段落 / 字符序列，并动态调整 $\texttt{total\_steps}$；
- 字符模式在 $\texttt{character\_teacher\_interval}$ 配置下周期性注入真实目标字符，实现可调节的 Teacher Forcing；
- 策略热身与日志逻辑保持不变，但 $\texttt{log\_metrics}$ 现已完整写入 $\texttt{capital\_value}$、$\texttt{budget\_remaining}$ 等指标，配合 $\texttt{:.6f}$ 精度便于诊断评分体系；
- $\texttt{render\_iterative\_summary()}$ 支持段落与字符模式，内部将 $\texttt{per\_round\_intervals}$ 展平成统一序列，便于输出全局资本演化轨迹；
- 章节模式仍按原逻辑运行，保证旧脚本无感知升级。

### 输出与快照

- 日志、CSV、HTML、快照等输出统一写入 $\texttt{RUN\_DIR}$；
- 每轮快照依旧在 $\texttt{round\_snapshots/}$ 下生成，路径相对运行目录，方便对比不同粒度下的策略表现；
- $\texttt{save\_agent\_snapshot()}$、$\texttt{save\_model\_artifact()}$ 亦指向 run 目录，避免跨实验混淆。

## 小结

通过引入可配置的粒度控制、段落划分与环境动态配置机制，系统可以：

1. 继续支持章节一次迭代的传统流程；
2. 在需要更细颗粒度时，可切换为“章节=轮、段落=步”或“章节=轮、字符=步”，模拟更频繁的资本交易；
3. 通过运行目录隔离和高精度日志，反向验证评分体系的稳定性。

后续若需扩展为句子级、图谱级等更细粒度，只需复用现有配置框架与环境切换逻辑即可。
