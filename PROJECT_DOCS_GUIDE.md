# 项目工程文档说明（根目录）

> 摘要：本文对根目录下的 5 篇工程设计文档给出作用定位、适用范围、关键字段与同步维护规则，形成“文档矩阵”。这 5 篇分别是：STEP_SCORING.md（奖励函数）、NETWORK_TOPOLOGY.md（网络拓扑）、ITERATIVE_SUMMARY_TEMPLATE.md（迭代摘要模板）、ITERATION_GRANULARITY_DESIGN.md（迭代粒度方案）与 INPUT_OUTPUT_SCHEME.md（输入输出方案）。当任意方案发生调整（术语/符号/流程/参数/版本等）时，需同步更新本文对应条目，并按需归档到 `engineering_docs_archive`。本文亦给出版本发布与归档的简要伪代码流程，确保工程信息的一致性与可追溯性。

---

## 一、文档矩阵与职责

- STEP_SCORING.md（Step 打分方案）
  - 职责：定义奖励函数 r_t 及子项（覆盖 cov、相似 sim、新颖 nov、简洁 conc、复读惩罚 rep）与权重聚合；给出数学公式、边界处理与 `compute_reward()` 伪代码。
  - 变更触发：子项定义、权重、归一化、长度/重复策略、编码器/相似度度量。

- NETWORK_TOPOLOGY.md（网络拓扑结构方案）
  - 职责：微型 LLM 头部（单向 Transformer）的模块结构、维度、注意力/前馈表达式、推理与训练（SFT + 策略梯度）伪代码，推荐超参与工程要点（KV Cache、RoPE、混精等）。
  - 变更触发：层数/宽度、注意力/位置编码变体、损失与优化策略、推理采样策略。

- ITERATIVE_SUMMARY_TEMPLATE.md（迭代摘要模板）
  - 职责：描述生成摘要时的提示模板/格式、占位符约定、语言输出偏好与采样策略对齐（如“仅用简体中文输出”）。
  - 变更触发：模板内容与字段、更换解码策略对模板的要求、输出风格与长度约束。

- ITERATION_GRANULARITY_DESIGN.md（迭代粒度设计）
  - 职责：定义“上一轮摘要 + 当前章节全文 → 新摘要”的粒度与循环策略，切分单位、步长与边界；与数据分段符的关系与容错策略。
  - 变更触发：分段规则、步长策略、对空段/短段的处理、截断/拼接策略变化。

- INPUT_OUTPUT_SCHEME.md（输入-输出方案）
  - 职责：定义 `data/sample_article.txt` 分段解析、输入拼接 `s_{t-1} ⊕ <SEP> ⊕ c_t`、最大长度 `L_in/L_out`、批处理/缓存、日志 JSONL 字段与示例。
  - 变更触发：分隔符、字段名、长度上限、日志结构、批/缓存策略。

## 二、跨文档的统一约定

- 术语与符号：s_t、c_t、n-gram、φ(·) 等在 5 篇文档中需保持一致；若变更，统一替换并在本文“文档矩阵”中注明。
- 模板与语言：输出语言与格式约束（如“仅用简体中文”）由模板文档主导，评分与拓扑需与之兼容。
- 版本规范：采用 `vMAJOR.MINOR.PATCH`；语义版本号反映兼容性。
- 归档目录：历史版本放置在 `engineering_docs_archive/` 下（不做自动维护），命名为原名后缀 `_vX.Y.Z.md`。

## 三、版本发布与归档（伪代码）

```python
def release_docs(version: str):
    # 1) 更新 5 篇文档中的“版本/变更历史”段落（如有）
    bump_versions(
        files=[
            'STEP_SCORING.md',
            'NETWORK_TOPOLOGY.md',
            'ITERATIVE_SUMMARY_TEMPLATE.md',
            'ITERATION_GRANULARITY_DESIGN.md',
            'INPUT_OUTPUT_SCHEME.md',
        ], version=version)

    # 2) 复制到历史归档目录（不自动维护）
    for f in files:
        copy(f, f'engineering_docs_archive/{stem(f)}_v{version}.md')

    # 3) 同步本说明文件（矩阵/约定/流程处的相关版本描述）
    update_project_docs_guide(version)

    # 4) 提交建议
    git_commit(msg=f'docs: 发布工程文档 {version} 并归档')
```

## 四、同步维护 Checklist

- 方案变更时：
  - 更新对应主文档；
  - 若影响术语/接口/长度/边界，联动更新其余受影响文档；
  - 更新本文“文档矩阵”相关条目；
  - 如为“版本发布”，复制到 `engineering_docs_archive/` 归档，并在提交信息中体现版本号。
- 文件名规范：统一使用小写 `.md` 扩展名；禁止 `.MD/.Md` 等变体。
- README 索引：README 文末的 docs 摘要索引仅覆盖 `docs/*.md`，不包含根目录工程文档与 `engineering_docs_archive`。

## 五、附注

- 本说明仅覆盖根目录工程文档；`docs/` 知识库文章仍按“文档摘要同步规范”与对齐脚本维护。
- 归档目录不参与自动规范化与标题校验，避免历史版本被改写。

