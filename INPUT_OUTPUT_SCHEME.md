# 当前输入-输出数据方案说明

## 总览
每个训练轮次以章节为粒度推进。第 t 步的环境观测由三部分组成：
1. 认知资本快照 `capital_snapshot_{t-1}`，以字符串形式呈现现有事实、验证记录和预算；
2. 当前章节原文 `chapter_t`；
3. 剩余预算 `budget_t`，被编码进资本快照的前缀。

策略网络输出一串字符，解析后得到一系列操作（如 ACQUIRE、EXTRACT、VERIFY 等），环境据此更新认知资本并扣减预算。

## 数据结构
| 符号 | 含义 | 具体来源 |
| --- | --- | --- |
| `capital_snapshot_0` | 初始认知资本（空集合与满额预算） | 环境 `reset()` 时由 `CognitiveCapital` 构造 |
| `chapter_t` | 第 t 章原文 | `load_article_features()` 从 `data/sample_article.txt` 切分 |
| `observation_t` | 观测结构 | `TextObservation(previous_summary=capital_snapshot_{t-1}, chapter_text=chapter_t)` |
| `action_t` | 字符级操作序列 | `TextAction(text=policy_output, token_ids, length)` |
| `capital_t` | 更新后的认知资本 | `CognitiveCapital.apply()` 依次执行解析出的操作 |
| `budget_t` | 剩余预算 | 初始值 `DEFAULT_INITIAL_BUDGET`，每个操作按 `OPERATION_COSTS` 扣减 |
| `valuation_t` | 资本估值 | `CapitalValuator.value()` 基于覆盖、多样、冗余与验证计算 |

## 操作解析与预算演化
- 策略输出按行划分操作，允许的指令包括 `ACQUIRE`, `EXTRACT`, `LINK`, `VERIFY`, `HEDGE`, `TRIM`, `COMMIT`，以及对应缩写（ACQ、EXT、LNK 等）。
- `OperationParser.parse()` 将文本映射为结构化 `Operation` 列表，环境逐条执行。
- 每条操作消耗固定成本，成本来源 `OPERATION_COSTS`，若未定义则使用 `DEFAULT_OPERATION_COST`。
- 预算为浮点数，可透支；当预算小于零时会触发额外惩罚项。

## 估值与奖励
- `CapitalValuator.metrics()` 提供覆盖率、多样性、冗余度、验证比例等指标；`value()` 在此基础上累加奖励权重并对冲对冲行为带来的折扣。
- 基础回报仅在终止步骤给出：`capital_value - COST_WEIGHT * cumulative_cost - BUDGET_PENALTY_WEIGHT * budget_breach`。
- 为保持最优策略不变，引入潜能塑形：每步奖励额外累加 `potential(capital_t) - potential(capital_{t-1})`。潜能函数即 `CapitalValuator.potential()`，与估值等价。

## 迭代流程伪代码
```pseudo
initial_capital <- CognitiveCapital()
budget <- DEFAULT_INITIAL_BUDGET
capital_text <- initial_capital.render_text(budget)
for t in 1..T:
    observation <- TextObservation(previous_summary=capital_text, chapter_text=chapter_t)
    obs_tokens <- tokenizer.encode_observation(observation)
    action_tokens <- policy.sample(obs_tokens)
    action_text <- tokenizer.decode_action(action_tokens)
    operations <- OperationParser.parse(action_text)
    step_cost <- 0
    for op in operations:
        capital.apply(op)
        step_cost += OPERATION_COSTS.get(op.kind, DEFAULT_OPERATION_COST)
    budget <- budget - step_cost
    valuation <- valuator.value(capital)
    reward_t <- shaped_reward(valuation, step_cost, budget)
    capital_text <- capital.render_text(budget)
    buffer.push(state=observation, action=TextAction(token_ids=action_tokens, text=action_text, length=len(action_text)), reward=reward_t, next_state=next_observation)
```

## 输入与输出要点
- **观测编码**：`CharTokenizer.encode_observation` 将资本快照、章节文本与界定符 `[BOS]/[SEP]/[EOS]` 拼接成字符 ID 序列。
- **行动编码**：策略输出同样通过字符级解码得到文本指令，仍使用统一词表，避免临时 token。
- **认知资本**：`CognitiveCapital` 维护事实集合、验证集合、对冲集合与链接集合，并提供 `render_text()` 用于形成下一步观测。
- **预算约束**：预算完整保留在环境内部，透支时在奖励中追加惩罚，同时通过日志字段 `budget_remaining`、`budget_breach` 对外暴露。
- **日志指标**：环境为每步记录资本估值、覆盖、多样、冗余、验证比例、操作数量与成本等指标，并写入 `STEP_CSV_HEADERS`。

> 若后续扩展操作集合或估值模型（例如引入风险度量、约束乘子），需同步更新本说明，保持实现与文档一致。
