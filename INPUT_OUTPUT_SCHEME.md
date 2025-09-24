# 当前输入-输出数据方案

## 符号约定
- $C_t = (F_t, V_t, H_t)$：第 $t$ 步结束后的认知资本，含事实、验证、对冲集合。
- $B_t$：第 $t$ 步后的预算余额；$B_0 = \texttt{DEFAULT\_INITIAL\_BUDGET}$。
- $\chi_t$：第 $t$ 段原始章节文本；字符模式下为空串。
- $O_t$：环境暴露给策略的观测，包含渲染文本与章节/目标提示。
- $x_t \in \mathbb{N}^{L_t}$：经 tokenizer 编码后的 token 序列。
- $y_t \in \mathbb{N}^{M_t}$：策略网络输出的 token 序列。
- $A_t = (a_1, \dots, a_{m_t})$：解析得到的结构化操作列表。
- $r_t$：根据 `STEP_SCORING.md` 计算的即时奖励。

## 端到端流程
1. 观测渲染：
   \[
   O_t = (\sigma(C_{t-1}, B_{t-1}), \chi_t),\qquad \sigma = \texttt{CognitiveCapital.render_text}
   \]
   字符模式时设 $\chi_t = \varnothing$，仅暴露上一轮摘要与预算。
2. 编码：
   \[
   x_t = f_{\text{enc}}(O_t) = \texttt{CharTokenizer.encode_observation}(O_t)
   \]
   该函数负责拼接 `[<bos>]`、`[<sep>]`、`[<eos>]` 等特殊标记并做零填充。
3. 决策：策略 $\pi_\theta$ 作用于 $x_t$，输出 token 序列
   \[
   y_t \sim \pi_\theta(\cdot \mid x_t)
   \]
   解码函数 `decode_action` 将索引转换为可读文本。
4. 解析：
   \[
   A_t = g_{\text{parse}}(y_t) = \texttt{OperationParser.parse}(y_t)
   \]
   每个 $a_i = (\mathrm{kind}, \mathrm{payload})$ 对应一次对 `CognitiveCapital.apply` 的调用。
5. 环境更新：
   \[
   C_t = \Gamma(C_{t-1}, A_t),\qquad B_t = B_{t-1} - \sum_{a \in A_t} \mathrm{cost}(a)
   \]
   更新后通过 `STEP_SCORE` 得到奖励 $r_t$ 与潜力增量。
6. 经验写入： `Transition(O_t, y_t, r_t, O_{t+1}, \text{done})` 存入 `SimpleReplayBuffer`。

## 数据结构映射
| 概念 | 描述 | 代码位置 |
| --- | --- | --- |
| $\sigma(C_{t-1}, B_{t-1})$ | 观测中的认知资本文本快照 | `src/train_demo.py:977` 附近的 `render_text` |
| $\chi_t$ | 当前章节或字符目标 | `ArticleEnvironment._chapters[t]` / `_char_truth_pairs[t]` |
| $O_t$ | 观测对象 | `TextObservation` 数据类 |
| $x_t$ | Token 序列 | `CharTokenizer.encode_observation` 返回的张量 |
| $y_t$ | 文本动作 | `TextPolicyNetwork.forward` + `decode_action` |
| $A_t$ | 解析后的操作 | `OperationParser.parse` |
| $C_t$ | 新的认知资本 | `CognitiveCapital.apply` 累积结果 |
| $B_t$ | 剩余预算 | `ArticleEnvironment._budget` |
| $r_t$ | 即时奖励 | `ArticleEnvironment.step` 调用 `compute_reward` |
| 经验缓存 | SAC 经验条目 | `SimpleReplayBuffer.add` |

## 观测与动作拼接规则
- 文本模式：`[<bos>] + summary + [<sep>] + chapter + [<eos>]`。
- 字符模式：`[<bos>] + summary + [<eos>]`，并额外记录 `target_char`（当前章节字符）。
- Token 化时保留 `<pad>/<bos>/<eos>/<sep>/<unk>`，长度超限时截断到 `max_observation_length`。
- 策略首 token 由 `first_step_distribution` 采样，后续 token 使用 Top-$p$ ($p=0.98$) 过滤与合规掩码。

## 字符模式 bigram 定义（更新）
- 候选计算：`bigram = chapter_char ⊕ raw_action_char`。
- raw_action_char 提取：
  - 教师动作（常见为 `chapter_char + next_char`）：若 `action.text` 以 `chapter_char` 开头且长度≥2，则取 `action.text` 的末字；
  - 其他情况：取 `action.text` 的首字。
- 记录位置：`metrics['lexical_bigram_candidate']` 记录该 bigram；命中词表时 `lexical_bigram_bonus` 取 `CHARACTER_LEXICAL_BIGRAM_BONUS`，否则在 `match_char` 时回退 `CHARACTER_TEACHER_BIGRAM_FALLBACK`。

## 输入输出伪代码
```pseudo
function ENV_STEP(state, policy, replay_buffer):
    observation = TextObservation(
        render_text(state.capital, state.budget),
        current_chapter(state.mode, state.index),
        state.index
    )
    tokens = tokenizer.encode_observation(observation)

    action_ids, logits = policy(tokens)
    action_text = tokenizer.decode_action(action_ids)
    operations = OperationParser.parse(action_text)

    capital_before = state.capital.clone()
    budget_before = state.budget
    capital_after = capital_before
    for op in operations:
        capital_after = capital_after.apply(op)
    budget_after = budget_before - total_cost(operations)

    reward, capital_after, budget_after = STEP_SCORE(
        state.capital, state.budget, operations, analyze_summary(action_text), state.done
    )

    # 字符模式 bigram：chapter + raw_action
    if mode == "character":
        chapter_char = target_char
        if startswith(action_text, chapter_char) and length(action_text) >= 2:
            raw_action_char = last(action_text)
        else:
            raw_action_char = first(action_text)
        bigram = chapter_char ++ raw_action_char

    next_observation = TextObservation(
        render_text(capital_after, budget_after),
        next_chapter(state.mode, state.index),
        state.index + 1
    )
    replay_buffer.add(observation, action_ids, reward, next_observation, state.done)

    return next_observation, reward, capital_after, budget_after
```

## 日志与缓存字段
- `metrics`: 记录 `similarity`、`coverage_ratio`、`novelty_ratio`、`lexical_cosine`、`lexical_js_similarity`、`garbled_ratio`、`word_noncompliance_ratio`，写入 `out/metrics_logs/*.csv`。
- `capital_metrics`: 由 `CapitalValuator.metrics` 产生，含 `capital_value`、`topic_coverage`、`fact_count` 等字段，用于调试面板。
- 预算轨迹：`budget_remaining`、`budget_breach`、`operation_cost`、`cumulative_cost` 持久化于 episode summary。
- 字符模式额外日志：`predicted_char`、`target_char`、`lexical_bigram_bonus`、`lexical_bigram_candidate`（按上节定义）。

## 一致性约束
- 若 `CharTokenizer` 的特殊标记或 `max_observation_length` 调整，需同步更新观测拼接公式。
- 若操作集合或成本表变动，需同步更新 $\mathrm{cost}(\cdot)$ 及预算更新逻辑。
- 新增指标时请在 `metrics` 字典中添加键名，并在 `STEP_SCORING.md` 中补充对应权重。
