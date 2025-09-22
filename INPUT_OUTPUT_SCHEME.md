# 当前输入-输出数据方案说明

## 数学化描述
设认知资本状态记为 $C_t=(F_t,V_t,H_t)$，预算为 $B_t$，章节文本序列为 $\{\chi_t\}_{t=1}^T$。
第 $t$ 步观测表示为 $O_t = (\sigma(C_{t-1}, B_{t-1}), \chi_t)$，其中 $\sigma$ 为文本渲染函数 `CognitiveCapital.render_text`。
字符级编码器 $f_{\text{enc}}$ 将观测转为 token 序列 $x_t = f_{\text{enc}}(O_t)$，其实现为 `CharTokenizer.encode_observation`。
策略网络 $\pi_\theta$ 在 $x_t$ 上解码得到文本 $y_t$，解析器 $g_{\text{parse}}$ 产出操作序列 $A_t = g_{\text{parse}}(y_t) = (a_1,\dots,a_{m_t})$。
环境根据 `OperationParser.parse` 解析结果逐条调用 `CognitiveCapital.apply`，得到 $C_t = \Gamma(C_{t-1}, A_t)$，预算更新为 $B_t = B_{t-1} - \sum_{a\in A_t} \mathrm{cost}(a)$。

## 数据结构映射
| 符号 | 描述 | 实现对照 |
| --- | --- | --- |
| $\sigma(C_{t-1}, B_{t-1})$ | 观测中的认知资本文本快照 | `CognitiveCapital.render_text` |
| $\chi_t$ | 当前章节原文（章节模式） | `ArticleEnvironment._chapters[t]` |
| $\hat\chi_t$ | 字符模式下的隐藏二元组（历史+候选） | `ArticleEnvironment._char_truth_pairs[t]` |
| $\tilde\chi_t$ | 字符模式下的目标字符 | `ArticleEnvironment._char_targets[t]` |
| $O_t$ | 完整观测 $(\sigma(C_{t-1},B_{t-1}), \chi_t)$，字符模式时 $\chi_t=\varnothing$ | `TextObservation(previous_summary, chapter_text)` |
| $x_t$ | Token 序列 | `CharTokenizer.encode_observation` 返回的整型数组 |
| $y_t$ | 策略生成的字符文本 | `CharTokenizer.decode_action` |
| $A_t$ | 解析后的操作列表 | `OperationParser.parse` |
| $C_t$ | 更新后的认知资本 | `CognitiveCapital.apply` 累积结果 |
| $B_t$ | 扣减后的预算 | `ArticleEnvironment._budget` |
| $r_t$ | 奖励 $(B_t, \Delta\Phi_t, S_t)$ | `ArticleEnvironment.step` 组装的 `metrics['reward']` |

## 编码与解析流程
- 观测拼接：章节模式下拼接 `[<bos>] + summary + [<sep>] + chapter + [<eos>]`；字符模式仅编码上一字符 `[<bos>] + summary + [<eos>]`，隐藏对 $(\hat\chi_t)$ 仅用于奖励与日志。
- Token 化：`CharTokenizer.encode_observation` 将字符映射到词表索引，词表包含 `<pad>/<bos>/<eos>/<sep>/<unk>` 以及章节字符。
- 策略解码：`TextPolicyNetwork` 以 $x_t$ 为输入，解码获得 `token_ids` 与文本 $y_t$。
- 操作解析：`OperationParser.parse` 将 $y_t$ 按行映射为结构化 `Operation(kind, payload)`。
- 预算演化：每条指令消耗 `OPERATION_COSTS` 中的费用，未命中时用 `DEFAULT_OPERATION_COST`，累计到 $B_t$ 与 $\bar{c}_t$。
- 潜能与奖励：`CapitalValuator.metrics` 与 `value` 生成估值、潜能差与日志字段，供奖励公式使用。

## 环境状态转移伪代码
```pseudo
function STEP(environment_state, policy):
    state_text = render(C_{t-1}, B_{t-1})
    if mode == "character":
        pair = hidden_pairs[t]                  # 例如 “意味”
        observation = TextObservation(pair[0], "", t)
        target_pair = pair
        source_text = pair
    else:
        observation = TextObservation(state_text, chi_t, t)
        target_pair = chi_t
        source_text = combine(state_text, chi_t)
    tokens = tokenizer.encode_observation(observation)
    action_ids, info = policy(tokens)
    action_text = tokenizer.decode_action(action_ids)
    operations = OperationParser.parse(action_text)           # 字符模式返回空列表
    cost = sum(OPERATION_COSTS.get(op.kind, DEFAULT_OPERATION_COST) for op in operations)
    capital_before = capital.clone()
    potential_before = valuator.value(capital_before)
    for op in operations:
        capital.apply(op)
    budget = budget - cost
    potential_after = valuator.value(capital)
    metrics = analyze_summary(action_text, source_text)
    if mode == "character" and source_text.length == 2 and source_text in lexical_bigram_set:
        lexical_bonus = CHARACTER_LEXICAL_BIGRAM_BONUS
    else:
        lexical_bonus = 0
    reward = compute_reward(metrics, capital, budget, potential_before, potential_after, cost)
    reward += lexical_bonus
    next_text = capital.render_text(budget)
    next_observation = TextObservation(next_text, chi_{t+1}, t+1)
    return Transition(observation, action, reward, next_observation, done)
```

## 日志与缓存字段
- 步级指标：`metrics` 中写入 `summary_length`、`similarity`、`coverage_ratio`、`lexical_cosine`、`lexical_js_similarity`、`garbled_ratio`、`word_noncompliance_ratio` 等字段并导出到 CSV。
- 资本估值：`CapitalValuator.metrics` 输出 `capital_value`、`capital_coverage`、`capital_diversity`、`capital_redundancy`、`capital_verification_ratio`、`capital_fact_count`。
- 预算记录：输出 `budget_remaining`、`budget_breach`、`operation_cost`、`cumulative_cost` 以便分析资源消耗。
- 回放缓存：`SimpleReplayBuffer.add` 存储 `Transition(state, action, reward, next_state, done)`，供 `DemoSACAgent.update` 抽样。
- 字符二元奖励：字符模式额外输出 `lexical_bigram_bonus`（匹配词汇表二元组合时的奖励），便于监控拓扑记忆效果。

若调整观测格式、操作类型或估值指标，需要同步修改代码与本文档以保持一致。
