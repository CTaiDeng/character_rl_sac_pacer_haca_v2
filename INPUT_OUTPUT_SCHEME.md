# 当前输入-输出数据方案

## 更新要点（需与代码保持一致）
- 字符模式观测采用 prev + <sep> + chapter，其中 chapter=当前目标字符，不再为空串。
- 历史回溯拓扑：在字符模式日志渲染中，若 `source=prev+chapter` 的前缀两字未命中词表，则沿“历史字符对”向左逐步扩展 prev，至多 16 次，直到前缀两字命中或达到上限。
- 允许的后缀词长来源于 `data/word_length_sets.json` 的并集长度集合 `union.lengths`，用于 raw_action/bigram 的后缀命中判定。

## 字符模式输入构成（prev+sep+chapter）
设时刻 t 的观测为 O_t=(s_t, χ_t, i_t)，其中 χ_t 为当前目标字符。
令编码器输出序列为
  x_t = [<bos>] ⊕ clip(s_t) ⊕ [<sep>] ⊕ χ_t ⊕ [<eos>]
并截断至 `max_observation_length`。

伪代码：
```pseudo
function ENCODE_OBSERVATION(prev_summary, chapter_char):
    tokens = [<bos>] + encode(prev_summary)
    tokens += [<sep>] + encode(chapter_char) + [<eos>]
    return clip(tokens, max_observation_length)
```

## 配置参数（与 res/config.json 同步）
- character_history_extension_limit = 16
  - 用于字符模式日志渲染时的“历史回溯拓扑”最大步数；达到阈值即停止左扩 prev。
- iteration_granularity ∈ {chapter, paragraph, character}，字符模式时 `chapter_text` 即为目标字符。


## 状态与观测
- 记忆状态 $C_t$ 由 `CognitiveCapital` 管理，预算 $B_t$ 为环境中的剩余成本额度。
- 第 $t$ 步观测 $O_t = (s_t, \chi_t, i_t)$：$s_t = \sigma(C_{t-1}, B_{t-1})$ 来自 `CognitiveCapital.render_text`，$\chi_t$ 为章节文本或字符目标，$i_t$ 为游标索引。
- 在章节模式下 $\chi_t = \text{chapters}[t]$；字符模式下 $\chi_t$ 为当前目标字符（`src/train_demo.py` 运行时由环境提供）。

## 编码函数
- `CharTokenizer.encode_observation` (`src/train_demo.py:543-625`) 构造 token 序列
  \[
  x_t = [\texttt{<bos>}] \oplus \mathrm{clip}(s_t) \oplus
  \begin{cases}
  [\texttt{<sep>}] \oplus \mathrm{clip}(\chi_t), & \text{章节模式},\\
  [], & \text{字符模式},
  \end{cases}
  \oplus [\texttt{<eos>}],
  \]
  并截断至 `max_observation_length`。
- Tokenizer 维护 `summary_token_ids` 供合规检查、`pad`/`unk` 等特殊标记。

## 策略输出
- `TextPolicyNetwork` (`src/train_demo.py:2371-2525`) 接收 batch 化的 $x_t$，先用 GRU 编码，再逐步解码 action tokens：
  \[
  y_t \sim \pi_\theta(\cdot \mid x_t), \qquad y_t = (y_{t,1},\dots,y_{t,m}).
  \]
- `first_step_distribution` 返回首个 token 的概率、对数概率与合规 mask，Top-$p$ 采样阈值默认 $p=0.98$ (`DemoSACAgent.top_p`)。

## 操作解析
- `OperationParser.parse` (`src/train_demo.py:740-947`) 将 $y_t$ 解析为结构化操作列表 $A_t = (a_1,\dots,a_{m_t})$，每个 $a_i = (\mathrm{kind}, \mathrm{payload})$。
- 合法指令集合包含 `ACQUIRE`/`EXTRACT`/`LINK`/`VERIFY`/`HEDGE`/`TRIM`/`COMMIT` 等，失败解析会返回空序列并保留原文本供日志分析。

## 环境转移
- 环境执行器 (`ArticleEnvironment.step`, `src/train_demo.py:2142-2280`) 根据 $A_t$ 更新资本与预算：
  \[
  C_t = \Gamma(C_{t-1}, A_t), \qquad B_t = B_{t-1} - \sum_{a\in A_t} c(a).
  \]
- 同步计算指标 `metrics_t = \mathrm{analyze\_summary}(y_t)` (`src/train_demo.py:1873-1985`)，并写入 CSV/HTML 仪表板。

## 输出与缓冲
- 奖励 $r_t$ 依照 `STEP_SCORING.md` 描述的公式生成，下一观测 $O_{t+1}`$ 构造规则见 `ArticleEnvironment.step`。
- 转移元组 `Transition(state, action, reward, next_state, done)` (`src/train_demo.py:2326-2370`) 被压入 `SimpleReplayBuffer` (`src/train_demo.py:2336-2357`)：
  \[
  \mathcal{D} \leftarrow \mathcal{D} \cup \{(O_t, y_t, r_t, O_{t+1}, \mathrm{done}_t)\}.
  \]

## 日志字段
- `metrics_logs/*.csv`：`similarity`、`coverage_ratio`、`novelty_ratio`、`lexical_cosine`、`lexical_js_similarity`、`garbled_ratio`、`word_noncompliance_ratio`、`reward`、`capital_value` 等。
- `character` 模式额外记录 `predicted_char`、`target_char`、`lexical_bigram_bonus`、`lexical_bigram_candidate` (`src/train_demo.py:2243-2264`)。
- `rewards.html` 仪表板汇总 `total_reward`、`average_reward` 等指标 (`src/train_demo.py:1432-1627`)。

## 伪代码
```pseudo
function ENV_STEP(state, policy, tokenizer, replay_buffer):
    observation = build_observation(state)           # TextObservation
    x_t = tokenizer.encode_observation(observation)  # token ids

    action_ids, extra = policy.sample(x_t)           # 或 first_step_distribution
    action_text = tokenizer.decode_action(action_ids)
    operations = OperationParser.parse(action_text)

    capital_next = apply_operations(state.capital, operations)
    budget_next = state.budget - total_cost(operations)
    metrics = analyze_summary(action_text)
    reward = compute_step_reward(state, capital_next, budget_next, metrics)

    next_observation = build_observation_next(capital_next, budget_next, state.cursor+1)
    transition = Transition(observation, TextAction(action_ids, action_text), reward, next_observation, done)
    replay_buffer.add(transition)

    return next_observation, reward
```

## 代码映射
- 观测结构：`TextObservation` 与 `CharTokenizer`，`src/train_demo.py:496-719`。
- 操作解析：`OperationParser`，`src/train_demo.py:740-947`。
- 环境核心逻辑：`ArticleEnvironment.step`，`src/train_demo.py:2142-2280`。
- 度量与日志：`analyze_summary`、`_write_rewards_dashboard`，`src/train_demo.py:1873-2084`、`src/train_demo.py:1432-1627`。
- 经验缓冲：`SimpleReplayBuffer`，`src/train_demo.py:2312-2357`。
