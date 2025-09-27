# 当前输入-输出数据方案

## 更新要点（需与代码保持一致）
- 字符模式观测采用 prev + <sep> + chapter，其中 chapter=当前目标字符，不再为空串。
- 历史回溯拓扑：在字符模式日志渲染中，若 $\texttt{source=prev+chapter}$ 的前缀两字未命中词表，则沿“历史字符对”向左逐步扩展 prev，至多 16 次，直到前缀两字命中或达到上限。
- 允许的后缀词长来源于 $\texttt{data/word\_length\_sets.json}$ 的并集长度集合 $\texttt{union.lengths}$，用于 raw_action/bigram 的后缀命中判定。

## 字符模式输入构成（prev+sep+chapter）
设时刻 t 的观测为 O_t=(s_t, χ_t, i_t)，其中 χ_t 为当前目标字符。
令编码器输出序列为 $$ x_t = [<bos>] ⊕ clip(s_t) ⊕ [<sep>] ⊕ χ_t ⊕ [<eos>]$$ 并截断至 $\texttt{max\_observation\_length}$。

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
- iteration_granularity ∈ {chapter, paragraph, character}，字符模式时 $\texttt{chapter\_text}$ 即为目标字符。


## 状态与观测
- 记忆状态 $C_t$ 由 $\texttt{CognitiveCapital}$ 管理，预算 $B_t$ 为环境中的剩余成本额度。
- 第 $t$ 步观测 $O_t = (s_t, \chi_t, i_t)$：$s_t = \sigma(C_{t-1}, B_{t-1})$ 来自 $\texttt{CognitiveCapital.render\_text}$，$\chi_t$ 为章节文本或字符目标，$i_t$ 为游标索引。
- 在章节模式下 $\chi_t = \text{chapters}[t]$；字符模式下 $\chi_t$ 为当前目标字符（$\texttt{src/character\_sac\_trainer.py}$ 运行时由环境提供）。

## 编码函数
- $\texttt{CharTokenizer.encode\_observation}$（见 $\texttt{src/character\_sac\_trainer.py}$）构造 token 序列
  $$
  x_t = [\texttt{<bos>}] \oplus \mathrm{clip}(s_t) \oplus
  \begin{cases}
  [\texttt{<sep>}] \oplus \mathrm{clip}(\chi_t), & \text{章节模式},\\
  [], & \text{字符模式},
  \end{cases}
  \oplus [\texttt{<eos>}],
  $$
  并截断至 $\texttt{max\_observation\_length}$。
- Tokenizer 维护 $\texttt{summary\_token\_ids}$ 供合规检查、$\texttt{pad}$/$\texttt{unk}$ 等特殊标记。

## 策略输出
- $\texttt{TextPolicyNetwork}$（见 $\texttt{src/character\_sac\_trainer.py}$）接收 batch 化的 $x_t$，先用 GRU 编码，再逐步解码 action tokens：
  $$
  y_t \sim \pi_\theta(\cdot \mid x_t), \qquad y_t = (y_{t,1},\dots,y_{t,m}).
  $$
- $\texttt{first\_step\_distribution}$ 返回首个 token 的概率、对数概率与合规 mask，Top-$p$ 采样阈值默认 $p=0.98$ ($\texttt{DemoSACAgent.top\_p}$)。

## 操作解析
- $\texttt{OperationParser.parse}$（见 $\texttt{src/character\_sac\_trainer.py}$）将 $y_t$ 解析为结构化操作列表 $A_t = (a_1,\dots,a_{m_t})$，每个 $a_i = (\mathrm{kind}, \mathrm{payload})$。
- 合法指令集合包含 $\texttt{ACQUIRE}$/$\texttt{EXTRACT}$/$\texttt{LINK}$/$\texttt{VERIFY}$/$\texttt{HEDGE}$/$\texttt{TRIM}$/$\texttt{COMMIT}$ 等，失败解析会返回空序列并保留原文本供日志分析。

## 环境转移
- 环境执行器（$\texttt{ArticleEnvironment.step}$，见 $\texttt{src/character\_sac\_trainer.py}$）根据 $A_t$ 更新资本与预算：
  $$
  C_t = \Gamma(C_{t-1}, A_t), \qquad B_t = B_{t-1} - \sum_{a\in A_t} c(a).
  $$
- 同步计算指标 $\texttt{metrics\_t} = \mathrm{analyze\_summary}(y_t)$（见 $\texttt{src/character\_sac\_trainer.py}$），并写入 CSV/HTML 仪表板。

## 输出与缓冲
- 奖励 $r_t$ 依照 $\texttt{STEP\_SCORING.md}$ 描述的公式生成，下一观测 $O_{t+1}$  构造规则见 $\texttt{ArticleEnvironment.step}$。
- 转移元组 $\texttt{Transition(state, action, reward, next\_state, done)}$（见 $\texttt{src/character\_sac\_trainer.py}$）被压入 $\texttt{SimpleReplayBuffer}$（见 $\texttt{src/character\_sac\_trainer.py}$）：
  $$
  \mathcal{D} \leftarrow \mathcal{D} \cup \{(O_t, y_t, r_t, O_{t+1}, \mathrm{done}_t)\}.
  $$

## 日志字段
- $\texttt{metrics\_logs/*.csv}$：$\texttt{similarity}$、$\texttt{coverage\_ratio}$、$\texttt{novelty\_ratio}$、$\texttt{lexical\_cosine}$、$\texttt{lexical\_js\_similarity}$、$\texttt{garbled\_ratio}$、$\texttt{word\_noncompliance\_ratio}$、$\texttt{reward}$、$\texttt{capital\_value}$ 等。
- $\texttt{character}$ 模式额外记录 $\texttt{predicted\_char}$、$\texttt{target\_char}$、$\texttt{lexical\_bigram\_bonus}$、$\texttt{lexical\_bigram\_candidate}$（见 $\texttt{src/character\_sac\_trainer.py}$）。
- $\texttt{rewards.html}$ 仪表板汇总 $\texttt{total\_reward}$、$\texttt{average\_reward}$ 等指标（见 $\texttt{src/character\_sac\_trainer.py}$）。

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
- 观测结构：$\texttt{TextObservation}$ 与 $\texttt{CharTokenizer}$，见 $\texttt{src/character\_sac\_trainer.py}$。
- 操作解析：$\texttt{OperationParser}$，见 $\texttt{src/character\_sac\_trainer.py}$。
- 环境核心逻辑：$\texttt{ArticleEnvironment.step}$，见 $\texttt{src/character\_sac\_trainer.py}$。
- 度量与日志：$\texttt{analyze\_summary}$、$\texttt{\_write\_rewards\_dashboard}$，见 $\texttt{src/character\_sac\_trainer.py}$。
- 经验缓冲：$\texttt{SimpleReplayBuffer}$，见 $\texttt{src/character\_sac\_trainer.py}$。

