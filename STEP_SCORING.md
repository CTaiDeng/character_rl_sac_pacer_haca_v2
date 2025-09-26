# 当前 Step 打分方案

## 符号约定
- $C_t = (F_t, L_t, V_t, H_t)$ 表示第 $t$ 步的认知资本，分别存储事实、链接、已验证事实、已对冲事实。
- $B_t$ 为剩余预算，初始值 $B_0 = \texttt{DEFAULT\_INITIAL\_BUDGET} = 1200$。
- $\mathcal{O}_t = \{o_1, \dots, o_{n_t}\}$ 为解析得到的操作序列。
- $c(o)$ 为单次操作成本，采用 `OPERATION_COSTS` 中的映射，未列出的操作使用 $\texttt{DEFAULT\_OPERATION\_COST} = 2$。
- $\bar c_t = \sum_{i=1}^t \sum_{o\in\mathcal{O}_i} c(o)$ 为累计成本。
- $\mathrm{metrics}_t$ 收集当前摘要质量指标（`similarity`、`coverage_ratio`、`novelty_ratio`、`lexical_cosine`、`lexical_js_similarity`、`garbled_ratio`、`word_noncompliance_ratio` 等）。
- $\mathcal{N}_\gamma(x) = 1 - (1 - x)^\gamma$ 为奖励放大量表。
- $V(C_t)$、$P(C_t)$ 分别表示 `CapitalValuator` 给出的价值与潜力。

## 成本与预算
$$
 c_t = \sum_{o\in\mathcal{O}_t} c(o), \qquad
 B_t = B_{t-1} - c_t, \qquad
 \psi_t = \beta \, \max(0, -B_t),
$$
其中 $\beta = \texttt{BUDGET\_PENALTY\_WEIGHT} = 0.02$。当 episode 结束时，成本惩罚改为 $\lambda_t = \omega_c \, \bar c_t$，否则 $\lambda_t = \omega_c \, c_t$，权重 $\omega_c = \texttt{COST\_WEIGHT} = 0.08$。

## 认知资本估值
`CapitalValuator` 在 `src/character_sac_trainer.py:1074` 起定义：
$$
\begin{aligned}
 \mathrm{coverage}_t &= \frac{\bigl|\bigcup_{f\in F_t} T(f)\bigr|}{\bigl|\bigcup_{p\in\mathcal{P}} T(p)\bigr|},\\
 \mathrm{diversity}_t &= \frac{|\{ \text{首词}(f):f\in F_t\}|}{|\mathcal{D}|},\\
 \mathrm{redundancy}_t &= \frac{1}{|F_t|(|F_t|-1)} \sum_{f\neq f'} \mathrm{Jaccard}(T(f),T(f')),\\
 \mathrm{verification}_t &= \frac{|V_t|}{|F_t|\vee 1},\quad
 \mathrm{hedge}_t = \frac{|H_t|}{|F_t|\vee 1}.
\end{aligned}
$$
价值函数
$$
 V(C_t) = \max\Bigl(0,\Bigl[
 1.5\,\mathrm{coverage}_t + 0.8\,\mathrm{diversity}_t - 0.6\,\mathrm{redundancy}_t
 + 0.4\,\mathrm{verification}_t + 0.45 \ln(1+|F_t|)
 \Bigr]\cdot(1 - 0.2\,\mathrm{hedge}_t)\Bigr),
$$
潜力 $P(C_t)$ 等同于 $V(C_t)$。潜力增益 $\Delta_t = P(C_t) - P(C_{t-1})$。

## 质量与语言项
质量项使用幂次 $\gamma_q = 4.0$，词汇项幂次 $\gamma_\ell = 3.5$，洁净项幂次 $\gamma_c = 5.0$：
$$
\begin{aligned}
 Q_t &= 0.6\,\mathcal{N}_{4.0}(\texttt{similarity}) + 0.3\,\mathcal{N}_{4.0}(\texttt{coverage\_ratio}) + 0.1\,\mathcal{N}_{4.0}(\max(0,\texttt{novelty\_ratio})),\\
 L_t &= 0.15\,\mathcal{N}_{3.5}(\texttt{lexical\_cosine}) + 0.10\,\mathcal{N}_{3.5}(\texttt{lexical\_js\_similarity}),\\
 P_t &= 0.5\,\mathcal{N}_{5.0}(\texttt{garbled\_ratio}) + 0.7\,\mathcal{N}_{5.0}(\texttt{word\_noncompliance\_ratio}).
\end{aligned}
$$
软奖励 $S_t = Q_t + L_t - P_t$。

## 字符级增益
在 `iteration_mode == "character"` 时额外计算
$$
 \chi_t = \max(0, Q_t + L_t),
$$
$$
 \delta_t = \mathbf{1}_{\text{match}} \cdot
 \begin{cases}
 1.0, & \text{bigram 命中词典},\\
 0.5, & \text{bigram 未命中且字符匹配教师},\\
 0, & \text{否则},
 \end{cases}
$$
其中 `match` 表示预测首字符等于教师目标字符，bigram 为 `chapter_char + raw_action_char`，构建逻辑见 `src/character_sac_trainer.py:2234-2264`。该模式下
$$
\begin{aligned}
 B_t^{\text{char}} &= B_t + 0.5\,\chi_t + \delta_t,\\
 \Delta_t^{\text{char}} &= \Delta_t + 0.25\,\chi_t.
\end{aligned}
$$

## Step 奖励合成
综合 `src/character_sac_trainer.py:2181-2268`：
$$
 R_t = V(C_t) - \lambda_t - \psi_t + S_t +
 \begin{cases}
 \Delta_t, & \text{章节模式},\\
 \Delta_t + 0.5\,\chi_t + 0.25\,\chi_t + \delta_t, & \text{字符模式}.
 \end{cases}
$$
环境同时记录 `capital_metrics`、`reward_*` 分量用于调试。

## 伪代码
```pseudo
function STEP_REWARD(state, operations, metrics, is_terminal):
    cost = sum(OPERATION_COSTS.get(op.kind, DEFAULT_OPERATION_COST) for op in operations)
    cumulative_cost = state.cumulative_cost + cost
    budget = state.budget - cost
    capital_before = state.capital.clone()
    capital_after = apply_operations(capital_before, operations)

    capital_value = valuator.value(capital_after)
    potential_gain = valuator.potential(capital_after) - valuator.potential(capital_before)

    cost_penalty = COST_WEIGHT * (cumulative_cost if is_terminal else cost)
    budget_penalty = BUDGET_PENALTY_WEIGHT * max(0.0, -budget)

    quality = quality_component(metrics)
    lexical = lexical_component(metrics)
    cleanliness = cleanliness_penalty(metrics)
    soft_reward = quality + lexical - cleanliness

    base_component = capital_value - cost_penalty - budget_penalty
    potential_component = potential_gain

    if iteration_mode == "character":
        quality_signal = max(0.0, quality + lexical)
        base_component += CHARACTER_BASE_QUALITY_WEIGHT * quality_signal
        potential_component += CHARACTER_POTENTIAL_QUALITY_WEIGHT * quality_signal
        base_component += compute_bigram_bonus(state, operations)

    total_reward = base_component + potential_component + soft_reward
    return total_reward, capital_after, budget, cumulative_cost
```

## 实现映射
- `操作成本与预算`：`src/character_sac_trainer.py:2147-2179`，`OPERATION_COSTS` 定义于 `src/character_sac_trainer.py:421-428`。
- `认知资本估值`：`CognitiveCapital` 与 `CapitalValuator` 定义于 `src/character_sac_trainer.py:959-1179`。
- `质量/语言/洁净组件`：`src/character_sac_trainer.py:2193-2212`。
- `字符模式加成与 bigram 奖励`：`src/character_sac_trainer.py:2220-2266`。
- `奖励写入与日志`：`ArticleEnvironment.step` 中 `src/character_sac_trainer.py:2142-2280`。
## 字符模式加成与词法二元组（bigram）奖励（补充说明）
在 `iteration_mode == "character"` 时，额外：
$$
 \chi_t = \max(0, Q_t + L_t)
$$

bigram 奖励基于“后缀命中”的可变长度词法集合 U（来自 `data/word_length_sets.json.union.lengths`）。设：
- 预测首字为 c，raw_action 通过前向拓扑得到序列 q（去重首字，沿未来字符扩展，遇到 q 的后缀在 Catalog 命中即停，长度从 U 按降序尝试）。
- bigram 候选串 s = chapter_char ⊕ q，经前向拓扑（与 raw_action 共用 U）在其后缀命中时给奖励。
- 命中注记记录来源词表与编号，如 `data/chinese_frequency_word.json#236703`。
- 若未命中但当前字符匹配教师目标字符，给降级奖励；否则为 0。

形式化地：
$$
 \delta_t =
 \begin{cases}
 1.0, & \exists L\in U,\ \mathrm{tail}(s, L)\in\text{Catalog},\\
 0.5, & \mathrm{tail}(s, 1)=\text{target\_char},\\
 0, & \text{otherwise}.
 \end{cases}
$$

字符模式下：
$$
\begin{aligned}
 B_t^{\text{char}} &= B_t + 0.5\,\chi_t + \delta_t,\\
 \Delta_t^{\text{char}} &= \Delta_t + 0.25\,\chi_t.
\end{aligned}
$$

实现要点（与日志一致）：
- raw_action 显示形如：`raw_action=3 chars "他喃喃" (后缀"喃喃": data/chinese_frequency_word.json未命中)`。
- bigram 显示形如：`bigram=4 chars "”他喃喃" (后缀"喃喃": data/chinese_frequency_word.json未命中)`。
- `character_history_extension_limit=16` 控制“source 前缀两字命中”的左扩历史步数上限（只影响日志渲染与注记，不改变策略输入）。

