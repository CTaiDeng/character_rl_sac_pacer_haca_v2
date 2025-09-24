# 当前 Step 打分方案

## 符号约定
- $C_t = (F_t, V_t, H_t)$：第 $t$ 步结束后的认知资本，依次表示事实集、验证证据集、对冲假设集。
- $B_t$：第 $t$ 步完成后的剩余预算；$B_0 = \texttt{DEFAULT\_INITIAL\_BUDGET}$。
- $\mathcal{O}_t = \{o_1, \dots, o_{n_t}\}$：当前步的操作序列，$\mathrm{kind}(o)$ 和 $\mathrm{payload}(o)$ 分别为操作类型与参数。
- $\mathrm{metrics}_t$：评估模块返回的指标字典，含 `similarity`、`coverage_ratio`、`novelty_ratio`、`lexical_cosine`、`lexical_js_similarity`、`garbled_ratio`、`word_noncompliance_ratio` 等键。

## 核心评价公式

### 1. 操作成本
\[
 c_t = \sum_{o \in \mathcal{O}_t} \mathrm{cost}(\mathrm{kind}(o)),\qquad
 \bar{c}_t = \sum_{i=1}^{t} c_i
\]
其中
\[
 \mathrm{cost}(k) =
 \begin{cases}
  \texttt{OPERATION\_COSTS}[k], & k \in \{\text{ACQUIRE}, \text{EXTRACT}, \text{LINK}, \text{VERIFY}, \text{HEDGE}, \text{TRIM}, \text{COMMIT}\} \\
  \texttt{DEFAULT\_OPERATION\_COST}, & \text{其它类型}
 \end{cases}
\]

### 2. 资本价值函数
定义覆盖率、深度、多样性、冗余、验证度与对冲占比：
\[
\begin{aligned}
 \kappa_t &= \frac{\bigl|\bigcup_{f \in F_t} T(f)\bigr|}{\bigl|\bigcup_{p \in \mathcal{P}} T(p)\bigr|}, &
 \delta_t &= \frac{\bigl|\{\mathrm{domain}(f) : f \in F_t\}\bigr|}{|\mathcal{D}|},\\
 \rho_t &= \mathrm{Jaccard}(F_t, F_{t-1}), &
 \nu_t &= \frac{|V_t|}{\max(1, |F_t|)},\\
 \eta_t &= \frac{|H_t|}{\max(1, |F_t| + |H_t|)}.
\end{aligned}
\]
资本价值定义为
\[
 V(C_t) = \sigma\Bigl(w_c\kappa_t + w_d \delta_t + w_v \nu_t - w_r \rho_t - w_h \eta_t + w_f \ln(1 + |F_t|)\Bigr),
\]
其中 $\sigma(x) = \dfrac{1}{1 + e^{-x}}$。潜力增量记为 $\Delta\Phi_t = V(C_t) - V(C_{t-1})$。

### 3. 预算约束与罚项
\[
 B_t = B_{t-1} - c_t,\qquad
 \lambda_t = \gamma_c\, c_t + \gamma_{\text{cum}}\, \bar{c}_t,\qquad
 \psi_t = \beta \max(0, -B_t).
\]

### 4. 软指标
引入非线性映射 $\mathcal{N}_\gamma(x) = 1 - (1-x)^{\gamma}$：
\[
\begin{aligned}
 Q_t &= w_s\, \mathcal{N}_{4.0}(\mathrm{metrics}_t[\texttt{similarity}]) + w_{cov}\, \mathcal{N}_{4.0}(\mathrm{metrics}_t[\texttt{coverage\_ratio}]) + w_{nov}\, \mathcal{N}_{4.5}(\max(0, \mathrm{metrics}_t[\texttt{novelty\_ratio}]))\\
 L_t &= w_{lex}\, \mathcal{N}_{3.5}(\mathrm{metrics}_t[\texttt{lexical\_cosine}]) + w_{js}\, \mathcal{N}_{3.0}(\mathrm{metrics}_t[\texttt{lexical\_js\_similarity}])\\
 P_t &= w_{gar}\, \mathcal{N}_{5.0}(\mathrm{metrics}_t[\texttt{garbled\_ratio}]) + w_{word}\, \mathcal{N}_{5.0}(\mathrm{metrics}_t[\texttt{word\_noncompliance\_ratio}])
\end{aligned}
\]

### 5. Step 总得分
\[
 R_t = \alpha_{\text{base}} V(C_t) + \Delta\Phi_t + Q_t + L_t - P_t - \lambda_t - \psi_t.
\]

## 参数取值
| 符号 | 数值 | 说明 |
| --- | --- | --- |
| $\texttt{OPERATION\_COSTS}$ | `{ACQUIRE:3, EXTRACT:3, LINK:2, VERIFY:4, HEDGE:1.5, TRIM:1, COMMIT:5}` | 见 `src/train_demo.py` 中的 `OPERATION_COSTS` |
| $\texttt{DEFAULT\_OPERATION\_COST}$ | $2$ | 未列出的操作默认成本 |
| $w_c, w_d, w_v, w_r, w_h, w_f$ | $1.6, 0.9, 0.5, 0.6, 0.3, 0.45$ | 资本价值权重 |
| $\gamma_c, \gamma_{\text{cum}}$ | $0.08, 0.02$ | 成本罚项系数 |
| $\beta$ | $0.02$ | 预算穿透罚项系数 |
| $w_s, w_{cov}, w_{nov}$ | $0.6, 0.25, 0.15$ | 语义质量权重 |
| $w_{lex}, w_{js}$ | $0.18, 0.12$ | 词法质量权重 |
| $w_{gar}, w_{word}$ | $0.5, 0.7$ | 清洁度惩罚权重 |
| $\alpha_{\text{base}}$ | $0.6$ | 基础资本权重 |

## α 伪代码
```pseudo
function STEP_SCORE(prev_capital, prev_budget, operations, metrics, is_terminal):
    cost = sum(cost_of(op.kind) for op in operations)
    cumulative_cost = update_cumulative(cost, is_terminal)
    budget = prev_budget - cost

    capital = apply_operations(prev_capital, operations)
    value_prev = capital_value(prev_capital)
    value_new = capital_value(capital)
    potential_gain = value_new - value_prev

    lambda_penalty = GAMMA_C * cost + GAMMA_CUM * cumulative_cost
    budget_penalty = BETA * max(0.0, -budget)

    quality = (
        W_S * nonlinear(metrics["similarity"], 4.0)
        + W_COV * nonlinear(metrics["coverage_ratio"], 4.0)
        + W_NOV * nonlinear(max(0.0, metrics["novelty_ratio"]), 4.5)
    )
    lexical = (
        W_LEX * nonlinear(metrics["lexical_cosine"], 3.5)
        + W_JS * nonlinear(metrics["lexical_js_similarity"], 3.0)
    )
    penalty = (
        W_GAR * nonlinear(metrics["garbled_ratio"], 5.0)
        + W_WORD * nonlinear(metrics["word_noncompliance_ratio"], 5.0)
    )

    reward = (
        ALPHA_BASE * value_new
        + potential_gain
        + quality + lexical - penalty
        - lambda_penalty - budget_penalty
    )

    if is_terminal:
        reward += terminal_adjustment(capital)

    return reward, capital, budget
```

## 实现映射
- 认知资本、潜力与价值评估在 `src/train_demo.py` 的 `CognitiveCapital` 与 `CapitalValuator` 中实现。
- 操作成本与预算更新位于 `ArticleEnvironment.step` 的 `_apply_operations` 链路。
- 软指标由 `SummarizationMetrics` 计算，位于 `src/train_demo.py` 的 `analyze_summary` 函数附近。
- 字符模式 bigram（更新）：`lexical_bigram_candidate = chapter_char + raw_action_char`；若 `action.text` 以 `chapter_char` 开头且长度≥2，则 `raw_action_char = last(action.text)`，否则取首字。

如在代码层调整参数或函数，请同步更新本文档。
