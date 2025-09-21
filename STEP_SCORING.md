# 当前 Step 打分方案说明

## 数学定义
记第 $t$ 步执行前的认知资本为 $C_{t-1}$，执行后为 $C_t$，执行后预算为 $B_t^{\text{raw}}$。解析得到的操作集合记为 $\mathcal{O}_t$，其元素 $o$ 的类别为 $\mathrm{kind}(o)$。

1. **操作成本**：
   \[
   c_t = \sum_{o \in \mathcal{O}_t} \mathrm{cost}\bigl(\mathrm{kind}(o)\bigr),\qquad
   \mathrm{cost}(k) = \begin{cases}
   \texttt{OPERATION\_COSTS}[k], & k \in \{\text{ACQUIRE},\text{EXTRACT},\text{LINK},\text{VERIFY},\text{HEDGE},\text{TRIM},\text{COMMIT}\} \\
   \texttt{DEFAULT\_OPERATION\_COST}, & \text{否则}
   \end{cases}
   \]
   非终止步骤的成本惩罚仅使用 $c_t$；终止步骤改用累计成本 $\bar{c}_t = \sum_{i=1}^t c_i$。

2. **资本估值**：认知资本 $C_t = (F_t,V_t,H_t)$ 包含事实、核实与对冲集合。
   \[
   V(C_t) = \max\Bigl(\bigl(w_c\,\kappa_t + w_d\,\delta_t - w_r\,\rho_t + w_v\,\nu_t + w_f\,\ln(1+|F_t|)\bigr) (1-0.2\,\eta_t),\;0\Bigr).
   \]
   其中：
   - 覆盖率 $\kappa_t = \dfrac{|\bigcup_{f\in F_t} T(f)|}{|\bigcup_{p\in\mathcal{P}} T(p)|}$，$T(\cdot)$ 为分词集合；
   - 多样性 $\delta_t = \dfrac{|\{\text{类别}(f):f\in F_t\}|}{|\text{类别宇集}|}$；
   - 冗余度 $\rho_t$ 为事实两两 Jaccard 均值；
   - 验证占比 $\nu_t = \dfrac{|V_t|}{\max(1, |F_t|)}$；
   - 对冲占比 $\eta_t = \dfrac{|H_t|}{\max(1, |F_t|+|H_t|)}$；
   - 权重 $w_c=1.5,\; w_d=0.8,\; w_r=0.6,\; w_v=0.4,\; w_f=0.45$。

3. **基础奖励**：
   \[
   B_t = V(C_t) - \lambda_t - \beta\,\max\bigl(0, -B_t^{\text{raw}}\bigr)。
   \]
   其中 $\beta = \texttt{BUDGET\_PENALTY\_WEIGHT}=0.02$，成本惩罚 $\lambda_t$ 为：
   \[
   \lambda_t = \texttt{COST\_WEIGHT} \times\begin{cases}
   \bar{c}_t, & t \text{ 为终止步} \\
   c_t, & \text{否则}
   \end{cases},\qquad \texttt{COST\_WEIGHT}=0.08。
   \]

4. **潜能塑形**：潜能函数与估值一致 $\Phi(C)=V(C)$，塑形项 $\Delta\Phi_t = V(C_t) - V(C_{t-1})$。

5. **软奖励**：记质量指标 $(\mathrm{sim}_t,\mathrm{cov}_t,\mathrm{nov}_t)$，词汇指标 $(\mathrm{cos}_t,\mathrm{js}_t)$，洁净指标 $(\mathrm{garble}_t,\mathrm{noncomp}_t)$。
   \[
   \begin{aligned}
   S_t &= Q_t + L_t - P_t, \\
   Q_t &= w_s\,\mathcal{N}_{4.0}(\mathrm{sim}_t) + w_{cov}\,\mathcal{N}_{4.0}(\mathrm{cov}_t) + w_{nov}\,\mathcal{N}_{4.0}(\mathrm{nov}_t), \\
   L_t &= w_{lex}\,\mathcal{N}_{3.5}(\mathrm{cos}_t) + w_{js}\,\mathcal{N}_{3.5}(\mathrm{js}_t), \\
   P_t &= w_{gar}\,\mathcal{N}_{5.0}(\mathrm{garble}_t) + w_{word}\,\mathcal{N}_{5.0}(\mathrm{noncomp}_t)。
   \end{aligned}
   \]
   其中 $\mathcal{N}_\gamma(x) = 1 - (1-x)^{\gamma}$，权重满足 $w_s=0.6,\; w_{cov}=0.3,\; w_{nov}=0.1,\; w_{lex}=0.15,\; w_{js}=0.1,\; w_{gar}=0.5,\; w_{word}=0.7$。

6. **总奖励**：
   \[
   R_t = B_t + \Delta\Phi_t + S_t。
   \]

## 伪代码
```pseudo
function STEP_SCORE(state_capital, budget_prev, operations, metrics, is_terminal):
    cost = sum(OPERATION_COSTS.get(op.kind, DEFAULT_OPERATION_COST) for op in operations)
    budget_raw = budget_prev - cost
    capital = apply_operations(state_capital, operations)
    potential_before = value(state_capital)
    potential_after = value(capital)

    step_cost = cumulative_cost + cost if is_terminal else cost
    if is_terminal:
        step_cost = cumulative_cost
    cost_penalty = COST_WEIGHT * step_cost
    budget_breach = max(0.0, -budget_raw)
    base_reward = value(capital) - cost_penalty - BUDGET_PENALTY_WEIGHT * budget_breach

    quality = (
        QUALITY_SIMILARITY_WEIGHT * nonlinear(metrics['similarity'], QUALITY_NONLINEAR_EXPONENT)
        + QUALITY_COVERAGE_WEIGHT * nonlinear(metrics['coverage_ratio'], QUALITY_NONLINEAR_EXPONENT)
        + QUALITY_NOVELTY_WEIGHT * nonlinear(max(0.0, metrics['novelty_ratio']), QUALITY_NONLINEAR_EXPONENT)
    )
    lexical = (
        LEXICAL_SIMILARITY_WEIGHT * nonlinear(metrics['lexical_cosine'], LEXICAL_NONLINEAR_EXPONENT)
        + LEXICAL_JS_WEIGHT * nonlinear(metrics['lexical_js_similarity'], LEXICAL_NONLINEAR_EXPONENT)
    )
    penalty = (
        GARBLED_REWARD_WEIGHT * nonlinear(metrics['garbled_ratio'], CLEANLINESS_NONLINEAR_EXPONENT)
        + WORD_COMPLIANCE_REWARD_WEIGHT * nonlinear(metrics['word_noncompliance_ratio'], CLEANLINESS_NONLINEAR_EXPONENT)
    )
    soft_reward = quality + lexical - penalty

    reward = base_reward + (potential_after - potential_before) + soft_reward
    return reward, capital, budget_raw
```

## 参数表
- `OPERATION_COSTS = {ACQUIRE: 3, EXTRACT: 3, LINK: 2, VERIFY: 4, HEDGE: 1.5, TRIM: 1, COMMIT: 5}`，`DEFAULT_OPERATION_COST = 2`。
- 初始预算 `DEFAULT_INITIAL_BUDGET = 1200`，透支惩罚系数 `BUDGET_PENALTY_WEIGHT = 0.02`。
- 非线性放大函数 `nonlinear(x, \gamma) = 1 - (1-x)^{\gamma}` 在实现中对应 `_nonlinear_reward`。
- 所有指标的计算与记录位于 `src/train_demo.py` 的 `ArticleEnvironment.step` 与 `CapitalValuator` 中。

若调整任何权重、成本或函数形式，需同步修改实现与本文档以保持一致。
