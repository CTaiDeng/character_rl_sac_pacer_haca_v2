# 当前 Step 打分方案说明

## 文字描述

每个 step 的得分由两部分组成：
1. **终值收益**：在遍历完整个章节序列后，依据认知资本估值扣除累计操作成本与预算透支罚项；
2. **潜能塑形**：在每个中间阶段，比较当前潜能与上一步潜能，形成奖励差值，保证最优策略不变的同时提供平滑梯度。

核心流程如下：

- 策略输出文本被 `OperationParser` 解析为指令列表。
- `CognitiveCapital.apply()` 逐条执行指令，实时更新事实集合、验证集合、对冲集合与链接集合。
- 每条指令按照 `OPERATION_COSTS` 扣除预算，未列出的指令使用 `DEFAULT_OPERATION_COST`。预算可以透支，透支额度在奖励中按 `BUDGET_PENALTY_WEIGHT` 计入惩罚。
- `CapitalValuator.metrics()` 计算覆盖率、多样性、冗余度、验证比例等指标，`value()` 根据指标加权得到资本估值，`potential()` 与 `value()` 等价，用于潜能塑形。
- 终止步骤的基础奖励：`capital_value - COST_WEIGHT * cumulative_cost - BUDGET_PENALTY_WEIGHT * budget_breach`，其中 `budget_breach = max(0, -budget_remaining)`。
- 中间步骤基础奖励仅包含预算惩罚项，其余部分依赖潜能差值。

## 伪代码

```pseudo
function compute_step_reward(capital_before, capital_after, valuator, step_cost, cumulative_cost, budget, is_terminal):
    potential_prev <- valuator.potential(capital_before)
    potential_now <- valuator.potential(capital_after)
    budget_breach <- max(0, -budget)
    if is_terminal:
        base_reward <- valuator.value(capital_after)
            - COST_WEIGHT * cumulative_cost
            - BUDGET_PENALTY_WEIGHT * budget_breach
    else:
        base_reward <- -BUDGET_PENALTY_WEIGHT * budget_breach
    shaped_reward <- base_reward + (potential_now - potential_prev)
    return shaped_reward
```

## 参数说明
- **估值权重**：`CAPITAL_COVERAGE_WEIGHT`、`CAPITAL_DIVERSITY_WEIGHT`、`CAPITAL_REDUNDANCY_WEIGHT`、`CAPITAL_VERIFICATION_BONUS` 控制覆盖、多样、冗余、验证对估值的影响。
- **成本权重**：`COST_WEIGHT` 用于限制过度操作；成本源自所有历史指令的总花费。
- **预算惩罚**：`BUDGET_PENALTY_WEIGHT` 对透支金额进行软约束。
- **潜能函数**：与估值相同，确保潜能塑形不改变最优策略，但能在训练早期提供正向信号。

> 若未来调整估值模型或成本结构，应同步更新本说明与实现，保持行为一致。
