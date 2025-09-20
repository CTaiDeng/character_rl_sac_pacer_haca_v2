# 当前网络拓扑结构设计方案

## 总览
- **观察编码**：上一轮的认知资本快照与当前章节字符序列按顺序拼接，形成 `[<bos>] + capital_snapshot + [<sep>] + chapter + [<eos>]`。
- **行动解码**：策略网络输出字符序列，对应一组按行排列的操作指令（ACQUIRE/EXTRACT/VERIFY/…）。
- **核心组件**：
  1. 字符级嵌入层（共用词表），将观察序列映射到向量空间；
  2. 编码器 GRU 提取观察上下文隐藏状态；
  3. 解码器 GRU 自回归地产生操作字符分布，并在采样前应用合规屏蔽与温度调节；
  4. 双路 Q 网络评估状态-动作对，用于最大熵 SAC 更新；
  5. `CapitalValuator` 提供潜能与估值，用于奖励塑形和日志记录。

## 策略网络拓扑
设字符词表大小为 `|V|`，最大解码长度为 `L_max`。策略网络 `π_θ` 的结构：
1. **Embedding**：`E ∈ ℝ^{|V| × d_emb}`，将观察序列映射为张量 `Z`。
2. **编码器 GRU**：`h_enc = GRU_enc(Z)`，输出最终隐藏状态作为上下文向量。
3. **解码器 GRU**：以 `h_enc` 作为初始状态，逐步生成字符；每一步：
   - 取上一字符嵌入 `y_emb`；
   - 解码器输出 `h_dec`；
   - 线性层得到 logits；
   - `WordComplianceChecker` 与 `OperationParser` 提供的允许集合用于屏蔽非法字符，并在允许集合上施加温度；
   - 通过 `Categorical(logits)` 采样（或取 argmax）。
4. **潜能塑形**：策略本身不直接使用潜能，但环境在奖励侧使用 `CapitalValuator.potential()`，与策略网络的输出长度无关。

伪代码如下：
```pseudo
function POLICY_FORWARD(x_tokens):
    z = embedding(x_tokens)
    _, h = encoder_gru(pack(z))
    y_prev = <bos>
    outputs = []
    log_probs = []
    finished = false
    for step in 1..L_max:
        y_emb = embedding(y_prev)
        h, dec_out = decoder_gru(y_emb, h)
        logits = linear(dec_out)
        logits = apply_compliance(logits, y_prev)
        y_curr ~ Categorical(logits)
        outputs.append(y_curr)
        log_probs.append(log_prob(y_curr))
        finished = finished or (y_curr == <eos>)
        if finished:
            break
        y_prev = y_curr
    return stack(outputs), sum(log_probs)
```

`apply_compliance` 会：
- 根据 `WordComplianceChecker` 提供的合法字符集合屏蔽非法 logits；
- 对合法集合按照温度 `τ=0.85` 重新缩放。

## 价值网络拓扑
价值网络 `Q_φ` 与 `Q_ψ` 结构保持轻量：
1. 共享嵌入层编码状态与动作字符序列；
2. 通过掩码平均池化得到状态向量 `u` 与动作向量 `v`；
3. 两层前馈网络输出标量 Q 值。

伪代码：
```pseudo
function Q_FORWARD(state_tokens, action_tokens):
    s = embedding(state_tokens)
    a = embedding(action_tokens)
    u = tanh(W_s * masked_mean(s))
    v = tanh(W_a * masked_mean(a))
    q = FFN(concat(u, v))
    return q
```

## 潜能塑形与奖励逻辑
- `CapitalValuator` 基于当前资本计算覆盖、多样、冗余、验证四项指标，生成估值 `value(capital)`；
- 环境每步奖励：`r_t = base_reward + potential(capital_t) - potential(capital_{t-1})`，其中 `base_reward` 仅在终止步骤包含估值与成本差；
- 成本由操作成本与预算透支罚项组成，使策略在最大化估值同时控制资源消耗。

## 数据流
1. 观察编码：`capital_snapshot_{t-1} + chapter_t → tokenizer → tokens`；
2. 策略解码：`tokens → π_θ → action_text`；
3. 操作执行：`OperationParser.parse(action_text)` → `CognitiveCapital.apply()` → 更新资本与预算；
4. 估值：`CapitalValuator.metrics()` 产出日志指标并为奖励塑形提供潜能值；
5. 价值更新：`Q_φ, Q_ψ` 在最大熵 SAC 中与策略共同训练。

任何关于策略或价值网络的结构调整（嵌入维度、RNN 类型、合规策略、潜能定义等），都需同步修改本文与输入输出说明文档。

## 估值与日志支路
- `CapitalValuator.metrics()` 输出 `capital_value`, `capital_coverage`, `capital_diversity`, `capital_redundancy`, `capital_verification_ratio` 等字段，并写入 step CSV。
- 预算与成本通过 `budget_remaining`, `budget_breach`, `operation_cost`, `cumulative_cost` 字段落盘，便于分析资源使用情况。
