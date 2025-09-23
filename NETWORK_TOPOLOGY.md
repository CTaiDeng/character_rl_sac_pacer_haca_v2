# 当前网络拓扑结构设计方案

## 策略网络 $\pi_\theta$
策略网络为字符级序列到序列结构，输入 Token 序列 $x_t \in \mathbb{N}^{L}$，embedding 维度 $d=96$，隐藏维度 $h=128$。
编码器 GRU 计算 $h^{\text{enc}} = \mathrm{GRU}_{\text{enc}}(E x_t)$，其中 $E \in \mathbb{R}^{|V|\times d}$。
解码端以 $h^{\text{enc}}$ 为初始状态，逐步应用 $\mathrm{GRU}_{\text{dec}}$，输出 logits $z_\tau = W h_\tau + b$，再经合规筛选。
合规模块根据 `WordComplianceChecker` 给出的允许字符集合 $\mathcal{A}(y_{\tau-1})$ 对 logits 进行遮罩，并按温度 $\tau_c=0.85$ 重新缩放。
最终动作分布 $p_\theta(y_\tau \mid y_{<\tau}, x_t) = \mathrm{Softmax}(\tilde{z}_\tau)$，在训练时采样，在评估时取 argmax。
- 嵌入层：共享词表大小 $|V| \approx 1600$（随语料自动统计），\`embedding\` 权重与价值网络共享初始化种子。
- 编码器：单层 GRU（隐藏 128），使用 `pack_padded_sequence` 处理变长输入。
- 解码器：单层 GRU 加线性层输出 logits，生成长度上限 $L_{\max}=512$（由章节长度截断）。
- 合规温度与惩罚：使用 `DEFAULT_COMPLIANCE_TEMPERATURE=0.85`，非法 logits 统一填充 `COMPLIANCE_MASK_FILL_VALUE=-1e9`，保证硬掩码稳定。
- 首步分布：`first_step_distribution` 直接输出经硬掩码与重标定后的 logits/probs/log-probs/合法掩码，供 Top-p 期望与熵更新使用。
## 价值网络 $Q_\phi, Q_\psi$
价值网络共享 `TextQNetwork` 拓扑，使用相同词表嵌入并计算遮罩平均。
状态摘要 $u = \tanh(W_s \cdot \mathrm{mean}_{\mathrm{mask}}(E x_t))$，动作摘要 $v = \tanh(W_a \cdot \mathrm{mean}_{\mathrm{mask}}(E y_t))$，连接后经两层 MLP 输出标量。
双路网络 $(Q_\phi, Q_\psi)$ 参数独立，目标网络通过软更新 $\theta^{\prime} \leftarrow \tau \theta + (1-\tau) \theta^{\prime}$，其中 $\tau = \texttt{TrainerConfig.tau}$。
## 采样与温度伪代码
```pseudo
function SAMPLE_POLICY(tokens, lengths):
    embedded = embedding(tokens)
    _, h_enc = encoder(embedded, lengths)
    prev = BOS
    outputs = []
    for step in range(L_max):
        logits, h_enc = decoder_step(prev, h_enc)
        allowed = compliance(prev)
        logits[not allowed] -= penalty
        logits[allowed] /= tau_c
        dist = Categorical(logits)
        sample = dist.sample()
        outputs.append(sample)
        if sample == EOS: break
        prev = sample
    return outputs
```
## 参数配置与训练超参
- 关键维度：`embedding_dim=96`，`hidden_dim=128`，`max_summary_length` 由章节最大长度截断（不超过 512）。
- 优化器：策略与 Q 网络均使用 Adam(LR=3e-4)，熵系数 `alpha` 来自 `AgentConfig` 初始设置并在更新过程中自适应。
- 更新节奏：每步收集 1 条样本，`DemoSACAgent.update` 在 `updates_per_round = steps_per_round` 条目上执行，软更新系数 `tau=TrainerConfig.tau`。
- 参数规模：`DemoSACAgent.parameter_count` 统计策略网络参数量并写入导出，便于推断模型大小 `MODEL_SIZE_BYTES`。
- 字符模式长度字段：`character_length_field_width` 控制日志中 `prev_summary=... chars` 的宽度（默认 1，无补零）。
- 字符二元奖励：若上一字符与目标字符组合命中 `data/chinese_frequency_word.json` 或 `data/chinese_name_frequency_word.json` 的二字词，或命中原文滑窗中的二元组合，则在 `reward_base` 上追加 `CHARACTER_LEXICAL_BIGRAM_BONUS=1.0`，强化拓扑记忆。
- 词频补全：训练前通过  `_augment_lexical_statistics_with_bigrams` 将原文的二字词写回词频缓存。 
- Top-p 采样：默认 `top_p=0.98`，使用 `first_step_distribution` 的概率并在选集上重新归一化后计算无偏期望。
- 温度自适应：维护 `log_alpha` 参数，按 `logα ← logα + η(H_target - H)`（梯度化实现）更新并限制在 `[10^{-4}, 2]`。
## 数据流摘要
- 观测编码：`CharTokenizer.encode_observation` 共享给策略与价值网络。
- 策略采样：`TextPolicyNetwork` 输出字符动作，经 `OperationParser.parse` 转化为结构化操作并更新 `CognitiveCapital`。
- 价值评估：双路 `TextQNetwork` 在训练时接收状态与策略采样动作，构造目标 $y = r + \gamma(\min(Q_\phi, Q_\psi) - \alpha \log \pi)$。
- 参数更新：策略目标使用 $J_\pi = \mathbb{E}[\alpha \log \pi_\theta(a\mid s) - Q_\phi(s,a)]$，Q 目标使用 MSE，软更新由 `tau` 控制。
- Top-p 期望：目标值与策略梯度均基于 Top-p 截断后的归一化分布，对应候选集合 `detach()` 后再组合 `q` 与 `\log \pi`。
- 熵目标：使用合法动作计数计算 $H_{\text{tgt}}=\kappa\log |\mathcal{A}(s)|$ 并驱动 `\alpha` 自适应。
如需调整网络维度、合规模块或目标定义，请同步修改实现与本文档。
