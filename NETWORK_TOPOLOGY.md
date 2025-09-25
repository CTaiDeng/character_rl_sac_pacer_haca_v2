# 当前网络拓扑结构方案

## 策略网络 $\pi_\theta$
- 观测拼接：字符模式下 tokens 由 prev + [<sep>] + chapter(目标字符) 组成，确保模型显式看到目标字符。
- 结构：`TextPolicyNetwork` (`src/train_demo.py:2371-2525`) 使用共享字符词表。
  1. 嵌入层 $E \in \mathbb{R}^{|V|\times d_{\text{emb}}}$ 将 tokens 映射为向量。
  2. 编码 GRU（单向，hidden dim $d_h$）对输入序列执行 `pack_padded_sequence` 编码，得到上下文隐状态 $h^{\text{enc}}$。
  3. 解码 GRU 以 $h^{\text{enc}}$ 为初始状态自回归地产生输出 token，步进时应用合规模块 `_mask_logits`。
  4. 输出层 $W_o \in \mathbb{R}^{|V|\times d_h}$，logits 经温度缩放 $\tau_c = \texttt{DEFAULT\_COMPLIANCE\_TEMPERATURE} = 0.85$ 与合规掩码处理。
- 合规模块：给定前一 token $y_{t,\tau-1}$、`summary_token_ids` 与 `WordComplianceChecker` (`src/train_demo.py:230-381`)，构造允许集合 $\mathcal{A}(y_{<\tau})$，对非法 token 赋值 $\texttt{COMPLIANCE\_MASK\_FILL\_VALUE}=-10^9$。
- 采样：`DemoSACAgent._select_top_p` (`src/train_demo.py:2709-2763`) 对 softmax 概率执行 Top-$p$ 过滤，$p$ 来自配置（默认 0.98），并返回候选集合、归一化权重和 log-prob。
- 确定性推理：`TextPolicyNetwork.deterministic` 使用贪心选取允许集合中概率最大的 token。

## 价值网络 $Q_\phi, Q_\psi$
- 结构：`TextQNetwork` (`src/train_demo.py:2572-2623`) 对状态与动作序列分别嵌入，使用掩码均值：
  \[
  u = \tanh(W_s \cdot \mathrm{mean\_mask}(E x_t)),\quad
  v = \tanh(W_a \cdot \mathrm{mean\_mask}(E y_t)).
  \]
- 将 $u \Vert v$ 拼接后经两层 MLP（ReLU-Linear-ReLU-Linear）输出标量值。
- 工厂类 `DemoNetworkFactory` (`src/train_demo.py:2625-2669`) 生成成对的 $Q$ 网络和策略网络，确保参数共享词表与超参数。

## SAC 损失构造
`DemoSACAgent.update` (`src/train_demo.py:2784-2897`) 实现软演员-评论家：
- 目标值：
  \[
  y = r + \gamma (1 - d) \cdot \mathbb{E}_{a'\sim\pi}\bigl[\min(Q_{\phi'}(s', a'), Q_{\psi'}(s', a')) - \alpha \log \pi(a'\mid s')\bigr].
  \]
  期望通过 Top-$p$ 候选集、权重 $w_i$ 与 log-prob $\log p_i$ 近似：
  $\sum_i w_i (\min(Q_{\phi'},Q_{\psi'}) - \alpha \log p_i)$。
- 评论家损失：$\mathcal{L}_{Q} = \mathrm{MSE}(Q_\phi(s,a), y) + \mathrm{MSE}(Q_\psi(s,a), y)$。
- 策略损失：
  \[
  \mathcal{L}_\pi = \mathbb{E}_{s\sim\mathcal{D}} \sum_i w_i (\alpha \log p_i - Q_\phi(s, a_i)).
  \]
- 温度自适应：$\alpha = e^{\log \alpha}$，目标熵 $H_{\text{tgt}} = \kappa \log |\mathcal{A}(s)|$，其中 $\kappa = \texttt{entropy\_kappa}$（默认 0.9）。损失
  $\mathcal{L}_\alpha = -\log\alpha \cdot (H_{\text{tgt}} - H_{\text{emp}})$。
- 目标网络通过指数滑动更新：$\theta' \leftarrow \tau \theta + (1-\tau) \theta'$ (`src/train_demo.py:2887-2895`)。

## 训练流程伪代码
```pseudo
function UPDATE(agent):
    batch = replay_buffer.sample(B)
    state_tokens, state_lengths = tokenizer.batch_encode(states)
    action_tokens, action_lengths = tokenizer.batch_encode(actions)

    with torch.no_grad():
        next_logits = policy.first_step_distribution(next_state_tokens)
        candidates = select_top_p(next_logits, top_p)
        q_expectation = combine_expectations(candidates, target_q1, target_q2, alpha)
        target = rewards + gamma * (1 - dones) * q_expectation

    q1 = q1_network(state_tokens, state_lengths, action_tokens, action_lengths)
    q2 = q2_network(state_tokens, state_lengths, action_tokens, action_lengths)
    optimize_mse(q1, target, optimizer_q1)
    optimize_mse(q2, target, optimizer_q2)

    freeze(q1, q2)
    policy_candidates = select_top_p(policy.first_step_distribution(state_tokens), top_p)
    policy_loss = expectation(alpha * log_prob - q1_candidate)
    optimize(policy_loss, policy_optimizer)
    unfreeze(q1, q2)

    entropy = -(prob * log_prob).sum()
    target_entropy = kappa * log(legal_token_count)
    alpha_loss = -(log_alpha * (target_entropy - entropy))
    optimize(alpha_loss, alpha_optimizer)
    clamp(log_alpha, [\log \alpha_{\min}, \log \alpha_{\max}])

    soft_update(target_q1, q1, tau)
    soft_update(target_q2, q2, tau)

    return metrics_dict
```

## 关键超参数
| 名称 | 默认来源 | 值/含义 |
| --- | --- | --- |
| `embedding_dim` | `DemoNetworkFactory.embedding_dim` | 字符嵌入维度，示例配置 96。 |
| `hidden_dim` | `DemoNetworkFactory.hidden_dim` | GRU/MLP 隐状态维度，示例配置 128。 |
| `max_summary_length` | `DemoNetworkFactory.max_summary_length` | 策略输出最大 token 长度。 |
| `COMPLIANCE_INVALID_LOGIT_PENALTY` | 常量 (`src/train_demo.py:413`) | 12.0，非法 token 罚分。 |
| `top_p` | `AgentConfig.top_p` | 默认 0.98，用于候选截断。 |
| `alpha` 范围 | `ALPHA_MIN/ALPHA_MAX` (`src/train_demo.py:390-392`) | $[10^{-4}, 2]$，更新时夹紧。 |
| `tau` | `AgentConfig.tau` | 软更新 EMA 系数，模板默认 0.01。 |

## 代码映射
- 合规与词典检查：`WordComplianceChecker`，`src/train_demo.py:230-381`。
- 策略网络定义与采样：`TextPolicyNetwork`，`src/train_demo.py:2371-2525`。
- 价值网络与工厂：`TextQNetwork`、`DemoNetworkFactory`，`src/train_demo.py:2572-2669`。
- SAC 代理与更新逻辑：`DemoSACAgent`，`src/train_demo.py:2669-2897`。
- 超参数常量：`src/train_demo.py:397-419`。
