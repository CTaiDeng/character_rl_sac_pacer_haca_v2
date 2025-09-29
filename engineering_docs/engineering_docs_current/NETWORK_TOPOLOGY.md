# 当前网络拓扑结构方案

## 字符模式拓扑规则
记时刻 t：
- prev = 上一轮摘要预览（取末尾 ≤1 字用于渲染与构造 source）；
- chapter = 当前目标字符；
- source = prev ⊕ chapter；
- 词表 Catalog 包含 `data/chinese_name_frequency_word.json` 与 `data/chinese_frequency_word.json`；
- 允许后缀长度集合 U = `data/word_length_sets.json.union.lengths`（例如 {2,3,4,5,6,7,8,9,10,11,13}）。

1) 前缀左扩（保证 source 头两字命中）
- 若 len(source)≥2 且 source[:2] 不在 Catalog，则沿“历史字符对”向左扩展 prev（每步把一个历史对的首字加到 prev 之前），至多 N=character_history_extension_limit 次；当 source[:2] 命中或达到 N 即停。

伪代码：
```pseudo
function EXTEND_PREV_FOR_PREFIX_HIT(prev, chapter, history_pairs, N):
    source = prev + chapter
    if len(source) < 2: return prev
    step = 0
    while source[:2] not in Catalog and step < N and history_pairs.has_left(step):
        pair = history_pairs.left(step)
        if prev and not prev.startswith(pair.tail):
            step += 1; continue
        prev = pair.head + prev
        source = prev + chapter
        step += 1
    return prev
```

2) raw_action 的后缀拓扑（去重首字，遇“有意义词”即停）
- 初始为当前预测字符 c；若后续预测串出现以 c 开头的重复首字，则去重一次；
- 按时间向前追加未来字符，形成候选串 q；
- 对每一步的 q，在 U 中按降序枚举长度 L，若 q 的后缀 q[-L:] 在 Catalog 命中则停（把该后缀作为“有意义词”）；
- 记录注记（包含来源词表与编号）。

伪代码：
```pseudo
function EXTEND_RAW_ACTION_SUFFIX(initial_char, future_chars, U):
    q = dedup_head_repeat(initial_char)
    suffix, ann = "", ""
    for ch in future_chars:
        q += ch
        for L in sort_desc(U ∩ [1..len(q)]):
            seg = tail(q, L)
            if is_cjk(seg) and in_catalog(seg):
                return q, seg, annotate(seg)
        if len(q) > max(U): continue
    # 未命中时保留最长连续 CJK 片段作为回退
    return q, longest_cjk_tail(q, max(U)), annotate_optional()
```

3) bigram 的前向拓扑（来自 raw_action 的扩展）
- 以 base = chapter ⊕ raw_action 开始，按未来字符向前拓展；
- 在每次拓展后，对 base 的后缀按 U 的降序判定命中；命中即停；
- bigram 的注记显示“后缀命中”的词与词表来源。

伪代码：
```pseudo
function FORWARD_EXTEND_BIGRAM(chapter, raw_action, future_chars, U):
    s = chapter + raw_action
    best = tail(s, min(2, len(s)))
    for ch in future_chars:
        s += ch
        for L in sort_desc(U ∩ [1..len(s)]):
            seg = tail(s, L)
            if is_cjk(seg) and in_catalog(seg):
                return seg, s
    return best, s
```

备注：
- 历史左扩的 N 由 `res/config.json`/`config_template.json` 的 `character_history_extension_limit`（默认 16）控制；
- Catalog 命中采用“最长可用”优先；注记形如 `data/chinese_frequency_word.json\#<id>`。


## 策略网络 $\pi_\theta$
- 观测拼接：字符模式下 tokens 由 prev + [<sep>] + chapter(目标字符) 组成，确保模型显式看到目标字符。
- 结构：`TextPolicyNetwork`（见 `src/character_sac_trainer.py`）使用共享字符词表。
  1. 嵌入层 $E \in \mathbb{R}^{|V|\times d_{\text{emb}}}$ 将 tokens 映射为向量。
  2. 编码 GRU（单向，hidden dim $d_h$）对输入序列执行 `pack_padded_sequence` 编码，得到上下文隐状态 $h^{\text{enc}}$。
  3. 解码 GRU 以 $h^{\text{enc}}$ 为初始状态自回归地产生输出 token，步进时应用合规模块 `_mask_logits`。
  4. 输出层 $W_o \in \mathbb{R}^{|V|\times d_h}$，logits 经温度缩放 $\tau_c = \texttt{DEFAULT\_COMPLIANCE\_TEMPERATURE} = 0.85$ 与合规掩码处理。
- 合规模块：给定前一 token $y_{t,\tau-1}$、`summary_token_ids` 与 `WordComplianceChecker`（见 `src/character_sac_trainer.py`），构造允许集合 $\mathcal{A}(y_{<\tau})$，对非法 token 赋值 $\texttt{COMPLIANCE\_MASK\_FILL\_VALUE}=-10^9$。
- 采样：`DemoSACAgent._select_top_p`（见 `src/character_sac_trainer.py`）对 softmax 概率执行 Top-$p$ 过滤，$p$ 来自配置（默认 0.98），并返回候选集合、归一化权重和 log-prob。
- 确定性推理：`TextPolicyNetwork.deterministic` 使用贪心选取允许集合中概率最大的 token。

## 价值网络 $Q_\phi, Q_\psi$
- 结构：`TextQNetwork`（见 `src/character_sac_trainer.py`）对状态与动作序列分别嵌入，使用掩码均值：
  $$
  u = \tanh(W_s \cdot \mathrm{mean\_mask}(E x_t)),\quad
  v = \tanh(W_a \cdot \mathrm{mean\_mask}(E y_t)).
  $$
- 将 $u \Vert v$ 拼接后经两层 MLP（ReLU-Linear-ReLU-Linear）输出标量值。
- 工厂类 `DemoNetworkFactory`（见 `src/character_sac_trainer.py`）生成成对的 $Q$ 网络和策略网络，确保参数共享词表与超参数。

## SAC 损失构造
`DemoSACAgent.update`（见 `src/character_sac_trainer.py`）实现软演员-评论家：
- 目标值：
  $$
  y = r + \gamma (1 - d) \cdot \mathbb{E}_{a'\sim\pi}\bigl[\min(Q_{\phi'}(s', a'), Q_{\psi'}(s', a')) - \alpha \log \pi(a'\mid s')\bigr].
  $$
  期望通过 Top-$p$ 候选集、权重 $w_i$ 与 log-prob $\log p_i$ 近似：
  $\sum_i w_i (\min(Q_{\phi'},Q_{\psi'}) - \alpha \log p_i)$。
- 评论家损失：$\mathcal{L}_{Q} = \mathrm{MSE}(Q_\phi(s,a), y) + \mathrm{MSE}(Q_\psi(s,a), y)$。
- 策略损失：
  $$
  \mathcal{L}_\pi = \mathbb{E}_{s\sim\mathcal{D}} \sum_i w_i (\alpha \log p_i - Q_\phi(s, a_i)).
  $$
- 温度自适应：$\alpha = e^{\log \alpha}$，目标熵 $H_{\text{tgt}} = \kappa \log |\mathcal{A}(s)|$，其中 $\kappa = \texttt{entropy\_kappa}$（默认 0.9）。损失
  $\mathcal{L}_\alpha = -\log\alpha \cdot (H_{\text{tgt}} - H_{\text{emp}})$。
- 目标网络通过指数滑动更新：$\theta' \leftarrow \tau \theta + (1-\tau) \theta'$（见 `src/character_sac_trainer.py`）。

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
| `COMPLIANCE_INVALID_LOGIT_PENALTY` | 常量（见 `src/character_sac_trainer.py`） | 12.0，非法 token 罚分。 |
| `top_p` | `AgentConfig.top_p` | 默认 0.98，用于候选截断。 |
| `alpha` 范围 | `ALPHA_MIN/ALPHA_MAX`（见 `src/character_sac_trainer.py`） | $[10^{-4}, 2]$，更新时夹紧。 |
| `tau` | `AgentConfig.tau` | 软更新 EMA 系数，模板默认 0.01。 |

## 代码映射
- 合规与词典检查：`WordComplianceChecker`，见 `src/character_sac_trainer.py`。
- 策略网络定义与采样：`TextPolicyNetwork`，见 `src/character_sac_trainer.py`。
- 价值网络与工厂：`TextQNetwork`、`DemoNetworkFactory`，见 `src/character_sac_trainer.py`。
- SAC 代理与更新逻辑：`DemoSACAgent`，见 `src/character_sac_trainer.py`。
- 超参数常量：见 `src/character_sac_trainer.py`。

