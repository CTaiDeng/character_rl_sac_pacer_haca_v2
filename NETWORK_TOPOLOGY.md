# 当前网络拓扑结构设计

## 策略网络 $\pi_\theta$
- 输入：token 序列 $x_t \in \mathbb{N}^{L_t}$，先经词嵌入矩阵 $E \in \mathbb{R}^{|V| \times d}$ 得到 $X_t = E x_t$。
- 编码器：双层 GRU，隐藏维度 $h=128$，得到上下文表示 $h^{\text{enc}} = \mathrm{GRU}_{\text{enc}}(X_t)$。
- 解码器：条件 GRU，以 $h^{\text{enc}}$ 为初始状态，逐步生成 $y_t$，每步 logits
  \[
  z_\tau = W_o h_\tau + b_o,\qquad W_o \in \mathbb{R}^{|V| \times h}
  \]
- 合规掩码：`WordComplianceChecker` 根据前缀 $y_{<\tau}$ 构造可选集合 $\mathcal{A}(y_{<\tau})$，对非法 token 赋值 $-10^9$，温度缩放 $\tau_c = 0.85$。
- 采样：
  \[
  p_\theta(y_\tau \mid y_{<\tau}, x_t) = \mathrm{Softmax}(z_\tau / \tau_c)\big|_{\mathcal{A}(y_{<\tau})}
  \]
  训练阶段采用 Gumbel-Softmax 近似，推理阶段使用 Top-$p$ ($p=0.98$) 抽样。

## 值函数 $Q_\phi, Q_\psi$
- 输入：状态 token $x_t$ 与动作 token $y_t$。
- 状态嵌入：
  \[
  u = \tanh\bigl(W_s\, \mathrm{mean\_pool}(E x_t) + b_s\bigr)
  \]
- 动作嵌入：
  \[
  v = \tanh\bigl(W_a\, \mathrm{mean\_pool}(E y_t) + b_a\bigr)
  \]
- 拼接后通过两层 MLP 得到 $Q_\phi(s, a)$；$Q_\psi$ 结构一致，参数独立。
- 目标网络使用指数滑动：
  \[
  \theta' \leftarrow \tau \theta + (1-\tau)\theta',\qquad \phi', \psi' \text{ 同理}
  \]

## 熵系数与损失
- 目标熵：
  \[
  H_{\text{tgt}} = \kappa \log |\mathcal{A}(x_t)|,\qquad \kappa = 0.9
  \]
- 自适应熵系数 $\alpha$ 更新：
  \[
  \mathcal{L}_{\alpha} = -\alpha \cdot (\log \pi_\theta(y_t \mid x_t) + H_{\text{tgt}})
  \]
- 策略损失：
  \[
  \mathcal{L}_{\pi} = \mathbb{E}_{(s,a) \sim \mathcal{D}}[\alpha \log \pi_\theta(a\mid s) - Q_\phi(s, a)]
  \]
- 值函数损失：
  \[
  \mathcal{L}_{Q} = \mathbb{E}\left[\bigl(Q_\phi(s,a) - y\bigr)^2 + \bigl(Q_\psi(s,a) - y\bigr)^2\right],
  \]
  其中
  \[
  y = r + \gamma\left(\min(Q_{\phi'}(s', a') , Q_{\psi'}(s', a')) - \alpha \log \pi_\theta(a'\mid s')\right).
  \]

## 前向伪代码
```pseudo
function POLICY_FORWARD(tokens, lengths):
    embedded = embedding(tokens)
    _, hidden = encoder_gru(embedded, lengths)
    prev = BOS
    outputs = []
    for step in range(MAX_DEC_LEN):
        logits, hidden = decoder_gru(prev, hidden)
        logits = apply_compliance_mask(logits, outputs)
        probs = top_p_filter(softmax(logits / TAU_C), TOP_P)
        sample = sample_from(probs)
        outputs.append(sample)
        if sample == EOS:
            break
        prev = sample
    return outputs
```

## 参数与超参
| 模块 | 关键参数 | 数值 | 说明 |
| --- | --- | --- | --- |
| 嵌入层 | `embedding_dim` | 96 | 词向量维度 |
| 编码 GRU | `hidden_dim` | 128 | 双向堆叠两层 |
| 解码 GRU | `hidden_dim` | 128 | 使用注意力初始化 |
| 解码上限 | `max_summary_length` | 512 | 文本模式最大步骤 |
| 优化器 | `Adam` 学习率 | $3\times10^{-4}$ | 策略与价值共享 |
| 熵系数 | `alpha` 初始值 | 0.2 | 通过 $\mathcal{L}_\alpha$ 自适应 |
| 软更新 | $\tau$ | 0.01 | 目标网络 EMA 系数 |

## 训练拓扑
1. 从 `SimpleReplayBuffer` 采样 batch，解码得到 $(s, a, r, s', \text{done})$。
2. 计算目标动作 $a'$：对 $s'$ 运行策略并采样，保留 `stop_gradient` 的 logits 与 log-prob。
3. 更新 $Q_\phi, Q_\psi$ 使其拟合目标 $y$；同步累积梯度裁剪到 $\pm 5$。
4. 更新策略：最小化 $\mathcal{L}_{\pi}$，并记录 `policy_loss`、`entropy`、`alpha`。
5. 更新熵系数 $\alpha$，保持在区间 $[10^{-4}, 2]$。
6. 每步完成后执行软更新，刷新 (encoder, decoder, embedding) 的目标副本。

## 子模块关系概览
- Tokenizer (`CharTokenizer`) 向策略提供索引，策略输出经 `OperationParser` 转化为结构化动作。
- `TextPolicyNetwork`、`TextQNetwork`、`ValueTargetNetwork` 共享嵌入矩阵，减少参数漂移。
- 字符模式下额外分支 `CharacterPolicyHead` 复用相同编码器，但输出 2-token 的 bigram logits，并与主策略共享熵参数。

若网络结构、超参或采样策略调整，请同步更新本文件与相关实现。
