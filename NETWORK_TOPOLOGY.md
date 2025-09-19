# 当前网络拓扑结构设计方案

## 总览
- **输入**：上一轮摘要与当前章节文本按顺序拼接，记为 $x = [s_{t-1}; c_t]$。
- **输出**：策略网络生成的下一轮摘要 $\hat{s}_t$，并同时提供动作对数概率等训练信号。
- **核心组件**：
  1. 字符级嵌入层，将离散字符索引映射到 $\mathbb{R}^{d_{emb}}$。
  2. 编码器 GRU，提取观察序列的全局语境向量 $h_t$。
  3. 解码器 GRU，自回归地产生摘要字符分布。
  4. 双路 Q 网络，对 $(x, \hat{s}_t)$ 的价值进行评估，用于 SAC 的软价值更新。

## 策略网络拓扑
记字符词表大小为 $|\mathcal{V}|$，最大摘要长度为 $L_{max}$。策略网络 $\pi_\theta$ 包含如下层次：

1. **嵌入层** $E \in \mathbb{R}^{|\mathcal{V}| \times d_{emb}}$：
   $$
   Z = E(x) \in \mathbb{R}^{T \times d_{emb}}, \quad T = |x|。
   $$
2. **编码器 GRU** $\text{GRU}_{enc}$：
   $$
   h_t = \text{GRU}_{enc}(Z) \in \mathbb{R}^{d_{hid}}。
   $$
3. **解码器 GRU** $\text{GRU}_{dec}$：以 $h_t$ 为初始隐藏状态，逐步生成输出：
   $$
   \begin{aligned}
   y_0 &= \texttt{<bos>}\\
   y_k, h_{k+1} &= \text{GRU}_{dec}(E(y_{k-1}), h_k),\quad k = 1,\ldots,L_{max}。
   \end{aligned}
   $$
4. **输出投影** $W_o \in \mathbb{R}^{d_{hid} \times |\mathcal{V}|}$：
   $$
   p(y_k \mid y_{<k}, x) = \text{Softmax}(W_o h_k)。
   $$

伪代码描述策略网络：
```pseudo
function POLICY_FORWARD(x_tokens):
    Z = embedding(x_tokens)
    _, h = encoder_gru(pack(Z))
    y_prev = <bos>
    summary_tokens = []
    log_probs = []
    for step in 1..L_max:
        y_emb = embedding(y_prev)
        h, output = decoder_gru(y_emb, h)
        logits = linear(output)
        y_curr ~ Categorical(logits)
        summary_tokens.append(y_curr)
        log_probs.append(log_prob(y_curr))
        if y_curr == <eos>:
            break
        y_prev = y_curr
    return summary_tokens, sum(log_probs)
```

## 价值网络拓扑
价值网络 $Q_\phi$ 与 $Q_{\psi}$ 为同构的轻量级前馈架构：

1. **共享嵌入层**：与策略网络一致的 $E$，分别编码状态 $x$ 与动作 $\hat{s}_t$。
2. **掩码平均池化**：对填充后的序列取均值得到 $\bar{z}_x, \bar{z}_a \in \mathbb{R}^{d_{emb}}$。
3. **线性变换**：
   $$
   u = \tanh(W_s \bar{z}_x), \quad v = \tanh(W_a \bar{z}_a)。
   $$
4. **前馈头**：
   $$
   q = W_2 \sigma(W_1 [u; v])。
   $$

伪代码如下：
```pseudo
function Q_FORWARD(state_tokens, action_tokens):
    S = embedding(state_tokens)
    A = embedding(action_tokens)
    u = tanh(W_s * masked_mean(S))
    v = tanh(W_a * masked_mean(A))
    q = FFN(concat(u, v))
    return q
```

## 拓扑与数据流关系
- 观察编码：$x \xrightarrow{\text{CharTokenizer}} x_{ids} \xrightarrow{\pi_\theta} \hat{s}_t$。
- 奖励评估：将 $\hat{s}_t$ 与原始文本结合，计算相似度、覆盖率、新颖度及合规惩罚。
- 价值更新：$Q_\phi(x, \hat{s}_t)$、$Q_{\psi}(x, \hat{s}_t)$ 共享相同的嵌入与池化结构，实现软演员-评论家循环。

当网络拓扑结构设计方案发生任何调整（例如更换编码器类型、修改解码器深度或价值网络结构），须同步更新本文件与相关输出数据方案文档，确保描述与实现保持一致。
