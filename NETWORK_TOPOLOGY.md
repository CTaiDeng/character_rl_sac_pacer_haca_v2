# 当前网络拓扑结构设计方案

## 总览
- **输入**：上一轮摘要与当前章节文本按顺序拼接，记为 $x = [s_{t-1}; c_t]$。
- **输出**：策略网络生成的下一轮摘要 $\hat{s}_t$，并同时提供动作对数概率等训练信号。
- **核心组件**：
  1. 字符级嵌入层，基于频率筛选得到的 $\mathcal{V}_{\text{summary}}$ 与标点集，将离散字符索引映射到 $\mathbb{R}^{d_{emb}}$。
  2. 编码器 GRU，提取观察序列的全局语境向量 $h_t$。
  3. 解码器 GRU，自回归地产生摘要字符分布，并在采样前应用合规性温度调节。
  4. 双路 Q 网络，对 $(x, \hat{s}_t)$ 的价值进行评估，用于 SAC 的软价值更新。
  5. 词频统计缓存单元：离线载入 $\mathcal{L}$，为奖励评估提供章节 TF-IDF 与词频分布向量。

## 策略网络拓扑
记字符词表大小为 $|\mathcal{V}_{\text{summary}}|$，最大摘要长度为 $L_{max}$。策略网络 $\pi_\theta$ 包含如下层次：

1. **嵌入层** $E \in \mathbb{R}^{|\mathcal{V}_{\text{summary}}| \times d_{emb}}$：
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
4. **输出投影** $W_o \in \mathbb{R}^{d_{hid} \times |\mathcal{V}_{\text{summary}}|}$：
   $$
   \tilde{p}(y_k \mid y_{<k}, x) = W_o h_k。
   $$
5. **合规筛选与温度调节**：记上一步字符为 $p=y_{k-1}$，允许集合
   $$
   \mathcal{A}(p) = \{\text{EOS}\} \cup \{\chi \in \mathcal{V}_{\text{summary}} \mid \text{checker.is\_candidate\_allowed}(p, \chi)\}。
   $$
   对于 $\mathcal{A}(p)$ 外的 token，将对数几率减去罚项 $\delta=12$；若 $|\mathcal{A}(p)| < |\mathcal{V}_{\text{summary}}| + 1$，再对允许集合内的 logits 以温度 $\tau=0.85$ 进行缩放后再归一化：
   $$
   p(y_k \mid y_{<k}, x) = \text{Softmax}(\text{adjust}(\tilde{p}, \mathcal{A}(p)))。
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
        logits = apply_compliance(logits, y_prev)
        y_curr ~ Categorical(logits)
        summary_tokens.append(y_curr)
        log_probs.append(log_prob(y_curr))
        if y_curr == <eos>:
            break
        y_prev = y_curr
    return summary_tokens, sum(log_probs)

function APPLY_COMPLIANCE(logits, prev_token):
    allowed = {eos_id}
    for token_id in tokenizer.summary_token_ids:
        char = tokenizer.token_from_id(token_id)
        if word_checker.is_candidate_allowed(tokenizer.token_from_id(prev_token), char):
            allowed.add(token_id)
    logits[not in allowed] -= 12
    if len(allowed) < len(tokenizer.summary_token_ids) + 1:
        logits[allowed] /= 0.85
    return logits
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
- 观察编码：$x \xrightarrow{\text{CharTokenizer}} x_{ids} \xrightarrow{\pi_\theta} \hat{s}_t$，其中词表已按频率裁剪，降低无意义字符的生成概率。
- 奖励评估：将 $\hat{s}_t$ 与原始文本结合，计算相似度、覆盖率、新颖度及合规惩罚。
- 价值更新：$Q_\phi(x, \hat{s}_t)$、$Q_{\psi}(x, \hat{s}_t)$ 共享相同的嵌入与池化结构，实现软演员-评论家循环。

当网络拓扑结构设计方案发生任何调整（例如更换编码器类型、修改解码器深度或价值网络结构），须同步更新本文件与相关输出数据方案文档，确保描述与实现保持一致。



## 词频奖励支路
- 通过 `scripts/compute_chapter_tfidf.py` 预处理 `data/sample_article.txt`，生成 `data/sample_article_lexical.json`，其中保存每章 TF-IDF、概率分布及 IDF。
- 训练时一次载入 $\mathcal{L}$ 与对应的分词器，使 `analyze_summary` 能计算 `lexical_cosine` 与 `lexical_js` 两项指标。
- 词频相似度被线性加权注入奖励：$0.15\cdot\mathrm{lex\_cos}+0.1\cdot\mathrm{lex\_js}$，既鼓励覆盖关键信息，又保持整体词频结构稳定。
