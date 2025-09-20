# 当前输入-输出数据方案说明

## 总览
训练环境以章节为单位推进，记文章共有 $T$ 个章节。第 $t$ 步（Iteration/Step）收到的观测为
$$
O_t = (S_{t-1}, C_t)
$$
其中：
- $S_{t-1}$ 表示上一轮策略生成的整段摘要文本；
- $C_t$ 表示当前章节的全文原文。

策略网络直接输出新的章节摘要 $S_t$，并作为后续步骤的历史上下文。

## 数据结构
| 符号 | 含义 | 具体来源 |
| --- | --- | --- |
| $S_0$ | 初始摘要，置为空字符串 | 训练开始时环境重置 |
| $C_t$ | 第 $t$ 章原文 | `load_article_features()` 读取 `data/sample_article.txt` 并按章节切分 |
| $O_t$ | 观测对 | `TextObservation(previous_summary=S_{t-1}, chapter_text=C_t)` |
| $A_t$ | 行动（模型输出） | `TextAction(text=S_t, token_ids, length)` |
| $R_t$ | 即时奖励 | `analyze_summary(S_t, S_{t-1}+C_t)` 产生的质量分数 |
| \mathcal{L} | 章节词频缓存（TF-IDF 与概率分布） | `data/sample_article_lexical.json`，由 `scripts/compute_chapter_tfidf.py` 预生成 |

`CharTokenizer` 在上述观测与行动上构造字符级序列：
- 观测编码：`[BOS] + chars(S_{t-1}) + [SEP] + chars(C_t) + [EOS]`。
- 行动编码：`[BOS] + chars(S_t) + [EOS]`。
- 汉字词表限定在 $\mathcal{V}_{\text{summary}}$（见下节），其余字符映射到 `<unk>` 以保持编码稳定。

## 字符集约束与采样调节
- 从 `data/sample_article.txt` 统计每个汉字的出现次数 $f(\chi)$，令 $\mu = \operatorname{mean}(f(\chi))$、$\sigma = \operatorname{pstdev}(f(\chi))$，抽取高频集合
  $$\mathcal{V}_{\text{common}} = \{\chi \mid f(\chi) \ge \mu + \sigma\}$$
  若集合为空，则回退到按频次排序的前 $200$ 个汉字。最终可用摘要字符集为
  $$\mathcal{V}_{\text{summary}} = \mathcal{V}_{\text{common}} \cup \Pi$$
  其中 $\Pi$ 为常用中英文标点集合。
- 采样阶段根据 `WordComplianceChecker` 的双字库限制可选字符。记上一步生成的字符为 $p$，允许集合
  $$\mathcal{A}(p) = \{\text{EOS}\} \cup \{\chi \in \mathcal{V}_{\text{summary}} \mid \text{checker.is_candidate_allowed}(p, \chi)\}$$
  对于 $\mathcal{A}(p)$ 外的 token，将其对数几率减去固定惩罚 $\delta = 12$；若 $\lvert \mathcal{A}(p)\rvert < \lvert\mathcal{V}_{\text{summary}}\rvert + 1$，则对允许集合的 logits 以温度 $\tau = 0.85$ 进行缩放，使概率集中于合规片段。
- 相关伪代码：
```pseudo
allowed <- {eos_id}
for token_id in tokenizer.summary_token_ids:
    char <- tokenizer.token_from_id(token_id)
    if word_checker.is_candidate_allowed(prev_char, char):
        allowed.add(token_id)
mask <- ones_like(logits)
mask[allowed] <- 0
logits[mask == 1] -= 12
if len(allowed) < len(tokenizer.summary_token_ids) + 1:
    logits[allowed] /= 0.85
```

## 迭代流程伪代码
```pseudo
S_0 <- ""
for round in 1..R:
    for t in 1..T:
        observation <- TextObservation(previous_summary=S_{t-1}, chapter_text=C_t)
        obs_tokens <- tokenizer.encode_observation(observation)
        action_tokens <- policy.sample(obs_tokens)
        S_t <- tokenizer.decode_action(action_tokens)
        reward_t, metrics_t <- analyze_summary(
            summary=S_t,
            source=concat(S_{t-1}, C_t),
            chapter_index=t,
            lexical_stats=\mathcal{L},
            lexical_tokenizer=lexical_tokenizer
        )
        buffer.push(
            state=observation,
            action=TextAction(token_ids=action_tokens, text=S_t, length=len(S_t)),
            reward=reward_t,
            next_state=TextObservation(S_t, C_{t+1}) if t < T else terminal_state
        )
        log_step(round, t, S_t, metrics_t)
    update_agent(buffer, episodes=post_round_updates)
    log_round(round, sum_{t=1}^T reward_t)
```

## 输入与输出要点
- **输入**：上一轮摘要与当前章节全文的直接拼接，既以原始字符串形式供日志使用，又以字符 ID 序列供模型计算。
- **输出**：策略生成的整段摘要文本 $S_t$，不再做长度裁剪，仅在日志中记录其字符数、质量指标与惩罚项。
- **辅助缓存**：\mathcal{L} 持久化章节词频，奖励计算阶段一次读取，多次复用，避免重复统计。
- **日志扩展**：在原有指标基础上新增 `lexical_cosine` 与 `lexical_js` 等词频相似度输出，便于离线诊断。
- **依赖关系**：$S_t$ 同时成为下一步的 $S_{t-1}$，保证“摘要递推”链条成立；回放缓冲区按 `Transition` 记录 $(O_t, A_t, R_t, O_{t+1})$。
- **终止条件**：当 $t = T$ 时标记 `done=True`，一轮结束后清空累积摘要并重新开始下一轮训练。

> 若未来调整观测或行动的组成（例如改用子词分词、引入多模态信号），需同步更新本文档，确保开发者始终掌握最新的数据接口约定。
