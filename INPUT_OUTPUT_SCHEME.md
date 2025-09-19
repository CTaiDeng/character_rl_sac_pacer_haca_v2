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

`CharTokenizer` 在上述观测与行动上构造字符级序列：
- 观测编码：`[BOS] + chars(S_{t-1}) + [SEP] + chars(C_t) + [EOS]`。
- 行动编码：`[BOS] + chars(S_t) + [EOS]`。
- 所有字符均基于 UTF-8 词表，并保留 `<unk>` 以处理未登录字符。

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
            source=concat(S_{t-1}, C_t)
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
- **依赖关系**：$S_t$ 同时成为下一步的 $S_{t-1}$，保证“摘要递推”链条成立；回放缓冲区按 `Transition` 记录 $(O_t, A_t, R_t, O_{t+1})$。
- **终止条件**：当 $t = T$ 时标记 `done=True`，一轮结束后清空累积摘要并重新开始下一轮训练。

> 若未来调整观测或行动的组成（例如改用子词分词、引入多模态信号），需同步更新本文档，确保开发者始终掌握最新的数据接口约定。
