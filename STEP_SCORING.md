# 当前 Step 打分方案说明

## 文字描述

在每个迭代 step 中，环境会拿到策略生成的摘要文本 $s$，并构造“上一轮摘要 $p$ 与当前章节 $c$ 的拼接文本”$x = p \Vert c$（若两者都存在，则在中间插入换行符）作为对照，基于二者的字符级匹配关系即时计算奖励。奖励不会依赖长度目标或截断规则，而是通过相似度、覆盖率与新颖度综合评估摘要质量，同时对乱码与词语合规性施加惩罚：

- **相似度 $\mathrm{sim}(s, x)$**：使用 `difflib.SequenceMatcher` 的 `ratio()` 作为字符序列的全局相似度。
- **覆盖率 $\mathrm{cov}(s, x)$**：统计匹配块的字符总数与源文本总字符数之比，衡量摘要对“上一轮摘要 + 当前章节”组合信息的涵盖程度。
- **复制率 $\mathrm{copy}(s, x)$**：取最长匹配块长度与摘要总长度之比，表示摘要中最大连续片段对源文本的直接复制程度；新颖度定义为 $\mathrm{nov}(s, x) = \max(0, 1 - \mathrm{copy}(s, x))$。
- **词向量余弦相似度 $\mathrm{lex\_cos}(s, c)$**：基于章节 TF-IDF 向量与摘要 TF-IDF 向量的余弦相似度，衡量摘要是否覆盖高权重词语。
- **词频分布相似度 $\mathrm{lex\_js}(s, c)$**：使用 Jensen-Shannon 相似度比较摘要与章节的概率分布，反映整体词频结构匹配度。
- **乱码比例 $\mathrm{garb}(s)$**：统计摘要中 `<unk>`、不可打印字符以及不在 `CharTokenizer` 字符集内的字符占比。计算时会将 `<unk>` 子串整体视作乱码，并排除换行、制表符等允许的控制字符。
- **词合规缺失率 $\mathrm{word\_nc}(s)$**：基于全部章节提取连续汉字 bigram 构成的词表，统计摘要中任一未出现过的汉字 bigram 或全新汉字占摘要全部汉字的比例，用于识别被随意拼接的词语或语序混乱的组合。

为了实现“只要表现稍有起色就给予重度奖励”的目标，所有指标都会先通过非线性放大奖励函数
\[
\phi(z; \alpha) = 1 - (1 - \operatorname{clip}(z, 0, 1))^{\alpha}
\]
再与各自权重相乘。$\operatorname{clip}(z, 0, 1)$ 表示把输入限制在 $[0,1]$ 区间。字符级指标（相似度、覆盖率、新颖度）使用 $\alpha = 4$，词汇指标（TF-IDF 余弦、Jensen-Shannon 相似度）使用 $\alpha = 3.5$。对过去的“惩罚项”改为奖励干净输出：先取“清洁度”$c = 1 - \operatorname{clip}(\mathrm{garb}(s), 0, 1)$ 与 $c' = 1 - \operatorname{clip}(\mathrm{word\_nc}(s), 0, 1)$，再套用指数 $\beta = 5$ 的同款放大函数。

因此最终奖励 $R(s, x)$ 为：
\[
\begin{aligned}
R(s, x) ={}& 0.6 \cdot \phi(\mathrm{sim}(s, x); 4) + 0.3 \cdot \phi(\mathrm{cov}(s, x); 4) + 0.1 \cdot \phi(\mathrm{nov}(s, x); 4)\\
&{}+ 0.15 \cdot \phi(\mathrm{lex\_cos}(s, c); 3.5) + 0.1 \cdot \phi(\mathrm{lex\_js}(s, c); 3.5)\\
&{}+ 0.5 \cdot \phi(1 - \mathrm{garb}(s); 5) + 0.7 \cdot \phi(1 - \mathrm{word\_nc}(s); 5).
\end{aligned}
\]

前三个正向项仍旧反映语义贴合度和信息覆盖的重要性，只是通过 $\phi$ 使得“略高于平均”的表现获得指数式奖励；词汇相关项帮助模型快速对齐关键词和词频结构。后两个“清洁奖励”保证只要摘要几乎没有乱码或不合规词语，就能立刻拿到大额奖励，而不是被线性扣分。

## 伪代码

```pseudo
function phi(value, exponent):
    clamped = clamp(value, 0, 1)
    return 1 - (1 - clamped) ** exponent

function compute_step_reward(summary_text, previous_summary, chapter_text, chapter_index, tokenizer, word_checker, lexical_stats, lexical_tokenizer):
    source_text = concatenate(previous_summary, "\n", chapter_text)  # 若任一为空则直接使用另一个
    matcher = SequenceMatcher(summary_text, source_text)
    similarity = matcher.ratio()

    match_blocks = matcher.get_matching_blocks()
    matched_chars = sum(block.size for block in match_blocks)
    longest_block = max(block.size for block in match_blocks, default=0)

    source_len = length(source_text)
    summary_len = length(summary_text)

    if source_len == 0:
        coverage = 0
    else:
        coverage = matched_chars / source_len

    if summary_len == 0:
        copy_ratio = 0
    else:
        copy_ratio = longest_block / summary_len

    novelty = max(0, 1 - copy_ratio)

    garbled_ratio = compute_garbled_ratio(summary_text, tokenizer)
    word_nc_ratio = compute_word_noncompliance_ratio(summary_text, word_checker)

    lexical_cosine = compute_lexical_cosine(summary_text, chapter_index, lexical_stats, lexical_tokenizer)
    lexical_js = compute_lexical_js(summary_text, chapter_index, lexical_stats, lexical_tokenizer)

    similarity_reward = 0.6 * phi(similarity, 4)
    coverage_reward = 0.3 * phi(coverage, 4)
    novelty_reward = 0.1 * phi(novelty, 4)
    lexical_cos_reward = 0.15 * phi(lexical_cosine, 3.5)
    lexical_js_reward = 0.1 * phi(lexical_js, 3.5)
    garbled_reward = 0.5 * phi(1 - garbled_ratio, 5)
    word_clean_reward = 0.7 * phi(1 - word_nc_ratio, 5)

    reward = (
        similarity_reward
        + coverage_reward
        + novelty_reward
        + lexical_cos_reward
        + lexical_js_reward
        + garbled_reward
        + word_clean_reward
    )
    return reward, {
        "similarity": similarity,
        "coverage_ratio": coverage,
        "copy_ratio": copy_ratio,
        "novelty_ratio": novelty,
        "garbled_ratio": garbled_ratio,
        "word_noncompliance_ratio": word_nc_ratio,
        "lexical_cosine": lexical_cosine,
        "lexical_js": lexical_js
    }
```

## 更新约定

若未来调整任一权重或改动指标的定义（例如改用其他相似度算法、覆盖率统计方式等），必须同步修改本文件中相应的数学公式、文字描述与伪代码，确保文档与实现保持一致。
