# 当前 Step 打分方案说明

## 文字描述

在每个迭代 step 中，环境会拿到策略生成的摘要文本 $s$，并构造“上一轮摘要 $p$ 与当前章节 $c$ 的拼接文本”$x = p \Vert c$（若两者都存在，则在中间插入换行符）作为对照，基于二者的字符级匹配关系即时计算奖励。奖励不会依赖长度目标或截断规则，而是通过相似度、覆盖率与新颖度综合评估摘要质量，同时对乱码与词语合规性施加惩罚：

- **相似度 $\mathrm{sim}(s, x)$**：使用 `difflib.SequenceMatcher` 的 `ratio()` 作为字符序列的全局相似度。
- **覆盖率 $\mathrm{cov}(s, x)$**：统计匹配块的字符总数与源文本总字符数之比，衡量摘要对“上一轮摘要 + 当前章节”组合信息的涵盖程度。
- **复制率 $\mathrm{copy}(s, x)$**：取最长匹配块长度与摘要总长度之比，表示摘要中最大连续片段对源文本的直接复制程度；新颖度定义为 $\mathrm{nov}(s, x) = \max(0, 1 - \mathrm{copy}(s, x))$。
- **乱码比例 $\mathrm{garb}(s)$**：统计摘要中 `<unk>`、不可打印字符以及不在 `CharTokenizer` 字符集内的字符占比。计算时会将 `<unk>` 子串整体视作乱码，并排除换行、制表符等允许的控制字符。
- **词合规缺失率 $\mathrm{word\_nc}(s)$**：基于全部章节提取连续汉字 bigram 构成的词表，统计摘要中任一未出现过的汉字 bigram 或全新汉字占摘要全部汉字的比例，用于识别被随意拼接的词语或语序混乱的组合。

最终奖励 $R(s, x)$ 将上述质量项与乱码惩罚结合：
\[
R(s, x) = 0.6 \cdot \mathrm{sim}(s, x) + 0.3 \cdot \mathrm{cov}(s, x) + 0.1 \cdot \mathrm{nov}(s, x) - 0.5 \cdot \mathrm{garb}(s) - 0.7 \cdot \mathrm{word\_nc}(s).
\]

前三个正向权重反映了我们优先追求整体语义相似和信息覆盖，同时对保持一定的新颖度给予次要奖励；两个负号项则针对编码质量与汉字组合进行约束：只要摘要里含有乱码或词语组合未在原文中出现，就会按照比例扣分，以此鼓励策略输出干净、语义连贯的中文句子。

## 伪代码

```pseudo
function compute_step_reward(summary_text, previous_summary, chapter_text, tokenizer, word_checker):
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

    reward = (
        0.6 * similarity
        + 0.3 * coverage
        + 0.1 * novelty
        - 0.5 * garbled_ratio
        - 0.7 * word_nc_ratio
    )
    return reward, {
        "similarity": similarity,
        "coverage_ratio": coverage,
        "copy_ratio": copy_ratio,
        "novelty_ratio": novelty,
        "garbled_ratio": garbled_ratio,
        "word_noncompliance_ratio": word_nc_ratio
    }
```

## 更新约定

若未来调整任一权重或改动指标的定义（例如改用其他相似度算法、覆盖率统计方式等），必须同步修改本文件中相应的数学公式、文字描述与伪代码，确保文档与实现保持一致。
