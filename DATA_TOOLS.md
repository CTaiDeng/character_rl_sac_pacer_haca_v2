#+ 数据工具与映射说明

- 输入-输出-打分映射（JSON）
  - 文件：`data/io_score_mapping.json`
  - 作用：给出最小映射 schema（input/output/score）与示例，便于脚本/服务统一记录样例。

- 词长集合生成器
  - 文件：`data/gen_word_length_sets.py`
  - 用法：`python -m data.gen_word_length_sets`
  - 输出：`data/word_length_sets.json`（names/freq/union 三块，供可变长度后缀命中使用）。

- 词表命中查询模块
  - 文件：`data/catalog_lookup.py`（可 `from data import catalog_lookup`）
  - 接口：`load_catalog()`、`annotate(term)`、`longest_prefix_hit(text,lengths)`、`suffix_hit(text,lengths)`
  - CLI 示例：
    - 标注：`python -m data.catalog_lookup --query "精妙"`
    - 前缀：`python -m data.catalog_lookup --prefix "精妙。如" --lengths 2,3,4`
    - 后缀：`python -m data.catalog_lookup --suffix "”他喃喃" --lengths 2,3,4`

- JSONLINE 转结构化 JSON
  - 文件：`data/jsonl_to_json.py`
  - 作用：将若干 `*.jsonl` 文件（逐行 JSON）合并为结构化 JSON 数组并美化缩进（UTF-8）。
  - 用法：
    - 转单个文件：`python -m data.jsonl_to_json out/train_123/teacher_trajectory.jsonl`
    - 批量（递归）：`python -m data.jsonl_to_json out --recursive`
    - 覆盖已有输出：`python -m data.jsonl_to_json out --recursive --force`
    - 控制缩进/ASCII：`python -m data.jsonl_to_json out --indent 2 --ascii`

> 说明：
> - `union.lengths` 被 `src/character_sac_trainer.py` 用作 raw_action/bigram 的可变长度后缀命中集合；
> - `catalog_lookup.annotate` 统一了“命中/未命中#编号”的注记显示；
> - 运行字符模式时，若 `action_source=teacher`，训练目录 `out/train_*/teacher_samples.jsonl` 会追加监督样本（输入→输出），输入为 `prev/<sep>/chapter`，输出为“下一字符”。
