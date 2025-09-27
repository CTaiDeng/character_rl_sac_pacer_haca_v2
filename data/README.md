# data 目录使用手册

> 摘要：本手册汇总 `data/` 目录的文件与脚本，按“语料与示例/生成物与缓存/词表与映射/脚本与 CLI”四类进行分类说明，逐一注明用途与典型用法，并标注文件间的生成关系（例如 `gen_word_length_sets.py` 生成 `word_length_sets.json`，`extract_chinese_names.py` 产出人名词表等）。新增、删除或重命名文件时，请同步更新本清单，确保与开发协议保持一致，便于团队快速查阅与维护。

---

## 语料与示例

- `sample_article.txt`
  - 用途：示例文章语料。多段落，通过分隔符 `"[----------------------------------------------------->"` 切分为“章节”。
  - 关联：被 `scripts/compute_chapter_tfidf.py`、`scripts/extract_chinese_names.py` 等脚本读取。

- `chapter_iterative_io_examples.txt`
  - 用途：字符模式下的“上一轮摘要 + 当前章节 → 当前操作/摘要”迭代 I/O 示例，展示 ACQUIRE/VERIFY/LINK/COMMIT 等操作序列与风格。

---

## 生成物与缓存

- 注：以下文件通常通过脚本生成或更新，不建议手工编辑。

- `sample_article_lexical.json`
  - 用途：章节级词法统计（TF‑IDF、概率分布、top 词项等）。
  - 来源：`scripts/compute_chapter_tfidf.py` 从 `sample_article.txt` 生成。
  - 下游：`scripts/evaluate_lexical_reward.py` 评估摘要与章节的词法相似度。

- `chinese_name_frequency_word.json`
  - 用途：中文实体/术语词表（以人名为主，含地名/机构名与 ASCII 术语）。
  - 来源：`scripts/extract_chinese_names.py` 从 `sample_article.txt` 与频次词表抽取、筛选而来。
  - 下游：作为 Catalog 与 `data/catalog_lookup.py` 的数据源之一。

- `word_length_sets.json`
  - 用途：统计两类词表（人名词表与频次词表）的长度集合，提供 `names/freq/union.lengths`（示例：$\texttt{{2,3,4,5,6,7,8,9,10,11,13}}$）。
  - 来源：`data/gen_word_length_sets.py` 基于 `chinese_name_frequency_word.json` 与 `chinese_frequency_word.json` 生成。
  - 下游：联合长度集 `union.lengths` 用于“可变长度后缀命中（lexical bigram）”等逻辑。

---

## 词表与映射

- `chinese_frequency_word.json`
  - 用途：通用高频词表/字词词频数据，作为 Catalog 参考与长度统计的数据源。

- `io_score_mapping.json`
  - 用途：输入‑输出‑评分的轻量 Schema 示例（input/prev|chapter|source；output/action_text；score/reward|components）。
  - 备注：用于演示统一的 I/O 与评分记录格式，便于日志与可视化。

---

## 脚本与 CLI

- `catalog_lookup.py`
  - 用途：词表 Catalog 查询与注记工具，统一从 `chinese_name_frequency_word.json` 与 `chinese_frequency_word.json` 读取并标注来源。
  - 典型用法：
    - 注记：`python -m data.catalog_lookup --query "示例词"`
    - 前缀命中：`python -m data.catalog_lookup --prefix "文本..." --lengths 2,3,4`
    - 后缀命中：`python -m data.catalog_lookup --suffix "...文本" --lengths 2,3,4`

- `gen_word_length_sets.py`
  - 用途：统计两类词表的长度集合，生成 `word_length_sets.json`（含 `names/freq/union.lengths`）。
  - 典型用法：`python data/gen_word_length_sets.py`

- `jsonl_to_json.py`
  - 用途：将 `*.jsonl` 批量转换为数组形式的 `*.json`，便于下游加载/可视化。
  - 典型用法：
    - 处理单文件：`python data/jsonl_to_json.py path/to/file.jsonl`
    - 递归遍历：`python data/jsonl_to_json.py data/ --recursive --pattern "*.jsonl"`

- `__init__.py`
  - 用途：打包入口与模块说明，便于以 `from data import catalog_lookup` 方式引用工具。

- `__pycache__/`
  - 用途：Python 字节码缓存目录（自动生成，已被忽略）。

---

## 维护与开发协议对接

- 当 `data/` 目录新增、删除或重命名文件（含脚本与数据集）时，必须同步更新本清单，保持与 `AGENTS.md` 的“数据目录维护规范”一致。
- 若 `word_length_sets.json.union.lengths` 或示例语料的分隔符/结构发生变更，需同步更新顶层方案文档与本清单的“生成物与来源”部分，避免文档‑实现不一致。

