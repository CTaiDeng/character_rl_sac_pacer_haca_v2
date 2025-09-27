# data 目录使用手册

> 摘要：本手册汇总 $\texttt{data/}$ 目录的文件与脚本，按“语料与示例/生成物与缓存/词表与映射/脚本与 CLI”四类进行分类说明，逐一注明用途与典型用法，并标注文件间的生成关系（例如 $\texttt{gen\_word\_length\_sets.py}$ 生成 $\texttt{word\_length\_sets.json}$，$\texttt{extract\_chinese\_names.py}$ 产出人名词表等）。新增、删除或重命名文件时，请同步更新本清单，确保与开发协议保持一致，便于团队快速查阅与维护。

---

## 语料与示例

- $\texttt{sample\_article.txt}$
  - 用途：示例文章语料。多段落，通过分隔符 $\texttt{"[----------------------------------------------------->"}$ 切分为“章节”。
  - 关联：被 $\texttt{scripts/compute\_chapter\_tfidf.py}$、$\texttt{scripts/extract\_chinese\_names.py}$ 等脚本读取。

- $\texttt{chapter\_iterative\_io\_examples.txt}$
  - 用途：字符模式下的“上一轮摘要 + 当前章节 → 当前操作/摘要”迭代 I/O 示例，展示 ACQUIRE/VERIFY/LINK/COMMIT 等操作序列与风格。

---

## 生成物与缓存

- 注：以下文件通常通过脚本生成或更新，不建议手工编辑。

- $\texttt{sample\_article\_lexical.json}$
  - 用途：章节级词法统计（TF‑IDF、概率分布、top 词项等）。
  - 来源：$\texttt{scripts/compute\_chapter\_tfidf.py}$ 从 $\texttt{sample\_article.txt}$ 生成。
  - 下游：$\texttt{scripts/evaluate\_lexical\_reward.py}$ 评估摘要与章节的词法相似度。

- $\texttt{chinese\_name\_frequency\_word.json}$
  - 用途：中文实体/术语词表（以人名为主，含地名/机构名与 ASCII 术语）。
  - 来源：$\texttt{scripts/extract\_chinese\_names.py}$ 从 $\texttt{sample\_article.txt}$ 与频次词表抽取、筛选而来。
  - 下游：作为 Catalog 与 $\texttt{data/catalog\_lookup.py}$ 的数据源之一。

- $\texttt{word\_length\_sets.json}$
  - 用途：统计两类词表（人名词表与频次词表）的长度集合，提供 $\texttt{names/freq/union.lengths}$（示例：$\texttt{{2,3,4,5,6,7,8,9,10,11,13}}$）。
  - 来源：$\texttt{data/gen\_word\_length\_sets.py}$ 基于 $\texttt{chinese\_name\_frequency\_word.json}$ 与 $\texttt{chinese\_frequency\_word.json}$ 生成。
  - 下游：联合长度集 $\texttt{union.lengths}$ 用于“可变长度后缀命中（lexical bigram）”等逻辑。

---

## 词表与映射

- $\texttt{chinese\_frequency\_word.json}$
  - 用途：通用高频词表/字词词频数据，作为 Catalog 参考与长度统计的数据源。

- $\texttt{io\_score\_mapping.json}$
  - 用途：输入‑输出‑评分的轻量 Schema 示例（input/prev|chapter|source；output/action_text；score/reward|components）。
  - 备注：用于演示统一的 I/O 与评分记录格式，便于日志与可视化。

---

## 脚本与 CLI

- $\texttt{catalog\_lookup.py}$
  - 用途：词表 Catalog 查询与注记工具，统一从 $\texttt{chinese\_name\_frequency\_word.json}$ 与 $\texttt{chinese\_frequency\_word.json}$ 读取并标注来源。
  - 典型用法：
    - 注记：$\texttt{python -m data.catalog\_lookup --query "示例词"}$
    - 前缀命中：$\texttt{python -m data.catalog\_lookup --prefix "文本..." --lengths 2,3,4}$
    - 后缀命中：$\texttt{python -m data.catalog\_lookup --suffix "...文本" --lengths 2,3,4}$

- $\texttt{gen\_word\_length\_sets.py}$
  - 用途：统计两类词表的长度集合，生成 $\texttt{word\_length\_sets.json}$（含 $\texttt{names/freq/union.lengths}$）。
  - 典型用法：$\texttt{python data/gen\_word\_length\_sets.py}$

- $\texttt{jsonl\_to\_json.py}$
  - 用途：将 $\texttt{*.jsonl}$ 批量转换为数组形式的 $\texttt{*.json}$，便于下游加载/可视化。
  - 典型用法：
    - 处理单文件：$\texttt{python data/jsonl\_to\_json.py path/to/file.jsonl}$
    - 递归遍历：$\texttt{python data/jsonl\_to\_json.py data/ --recursive --pattern "*.jsonl"}$

- $\texttt{\_\_init\_\_.py}$
  - 用途：打包入口与模块说明，便于以 $\texttt{from data import catalog\_lookup}$ 方式引用工具。

- $\texttt{\_\_pycache\_\_/}$
  - 用途：Python 字节码缓存目录（自动生成，已被忽略）。

---

## 维护与开发协议对接

- 当 $\texttt{data/}$ 目录新增、删除或重命名文件（含脚本与数据集）时，必须同步更新本清单，保持与 $\texttt{AGENTS.md}$ 的“数据目录维护规范”一致。
- 若 $\texttt{word\_length\_sets.json.union.lengths}$ 或示例语料的分隔符/结构发生变更，需同步更新顶层方案文档与本清单的“生成物与来源”部分，避免文档‑实现不一致。

