# scripts 目录使用手册

> 摘要：本手册汇总 $\texttt{scripts/}$ 目录中的工具脚本，覆盖数据预处理（章节切分、TF‑IDF 统计与相似度评估）、中文实体抽取、PyTorch 环境安装，以及文档规范化与索引同步、README 协议迁移与翻译修补等常见工程任务。每个脚本均给出用途要点与一行示例命令，便于团队快速查阅、复用与自动化集成。

---

## 总览

- 适用范围：本手册仅针对 $\texttt{scripts/}$ 下的可执行脚本；项目顶层协议与文档规范见 $\texttt{AGENTS.md}$。
- 编码与规范：Markdown 采用 UTF‑8（带 BOM）；行内代码以反引号标注（提交前会被规范化为数学打字体）。

---

## 脚本明细

- $\texttt{check\_docs\_sync.py}$ — 文档一致性检查
  - 用途：校验顶层方案文档是否同步提及关键配置（如 $\texttt{character\_history\_extension\_limit}$、$\texttt{data/word\_length\_sets.json.union.lengths}$）。
  - 用法：$\texttt{python scripts/check\_docs\_sync.py}$

- $\texttt{compute\_chapter\_tfidf.py}$ — 章节级 TF‑IDF 统计
  - 用途：按段落分隔符切分文章，计算章节级 TF‑IDF/概率等词法统计并写入 JSON。
  - 用法：$\texttt{python scripts/compute\_chapter\_tfidf.py --article-path data/sample\_article.txt --output data/sample\_article\_lexical.json}$

- $\texttt{evaluate\_lexical\_reward.py}$ — 摘要 vs 章节 词法相似度
  - 用途：基于上一脚本的统计，对多个摘要计算 TF‑IDF 余弦与 Jensen‑Shannon 相似度，并打印 Top 词项。
  - 用法：$\texttt{python scripts/evaluate\_lexical\_reward.py --stats data/sample\_article\_lexical.json --chapter-index 1 summaries out/summary1.txt out/summary2.txt}$

- $\texttt{extract\_chinese\_names.py}$ — 中文实体/术语抽取
  - 用途：使用 LTP NER 从样例文章中抽取人名/地名/机构名与 ASCII 术语，过滤后写入 $\texttt{data/chinese\_name\_frequency\_word.json}$。
  - 用法：$\texttt{python scripts/extract\_chinese\_names.py}$

- $\texttt{install\_pytorch.sh}$ — 安装 CPU 版 PyTorch
  - 用途：升级 $\texttt{pip}$、$\texttt{numpy}$ 并安装 CPU 版 PyTorch（可通过 $\texttt{PYTHON}$ 环境变量指定解释器）。
  - 用法：$\texttt{bash scripts/install\_pytorch.sh}$

- $\texttt{md\_normalize.py}$ — Markdown 规范化
  - 用途：
```
- 围栏外执行规范化：
    \[...\]→$$...$$，
    \(...\)→$...$，
    `→$；
- 并将 · 兼容替换为 $\cdot$；
- 读写 UTF‑8（BOM）；
```
  - 用法：$\texttt{python scripts/md\_normalize.py [README.md docs/a.md ...]}$（缺省处理全仓 $\texttt{.md}$）

- $\texttt{update\_readme\_index.py}$ — README 文末摘要索引
  - 用途：提取 $\texttt{docs/*.md}$ 顶部摘要块，维护 $\texttt{README.md}$ 文末“文档摘要索引”；缺失时在文首自动插入占位摘要；清理历史重复索引段。
  - 用法：$\texttt{python scripts/update\_readme\_index.py}$

- $\texttt{move\_dev\_protocol.py}$ — README 协议统一指引
  - 用途：将 README 中的“开发协议”小节替换为指向 $\texttt{AGENTS.md}$ 的统一指引，集中化协议维护。
  - 用法：$\texttt{python scripts/move\_dev\_protocol.py}$

- $\texttt{patch\_readme\_translate.py}$ — README 英文化段修补
  - 用途：将 README 中特定英文说明段替换为中文，统一语言表达（保留代码块）。
  - 用法：$\texttt{python scripts/patch\_readme\_translate.py}$

- $\texttt{fix\_texttt.py}$
```
 — 修复 $\texttt{$ texttt}$ 误写
  - 用途：将 README 中 $    exttt{...}$统一修正为 $\texttt{...}$ 写法，避免解析异常。
  - 用法：python scripts/fix_texttt.py
```



- $\texttt{fix\_top\_p\_readme.py}$ — 统一“Top‑p”为纯文本
  - 用途：将 README 中 $\texttt{Top‑p/Top–p/Top‒p}$ 等非 ASCII 连字符统一为纯文字 $\texttt{Top-p}$，避免数学解析干扰。
  - 用法：$\texttt{python scripts/fix\_top\_p\_readme.py}$

---

## 建议的本地工作流

1) 先运行 $\texttt{install\_pytorch.sh}$ 完成依赖安装；
2) 进行数据准备与章节词法统计：$\texttt{compute\_chapter\_tfidf.py}$；
3) 评估摘要质量与相似度：$\texttt{evaluate\_lexical\_reward.py}$；
4) 提交前执行：$\texttt{md\_normalize.py}$ 与 $\texttt{update\_readme\_index.py}$（预提交钩子已自动化）。

