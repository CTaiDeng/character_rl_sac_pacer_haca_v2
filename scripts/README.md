# scripts 目录使用手册

> 摘要：本手册汇总 `scripts/` 目录中的工具脚本，覆盖数据预处理（章节切分、TF‑IDF 统计与相似度评估）、中文实体抽取、PyTorch 环境安装，以及文档规范化与索引同步、README 协议迁移与翻译修补等常见工程任务。每个脚本均给出用途要点与一行示例命令，便于团队快速查阅、复用与自动化集成。

---

## 总览

- 适用范围：本手册仅针对 `scripts/` 下的可执行脚本；项目顶层协议与文档规范见 `AGENTS.md`。
- 编码与规范：Markdown 采用 UTF‑8（带 BOM）；行内代码以反引号标注（提交前会被规范化为数学打字体）。

---

## 脚本明细

- `check_docs_sync.py` — 文档一致性检查
  - 用途：校验顶层方案文档是否同步提及关键配置（如 `character_history_extension_limit`、`data/word_length_sets.json.union.lengths`）。
  - 用法：`python scripts/check_docs_sync.py`

- `compute_chapter_tfidf.py` — 章节级 TF‑IDF 统计
  - 用途：按段落分隔符切分文章，计算章节级 TF‑IDF/概率等词法统计并写入 JSON。
  - 用法：`python scripts/compute_chapter_tfidf.py --article-path data/sample_article.txt --output data/sample_article_lexical.json`

- `evaluate_lexical_reward.py` — 摘要 vs 章节 词法相似度
  - 用途：基于上一脚本的统计，对多个摘要计算 TF‑IDF 余弦与 Jensen‑Shannon 相似度，并打印 Top 词项。
  - 用法：`python scripts/evaluate_lexical_reward.py --stats data/sample_article_lexical.json --chapter-index 1 summaries out/summary1.txt out/summary2.txt`

- `extract_chinese_names.py` — 中文实体/术语抽取
  - 用途：使用 LTP NER 从样例文章中抽取人名/地名/机构名与 ASCII 术语，过滤后写入 `data/chinese_name_frequency_word.json`。
  - 用法：`python scripts/extract_chinese_names.py`

- `install_pytorch.sh` — 安装 CPU 版 PyTorch
  - 用途：升级 `pip`、`numpy` 并安装 CPU 版 PyTorch（可通过 `PYTHON` 环境变量指定解释器）。
  - 用法：`bash scripts/install_pytorch.sh`

- `md_normalize.py` — Markdown 规范化
  - 用途：
```
- 围栏外执行规范化：
    \[...\]→$$...$$，
    \(...\)→$...$，
    `→$；
- 并将 · 兼容替换为 $\cdot$；
- 读写 UTF‑8（BOM）；
```
  - 用法：`python scripts/md_normalize.py [README.md docs/a.md ...]`（缺省处理全仓 `.md`）

- `update_readme_index.py` — README 文末摘要索引
  - 用途：提取 `docs/*.md` 顶部摘要块，维护 `README.md` 文末“文档摘要索引”；缺失时在文首自动插入占位摘要；清理历史重复索引段。
  - 用法：`python scripts/update_readme_index.py`

- `move_dev_protocol.py` — README 协议统一指引
  - 用途：将 README 中的“开发协议”小节替换为指向 `AGENTS.md` 的统一指引，集中化协议维护。
  - 用法：`python scripts/move_dev_protocol.py`

- `patch_readme_translate.py` — README 英文化段修补
  - 用途：将 README 中特定英文说明段替换为中文，统一语言表达（保留代码块）。
  - 用法：`python scripts/patch_readme_translate.py`

- `fix_texttt.py`
```
 — 修复 $\texttt{$ texttt}$ 误写
  - 用途：将 README 中 $    exttt{...}$统一修正为 $\texttt{...}$ 写法，避免解析异常。
  - 用法：python scripts/fix_texttt.py
```



- `fix_top_p_readme.py` — 统一“Top‑p”为纯文本
  - 用途：将 README 中 `Top‑p/Top–p/Top‒p` 等非 ASCII 连字符统一为纯文字 `Top-p`，避免数学解析干扰。
  - 用法：`python scripts/fix_top_p_readme.py`

---

## 建议的本地工作流

1) 先运行 `install_pytorch.sh` 完成依赖安装；
2) 进行数据准备与章节词法统计：`compute_chapter_tfidf.py`；
3) 评估摘要质量与相似度：`evaluate_lexical_reward.py`；
4) 提交前执行：`md_normalize.py` 与 `update_readme_index.py`（预提交钩子已自动化）。

