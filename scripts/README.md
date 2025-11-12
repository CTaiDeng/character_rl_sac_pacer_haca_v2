# scripts 目录使用手册

> 摘要：本文对仓库 `scripts/` 目录的脚本做统一说明与对齐，涵盖 docs 对齐与规范化、README 摘要索引维护、Markdown 清理与转换、文本统计与演示辅助、提交信息生成与环境安装等。每个脚本提供用途与最小可用示例命令，便于团队快速查阅与复用。

---

## 重要：check_gpl_and_arms_length.py（许可证检查与“臂长”边界）

- 作用：
  - 全仓默认检查未被 `.gitignore` 忽略的全部 Python 文件（.py），统计顶层导入并基于本机包元数据识别第三方分发包的许可证，显式标注 GPL/AGPL/LGPL 家族依赖；
  - 可选在 `--scope src` 下启用“臂长通信”规则：禁止 `src/` 以库方式直接 `import data/scripts/tests`，并检测针对这些目录的 `sys.path` 注入。
- 检查范围：
  - 默认 `--scope repo`（全仓），使用 `git ls-files` 与 `git ls-files -o --exclude-standard`，覆盖“已跟踪 + 未被忽略的未跟踪” `.py`，严格尊重 `.gitignore`；
  - `--scope src` 时仅扫描 `src/`。
- 退出码（用于 CI）：
  - 正常：`0`；
  - 发现 GPL 家族依赖且启用 `--fail-on-gpl`：`2`；
  - 发现臂长违规且启用 `--fail-on-armlength`（仅 `--scope src` 有效）：`3`；
  - 同时触发时保留最严重的非零码（以脚本实现为准）。
- 最小用法：
  - 全仓文本报告：`python3 check_gpl_and_arms_length.py`
  - JSON 输出：`python3 check_gpl_and_arms_length.py --json`
  - 仅源码并启用臂长规则：`python3 check_gpl_and_arms_length.py --scope src --fail-on-armlength`
  - CI 严格模式：`python3 check_gpl_and_arms_length.py --json --fail-on-gpl`
- 维护约定：该段说明与脚本实现保持同步维护；当脚本的 CLI 选项、默认范围、退出码或输出结构调整时，必须在本节同步更新。

---

## 快速开始
- 运行全量对齐流程：`python scripts/align_docs.py`
- 启用预提交钩子：`git config core.hooksPath .githooks`
- Markdown 规范化（全仓）：`python scripts/md_normalize.py`

---

## 文档对齐与索引
- `align_docs.py`：一键对齐 docs 与 README 索引
  - 顺序执行：`rename_docs_to_git_ts.py`、`insert_doc_date_from_prefix.py`、`insert_o3_citation_note.py`、`update_readme_index.py`、`fix_readme_index_style.py`、`convert_texttt_to_backticks.py`、`md_normalize.py`
  - 用法：`python scripts/align_docs.py`

- `rename_docs_to_git_ts.py`：重写 docs 文件名前缀为 Git 首次入库时间戳（秒）
  - 仅处理 `^\d+_.*\.md$` 文件；冲突时自动追加后缀
  - 用法：`python scripts/rename_docs_to_git_ts.py`

- `insert_doc_date_from_prefix.py`：将“日期：YYYY-MM-DD”插入到文档首个标题下一行
  - 日期来自文件名时间戳前缀；已存在则就地更新
  - 用法：`python scripts/insert_doc_date_from_prefix.py`

- `insert_o3_citation_note.py`：当文中出现“O3理论/O3元数学理论/主纤维丛版广义非交换李代数(PFB-GNLA)”关键字时，在日期行下自动插入统一注释
  - 跳过只读目录见“配置”一节；重复注释不再插入
  - 用法：`python scripts/insert_o3_citation_note.py [docs]`

- `update_readme_index.py`：重建根 `README.md` 文末的 docs 摘要索引
  - 优先读取 `<!-- SUMMARY-START -->...<!-- SUMMARY-END -->`；否则使用 `> 摘要：...` 段；缺失时为文档插入占位摘要
  - 用法：`python scripts/update_readme_index.py`

- `fix_readme_index_style.py`：将索引块内的 `docs/...` 路径统一为反引号行内代码风格
  - 用法：`python scripts/fix_readme_index_style.py`

- `ensure_title_equals_filename.py`：保证 docs 每篇文档的首个 H1 与文件名（去掉时间戳与扩展名）一致
  - 用法：`python scripts/ensure_title_equals_filename.py [<files_or_dirs>...]`

- `move_dev_protocol.py`：将 README 中“开发协议/Development Protocol”小节替换为指向 `AGENTS.md` 的统一说明
  - 用法：`python scripts/move_dev_protocol.py`

---

## Markdown 清理与规范
- `md_normalize.py`：围栏外数学/代码风格规范化
  - 转换：将 `$$...$$` 统一为 `$$...$$`，将 `$...$` 统一为 `$...$`；不改动代码围栏内部；`\cdot` 规范为 `\cdot`
  - 用法：`python scripts/md_normalize.py [README.md docs/a.md ...]`（缺省遍历全仓 `.md`）

- `convert_texttt_to_backticks.py`：将围栏外 `` ... `` 统一替换为反引号 `...`
  - 用法：`python scripts/convert_texttt_to_backticks.py [<files...>]`

- `fix_texttt.py`：修复 README 中 `$ texttt{...}` 等异常写法为标准 `$\texttt{...}`
  - 用法：`python scripts/fix_texttt.py`

- `fix_top_p_readme.py`：修正 README 中 `Top?p/Top‑p` 等连字符变体为标准 `Top-p`
  - 用法：`python scripts/fix_top_p_readme.py`

- `patch_readme_translate.py`：将 README 中若干英文固定句替换为中文版
  - 用法：`python scripts/patch_readme_translate.py`

---

## 文本统计与演示
- `compute_chapter_tfidf.py`：按章节计算 TF‑IDF/概率分布并保存 JSON
  - 用法：`python scripts/compute_chapter_tfidf.py --article-path data/sample_article.txt --output data/sample_article_lexical.json`

- `evaluate_lexical_reward.py`：在给定章节上评估多个摘要的词汇相似度（余弦/Jensen‑Shannon）
  - 用法：`python scripts/evaluate_lexical_reward.py --stats data/sample_article_lexical.json --chapter-index 1 summaries out/summary1.txt out/summary2.txt`

- `extract_chinese_names.py`：基于 LTP 的人名/地名/机构名抽取与 ASCII 词统计
  - 依赖：`pip install ltp`；输出写入 `data/chinese_name_frequency_word.json`
  - 用法：`python scripts/extract_chinese_names.py`

---

## 提交信息与其他
- `gen_commit_msg_googleai.py`：用 Gemini 读取已暂存 diff 生成提交信息（无凭据时回退到本地摘要）。支持通过 `scripts/docs_processing_config.json` 的 `commit_msg_include_prefixes`/`commit_msg_exclude_prefixes` 过滤路径；未配置排除清单时默认复用 `skip_paths`（如排除 `docs/kernel_reference/`）。
  - 环境：设置 `GEMINI_API_KEY` 与 `GOOGLE_API_KEY`；可选 `GEMINI_MODEL`（默认 `gemini-1.5-flash`）
  - 用法：`python scripts/gen_commit_msg_googleai.py`

- `install_pytorch.sh`：安装 CPU 版 PyTorch（附带安装 numpy）
  - 用法：`bash scripts/install_pytorch.sh`（可用环境变量 `PYTHON` 指定解释器）

- `check_docs_sync.py`：工程文档与数据项的一致性自检
  - 用法：`python scripts/check_docs_sync.py`

- 其他辅助脚本：`commit_filters.py`、`fix_pip_binding.ps1`

---

## 配置
- 路径跳过白名单：`scripts/docs_processing_config.json`
  - 字段 `skip_paths` 列表中的目录在递归处理 Markdown 时会被跳过（默认包含 `docs/kernel_reference/` 只读目录）
- 内部模块：`scripts/_docs_config.py`（加载上述配置并提供 `is_under()` 路径判断）

---

## 约定
- 编码与行尾：统一 UTF-8（带 BOM）+ LF
- 自动化：预提交钩子会规范化已暂存的 `.md` 并更新 README 摘要索引；详见 `AGENTS.md`
