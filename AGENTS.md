# AGENTS 指南（仓库内优先级指令｜唯一规范来源）

> 作用域：仓库根目录（含全部子目录）；本文件为代理在仓内工作的首要指引与规范汇总。

- 对话语言：所有协作一律使用“简体中文”。
- 规范优先：本文件为唯一规范来源；“开发协议与规范”章节已并入（原 PROJECT_DOCS_GUIDE.md 已合并并删除）。
- 工程文档位置：工程类文档统一放在 `engineering_docs/` 中管理：
  - 当前版本：`engineering_docs/engineering_docs_current/`
  - 历史归档：`engineering_docs/engineering_docs_archive/`
- README 摘要索引：根 `README.md` 文末的“docs 摘要索引”仅索引 `docs/*.md`；不覆盖 `engineering_docs/*`。
- Markdown 约定：遵循“开发协议与规范”的数学/代码规范；行间数学用 `$$ … $$`，行内用 `$ … $`，围栏代码块内不做转换。
- 同步要求：当以下方案变更时，需同步更新 `engineering_docs/engineering_docs_current/` 对应工程文档，并在必要时复制至归档：
  - `STEP_SCORING.md`（Step 打分方案）
  - `INPUT_OUTPUT_SCHEME.md`（输入-输出数据方案）
  - `NETWORK_TOPOLOGY.md`（网络拓扑结构方案）

**关键规范摘要**
- 开发协议：三份方案文档需使用简体中文，配数学语言与伪代码；发生变更时必须同步更新。
- 文档摘要同步：`docs/*.md` 顶部维护“摘要”块；`README.md` 文末维护统一索引；任一侧调整需双向同步。
- 对齐流程：新增/重命名/迁移 `docs` 后执行 `python scripts/align_docs.py` 重写时间戳前缀、更新日期行、重建 README 索引并规范化 Markdown。
- Markdown 规范：行间 `$$ … $$`、行内 `$ … $`；围栏代码块不转换；行内代码使用反引号。
- 编码与行尾：统一 UTF-8（带 BOM）+ LF。

**常用指令**
- 对齐脚本：`python scripts/align_docs.py`
- 预提交钩子：`git config core.hooksPath .githooks`（提交前自动规范化已暂存的 `.md` 并同步 README 索引）

---

# 开发协议与规范（原 PROJECT_DOCS_GUIDE.md 内容）

> 说明：本节为原 `PROJECT_DOCS_GUIDE.md` 的完整内容，现已合并至 AGENTS.md 并作为唯一来源维护。

> 摘要：本文定义本项目的统一开发协议、文档规范、摘要同步机制、对齐流程、Markdown 数学/代码格式化标准、演示与环境约定、临时文件清理、数据目录维护、工程文档归档策略、扩展名约定与工程文档同步维护要求。工程类设计文档集中放置于 `engineering_docs/`：当前版本位于 `engineering_docs/engineering_docs_current/`，历史归档位于 `engineering_docs/engineering_docs_archive/`。当 Step 打分方案、输入-输出方案或网络拓扑方案发生调整时，必须同步更新对应工程文档，并按需归档。README 文末索引仅覆盖 `docs/*.md`，与工程文档互不干扰。

## 开发协议

对应工程文档（当前版本）清单：

- `engineering_docs/engineering_docs_current/STEP_SCORING.md`
- `engineering_docs/engineering_docs_current/INPUT_OUTPUT_SCHEME.md`
- `engineering_docs/engineering_docs_current/NETWORK_TOPOLOGY.md`
- `engineering_docs/engineering_docs_current/ITERATIVE_SUMMARY_TEMPLATE.md`
- `engineering_docs/engineering_docs_current/ITERATION_GRANULARITY_DESIGN.md`
- `engineering_docs/engineering_docs_current/DATA_TOOLS.md`

## 文档摘要同步规范

- 在 `README.md` 文末追加 `docs` 目录下每篇文章的“摘要”索引。
- 每篇 `docs/*.md` 顶部需维护“摘要”块，格式建议：在标题后插入一段 `> 摘要：...`；或使用 `<!-- SUMMARY-START --> ... <!-- SUMMARY-END -->` 注释包裹。
- 摘要长度要求：尽量完整，建议 120–300 字（约 3–6 句）；可包含 3–5 条要点列表补充关键信息。
- 任何一侧（`docs` 内文或 `README.md` 索引）摘要发生调整时，必须同步更新另一侧，保持一致。
- 新增或重命名文档时，应同时在 `README.md` 的摘要索引中增删对应条目。
- 禁止维护 `docs/SUMMARIES.md` 等独立摘要清单，避免三方不同步。
- 文档编码统一为 UTF-8（带 BOM），并统一写回 LF 行尾（跨平台一致，避免 CRLF 警告）。

## 文档对齐指令与整理流程（docs/*）

- 目标：将 `docs` 知识库的文章统一成“文件名前缀为入库时间戳（秒）”，并在文档主标题下方写入对应日期（YYYY‑MM‑DD）。
- 时间戳前缀重写：以 git 首次入库时间为准（`git log --diff-filter=A --follow --format=%at -n 1 <file>`），仅处理匹配 `^\d+_.*\.md$` 的文件。
- 日期写入规则：
  - 位置：文档首个标题（以 `#` 开头）之后一行；
  - 形式：`日期：YYYY-MM-DD`；
  - 若该位置已有“日期：”行，则就地更新。
- 对齐指令入口：`python scripts/align_docs.py`
  - 依次执行：
    1) `scripts/rename_docs_to_git_ts.py`（重写时间戳前缀）；
    2) `scripts/insert_doc_date_from_prefix.py`（写入/更新日期行）；
    3) `scripts/update_readme_index.py`（重建 README 文末索引）；
    4) `scripts/md_normalize.py`（对 README 与 docs 做 Markdown 规范化）。
- 触发时机：新增/重命名文档后，或批量文档迁移后，执行一次对齐指令；PR 合并前建议手动执行并提交。

## Markdown 规范（数学/代码格式实时审查）

```
- 数学分隔符统一：
  - 行间公式使用 $$ … $$（将遗留的 \[ … \] 统一转换为 $$ … $$）。
  - 行内公式使用 $ … $（将遗留的 \( … \) 统一转换为 $ … $）。
- 行内代码：保持反引号表示（例如 `a_b`），不转换为数学字体。
- 代码块保护：三反引号/三波浪线围栏代码块内部（``` 或 ~~~）不做上述转换。
- 维护要求：当任一 Markdown 文档改动时，需同步执行上述规范转换并自查（或在提交评审中检查）以保证全仓库风格一致。
```

### 落实方式

- 工具脚本：`python scripts/md_normalize.py [<files...>]`
  - 缺省遍历仓库内全部 `.md`，支持传入改动文件列表，仅对围栏外文本做以下转换：
    1) `$$ ... $$` → `$$ ... $$`（忽略以双反斜杠开头的 `\\[2pt]` 等行内可选参数）；
    2) `$ ... $` → `$ ... $`；
    3) `` `inline_code` `` → ``inline_code``。
  - 读写编码使用 UTF-8（带 BOM），并统一写回 LF 行尾（跨平台一致，避免 CRLF 警告）。
- 提交钩子：`.githooks/pre-commit` 会在提交前自动：
  - 仅对已暂存的 `.md` 执行规范化；
  - 运行 `scripts/update_readme_index.py` 同步 `README.md` 文末的 docs 摘要索引；
  - 自动 `git add` 相关文件。
- 启用钩子：`git config core.hooksPath .githooks`

## 演示与环境约定（Development Protocol）

- 演示脚本将策略网络视作微型 LLM 头部，直接读取“上一轮摘要 + 当前章节全文”的拼接文本并生成新的摘要。
- 数据样例文件 `data/sample_article.txt` 使用 "[----------------------------------------------------->" 作为段落分割符号，模拟教师模型输出的分段提示。
- 训练过程中对每个分割执行迭代摘要：第 1 个摘要默认为空字符串，将其与第 1 个分割（两个分隔符之间的内容）拼接后得到第 1 次输出；随后把该摘要与第 2 个分割组合生成第 2 次输出，如此迭代，模拟蒸馏时“上一次摘要 + 间隔内容 → 新摘要”的累积推理轨迹。环境不会裁剪策略给出的文本，奖励函数依据章节覆盖率、语义相似度与文本新颖度综合打分。
- 环境准备：开发前请先在当前环境中安装 `numpy` 与 `pytorch`（可直接运行 `scripts/install_pytorch.sh`，该脚本会顺带安装 `numpy`）。

## 临时文件与清理规范

- 禁止将临时/对比用产物纳入版本库，提交前需清理：
  - `out_prev_readme.txt`（对比 README 旧版本时产生的临时文件）。
  - `out_block.txt`（导出 README 摘要索引块用于排查时产生的临时文件）。
  - `last_gen_msg.txt`（生成提交信息时的调试/留存输出临时文件）。
  - `TMP_COMMIT_MSG.txt`（模拟 prepare-commit-msg 钩子时的临时提交信息文件）。
- `.gitignore` 已忽略上述文件；预提交钩子会在提交前自动删除该文件。

## 数据目录维护规范

- 在 `data/` 目录新增、删除、重命名文件（含脚本与数据集）时，必须同步更新 `data/README.md` 中的清单、分类与用途说明。
- 若 `data/word_length_sets.json.union.lengths` 或示例语料的结构/分隔符发生变更，需同步更新相关设计文档与本清单的“生成物与来源”说明。
- 涉及 CLI 的脚本（如 `catalog_lookup.py`、`gen_word_length_sets.py`、`jsonl_to_json.py`）应在 `data/README.md` 中给出最小可用示例命令。

## 工程文档目录与存档（engineering_docs）

- 当前版本：`engineering_docs/engineering_docs_current/` 存放 5 篇工程文档的最新版本。
- 历史归档：`engineering_docs/engineering_docs_archive/` 用于存放项目文档的历史版本快照，仅作为归档与检索使用；不做自动维护。
- 命名约定：归档文件以原名加版本尾缀 `_vMAJOR.MINOR.PATCH`，示例：`STEP_SCORING_v1.0.0.md`。
- README 摘要索引：仅覆盖 `docs/*.md`，不包含 `engineering_docs/*`。
- 操作建议：当工程文档发布新版本时，请手动复制到归档目录并按上述命名规范命名。

## Markdown 文件扩展名约定

- 统一使用小写 `.md` 作为 Markdown 扩展名。
- 禁止使用 `.MD`、`.Md`、`.MkD` 等大小写或变体；存量文件需更名为 `.md`。
- 工具链（预提交钩子与脚本）默认仅处理 `.md` 小写扩展名文件；大写扩展名不会参与自动规范化与索引。
- 新增/迁移文档时，请自查扩展名是否符合本规范。

## 工程文档（engineering_docs）同步维护规范

- 范围：`engineering_docs/engineering_docs_current/` 下的 5 篇主文档与本说明。
- 同步要求：上述任一文档发生方案/术语/接口/参数/流程变化时，需同步更新其他受影响文档与本节（开发协议与规范）的对应说明条目。
- 版本与归档：采用语义化版本；发布版本时，将 5 篇主文档复制到 `engineering_docs/engineering_docs_archive/`（命名 `_vMAJOR.MINOR.PATCH.md`）。归档目录不做自动维护。
- README 摘要索引：仅覆盖 `docs/*.md`；工程文档与归档不纳入该索引。
- 提交信息约定：工程文档更新建议使用前缀 `docs:`，发布版本建议包含版本号，如 `docs: 发布工程文档 v1.0.1 并归档`。

---

系统提示：本仓库与代理交互统一使用“简体中文”。

## 自动注释注入规则（docs/*）
- 触发条件：当 docs/*.md 正文任意位置出现以下关键词之一时，自动在“日期：YYYY-MM-DD”下一行插入统一注释：
  - O3理论、O3元数学理论、主纤维丛版广义非交换李代数、PFB-GNLA
- 注释内容：
  #### ***注：“O3理论/O3元数学理论/主纤维丛版广义非交换李代数(PFB-GNLA)”相关理论参见： [作者（GaoZheng）网盘分享](https://drive.google.com/drive/folders/1lrgVtvhEq8cNal0Aa0AjeCNQaRA8WERu?usp=sharing) 或 [作者（GaoZheng）主页](https://mymetamathematics.blogspot.com)***
- 执行位置：由 scripts/align_docs.py 调用 scripts/insert_o3_citation_note.py 在日期行下方插入；已存在同样注释时不重复插入；若缺少日期行则退化为插入在首个标题后一行。

## 临时文件与清理规范（补充）
- out_align_docs_prev.txt：对齐脚本运行时可能生成的临时对比文件；已加入 .gitignore 忽略，并在预提交钩子 .githooks/pre-commit 中自动删除，避免误提交。

## 知识库范围与只读目录（新增）
- 知识库定义：仓库 `docs/` 目录整体视为“项目知识库”，其中包含只读引用子目录 `docs/kernel_reference/`。
- 只读引用目录：`docs/kernel_reference/` 指向外部源，属于知识库的只读引用，不纳入本项目的维护、索引与任何自动修改；不得写入或改动其内容。
- 索引范围：`README.md` 文末“docs 摘要索引”不包含 `docs/kernel_reference/`。
- 脚本约束：所有递归遍历/修改 Markdown 的脚本必须跳过该目录并视为只读。统一通过 `scripts/docs_processing_config.json` 的 `skip_paths` 管理（默认已包含 `docs/kernel_reference/`）。对齐流程中涉及的 `scripts/md_normalize.py`、`scripts/convert_texttt_to_backticks.py`、`scripts/insert_o3_citation_note.py` 等脚本遵循该白名单；而仅处理 `docs/*.md` 顶层文件的 `scripts/rename_docs_to_git_ts.py`、`scripts/insert_doc_date_from_prefix.py`、`scripts/update_readme_index.py` 本身不会进入该目录。
- AI 助手引用范围：在与 Codex 或其他 AI 助手协作时，如未明确限定“知识库”范围，应默认将 `docs/` 及其所有递归子目录的文档一并纳入参考范围（包含只读目录 `docs/kernel_reference/`）；仅编辑、索引与自动修改保持对 `docs/kernel_reference/` 的只读约束不变。
