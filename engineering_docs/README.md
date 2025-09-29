# 工程文档目录（engineering_docs）

> 摘要：本目录统一管理工程类设计文档的“当前版本”与“历史归档”。当前有效版本位于 `engineering_docs_current/`，历史发布版本位于 `engineering_docs/engineering_docs_archive/`。当打分方案、I/O 方案或网络拓扑等发生变更时，请先在 `engineering_docs_current/` 更新对应文档，并在发布版本时将旧版复制到归档目录，保持可追溯性与一致性。

## 当前与归档结构
- 当前工程文档：`engineering_docs/engineering_docs_current/`
  - `STEP_SCORING.md`（Step 打分方案）
  - `INPUT_OUTPUT_SCHEME.md`（输入-输出数据方案）
  - `NETWORK_TOPOLOGY.md`（网络拓扑方案）
  - `ITERATIVE_SUMMARY_TEMPLATE.md`（迭代摘要模板）
  - `ITERATION_GRANULARITY_DESIGN.md`（迭代粒度设计）
  - `DATA_TOOLS.md`（数据工具与脚本说明）
- 历史归档：`engineering_docs/engineering_docs_archive/`（手动复制 `_vMAJOR.MINOR.PATCH.md` 版本文件）

## 维护与发布流程
- 修改方案：在 `engineering_docs/engineering_docs_current/` 中更新相应文档内容。
- 同步说明：若涉及术语/接口/流程变更，需同步更新 `AGENTS.md`（开发协议与规范）与其他受影响的工程文档。
- 版本发布：将当前文档复制到 `engineering_docs/engineering_docs_archive/`，并以原名加版本后缀 `_vMAJOR.MINOR.PATCH.md` 命名。
- 提交信息：建议使用前缀 `docs:`，并包含版本号，例如 `docs: 发布工程文档 v1.1.0 并归档`。

## 规范与边界
- 编码与行尾：统一使用 UTF-8（带 BOM），LF 行尾。
- 文件扩展名：统一使用小写 `.md`。
- README 摘要索引：根目录 `README.md` 的“docs 摘要索引”仅覆盖 `docs/*.md`；本目录不纳入该索引。

## 历史归档示例
- `engineering_docs/engineering_docs_archive/DATA_TOOLS_v1.0.0.md`（数据工具与脚本说明，v1.0.0 归档）
