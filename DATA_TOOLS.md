#+ 数据工具与映射说明

- 输入-输出-打分映射（JSON）
  - 文件：$\texttt{data/io\_score\_mapping.json}$
  - 作用：给出最小映射 schema（input/output/score）与示例，便于脚本/服务统一记录样例。

- 词长集合生成器
  - 文件：$\texttt{data/gen\_word\_length\_sets.py}$
  - 用法：$\texttt{python -m data.gen\_word\_length\_sets}$
  - 输出：$\texttt{data/word\_length\_sets.json}$（names/freq/union 三块，供可变长度后缀命中使用）。

- 词表命中查询模块
  - 文件：$\texttt{data/catalog\_lookup.py}$（可 $\texttt{from data import catalog\_lookup}$）
  - 接口：$\texttt{load\_catalog()}$、$\texttt{annotate(term)}$、$\texttt{longest\_prefix\_hit(text,lengths)}$、$\texttt{suffix\_hit(text,lengths)}$
  - CLI 示例：
    - 标注：$\texttt{python -m data.catalog\_lookup --query "精妙"}$
    - 前缀：$\texttt{python -m data.catalog\_lookup --prefix "精妙。如" --lengths 2,3,4}$
    - 后缀：$\texttt{python -m data.catalog\_lookup --suffix "”他喃喃" --lengths 2,3,4}$

- JSONLINE 转结构化 JSON
  - 文件：$\texttt{data/jsonl\_to\_json.py}$
  - 作用：将若干 $\texttt{*.jsonl}$ 文件（逐行 JSON）合并为结构化 JSON 数组并美化缩进（UTF-8）。
  - 用法：
    - 转单个文件：$\texttt{python -m data.jsonl\_to\_json out/train\_123/teacher\_trajectory.jsonl}$
    - 批量（递归）：$\texttt{python -m data.jsonl\_to\_json out --recursive}$
    - 覆盖已有输出：$\texttt{python -m data.jsonl\_to\_json out --recursive --force}$
    - 控制缩进/ASCII：$\texttt{python -m data.jsonl\_to\_json out --indent 2 --ascii}$

> 说明：
> - $\texttt{union.lengths}$ 被 $\texttt{src/character\_sac\_trainer.py}$ 用作 raw_action/bigram 的可变长度后缀命中集合；
> - $\texttt{catalog\_lookup.annotate}$ 统一了“命中/未命中#编号”的注记显示；
> - 运行字符模式时，若 $\texttt{action\_source=teacher}$，训练目录 $\texttt{out/train\_*/teacher\_samples.jsonl}$ 会追加监督样本（输入→输出），输入为 $\texttt{prev/<sep>/chapter}$，输出为“下一字符”。
