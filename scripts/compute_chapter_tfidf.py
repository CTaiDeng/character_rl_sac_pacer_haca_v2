"""预计算章节词频与 TF-IDF 缓存。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.lexical_stats import (
    ChapterLexicalStatistics,
    LexicalTokenizer,
    TokenizerUnavailableError,
    compute_chapter_statistics,
    load_stopwords,
)

ARTICLE_SEGMENT_SEPARATOR = "[----------------------------------------------------->"


def _split_chapters(text: str) -> Sequence[str]:
    if ARTICLE_SEGMENT_SEPARATOR in text:
        segments = text.split(ARTICLE_SEGMENT_SEPARATOR)
    else:
        segments = text.split("\n\n")
    return [segment.strip() for segment in segments if segment.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成章节 TF-IDF 缓存")
    parser.add_argument(
        "--article-path",
        default="data/sample_article.txt",
        type=Path,
        help="原始长文路径",
    )
    parser.add_argument(
        "--output",
        default="data/sample_article_lexical.json",
        type=Path,
        help="输出 JSON 路径",
    )
    parser.add_argument(
        "--stopwords",
        type=Path,
        default=None,
        help="停用词表路径，可选",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "jieba", "regex"),
        default="auto",
        help="分词后端优先级",
    )
    args = parser.parse_args()

    article_path: Path = args.article_path
    if not article_path.exists():
        raise FileNotFoundError(f"未找到文章文件：{article_path}")

    text = article_path.read_text(encoding="utf-8")
    chapters = _split_chapters(text)
    if not chapters:
        raise ValueError("未能从文章中解析出任何章节")

    stopwords = load_stopwords(args.stopwords)
    try:
        tokenizer = LexicalTokenizer(stopwords=stopwords, force_backend=args.backend if args.backend != "auto" else None)
    except TokenizerUnavailableError as exc:  # pragma: no cover - CLI 交互
        raise SystemExit(str(exc))

    stats: ChapterLexicalStatistics = compute_chapter_statistics(chapters, tokenizer)
    stats.save(args.output)

    print(
        "已生成章节词频缓存："
        f"章节数={stats.total_documents} 词汇量={len(stats.vocabulary)} 输出={args.output}"
    )


if __name__ == "__main__":
    main()
