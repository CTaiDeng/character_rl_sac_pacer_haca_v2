"""离线评估摘要与章节词频相似度。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from src.lexical_stats import (
    ChapterLexicalStatistics,
    LexicalTokenizer,
    TokenizerUnavailableError,
    cosine_similarity,
    jensen_shannon_similarity,
)


def _load_tokenizer(stats: ChapterLexicalStatistics) -> LexicalTokenizer:
    try:
        return LexicalTokenizer(
            stopwords=stats.stopwords,
            force_backend=stats.tokenizer_backend,
        )
    except TokenizerUnavailableError:
        print("警告：无法初始化缓存指定的分词后端，改用正则切分。")
        return LexicalTokenizer(stopwords=stats.stopwords, force_backend="regex")


def _format_top_items(items: Iterable[tuple[str, float]], limit: int = 5) -> str:
    pairs = list(items)
    if not pairs:
        return "<none>"
    top = pairs[:limit]
    return ", ".join(f"{token}:{score:.3f}" for token, score in top)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估摘要的词频奖励指标")
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("data/sample_article_lexical.json"),
        help="章节词频缓存 JSON 路径",
    )
    parser.add_argument(
        "--chapter-index",
        type=int,
        required=True,
        help="要对比的章节索引（从 1 开始）",
    )
    parser.add_argument(
        "summaries",
        type=Path,
        nargs="+",
        help="待评估摘要文件，可以提供多个",
    )
    args = parser.parse_args()

    if not args.stats.exists():
        raise FileNotFoundError(f"未找到词频缓存：{args.stats}")
    stats = ChapterLexicalStatistics.load(args.stats)
    tokenizer = _load_tokenizer(stats)
    chapter_entry = stats.chapter_by_index(args.chapter_index)

    print(
        f"章节 {args.chapter_index} | token_count={chapter_entry.token_count} "
        f"top_tokens={_format_top_items(sorted(chapter_entry.tfidf.items(), key=lambda kv: kv[1], reverse=True))}"
    )

    for summary_path in args.summaries:
        if not summary_path.exists():
            raise FileNotFoundError(f"未找到摘要文件：{summary_path}")
        summary_text = summary_path.read_text(encoding="utf-8")
        vector = stats.vectorize_text(summary_text, tokenizer)
        cosine = cosine_similarity(vector.tfidf, chapter_entry.tfidf)
        js = jensen_shannon_similarity(vector.probability, chapter_entry.probability)
        tfidf_sorted = sorted(vector.tfidf.items(), key=lambda kv: kv[1], reverse=True)
        prob_sorted = sorted(vector.probability.items(), key=lambda kv: kv[1], reverse=True)
        print("-" * 60)
        print(f"摘要：{summary_path}")
        print(
            f"  tokens={vector.token_count} cosine={cosine:.4f} js_similarity={js:.4f}"
        )
        print(
            "  top_tfidf="
            + _format_top_items(tfidf_sorted)
        )
        print(
            "  top_prob="
            + _format_top_items(prob_sorted)
        )


if __name__ == "__main__":
    main()
