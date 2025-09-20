"""Lexical statistics utilities for chapter-level TF-IDF analysis."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_STOPWORDS: Tuple[str, ...] = (
    "的",
    "了",
    "和",
    "是",
    "在",
    "有",
    "我",
    "也",
    "就",
    "都",
    "而",
    "及",
    "与",
    "着",
    "或",
    "一个",
    "一些",
    "我们",
    "你",
    "你们",
    "他们",
    "她们",
    "它们",
    "这",
    "那",
    "这些",
    "那些",
    "\n",
    "\r",
    "\t",
)


_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")


class TokenizerUnavailableError(RuntimeError):
    """Raised when a requested tokenizer backend cannot be initialized."""


class LexicalTokenizer:
    """Tokenize Chinese text with optional jieba support and stopword filtering."""

    def __init__(
        self,
        *,
        stopwords: Optional[Iterable[str]] = None,
        force_backend: Optional[str] = None,
    ) -> None:
        self.stopwords = set(stopwords) if stopwords is not None else set(DEFAULT_STOPWORDS)
        backend = force_backend or "auto"
        self._jieba = None
        self._backend = "regex"
        if backend in ("auto", "jieba"):
            try:
                import jieba  # type: ignore

                self._jieba = jieba
                self._backend = "jieba"
            except ImportError:
                if force_backend == "jieba":
                    raise TokenizerUnavailableError(
                        "无法加载 jieba，请先安装：python -m pip install jieba"
                    )
        if self._jieba is None:
            self._backend = "regex"

    @property
    def backend(self) -> str:
        return self._backend

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        if self._jieba is not None:
            raw_tokens = self._jieba.lcut(text, cut_all=False)
        else:
            raw_tokens = _TOKEN_PATTERN.findall(text)
        filtered: List[str] = []
        for token in raw_tokens:
            token = token.strip()
            if not token:
                continue
            if token in self.stopwords:
                continue
            if _is_punctuation(token):
                continue
            filtered.append(token)
        return filtered


def _is_punctuation(token: str) -> bool:
    if len(token) == 1:
        category = _char_category(token)
        return category.startswith("P")
    return False


def _char_category(char: str) -> str:
    import unicodedata

    return unicodedata.category(char)


@dataclass
class ChapterLexicalEntry:
    index: int
    token_count: int
    term_freq: Dict[str, int]
    probability: Dict[str, float]
    tfidf: Dict[str, float]


@dataclass
class ChapterLexicalStatistics:
    total_documents: int
    vocabulary: List[str]
    document_frequency: Dict[str, int]
    corpus_frequency: Dict[str, int]
    idf: Dict[str, float]
    corpus_token_count: int
    stopwords: List[str] = field(default_factory=list)
    tokenizer_backend: str = "regex"
    chapters: List[ChapterLexicalEntry] = field(default_factory=list)

    def to_json(self) -> Dict[str, object]:
        return {
            "total_documents": self.total_documents,
            "vocabulary": self.vocabulary,
            "document_frequency": self.document_frequency,
            "corpus_frequency": self.corpus_frequency,
            "idf": self.idf,
            "corpus_token_count": self.corpus_token_count,
            "stopwords": self.stopwords,
            "tokenizer_backend": self.tokenizer_backend,
            "chapters": [
                {
                    "index": entry.index,
                    "token_count": entry.token_count,
                    "term_freq": entry.term_freq,
                    "probability": entry.probability,
                    "tfidf": entry.tfidf,
                }
                for entry in self.chapters
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_json(), handle, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "ChapterLexicalStatistics":
        chapters_payload = payload.get("chapters", [])
        chapters: List[ChapterLexicalEntry] = []
        for entry in chapters_payload:
            data = dict(entry)  # type: ignore[arg-type]
            chapters.append(
                ChapterLexicalEntry(
                    index=int(data.get("index", 0)),
                    token_count=int(data.get("token_count", 0)),
                    term_freq={str(k): int(v) for k, v in dict(data.get("term_freq", {})).items()},
                    probability={str(k): float(v) for k, v in dict(data.get("probability", {})).items()},
                    tfidf={str(k): float(v) for k, v in dict(data.get("tfidf", {})).items()},
                )
            )
        return cls(
            total_documents=int(payload.get("total_documents", 0)),
            vocabulary=list(payload.get("vocabulary", [])),
            document_frequency={str(k): int(v) for k, v in dict(payload.get("document_frequency", {})).items()},
            corpus_frequency={str(k): int(v) for k, v in dict(payload.get("corpus_frequency", {})).items()},
            idf={str(k): float(v) for k, v in dict(payload.get("idf", {})).items()},
            corpus_token_count=int(payload.get("corpus_token_count", 0)),
            stopwords=list(payload.get("stopwords", [])),
            tokenizer_backend=str(payload.get("tokenizer_backend", "regex")),
            chapters=chapters,
        )

    @classmethod
    def load(cls, path: Path) -> "ChapterLexicalStatistics":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_json(data)

    def chapter_by_index(self, index: int) -> ChapterLexicalEntry:
        for entry in self.chapters:
            if entry.index == index:
                return entry
        raise KeyError(f"未找到章节索引 {index}")

    def vectorize_text(
        self,
        text: str,
        tokenizer: LexicalTokenizer,
        *,
        default_idf: Optional[float] = None,
    ) -> "VectorizedText":
        tokens = tokenizer.tokenize(text)
        counter = Counter(tokens)
        total = sum(counter.values())
        probabilities: Dict[str, float] = {}
        tfidf: Dict[str, float] = {}
        if total == 0:
            return VectorizedText(tokens=counter, probability=probabilities, tfidf=tfidf, token_count=0)
        default = default_idf if default_idf is not None else _default_idf(self.total_documents)
        for token, count in counter.items():
            probability = count / total
            probabilities[token] = probability
            weight = self.idf.get(token, default)
            tfidf[token] = probability * weight
        return VectorizedText(tokens=counter, probability=probabilities, tfidf=tfidf, token_count=total)


@dataclass
class VectorizedText:
    tokens: Mapping[str, int]
    probability: Mapping[str, float]
    tfidf: Mapping[str, float]
    token_count: int


def _default_idf(total_documents: int) -> float:
    return math.log((total_documents + 1) / 1) + 1


def compute_chapter_statistics(
    chapters: Sequence[str],
    tokenizer: LexicalTokenizer,
) -> ChapterLexicalStatistics:
    if not chapters:
        raise ValueError("章节列表不能为空")
    document_frequency: Counter[str] = Counter()
    corpus_frequency: Counter[str] = Counter()
    entries: List[ChapterLexicalEntry] = []
    vocabulary: set[str] = set()
    corpus_token_count = 0
    for index, text in enumerate(chapters, start=1):
        tokens = tokenizer.tokenize(text)
        counter = Counter(tokens)
        token_total = sum(counter.values())
        corpus_token_count += token_total
        vocabulary.update(counter.keys())
        document_frequency.update(counter.keys())
        corpus_frequency.update(counter)
        probability = {
            token: (count / token_total) if token_total else 0.0
            for token, count in counter.items()
        }
        # tf-idf will be filled later after IDF is known
        entries.append(
            ChapterLexicalEntry(
                index=index,
                token_count=token_total,
                term_freq=dict(counter),
                probability=probability,
                tfidf={},
            )
        )
    total_documents = len(chapters)
    idf: Dict[str, float] = {}
    for token in vocabulary:
        df = document_frequency[token]
        idf[token] = math.log((total_documents + 1) / (df + 1)) + 1
    for entry in entries:
        tfidf = {}
        for token, probability in entry.probability.items():
            tfidf[token] = probability * idf[token]
        entry.tfidf = tfidf
    return ChapterLexicalStatistics(
        total_documents=total_documents,
        vocabulary=sorted(vocabulary),
        document_frequency=dict(document_frequency),
        corpus_frequency=dict(corpus_frequency),
        idf=idf,
        corpus_token_count=corpus_token_count,
        stopwords=sorted(tokenizer.stopwords),
        tokenizer_backend=tokenizer.backend,
        chapters=entries,
    )


def cosine_similarity(vec_a: Mapping[str, float], vec_b: Mapping[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    numerator = 0.0
    for token, weight in vec_a.items():
        if token in vec_b:
            numerator += weight * vec_b[token]
    if numerator == 0.0:
        return 0.0
    norm_a = math.sqrt(sum(weight * weight for weight in vec_a.values()))
    norm_b = math.sqrt(sum(weight * weight for weight in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return numerator / (norm_a * norm_b)


def jensen_shannon_similarity(
    prob_a: Mapping[str, float],
    prob_b: Mapping[str, float],
    *,
    epsilon: float = 1e-12,
) -> float:
    if not prob_a or not prob_b:
        return 0.0
    all_tokens = set(prob_a.keys()) | set(prob_b.keys())
    kl_a = 0.0
    kl_b = 0.0
    for token in all_tokens:
        pa = max(prob_a.get(token, 0.0), epsilon)
        pb = max(prob_b.get(token, 0.0), epsilon)
        m = 0.5 * (pa + pb)
        kl_a += pa * math.log(pa / m)
        kl_b += pb * math.log(pb / m)
    js_divergence = 0.5 * (kl_a + kl_b)
    return max(0.0, 1.0 - js_divergence / math.log(2))


def load_stopwords(path: Optional[Path]) -> List[str]:
    if path is None or not path.exists():
        return list(DEFAULT_STOPWORDS)
    stopwords: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if token:
                stopwords.append(token)
    return stopwords


__all__ = [
    "ChapterLexicalEntry",
    "ChapterLexicalStatistics",
    "LexicalTokenizer",
    "TokenizerUnavailableError",
    "VectorizedText",
    "compute_chapter_statistics",
    "cosine_similarity",
    "jensen_shannon_similarity",
    "load_stopwords",
]
