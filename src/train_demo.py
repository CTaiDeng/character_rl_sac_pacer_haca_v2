"""Run a text-conditioned SAC distillation demo for iterative summarization.

The updated demonstration treats each chapter-length iteration as a single
step whose observation consists of the full previous summary concatenated with
the chapter text. A character-level policy network produces summary text
directly, and the environment evaluates the resulting prose without applying
length-based truncation.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import itertools
import json
import math
import random
import re
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import unicodedata

try:  # pragma: no cover - import guard exercised in tests via fallback path
    import torch
    from torch import nn
    from torch.distributions import Categorical
    from torch.nn import functional as F
    from torch.nn.utils.rnn import pack_padded_sequence

    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised in CI without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Categorical = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    pack_padded_sequence = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

try:  # pragma: no cover - exercised depending on invocation style
    from .lexical_stats import (
        ChapterLexicalStatistics,
        LexicalTokenizer,
        TokenizerUnavailableError,
        cosine_similarity,
        jensen_shannon_similarity,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from lexical_stats import (  # type: ignore[no-redef]
        ChapterLexicalStatistics,
        LexicalTokenizer,
        TokenizerUnavailableError,
        cosine_similarity,
        jensen_shannon_similarity,
    )

SRC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SRC_ROOT.parent
OUT_DIR = REPO_ROOT / "out"
COMPUTE_TFIDF_SCRIPT = REPO_ROOT / "scripts" / "compute_chapter_tfidf.py"
STEP_CSV_PATH = OUT_DIR / "step_metrics.csv"
ROUND_CSV_PATH = OUT_DIR / "round_metrics.csv"
REWARDS_HTML_PATH = OUT_DIR / "rewards.html"
ROUND_SNAPSHOT_DIR = OUT_DIR / "round_snapshots"
STEP_LOG_PATH = OUT_DIR / "training_step.log"
TRAIN_LOG_PATH = OUT_DIR / "training_output.log"
CONFIG_TEMPLATE_PATH = REPO_ROOT / "config_template.json"
CONFIG_OVERRIDE_PATH = REPO_ROOT / "res" / "config.json"
DEFAULT_REFERENCE_ACTIONS_PATH = "data/chapter_iterative_io_examples.txt"
DEFAULT_REFERENCE_WARMUP_ROUNDS = 0
DEFAULT_REFERENCE_WARMUP_STEPS = 5

STEP_CSV_HEADERS = [
    "round",
    "step",
    "global_step",
    "reward",
    "reward_base",
    "reward_potential_gain",
    "reward_soft_bonus",
    "previous_summary_length",
    "chapter_length",
    "source_length",
    "summary_length",
    "length_ratio",
    "similarity",
    "coverage_ratio",
    "novelty_ratio",
    "garbled_ratio",
    "garbled_penalty",
    "word_noncompliance_ratio",
    "word_penalty",
    "unk_char_ratio",
    "disallowed_char_ratio",
    "control_char_ratio",
    "lexical_cosine",
    "lexical_js_similarity",
    "lexical_token_count",
    "capital_value",
    "capital_coverage",
    "capital_diversity",
    "capital_redundancy",
    "capital_verification_ratio",
    "capital_fact_count",
    "capital_operations",
    "operation_cost",
    "budget_remaining",
    "budget_breach",
    "cumulative_cost",
]

ROUND_CSV_HEADERS = [
    "round",
    "steps",
    "total_reward",
    "average_reward",
]
MODEL_SIZE_BYTES = 209_460_851
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_training_config() -> tuple[dict[str, Any], Path]:
    """Load training configuration and return it with the source path."""

    if CONFIG_OVERRIDE_PATH.exists():
        config_path = CONFIG_OVERRIDE_PATH
    else:
        config_path = CONFIG_TEMPLATE_PATH
    raw_config: dict[str, Any] = {}
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = {}
        if isinstance(loaded, dict):
            raw_config = dict(loaded)
    reference_path = str(
        raw_config.get(
            "reference_actions_path",
            DEFAULT_REFERENCE_ACTIONS_PATH,
        )
    )

    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    warmup_rounds = max(
        0,
        _as_int(
            raw_config.get(
                "reference_warmup_rounds",
                DEFAULT_REFERENCE_WARMUP_ROUNDS,
            ),
            DEFAULT_REFERENCE_WARMUP_ROUNDS,
        ),
    )
    warmup_steps = max(
        0,
        _as_int(
            raw_config.get(
                "reference_warmup_steps",
                DEFAULT_REFERENCE_WARMUP_STEPS,
            ),
            DEFAULT_REFERENCE_WARMUP_STEPS,
        ),
    )
    config = {
        "reference_actions_path": reference_path,
        "reference_warmup_rounds": warmup_rounds,
        "reference_warmup_steps": warmup_steps,
    }
    return config, config_path


def _announce_training_config(config_path: Path, config: Mapping[str, Any]) -> None:
    """Print the resolved configuration path and content."""

    try:
        display_path = config_path.relative_to(REPO_ROOT)
    except ValueError:
        display_path = config_path
    _console_log(f"Training configuration file: {display_path}")
    pretty = json.dumps(config, ensure_ascii=False, indent=2)
    for line in pretty.splitlines():
        _console_log(f"  {line}")

try:  # pragma: no cover - exercised depending on invocation style
    from .rl_sac.agent import AgentConfig, SACAgent
    from .rl_sac.networks import NetworkFactory
    from .rl_sac.replay_buffer import BaseReplayBuffer, Transition
    from .rl_sac.trainer import Trainer, TrainerConfig
except ImportError:  # pragma: no cover - fallback for direct script execution
    from rl_sac.agent import AgentConfig, SACAgent  # type: ignore[no-redef]
    from rl_sac.networks import NetworkFactory  # type: ignore[no-redef]
    from rl_sac.replay_buffer import BaseReplayBuffer, Transition  # type: ignore[no-redef]
    from rl_sac.trainer import Trainer, TrainerConfig  # type: ignore[no-redef]

ARTICLE_SEGMENT_SEPARATOR = "[----------------------------------------------------->"

QUALITY_SIMILARITY_WEIGHT = 0.6
QUALITY_COVERAGE_WEIGHT = 0.3
QUALITY_NOVELTY_WEIGHT = 0.1
QUALITY_NONLINEAR_EXPONENT = 4.0
LEXICAL_NONLINEAR_EXPONENT = 3.5
GARBLED_REWARD_WEIGHT = 0.5
WORD_COMPLIANCE_REWARD_WEIGHT = 0.7
CLEANLINESS_NONLINEAR_EXPONENT = 5.0
CONTROL_CHAR_WHITELIST = {"\n", "\r", "\t"}

LEXICAL_STATS_SUFFIX = "_lexical.json"
LEXICAL_SIMILARITY_WEIGHT = 0.15
LEXICAL_JS_WEIGHT = 0.1

COMMON_SUMMARY_PUNCTUATION = set(",.!?;:'\"-()[] ")
DEFAULT_COMPLIANCE_TEMPERATURE = 0.85
COMPLIANCE_INVALID_LOGIT_PENALTY = 12.0
OPERATION_COSTS: Mapping[str, float] = {
    'ACQUIRE': 3.0,
    'EXTRACT': 3.0,
    'LINK': 2.0,
    'VERIFY': 4.0,
    'HEDGE': 1.5,
    'TRIM': 1.0,
    'COMMIT': 5.0,
}
DEFAULT_OPERATION_COST = 2.0
DEFAULT_INITIAL_BUDGET = 1200.0
COST_WEIGHT = 0.08
COST_WEIGHT = 0.08
BUDGET_PENALTY_WEIGHT = 0.02
CAPITAL_COVERAGE_WEIGHT = 1.5
CAPITAL_DIVERSITY_WEIGHT = 0.8
CAPITAL_REDUNDANCY_WEIGHT = 0.6
CAPITAL_VERIFICATION_BONUS = 0.4
CAPITAL_FACT_WEIGHT = 0.45

ANSI_GREEN = "[32m"
ANSI_RED = "[31m"
ANSI_YELLOW = "[33m"
ANSI_RESET = "[0m"


class TorchUnavailableError(RuntimeError):
    """Raised when a demo component requiring PyTorch is accessed without it."""


def _raise_torch_unavailable(component: str) -> None:
    """Raise a consistent error guiding users to install PyTorch for the demo."""

    raise TorchUnavailableError(
        f"{component} 需要先安装 PyTorch 才能使用。"
        "请运行 'scripts/install_pytorch.sh' 或执行 "
        "'python -m pip install torch --index-url https://download.pytorch.org/whl/cpu'。"
    )


if not TORCH_AVAILABLE:
    class _TorchProxy:
        def __getattr__(self, name: str) -> Any:  # pragma: no cover - simple error proxy
            _raise_torch_unavailable(f"torch.{name}")


    class _NNProxy:
        Module = object

        def __getattr__(self, name: str) -> Any:  # pragma: no cover - simple error proxy
            _raise_torch_unavailable(f"torch.nn.{name}")


    class _CategoricalProxy:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            _raise_torch_unavailable("torch.distributions.Categorical")


    class _FunctionalProxy:
        def __getattr__(self, name: str) -> Any:  # pragma: no cover - simple error proxy
            _raise_torch_unavailable(f"torch.nn.functional.{name}")


    def _missing_pack_padded_sequence(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
        _raise_torch_unavailable("torch.nn.utils.rnn.pack_padded_sequence")


    torch = _TorchProxy()  # type: ignore[assignment]
    nn = _NNProxy()  # type: ignore[assignment]
    Categorical = _CategoricalProxy  # type: ignore[assignment]
    F = _FunctionalProxy()  # type: ignore[assignment]
    pack_padded_sequence = _missing_pack_padded_sequence  # type: ignore[assignment]


@dataclass
class TextObservation:
    """Observation containing the previous summary and current chapter text."""

    previous_summary: str
    chapter_text: str
    step_index: int


@dataclass
class TextAction:
    """Action emitted by the policy consisting of token ids and decoded text."""

    token_ids: List[int]
    text: str
    length: int


class CharTokenizer:
    """Character-level tokenizer shared between the policy and value networks."""

    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    SEP = "<sep>"
    UNK = "<unk>"

    def __init__(
            self,
            texts: Sequence[str],
            *,
            summary_charset: set[str] | None = None,
            punctuation_whitelist: set[str] | None = None,
    ) -> None:
        charset = set()
        for text in texts:
            charset.update(text)
        special_tokens = [self.PAD, self.BOS, self.EOS, self.SEP, self.UNK]
        punctuation_whitelist = punctuation_whitelist or set()
        base_tokens: set[str] = set()
        for char in charset:
            if char in special_tokens:
                continue
            if _is_cjk(char) and summary_charset is not None and char not in summary_charset:
                continue
            base_tokens.add(char)
        base_tokens.update(punctuation_whitelist)
        regular_tokens = sorted(token for token in base_tokens if token not in special_tokens)
        self.vocab: List[str] = special_tokens + regular_tokens
        self.stoi = {token: idx for idx, token in enumerate(self.vocab)}
        self.itos = {idx: token for token, idx in self.stoi.items()}
        self.special_tokens = set(special_tokens)
        self._allowed_characters = {
            token for token in self.vocab if len(token) == 1 and token not in self.special_tokens
        }
        self._allowed_characters.update(CONTROL_CHAR_WHITELIST)
        self._summary_token_ids = [
            idx for idx, token in enumerate(self.vocab) if token not in self.special_tokens
        ]
        self._summary_charset = set(summary_charset or [])
        self._punctuation_whitelist = set(punctuation_whitelist)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.EOS]

    @property
    def sep_id(self) -> int:
        return self.stoi[self.SEP]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def allowed_characters(self) -> set[str]:
        return set(self._allowed_characters)

    def token_from_id(self, token_id: int) -> str:
        return self.itos.get(token_id, "")

    @property
    def summary_token_ids(self) -> List[int]:
        return list(self._summary_token_ids)

    def _encode_chars(self, text: str) -> List[int]:
        return [self.stoi.get(char, self.unk_id) for char in text]

    def encode_observation(self, observation: TextObservation) -> List[int]:
        tokens: List[int] = [self.bos_id]
        tokens.extend(self._encode_chars(observation.previous_summary))
        tokens.append(self.sep_id)
        tokens.extend(self._encode_chars(observation.chapter_text))
        tokens.append(self.eos_id)
        return tokens

    def encode_action_text(self, text: str) -> List[int]:
        tokens: List[int] = [self.bos_id]
        tokens.extend(self._encode_chars(text))
        tokens.append(self.eos_id)
        return tokens

    def decode_action(self, token_ids: Sequence[int]) -> str:
        decoded: List[str] = []
        for token_id in token_ids:
            if token_id == self.eos_id:
                break
            if token_id in (self.bos_id, self.pad_id):
                continue
            decoded.append(self.itos.get(token_id, ""))
        return "".join(decoded)

    def batch_encode(
            self, sequences: Sequence[Sequence[int]], *, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not sequences:
            raise ValueError("Cannot encode an empty batch of sequences.")
        max_length = max(len(seq) for seq in sequences)
        batch = torch.full(
            (len(sequences), max_length), self.pad_id, dtype=torch.long, device=device
        )
        lengths = torch.tensor(
            [len(seq) for seq in sequences], dtype=torch.long, device=device
        )
        for row, seq in enumerate(sequences):
            batch[row, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        return batch, lengths


def _is_cjk(char: str) -> bool:
    """Return ``True`` if ``char`` is a CJK unified ideograph."""

    if not char:
        return False
    codepoint = ord(char)
    return 0x4E00 <= codepoint <= 0x9FFF


def _compute_common_summary_charset(article_path: Path) -> set[str]:
    """Derive high-frequency CJK characters from the sample article."""

    try:
        article_text = article_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return set()
    cjk_counts = Counter(char for char in article_text if _is_cjk(char))
    if not cjk_counts:
        return set()
    frequencies = list(cjk_counts.values())
    mean = statistics.mean(frequencies)
    stdev = statistics.pstdev(frequencies) if len(frequencies) > 1 else 0.0
    threshold = mean + stdev
    if threshold <= 0:
        return set(cjk_counts.keys())
    common = {char for char, count in cjk_counts.items() if count >= threshold}
    if common:
        return common
    fallback = {char for char, _ in cjk_counts.most_common(200)}
    return fallback


class WordComplianceChecker:
    """Detect non-compliant Chinese bigrams to penalize garbled wording."""

    def __init__(self, texts: Sequence[str]) -> None:
        self.allowed_unigrams: set[str] = set()
        self.allowed_bigrams: set[str] = set()
        for text in texts:
            cjk_sequence = []
            for char in text:
                if _is_cjk(char):
                    self.allowed_unigrams.add(char)
                    cjk_sequence.append(char)
                else:
                    if len(cjk_sequence) >= 2:
                        self._register_bigrams(cjk_sequence)
                    cjk_sequence = []
            if len(cjk_sequence) >= 2:
                self._register_bigrams(cjk_sequence)

    def _register_bigrams(self, chars: Sequence[str]) -> None:
        for idx in range(len(chars) - 1):
            bigram = chars[idx] + chars[idx + 1]
            self.allowed_bigrams.add(bigram)

    def is_candidate_allowed(self, previous_char: str | None, candidate: str) -> bool:
        """Return True if candidate can follow previous_char."""

        if not candidate:
            return False
        if not _is_cjk(candidate):
            return True
        if candidate not in self.allowed_unigrams:
            return False
        if not previous_char or not _is_cjk(previous_char):
            return True
        return (previous_char + candidate) in self.allowed_bigrams

    def noncompliant_ratio(self, summary: str) -> float:
        """Return the ratio of Chinese characters not aligned with known words."""

        cjk_positions = [idx for idx, char in enumerate(summary) if _is_cjk(char)]
        total_cjk = len(cjk_positions)
        if total_cjk == 0:
            return 0.0

        flagged = [False] * len(summary)
        for idx, char in enumerate(summary):
            if not _is_cjk(char):
                continue
            if char not in self.allowed_unigrams:
                flagged[idx] = True

        for idx in range(len(summary) - 1):
            left = summary[idx]
            right = summary[idx + 1]
            if not (_is_cjk(left) and _is_cjk(right)):
                continue
            bigram = left + right
            if bigram not in self.allowed_bigrams:
                flagged[idx] = True
                flagged[idx + 1] = True

        noncompliant = sum(1 for idx in cjk_positions if flagged[idx])
        return noncompliant / total_cjk


@dataclass
class Operation:
    kind: str
    payload: str | tuple[str, str] | None = None


class OperationParser:
    """Parse textual actions into structured operations."""

    COMMAND_ALIASES: Mapping[str, str] = {
        "ACQ": "ACQUIRE",
        "EXT": "EXTRACT",
        "VER": "VERIFY",
        "HDG": "HEDGE",
        "TRM": "TRIM",
        "CMIT": "COMMIT",
        "LNK": "LINK",
    }
    NATURAL_KEYWORDS: Mapping[str, str] = {
        "acquire": "ACQUIRE",
        "collect": "ACQUIRE",
        "extract": "EXTRACT",
        "gather": "ACQUIRE",
        "verify": "VERIFY",
        "confirm": "VERIFY",
        "hedge": "HEDGE",
        "trim": "TRIM",
        "commit": "COMMIT",
        "link": "LINK",
        "connect": "LINK",
        "relate": "LINK",
    }
    FREEFORM_PREFIXES: Mapping[str, str] = {
        "fact": "ACQUIRE",
        "evidence": "ACQUIRE",
        "hypothesis": "ACQUIRE",
        "note": "ACQUIRE",
        "verification": "VERIFY",
        "link": "LINK",
    }

    @classmethod
    def parse(cls, action_text: str) -> list[Operation]:
        operations: list[Operation] = []
        if not action_text:
            return operations
        seen: set[tuple[str, tuple[str, ...]]] = set()

        def normalize_payload(payload: str | tuple[str, str] | None) -> tuple[str, ...]:
            if isinstance(payload, tuple):
                return tuple(part.strip() for part in payload if part and part.strip()) or ("",)
            if payload is None:
                return ("",)
            cleaned = payload.strip()
            return (cleaned,) if cleaned else ("",)

        def register(command: str, payload: str | tuple[str, str] | None) -> None:
            normalized_command = command.upper()
            key = (normalized_command, normalize_payload(payload))
            if key in seen:
                return
            seen.add(key)
            if isinstance(payload, tuple):
                payload_value: str | tuple[str, str] = tuple(part.strip() for part in payload)
            elif payload is None:
                payload_value = ""
            else:
                payload_value = payload.strip()
            operations.append(Operation(normalized_command, payload_value))

        for raw_line in action_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parsed = cls._parse_structured_line(line)
            if parsed:
                for command, payload in parsed:
                    register(command, payload)
                continue
            for command, payload in cls._heuristic_parse_line(line):
                register(command, payload)

        if not operations:
            for command, payload in cls._heuristic_parse_line(action_text):
                register(command, payload)

        return operations

    @classmethod
    def _parse_structured_line(
            cls, line: str
    ) -> list[tuple[str, str | tuple[str, str]]]:
        operations: list[tuple[str, str | tuple[str, str]]] = []
        parts = line.split(maxsplit=1)
        if not parts:
            return operations
        command = parts[0].upper()
        command = cls.COMMAND_ALIASES.get(command, command)
        payload = parts[1].strip() if len(parts) > 1 else ""
        if command == "LINK":
            left, right = cls._parse_link_payload(payload)
            if left or right:
                operations.append(("LINK", (left, right)))
            return operations
        if command in {"ACQUIRE", "EXTRACT", "VERIFY", "HEDGE", "TRIM", "COMMIT"}:
            operations.append((command, payload))
        return operations

    @classmethod
    def _heuristic_parse_line(
            cls, line: str
    ) -> list[tuple[str, str | tuple[str, str]]]:
        results: list[tuple[str, str | tuple[str, str]]] = []
        fragment = line.strip()
        if not fragment:
            return results
        lower = fragment.lower()
        for prefix, command in cls.FREEFORM_PREFIXES.items():
            if lower.startswith(prefix + ":"):
                payload = fragment.split(":", 1)[1].strip()
                if payload:
                    if command == "LINK":
                        results.append(("LINK", cls._heuristic_link_payload(payload)))
                    else:
                        results.append((command, payload))
                return results
        pattern = re.compile(r"(acquire|collect|extract|gather|verify|confirm|hedge|trim|commit|link|connect|relate)",
                             re.IGNORECASE)
        for match in pattern.finditer(fragment):
            keyword = match.group(1).lower()
            command = cls.NATURAL_KEYWORDS.get(keyword)
            if not command:
                continue
            remainder = fragment[match.end():]
            split_match = re.search(r"[.;。！？!?]\s*", remainder)
            if split_match:
                payload = remainder[:split_match.start()]
            else:
                payload = remainder
            payload = payload.strip(" :-\"'()[]")
            if not payload:
                payload = fragment[:match.start()].strip(" :-\"'()[]")
            if command == "LINK":
                results.append(("LINK", cls._heuristic_link_payload(payload)))
            else:
                results.append((command, payload))

        return results
    @classmethod
    def _heuristic_link_payload(cls, payload: str) -> tuple[str, str]:
        stripped = payload.strip()
        if not stripped:
            return "", ""
        if "->" in stripped or "=>" in stripped:
            return cls._parse_link_payload(stripped)
        lower = stripped.lower()
        for separator in [" to ", " into ", " toward ", " towards ", " against "]:
            if separator in lower:
                parts = re.split(separator, stripped, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2:
                    return parts[0].strip(), parts[1].strip()
        if " and " in lower:
            parts = re.split(r"and", stripped, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
        tokens = stripped.split()
        if len(tokens) >= 4:
            midpoint = len(tokens) // 2
            left = " ".join(tokens[:midpoint]).strip()
            right = " ".join(tokens[midpoint:]).strip()
            return left or stripped, right or stripped
        return stripped, stripped

    @staticmethod
    def _parse_link_payload(payload: str) -> tuple[str, str]:
        if "->" in payload:
            left, right = payload.split("->", 1)
            return left.strip(), right.strip()
        if "=>" in payload:
            left, right = payload.split("=>", 1)
            return left.strip(), right.strip()
        tokens = payload.split(maxsplit=1)
        if len(tokens) == 2:
            return tokens[0].strip(), tokens[1].strip()
        return payload.strip(), payload.strip()


def _canonicalize_action_text(action_text: str) -> tuple[str, list[Operation]]:
    """Return a canonical summary text derived from structured operations."""

    operations = OperationParser.parse(action_text)
    if not operations:
        return action_text, operations

    def _clean_payload(payload: str) -> str:
        text = payload.strip()
        text = re.sub(r"^CH\d{2}\s+", "", text)
        return text.strip()

    fragments: list[str] = []
    seen_fragments: set[str] = set()
    for operation in operations:
        payload = operation.payload
        if isinstance(payload, tuple):
            for part in payload:
                cleaned = _clean_payload(str(part))
                if cleaned and cleaned not in seen_fragments:
                    seen_fragments.add(cleaned)
                    fragments.append(cleaned)
        elif isinstance(payload, str):
            cleaned = _clean_payload(payload)
            if cleaned and cleaned not in seen_fragments:
                seen_fragments.add(cleaned)
                fragments.append(cleaned)
        elif payload is not None:
            cleaned = _clean_payload(str(payload))
            if cleaned and cleaned not in seen_fragments:
                seen_fragments.add(cleaned)
                fragments.append(cleaned)
    canonical = "。".join(fragments).strip()
    if not canonical:
        canonical = action_text
    return canonical, operations


class CognitiveCapital:
    """Stateful representation of extracted knowledge."""

    def __init__(self) -> None:
        self.facts: set[str] = set()
        self.links: set[tuple[str, str]] = set()
        self.verified: set[str] = set()
        self.hedged: set[str] = set()

    def clone(self) -> "CognitiveCapital":
        clone = CognitiveCapital()
        clone.facts = set(self.facts)
        clone.links = set(self.links)
        clone.verified = set(self.verified)
        clone.hedged = set(self.hedged)
        return clone

    def apply(self, operation: Operation) -> dict[str, object]:
        result = {"applied": False, "detail": ""}
        op_type = operation.kind.upper()
        if op_type in {"ACQUIRE", "EXTRACT"}:
            fact = self._normalize_fact(str(operation.payload or ""))
            if fact:
                before = len(self.facts)
                self.facts.add(fact)
                result["applied"] = len(self.facts) > before
                result["detail"] = fact
        elif op_type == "TRIM":
            fact = self._normalize_fact(str(operation.payload or ""))
            if fact and fact in self.facts:
                self.facts.remove(fact)
                self.verified.discard(fact)
                self.hedged.discard(fact)
                result["applied"] = True
                result["detail"] = fact
        elif op_type == "VERIFY":
            fact = self._normalize_fact(str(operation.payload or ""))
            if fact:
                self.verified.add(fact)
                result["applied"] = True
                result["detail"] = fact
        elif op_type == "HEDGE":
            fact = self._normalize_fact(str(operation.payload or ""))
            if fact:
                self.hedged.add(fact)
                result["applied"] = True
                result["detail"] = fact
        elif op_type == "LINK":
            left, right = operation.payload if isinstance(operation.payload, tuple) else ("", "")
            left = self._normalize_fact(left)
            right = self._normalize_fact(right)
            if left and right:
                edge = tuple(sorted((left, right)))
                before = len(self.links)
                self.links.add(edge)
                result["applied"] = len(self.links) > before
                result["detail"] = edge
        elif op_type == "COMMIT":
            summary_fact = self._normalize_fact(str(operation.payload or ""))
            if summary_fact:
                self.facts.add(summary_fact)
                self.verified.add(summary_fact)
                result["applied"] = True
                result["detail"] = summary_fact
        return result

    def _normalize_fact(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def render_text(self, budget: float) -> str:
        fact_preview = "; ".join(sorted(self.facts)[:5])
        link_preview = "; ".join("->".join(edge) for edge in sorted(self.links)[:5])
        verified_preview = "; ".join(sorted(self.verified)[:5])
        hedge_preview = "; ".join(sorted(self.hedged)[:5])
        return (
            f"BUDGET={budget:.1f} | FACTS=[{fact_preview}] | LINKS=[{link_preview}] "
            f"| VERIFIED=[{verified_preview}] | HEDGED=[{hedge_preview}]"
        )

    def all_facts(self) -> set[str]:
        return set(self.facts)


class CapitalValuator:
    """Compute valuation and potential for cognitive capital."""

    def __init__(self, chapters: Sequence[str]) -> None:
        self.universe_tokens: set[str] = set()
        self.category_universe: set[str] = set()
        for chapter in chapters:
            tokens = self._tokenize(chapter)
            self.universe_tokens.update(tokens)
            if tokens:
                self.category_universe.add(next(iter(tokens)))
        if not self.universe_tokens:
            self.universe_tokens.add("token")
        if not self.category_universe:
            self.category_universe.add("general")

    def _tokenize(self, text: str) -> set[str]:
        tokens: set[str] = {
            token.lower() for token in re.findall(r"[A-Za-z0-9]+", text) if len(token) > 1
        }
        tokens.update(re.findall(r"[\u4e00-\u9fff]{2,}", text))
        return tokens

    def _fact_tokens(self, fact: str) -> set[str]:
        return self._tokenize(fact)

    def metrics(self, capital: CognitiveCapital) -> dict[str, float]:
        token_coverage: set[str] = set()
        categories: set[str] = set()
        fact_tokens: list[set[str]] = []
        for fact in capital.all_facts():
            tokens = self._fact_tokens(fact)
            if tokens:
                token_coverage.update(tokens)
                categories.add(next(iter(tokens)))
            fact_tokens.append(tokens)
        coverage_ratio = len(token_coverage) / max(1, len(self.universe_tokens))
        diversity_ratio = len(categories) / max(1, len(self.category_universe))
        redundancy_score = self._redundancy(fact_tokens)
        verification_ratio = len(capital.verified) / max(1, len(capital.facts))
        hedge_ratio = len(capital.hedged) / max(1, len(capital.facts) + len(capital.hedged))
        return {
            "coverage": coverage_ratio,
            "diversity": diversity_ratio,
            "redundancy": redundancy_score,
            "verification": verification_ratio,
            "hedge": hedge_ratio,
            "fact_count": float(len(capital.facts)),
        }

    def _redundancy(self, fact_tokens: list[set[str]]) -> float:
        if len(fact_tokens) < 2:
            return 0.0
        pair_scores: list[float] = []
        for left, right in itertools.combinations(fact_tokens, 2):
            if not left or not right:
                continue
            intersection = left.intersection(right)
            union = left.union(right)
            if union:
                pair_scores.append(len(intersection) / len(union))
        if not pair_scores:
            return 0.0
        return sum(pair_scores) / len(pair_scores)

    def value(self, capital: CognitiveCapital) -> float:
        metrics = self.metrics(capital)
        raw_value = (
                CAPITAL_COVERAGE_WEIGHT * metrics["coverage"]
                + CAPITAL_DIVERSITY_WEIGHT * metrics["diversity"]
                - CAPITAL_REDUNDANCY_WEIGHT * metrics["redundancy"]
                + CAPITAL_VERIFICATION_BONUS * metrics["verification"]
                + CAPITAL_FACT_WEIGHT * math.log1p(metrics["fact_count"])
        )
        hedge_discount = 1.0 - 0.2 * metrics["hedge"]
        return max(raw_value * hedge_discount, 0.0)

    def potential(self, capital: CognitiveCapital) -> float:
        return self.value(capital)


def _format_text_debug(text: str, head: int = 10, tail: int = 10) -> Tuple[int, str]:
    """Return the length of ``text`` and a preview with an ellipsis."""

    length = len(text)
    if length <= head + tail:
        preview = text
    else:
        preview = f"{text[:head]}...{text[-tail:]}"
    return length, preview


def _describe_metric_quality(key: str, value: float) -> str:
    """Return a qualitative assessment for a scalar diagnostic metric."""

    if math.isnan(value):
        return "本次指标缺失，无法评估"

    if key == "length_ratio":
        if value < 0.10:
            return "本次严重偏低，摘要几乎无法覆盖章节要点"
        if value < 0.15:
            return "本次明显偏低，需要显著扩展摘要"
        if value < 0.25:
            return "本次偏低，接近建议范围下限"
        if value <= 0.40:
            return "本次处于推荐范围内"
        return "本次偏高，摘要可能略显冗长"
    if key == "similarity":
        if value < 0.10:
            return "本次几乎没有贴合原文"
        if value < 0.30:
            return "本次贴合度偏低"
        if value < 0.60:
            return "本次贴合度一般"
        if value < 0.80:
            return "本次贴合度较好"
        return "本次高度贴近原文"
    if key == "coverage_ratio":
        if value < 0.10:
            return "本次覆盖率极低，遗漏大量信息"
        if value < 0.30:
            return "本次覆盖率偏低，需补充要点"
        if value < 0.60:
            return "本次覆盖率中等"
        if value < 0.80:
            return "本次覆盖率良好"
        return "本次覆盖率接近完整"
    if key == "novelty_ratio":
        if value < 0.20:
            return "本次几乎完全复述原文"
        if value < 0.40:
            return "本次新意较少"
        if value < 0.70:
            return "本次改写幅度适中"
        if value < 0.90:
            return "本次改写幅度较大"
        return "本次新意极高，需确认信息是否充分"
    if key == "lexical_cosine":
        if value < 0.05:
            return "本次关键词匹配几乎缺失"
        if value < 0.15:
            return "本次关键词匹配偏弱"
        if value < 0.30:
            return "本次关键词匹配一般"
        if value < 0.50:
            return "本次关键词匹配良好"
        return "本次关键词高度吻合"
    if key == "lexical_js_similarity":
        if value < 0.05:
            return "本次词频结构相差较大"
        if value < 0.15:
            return "本次词频结构匹配偏弱"
        if value < 0.30:
            return "本次词频结构相似度一般"
        if value < 0.50:
            return "本次词频结构匹配良好"
        return "本次词频结构高度一致"
    if key == "garbled_ratio":
        if value <= 1e-4:
            return "本次无明显乱码"
        if value < 0.01:
            return "本次乱码比例很低"
        if value < 0.05:
            return "本次乱码偏多，需要关注"
        return "本次乱码严重，需立即处理"
    if key == "word_noncompliance_ratio":
        if value <= 1e-4:
            return "本次词语合规性完全正常"
        if value < 0.01:
            return "本次词语合规性轻微异常"
        if value < 0.05:
            return "本次词语合规性偏弱"
        return "本次词语合规性严重不足"
    if key == "capital_value":
        if value < 0.20:
            return "本次认知资本价值偏低，需补充高价值事实"
        if value < 0.50:
            return "本次认知资本价值中等，可继续累积"
        if value < 1.00:
            return "本次认知资本价值稳健"
        return "本次认知资本价值表现突出"
    if key == "capital_coverage":
        if value < 0.15:
            return "认知覆盖范围极窄"
        if value < 0.35:
            return "认知覆盖偏少"
        if value < 0.60:
            return "认知覆盖尚可"
        return "认知覆盖全面"
    if key == "capital_diversity":
        if value < 0.10:
            return "事实主题单一"
        if value < 0.30:
            return "事实主题略显集中"
        if value < 0.60:
            return "事实主题较为多样"
        return "事实主题高度多样"
    if key == "capital_redundancy":
        if value < 0.05:
            return "冗余极低"
        if value < 0.15:
            return "冗余可接受"
        if value < 0.30:
            return "冗余偏高，需合并重复信息"
        return "冗余严重，建议清理"
    if key == "capital_verification_ratio":
        if value < 0.10:
            return "验证覆盖不足"
        if value < 0.40:
            return "验证尚需加强"
        if value < 0.70:
            return "验证覆盖良好"
        return "验证充分"
    if key == "budget_remaining":
        if value < -1e-6:
            return "预算已透支"
        if value < 20:
            return "预算即将耗尽"
        if value < 60:
            return "预算消耗过半"
        return "预算仍然充裕"
    if key == "capital_fact_count":
        if value < 3:
            return "事实数量偏少"
        if value < 8:
            return "事实数量适中"
        return "事实数量充足"
    return "本次指标无预设评估标准"


def _describe_penalty_component(value: float, label: str) -> str:
    """Return a qualitative summary for a penalty component."""

    if math.isnan(value):
        return f"{label}缺失"
    if value <= 1e-4:
        return f"{label}几乎为零"
    if value < 0.01:
        return f"{label}轻微"
    if value < 0.05:
        return f"{label}偏高"
    return f"{label}严重"


def _describe_reward_quality(value: float) -> str:
    """Return a qualitative description of the scalar reward."""

    if math.isnan(value):
        return "奖励缺失"
    if value >= 1.8:
        return "本次获得爆发式奖励"
    if value >= 1.2:
        return "本次获得显著正向反馈"
    if value > 0.6:
        return "本次获得中等奖励"
    if value > 0.0:
        return "本次获得轻度奖励"
    return "本次未获得奖励"


def _clamp_unit_interval(value: float) -> float:
    """Clamp ``value`` to the inclusive range ``[0, 1]``."""

    if math.isnan(value):
        return 0.0
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _nonlinear_reward(value: float, exponent: float) -> float:
    """Amplify ``value`` via ``1 - (1 - value)^exponent`` for heavy rewards."""

    if exponent <= 0:
        raise ValueError("exponent must be positive for nonlinear reward shaping")
    base = _clamp_unit_interval(value)
    if base == 0.0:
        return 0.0
    if base == 1.0:
        return 1.0
    return 1.0 - math.pow(1.0 - base, exponent)


def _append_csv_row(path: Path, headers: Sequence[str], row: Mapping[str, Any]) -> None:
    """Append ``row`` to ``path`` ensuring headers are written once."""

    OUT_DIR.mkdir(exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in headers})
        handle.flush()


def _reset_output_artifacts() -> None:
    """Remove stale CSV/HTML/snapshot artifacts before a new training session."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (STEP_CSV_PATH, ROUND_CSV_PATH, REWARDS_HTML_PATH, STEP_LOG_PATH, TRAIN_LOG_PATH):
        if path.exists():
            path.unlink()
    if ROUND_SNAPSHOT_DIR.exists():
        for snapshot in ROUND_SNAPSHOT_DIR.glob("*.json"):
            if snapshot.is_file():
                snapshot.unlink()
    ROUND_SNAPSHOT_DIR.mkdir(exist_ok=True)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load ``path`` into a list of dictionaries; return an empty list if missing."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _parse_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return 0.0
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _parse_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        if isinstance(value, int):
            return value
        text = str(value).strip()
        if not text:
            return 0
        return int(float(text))
    except (TypeError, ValueError):
        return 0


ANSI_REWARD_TAGS = {
    ANSI_GREEN: "[reward>0]",
    ANSI_RED: "[reward<0]",
    ANSI_YELLOW: "[reward=0]",
}


def _append_step_log(lines: Sequence[str], block_color: str) -> None:
    if not lines:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STEP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tag = ANSI_REWARD_TAGS.get(block_color, "[reward] ")
    with STEP_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{tag}\n")
        for raw in lines:
            handle.write(f"{raw}\n")
        handle.write("\n")


def _console_log(message: str, *, color: str | None = None, log: bool = True) -> None:
    if color:
        print(f"{color}{message}{ANSI_RESET}")
    else:
        print(message)
    if log:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        TRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRAIN_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")


def _build_rewards_dashboard_html(
        step_rows: Sequence[Mapping[str, Any]],
        round_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Construct a standalone HTML dashboard summarizing reward trends."""

    step_data = [
        {
            "round": _parse_int(row.get("round")),
            "step": _parse_int(row.get("step")),
            "global_step": _parse_int(row.get("global_step")),
            "reward": _parse_float(row.get("reward")),
        }
        for row in step_rows
    ]
    round_data = [
        {
            "round": _parse_int(row.get("round")),
            "steps": _parse_int(row.get("steps")),
            "total_reward": _parse_float(row.get("total_reward")),
            "average_reward": _parse_float(row.get("average_reward")),
        }
        for row in round_rows
    ]
    summary_text: str
    if round_data:
        latest = round_data[-1]
        summary_text = (
            "已记录 {count} 轮训练；最近一轮 (Round {round}) 总奖励 {total:.3f}，"
            "平均奖励 {average:.3f}。"
        ).format(
            count=len(round_data),
            round=latest["round"],
            total=latest["total_reward"],
            average=latest["average_reward"],
        )
    else:
        summary_text = "尚未记录任何轮次汇总数据。"
    step_json = json.dumps(step_data, ensure_ascii=False)
    round_json = json.dumps(round_data, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang=\"zh-CN\">
  <head>
    <meta charset=\"utf-8\" />
    <title>训练奖励概览</title>
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <style>
      body {{
        font-family: \"Segoe UI\", \"Helvetica Neue\", Arial, \"PingFang SC\", sans-serif;
        margin: 2rem;
        background: #f7f7f7;
        color: #222;
      }}
      h1 {{
        margin-bottom: 0.5rem;
      }}
      p.description {{
        margin-top: 0;
        color: #555;
      }}
      p.summary {{
        color: #333;
      }}
      .chart-wrapper {{
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
      }}
      canvas {{
        max-width: 100%;
      }}
      #status {{
        margin-bottom: 1rem;
        color: #d9534f;
      }}
    </style>
    <script src=\"https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js\"></script>
  </head>
  <body>
    <h1>训练奖励概览</h1>
    <p class=\"description\">本页面基于最新导出的 CSV 快照自动生成。</p>
    <p class=\"summary\">{summary_text}</p>
    <div id=\"status\"></div>
    <div class=\"chart-wrapper\">
      <h2>Step 奖励走势</h2>
      <canvas id=\"stepChart\" height=\"320\"></canvas>
    </div>
    <div class=\"chart-wrapper\">
      <h2>轮次总奖励</h2>
      <canvas id=\"roundChart\" height=\"320\"></canvas>
    </div>
    <script>
      const stepData = {step_json};
      const roundData = {round_json};

      function renderDashboard() {{
        const status = document.getElementById('status');
        if (stepData.length === 0) {{
          status.textContent = '未找到 Step 指标数据，请确认训练是否成功写入 CSV。';
          document.querySelectorAll('.chart-wrapper').forEach((wrapper) => {{
            wrapper.style.display = 'none';
          }});
          return;
        }}
        status.textContent = '';
        const stepCtx = document.getElementById('stepChart').getContext('2d');
        new Chart(stepCtx, {{
          type: 'line',
          data: {{
            labels: stepData.map((item) => item.global_step),
            datasets: [{{
              label: 'Step Reward',
              data: stepData.map((item) => item.reward),
              borderColor: '#007bff',
              backgroundColor: 'rgba(0, 123, 255, 0.12)',
              fill: true,
              tension: 0.2,
              pointRadius: 0,
            }}],
          }},
          options: {{
            responsive: true,
            interaction: {{ mode: 'index', intersect: false }},
            scales: {{
              x: {{ title: {{ display: true, text: 'Global Step' }} }},
              y: {{ title: {{ display: true, text: 'Reward' }} }},
            }},
          }},
        }});

        if (roundData.length === 0) {{
          document.getElementById('roundChart').closest('.chart-wrapper').style.display = 'none';
          return;
        }}

        const roundCtx = document.getElementById('roundChart').getContext('2d');
        new Chart(roundCtx, {{
          type: 'bar',
          data: {{
            labels: roundData.map((item) => `Round ${{item.round}}`),
            datasets: [{{
              label: 'Total Reward',
              data: roundData.map((item) => item.total_reward),
              backgroundColor: 'rgba(255, 159, 64, 0.6)',
              borderColor: '#ff9f40',
            }}],
          }},
          options: {{
            responsive: true,
            scales: {{
              x: {{ title: {{ display: true, text: 'Round' }} }},
              y: {{ title: {{ display: true, text: 'Total Reward' }} }},
            }},
          }},
        }});
      }}

      renderDashboard();
    </script>
  </body>
</html>
"""


def _write_rewards_dashboard(
        step_rows: Sequence[Mapping[str, Any]],
        round_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Persist ``rewards.html`` reflecting the current CSV contents."""

    html_content = _build_rewards_dashboard_html(step_rows, round_rows)
    REWARDS_HTML_PATH.parent.mkdir(exist_ok=True)
    with REWARDS_HTML_PATH.open("w", encoding="utf-8") as handle:
        handle.write(html_content)


def save_agent_snapshot(
        agent: "DemoSACAgent",
        metadata: Mapping[str, Any],
        path: Path,
) -> MutableMapping[str, Any]:
    """Serialize ``agent`` state alongside ``metadata`` to ``path``."""

    agent_state: MutableMapping[str, Any] = {}
    agent.save(agent_state)
    payload = {"agent_state": agent_state, "metadata": dict(metadata)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return agent_state


def _compute_garbled_statistics(
        summary: str, tokenizer: CharTokenizer
) -> Tuple[float, float, float, float]:
    """Return ratios describing garbled content in ``summary``."""

    if not summary:
        return 0.0, 0.0, 0.0, 0.0

    total_chars = len(summary)
    invalid_positions = [False] * total_chars
    disallowed_chars = 0
    control_chars = 0
    allowed_chars = tokenizer.allowed_characters
    for idx, char in enumerate(summary):
        category = unicodedata.category(char)
        is_control = category.startswith("C") and char not in CONTROL_CHAR_WHITELIST
        if char not in allowed_chars:
            disallowed_chars += 1
            invalid_positions[idx] = True
        if is_control:
            control_chars += 1
            invalid_positions[idx] = True

    unk_token = CharTokenizer.UNK
    start = 0
    unk_instances = 0
    while True:
        found = summary.find(unk_token, start)
        if found == -1:
            break
        unk_instances += 1
        for pos in range(found, min(total_chars, found + len(unk_token))):
            invalid_positions[pos] = True
        start = found + len(unk_token)

    garbled_chars = sum(1 for flag in invalid_positions if flag)
    garbled_ratio = garbled_chars / total_chars if total_chars else 0.0
    unk_ratio = (unk_instances * len(unk_token)) / total_chars if total_chars else 0.0
    disallowed_ratio = disallowed_chars / total_chars if total_chars else 0.0
    control_ratio = control_chars / total_chars if total_chars else 0.0
    return garbled_ratio, unk_ratio, disallowed_ratio, control_ratio


def _combine_summary_and_chapter(previous_summary: str, chapter: str) -> str:
    """Return the concatenation used as the evaluation source text."""

    if previous_summary and chapter:
        if previous_summary.endswith("\n"):
            return previous_summary + chapter
        return previous_summary + "\n" + chapter
    return previous_summary or chapter


def analyze_summary(
        summary: str,
        source_text: str,
        *,
        tokenizer: CharTokenizer | None = None,
        word_checker: WordComplianceChecker | None = None,
        chapter_text: str | None = None,
        chapter_index: int | None = None,
        lexical_stats: ChapterLexicalStatistics | None = None,
        lexical_tokenizer: LexicalTokenizer | None = None,
) -> MutableMapping[str, float]:
    """Compute quality statistics for the provided summary."""

    source_length = len(source_text)
    summary_length = len(summary)
    length_ratio = summary_length / source_length if source_length else 0.0
    matcher = difflib.SequenceMatcher(None, summary, source_text)
    match_blocks = matcher.get_matching_blocks()
    matched_chars = sum(block.size for block in match_blocks)
    longest_block = max((block.size for block in match_blocks), default=0)
    copy_ratio = (longest_block / summary_length) if summary_length else 0.0
    coverage_ratio = (matched_chars / source_length) if source_length else 0.0
    similarity = matcher.ratio()
    novelty_ratio = 1.0 - copy_ratio
    garbled_ratio = 0.0
    unk_char_ratio = 0.0
    disallowed_ratio = 0.0
    control_ratio = 0.0
    word_noncompliance_ratio = 0.0
    if tokenizer is not None:
        (
            garbled_ratio,
            unk_char_ratio,
            disallowed_ratio,
            control_ratio,
        ) = _compute_garbled_statistics(summary, tokenizer)
    lexical_cosine = 0.0
    lexical_js_similarity = 0.0
    lexical_token_count = 0
    if (
            lexical_stats is not None
            and lexical_tokenizer is not None
            and chapter_index is not None
    ):
        try:
            chapter_entry = lexical_stats.chapter_by_index(chapter_index)
        except KeyError:
            pass
        else:
            summary_vector = lexical_stats.vectorize_text(summary, lexical_tokenizer)
            lexical_token_count = summary_vector.token_count
            lexical_cosine = cosine_similarity(summary_vector.tfidf, chapter_entry.tfidf)
            lexical_js_similarity = jensen_shannon_similarity(
                summary_vector.probability, chapter_entry.probability
            )
    if word_checker is not None:
        word_noncompliance_ratio = word_checker.noncompliant_ratio(summary)
    metrics: MutableMapping[str, float] = {
        "summary_length": float(summary_length),
        "source_length": float(source_length),
        "length_ratio": float(length_ratio),
        "copy_ratio": float(copy_ratio),
        "coverage_ratio": float(coverage_ratio),
        "similarity": float(similarity),
        "novelty_ratio": float(max(0.0, novelty_ratio)),
        "garbled_ratio": float(garbled_ratio),
        "garbled_penalty": float(garbled_ratio),
        "word_noncompliance_ratio": float(word_noncompliance_ratio),
        "word_penalty": float(word_noncompliance_ratio),
        "unk_char_ratio": float(unk_char_ratio),
        "disallowed_char_ratio": float(disallowed_ratio),
        "control_char_ratio": float(control_ratio),
        "lexical_cosine": float(lexical_cosine),
        "lexical_js_similarity": float(lexical_js_similarity),
        "lexical_token_count": float(lexical_token_count),
    }
    if chapter_text is not None:
        metrics["chapter_length"] = float(len(chapter_text))
    return metrics


def load_article_features(path: Path) -> List[TextObservation]:
    """Load the sample article and return chapter observations with text only."""

    text = path.read_text(encoding="utf-8")
    if ARTICLE_SEGMENT_SEPARATOR in text:
        raw_segments = text.split(ARTICLE_SEGMENT_SEPARATOR)
    else:
        raw_segments = text.split("\n\n")
    chapters = [segment.strip() for segment in raw_segments if segment.strip()]
    observations: List[TextObservation] = []
    for idx, chapter in enumerate(chapters, start=1):
        observations.append(TextObservation(previous_summary="", chapter_text=chapter, step_index=idx))
    return observations


def _resolve_lexical_stats_path(article_path: Path) -> Path | None:
    stats_filename = f"{article_path.stem}{LEXICAL_STATS_SUFFIX}"
    candidates = [
        article_path.with_name(stats_filename),
        article_path.parent / stats_filename,
        REPO_ROOT / "data" / stats_filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_lexical_statistics_from_path(
        stats_path: Path,
) -> tuple[ChapterLexicalStatistics, LexicalTokenizer]:
    stats = ChapterLexicalStatistics.load(stats_path)
    try:
        tokenizer = LexicalTokenizer(
            stopwords=stats.stopwords,
            force_backend=stats.tokenizer_backend,
        )
    except TokenizerUnavailableError:
        print("无法使用缓存中的分词后端，回退至正则切分。请确保已安装 jieba 以获得一致的奖励评估。")
        tokenizer = LexicalTokenizer(
            stopwords=stats.stopwords,
            force_backend='regex',
        )
    return stats, tokenizer


def _load_lexical_statistics(
        article_path: Path,
) -> tuple[ChapterLexicalStatistics | None, LexicalTokenizer | None]:
    stats_path = _resolve_lexical_stats_path(article_path)
    if stats_path is None:
        return None, None
    return _load_lexical_statistics_from_path(stats_path)


def _ensure_lexical_statistics(
        article_path: Path,
        *,
        recompute: bool = False,
) -> tuple[ChapterLexicalStatistics | None, LexicalTokenizer | None]:
    stats_path = _resolve_lexical_stats_path(article_path)
    output_path = article_path.parent / f"{article_path.stem}{LEXICAL_STATS_SUFFIX}"
    if stats_path is None and output_path.exists():
        stats_path = output_path
    needs_compute = recompute or stats_path is None
    if not needs_compute and stats_path is not None:
        try:
            needs_compute = stats_path.stat().st_mtime < article_path.stat().st_mtime
        except OSError:
            needs_compute = True
        else:
            if not needs_compute:
                try:
                    relative = stats_path.relative_to(REPO_ROOT)
                except ValueError:
                    relative = stats_path
                print(f"检测到现有词频缓存：{relative}")
    if needs_compute:
        if not COMPUTE_TFIDF_SCRIPT.exists():
            try:
                rel_script = COMPUTE_TFIDF_SCRIPT.relative_to(REPO_ROOT)
            except ValueError:
                rel_script = COMPUTE_TFIDF_SCRIPT
            print(f"警告：未找到 {rel_script}，无法自动生成词频缓存。")
        else:
            cmd = [
                sys.executable,
                str(COMPUTE_TFIDF_SCRIPT),
                '--article-path',
                str(article_path),
                '--output',
                str(output_path),
            ]
            print('自动执行词频缓存生成：' + ' '.join(str(arg) for arg in cmd))
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                print(
                    f"词频缓存生成失败（退出码 {exc.returncode}），将继续尝试使用现有缓存。"
                )
            else:
                stats_path = output_path
    if stats_path is not None and stats_path.exists():
        return _load_lexical_statistics_from_path(stats_path)
    return _load_lexical_statistics(article_path)


def _run_inline_lexical_evaluation(
        lexical_stats: ChapterLexicalStatistics | None,
        lexical_tokenizer: LexicalTokenizer | None,
        chapter_index: int,
        summary_paths: Sequence[Path],
) -> None:
    if lexical_stats is None or lexical_tokenizer is None:
        print('警告：缺少词频缓存，跳过词频指标评估。')
        return
    chapter_entry = lexical_stats.chapter_by_index(chapter_index)
    top_tfidf = sorted(
        chapter_entry.tfidf.items(), key=lambda kv: kv[1], reverse=True
    )[:5]

    def _fmt(items: Sequence[tuple[str, float]]) -> str:
        if not items:
            return '<none>'
        return ', '.join(f'{token}:{score:.3f}' for token, score in items)

    print(
        f"词频参考 | 章节 {chapter_index:02d} tokens={chapter_entry.token_count} "
        f"top_tfidf={_fmt(top_tfidf)}"
    )
    for summary_path in summary_paths:
        try:
            summary_text = summary_path.read_text(encoding='utf-8')
        except OSError as exc:
            print(f"  摘要 {summary_path} 读取失败：{exc}，跳过。")
            continue
        vector = lexical_stats.vectorize_text(summary_text, lexical_tokenizer)
        cosine = cosine_similarity(vector.tfidf, chapter_entry.tfidf)
        js = jensen_shannon_similarity(
            vector.probability, chapter_entry.probability
        )
        summary_top = sorted(
            vector.tfidf.items(), key=lambda kv: kv[1], reverse=True
        )[:5]
        print('-' * 60)
        print(
            f"摘要 {summary_path} | tokens={vector.token_count} "
            f"cosine={cosine:.4f} js={js:.4f}"
        )
        print('  top_tfidf=' + _fmt(summary_top))


class ArticleEnvironment:
    """Environment emitting text observations and accepting text actions."""

    def __init__(
            self,
            chapters: Sequence[str],
            *,
            tokenizer: CharTokenizer,
            lexical_statistics: ChapterLexicalStatistics | None = None,
            lexical_tokenizer: LexicalTokenizer | None = None,
            initial_budget: float = DEFAULT_INITIAL_BUDGET,
            cost_weight: float = COST_WEIGHT,
    ) -> None:
        if not chapters:
            raise ValueError("The environment requires at least one chapter.")
        self._chapters = list(chapters)
        self._cursor = 0
        self._tokenizer = tokenizer
        self._word_checker = WordComplianceChecker(self._chapters)
        self._lexical_statistics = lexical_statistics
        if self._lexical_statistics is not None and lexical_tokenizer is None:
            try:
                lexical_tokenizer = LexicalTokenizer(
                    stopwords=self._lexical_statistics.stopwords,
                    force_backend=self._lexical_statistics.tokenizer_backend,
                )
            except TokenizerUnavailableError:
                lexical_tokenizer = LexicalTokenizer(
                    stopwords=self._lexical_statistics.stopwords,
                    force_backend="regex",
                )
        self._lexical_tokenizer = lexical_tokenizer
        self._valuator = CapitalValuator(self._chapters)
        self._initial_budget = float(initial_budget)
        self._cost_weight = float(cost_weight)
        self._capital = CognitiveCapital()
        self._budget = self._initial_budget
        self._cumulative_cost = 0.0
        self._current_summary = self._capital.render_text(self._budget)
        self._last_metrics: MutableMapping[str, float] = {}

    def reset(self) -> TextObservation:
        self._cursor = 0
        self._capital = CognitiveCapital()
        self._budget = self._initial_budget
        self._cumulative_cost = 0.0
        self._last_metrics = {}
        self._current_summary = self._capital.render_text(self._budget)
        return TextObservation(self._current_summary, self._chapters[0], 1)

    def step(self, action: TextAction) -> Transition:
        state = TextObservation(
            previous_summary=self._current_summary,
            chapter_text=self._chapters[self._cursor],
            step_index=self._cursor + 1,
        )
        source_text = _combine_summary_and_chapter(
            state.previous_summary, state.chapter_text
        )
        canonical_summary, operations = _canonicalize_action_text(action.text)
        metrics = analyze_summary(
            canonical_summary,
            source_text,
            tokenizer=self._tokenizer,
            word_checker=self._word_checker,
            chapter_text=state.chapter_text,
            chapter_index=state.step_index,
            lexical_stats=self._lexical_statistics,
            lexical_tokenizer=self._lexical_tokenizer,
        )
        capital_before = self._capital.clone()
        potential_before = self._valuator.potential(capital_before)
        step_cost = 0.0
        applied_operations = 0
        for operation in operations:
            op_kind = operation.kind.upper()
            cost = OPERATION_COSTS.get(op_kind, DEFAULT_OPERATION_COST)
            result = self._capital.apply(operation)
            if result.get("applied", False):
                applied_operations += 1
            step_cost += cost
        self._budget -= step_cost
        self._cumulative_cost += step_cost
        budget_breach = max(0.0, -self._budget)

        capital_metrics = self._valuator.metrics(self._capital)
        capital_value = self._valuator.value(self._capital)
        potential_after = self._valuator.potential(self._capital)
        potential_gain = potential_after - potential_before
        done = self._cursor + 1 >= len(self._chapters)
        cost_penalty = self._cost_weight * step_cost
        if done:
            cost_penalty = self._cost_weight * self._cumulative_cost
        base_reward = (
            capital_value
            - cost_penalty
            - BUDGET_PENALTY_WEIGHT * budget_breach
        )

        similarity_score = metrics.get("similarity", 0.0)
        coverage_score = metrics.get("coverage_ratio", 0.0)
        novelty_score = max(0.0, metrics.get("novelty_ratio", 0.0))
        lexical_cosine = metrics.get("lexical_cosine", 0.0)
        lexical_js = metrics.get("lexical_js_similarity", 0.0)
        garbled_ratio = metrics.get("garbled_ratio", 0.0)
        word_noncompliance = metrics.get("word_noncompliance_ratio", 0.0)

        quality_component = (
                _nonlinear_reward(similarity_score, QUALITY_NONLINEAR_EXPONENT) * QUALITY_SIMILARITY_WEIGHT
                + _nonlinear_reward(coverage_score, QUALITY_NONLINEAR_EXPONENT) * QUALITY_COVERAGE_WEIGHT
                + _nonlinear_reward(novelty_score, QUALITY_NONLINEAR_EXPONENT) * QUALITY_NOVELTY_WEIGHT
        )
        lexical_component = (
                _nonlinear_reward(lexical_cosine, LEXICAL_NONLINEAR_EXPONENT) * LEXICAL_SIMILARITY_WEIGHT
                + _nonlinear_reward(lexical_js, LEXICAL_NONLINEAR_EXPONENT) * LEXICAL_JS_WEIGHT
        )
        cleanliness_penalty = (
                _nonlinear_reward(garbled_ratio, CLEANLINESS_NONLINEAR_EXPONENT) * GARBLED_REWARD_WEIGHT
                + _nonlinear_reward(word_noncompliance, CLEANLINESS_NONLINEAR_EXPONENT) * WORD_COMPLIANCE_REWARD_WEIGHT
        )
        soft_reward = quality_component + lexical_component - cleanliness_penalty

        reward = base_reward + potential_gain + soft_reward
        next_summary = self._capital.render_text(self._budget)
        self._current_summary = next_summary
        self._cursor += 1
        if not done:
            next_state = TextObservation(
                previous_summary=self._current_summary,
                chapter_text=self._chapters[self._cursor],
                step_index=self._cursor + 1,
            )
        else:
            next_state = TextObservation(
                previous_summary=self._current_summary,
                chapter_text="",
                step_index=self._cursor + 1,
            )
        metrics.update({
            "capital_value": float(capital_value),
            "capital_operations": float(applied_operations),
            "operation_cost": float(step_cost),
            "budget_remaining": float(self._budget),
            "cumulative_cost": float(self._cumulative_cost),
            "budget_breach": float(budget_breach),
            "reward_base": float(base_reward),
            "reward_potential_gain": float(potential_gain),
            "reward_soft_bonus": float(soft_reward),
            "reward": reward,
            "canonical_summary_text": canonical_summary,
            "summary_raw_length": float(len(action.text)),
        })
        metrics["capital_coverage"] = capital_metrics["coverage"]
        metrics["capital_diversity"] = capital_metrics["diversity"]
        metrics["capital_redundancy"] = capital_metrics["redundancy"]
        metrics["capital_verification_ratio"] = capital_metrics["verification"]
        metrics["capital_fact_count"] = capital_metrics["fact_count"]
        self._last_metrics = metrics
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        return transition

    @property
    def last_metrics(self) -> MutableMapping[str, float]:
        return dict(self._last_metrics)

    @property
    def word_checker(self) -> WordComplianceChecker:
        return self._word_checker

    @property
    def lexical_statistics(self) -> ChapterLexicalStatistics | None:
        return self._lexical_statistics

    @property
    def lexical_tokenizer(self) -> LexicalTokenizer | None:
        return self._lexical_tokenizer


class SimpleReplayBuffer(BaseReplayBuffer):
    """In-memory FIFO replay buffer used solely for the demonstration."""

    def add(self, transition: Transition) -> None:
        if len(self._storage) >= self._capacity:
            self._storage.pop(0)
        self._storage.append(transition)

    def sample(self, batch_size: int) -> Iterable[Transition]:
        if not self._storage:
            return []
        size = min(len(self._storage), batch_size)
        return random.sample(self._storage, size)


class TextPolicyNetwork(nn.Module):
    """Stochastic policy operating on character token sequences."""

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            hidden_dim: int,
            max_summary_length: int,
            bos_token_id: int,
            eos_token_id: int,
            *,
            tokenizer: CharTokenizer | None = None,
            word_checker: WordComplianceChecker | None = None,
            compliance_temperature: float = DEFAULT_COMPLIANCE_TEMPERATURE,
            invalid_logit_penalty: float = COMPLIANCE_INVALID_LOGIT_PENALTY,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.max_summary_length = max_summary_length
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self._tokenizer_ref = tokenizer
        self._word_checker = word_checker
        self._compliance_temperature = max(1e-4, compliance_temperature)
        self._invalid_logit_penalty = abs(invalid_logit_penalty)

    def _adjust_logits_with_compliance(
            self,
            logits: torch.Tensor,
            prev_tokens: torch.Tensor,
            finished: torch.Tensor,
    ) -> torch.Tensor:
        if self._word_checker is None or self._tokenizer_ref is None:
            return logits
        summary_token_ids = self._tokenizer_ref.summary_token_ids
        if not summary_token_ids:
            return logits
        adjusted = logits.clone()
        eos_id = self.eos_token_id
        candidate_count = len(summary_token_ids) + 1
        for batch_idx in range(logits.size(0)):
            if finished[batch_idx]:
                continue
            prev_token_id = int(prev_tokens[batch_idx].item())
            prev_char = self._tokenizer_ref.token_from_id(prev_token_id)
            if prev_char in self._tokenizer_ref.special_tokens or len(prev_char) != 1:
                prev_char = None
            allowed_ids: set[int] = {eos_id}
            for token_id in summary_token_ids:
                candidate_char = self._tokenizer_ref.token_from_id(token_id)
                if len(candidate_char) != 1:
                    continue
                if self._word_checker.is_candidate_allowed(prev_char, candidate_char):
                    allowed_ids.add(token_id)
            if not allowed_ids:
                allowed_ids.add(eos_id)
            mask = torch.ones_like(adjusted[batch_idx], dtype=torch.bool)
            mask[list(allowed_ids)] = False
            if not torch.any(mask).item():
                continue
            adjusted[batch_idx, mask] = adjusted[batch_idx, mask] - self._invalid_logit_penalty
            if (
                    self._compliance_temperature < 1.0
                    and len(allowed_ids) < candidate_count
            ):
                allowed_tensor = torch.tensor(
                    list(allowed_ids), dtype=torch.long, device=adjusted.device
                )
                adjusted[batch_idx, allowed_tensor] = (
                        adjusted[batch_idx, allowed_tensor] / self._compliance_temperature
                )
        return adjusted

    def forward(
            self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, MutableMapping[str, torch.Tensor]]:
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.encoder(packed)
        batch_size = tokens.size(0)
        prev_tokens = torch.full(
            (batch_size,),
            fill_value=self.bos_token_id,
            dtype=torch.long,
            device=tokens.device,
        )
        outputs: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        hidden_state = hidden
        finished = torch.zeros(batch_size, dtype=torch.bool, device=tokens.device)
        for _ in range(self.max_summary_length):
            prev_emb = self.embedding(prev_tokens).unsqueeze(1)
            decoder_out, hidden_state = self.decoder(prev_emb, hidden_state)
            logits = self.output_layer(decoder_out.squeeze(1))
            logits = self._adjust_logits_with_compliance(logits, prev_tokens, finished)
            dist = Categorical(logits=logits)
            sampled = dist.sample()
            outputs.append(sampled)
            log_probs.append(dist.log_prob(sampled))
            finished = finished | sampled.eq(self.eos_token_id)
            prev_tokens = sampled
            if torch.all(finished):
                break
        action_tensor = torch.stack(outputs, dim=1)
        log_prob_tensor = torch.stack(log_probs, dim=1)
        mask = self._sequence_mask(action_tensor)
        log_prob = (log_prob_tensor * mask).sum(dim=-1, keepdim=True)
        info: MutableMapping[str, torch.Tensor] = {
            "log_prob": log_prob,
            "action_lengths": mask.sum(dim=-1),
        }
        return action_tensor, info

    def deterministic(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.encoder(packed)
        batch_size = tokens.size(0)
        prev_tokens = torch.full(
            (batch_size,),
            fill_value=self.bos_token_id,
            dtype=torch.long,
            device=tokens.device,
        )
        outputs: list[torch.Tensor] = []
        hidden_state = hidden
        finished = torch.zeros(batch_size, dtype=torch.bool, device=tokens.device)
        for _ in range(self.max_summary_length):
            prev_emb = self.embedding(prev_tokens).unsqueeze(1)
            decoder_out, hidden_state = self.decoder(prev_emb, hidden_state)
            logits = self.output_layer(decoder_out.squeeze(1))
            logits = self._adjust_logits_with_compliance(logits, prev_tokens, finished)
            chosen = torch.argmax(logits, dim=-1)
            outputs.append(chosen)
            finished = finished | chosen.eq(self.eos_token_id)
            prev_tokens = chosen
            if torch.all(finished):
                break
        if outputs:
            return torch.stack(outputs, dim=1)
        return torch.empty((batch_size, 0), dtype=torch.long, device=tokens.device)

    def _sequence_mask(self, samples: torch.Tensor) -> torch.Tensor:
        eos_hits = (samples == self.eos_token_id).int()
        cumulative = torch.cumsum(eos_hits, dim=-1)
        mask = (cumulative <= 1).float()
        return mask

    def infer_lengths(self, samples: torch.Tensor) -> torch.Tensor:
        mask = self._sequence_mask(samples)
        lengths = mask.sum(dim=-1).long()
        return torch.clamp(lengths, min=1)


class TextQNetwork(nn.Module):
    """Lightweight Q-network aggregating token embeddings without recurrent loops."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.state_proj = nn.Linear(embedding_dim, hidden_dim)
        self.action_proj = nn.Linear(embedding_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _masked_mean(self, embeddings: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(0).unsqueeze(-1).float()
        masked = embeddings * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(
            self,
            state_tokens: torch.Tensor,
            state_lengths: torch.Tensor,
            action_tokens: torch.Tensor,
            action_lengths: torch.Tensor,
    ) -> torch.Tensor:
        del state_lengths, action_lengths  # lengths are implicit in the masking
        state_embedded = self.embedding(state_tokens)
        action_embedded = self.embedding(action_tokens)
        state_summary = torch.tanh(self.state_proj(self._masked_mean(state_embedded, state_tokens)))
        action_summary = torch.tanh(
            self.action_proj(self._masked_mean(action_embedded, action_tokens))
        )
        combined = torch.cat([state_summary, action_summary], dim=-1)
        return self.head(combined)


@dataclass
class DemoNetworkFactory(NetworkFactory):
    """Factory returning PyTorch networks sized for the demonstration."""

    policy_builder: Any | None = field(default=None, init=False, repr=False)
    q1_builder: Any | None = field(default=None, init=False, repr=False)
    q2_builder: Any | None = field(default=None, init=False, repr=False)
    vocab_size: int
    embedding_dim: int
    hidden_dim: int
    max_summary_length: int
    bos_token_id: int
    eos_token_id: int
    compliance_temperature: float = DEFAULT_COMPLIANCE_TEMPERATURE
    invalid_logit_penalty: float = COMPLIANCE_INVALID_LOGIT_PENALTY

    def build_policy(
            self,
            *args: Any,
            tokenizer: CharTokenizer | None = None,
            word_checker: WordComplianceChecker | None = None,
            compliance_temperature: float | None = None,
            invalid_logit_penalty: float | None = None,
            **kwargs: Any,
    ) -> TextPolicyNetwork:
        temperature = (
            compliance_temperature
            if compliance_temperature is not None
            else self.compliance_temperature
        )
        penalty = (
            invalid_logit_penalty
            if invalid_logit_penalty is not None
            else self.invalid_logit_penalty
        )
        return TextPolicyNetwork(
            self.vocab_size,
            self.embedding_dim,
            self.hidden_dim,
            self.max_summary_length,
            self.bos_token_id,
            self.eos_token_id,
            tokenizer=tokenizer,
            word_checker=word_checker,
            compliance_temperature=temperature,
            invalid_logit_penalty=penalty,
        )

    def build_q_functions(
            self, *args: Any, **kwargs: Any
    ) -> tuple[TextQNetwork, TextQNetwork]:
        return (
            TextQNetwork(self.vocab_size, self.embedding_dim, self.hidden_dim),
            TextQNetwork(self.vocab_size, self.embedding_dim, self.hidden_dim),
        )


class DemoSACAgent(SACAgent):
    """Concrete SAC agent operating on text observations and actions."""

    def __init__(
            self,
            policy: TextPolicyNetwork,
            q1: TextQNetwork,
            q2: TextQNetwork,
            target_q1: TextQNetwork,
            target_q2: TextQNetwork,
            replay_buffer: BaseReplayBuffer,
            config: AgentConfig,
            *,
            tokenizer: CharTokenizer,
            update_batch_size: int = 4,
            device: str = "cpu",
    ) -> None:
        super().__init__(policy, q1, q2, target_q1, target_q2, replay_buffer, config)
        self.tokenizer = tokenizer
        self.update_batch_size = update_batch_size
        self.device = torch.device(device)
        self.device_str = str(self.device)
        self.policy.to(self.device)
        self.q1.to(self.device)
        self.q2.to(self.device)
        self.target_q1.to(self.device)
        self.target_q2.to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        for target in (self.target_q1, self.target_q2):
            for parameter in target.parameters():
                parameter.requires_grad = False
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=3e-4)
        self.alpha = self.config.alpha
        self.parameter_count = sum(
            parameter.numel() for parameter in self.policy.parameters()
        )
        self.model_size_bytes = MODEL_SIZE_BYTES

    def _encode_observation(self, observation: TextObservation) -> List[int]:
        return self.tokenizer.encode_observation(observation)

    def _encode_action(self, action: TextAction) -> List[int]:
        return action.token_ids

    def act(self, state: TextObservation, deterministic: bool = False) -> TextAction:
        tokens, lengths = self.tokenizer.batch_encode(
            [self._encode_observation(state)], device=self.device
        )
        with torch.no_grad():
            if deterministic:
                action_ids = self.policy.deterministic(tokens, lengths)
                action_lengths = self.policy.infer_lengths(action_ids)
            else:
                action_ids, info = self.policy(tokens, lengths)
                action_lengths = info["action_lengths"].long()
        token_ids = action_ids.squeeze(0).cpu().tolist()
        length = int(action_lengths.squeeze(0).item())
        text = self.tokenizer.decode_action(token_ids)
        return TextAction(token_ids=token_ids, text=text, length=length)

    def update(self) -> MutableMapping[str, float]:
        if len(self.replay_buffer) == 0:
            return {
                "policy_loss": 0.0,
                "q1_loss": 0.0,
                "q2_loss": 0.0,
                "average_reward": 0.0,
            }

        batch = list(self.replay_buffer.sample(self.update_batch_size))
        states = [self._encode_observation(transition.state) for transition in batch]
        actions = [self._encode_action(transition.action) for transition in batch]
        rewards = torch.tensor(
            [transition.reward for transition in batch],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(-1)
        next_states = [self._encode_observation(transition.next_state) for transition in batch]
        dones = torch.tensor(
            [transition.done for transition in batch],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(-1)

        state_tokens, state_lengths = self.tokenizer.batch_encode(
            states, device=self.device
        )
        action_tokens, action_lengths = self.tokenizer.batch_encode(
            actions, device=self.device
        )
        next_state_tokens, next_state_lengths = self.tokenizer.batch_encode(
            next_states, device=self.device
        )

        with torch.no_grad():
            next_action_tokens, next_info = self.policy(next_state_tokens, next_state_lengths)
            next_action_lengths = next_info["action_lengths"].long().clamp(min=1)
            target_q1 = self.target_q1(
                next_state_tokens,
                next_state_lengths,
                next_action_tokens,
                next_action_lengths,
            )
            target_q2 = self.target_q2(
                next_state_tokens,
                next_state_lengths,
                next_action_tokens,
                next_action_lengths,
            )
            target_value = torch.min(target_q1, target_q2) - self.alpha * next_info[
                "log_prob"
            ]
            target_q = rewards + self.config.gamma * (1.0 - dones) * target_value

        current_q1 = self.q1(state_tokens, state_lengths, action_tokens, action_lengths)
        current_q2 = self.q2(state_tokens, state_lengths, action_tokens, action_lengths)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        for parameter in self.q1.parameters():
            parameter.requires_grad_(False)
        new_action_tokens, policy_info = self.policy(state_tokens, state_lengths)
        new_action_lengths = policy_info["action_lengths"].long().clamp(min=1)
        q1_for_policy = self.q1(
            state_tokens,
            state_lengths,
            new_action_tokens,
            new_action_lengths,
        )
        policy_loss = (
                self.alpha * policy_info["log_prob"] - q1_for_policy
        ).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        for parameter in self.q1.parameters():
            parameter.requires_grad_(True)

        with torch.no_grad():
            for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
                target_param.copy_(
                    self.config.tau * param + (1 - self.config.tau) * target_param
                )
            for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
                target_param.copy_(
                    self.config.tau * param + (1 - self.config.tau) * target_param
                )

        average_reward = rewards.mean().item()
        return {
            "policy_loss": float(policy_loss.item()),
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "average_reward": average_reward,
        }

    def save(self, destination: MutableMapping[str, Any]) -> None:  # pragma: no cover - placeholder
        weights: List[float] = []
        for tensor in self.policy.state_dict().values():
            weights.extend(tensor.detach().cpu().reshape(-1).tolist())
        weights = weights[: self.parameter_count]
        destination.update(
            {
                "device": self.device_str,
                "model_size_bytes": self.model_size_bytes,
                "policy_state": {
                    "parameter_count": self.parameter_count,
                    "weights": weights,
                },
            }
        )

    def load(self, source: MutableMapping[str, Any]) -> None:  # pragma: no cover - placeholder
        _ = source

    @classmethod
    def from_factory(
            cls,
            factory: DemoNetworkFactory,
            replay_buffer: BaseReplayBuffer,
            config: AgentConfig,
            *,
            tokenizer: CharTokenizer,
            word_checker: WordComplianceChecker | None = None,
            update_batch_size: int = 4,
            device: str = "cpu",
    ) -> "DemoSACAgent":
        policy = factory.build_policy(
            tokenizer=tokenizer, word_checker=word_checker
        )
        q1, q2 = factory.build_q_functions()
        target_q1, target_q2 = factory.build_q_functions()
        return cls(
            policy,
            q1,
            q2,
            target_q1,
            target_q2,
            replay_buffer,
            config,
            tokenizer=tokenizer,
            update_batch_size=update_batch_size,
            device=device,
        )


class DemoTrainer(Trainer):
    """Trainer that runs rollouts for the iterative text summarization demo."""

    def __init__(
            self,
            agent: DemoSACAgent,
            environment: ArticleEnvironment,
            config: TrainerConfig,
            *,
            intervals: Sequence[str],
            logger: MutableMapping[str, Any] | None = None,
            reference_actions: Mapping[int, str] | None = None,
            reference_warmup_rounds: int = DEFAULT_REFERENCE_WARMUP_ROUNDS,
            reference_warmup_steps: int = DEFAULT_REFERENCE_WARMUP_STEPS,
    ) -> None:
        super().__init__(agent, environment, config, logger)
        self._intervals = list(intervals)
        if not self._intervals:
            raise ValueError("Intervals cannot be empty for the trainer.")
        self._reference_actions = dict(reference_actions or {})
        self._reference_warmup_rounds = max(0, reference_warmup_rounds)
        self._reference_warmup_steps = max(0, reference_warmup_steps)

    def run(self, *, round_index: int = 1) -> None:
        state = self.environment.reset()
        total_steps = len(self._intervals)
        if self.config.total_steps != total_steps:
            print(
                "Adjusting total steps to match interval segments: "
                f"{self.config.total_steps} -> {total_steps}"
            )
            self.config.total_steps = total_steps
        print(f"=== Training round {round_index} | steps={total_steps} ===")
        total_reward = 0.0
        for step in range(1, total_steps + 1):
            prev_len, prev_preview = _format_text_debug(state.previous_summary, 20, 20)
            chapter_len, chapter_preview = _format_text_debug(state.chapter_text, 20, 20)
            source_text = _combine_summary_and_chapter(
                state.previous_summary, state.chapter_text
            )
            source_len, source_preview = _format_text_debug(source_text, 20, 20)
            block_color = ANSI_YELLOW

            def _colorize(line: str) -> str:
                return f"{block_color}{line}{ANSI_RESET}"

            lines_to_print: list[str] = []
            lines_to_print.append(
                f"  Step {step:02d} | prev_summary={prev_len:04d} chars \"{prev_preview}\""
            )
            lines_to_print.append(
                f"           | chapter={chapter_len:04d} chars \"{chapter_preview}\""
            )
            lines_to_print.append(
                f"           | source={source_len:04d} chars \"{source_preview}\""
            )

            global_step = (round_index - 1) * total_steps + step
            reference_available = (
                self._reference_actions
                and state.step_index in self._reference_actions
            )
            warmup_round_active = (
                self._reference_warmup_rounds > 0
                and round_index <= self._reference_warmup_rounds
            )
            warmup_step_active = (
                self._reference_warmup_steps > 0
                and global_step <= self._reference_warmup_steps
            )
            use_reference = reference_available and (
                warmup_round_active or warmup_step_active
            )
            if use_reference:
                reference_text = self._reference_actions[state.step_index]
                action = _create_text_action(reference_text, self.agent.tokenizer)
                source_reason = "warmup-round" if warmup_round_active else "warmup-step"
                lines_to_print.append(
                    f"           | action_source=reference-template ({source_reason})"
                )
            else:
                action = self.agent.act(state)
                lines_to_print.append("           | action_source=policy")
            transition = self.environment.step(action)
            self.agent.record(transition)
            metrics = self.environment.last_metrics

            canonical_summary_text = metrics.get("canonical_summary_text", action.text)
            summary_len, summary_preview = _format_text_debug(canonical_summary_text, 20, 20)
            raw_summary_len, raw_summary_preview = _format_text_debug(action.text, 20, 20)
            if raw_summary_preview != summary_preview:
                lines_to_print.append(
                    f"           | raw_action={raw_summary_len:04d} chars \"{raw_summary_preview}\""
                )
            total_reward += transition.reward
            summary_length_value = float(metrics.get("summary_length", summary_len))
            log_metrics: MutableMapping[str, Any] = {
                "reward": transition.reward,
                "reward_base": metrics.get("reward_base", 0.0),
                "reward_potential_gain": metrics.get("reward_potential_gain", 0.0),
                "reward_soft_bonus": metrics.get("reward_soft_bonus", 0.0),
                "buffer_size": len(self.agent.replay_buffer),
                "summary_length": summary_length_value,
                "source_length": metrics.get("source_length", float(source_len)),
                "length_ratio": metrics.get("length_ratio", 0.0),
                "similarity": metrics.get("similarity", 0.0),
                "coverage_ratio": metrics.get("coverage_ratio", 0.0),
                "novelty_ratio": metrics.get("novelty_ratio", 0.0),
                "garbled_ratio": metrics.get("garbled_ratio", 0.0),
                "garbled_penalty": metrics.get("garbled_penalty", 0.0),
                "word_noncompliance_ratio": metrics.get("word_noncompliance_ratio", 0.0),
                "word_penalty": metrics.get("word_penalty", 0.0),
                "unk_char_ratio": metrics.get("unk_char_ratio", 0.0),
                "disallowed_char_ratio": metrics.get("disallowed_char_ratio", 0.0),
                "control_char_ratio": metrics.get("control_char_ratio", 0.0),
                "lexical_cosine": metrics.get("lexical_cosine", 0.0),
                "lexical_js_similarity": metrics.get("lexical_js_similarity", 0.0),
                "lexical_token_count": metrics.get("lexical_token_count", 0.0),
            }
            summary_line = (
                f"           -> summary={summary_len:04d} chars \"{summary_preview}\""
            )

            metric_indent = "           "
            metric_descriptions: list[tuple[str, str, str]] = [
                ("reward_base", "reward_base", "基础奖励，结合资本价值与预算成本"),
                ("reward_potential", "reward_potential_gain", "潜在价值增量"),
                ("reward_soft", "reward_soft_bonus", "摘要质量软奖励"),
                ("len_ratio", "length_ratio", "摘要长度与信息源比值，偏低会导致覆盖不足"),
                ("sim", "similarity", "字符级相似度，衡量摘要整体贴近原文的程度"),
                ("coverage", "coverage_ratio", "覆盖率，统计摘要覆盖原文字符的比例"),
                ("novelty", "novelty_ratio", "新颖度，越高表示抄写成分越少"),
                ("lex_cos", "lexical_cosine", "章节 TF-IDF 余弦相似度，反映高权重词是否匹配"),
                ("lex_js", "lexical_js_similarity", "词频 Jensen-Shannon 相似度，衡量整体词频结构的接近程度"),
                ("garbled", "garbled_ratio", "乱码比率，非法或不可打印字符占比"),
                ("word_nc", "word_noncompliance_ratio", "词合规缺失率，识别异常汉字或未见过的双字组合"),
                ("cap_val", "capital_value", "认知资本估值，综合覆盖、多样与验证实力"),
                ("cap_cov", "capital_coverage", "认知覆盖率，衡量知识对主题的覆盖"),
                ("cap_div", "capital_diversity", "认知多样性，反映事实主题的广度"),
                ("cap_red", "capital_redundancy", "冗余度，越低越精炼"),
                ("cap_ver", "capital_verification_ratio", "验证比例，越高越可靠"),
                ("budget", "budget_remaining", "剩余预算，反映资源消耗进度"),
            ]
            metric_lines: list[str] = []
            for label, key, description in metric_descriptions:
                value = float(log_metrics.get(key, 0.0))
                quality = _describe_metric_quality(key, value)
                metric_lines.append(
                    f"{metric_indent}{label}={value:.3f} （{description}；{quality}）"
                )

            garbled_penalty = float(log_metrics["garbled_penalty"])
            word_penalty = float(log_metrics["word_penalty"])
            penalty_quality = "；".join(
                [
                    _describe_penalty_component(garbled_penalty, "乱码惩罚"),
                    _describe_penalty_component(word_penalty, "词合规惩罚"),
                ]
            )
            penalty_line = (
                f"{metric_indent}penalties={garbled_penalty:.3f}/{word_penalty:.3f} "
                f"（乱码与词合规惩罚项，越高惩罚越重；{penalty_quality}）"
            )

            reward_quality = _describe_reward_quality(transition.reward)
            if transition.reward > 1e-6:
                block_color = ANSI_GREEN
            elif transition.reward < -1e-6:
                block_color = ANSI_RED
            else:
                block_color = ANSI_YELLOW
            reward_line = (
                f"{metric_indent}reward={transition.reward:.3f} "
                f"(base={metrics.get('reward_base', 0.0):+.3f}, "
                f"potential={metrics.get('reward_potential_gain', 0.0):+.3f}, "
                f"soft={metrics.get('reward_soft_bonus', 0.0):+.3f}; {reward_quality})"
            )

            lines_to_print.append(summary_line)
            lines_to_print.extend(metric_lines)
            lines_to_print.append(penalty_line)
            lines_to_print.append(reward_line)
            _append_step_log(lines_to_print, block_color)
            for stored_line in lines_to_print:
                _console_log(stored_line, color=block_color)

            if log_metrics:
                self.log(log_metrics, global_step)
            step_csv_row = {
                "round": round_index,
                "step": step,
                "global_step": global_step,
                "reward": transition.reward,
                "reward_base": log_metrics.get("reward_base", 0.0),
                "reward_potential_gain": log_metrics.get("reward_potential_gain", 0.0),
                "reward_soft_bonus": log_metrics.get("reward_soft_bonus", 0.0),
                "previous_summary_length": prev_len,
                "chapter_length": chapter_len,
                "source_length": source_len,
                "summary_length": int(summary_length_value),
                "length_ratio": log_metrics.get("length_ratio", 0.0),
                "similarity": log_metrics.get("similarity", 0.0),
                "coverage_ratio": log_metrics.get("coverage_ratio", 0.0),
                "novelty_ratio": log_metrics.get("novelty_ratio", 0.0),
                "garbled_ratio": log_metrics.get("garbled_ratio", 0.0),
                "garbled_penalty": log_metrics.get("garbled_penalty", 0.0),
                "word_noncompliance_ratio": log_metrics.get("word_noncompliance_ratio", 0.0),
                "word_penalty": log_metrics.get("word_penalty", 0.0),
                "unk_char_ratio": log_metrics.get("unk_char_ratio", 0.0),
                "disallowed_char_ratio": log_metrics.get("disallowed_char_ratio", 0.0),
                "control_char_ratio": log_metrics.get("control_char_ratio", 0.0),
                "lexical_cosine": log_metrics.get("lexical_cosine", 0.0),
                "lexical_js_similarity": log_metrics.get("lexical_js_similarity", 0.0),
                "lexical_token_count": log_metrics.get("lexical_token_count", 0.0),
                "capital_value": metrics.get("capital_value", 0.0),
                "capital_coverage": metrics.get("capital_coverage", 0.0),
                "capital_diversity": metrics.get("capital_diversity", 0.0),
                "capital_redundancy": metrics.get("capital_redundancy", 0.0),
                "capital_verification_ratio": metrics.get("capital_verification_ratio", 0.0),
                "capital_fact_count": metrics.get("capital_fact_count", 0.0),
                "capital_operations": metrics.get("capital_operations", 0.0),
                "operation_cost": metrics.get("operation_cost", 0.0),
                "budget_remaining": metrics.get("budget_remaining", 0.0),
                "budget_breach": metrics.get("budget_breach", 0.0),
                "cumulative_cost": metrics.get("cumulative_cost", 0.0),
            }
            _append_csv_row(STEP_CSV_PATH, STEP_CSV_HEADERS, step_csv_row)
            state = transition.next_state
            if transition.done:
                steps_completed = step
                round_total = total_reward
                print(
                    f"=== Training round {round_index} complete | "
                    f"total_reward={round_total:.2f} ==="
                )
                round_csv_row = {
                    "round": round_index,
                    "steps": steps_completed,
                    "total_reward": round_total,
                    "average_reward": round_total / steps_completed if steps_completed else 0.0,
                }
                _append_csv_row(ROUND_CSV_PATH, ROUND_CSV_HEADERS, round_csv_row)
                snapshot_metadata = {
                    "round": round_index,
                    "steps_completed": steps_completed,
                    "total_reward": round_total,
                    "average_reward": round_csv_row["average_reward"],
                    "global_step": global_step,
                    "replay_buffer_size": len(self.agent.replay_buffer),
                    "updates_per_round": self.config.updates_per_round,
                }
                snapshot_path = ROUND_SNAPSHOT_DIR / (
                    f"demo_agent_snapshot_round_{round_index:04d}.json"
                )
                save_agent_snapshot(self.agent, snapshot_metadata, snapshot_path)
                print(
                    "           "
                    f"Saved round snapshot to {snapshot_path.relative_to(REPO_ROOT)}"
                )
                total_reward = 0.0
                state = self.environment.reset()
        if (
                self.config.updates_per_round > 0
                and len(self.agent.replay_buffer) >= self.config.batch_size
        ):
            print(
                f"=== Post-round updates (round {round_index}) "
                f"x{self.config.updates_per_round} ==="
            )
            post_round_metrics: list[MutableMapping[str, float]] = []
            for update_idx in range(1, self.config.updates_per_round + 1):
                update_metrics = self.agent.update()
                post_round_metrics.append(update_metrics)
                print(
                    f"    Update {update_idx:03d} | "
                    f"policy_loss={update_metrics.get('policy_loss', float('nan')):.4f} "
                    f"q1_loss={update_metrics.get('q1_loss', float('nan')):.4f} "
                    f"q2_loss={update_metrics.get('q2_loss', float('nan')):.4f} "
                    f"avg_reward={update_metrics.get('average_reward', float('nan')):.4f}"
                )
            aggregated: MutableMapping[str, float] = {}
            for key in {key for metrics in post_round_metrics for key in metrics}:
                values = [metrics[key] for metrics in post_round_metrics if key in metrics]
                if values:
                    aggregated[key] = statistics.fmean(values)
            if aggregated:
                summary_metrics = {f"post_round_{key}": value for key, value in aggregated.items()}
                summary_step = round_index * total_steps
                self.log(summary_metrics, summary_step)
                print(
                    "    Post-round metric averages | "
                    + " ".join(
                        f"{key}={value:.4f}" for key, value in aggregated.items()
                    )
                )

    def render_iterative_summary(self) -> List[str]:
        """Render iterative capital accrual generated by the deterministic policy."""

        initial_budget = getattr(self.environment, "_initial_budget", DEFAULT_INITIAL_BUDGET)
        valuator = getattr(self.environment, "_valuator", CapitalValuator(self._intervals))
        capital = CognitiveCapital()
        budget = float(initial_budget)
        cumulative_cost = 0.0
        rendered_iterations: List[str] = [
            f"Iteration 00 | capital_value=0.000 budget={budget:.1f} | <empty>"
        ]
        for idx, chapter in enumerate(self._intervals, start=1):
            observation = TextObservation(
                previous_summary=capital.render_text(budget),
                chapter_text=chapter,
                step_index=idx,
            )
            action = self.agent.act(observation, deterministic=True)
            operations = OperationParser.parse(action.text)
            step_cost = 0.0
            for operation in operations:
                op_kind = operation.kind.upper()
                cost = OPERATION_COSTS.get(op_kind, DEFAULT_OPERATION_COST)
                capital.apply(operation)
                step_cost += cost
            budget -= step_cost
            cumulative_cost += step_cost
            capital_value = valuator.value(capital)
            capital_metrics = valuator.metrics(capital)
            preview_source = action.text.replace("\n", " ")
            preview = preview_source[:48] + ("..." if len(preview_source) > 48 else "")
            rendered_iterations.append(
                f"Iteration {idx:02d} | capital_value={capital_value:.3f} "
                f"budget={budget:.1f} cost={step_cost:.2f} "
                f"coverage={capital_metrics['coverage']:.2f} "
                f"diversity={capital_metrics['diversity']:.2f} "
                f"redundancy={capital_metrics['redundancy']:.2f} | {preview}"
            )
        return rendered_iterations

    def _print_iterative_summary(self, step: int, round_index: int) -> None:
        print(
            "  Iterative distillation summary after "
            f"round {round_index} step {step:02d}:"
        )
        for line in self.render_iterative_summary():
            print(f"    {line}")


def _normalize_fact_snippet(text: str, max_chars: int = 120) -> str:
    collapsed = " ".join(text.strip().split())
    return collapsed[:max_chars]


def _extract_candidate_sentences(chapter_text: str, max_sentences: int = 3) -> list[str]:
    sentences = re.split(r"[。！？!?\.]+", chapter_text)
    candidates: list[str] = []
    for sentence in sentences:
        normalized = _normalize_fact_snippet(sentence)
        if len(normalized) >= 12:
            candidates.append(normalized)
        if len(candidates) >= max_sentences:
            break
    if not candidates:
        fallback = _normalize_fact_snippet(chapter_text)
        if fallback:
            candidates.append(fallback)
    return candidates


def _build_template_action(chapter_text: str, chapter_index: int) -> str:
    sentences = _extract_candidate_sentences(chapter_text, max_sentences=3)
    if not sentences:
        return ""
    commands: list[str] = []
    fact_label = f"CH{chapter_index:02d}"
    primary = sentences[0]
    commands.append(f"ACQUIRE {fact_label} {primary}")
    if len(sentences) > 1:
        secondary = sentences[1]
        commands.append(f"VERIFY {fact_label} {secondary}")
        left = primary[:60]
        right = secondary[:60]
        commands.append(f"LINK {left} -> {right}")
    if len(sentences) > 2:
        tertiary = sentences[2]
        commands.append(f"COMMIT {fact_label} {tertiary}")
    return "".join(cmd for cmd in commands if cmd.strip())


def _create_text_action(action_text: str, tokenizer: CharTokenizer) -> TextAction:
    token_ids = tokenizer.encode_action_text(action_text)
    return TextAction(token_ids=token_ids, text=action_text, length=len(token_ids))


def _create_text_action(action_text: str, tokenizer: CharTokenizer) -> TextAction:
    token_ids = tokenizer.encode_action_text(action_text)
    return TextAction(token_ids=token_ids, text=action_text, length=len(token_ids))


def _seed_replay_buffer_with_templates(
        environment: ArticleEnvironment,
        replay_buffer: BaseReplayBuffer,
        tokenizer: CharTokenizer,
        observations: Sequence[TextObservation],
        *,
        max_seed_steps: int = 4,
) -> list[tuple[str, float]]:
    if max_seed_steps <= 0:
        return []
    seeds: list[tuple[str, float]] = []
    environment.reset()
    for index, observation in enumerate(observations[:max_seed_steps], start=1):
        template_text = _build_template_action(observation.chapter_text, observation.step_index)
        if not template_text.strip():
            continue
        action = _create_text_action(template_text, tokenizer)
        transition = environment.step(action)
        replay_buffer.add(transition)
        seeds.append((template_text, transition.reward))
        if transition.done:
            break
    environment.reset()
    return seeds


def _create_text_action(action_text: str, tokenizer: CharTokenizer) -> TextAction:
    token_ids = tokenizer.encode_action_text(action_text)
    return TextAction(token_ids=token_ids, text=action_text, length=len(token_ids))


def _load_reference_actions(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    chapter_pattern = re.compile(r"^Chapter\s+(\d{1,3})\b")
    actions: dict[int, str] = {}
    current_idx: int | None = None
    buffer: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = chapter_pattern.match(raw_line)
        if match:
            if current_idx is not None and buffer:
                actions[current_idx] = "\n".join(buffer).strip()
            current_idx = int(match.group(1))
            buffer = []
            continue
        if raw_line.startswith("  "):
            stripped = raw_line.strip()
            if stripped:
                buffer.append(stripped)
            continue
        if buffer and raw_line.strip() == "":
            # blank line denotes end of current block
            if current_idx is not None:
                actions[current_idx] = "\n".join(buffer).strip()
            current_idx = None
            buffer = []
    if current_idx is not None and buffer:
        actions[current_idx] = "\n".join(buffer).strip()
    return actions


def _seed_replay_buffer_with_templates(
        environment: ArticleEnvironment,
        replay_buffer: BaseReplayBuffer,
        tokenizer: CharTokenizer,
        observations: Sequence[TextObservation],
        *,
        max_seed_steps: int = 4,
) -> list[tuple[str, float]]:
    if max_seed_steps <= 0:
        return []
    seeds: list[tuple[str, float]] = []
    environment.reset()
    for index, observation in enumerate(observations[:max_seed_steps], start=1):
        template_text = _build_template_action(observation.chapter_text, observation.step_index)
        if not template_text.strip():
            continue
        action = _create_text_action(template_text, tokenizer)
        transition = environment.step(action)
        replay_buffer.add(transition)
        seeds.append((template_text, transition.reward))
        if transition.done:
            break
    environment.reset()
    return seeds


def build_demo_components(
        article_path: Path,
        capacity: int,
        *,
        precomputed: Sequence[TextObservation] | None = None,
        lexical_stats: ChapterLexicalStatistics | None = None,
        lexical_tokenizer: LexicalTokenizer | None = None,
        training_config: Mapping[str, Any] | None = None,
) -> tuple[DemoSACAgent, DemoTrainer]:
    if precomputed is None:
        observations = load_article_features(article_path)
    else:
        observations = list(precomputed)
    chapters = [ob.chapter_text for ob in observations]
    if training_config is None:
        training_config, _ = _load_training_config()
    reference_actions_path = Path(
        training_config["reference_actions_path"]
    )
    if not reference_actions_path.is_absolute():
        reference_actions_path = REPO_ROOT / reference_actions_path
    reference_actions = _load_reference_actions(reference_actions_path)
    tokenizer_corpus = list(chapters)
    tokenizer_corpus.extend(reference_actions.values())
    tokenizer_corpus.append(CognitiveCapital().render_text(DEFAULT_INITIAL_BUDGET))
    common_charset = _compute_common_summary_charset(article_path)
    reference_charset = {
        char
        for text in reference_actions.values()
        for char in text
        if _is_cjk(char)
    }
    combined_charset = set(common_charset or [])
    combined_charset.update(reference_charset)
    tokenizer = CharTokenizer(
        tokenizer_corpus,
        summary_charset=combined_charset or None,
        punctuation_whitelist=COMMON_SUMMARY_PUNCTUATION,
    )
    if lexical_stats is None or lexical_tokenizer is None:
        fallback_stats, fallback_tokenizer = _load_lexical_statistics(article_path)
        lexical_stats = lexical_stats or fallback_stats
        lexical_tokenizer = lexical_tokenizer or fallback_tokenizer
        if lexical_stats is None:
            stats_name = f"{article_path.stem}{LEXICAL_STATS_SUFFIX}"
            print(f"警告：未找到 {stats_name}，词频奖励将被禁用。")
    max_summary_length = max(64, min(512, max(len(chapter) for chapter in chapters)))
    environment = ArticleEnvironment(
        chapters,
        tokenizer=tokenizer,
        lexical_statistics=lexical_stats,
        lexical_tokenizer=lexical_tokenizer,
        initial_budget=DEFAULT_INITIAL_BUDGET,
        cost_weight=COST_WEIGHT,
    )
    replay_buffer = SimpleReplayBuffer(capacity)
    seeded_samples = _seed_replay_buffer_with_templates(
        environment,
        replay_buffer,
        tokenizer,
        observations,
        max_seed_steps=min(4, len(observations)),
    )
    if seeded_samples:
        _console_log(
            f"Seeded replay buffer with {len(seeded_samples)} template transitions"
        )
        for seed_index, (template_text, reward_value) in enumerate(seeded_samples, start=1):
            preview = template_text.splitlines()[0][:80]
            _console_log(
                f"  Seed {seed_index:02d}: reward={reward_value:.3f} | {preview}"
            )
    network_factory = DemoNetworkFactory(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=96,
        hidden_dim=128,
        max_summary_length=max_summary_length,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        compliance_temperature=DEFAULT_COMPLIANCE_TEMPERATURE,
        invalid_logit_penalty=COMPLIANCE_INVALID_LOGIT_PENALTY,
    )
    agent_config = AgentConfig()
    agent = DemoSACAgent.from_factory(
        network_factory,
        replay_buffer,
        agent_config,
        tokenizer=tokenizer,
        word_checker=environment.word_checker,
        update_batch_size=1,
        device='cpu',
    )
    steps_per_round = len(chapters)
    trainer_config = TrainerConfig(
        total_steps=steps_per_round,
        warmup_steps=0,
        batch_size=1,
        updates_per_step=0,
        updates_per_round=steps_per_round,
    )
    trainer = DemoTrainer(
        agent,
        environment,
        trainer_config,
        intervals=chapters,
        reference_actions=reference_actions,
        reference_warmup_rounds=training_config["reference_warmup_rounds"],
        reference_warmup_steps=training_config["reference_warmup_steps"],
    )
    return agent, trainer


def save_model_artifact(path: Path, size: int) -> None:
    """Persist a deterministic binary blob representing the trained model."""

    path.parent.mkdir(exist_ok=True)
    pattern = bytes(range(256))
    with path.open("wb") as fh:
        full_chunks, remainder = divmod(size, len(pattern))
        for _ in range(full_chunks):
            fh.write(pattern)
        if remainder:
            fh.write(pattern[:remainder])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SAC scaffolding demo")
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=32,
        help="Maximum number of transitions stored in the replay buffer.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1000,
        help="Number of training rounds to execute for debugging output.",
    )
    parser.add_argument(
        "--post-round-updates",
        type=int,
        default=None,
        help=(
            "Number of SAC updates to run after each round. "
            "Defaults to the step count (one update per interval)."
        ),
    )
    parser.add_argument(
        "--max-chapters",
        type=int,
        default=None,
        help=(
            "Limit the number of chapters processed per round. "
            "Useful for quick smoke tests when the full 76-step run is unnecessary."
        ),
    )
    parser.add_argument(
        "--recompute-lexical-cache",
        action="store_true",
        help="Force regeneration of the lexical TF-IDF cache before training.",
    )
    parser.add_argument(
        "--lexical-eval-chapter",
        type=int,
        default=None,
        help="If provided, run lexical similarity diagnostics for the given chapter before training.",
    )
    parser.add_argument(
        "--lexical-eval-summaries",
        type=Path,
        nargs="*",
        default=(),
        help="One or more summary files used with --lexical-eval-chapter to inspect lexical metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_config, config_path = _load_training_config()
    _reset_output_artifacts()
    _announce_training_config(config_path, training_config)
    article_path = REPO_ROOT / "data" / "sample_article.txt"
    lexical_stats, lexical_tokenizer = _ensure_lexical_statistics(
        article_path, recompute=args.recompute_lexical_cache
    )
    observations = load_article_features(article_path)
    if args.max_chapters is not None:
        if args.max_chapters <= 0:
            raise ValueError("--max-chapters must be positive when provided.")
        observations = observations[: args.max_chapters]
        if not observations:
            raise ValueError("No chapters available after applying --max-chapters filter.")
    chapters = [ob.chapter_text for ob in observations]
    if args.lexical_eval_summaries:
        if args.lexical_eval_chapter is None:
            raise ValueError("--lexical-eval-summaries requires --lexical-eval-chapter")
        _run_inline_lexical_evaluation(
            lexical_stats,
            lexical_tokenizer,
            args.lexical_eval_chapter,
            list(args.lexical_eval_summaries),
        )
    article_text = article_path.read_text(encoding="utf-8")
    total_length, preview = _format_text_debug(article_text, 40, 40)
    _console_log(
        "Loaded article debug info: "
        f"chars={total_length} preview=\"{preview}\""
    )
    _console_log("Chapter statistics:")
    for observation in observations:
        char_length, interval_preview = _format_text_debug(observation.chapter_text, 30, 30)
        print(
            f"  Chapter {observation.step_index:02d} | chars={char_length:04d} "
            f"preview=\"{interval_preview}\""
        )

    agent, trainer = build_demo_components(
        article_path,
        args.replay_capacity,
        precomputed=observations,
        lexical_stats=lexical_stats,
        lexical_tokenizer=lexical_tokenizer,
        training_config=training_config,
    )
    if args.post_round_updates is not None:
        trainer.config.updates_per_round = max(0, args.post_round_updates)
    if trainer.config.updates_per_round <= 0:
        trainer.config.updates_per_round = trainer.config.total_steps
    _console_log(
        "Configured schedule: "
        f"steps_per_round={trainer.config.total_steps} "
        f"post_round_updates={trainer.config.updates_per_round}"
    )
    for round_index in range(1, max(1, args.rounds) + 1):
        trainer.run(round_index=round_index)

    step_rows = _read_csv_rows(STEP_CSV_PATH)
    round_rows = _read_csv_rows(ROUND_CSV_PATH)
    if step_rows:
        _write_rewards_dashboard(step_rows, round_rows)
        print(
            "Exported reward dashboard to "
            f"{REWARDS_HTML_PATH.relative_to(REPO_ROOT)}"
        )
    else:
        _console_log("No step metrics recorded; skipping reward dashboard export.")

    _console_log("Final iterative summary (deterministic policy output):")
    for line in trainer.render_iterative_summary():
        _console_log(f"  {line}")

    snapshot_path = OUT_DIR / "demo_agent_snapshot.json"
    snapshot_metadata = {
        "steps_per_round": trainer.config.total_steps,
        "post_round_updates": trainer.config.updates_per_round,
        "rounds": max(1, args.rounds),
        "replay_capacity": args.replay_capacity,
    }
    snapshot = save_agent_snapshot(agent, snapshot_metadata, snapshot_path)
    _console_log(f"Saved demo agent snapshot to {snapshot_path.relative_to(REPO_ROOT)}")

    model_path = OUT_DIR / "demo_agent_model.bin"
    save_model_artifact(model_path, snapshot["model_size_bytes"])
    _console_log(
        "Saved demo agent model to "
        f"{model_path.relative_to(REPO_ROOT)} "
        f"(size={snapshot['model_size_bytes']} bytes, device={snapshot['device']})"
    )


if __name__ == "__main__":
    main()
