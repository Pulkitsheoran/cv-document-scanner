"""OCR evaluation: character- and word-level error rates against ground truth.

Metrics
-------
- CER (character error rate):  Levenshtein distance / reference length
- WER (word error rate):        word-edit distance / reference word count
- accuracy = 1 - error rate     (clipped at zero)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class Metrics:
    reference: str
    hypothesis: str
    cer: float
    wer: float
    char_accuracy: float
    word_accuracy: float
    ref_chars: int
    hyp_chars: int


class TextNormalizer:
    """Normalises OCR output and reference text before scoring."""

    @staticmethod
    def normalize(text: str) -> str:
        value = text or ""
        value = re.sub(r"\s+", " ", value).strip()
        return value

    @staticmethod
    def tokens(text: str) -> list[str]:
        return TextNormalizer.normalize(text).split() if text else []


class OCREvaluator:
    """Computes standard OCR quality metrics between hypothesis and reference."""

    @staticmethod
    def edit_distance(left: str, right: str) -> int:
        """Levenshtein distance using two rolling rows (O(m*n), O(n) memory)."""
        if left == right:
            return 0
        if not left:
            return len(right)
        if not right:
            return len(left)
        previous = list(range(len(right) + 1))
        for i, char_a in enumerate(left, start=1):
            current = [i] + [0] * len(right)
            for j, char_b in enumerate(right, start=1):
                current[j] = min(
                    previous[j] + 1,          # deletion
                    current[j - 1] + 1,       # insertion
                    previous[j - 1] + (char_a != char_b),  # substitution
                )
            previous = current
        return previous[-1]

    @classmethod
    def cer(cls, reference: str, hypothesis: str) -> float:
        ref = TextNormalizer.normalize(reference)
        hyp = TextNormalizer.normalize(hypothesis)
        if not ref and not hyp:
            return 0.0
        if not ref:
            return 1.0
        return cls.edit_distance(ref, hyp) / float(len(ref))

    @classmethod
    def wer(cls, reference: str, hypothesis: str) -> float:
        ref_tokens = TextNormalizer.tokens(reference)
        hyp_tokens = TextNormalizer.tokens(hypothesis)
        if not ref_tokens and not hyp_tokens:
            return 0.0
        if not ref_tokens:
            return 1.0
        distance = cls._word_edit_distance(ref_tokens, hyp_tokens)
        return distance / float(len(ref_tokens))

    @staticmethod
    def _word_edit_distance(reference: list[str], hypothesis: list[str]) -> int:
        """Standard DP edit distance over token lists (1 per word operation)."""
        rows, cols = len(reference) + 1, len(hypothesis) + 1
        table = [[0] * cols for _ in range(rows)]
        for i in range(rows):
            table[i][0] = i
        for j in range(cols):
            table[0][j] = j
        for i in range(1, rows):
            for j in range(1, cols):
                cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
                table[i][j] = min(
                    table[i - 1][j] + 1,          # deletion
                    table[i][j - 1] + 1,          # insertion
                    table[i - 1][j - 1] + cost,   # substitution
                )
        return table[-1][-1]

    @classmethod
    def metrics(cls, reference: str, hypothesis: str) -> Metrics:
        cer = cls.cer(reference, hypothesis)
        wer = cls.wer(reference, hypothesis)
        return Metrics(
            reference=reference or "",
            hypothesis=hypothesis or "",
            cer=cer,
            wer=wer,
            char_accuracy=max(0.0, 1.0 - cer),
            word_accuracy=max(0.0, 1.0 - wer),
            ref_chars=len(TextNormalizer.normalize(reference)),
            hyp_chars=len(TextNormalizer.normalize(hypothesis)),
        )

    @staticmethod
    def summarize(rows: Iterable[Metrics]) -> dict:
        values = [r for r in rows if r is not None]
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "mean_cer": float(np.mean([r.cer for r in values])),
            "mean_wer": float(np.mean([r.wer for r in values])),
            "mean_char_accuracy": float(np.mean([r.char_accuracy for r in values])),
            "mean_word_accuracy": float(np.mean([r.word_accuracy for r in values])),
            "best_cer": float(min(r.cer for r in values)),
            "worst_cer": float(max(r.cer for r in values)),
        }