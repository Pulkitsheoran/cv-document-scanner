"""Tests for the OCR evaluator (CER/WER metrics)."""

from __future__ import annotations

import pytest

from scanvision.evaluator import OCREvaluator, TextNormalizer


@pytest.mark.parametrize(
    "reference,hypothesis,expected_cer",
    [
        ("hello world", "hello world", 0.0),
        ("hello", "hell", 1 / 5),
        ("hello", "hallo", 1 / 5),  # one substitution
        ("hello", "", 1.0),
        ("", "", 0.0),
        ("", "abc", 1.0),
    ],
)
def test_cer_cases(reference, hypothesis, expected_cer):
    assert OCREvaluator.cer(reference, hypothesis) == pytest.approx(expected_cer)


@pytest.mark.parametrize(
    "reference,hypothesis,expected_wer",
    [
        ("the cat sat on the mat", "the cat sat on the mat", 0.0),
        ("the cat", "a cat", 1 / 2),  # one word substitution
        ("a b c", "a b c d", 1 / 3),  # one insertion
        ("a b c", "a c", 1 / 3),  # one word deletion (3 reference tokens)      # noqa: E501
        ("one two", "three", 1.0),
        ("", "", 0.0),
    ],
)
def test_wer_cases(reference, hypothesis, expected_wer):
    assert OCREvaluator.wer(reference, hypothesis) == pytest.approx(expected_wer)


def test_accuracies_are_complementary():
    metrics = OCREvaluator.metrics("correct text here", "correct test here")
    assert metrics.char_accuracy == pytest.approx(1 - metrics.cer)
    assert metrics.word_accuracy == pytest.approx(1 - metrics.wer)


def test_normalization_collapses_whitespace_and_case():
    normalized = TextNormalizer.normalize("  Hello\n\tWorld   Foo ")
    assert normalized == "Hello World Foo"
    assert TextNormalizer.tokens("  a    b ") == ["a", "b"]


def test_summarize():
    rows = [
        OCREvaluator.metrics("same same same", "same same same"),
        OCREvaluator.metrics("abc def ghi", "abc def"),
    ]
    summary = OCREvaluator.summarize(rows)
    assert summary["count"] == 2
    assert 0 <= summary["mean_cer"] <= 1