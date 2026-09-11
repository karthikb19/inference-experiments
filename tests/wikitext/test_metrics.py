"""Tests for window alignment and token-weighted perplexity."""

import math

import pytest

from inference_experiments.wikitext.metrics import (
    aggregate_perplexity,
    score_windows,
)
from inference_experiments.wikitext.models import (
    DataValidationError,
    PromptLikelihood,
    PromptTokenLogprob,
    ScoringWindow,
    WindowScore,
)


def test_score_and_aggregate_uses_each_target_once() -> None:
    window = ScoringWindow("window:000000", 0, 3, 1, 3, (10, 11, 12))
    likelihood = PromptLikelihood(
        "window:000000",
        (10, 11, 12),
        (
            None,
            PromptTokenLogprob(11, -math.log(2)),
            PromptTokenLogprob(12, -math.log(8)),
        ),
    )

    scores = score_windows((window,), (likelihood,))
    metrics = aggregate_perplexity(3, scores)

    assert scores[0].target_token_ids == (11, 12)
    assert metrics.total_nll == pytest.approx(math.log(16))
    assert metrics.mean_nll == pytest.approx(math.log(4))
    assert metrics.perplexity == pytest.approx(4)
    assert metrics.bits_per_token == pytest.approx(2)


def test_aggregate_rejects_gap_and_nonfinite_score() -> None:
    gap = WindowScore("window:000000", 2, 3, (12,), (-1.0,), 1.0)
    with pytest.raises(DataValidationError, match="gap or overlap"):
        aggregate_perplexity(3, (gap,))

    invalid = WindowScore(
        "window:000000", 1, 3, (11, 12), (-1.0, float("nan")), float("nan")
    )
    with pytest.raises(DataValidationError, match="non-finite NLL"):
        aggregate_perplexity(3, (invalid,))

    inconsistent = WindowScore("window:000000", 1, 3, (11, 12), (-1.0, -1.0), 99.0)
    with pytest.raises(DataValidationError, match="window NLL mismatch"):
        aggregate_perplexity(3, (inconsistent,))


def test_aggregate_rejects_perplexity_overflow() -> None:
    score = WindowScore("window:000000", 1, 2, (11,), (-1000.0,), 1000.0)

    with pytest.raises(DataValidationError, match="perplexity overflowed"):
        aggregate_perplexity(2, (score,))
