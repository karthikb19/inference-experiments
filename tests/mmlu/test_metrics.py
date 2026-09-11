"""Tests for weighted and diagnostic MMLU metrics."""

import pytest

from inference_experiments.mmlu.metrics import aggregate_metrics
from inference_experiments.mmlu.models import DataValidationError, Prediction


def prediction(index: int, subject: str, correct: bool) -> Prediction:
    return Prediction(
        row_id=f"{subject}:{index}",
        source_index=index,
        subject=subject,
        expected="A",
        predicted="A" if correct else "B",
        correct=correct,
        choice_logprobs=(0.0, -1.0, -2.0, -3.0),
        prompt_tokens=10,
        output_tokens=1,
    )


def test_aggregate_metrics_weights_examples_and_macros_subjects() -> None:
    rows = (
        prediction(0, "abstract_algebra", True),
        prediction(1, "abstract_algebra", True),
        prediction(2, "astronomy", False),
    )

    metrics = aggregate_metrics(rows)

    assert metrics.weighted_accuracy == pytest.approx(2 / 3)
    assert metrics.subject_macro_accuracy == pytest.approx(0.5)
    assert metrics.categories[0].name == "stem"
    assert metrics.categories[0].weighted_accuracy == pytest.approx(2 / 3)
    assert metrics.categories[0].subject_macro_accuracy == pytest.approx(0.5)
    assert metrics.invalid_predictions == 0


def test_aggregate_metrics_rejects_unknown_subject() -> None:
    with pytest.raises(DataValidationError, match="no MMLU category"):
        aggregate_metrics((prediction(0, "unknown", True),))
