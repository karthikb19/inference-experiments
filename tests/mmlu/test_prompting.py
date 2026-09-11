"""Golden tests for the original MMLU prompt."""

import pytest

from inference_experiments.mmlu.models import MMLUExample
from inference_experiments.mmlu.prompting import render_five_shot_prompt


def example(index: int, *, subject: str = "abstract_algebra") -> MMLUExample:
    return MMLUExample(
        row_id=f"{subject}:{index}",
        source_index=index,
        question=f"Question {index}?",
        subject=subject,
        choices=("zero", "one", "two", "three"),
        answer_index=index % 4,
    )


def test_render_five_shot_prompt_matches_original_format() -> None:
    prompt = render_five_shot_prompt(example(5), tuple(example(i) for i in range(5)))

    assert prompt.startswith(
        "The following are multiple choice questions (with answers) about "
        "abstract algebra.\n\n"
        "Question 0?\nA. zero\nB. one\nC. two\nD. three\nAnswer: A\n\n"
    )
    assert prompt.endswith("Question 5?\nA. zero\nB. one\nC. two\nD. three\nAnswer:")


def test_render_five_shot_prompt_rejects_cross_subject_examples() -> None:
    demonstrations = tuple(example(i) for i in range(4)) + (
        example(4, subject="astronomy"),
    )

    with pytest.raises(ValueError, match="match the test subject"):
        render_five_shot_prompt(example(5), demonstrations)
