"""Strict loading for the pinned MMLU parquet files."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pyarrow.parquet as pq

from inference_experiments.mmlu.models import DataValidationError, MMLUExample

REQUIRED_COLUMNS = ("question", "subject", "choices", "answer")


def split_path(data_dir: Path, split: str) -> Path:
    """Return the repository parquet path for a named MMLU split."""
    return data_dir / f"{split}-00000-of-00001.parquet"


def load_mmlu_split(
    path: Path,
    *,
    subjects: Iterable[str] = (),
    limit: int | None = None,
) -> tuple[MMLUExample, ...]:
    """Load, validate, filter, and stably identify one MMLU split."""
    if not path.is_file():
        raise DataValidationError(f"MMLU split does not exist: {path}")
    table = pq.read_table(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(table.column_names))
    if missing:
        raise DataValidationError(f"MMLU split is missing columns: {missing}")

    questions = cast(list[str], table.column("question").to_pylist())
    row_subjects = cast(list[str], table.column("subject").to_pylist())
    row_choices = cast(list[list[str]], table.column("choices").to_pylist())
    answers = cast(list[int], table.column("answer").to_pylist())
    selected_subjects = frozenset(subjects)
    examples: list[MMLUExample] = []

    for index, (question, subject, choices, answer) in enumerate(
        zip(questions, row_subjects, row_choices, answers, strict=True)
    ):
        _validate_row(index, question, subject, choices, answer)
        if selected_subjects and subject not in selected_subjects:
            continue
        typed_choices = cast(tuple[str, str, str, str], tuple(choices))
        examples.append(
            MMLUExample(
                row_id=f"{subject}:{index}",
                source_index=index,
                question=question,
                subject=subject,
                choices=typed_choices,
                answer_index=answer,
            )
        )
        if limit is not None and len(examples) == limit:
            break

    if not examples:
        raise DataValidationError("MMLU selection contains no examples")
    return tuple(examples)


def group_demonstrations(
    examples: tuple[MMLUExample, ...],
) -> dict[str, tuple[MMLUExample, ...]]:
    """Group exactly five development examples for every subject."""
    grouped: dict[str, list[MMLUExample]] = {}
    for example in examples:
        grouped.setdefault(example.subject, []).append(example)
    invalid = sorted(subject for subject, rows in grouped.items() if len(rows) != 5)
    if invalid:
        raise DataValidationError(
            f"subjects must have exactly five development examples: {invalid}"
        )
    return {subject: tuple(rows) for subject, rows in grouped.items()}


def _validate_row(
    index: int,
    question: object,
    subject: object,
    choices: object,
    answer: object,
) -> None:
    if not isinstance(question, str) or not question.strip():
        raise DataValidationError(f"row {index} has an invalid question")
    if not isinstance(subject, str) or not subject.strip():
        raise DataValidationError(f"row {index} has an invalid subject")
    if not isinstance(choices, list) or len(choices) != 4:
        raise DataValidationError(f"row {index} must have exactly four choices")
    if not all(isinstance(choice, str) and choice.strip() for choice in choices):
        raise DataValidationError(f"row {index} has an invalid choice")
    if (
        not isinstance(answer, int)
        or isinstance(answer, bool)
        or answer not in range(4)
    ):
        raise DataValidationError(f"row {index} has an invalid answer index")
