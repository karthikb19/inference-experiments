"""Tests for strict MMLU parquet loading."""

from pathlib import Path

import pytest

from inference_experiments.mmlu.data import group_demonstrations, load_mmlu_split
from inference_experiments.mmlu.models import DataValidationError
from tests.mmlu.conftest import write_split


def test_load_mmlu_split_preserves_source_indices_after_filtering(
    tmp_path: Path,
) -> None:
    path = tmp_path / "test.parquet"
    write_split(
        path,
        [
            ("First?", "astronomy", ["a", "b", "c", "d"], 0),
            ("Second?", "abstract_algebra", ["a", "b", "c", "d"], 2),
            ("Third?", "abstract_algebra", ["a", "b", "c", "d"], 3),
        ],
    )

    examples = load_mmlu_split(path, subjects=("abstract_algebra",), limit=1)

    assert len(examples) == 1
    assert examples[0].row_id == "abstract_algebra:1"
    assert examples[0].answer == "C"


@pytest.mark.parametrize(
    ("choices", "answer", "message"),
    [
        (["a", "b", "c"], 0, "exactly four choices"),
        (["a", "b", "c", "d"], 4, "invalid answer index"),
    ],
)
def test_load_mmlu_split_rejects_malformed_rows(
    tmp_path: Path,
    choices: list[str],
    answer: int,
    message: str,
) -> None:
    path = tmp_path / "bad.parquet"
    write_split(path, [("Question?", "astronomy", choices, answer)])

    with pytest.raises(DataValidationError, match=message):
        load_mmlu_split(path)


def test_group_demonstrations_requires_exactly_five_rows(tmp_path: Path) -> None:
    path = tmp_path / "dev.parquet"
    write_split(
        path,
        [
            (f"Question {index}?", "astronomy", ["a", "b", "c", "d"], 0)
            for index in range(4)
        ],
    )

    with pytest.raises(DataValidationError, match="exactly five"):
        group_demonstrations(load_mmlu_split(path))
