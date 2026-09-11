"""Tests for tokenizer assumptions at the vLLM boundary."""

import pytest

from inference_experiments.mmlu.engine import _answer_token_ids
from inference_experiments.mmlu.models import InferenceError


class FakeTokenizer:
    def __init__(self, encodings: dict[str, list[int]]) -> None:
        self.encodings = encodings

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return self.encodings[text]


def test_answer_token_ids_requires_four_distinct_single_tokens() -> None:
    tokenizer = FakeTokenizer({" A": [1], " B": [2], " C": [3], " D": [4]})

    assert _answer_token_ids(tokenizer) == (1, 2, 3, 4)


def test_answer_token_ids_rejects_multi_token_choice() -> None:
    tokenizer = FakeTokenizer({" A": [1], " B": [2, 3], " C": [4], " D": [5]})

    with pytest.raises(InferenceError, match="not one tokenizer token"):
        _answer_token_ids(tokenizer)
