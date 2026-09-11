"""Tests for tokenizer assumptions and answer parsing at the vLLM boundary."""

from dataclasses import dataclass

import pytest

from inference_experiments.mmlu.engine import (
    VLLMEngine,
    _answer_token_ids,
    _Completion,
    _Logprob,
)
from inference_experiments.mmlu.models import CHOICES, InferenceError, InferenceRequest


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


@dataclass
class FakeLogprob:
    logprob: float


@dataclass
class FakeCompletion:
    token_ids: list[int]
    logprobs: list[dict[int, _Logprob]] | None


@dataclass
class FakeOutput:
    prompt_token_ids: list[int]
    outputs: list[_Completion]


def scored_output(
    selected_token: int, scores: tuple[float, float, float, float]
) -> FakeOutput:
    return FakeOutput(
        prompt_token_ids=[10, 11],
        outputs=[
            FakeCompletion(
                token_ids=[selected_token],
                logprobs=[
                    {index: FakeLogprob(score) for index, score in enumerate(scores, 1)}
                ],
            )
        ],
    )


@pytest.fixture
def engine() -> VLLMEngine:
    instance = VLLMEngine.__new__(VLLMEngine)
    instance._answer_token_ids = (1, 2, 3, 4)
    return instance


@pytest.mark.parametrize("selected_token", [2, 4])
@pytest.mark.parametrize("all_tied", [False, True])
def test_parse_output_preserves_emitted_maximum_on_exact_tie(
    engine: VLLMEngine, selected_token: int, all_tied: bool
) -> None:
    tied = -0.6932079792
    scores = (tied, tied, tied, tied) if all_tied else (-9.0, tied, -8.0, tied)
    answer = engine._parse_output(
        InferenceRequest("abstract_algebra:35", "prompt"),
        scored_output(selected_token, scores),
    )

    assert answer.choice == CHOICES[selected_token - 1]
    assert answer.choice_logprobs == scores
    assert answer.row_id == "abstract_algebra:35"
    assert answer.prompt_tokens == 2
    assert answer.output_tokens == 1


def test_parse_output_accepts_unique_maximum(engine: VLLMEngine) -> None:
    answer = engine._parse_output(
        InferenceRequest("row:1", "prompt"),
        scored_output(3, (-4.0, -3.0, -1.0, -2.0)),
    )

    assert answer.choice == "C"


@pytest.mark.parametrize("selected_score", [-2.0, -1.0000000001])
def test_parse_output_rejects_strictly_lower_score(
    engine: VLLMEngine, selected_score: float
) -> None:
    with pytest.raises(
        InferenceError, match="row:1: selected token and scores disagree"
    ):
        engine._parse_output(
            InferenceRequest("row:1", "prompt"),
            scored_output(4, (-4.0, -1.0, -3.0, selected_score)),
        )


def test_parse_output_rejects_token_outside_answer_set(engine: VLLMEngine) -> None:
    with pytest.raises(
        InferenceError, match="row:1: selected token is not an A-D answer"
    ):
        engine._parse_output(
            InferenceRequest("row:1", "prompt"),
            scored_output(99, (-1.0, -1.0, -1.0, -1.0)),
        )
