"""Tests for strict vLLM prompt-logprob response parsing."""

from dataclasses import dataclass

import pytest

from inference_experiments.wikitext.engine import VLLMLikelihoodEngine, _Logprob
from inference_experiments.wikitext.models import InferenceError, ScoringWindow


@dataclass
class FakeLogprob:
    logprob: float


@dataclass
class FakeCompletion:
    token_ids: list[int]


@dataclass
class FakeOutput:
    prompt_token_ids: list[int] | None
    prompt_logprobs: list[dict[int, _Logprob] | None] | None
    outputs: list[FakeCompletion]


def request() -> ScoringWindow:
    return ScoringWindow(
        window_id="window:000000",
        corpus_start=0,
        corpus_end=3,
        target_start=1,
        target_end=3,
        prompt_token_ids=(10, 11, 12),
    )


def output() -> FakeOutput:
    return FakeOutput(
        prompt_token_ids=[10, 11, 12],
        prompt_logprobs=[
            None,
            {11: FakeLogprob(-1.0)},
            {12: FakeLogprob(-2.0)},
        ],
        outputs=[FakeCompletion([99])],
    )


def test_parse_output_keeps_observed_prompt_scores_and_ignores_decode() -> None:
    engine = VLLMLikelihoodEngine.__new__(VLLMLikelihoodEngine)

    likelihood = engine._parse_output(request(), output())

    assert likelihood.prompt_logprobs[0] is None
    assert likelihood.prompt_logprobs[1] is not None
    assert likelihood.prompt_logprobs[1].token_id == 11
    assert likelihood.prompt_logprobs[1].logprob == -1.0


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), 0.1])
def test_parse_output_rejects_invalid_observed_score(bad_score: float) -> None:
    engine = VLLMLikelihoodEngine.__new__(VLLMLikelihoodEngine)
    result = output()
    result.prompt_logprobs = [
        None,
        {11: FakeLogprob(bad_score)},
        {12: FakeLogprob(-2.0)},
    ]

    with pytest.raises(InferenceError, match="invalid log-probability"):
        engine._parse_output(request(), result)


def test_parse_output_rejects_missing_observed_token() -> None:
    engine = VLLMLikelihoodEngine.__new__(VLLMLikelihoodEngine)
    result = output()
    result.prompt_logprobs = [None, {88: FakeLogprob(-1.0)}, {12: FakeLogprob(-2.0)}]

    with pytest.raises(InferenceError, match="missing observed token score"):
        engine._parse_output(request(), result)


def test_parse_output_rejects_changed_prompt_and_invalid_first_score() -> None:
    engine = VLLMLikelihoodEngine.__new__(VLLMLikelihoodEngine)
    changed = output()
    changed.prompt_token_ids = [10, 99, 12]
    with pytest.raises(InferenceError, match="prompt token IDs changed"):
        engine._parse_output(request(), changed)

    first_scored = output()
    first_scored.prompt_logprobs = [
        {10: FakeLogprob(-1.0)},
        {11: FakeLogprob(-1.0)},
        {12: FakeLogprob(-2.0)},
    ]
    with pytest.raises(InferenceError, match="first prompt log-probability"):
        engine._parse_output(request(), first_scored)
