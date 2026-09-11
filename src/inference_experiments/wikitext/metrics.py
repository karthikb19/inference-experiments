"""Window validation and token-weighted perplexity aggregation."""

from __future__ import annotations

import math

from inference_experiments.wikitext.models import (
    DataValidationError,
    InferenceError,
    PerplexityMetrics,
    PromptLikelihood,
    ScoringWindow,
    WindowScore,
)


def score_windows(
    windows: tuple[ScoringWindow, ...],
    likelihoods: tuple[PromptLikelihood, ...],
) -> tuple[WindowScore, ...]:
    """Validate engine alignment and retain only each new target suffix."""
    if len(windows) != len(likelihoods):
        raise InferenceError("engine returned a different number of likelihoods")
    return tuple(
        _score_window(window, likelihood)
        for window, likelihood in zip(windows, likelihoods, strict=True)
    )


def aggregate_perplexity(
    corpus_tokens: int, scores: tuple[WindowScore, ...]
) -> PerplexityMetrics:
    """Aggregate ordered target scores exactly once with stable summation."""
    if corpus_tokens < 2:
        raise DataValidationError("corpus must contain at least two tokens")
    if not scores:
        raise DataValidationError("cannot aggregate empty window scores")

    expected_start = 1
    logprobs: list[float] = []
    for score in scores:
        if score.target_start != expected_start:
            raise DataValidationError(
                f"target coverage gap or overlap at token {expected_start}"
            )
        if score.target_end <= score.target_start:
            raise DataValidationError(f"{score.window_id}: empty target range")
        expected_count = score.target_end - score.target_start
        if (
            len(score.target_token_ids) != expected_count
            or len(score.target_logprobs) != expected_count
        ):
            raise DataValidationError(f"{score.window_id}: target length mismatch")
        if not math.isfinite(score.total_nll):
            raise DataValidationError(f"{score.window_id}: non-finite NLL")
        calculated_window_nll = math.fsum(-value for value in score.target_logprobs)
        if calculated_window_nll != score.total_nll:
            raise DataValidationError(f"{score.window_id}: window NLL mismatch")
        logprobs.extend(score.target_logprobs)
        expected_start = score.target_end
    if expected_start != corpus_tokens:
        raise DataValidationError(f"target coverage ends at {expected_start}")
    if any(not math.isfinite(value) or value > 0 for value in logprobs):
        raise DataValidationError(
            "target log-probabilities must be finite and non-positive"
        )

    total_nll = math.fsum(-value for value in logprobs)
    scored_tokens = len(logprobs)
    if scored_tokens != corpus_tokens - 1:
        raise DataValidationError("scored token count does not match corpus")
    mean_nll = total_nll / scored_tokens
    try:
        perplexity = math.exp(mean_nll)
    except OverflowError as error:
        raise DataValidationError("perplexity overflowed") from error
    bits_per_token = mean_nll / math.log(2)
    if not all(math.isfinite(value) for value in (total_nll, perplexity)):
        raise DataValidationError("aggregate metrics are non-finite")
    return PerplexityMetrics(
        corpus_tokens=corpus_tokens,
        scored_tokens=scored_tokens,
        total_nll=total_nll,
        mean_nll=mean_nll,
        perplexity=perplexity,
        bits_per_token=bits_per_token,
    )


def _score_window(window: ScoringWindow, likelihood: PromptLikelihood) -> WindowScore:
    if window.window_id != likelihood.window_id:
        raise InferenceError(
            f"engine response order mismatch: {window.window_id} != "
            f"{likelihood.window_id}"
        )
    if window.prompt_token_ids != likelihood.prompt_token_ids:
        raise InferenceError(f"{window.window_id}: response prompt IDs changed")
    if len(likelihood.prompt_logprobs) != len(window.prompt_token_ids):
        raise InferenceError(f"{window.window_id}: response score length mismatch")
    local_start = window.target_start - window.corpus_start
    local_end = window.target_end - window.corpus_start
    if not 1 <= local_start < local_end <= len(window.prompt_token_ids):
        raise InferenceError(f"{window.window_id}: invalid target offsets")

    token_ids = window.prompt_token_ids[local_start:local_end]
    values = likelihood.prompt_logprobs[local_start:local_end]
    logprobs: list[float] = []
    for offset, (token_id, value) in enumerate(
        zip(token_ids, values, strict=True), start=local_start
    ):
        if value is None or value.token_id != token_id:
            raise InferenceError(
                f"{window.window_id}: target score mismatch at {offset}"
            )
        if not math.isfinite(value.logprob) or value.logprob > 0:
            raise InferenceError(
                f"{window.window_id}: invalid target score at {offset}"
            )
        logprobs.append(value.logprob)
    typed_logprobs = tuple(logprobs)
    return WindowScore(
        window_id=window.window_id,
        target_start=window.target_start,
        target_end=window.target_end,
        target_token_ids=token_ids,
        target_logprobs=typed_logprobs,
        total_nll=math.fsum(-value for value in typed_logprobs),
    )
