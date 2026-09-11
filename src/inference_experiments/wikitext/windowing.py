"""Deterministic finite-context scoring-window planning."""

from __future__ import annotations

from inference_experiments.wikitext.models import (
    DataValidationError,
    ScoringWindow,
    TokenizedCorpus,
)


def plan_scoring_windows(
    corpus: TokenizedCorpus, *, context_length: int, stride: int
) -> tuple[ScoringWindow, ...]:
    """Cover every eligible corpus target exactly once with left context."""
    if context_length < 2:
        raise ValueError("context_length must be at least 2")
    if not 1 <= stride < context_length:
        raise ValueError("stride must be in [1, context_length)")
    if len(corpus.token_ids) < 2:
        raise DataValidationError("corpus must contain at least two tokens")

    corpus_tokens = len(corpus.token_ids)
    first_end = min(context_length, corpus_tokens)
    spans = [(0, first_end, 1, first_end)]
    target_start = first_end
    while target_start < corpus_tokens:
        target_end = min(target_start + stride, corpus_tokens)
        corpus_start = max(0, target_end - context_length)
        spans.append((corpus_start, target_end, target_start, target_end))
        target_start = target_end

    return tuple(
        ScoringWindow(
            window_id=f"window:{index:06d}",
            corpus_start=corpus_start,
            corpus_end=corpus_end,
            target_start=target_start,
            target_end=target_end,
            prompt_token_ids=corpus.token_ids[corpus_start:corpus_end],
        )
        for index, (corpus_start, corpus_end, target_start, target_end) in enumerate(
            spans
        )
    )
