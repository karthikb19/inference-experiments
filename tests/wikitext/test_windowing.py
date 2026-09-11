"""Tests for gap-free, non-overlapping target-window planning."""

import pytest

from inference_experiments.wikitext.models import TokenizedCorpus
from inference_experiments.wikitext.windowing import plan_scoring_windows


def corpus(size: int) -> TokenizedCorpus:
    return TokenizedCorpus(
        split="test",
        row_count=1,
        utf8_bytes=size,
        text_sha256="text",
        tokenizer_sha256="tokenizer",
        token_ids_sha256="tokens",
        token_ids=tuple(range(size)),
    )


def test_plan_scoring_windows_matches_hugging_face_stride_semantics() -> None:
    windows = plan_scoring_windows(corpus(13), context_length=6, stride=2)

    assert [
        (
            window.corpus_start,
            window.corpus_end,
            window.target_start,
            window.target_end,
        )
        for window in windows
    ] == [
        (0, 6, 1, 6),
        (2, 8, 6, 8),
        (4, 10, 8, 10),
        (6, 12, 10, 12),
        (7, 13, 12, 13),
    ]
    targets = [
        index
        for window in windows
        for index in range(window.target_start, window.target_end)
    ]
    assert targets == list(range(1, 13))
    assert windows[-1].prompt_token_ids == tuple(range(7, 13))


def test_short_corpus_uses_one_window_and_excludes_first_token() -> None:
    windows = plan_scoring_windows(corpus(2), context_length=4096, stride=512)

    assert len(windows) == 1
    assert windows[0].target_start == 1
    assert windows[0].target_end == 2


def test_stride_must_leave_at_least_one_context_token() -> None:
    with pytest.raises(ValueError, match="stride must be"):
        plan_scoring_windows(corpus(12), context_length=6, stride=6)
