"""Tests for completed-artifact loading and paired quantization deltas."""

import math
from dataclasses import replace
from pathlib import Path

import pytest

from inference_experiments.wikitext.comparison import (
    compare_runs,
    load_completed_run,
)
from inference_experiments.wikitext.evaluate import run_evaluation
from inference_experiments.wikitext.models import ComparisonError, CompletedRun
from tests.wikitext.test_evaluate import FakeEngine, FakeTokenizer, create_config


def create_run(tmp_path: Path, name: str, logprob: float) -> CompletedRun:
    config = create_config(tmp_path, name)
    run_evaluation(
        config,
        lambda: FakeEngine(logprob),
        tokenizer_factory=lambda _: FakeTokenizer(),
    )
    return load_completed_run(config.output_dir)


def test_compare_runs_reports_aligned_token_and_perplexity_changes(
    tmp_path: Path,
) -> None:
    baseline = create_run(tmp_path, "baseline", -math.log(2))
    candidate = create_run(tmp_path, "candidate", -math.log(4))

    comparison = compare_runs(baseline, candidate)

    assert comparison.scored_tokens == 6
    assert comparison.mean_nll_delta == pytest.approx(math.log(2))
    assert comparison.mean_token_nll_delta == pytest.approx(math.log(2))
    assert comparison.perplexity_ratio == pytest.approx(2)
    assert comparison.worsened_tokens == 6
    assert comparison.improved_tokens == 0


def test_compare_runs_rejects_different_protocol_contracts(tmp_path: Path) -> None:
    baseline = create_run(tmp_path, "baseline", -1.0)
    candidate = create_run(tmp_path, "candidate", -1.0)
    incompatible_identity = replace(
        candidate.identity, stride=candidate.identity.stride + 1
    )
    incompatible = replace(candidate, identity=incompatible_identity)

    with pytest.raises(ComparisonError, match="contracts do not match"):
        compare_runs(baseline, incompatible)


def test_compare_runs_rejects_changed_target_token(tmp_path: Path) -> None:
    baseline = create_run(tmp_path, "baseline", -1.0)
    candidate = create_run(tmp_path, "candidate", -1.0)
    first = candidate.scores[0]
    changed_first = replace(
        first,
        target_token_ids=(999,) + first.target_token_ids[1:],
    )
    incompatible = replace(candidate, scores=(changed_first,) + candidate.scores[1:])

    with pytest.raises(ComparisonError, match="token plans do not match"):
        compare_runs(baseline, incompatible)
