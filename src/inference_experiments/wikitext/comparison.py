"""Strict loading and paired comparison of completed perplexity runs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from inference_experiments.wikitext.metrics import aggregate_perplexity
from inference_experiments.wikitext.models import (
    ArtifactError,
    ComparisonError,
    CompletedRun,
    PerplexityComparison,
    PerplexityMetrics,
    RunIdentity,
    WindowScore,
)


def load_completed_run(path: Path) -> CompletedRun:
    """Load and cross-check the quality artifacts from a finalized run."""
    if not path.is_dir() or path.name.endswith(".partial"):
        raise ArtifactError(f"completed run directory does not exist: {path}")
    config = _read_object(path / "config.json")
    corpus = _read_object(path / "corpus.json")
    recorded_metrics = _parse_metrics(_read_object(path / "metrics.json"))
    scores = _read_scores(path / "scores.jsonl")
    calculated_metrics = aggregate_perplexity(recorded_metrics.corpus_tokens, scores)
    if calculated_metrics != recorded_metrics:
        raise ArtifactError(f"metrics do not match scores: {path}")

    identity = RunIdentity(
        protocol=_string(config, "protocol"),
        source_sha256=_string(config, "source_sha256"),
        dataset_revision=_string(config, "dataset_revision"),
        dataset_split=_string(config, "dataset_split"),
        text_sha256=_matching_string(config, corpus, "text_sha256"),
        tokenizer_sha256=_matching_string(config, corpus, "tokenizer_sha256"),
        token_ids_sha256=_matching_string(config, corpus, "token_ids_sha256"),
        context_length=_integer(config, "context_length"),
        stride=_integer(config, "stride"),
        corpus_tokens=_matching_integer(
            corpus, recorded_metrics.corpus_tokens, "corpus_tokens"
        ),
        scored_tokens=_matching_integer(
            corpus, recorded_metrics.scored_tokens, "scored_tokens"
        ),
    )
    return CompletedRun(
        path=path,
        model_sha256=_string(config, "model_sha256"),
        identity=identity,
        metrics=recorded_metrics,
        scores=scores,
    )


def compare_runs(
    baseline: CompletedRun, candidate: CompletedRun
) -> PerplexityComparison:
    """Compare aligned token NLLs after enforcing the full quality contract."""
    if baseline.identity != candidate.identity:
        raise ComparisonError("run evaluation contracts do not match")
    if len(baseline.scores) != len(candidate.scores):
        raise ComparisonError("run window counts do not match")

    deltas: list[float] = []
    for baseline_score, candidate_score in zip(
        baseline.scores, candidate.scores, strict=True
    ):
        if (
            baseline_score.window_id != candidate_score.window_id
            or baseline_score.target_start != candidate_score.target_start
            or baseline_score.target_end != candidate_score.target_end
            or baseline_score.target_token_ids != candidate_score.target_token_ids
        ):
            raise ComparisonError("run target token plans do not match")
        deltas.extend(
            baseline_logprob - candidate_logprob
            for baseline_logprob, candidate_logprob in zip(
                baseline_score.target_logprobs,
                candidate_score.target_logprobs,
                strict=True,
            )
        )
    if len(deltas) != baseline.identity.scored_tokens:
        raise ComparisonError("paired token count does not match run identity")

    mean_delta = math.fsum(deltas) / len(deltas)
    mean_nll_delta = candidate.metrics.mean_nll - baseline.metrics.mean_nll
    if not math.isclose(mean_delta, mean_nll_delta, rel_tol=1e-10, abs_tol=1e-10):
        raise ComparisonError("token deltas do not match aggregate NLL delta")
    return PerplexityComparison(
        scored_tokens=len(deltas),
        baseline_mean_nll=baseline.metrics.mean_nll,
        candidate_mean_nll=candidate.metrics.mean_nll,
        mean_nll_delta=mean_nll_delta,
        baseline_perplexity=baseline.metrics.perplexity,
        candidate_perplexity=candidate.metrics.perplexity,
        perplexity_delta=(candidate.metrics.perplexity - baseline.metrics.perplexity),
        perplexity_ratio=(candidate.metrics.perplexity / baseline.metrics.perplexity),
        mean_token_nll_delta=mean_delta,
        mean_absolute_token_nll_delta=(
            math.fsum(abs(delta) for delta in deltas) / len(deltas)
        ),
        max_absolute_token_nll_delta=max(abs(delta) for delta in deltas),
        improved_tokens=sum(delta < 0 for delta in deltas),
        unchanged_tokens=sum(delta == 0 for delta in deltas),
        worsened_tokens=sum(delta > 0 for delta in deltas),
    )


def _read_scores(path: Path) -> tuple[WindowScore, ...]:
    if not path.is_file():
        raise ArtifactError(f"artifact does not exist: {path}")
    scores: list[WindowScore] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            raise ArtifactError(f"{path}:{line_number}: blank score line")
        payload = _parse_object(line, f"{path}:{line_number}")
        token_ids = _integer_list(payload, "target_token_ids")
        logprobs = _float_list(payload, "target_logprobs")
        scores.append(
            WindowScore(
                window_id=_string(payload, "window_id"),
                target_start=_integer(payload, "target_start"),
                target_end=_integer(payload, "target_end"),
                target_token_ids=token_ids,
                target_logprobs=logprobs,
                total_nll=_number(payload, "total_nll"),
            )
        )
    return tuple(scores)


def _parse_metrics(payload: Mapping[str, object]) -> PerplexityMetrics:
    return PerplexityMetrics(
        corpus_tokens=_integer(payload, "corpus_tokens"),
        scored_tokens=_integer(payload, "scored_tokens"),
        total_nll=_number(payload, "total_nll"),
        mean_nll=_number(payload, "mean_nll"),
        perplexity=_number(payload, "perplexity"),
        bits_per_token=_number(payload, "bits_per_token"),
    )


def _read_object(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise ArtifactError(f"artifact does not exist: {path}")
    return _parse_object(path.read_text(encoding="utf-8"), str(path))


def _parse_object(raw: str, source: str) -> Mapping[str, object]:
    try:
        payload: object = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ArtifactError(f"invalid JSON in {source}") from error
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        raise ArtifactError(f"expected JSON object in {source}")
    return cast(dict[str, object], payload)


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ArtifactError(f"{key} must be a string")
    return value


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArtifactError(f"{key} must be an integer")
    return value


def _number(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ArtifactError(f"{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ArtifactError(f"{key} must be finite")
    return result


def _integer_list(payload: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ArtifactError(f"{key} must be an integer list")
    return tuple(cast(list[int], value))


def _float_list(payload: Mapping[str, object], key: str) -> tuple[float, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    ):
        raise ArtifactError(f"{key} must be a number list")
    return tuple(float(item) for item in cast(list[int | float], value))


def _matching_string(
    first: Mapping[str, object], second: Mapping[str, object], key: str
) -> str:
    first_value = _string(first, key)
    if first_value != _string(second, key):
        raise ArtifactError(f"{key} does not match across artifacts")
    return first_value


def _matching_integer(first: Mapping[str, object], second_value: int, key: str) -> int:
    first_value = _integer(first, key)
    if first_value != second_value:
        raise ArtifactError(f"{key} does not match across artifacts")
    return first_value
