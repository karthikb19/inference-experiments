"""Offline WikiText perplexity evaluation orchestration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import cast

from inference_experiments.wikitext.artifacts import (
    append_scores,
    checkpoint_sha256,
    finalize_partial_directory,
    prepare_partial_directory,
    write_config,
    write_corpus,
    write_failure,
    write_metrics,
    write_runtime,
)
from inference_experiments.wikitext.data import (
    Tokenizer,
    load_wikitext_rows,
    tokenize_corpus,
)
from inference_experiments.wikitext.engine import LikelihoodEngine
from inference_experiments.wikitext.metrics import (
    aggregate_perplexity,
    score_windows,
)
from inference_experiments.wikitext.models import (
    EvaluationConfig,
    PerplexityMetrics,
    RuntimeMetrics,
)
from inference_experiments.wikitext.windowing import plan_scoring_windows

EngineFactory = Callable[[], LikelihoodEngine]
TokenizerFactory = Callable[[Path], Tokenizer]
Clock = Callable[[], float]


def load_tokenizer(path: Path) -> Tokenizer:
    """Load a local tokenizer lazily without allowing network fallback."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return cast(Tokenizer, tokenizer)


def run_evaluation(
    config: EvaluationConfig,
    engine_factory: EngineFactory,
    *,
    tokenizer_factory: TokenizerFactory = load_tokenizer,
    clock: Clock = perf_counter,
) -> PerplexityMetrics:
    """Score WikiText while retaining ordered partial output on failure."""
    partial_dir = prepare_partial_directory(config.output_dir)
    completed_windows = 0
    completed_tokens = 0
    engine: LikelihoodEngine | None = None
    try:
        rows = load_wikitext_rows(config.data_dir, config.split)
        tokenizer = tokenizer_factory(config.tokenizer)
        corpus = tokenize_corpus(
            rows,
            tokenizer,
            config.tokenizer,
            config.split,
            limit_tokens=config.limit_tokens,
        )
        windows = plan_scoring_windows(
            corpus,
            context_length=config.context_length,
            stride=config.stride,
        )
        model_sha256 = checkpoint_sha256(config.model)
        write_config(partial_dir / "config.json", config, corpus, model_sha256)
        write_corpus(partial_dir / "corpus.json", corpus)
        scores_path = partial_dir / "scores.jsonl"
        scores_path.touch()

        initialization_start = clock()
        engine = engine_factory()
        initialization_seconds = clock() - initialization_start
        provenance = engine.provenance()

        evaluation_start = clock()
        all_scores = []
        for offset in range(0, len(windows), config.batch_size):
            batch = windows[offset : offset + config.batch_size]
            likelihoods = engine.score(batch)
            scores = score_windows(batch, likelihoods)
            append_scores(scores_path, scores)
            all_scores.extend(scores)
            completed_windows += len(scores)
            completed_tokens += sum(
                score.target_end - score.target_start for score in scores
            )
        evaluation_seconds = clock() - evaluation_start

        metrics = aggregate_perplexity(len(corpus.token_ids), tuple(all_scores))
        prompt_tokens = sum(len(window.prompt_token_ids) for window in windows)
        runtime = RuntimeMetrics(
            engine_initialization_seconds=initialization_seconds,
            evaluation_seconds=evaluation_seconds,
            windows_per_second=_rate(len(windows), evaluation_seconds),
            prompt_tokens_per_second=_rate(prompt_tokens, evaluation_seconds),
            scored_tokens_per_second=_rate(metrics.scored_tokens, evaluation_seconds),
            windows=len(windows),
            prompt_tokens=prompt_tokens,
            scored_tokens=metrics.scored_tokens,
            engine=provenance,
        )
        engine.close()
        engine = None
        write_metrics(partial_dir / "metrics.json", metrics)
        write_runtime(partial_dir / "runtime.json", runtime)
        finalize_partial_directory(partial_dir, config.output_dir)
        return metrics
    except Exception as error:
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass  # Preserve the original evaluation failure.
        (partial_dir / "metrics.json").unlink(missing_ok=True)
        (partial_dir / "runtime.json").unlink(missing_ok=True)
        write_failure(
            partial_dir / "failure.json",
            error,
            completed_windows,
            completed_tokens,
        )
        raise


def _rate(count: int, seconds: float) -> float:
    if seconds < 0:
        raise ValueError("clock moved backwards during evaluation")
    return count / seconds if seconds else 0.0
