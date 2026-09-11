"""Offline MMLU evaluation orchestration."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from inference_experiments.mmlu.artifacts import (
    append_predictions,
    checkpoint_sha256,
    finalize_partial_directory,
    prepare_partial_directory,
    write_config,
    write_failure,
    write_metrics,
    write_runtime,
)
from inference_experiments.mmlu.data import (
    group_demonstrations,
    load_mmlu_split,
    split_path,
)
from inference_experiments.mmlu.engine import InferenceEngine
from inference_experiments.mmlu.metrics import aggregate_metrics
from inference_experiments.mmlu.models import (
    EvaluationConfig,
    EvaluationMetrics,
    InferenceAnswer,
    InferenceRequest,
    MMLUExample,
    Prediction,
    RuntimeMetrics,
)
from inference_experiments.mmlu.prompting import render_five_shot_prompt

EngineFactory = Callable[[], InferenceEngine]
Clock = Callable[[], float]


def run_evaluation(
    config: EvaluationConfig,
    engine_factory: EngineFactory,
    *,
    clock: Clock = perf_counter,
) -> EvaluationMetrics:
    """Run MMLU, retaining partial output on any failure."""
    partial_dir = prepare_partial_directory(config.output_dir)
    completed = 0
    engine: InferenceEngine | None = None
    try:
        model_sha256 = checkpoint_sha256(config.model)
        write_config(partial_dir / "config.json", config, model_sha256)
        predictions_path = partial_dir / "predictions.jsonl"
        predictions_path.touch()
        requests, examples = _load_requests(config)

        initialization_start = clock()
        engine = engine_factory()
        initialization_seconds = clock() - initialization_start
        provenance = engine.provenance()

        evaluation_start = clock()
        predictions: list[Prediction] = []
        for offset in range(0, len(requests), config.batch_size):
            request_batch = requests[offset : offset + config.batch_size]
            example_batch = examples[offset : offset + config.batch_size]
            answers = engine.predict(request_batch)
            batch_predictions = _score_batch(example_batch, answers)
            append_predictions(predictions_path, batch_predictions)
            predictions.extend(batch_predictions)
            completed += len(batch_predictions)
        evaluation_seconds = clock() - evaluation_start

        metrics = aggregate_metrics(tuple(predictions))
        prompt_tokens = sum(row.prompt_tokens for row in predictions)
        output_tokens = sum(row.output_tokens for row in predictions)
        runtime = RuntimeMetrics(
            engine_initialization_seconds=initialization_seconds,
            evaluation_seconds=evaluation_seconds,
            examples_per_second=_rate(len(predictions), evaluation_seconds),
            prompt_tokens_per_second=_rate(prompt_tokens, evaluation_seconds),
            output_tokens_per_second=_rate(output_tokens, evaluation_seconds),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            engine=provenance,
        )
        engine.close()
        engine = None
        write_runtime(partial_dir / "runtime.json", runtime)
        write_metrics(partial_dir / "metrics.json", metrics)
        finalize_partial_directory(partial_dir, config.output_dir)
        return metrics
    except Exception as error:
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass  # Preserve the original evaluation failure.
        (partial_dir / "metrics.json").unlink(missing_ok=True)
        write_failure(partial_dir / "failure.json", error, completed)
        raise


def _load_requests(
    config: EvaluationConfig,
) -> tuple[tuple[InferenceRequest, ...], tuple[MMLUExample, ...]]:
    examples = load_mmlu_split(
        split_path(config.data_dir, config.split),
        subjects=config.subjects,
        limit=config.limit,
    )
    dev_examples = load_mmlu_split(
        split_path(config.data_dir, "dev"), subjects=config.subjects
    )
    demonstrations = group_demonstrations(dev_examples)
    missing = sorted({example.subject for example in examples} - demonstrations.keys())
    if missing:
        raise ValueError(f"test subjects have no demonstrations: {missing}")
    requests = tuple(
        InferenceRequest(
            row_id=example.row_id,
            prompt=render_five_shot_prompt(
                example,
                demonstrations[example.subject],
            ),
        )
        for example in examples
    )
    return requests, examples


def _score_batch(
    examples: tuple[MMLUExample, ...],
    answers: tuple[InferenceAnswer, ...],
) -> tuple[Prediction, ...]:
    if len(examples) != len(answers):
        raise ValueError("engine returned a different number of answers")
    predictions: list[Prediction] = []
    for example, answer in zip(examples, answers, strict=True):
        if example.row_id != answer.row_id:
            raise ValueError(
                f"engine answer order mismatch: {example.row_id} != {answer.row_id}"
            )
        predictions.append(
            Prediction(
                row_id=example.row_id,
                source_index=example.source_index,
                subject=example.subject,
                expected=example.answer,
                predicted=answer.choice,
                correct=example.answer == answer.choice,
                choice_logprobs=answer.choice_logprobs,
                prompt_tokens=answer.prompt_tokens,
                output_tokens=answer.output_tokens,
            )
        )
    return tuple(predictions)


def _rate(count: int, seconds: float) -> float:
    if seconds < 0:
        raise ValueError("clock moved backwards during evaluation")
    return count / seconds if seconds else 0.0
