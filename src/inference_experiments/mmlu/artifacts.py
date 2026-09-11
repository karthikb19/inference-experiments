"""Deterministic quality and separate volatile runtime artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import NotRequired, TypedDict

from inference_experiments.mmlu.models import (
    AccuracyMetric,
    ArtifactError,
    CategoryMetric,
    EvaluationConfig,
    EvaluationMetrics,
    Prediction,
    RuntimeMetrics,
)

MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"


class ChoiceLogprobsJSON(TypedDict):
    A: float
    B: float
    C: float
    D: float


class PredictionJSON(TypedDict):
    row_id: str
    source_index: int
    subject: str
    expected: str
    predicted: str
    correct: bool
    choice_logprobs: ChoiceLogprobsJSON
    prompt_tokens: int
    output_tokens: int


class AccuracyJSON(TypedDict):
    name: str
    correct: int
    total: int
    accuracy: float


class CategoryJSON(TypedDict):
    name: str
    correct: int
    total: int
    weighted_accuracy: float
    subject_macro_accuracy: float


class MetricsJSON(TypedDict):
    correct: int
    total: int
    weighted_accuracy: float
    subject_macro_accuracy: float
    invalid_predictions: int
    subjects: list[AccuracyJSON]
    categories: list[CategoryJSON]


class ConfigJSON(TypedDict):
    protocol: str
    dataset_revision: str
    dataset_split: str
    model: str
    model_sha256: str
    tensor_parallel_size: int
    max_model_len: int
    batch_size: int
    gpu_memory_utilization: float
    seed: int
    limit: int | None
    subjects: list[str]
    git_commit: str


class EngineJSON(TypedDict):
    python_version: str
    vllm_version: str
    torch_version: str
    cuda_version: str | None
    gpu_names: list[str]


class RuntimeJSON(TypedDict):
    engine_initialization_seconds: float
    evaluation_seconds: float
    examples_per_second: float
    prompt_tokens_per_second: float
    output_tokens_per_second: float
    prompt_tokens: int
    output_tokens: int
    engine: EngineJSON


class FailureJSON(TypedDict):
    error_type: str
    message: str
    completed_predictions: NotRequired[int]


def prepare_partial_directory(output_dir: Path) -> Path:
    """Create an empty partial directory without overwriting prior work."""
    partial_dir = output_dir.with_name(f"{output_dir.name}.partial")
    if output_dir.exists():
        raise ArtifactError(f"completed output already exists: {output_dir}")
    if partial_dir.exists():
        raise ArtifactError(f"partial output already exists: {partial_dir}")
    partial_dir.mkdir(parents=True)
    return partial_dir


def finalize_partial_directory(partial_dir: Path, output_dir: Path) -> None:
    """Atomically rename a successful partial run to its final destination."""
    partial_dir.rename(output_dir)


def write_config(path: Path, config: EvaluationConfig, model_sha256: str) -> None:
    """Write the stable run contract."""
    payload: ConfigJSON = {
        "protocol": "mmlu-original-5shot-nonthinking-v1",
        "dataset_revision": MMLU_REVISION,
        "dataset_split": config.split,
        "model": str(config.model),
        "model_sha256": model_sha256,
        "tensor_parallel_size": config.tensor_parallel_size,
        "max_model_len": config.max_model_len,
        "batch_size": config.batch_size,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "seed": config.seed,
        "limit": config.limit,
        "subjects": list(config.subjects),
        "git_commit": git_commit(),
    }
    _write_json(path, payload)


def append_predictions(path: Path, predictions: tuple[Prediction, ...]) -> None:
    """Append a completed batch in stable row order."""
    with path.open("a", encoding="utf-8") as output:
        for prediction in predictions:
            output.write(_json_line(prediction_json(prediction)))


def write_metrics(path: Path, metrics: EvaluationMetrics) -> None:
    """Write deterministic aggregate quality metrics."""
    payload: MetricsJSON = {
        "correct": metrics.correct,
        "total": metrics.total,
        "weighted_accuracy": metrics.weighted_accuracy,
        "subject_macro_accuracy": metrics.subject_macro_accuracy,
        "invalid_predictions": metrics.invalid_predictions,
        "subjects": [_accuracy_json(metric) for metric in metrics.subjects],
        "categories": [_category_json(metric) for metric in metrics.categories],
    }
    _write_json(path, payload)


def write_runtime(path: Path, runtime: RuntimeMetrics) -> None:
    """Write volatile timing and environment details separately."""
    payload: RuntimeJSON = {
        "engine_initialization_seconds": runtime.engine_initialization_seconds,
        "evaluation_seconds": runtime.evaluation_seconds,
        "examples_per_second": runtime.examples_per_second,
        "prompt_tokens_per_second": runtime.prompt_tokens_per_second,
        "output_tokens_per_second": runtime.output_tokens_per_second,
        "prompt_tokens": runtime.prompt_tokens,
        "output_tokens": runtime.output_tokens,
        "engine": {
            "python_version": runtime.engine.python_version,
            "vllm_version": runtime.engine.vllm_version,
            "torch_version": runtime.engine.torch_version,
            "cuda_version": runtime.engine.cuda_version,
            "gpu_names": list(runtime.engine.gpu_names),
        },
    }
    _write_json(path, payload)


def write_failure(path: Path, error: Exception, completed: int) -> None:
    """Mark a run as partial without presenting aggregate results."""
    payload: FailureJSON = {
        "error_type": type(error).__name__,
        "message": str(error),
        "completed_predictions": completed,
    }
    _write_json(path, payload)


def prediction_json(prediction: Prediction) -> PredictionJSON:
    """Convert one prediction to its stable serialized schema."""
    scores = prediction.choice_logprobs
    return {
        "row_id": prediction.row_id,
        "source_index": prediction.source_index,
        "subject": prediction.subject,
        "expected": prediction.expected,
        "predicted": prediction.predicted,
        "correct": prediction.correct,
        "choice_logprobs": {
            "A": scores[0],
            "B": scores[1],
            "C": scores[2],
            "D": scores[3],
        },
        "prompt_tokens": prediction.prompt_tokens,
        "output_tokens": prediction.output_tokens,
    }


def checkpoint_sha256(model_dir: Path) -> str:
    """Hash checkpoint, configuration, and tokenizer files in stable order."""
    if not model_dir.is_dir():
        raise ArtifactError(f"model directory does not exist: {model_dir}")
    files = sorted(
        path
        for path in model_dir.iterdir()
        if path.is_file() and (path.suffix == ".safetensors" or path.suffix == ".json")
    )
    if not files:
        raise ArtifactError(f"model directory has no checkpoint files: {model_dir}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as model_file:
            while chunk := model_file.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    """Return the repository commit that produced the run."""
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _accuracy_json(metric: AccuracyMetric) -> AccuracyJSON:
    return {
        "name": metric.name,
        "correct": metric.correct,
        "total": metric.total,
        "accuracy": metric.accuracy,
    }


def _category_json(metric: CategoryMetric) -> CategoryJSON:
    return {
        "name": metric.name,
        "correct": metric.correct,
        "total": metric.total,
        "weighted_accuracy": metric.weighted_accuracy,
        "subject_macro_accuracy": metric.subject_macro_accuracy,
    }


def _json_line(payload: PredictionJSON) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
