"""Stable WikiText quality artifacts and separate volatile runtime data."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import NotRequired, TypedDict

from inference_experiments.wikitext.models import (
    ArtifactError,
    EvaluationConfig,
    PerplexityComparison,
    PerplexityMetrics,
    RuntimeMetrics,
    TokenizedCorpus,
    WindowScore,
)

PROTOCOL = "wikitext-103-raw-qwen-token-ppl-sliding-v1"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


class ConfigJSON(TypedDict):
    protocol: str
    source_sha256: str
    dataset_revision: str
    dataset_split: str
    model: str
    model_sha256: str
    tokenizer: str
    tokenizer_sha256: str
    text_sha256: str
    token_ids_sha256: str
    context_length: int
    stride: int
    max_model_len: int
    tensor_parallel_size: int
    batch_size: int
    gpu_memory_utilization: float
    quantization: str | None
    seed: int
    limit_tokens: int | None
    git_commit: str


class CorpusJSON(TypedDict):
    row_count: int
    utf8_bytes: int
    corpus_tokens: int
    scored_tokens: int
    text_sha256: str
    tokenizer_sha256: str
    token_ids_sha256: str


class WindowScoreJSON(TypedDict):
    window_id: str
    target_start: int
    target_end: int
    target_token_ids: list[int]
    target_logprobs: list[float]
    total_nll: float


class MetricsJSON(TypedDict):
    corpus_tokens: int
    scored_tokens: int
    total_nll: float
    mean_nll: float
    perplexity: float
    bits_per_token: float


class EngineJSON(TypedDict):
    python_version: str
    vllm_version: str
    torch_version: str
    cuda_version: str | None
    gpu_names: list[str]


class RuntimeJSON(TypedDict):
    engine_initialization_seconds: float
    evaluation_seconds: float
    windows_per_second: float
    prompt_tokens_per_second: float
    scored_tokens_per_second: float
    windows: int
    prompt_tokens: int
    scored_tokens: int
    engine: EngineJSON


class FailureJSON(TypedDict):
    error_type: str
    message: str
    completed_windows: NotRequired[int]
    completed_scored_tokens: NotRequired[int]


class ComparisonJSON(TypedDict):
    baseline_run: str
    candidate_run: str
    baseline_model_sha256: str
    candidate_model_sha256: str
    scored_tokens: int
    baseline_mean_nll: float
    candidate_mean_nll: float
    mean_nll_delta: float
    baseline_perplexity: float
    candidate_perplexity: float
    perplexity_delta: float
    perplexity_ratio: float
    mean_token_nll_delta: float
    mean_absolute_token_nll_delta: float
    max_absolute_token_nll_delta: float
    improved_tokens: int
    unchanged_tokens: int
    worsened_tokens: int


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
    """Atomically rename a successful partial run."""
    partial_dir.rename(output_dir)


def write_config(
    path: Path,
    config: EvaluationConfig,
    corpus: TokenizedCorpus,
    model_sha256: str,
) -> None:
    """Write the stable run and comparison contract."""
    payload: ConfigJSON = {
        "protocol": PROTOCOL,
        "source_sha256": source_sha256(),
        "dataset_revision": WIKITEXT_REVISION,
        "dataset_split": config.split,
        "model": str(config.model),
        "model_sha256": model_sha256,
        "tokenizer": str(config.tokenizer),
        "tokenizer_sha256": corpus.tokenizer_sha256,
        "text_sha256": corpus.text_sha256,
        "token_ids_sha256": corpus.token_ids_sha256,
        "context_length": config.context_length,
        "stride": config.stride,
        "max_model_len": config.max_model_len,
        "tensor_parallel_size": config.tensor_parallel_size,
        "batch_size": config.batch_size,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "quantization": config.quantization,
        "seed": config.seed,
        "limit_tokens": config.limit_tokens,
        "git_commit": git_commit(),
    }
    _write_json(path, payload)


def write_corpus(path: Path, corpus: TokenizedCorpus) -> None:
    """Write token-plan metadata without duplicating every token ID."""
    payload: CorpusJSON = {
        "row_count": corpus.row_count,
        "utf8_bytes": corpus.utf8_bytes,
        "corpus_tokens": len(corpus.token_ids),
        "scored_tokens": len(corpus.token_ids) - 1,
        "text_sha256": corpus.text_sha256,
        "tokenizer_sha256": corpus.tokenizer_sha256,
        "token_ids_sha256": corpus.token_ids_sha256,
    }
    _write_json(path, payload)


def append_scores(path: Path, scores: tuple[WindowScore, ...]) -> None:
    """Append a completed batch of window scores in stable order."""
    with path.open("a", encoding="utf-8") as output:
        for score in scores:
            output.write(
                json.dumps(
                    window_score_json(score), sort_keys=True, separators=(",", ":")
                )
                + "\n"
            )


def window_score_json(score: WindowScore) -> WindowScoreJSON:
    """Convert one typed score into its serialized shape."""
    return {
        "window_id": score.window_id,
        "target_start": score.target_start,
        "target_end": score.target_end,
        "target_token_ids": list(score.target_token_ids),
        "target_logprobs": list(score.target_logprobs),
        "total_nll": score.total_nll,
    }


def write_metrics(path: Path, metrics: PerplexityMetrics) -> None:
    """Write complete deterministic quality metrics."""
    payload: MetricsJSON = metrics_json(metrics)
    _write_json(path, payload)


def metrics_json(metrics: PerplexityMetrics) -> MetricsJSON:
    """Convert perplexity metrics to their stable JSON representation."""
    return {
        "corpus_tokens": metrics.corpus_tokens,
        "scored_tokens": metrics.scored_tokens,
        "total_nll": metrics.total_nll,
        "mean_nll": metrics.mean_nll,
        "perplexity": metrics.perplexity,
        "bits_per_token": metrics.bits_per_token,
    }


def write_runtime(path: Path, runtime: RuntimeMetrics) -> None:
    """Write volatile performance and environment data separately."""
    payload: RuntimeJSON = {
        "engine_initialization_seconds": runtime.engine_initialization_seconds,
        "evaluation_seconds": runtime.evaluation_seconds,
        "windows_per_second": runtime.windows_per_second,
        "prompt_tokens_per_second": runtime.prompt_tokens_per_second,
        "scored_tokens_per_second": runtime.scored_tokens_per_second,
        "windows": runtime.windows,
        "prompt_tokens": runtime.prompt_tokens,
        "scored_tokens": runtime.scored_tokens,
        "engine": {
            "python_version": runtime.engine.python_version,
            "vllm_version": runtime.engine.vllm_version,
            "torch_version": runtime.engine.torch_version,
            "cuda_version": runtime.engine.cuda_version,
            "gpu_names": list(runtime.engine.gpu_names),
        },
    }
    _write_json(path, payload)


def write_failure(
    path: Path, error: Exception, completed_windows: int, completed_tokens: int
) -> None:
    """Describe a failed run without presenting partial aggregate metrics."""
    payload: FailureJSON = {
        "error_type": type(error).__name__,
        "message": str(error),
        "completed_windows": completed_windows,
        "completed_scored_tokens": completed_tokens,
    }
    _write_json(path, payload)


def write_comparison(
    path: Path,
    comparison: PerplexityComparison,
    *,
    baseline_run: Path,
    candidate_run: Path,
    baseline_model_sha256: str,
    candidate_model_sha256: str,
) -> None:
    """Write one validated paired-run comparison without overwriting."""
    if path.exists():
        raise ArtifactError(f"comparison output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: ComparisonJSON = {
        "baseline_run": str(baseline_run),
        "candidate_run": str(candidate_run),
        "baseline_model_sha256": baseline_model_sha256,
        "candidate_model_sha256": candidate_model_sha256,
        "scored_tokens": comparison.scored_tokens,
        "baseline_mean_nll": comparison.baseline_mean_nll,
        "candidate_mean_nll": comparison.candidate_mean_nll,
        "mean_nll_delta": comparison.mean_nll_delta,
        "baseline_perplexity": comparison.baseline_perplexity,
        "candidate_perplexity": comparison.candidate_perplexity,
        "perplexity_delta": comparison.perplexity_delta,
        "perplexity_ratio": comparison.perplexity_ratio,
        "mean_token_nll_delta": comparison.mean_token_nll_delta,
        "mean_absolute_token_nll_delta": (comparison.mean_absolute_token_nll_delta),
        "max_absolute_token_nll_delta": comparison.max_absolute_token_nll_delta,
        "improved_tokens": comparison.improved_tokens,
        "unchanged_tokens": comparison.unchanged_tokens,
        "worsened_tokens": comparison.worsened_tokens,
    }
    _write_json(path, payload)


def checkpoint_sha256(model_dir: Path) -> str:
    """Hash checkpoint, configuration, and tokenizer files in stable order."""
    if not model_dir.is_dir():
        raise ArtifactError(f"model directory does not exist: {model_dir}")
    paths = sorted(
        path
        for path in model_dir.iterdir()
        if path.is_file() and (path.suffix == ".safetensors" or path.suffix == ".json")
    )
    if not paths:
        raise ArtifactError(f"model directory has no checkpoint files: {model_dir}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as model_file:
            while chunk := model_file.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    """Hash the complete WikiText evaluator source in stable order."""
    source_dir = Path(__file__).parent
    paths = sorted(source_dir.glob("*.py"))
    if not paths:
        raise ArtifactError(f"WikiText source files do not exist: {source_dir}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
