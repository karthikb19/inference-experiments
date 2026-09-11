"""Typed inputs and outputs for WikiText perplexity evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Split = Literal["validation", "test"]


class EvaluationError(RuntimeError):
    """Base class for expected WikiText evaluation failures."""


class DataValidationError(EvaluationError):
    """Raised when source data or a token plan is invalid."""


class InferenceError(EvaluationError):
    """Raised when an inference response cannot be scored."""


class ArtifactError(EvaluationError):
    """Raised when run artifacts are missing, malformed, or unsafe."""


class ComparisonError(EvaluationError):
    """Raised when two perplexity runs are not comparable."""


@dataclass(frozen=True)
class EvaluationConfig:
    """Inputs that define one WikiText quality run."""

    model: Path
    tokenizer: Path
    data_dir: Path
    output_dir: Path
    split: Split = "test"
    context_length: int = 4096
    stride: int = 512
    tensor_parallel_size: int = 2
    batch_size: int = 8
    gpu_memory_utilization: float = 0.9
    quantization: str | None = None
    seed: int = 0
    limit_tokens: int | None = None

    def __post_init__(self) -> None:
        """Reject configurations that cannot produce perplexity."""
        if self.context_length < 2:
            raise ValueError("context_length must be at least 2")
        if not 1 <= self.stride < self.context_length:
            raise ValueError("stride must be in [1, context_length)")
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.limit_tokens is not None and self.limit_tokens < 2:
            raise ValueError("limit_tokens must be at least 2 when provided")
        if self.quantization is not None and not self.quantization.strip():
            raise ValueError("quantization must be non-empty when provided")

    @property
    def max_model_len(self) -> int:
        """Reserve one engine position for vLLM's ignored decode token."""
        return self.context_length + 1


@dataclass(frozen=True)
class WikiTextRow:
    """One validated source row in stable split order."""

    source_index: int
    text: str


@dataclass(frozen=True)
class TokenizedCorpus:
    """One rendered corpus and its immutable Qwen token plan."""

    split: Split
    row_count: int
    utf8_bytes: int
    text_sha256: str
    tokenizer_sha256: str
    token_ids_sha256: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class ScoringWindow:
    """One prompt span and its non-overlapping target suffix."""

    window_id: str
    corpus_start: int
    corpus_end: int
    target_start: int
    target_end: int
    prompt_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class PromptTokenLogprob:
    """The model score returned for one observed prompt token."""

    token_id: int
    logprob: float


@dataclass(frozen=True)
class PromptLikelihood:
    """Prompt scores aligned one-to-one with a scoring request."""

    window_id: str
    prompt_token_ids: tuple[int, ...]
    prompt_logprobs: tuple[PromptTokenLogprob | None, ...]


@dataclass(frozen=True)
class WindowScore:
    """The retained target suffix scores for one window."""

    window_id: str
    target_start: int
    target_end: int
    target_token_ids: tuple[int, ...]
    target_logprobs: tuple[float, ...]
    total_nll: float


@dataclass(frozen=True)
class PerplexityMetrics:
    """Complete token-weighted corpus likelihood metrics."""

    corpus_tokens: int
    scored_tokens: int
    total_nll: float
    mean_nll: float
    perplexity: float
    bits_per_token: float


@dataclass(frozen=True)
class EngineProvenance:
    """Volatile software and accelerator details."""

    python_version: str
    vllm_version: str
    torch_version: str
    cuda_version: str | None
    gpu_names: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeMetrics:
    """Volatile timing and throughput measurements."""

    engine_initialization_seconds: float
    evaluation_seconds: float
    windows_per_second: float
    prompt_tokens_per_second: float
    scored_tokens_per_second: float
    windows: int
    prompt_tokens: int
    scored_tokens: int
    engine: EngineProvenance


@dataclass(frozen=True)
class RunIdentity:
    """Stable fields that must match before paired comparison."""

    protocol: str
    source_sha256: str
    dataset_revision: str
    dataset_split: str
    text_sha256: str
    tokenizer_sha256: str
    token_ids_sha256: str
    context_length: int
    stride: int
    corpus_tokens: int
    scored_tokens: int


@dataclass(frozen=True)
class CompletedRun:
    """Validated quality data loaded from a completed run directory."""

    path: Path
    model_sha256: str
    identity: RunIdentity
    metrics: PerplexityMetrics
    scores: tuple[WindowScore, ...]


@dataclass(frozen=True)
class PerplexityComparison:
    """Aggregate and paired-token changes from baseline to candidate."""

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
