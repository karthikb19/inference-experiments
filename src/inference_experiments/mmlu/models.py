"""Typed models for MMLU evaluation inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Choice = Literal["A", "B", "C", "D"]
CHOICES: tuple[Choice, Choice, Choice, Choice] = ("A", "B", "C", "D")


class EvaluationError(RuntimeError):
    """Base class for expected evaluation failures."""


class DataValidationError(EvaluationError):
    """Raised when MMLU input data does not satisfy its schema."""


class InferenceError(EvaluationError):
    """Raised when an inference response cannot be scored."""


class ArtifactError(EvaluationError):
    """Raised when a run artifact destination is unsafe to use."""


@dataclass(frozen=True)
class MMLUExample:
    """One validated MMLU row."""

    row_id: str
    source_index: int
    question: str
    subject: str
    choices: tuple[str, str, str, str]
    answer_index: int

    @property
    def answer(self) -> Choice:
        """Return the labelled correct answer."""
        return CHOICES[self.answer_index]


@dataclass(frozen=True)
class EvaluationConfig:
    """Stable inputs that define one MMLU quality run."""

    model: Path
    data_dir: Path
    output_dir: Path
    split: Literal["validation", "test"] = "test"
    tensor_parallel_size: int = 2
    max_model_len: int = 4096
    batch_size: int = 64
    gpu_memory_utilization: float = 0.9
    quantization: str | None = None
    seed: int = 0
    limit: int | None = None
    subjects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject configurations that cannot define a valid run."""
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be positive")
        if self.max_model_len < 1:
            raise ValueError("max_model_len must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.quantization is not None and not self.quantization.strip():
            raise ValueError("quantization must be non-empty when provided")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be positive when provided")


@dataclass(frozen=True)
class InferenceRequest:
    """A stable prompt submitted to an inference engine."""

    row_id: str
    prompt: str


@dataclass(frozen=True)
class InferenceAnswer:
    """One constrained engine answer and its raw A-D log probabilities."""

    row_id: str
    choice: Choice
    choice_logprobs: tuple[float, float, float, float]
    prompt_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Prediction:
    """One scored MMLU prediction."""

    row_id: str
    source_index: int
    subject: str
    expected: Choice
    predicted: Choice
    correct: bool
    choice_logprobs: tuple[float, float, float, float]
    prompt_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class AccuracyMetric:
    """Accuracy for one named slice."""

    name: str
    correct: int
    total: int
    accuracy: float


@dataclass(frozen=True)
class CategoryMetric:
    """Weighted and subject-macro accuracy for an MMLU category."""

    name: str
    correct: int
    total: int
    weighted_accuracy: float
    subject_macro_accuracy: float


@dataclass(frozen=True)
class EvaluationMetrics:
    """Complete deterministic quality metrics for a run."""

    correct: int
    total: int
    weighted_accuracy: float
    subject_macro_accuracy: float
    invalid_predictions: int
    subjects: tuple[AccuracyMetric, ...]
    categories: tuple[CategoryMetric, ...]


@dataclass(frozen=True)
class EngineProvenance:
    """Volatile runtime environment reported by an engine."""

    python_version: str
    vllm_version: str
    torch_version: str
    cuda_version: str | None
    gpu_names: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeMetrics:
    """Volatile timing and token-throughput measurements."""

    engine_initialization_seconds: float
    evaluation_seconds: float
    examples_per_second: float
    prompt_tokens_per_second: float
    output_tokens_per_second: float
    prompt_tokens: int
    output_tokens: int
    engine: EngineProvenance
