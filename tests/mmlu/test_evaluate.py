"""Integration tests for deterministic evaluation and partial artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_experiments.mmlu.evaluate import run_evaluation
from inference_experiments.mmlu.models import (
    EngineProvenance,
    EvaluationConfig,
    InferenceAnswer,
    InferenceRequest,
)
from tests.mmlu.conftest import write_split


class SequenceClock:
    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class FakeEngine:
    def __init__(self, *, fail_after_calls: int | None = None) -> None:
        self.fail_after_calls = fail_after_calls
        self.calls = 0
        self.closed = False

    def predict(
        self, requests: tuple[InferenceRequest, ...]
    ) -> tuple[InferenceAnswer, ...]:
        self.calls += 1
        if self.fail_after_calls is not None and self.calls > self.fail_after_calls:
            raise RuntimeError("synthetic engine failure")
        return tuple(
            InferenceAnswer(
                row_id=request.row_id,
                choice="A",
                choice_logprobs=(0.0, -1.0, -2.0, -3.0),
                prompt_tokens=20,
                output_tokens=1,
            )
            for request in requests
        )

    def provenance(self) -> EngineProvenance:
        return EngineProvenance(
            python_version="3.12.3",
            vllm_version="0.29.0",
            torch_version="2.13.0",
            cuda_version="13.0",
            gpu_names=("GPU 0", "GPU 1"),
        )

    def close(self) -> None:
        self.closed = True


def create_inputs(tmp_path: Path) -> EvaluationConfig:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}\n')
    data = tmp_path / "data"
    choices = ["zero", "one", "two", "three"]
    write_split(
        data / "dev-00000-of-00001.parquet",
        [
            (f"Development {index}?", "abstract_algebra", choices, index % 4)
            for index in range(5)
        ],
    )
    write_split(
        data / "test-00000-of-00001.parquet",
        [
            ("Test zero?", "abstract_algebra", choices, 0),
            ("Test one?", "abstract_algebra", choices, 1),
        ],
    )
    return EvaluationConfig(
        model=model,
        data_dir=data,
        output_dir=tmp_path / "run",
        batch_size=1,
    )


def test_run_evaluation_writes_complete_quality_and_runtime_artifacts(
    tmp_path: Path,
) -> None:
    config = create_inputs(tmp_path)
    engine = FakeEngine()

    metrics = run_evaluation(
        config,
        lambda: engine,
        clock=SequenceClock((0.0, 2.0, 3.0, 5.0)),
    )

    assert engine.closed
    assert metrics.correct == 1
    assert metrics.total == 2
    assert metrics.weighted_accuracy == 0.5
    assert config.output_dir.is_dir()
    assert not config.output_dir.with_name("run.partial").exists()
    prediction_lines = (
        (config.output_dir / "predictions.jsonl").read_text().splitlines()
    )
    assert len(prediction_lines) == 2
    assert json.loads(prediction_lines[0])["row_id"] == "abstract_algebra:0"
    quality = json.loads((config.output_dir / "metrics.json").read_text())
    runtime = json.loads((config.output_dir / "runtime.json").read_text())
    assert quality["weighted_accuracy"] == 0.5
    assert "evaluation_seconds" not in quality
    assert runtime["evaluation_seconds"] == 2.0
    assert runtime["examples_per_second"] == 1.0
    assert runtime["engine"]["gpu_names"] == ["GPU 0", "GPU 1"]


def test_run_evaluation_retains_partial_batch_output_on_failure(tmp_path: Path) -> None:
    config = create_inputs(tmp_path)
    engine = FakeEngine(fail_after_calls=1)

    with pytest.raises(RuntimeError, match="synthetic engine failure"):
        run_evaluation(
            config,
            lambda: engine,
            clock=SequenceClock((0.0, 1.0, 2.0)),
        )

    partial_dir = config.output_dir.with_name("run.partial")
    assert engine.closed
    assert partial_dir.is_dir()
    assert not config.output_dir.exists()
    assert len((partial_dir / "predictions.jsonl").read_text().splitlines()) == 1
    failure = json.loads((partial_dir / "failure.json").read_text())
    assert failure == {
        "completed_predictions": 1,
        "error_type": "RuntimeError",
        "message": "synthetic engine failure",
    }
    assert not (partial_dir / "metrics.json").exists()
