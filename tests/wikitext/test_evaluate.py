"""Integration tests for complete and failed perplexity runs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from inference_experiments.wikitext.evaluate import run_evaluation
from inference_experiments.wikitext.models import (
    ArtifactError,
    EngineProvenance,
    EvaluationConfig,
    PromptLikelihood,
    PromptTokenLogprob,
    ScoringWindow,
)
from tests.wikitext.conftest import write_text_shard


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert text == "source text"
        assert not add_special_tokens
        return [10, 11, 12, 13, 14, 15, 16]


class SequenceClock:
    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class FakeEngine:
    def __init__(self, logprob: float, fail_after_calls: int | None = None) -> None:
        self.logprob = logprob
        self.fail_after_calls = fail_after_calls
        self.calls = 0
        self.closed = False

    def score(
        self, requests: tuple[ScoringWindow, ...]
    ) -> tuple[PromptLikelihood, ...]:
        self.calls += 1
        if self.fail_after_calls is not None and self.calls > self.fail_after_calls:
            raise RuntimeError("synthetic likelihood failure")
        return tuple(
            PromptLikelihood(
                window_id=request.window_id,
                prompt_token_ids=request.prompt_token_ids,
                prompt_logprobs=(None,)
                + tuple(
                    PromptTokenLogprob(token_id, self.logprob)
                    for token_id in request.prompt_token_ids[1:]
                ),
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


def create_config(tmp_path: Path, name: str = "run") -> EvaluationConfig:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text('{"model_type":"qwen3"}\n')
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir(exist_ok=True)
    (tokenizer / "tokenizer.json").write_text("{}\n")
    data = tmp_path / "data"
    write_text_shard(data / "test-00000-of-00001.parquet", ["source text"])
    return EvaluationConfig(
        model=model,
        tokenizer=tokenizer,
        data_dir=data,
        output_dir=tmp_path / name,
        context_length=4,
        stride=2,
        batch_size=1,
    )


def test_run_evaluation_writes_complete_separated_artifacts(tmp_path: Path) -> None:
    config = create_config(tmp_path)
    engine = FakeEngine(-math.log(2))

    metrics = run_evaluation(
        config,
        lambda: engine,
        tokenizer_factory=lambda _: FakeTokenizer(),
        clock=SequenceClock((0.0, 2.0, 3.0, 5.0)),
    )

    assert engine.closed
    assert metrics.perplexity == pytest.approx(2)
    assert metrics.scored_tokens == 6
    assert len((config.output_dir / "scores.jsonl").read_text().splitlines()) == 3
    quality = json.loads((config.output_dir / "metrics.json").read_text())
    runtime = json.loads((config.output_dir / "runtime.json").read_text())
    assert "evaluation_seconds" not in quality
    assert runtime["evaluation_seconds"] == 2.0
    assert runtime["scored_tokens_per_second"] == 3.0
    assert not config.output_dir.with_name("run.partial").exists()


def test_run_evaluation_retains_partial_scores_without_metrics(tmp_path: Path) -> None:
    config = create_config(tmp_path)
    engine = FakeEngine(-1.0, fail_after_calls=1)

    with pytest.raises(RuntimeError, match="synthetic likelihood failure"):
        run_evaluation(
            config,
            lambda: engine,
            tokenizer_factory=lambda _: FakeTokenizer(),
            clock=SequenceClock((0.0, 1.0, 2.0)),
        )

    partial = config.output_dir.with_name("run.partial")
    assert engine.closed
    assert len((partial / "scores.jsonl").read_text().splitlines()) == 1
    assert not (partial / "metrics.json").exists()
    failure = json.loads((partial / "failure.json").read_text())
    assert failure["completed_windows"] == 1
    assert failure["completed_scored_tokens"] == 3


def test_run_evaluation_refuses_to_overwrite_completed_run(tmp_path: Path) -> None:
    config = create_config(tmp_path)
    run_evaluation(
        config,
        lambda: FakeEngine(-1.0),
        tokenizer_factory=lambda _: FakeTokenizer(),
    )

    with pytest.raises(ArtifactError, match="completed output already exists"):
        run_evaluation(
            config,
            lambda: FakeEngine(-1.0),
            tokenizer_factory=lambda _: FakeTokenizer(),
        )
