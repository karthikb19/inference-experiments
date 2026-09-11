"""Tests for evaluation-configuration failure gates."""

from pathlib import Path

import pytest

from inference_experiments.wikitext.models import EvaluationConfig


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"context_length": 1}, "context_length"),
        ({"stride": 0}, "stride"),
        ({"stride": 4096}, "stride"),
        ({"tensor_parallel_size": 0}, "tensor_parallel_size"),
        ({"batch_size": 0}, "batch_size"),
        ({"gpu_memory_utilization": 0.0}, "gpu_memory_utilization"),
        ({"limit_tokens": 1}, "limit_tokens"),
        ({"quantization": " "}, "quantization"),
    ],
)
def test_evaluation_config_rejects_invalid_values(
    overrides: dict[str, int | float | str], message: str
) -> None:
    values: dict[str, object] = {
        "model": Path("model"),
        "tokenizer": Path("tokenizer"),
        "data_dir": Path("data"),
        "output_dir": Path("output"),
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        EvaluationConfig(**values)  # type: ignore[arg-type]
