"""Tests for MMLU command-line defaults."""

from inference_experiments.mmlu.cli import parse_config


def test_parse_config_defaults_to_two_gpu_test_evaluation() -> None:
    config = parse_config(["--output-dir", "artifacts/test-run"])

    assert config.tensor_parallel_size == 2
    assert config.split == "test"
    assert config.batch_size == 64
    assert config.subjects == ()
