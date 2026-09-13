"""Tests for WikiText perplexity command-line defaults."""

from inference_experiments.wikitext.cli import parse_config


def test_parse_config_defaults_to_accepted_two_gpu_protocol() -> None:
    config = parse_config(["--output-dir", "artifacts/wikitext/test-run"])

    assert config.split == "test"
    assert config.context_length == 4096
    assert config.stride == 512
    assert config.max_model_len == 4097
    assert config.tensor_parallel_size == 2
    assert config.tokenizer == config.model


def test_parse_config_accepts_fake_quantization() -> None:
    config = parse_config(
        [
            "--output-dir",
            "artifacts/wikitext/test-run",
            "--quantization",
            "int8-fake-quant",
        ]
    )

    assert config.quantization == "int8-fake-quant"


def test_parse_config_accepts_int4_fake_quantization() -> None:
    config = parse_config(
        ["--output-dir", "artifacts/test-run", "--quantization", "int4-fake-quant"]
    )

    assert config.quantization == "int4-fake-quant"
