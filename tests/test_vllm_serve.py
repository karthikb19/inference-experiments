"""Tests for the two-GPU vLLM serving wrapper."""

import pytest

from inference_experiments.vllm_serve import ServeConfig, build_serve_command


def test_build_serve_command_uses_two_gpu_local_defaults() -> None:
    command = build_serve_command(ServeConfig())

    assert command[:3] == ("vllm", "serve", "models/Qwen3-8B")
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert command[command.index("--host") + 1] == "127.0.0.1"


def test_serve_config_rejects_invalid_port() -> None:
    with pytest.raises(ValueError, match="port"):
        ServeConfig(port=0)


def test_build_serve_command_includes_fake_quantization() -> None:
    command = build_serve_command(ServeConfig(quantization="int8-fake-quant"))

    assert command[-2:] == ("--quantization", "int8-fake-quant")
