"""Serve the local Qwen3-8B checkpoint through vLLM on two GPUs."""

from __future__ import annotations

import os
from argparse import ArgumentParser
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServeConfig:
    """Validated inputs for the vLLM OpenAI-compatible server."""

    model: Path = Path("models/Qwen3-8B")
    served_model_name: str = "qwen3-8b"
    tensor_parallel_size: int = 2
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    quantization: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000

    def __post_init__(self) -> None:
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be positive")
        if self.max_model_len < 1:
            raise ValueError("max_model_len must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.quantization is not None and not self.quantization.strip():
            raise ValueError("quantization must be non-empty when provided")
        if self.port not in range(1, 65536):
            raise ValueError("port must be between 1 and 65535")


def build_serve_command(config: ServeConfig) -> tuple[str, ...]:
    """Build the explicit vLLM server command."""
    command = (
        "vllm",
        "serve",
        str(config.model),
        "--served-model-name",
        config.served_model_name,
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--max-model-len",
        str(config.max_model_len),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--host",
        config.host,
        "--port",
        str(config.port),
    )
    if config.quantization is not None:
        command += ("--quantization", config.quantization)
    return command


def parse_config(arguments: Sequence[str] | None = None) -> ServeConfig:
    """Parse server options with two-GPU local defaults."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-8B"))
    parser.add_argument("--served-model-name", default="qwen3-8b")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--quantization")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    options = parser.parse_args(arguments)
    return ServeConfig(
        model=options.model,
        served_model_name=options.served_model_name,
        tensor_parallel_size=options.tensor_parallel_size,
        max_model_len=options.max_model_len,
        gpu_memory_utilization=options.gpu_memory_utilization,
        quantization=options.quantization,
        host=options.host,
        port=options.port,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Replace this process with the configured vLLM server."""
    command = build_serve_command(parse_config(arguments))
    os.execvp(command[0], command)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
