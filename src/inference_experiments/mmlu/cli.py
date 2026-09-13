"""Command-line entry point for the in-process MMLU baseline."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from inference_experiments.mmlu.engine import VLLMEngine
from inference_experiments.mmlu.evaluate import run_evaluation
from inference_experiments.mmlu.models import EvaluationConfig


def parse_config(arguments: Sequence[str] | None = None) -> EvaluationConfig:
    """Parse a complete evaluation contract from CLI arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-8B"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/mmlu/all"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--quantization")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--subjects", nargs="*", default=())
    options = parser.parse_args(arguments)
    return EvaluationConfig(
        model=options.model,
        data_dir=options.data_dir,
        output_dir=options.output_dir,
        split=options.split,
        tensor_parallel_size=options.tensor_parallel_size,
        max_model_len=options.max_model_len,
        batch_size=options.batch_size,
        gpu_memory_utilization=options.gpu_memory_utilization,
        quantization=options.quantization,
        seed=options.seed,
        limit=options.limit,
        subjects=tuple(options.subjects),
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Evaluate the local model and print its headline weighted accuracy."""
    config = parse_config(arguments)

    def create_engine() -> VLLMEngine:
        return VLLMEngine(
            model=str(config.model),
            tensor_parallel_size=config.tensor_parallel_size,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            quantization=config.quantization,
            seed=config.seed,
        )

    metrics = run_evaluation(config, create_engine)
    print(
        f"MMLU weighted accuracy: {metrics.weighted_accuracy:.4f} "
        f"({metrics.correct}/{metrics.total})"
    )
    print(f"Artifacts: {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
