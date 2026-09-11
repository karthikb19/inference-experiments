"""Run in-process WikiText-103 Qwen-token perplexity evaluation."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from inference_experiments.wikitext.engine import VLLMLikelihoodEngine
from inference_experiments.wikitext.evaluate import run_evaluation
from inference_experiments.wikitext.models import EvaluationConfig


def parse_config(arguments: Sequence[str] | None = None) -> EvaluationConfig:
    """Parse the complete perplexity contract from command-line arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-8B"))
    parser.add_argument("--tokenizer", type=Path, default=Path("models/Qwen3-8B"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/wikitext/wikitext-103-raw-v1"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--quantization")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-tokens", type=int)
    options = parser.parse_args(arguments)
    return EvaluationConfig(
        model=options.model,
        tokenizer=options.tokenizer,
        data_dir=options.data_dir,
        output_dir=options.output_dir,
        split=options.split,
        context_length=options.context_length,
        stride=options.stride,
        tensor_parallel_size=options.tensor_parallel_size,
        batch_size=options.batch_size,
        gpu_memory_utilization=options.gpu_memory_utilization,
        quantization=options.quantization,
        seed=options.seed,
        limit_tokens=options.limit_tokens,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Run perplexity and print the primary quality metrics."""
    config = parse_config(arguments)

    def create_engine() -> VLLMLikelihoodEngine:
        return VLLMLikelihoodEngine(
            model=str(config.model),
            tokenizer=str(config.tokenizer),
            tensor_parallel_size=config.tensor_parallel_size,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            quantization=config.quantization,
            seed=config.seed,
        )

    metrics = run_evaluation(config, create_engine)
    print(f"WikiText mean NLL: {metrics.mean_nll:.6f}")
    print(f"WikiText perplexity: {metrics.perplexity:.6f}")
    print(f"WikiText bits/token: {metrics.bits_per_token:.6f}")
    print(f"Scored tokens: {metrics.scored_tokens}")
    print(f"Artifacts: {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
