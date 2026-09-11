"""Compare compatible BF16 and quantized WikiText perplexity runs."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from inference_experiments.wikitext.artifacts import write_comparison
from inference_experiments.wikitext.comparison import compare_runs, load_completed_run


def main(arguments: Sequence[str] | None = None) -> int:
    """Validate, compare, persist, and print paired perplexity changes."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(arguments)
    baseline = load_completed_run(options.baseline_dir)
    candidate = load_completed_run(options.candidate_dir)
    comparison = compare_runs(baseline, candidate)
    write_comparison(
        options.output,
        comparison,
        baseline_run=baseline.path,
        candidate_run=candidate.path,
        baseline_model_sha256=baseline.model_sha256,
        candidate_model_sha256=candidate.model_sha256,
    )
    print(f"Mean NLL delta: {comparison.mean_nll_delta:+.6f}")
    print(f"Perplexity delta: {comparison.perplexity_delta:+.6f}")
    print(f"Perplexity ratio: {comparison.perplexity_ratio:.6f}")
    print(f"Artifact: {options.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
