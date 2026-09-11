# Qwen3-8B BF16 WikiText-103 perplexity

## Result

The complete `wikitext-103-raw-qwen-token-ppl-sliding-v1` test run produced:

| Metric | Value |
|---|---:|
| Corpus tokens | 299,078 |
| Scored tokens | 299,077 |
| Total NLL | 632,044.801767 |
| Mean NLL | 2.113317981 |
| Perplexity | **8.275654244** |
| Bits per token | 3.048873371 |

Mean NLL is the primary future quantization comparison. A candidate with a
positive `candidate - BF16` mean-NLL delta is worse; its perplexity ratio is
exactly `exp(mean_nll_delta)`.

## Protocol and provenance

- Dataset: pinned `Salesforce/wikitext` `wikitext-103-raw-v1` test split at
  revision `b08601e04326c79dfdd32d625aee71d232d685c3`.
- Rendering: exact `"\n\n".join(text_rows)`, retaining empty rows and all raw
  WikiText whitespace/placeholders.
- Tokenizer: local `models/Qwen3-8B`, without added special tokens.
- Context and targets: 4,096-token prompts, 512-token stride, 578 windows, and
  every eligible target scored once.
- Runtime: in-process vLLM 0.29.0, BF16, tensor parallel size 2, batch size 8,
  seed 0, and GPU memory utilization 0.9.
- Hardware: two of four visible NVIDIA RTX 4000 Ada Generation GPUs.
- Git commit: `f174961ea87012a5aedfe7153dc3512f2c064f1d`.
- Evaluator source SHA-256:
  `a97ab860da4c48c45ae33b8cadb4c2f5ccf8de34ad79c0e1689bc0b1394b6258`.
- Model SHA-256:
  `ac81fb6641aeb3b66128a9c24602b4123a76b86f33d10653efe9813e61e39195`.
- Tokenizer SHA-256:
  `7a50b53b219d9620306d5534cbe74327e4cfb72faf5578fc52fe56f9bc930791`.
- Corpus-text SHA-256:
  `696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83`.
- Token-plan SHA-256:
  `faf5b857d23194c6adb3acf87aae15fdfd9c2635c39a7bfab4d153889e8f415e`.

The ignored one-token decode required by vLLM is not included in any metric or
score artifact.

## Runtime

Engine initialization took 43.35 seconds. Scoring took 440.02 seconds and
processed 2,367,488 prompt tokens at 5,380.41 prompt tokens/second. The target
rate was 679.69 scored tokens/second. `scores.jsonl` retains all 299,077 target
IDs and log probabilities in 7,510,331 bytes.

The successful local artifact is
`artifacts/wikitext/qwen3-8b-bf16/`. Artifacts are intentionally ignored by Git;
this report preserves the result and comparison-critical provenance.

## Validation

- `uv run ruff format --check .`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: 57 passed.
- An 8,192-token validation smoke crossed all nine planned windows and scored
  8,191 targets without gaps or duplicates.
- A separate Transformers BF16 forward pass checked the first 15 observed-token
  log probabilities. Against two-GPU vLLM, mean absolute difference was
  0.021203 and maximum absolute difference was 0.094869, within the recorded
  0.1 smoke tolerance for different fused/tensor-parallel kernels.
- The completed-run loader reread all 578 score records, recomputed aggregate
  metrics, verified cross-artifact hashes/counts, and accepted the canonical
  artifact.
- A second complete pre-commit run produced identical token scores and metrics;
  it is retained locally as
  `artifacts/wikitext/qwen3-8b-bf16-provisional-precommit/`.

## Interpretation and remaining work

This is Qwen-token perplexity under the repository's accepted finite-context
protocol. It is not directly comparable to word-level WikiText-103 leaderboard
numbers or runs using different tokenizers, rendering, special tokens, context
lengths, or strides.

The BF16 baseline is proven. Quantized perplexity and the BF16-to-quantized mean
NLL/perplexity deltas remain **UNMEASURED** until a quantized checkpoint is
provided and evaluated through the same token plan.
