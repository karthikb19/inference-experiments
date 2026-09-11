# WikiText-103 perplexity runbook

This runbook operates the accepted
`wikitext-103-raw-qwen-token-ppl-sliding-v1` protocol. It measures the local
Qwen tokenizer's perplexity on the pinned raw WikiText corpus; it does not
claim compatibility with word-level WikiText leaderboards.

## Environment and validation

Enter the required GPU environment and activate the repository environment:

```bash
module load apptainer
gdevbox
cd ~/inference-experiments
export PATH="/root/.local/bin:$PATH"
source .venv/bin/activate
```

Before a benchmark, validate the exact commit:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

## Protocol

The loader retains every raw row and renders exact `"\n\n".join(text_rows)`.
The BF16 Qwen tokenizer runs without BOS, EOS, or chat-template tokens. The
complete test corpus contains 299,078 tokens, of which 299,077 are scored.

Input windows move by 512 tokens while remaining 4,096 tokens wide:

| Input prompt | Targets included in loss |
|---|---|
| `[0, 4096)` | `[1, 4096)` |
| `[512, 4608)` | `[4096, 4608)` |
| `[1024, 5120)` | `[4608, 5120)` |

Overlapping tokens supply context but are not counted twice. vLLM must generate
one extra token to return prompt log probabilities; the evaluator validates its
presence and ignores its identity and probability.

## Smoke and complete BF16 runs

Use a new output path for every attempt. A validation smoke run with 8,192
corpus tokens exercises nine windows:

```bash
uv run wikitext-perplexity \
  --output-dir artifacts/wikitext/qwen3-8b-bf16-smoke \
  --split validation \
  --limit-tokens 8192
```

Run the complete test split only after the smoke artifact is valid:

```bash
uv run wikitext-perplexity \
  --output-dir artifacts/wikitext/qwen3-8b-bf16
```

Do not rename a `.partial` directory to remove the suffix. Diagnose or archive
it, then rerun with a new output path.

## Artifacts

- `config.json` freezes dataset, model, tokenizer, token-plan, evaluator-source,
  window and engine inputs.
- `corpus.json` records row, byte and token counts plus comparison hashes.
- `scores.jsonl` retains each ordered target token ID and log probability.
- `metrics.json` contains complete-run mean NLL, perplexity, and bits per token.
- `runtime.json` contains volatile timing, throughput, versions, and GPU names.
- `failure.json` exists only in failed `.partial` runs and records completed
  windows and targets.

Interpret lower mean NLL, perplexity, and bits per token as better. Mean NLL is
the primary quantization comparison. Perplexity is `exp(mean_nll)`, and bits per
token is `mean_nll / log(2)`. Never average per-window perplexities.

## Quantized comparison

Always point `--tokenizer` at the BF16 tokenizer, even when the quantized
checkpoint contains tokenizer files:

```bash
uv run wikitext-perplexity \
  --model models/Qwen3-8B-quantized \
  --tokenizer models/Qwen3-8B \
  --quantization compressed-tensors \
  --output-dir artifacts/wikitext/qwen3-8b-quantized
```

Use the comparison gate rather than subtracting displayed values manually:

```bash
uv run wikitext-perplexity-compare \
  --baseline-dir artifacts/wikitext/qwen3-8b-bf16 \
  --candidate-dir artifacts/wikitext/qwen3-8b-quantized \
  --output artifacts/wikitext/bf16-vs-quantized.json
```

A positive mean-NLL delta and perplexity ratio above one are regressions. The
comparison also reports mean/max absolute token changes and counts improved,
unchanged, and worsened targets.

## Troubleshooting

- Existing output paths are never overwritten. Choose a new path or explicitly
  archive the prior run.
- A compatibility refusal means at least one quality-contract field changed.
  Rerun the candidate with the baseline split, tokenizer, context, stride, and
  token limit.
- Missing prompt scores, token-ID changes, non-finite values, reordered
  responses, gaps, and duplicate target coverage are hard failures.
- Reduce `--batch-size` if prompt-logprob responses exhaust host or GPU memory.
  Batch size affects throughput, not target selection.
- Context length and stride define the named protocol. Changing either requires
  a distinct BF16 baseline and must not be compared to the default protocol.
