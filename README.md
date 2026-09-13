# inference-experiments

Experiments quantizing models, training speculative-decoding models, and
measuring inference behavior.

## Download Qwen3-8B

Install the project and development dependencies with:

```bash
uv sync
```

Then download the model snapshot from Hugging Face Hub:

```bash
uv run python scripts/download_qwen3_8b.py
```

The files are stored in `models/Qwen3-8B` by default. Use `--output-dir` to
choose a different location, or `--token` for a private or gated repository.

## gdevbox environment

Run Python and GPU commands inside gdevbox:

```bash
module load apptainer
gdevbox
cd ~/inference-experiments
export PATH="/root/.local/bin:$PATH"
uv sync
source .venv/bin/activate
```

If the venv Python points to a host `/sw/...` path that is unavailable inside
the container, recreate it once before syncing:

```bash
uv venv --clear .venv --python /usr/bin/python3
source .venv/bin/activate
uv sync
```

## Serve Qwen3-8B with vLLM

The serving command defaults to tensor parallelism across two GPUs and exposes
an OpenAI-compatible API on `127.0.0.1:8000`:

```bash
uv run serve-qwen3-8b
```

Model path, served name, port, maximum context length, GPU memory utilization,
and tensor-parallel size are configurable CLI options. For example:

```bash
uv run serve-qwen3-8b --port 8030 --max-model-len 8192
```

## Run the MMLU baseline

The evaluator runs vLLM directly in the Python process, uses the original
five-shot MMLU prompt and A-D probability scoring, and defaults to two GPUs.
See the [MMLU baseline runbook](knowledge/runbooks/mmlu-baseline.md) for complete
gdevbox setup, validation, artifact inspection, and troubleshooting guidance.
Start with an explicitly named smoke artifact:

```bash
uv run mmlu-baseline \
  --output-dir artifacts/mmlu/qwen3-8b-bf16-smoke \
  --subjects abstract_algebra \
  --limit 10
```

Run all 14,042 test questions with:

```bash
uv run mmlu-baseline \
  --output-dir artifacts/mmlu/qwen3-8b-bf16
```

Each successful output directory contains:

- `config.json` — stable dataset, model, prompt, and engine configuration;
- `predictions.jsonl` — stable row-level answers and A-D log probabilities;
- `metrics.json` — weighted overall, subject, and category accuracy;
- `runtime.json` — volatile timings, token throughput, versions, and GPUs.

The primary score is weighted accuracy: total correct divided by total test
questions. If a run fails, its `<output>.partial` directory and completed batch
predictions remain for diagnosis; incomplete runs never receive finalized
aggregate metrics. Generated artifacts and model checkpoints are not committed.

To compare a quantized checkpoint, change `--model` and give it a distinct
output directory while keeping the remaining evaluation options identical.

## Run the INT8 fake-quant experiment

The `int8-fake-quant` vLLM plugin loads the ordinary BF16 checkpoint, rounds
every vLLM linear weight row through symmetric signed INT8 with one FP32 scale
per output channel, stores the reconstructed values as BF16, and then uses the
ordinary BF16 matrix multiplication. It does not quantize activations,
embeddings, attention kernels, the KV cache, or the vocabulary head. Attention
QKV and output projections are included because they are linear layers.

Run paired MMLU and WikiText candidates without creating another checkpoint:

```bash
uv run mmlu-baseline \
  --quantization int8-fake-quant \
  --output-dir artifacts/mmlu/qwen3-8b-int8-fake-quant

uv run wikitext-perplexity \
  --quantization int8-fake-quant \
  --output-dir artifacts/wikitext/qwen3-8b-int8-fake-quant
```

Keep all non-quantization options equal to the BF16 baseline. The fake-quant
pass runs once after each tensor-parallel shard is loaded; doing the same
deterministic round trip before every forward would produce the same BF16
weights but would add repeated quantization overhead unrelated to quality.
Row-parallel layers synchronize their maxima across tensor-parallel workers,
so each scale represents a row of the original unsharded matrix.

The serving wrapper accepts the same method when interactive inspection is
useful:

```bash
uv run serve-qwen3-8b --quantization int8-fake-quant
```

## Run WikiText-103 perplexity

The WikiText evaluator uses the accepted Hugging Face-style strided-window
protocol with in-process vLLM prompt log probabilities. It defaults to a
4,096-token context, 512-token stride, and two GPUs. Start with a bounded smoke
run:

```bash
uv run wikitext-perplexity \
  --output-dir artifacts/wikitext/qwen3-8b-bf16-smoke \
  --split validation \
  --limit-tokens 8192
```

Run the complete 299,077-target test baseline with:

```bash
uv run wikitext-perplexity \
  --output-dir artifacts/wikitext/qwen3-8b-bf16
```

Every target token ID and prompt log probability is retained in
`scores.jsonl`. Aggregate mean NLL, perplexity, and bits per token are written
to `metrics.json`; volatile timings and environment details stay in
`runtime.json`. Failed runs remain under `<output>.partial` without aggregate
metrics.

Quantized runs must keep the BF16 tokenizer and every evaluation option fixed:

```bash
uv run wikitext-perplexity \
  --model models/Qwen3-8B-quantized \
  --tokenizer models/Qwen3-8B \
  --quantization compressed-tensors \
  --output-dir artifacts/wikitext/qwen3-8b-quantized

uv run wikitext-perplexity-compare \
  --baseline-dir artifacts/wikitext/qwen3-8b-bf16 \
  --candidate-dir artifacts/wikitext/qwen3-8b-quantized \
  --output artifacts/wikitext/bf16-vs-quantized.json
```

The comparison command refuses different corpus, tokenizer, token-plan,
context, stride, split, or scored-token contracts. See the
[WikiText perplexity runbook](knowledge/runbooks/wikitext-perplexity.md) for
metric interpretation, validation, and troubleshooting.
