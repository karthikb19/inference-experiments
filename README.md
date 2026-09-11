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
source .venv/bin/activate
uv sync
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
