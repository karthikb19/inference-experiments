# MMLU baseline runbook

This runbook explains how to evaluate the local Qwen3-8B checkpoint with vLLM,
starting with a small smoke test and progressing to the complete MMLU baseline.
GPU and Python commands must run inside `gdevbox`; running `uv` on the host can
select a host Python that is unavailable inside the container.

## Prerequisites

- Run on a CS GPU server with at least two available GPUs.
- Keep the model at `models/Qwen3-8B` and MMLU parquet files at
  `data/mmlu/all`.
- Use the revision containing the MMLU implementation. Before the feature is
  merged, the primary checkout can inspect it with:

  ```bash
  git switch --detach kb/vllm-mmlu-baseline
  ```

  Detached mode avoids conflicting with the branch's existing development
  worktree. After testing, return to the previous branch with `git switch main`.

## Enter the GPU environment

Run these commands from the host:

```bash
cd ~/inference-experiments
module load apptainer
gdevbox
```

The repository is mounted at `/root/inference-experiments` inside the
container:

```bash
cd /root/inference-experiments
export PATH="/root/.local/bin:$PATH"
```

`uv` is installed at `/root/.local/bin/uv`, but that directory is not always on
the default `PATH`.

## Create or repair the environment

For the first setup, or if `.venv/bin/python` is absent or refers to a host
Python, recreate the disposable virtual environment with the container's
Python 3.12:

```bash
uv venv --clear .venv --python /usr/bin/python3
uv sync
source .venv/bin/activate
```

For later sessions, activate the existing environment:

```bash
source .venv/bin/activate
```

Confirm that the container environment is active:

```bash
python --version
which mmlu-baseline
which vllm
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
```

Python should report 3.12. Do not continue if it reports the host's Python 3.14
or resolves Python under `/u/` or `/sw/`.

Once the environment is active, invoke `mmlu-baseline`, `pytest`, and `vllm`
directly. `uv run` is not required.

## Validation ladder

### 1. Offline checks

These tests do not load the model or require GPUs:

```bash
pytest
ruff format --check .
ruff check .
```

### 2. Two-question GPU smoke test

Restrict vLLM to two GPUs and evaluate two questions from one subject:

```bash
CUDA_VISIBLE_DEVICES=0,1 mmlu-baseline \
  --output-dir artifacts/mmlu/qwen3-8b-smoke-2 \
  --subjects abstract_algebra \
  --limit 2
```

This exercises local parquet loading, five-shot prompt construction, two-GPU
tensor parallelism, answer-token probability scoring, metrics, and artifact
writing. The first invocation can take longer while vLLM loads weights and
compiles kernels. Accuracy from two questions is not a useful model-quality
measurement.

The implementation was initially validated with this equivalent smoke run. It
completed with one correct answer out of two and 699 prompt tokens. That result
is evidence that the pipeline worked, not a baseline score.

### 3. Complete one subject

Run all test examples from one subject before committing to the full suite:

```bash
CUDA_VISIBLE_DEVICES=0,1 mmlu-baseline \
  --output-dir artifacts/mmlu/qwen3-8b-abstract-algebra \
  --subjects abstract_algebra
```

### 4. Complete MMLU baseline

Omitting `--subjects` and `--limit` evaluates all 57 subjects and 14,042 test
questions:

```bash
CUDA_VISIBLE_DEVICES=0,1 mmlu-baseline \
  --output-dir artifacts/mmlu/qwen3-8b-bf16-full
```

The evaluator defaults to `--tensor-parallel-size 2`, a maximum model length
of 4096 tokens, and batches of 64. Record any non-default arguments when
comparing this result with a quantized model.

## Inspect results

A successful output contains four files:

```bash
python -m json.tool artifacts/mmlu/qwen3-8b-smoke-2/config.json
python -m json.tool artifacts/mmlu/qwen3-8b-smoke-2/metrics.json
python -m json.tool artifacts/mmlu/qwen3-8b-smoke-2/runtime.json
sed -n '1,2p' artifacts/mmlu/qwen3-8b-smoke-2/predictions.jsonl
```

- `config.json` records stable model, dataset, prompt, and engine provenance.
- `predictions.jsonl` records each expected answer, prediction, correctness,
  and A-D log probabilities.
- `metrics.json` records example-weighted overall accuracy, macro subject
  accuracy, and subject/category breakdowns.
- `runtime.json` records volatile timing, throughput, package, and GPU data.

Evaluation results and runtime results are separate so model-quality
comparisons do not depend on a particular server load or hardware timing.

## Failed and repeated runs

The evaluator never overwrites an existing output. Give each attempt a unique
output directory. If evaluation fails, it preserves completed predictions and
failure metadata under `<output-dir>.partial`. Inspect that directory before
moving it aside or choosing a new output name.

Final aggregate metrics are only written for successful runs. A partial run is
diagnostic evidence and must not be reported as a completed baseline.

## Serve Qwen3-8B normally with vLLM

MMLU evaluation uses vLLM in-process and does not require an HTTP server. For
interactive inference or an OpenAI-compatible endpoint, run:

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve models/Qwen3-8B \
  --served-model-name qwen3-8b \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --host 127.0.0.1 \
  --port 8000
```

The project wrapper supplies the same defaults:

```bash
CUDA_VISIBLE_DEVICES=0,1 serve-qwen3-8b
```

From another terminal on the same server, check health and inference:

```bash
curl -f http://127.0.0.1:8000/health
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-8b",
    "messages": [{
      "role": "user",
      "content": "/no_think\nReply with exactly: vLLM works"
    }],
    "temperature": 0,
    "max_tokens": 32
  }'
```

Stop the server with `Ctrl-C` when finished.

## Common failures

### `Using CPython 3.14` appears

The command is running on the host. Enter `gdevbox`, recreate `.venv` with
`/usr/bin/python3`, and retry from inside the container.

### `uv: command not found`

Add the container installation to `PATH`:

```bash
export PATH="/root/.local/bin:$PATH"
```

### `.venv/bin/python: No such file or directory`

The shared environment is incomplete or was replaced by host `uv`. Recreate it
with the commands in "Create or repair the environment."

### CUDA out-of-memory error

Confirm that only the intended GPUs are selected and are not occupied by other
jobs. If necessary, reduce `--gpu-memory-utilization` or `--batch-size`; record
the changed configuration with the result.

### Output directory already exists

Choose a new output name. This safeguard prevents an old baseline from being
silently replaced.
