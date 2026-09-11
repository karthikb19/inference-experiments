# Qwen3-8B vLLM MMLU baseline implementation plan

## Goal

Add a reproducible, offline MMLU evaluation for the local `models/Qwen3-8B`
checkpoint. The baseline must be easy to repeat for a quantized checkpoint and
must preserve enough run metadata and row-level output to explain an accuracy
or performance delta.

## Verified starting point

- The checkpoint is Qwen3-8B in BF16, has a 40,960-token configured context,
  and occupies 15.26 GiB on disk.
- The pinned `cais/mmlu` snapshot has 14,042 test questions in 57 subjects,
  285 dev questions (five per subject), and 1,531 validation questions.
- Each parquet row has `question: string`, `subject: string`,
  `choices: list[string]`, and `answer: int64`.
- In gdevbox on `gpusrv02`, vLLM 0.29.0 and Torch 2.13.0+cu130 see four
  RTX 4000 Ada GPUs. A one-GPU eager-mode smoke test loaded the local checkpoint
  in BF16, used 15.27 GiB for weights, created a 2.26 GiB KV cache, and generated
  `vLLM works` from a deterministic prompt.
- The original `.venv` referred to an unavailable host `/sw/...` interpreter.
  Recreating it inside gdevbox with `/usr/bin/python3` produced the required
  `/root/inference-experiments/.venv/bin/python` environment.

## Changes included in this feasibility spike

- `pyproject.toml` now declares vLLM 0.29.x and PyArrow 25.x as runtime
  dependencies.
- `uv.lock` captures the resolved GPU/runtime dependency graph, including the
  exact vLLM 0.29.0 and PyArrow 25.0.1 versions validated in gdevbox.
- This plan and the proposed ADR record the implementation boundary. No MMLU
  evaluator or full benchmark result is claimed yet.

## Evaluation protocol

Use the canonical five-shot MMLU shape: prepend the five same-subject dev rows,
render the test question with choices labelled A through D, and end at the
answer position. Run Qwen3 in non-thinking mode and choose exactly one of the
four answer tokens with greedy decoding. Record the four choice log
probabilities when vLLM exposes them.

The protocol is intentionally fixed for baseline-to-quantized comparisons:

- test is the scored split; dev supplies demonstrations; validation is only for
  development smoke runs;
- seed is zero, temperature is zero, and one answer token is generated;
- the prompt template, tokenizer fingerprint, dataset revision, model path and
  checkpoint fingerprint are part of the run manifest;
- row order is stable and each prediction has a stable
  `<subject>:<source-row-index>` identifier;
- malformed rows, missing subjects, non-four-choice rows, missing answer-token
  log probabilities, and unparseable answers fail the run explicitly;
- partial artifacts are written to a temporary run directory and renamed only
  after all rows and metrics are complete.

Before implementation, compare a small fixture against the EleutherAI
lm-evaluation-harness MMLU prompt format. This project should not depend on the
harness at runtime, but matching a widely used prompt prevents an accidental
project-specific baseline.

## Reported metrics

The primary quality metric is micro accuracy over all 14,042 test questions.
Also report:

- correct, total, and accuracy for each of the 57 subjects;
- micro accuracy for the standard STEM, humanities, social-sciences, and other
  subject groups;
- macro accuracy across subjects, plus macro accuracy within each group;
- invalid prediction count, which must be zero for a successful run;
- engine initialization seconds, evaluation wall seconds, examples/second,
  prompt tokens/second, output tokens/second, and total prompt/output tokens;
- runtime provenance: vLLM, Torch, CUDA and Python versions, visible GPU names,
  tensor-parallel size, dtype, max model length, batch settings, seed, and Git
  commit.

Do not call timing numbers “latency” because the offline runner batches work.
If online serving latency is needed later, add a separate load-test protocol
with concurrency, warm-up and percentile definitions.

## Proposed code shape

Production code remains under `src/inference_experiments/`; scripts are thin
entry points only. Structured records use frozen dataclasses or Pydantic models,
not bare dictionaries.

```python
def load_mmlu_split(path: Path) -> tuple[MMLUExample, ...]: ...

def render_five_shot_prompt(
    example: MMLUExample,
    demonstrations: tuple[MMLUExample, ...],
) -> str: ...

def evaluate_mmlu(config: EvaluationConfig, engine: InferenceEngine) -> RunReport: ...

def aggregate_metrics(
    predictions: tuple[Prediction, ...],
) -> EvaluationMetrics: ...
```

Keep vLLM behind an `InferenceEngine` protocol. Unit tests use a deterministic
fake engine, so imports and tests do not initialize CUDA. The concrete
`VLLMEngine` performs local batched inference. This seam also lets a later
quantized engine use the identical data, prompts, answer decoding and metrics.

## Files to add

- `src/inference_experiments/mmlu/__init__.py` — public evaluation types and
  entry points.
- `src/inference_experiments/mmlu/models.py` — typed examples, configuration,
  predictions, timing/provenance, metrics and report models.
- `src/inference_experiments/mmlu/data.py` — strict parquet loading and split
  validation.
- `src/inference_experiments/mmlu/prompting.py` — canonical five-shot prompt
  rendering and A–D token validation.
- `src/inference_experiments/mmlu/engine.py` — `InferenceEngine` protocol and
  lazy vLLM adapter.
- `src/inference_experiments/mmlu/evaluate.py` — batching, timing, stable row
  IDs, failure handling and artifact finalization.
- `src/inference_experiments/mmlu/metrics.py` — overall, subject and standard
  category aggregation.
- `src/inference_experiments/mmlu/cli.py` — typed CLI configuration and exit
  codes for the offline baseline.
- `src/inference_experiments/vllm_serve.py` — small typed wrapper around the
  OpenAI-compatible `vllm serve` command with safe local defaults.
- `tests/mmlu/test_data.py` — schemas, row validation, ordering and failures.
- `tests/mmlu/test_prompting.py` — exact prompt golden tests, escaping and token
  assumptions.
- `tests/mmlu/test_evaluate.py` — fake-engine batching, deterministic output,
  failure behavior and atomic artifacts.
- `tests/mmlu/test_metrics.py` — micro/macro/category arithmetic and empty or
  invalid inputs.
- `tests/test_vllm_serve.py` — serving argument construction without launching
  a server or GPU work.
- `tests/fixtures/mmlu/` — tiny synthetic parquet fixtures covering multiple
  subjects and malformed cases.

## Additional files to update during implementation

- `pyproject.toml` — add console entry points such as `mmlu-baseline` and
  `serve-qwen3-8b`; vLLM and PyArrow dependencies are already included.
- `uv.lock` — refresh if entry points or dependency constraints change.
- `.gitignore` — ignore generated `artifacts/` output.
- `README.md` — document gdevbox setup, serving, smoke evaluation, full
  evaluation, output artifacts and quantized-model substitution.
- `data/mmlu/README.md` — document split roles and the dataset revision used in
  the manifest.

No checkpoint or generated result belongs in Git.

## CLI sketch

```bash
mmlu-baseline \
  --model models/Qwen3-8B \
  --data-dir data/mmlu/all \
  --output-dir artifacts/mmlu/qwen3-8b-bf16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --batch-size 64

serve-qwen3-8b \
  --model models/Qwen3-8B \
  --served-model-name qwen3-8b \
  --tensor-parallel-size 1
```

The offline evaluator should also accept `--limit` and `--subjects` for smoke
runs. A quantized run changes the model path and explicit quantization engine
options, not the prompt or scoring protocol.

## Delivery sequence

1. Resolve the ADR questions and lock dependency/version policy.
2. Add typed data models, strict parquet loading and tiny offline fixtures.
3. Add prompt rendering with exact golden tests and tokenizer answer-token
   validation against the local Qwen3 tokenizer.
4. Add the fake engine, evaluator, metrics and atomic JSON/JSONL artifacts.
5. Add the lazy vLLM adapter and serving wrapper.
6. Run a validation-split smoke evaluation, then the full test split in
   gdevbox.
7. Run `uv run ruff format --check .`, `uv run ruff check .`, and
   `uv run pytest` in gdevbox; record exact commands, hardware, run artifact and
   metrics in the PR.

## Acceptance criteria

- A fresh gdevbox `uv sync` installs vLLM and the parquet reader from the lock.
- The serving command starts an OpenAI-compatible endpoint for the local model
  and a deterministic request succeeds.
- The offline smoke command scores a selected subject without network access.
- Two runs with identical inputs have byte-identical quality outputs; timing
  and environment fields may differ and are isolated from that comparison.
- A BF16 and quantized report can be joined by stable row ID to calculate
  accuracy deltas and changed answers.
- All unit tests are deterministic, fast, offline, and do not initialize Torch
  or CUDA.
