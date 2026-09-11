# ADR — Reproducible vLLM MMLU baselines

| | |
|---|---|
| **Date** | 2026-09-10 |
| **Status** | Proposed |
| **Author** | Codex session |
| **Touches** | `src/inference_experiments/mmlu/`, `tests/mmlu/`, `pyproject.toml`, `uv.lock`, `README.md` |
| **Invariants** | none |
| **CONTEXT** | none |

## Original prompt

> Okay so I need your help developing a pipeline with the MMLU data that we recently downloaded okay?
>
> 1. i want you to install vllm and get a basic model of it working with the model that we downloaded below. It's important to me that you do this because i wnat to be able to easily use vllm to serve inference and i would use a script that basically gets the useful stats for mmlu that people care about so we get some baseline metrics for Qwen3-8B and that specific baseline so when we do a quantized model it looks a little different
> 2. can you scope out what that looks like, in a new worktree scope out the implementation and what it would look like and what new files oyu would need to create. create an ADR as well

## Context

The repository contains a local BF16 Qwen3-8B checkpoint under
`models/Qwen3-8B` and the pinned `cais/mmlu` parquet snapshot under
`data/mmlu/all`, but it has no inference runtime, evaluator, result schema or
serving entry point. The test split has 14,042 rows across 57 subjects and the
dev split supplies five demonstrations per subject. Consequently, an ad hoc
script could produce a number, but it would not establish whether later
quantized-model differences came from weights, prompts, decoding, aggregation
or runtime settings. Baseline quality and throughput are currently unmeasured.
A gdevbox feasibility check on 2026-09-10 installed vLLM 0.29.0, loaded the
15.26 GiB checkpoint on one RTX 4000 Ada in BF16, and deterministically generated
`vLLM works`.

## Decision

Build a typed, offline-first evaluator that owns MMLU loading, canonical
five-shot prompting, constrained A–D decoding, aggregation and durable run
artifacts, while placing vLLM behind an inference protocol. Use the same local
checkpoint through a separately exposed `vllm serve` convenience command for
interactive OpenAI-compatible inference. Treat the prompt/scoring protocol and
run manifest as the comparison contract: a future quantized run changes engine
configuration and checkpoint identity, not evaluation semantics.

### Touched / untouched

- **Touched** — `src/inference_experiments/mmlu/`: typed dataset, prompt,
  inference, evaluation and metrics components.
- **Touched** — `src/inference_experiments/vllm_serve.py`: validated arguments
  for launching the local OpenAI-compatible server.
- **Touched** — `tests/mmlu/` and `tests/test_vllm_serve.py`: offline tests with
  fake inference and tiny parquet fixtures.
- **Touched** — `pyproject.toml` and `uv.lock`: reproducible vLLM and parquet
  dependencies plus console entry points.
- **Touched** — `.gitignore`, `README.md`, and `data/mmlu/README.md`: artifact
  policy and operator documentation.
- **Untouched** — `models/Qwen3-8B/`: model files remain external, immutable
  inputs and stay ignored by Git.
- **Untouched** — `data/mmlu/all/*.parquet`: the pinned source dataset remains
  byte-for-byte unchanged.
- **Untouched** — `scripts/download_qwen3_8b.py`: checkpoint acquisition is
  independent of evaluation and serving.
- **Untouched** — auxiliary-training data: it is not part of the MMLU baseline.

### Promises / non-promises

- **Promises** — the evaluator is offline after environment setup; validates
  every input row; uses stable row IDs, prompts and greedy answer decoding;
  reports overall, per-subject and standard-category accuracy; records runtime
  and checkpoint provenance; preserves row-level predictions; refuses
  incomplete runs; and can compare BF16 and quantized runs question by question.
- **Promises** — unit tests mock vLLM, CUDA, clocks and expensive work; the full
  repository formatting, lint and test gates run before handoff.
- **Non-promises** — this does not claim equivalence to every published Qwen3
  number, whose chat template, thinking budget and MMLU variant may differ.
- **Non-promises** — offline batched throughput is not online request latency,
  and the initial implementation will not load-test the serving endpoint.
- **Non-promises** — the feasibility smoke test proves BF16 loading and one
  generation on the observed node; a full MMLU score and quantized result remain
  UNPROVEN until implementation and benchmark execution.

### Interfaces

`MMLUExample`, `EvaluationConfig`, `Prediction`, `EvaluationMetrics` and
`RunManifest` define the serialized boundary. `InferenceEngine.predict()` is
the only evaluator-to-runtime seam; `VLLMEngine` implements it and tests use a
fake. `mmlu-baseline` writes `manifest.json`, `predictions.jsonl` and
`metrics.json` to an atomically finalized run directory. `serve-qwen3-8b`
launches the vLLM OpenAI-compatible server without coupling the benchmark to a
network service.

### Sketches

```python
class InferenceEngine(Protocol):
    def predict(self, requests: tuple[InferenceRequest, ...]) -> tuple[Answer, ...]: ...

def evaluate_mmlu(config: EvaluationConfig, engine: InferenceEngine) -> RunReport: ...

def aggregate_metrics(predictions: tuple[Prediction, ...]) -> EvaluationMetrics: ...
```

## Questions

**Q1. Should the comparison contract use five-shot, non-thinking, constrained
A–D decoding?**
Recommendation: agree. It is deterministic, inexpensive, close to the
traditional MMLU protocol, and prevents free-form answer-parser differences
from obscuring weight-quantization effects.
If the other branch: define a zero-shot, thinking-enabled or generated-answer
protocol as a separate named benchmark; do not silently mix its results with
this baseline.

> **Samarth:**

**Q2. Should the primary overall score be micro accuracy, with macro subject and
standard category scores reported alongside it?**
Recommendation: agree. Micro accuracy is the clearest all-question score while
the additional aggregations expose subject imbalance and localized regressions.
If the other branch: make macro subject accuracy primary and label historical
or external comparisons carefully because aggregation conventions differ.

> **Samarth:**

**Q3. Should the evaluator call vLLM in-process and keep OpenAI-compatible
serving as a separate operator command?**
Recommendation: agree. In-process batching exposes token/log-probability data
and removes HTTP variability from the quality baseline, while the separate
server still supports interactive inference.
If the other branch: evaluate through the HTTP endpoint and add server lifecycle,
health checks, retries and request-order restoration to the benchmark contract.

> **Samarth:**

**Q4. Should successful artifacts finalize atomically, with resume deferred?**
Recommendation: agree. A failed run remains visibly partial and cannot be
mistaken for a complete baseline; the 14,042-row run is small enough to rerun.
If the other branch: add checkpointed batches, a prompt/config fingerprint,
duplicate rejection and exact resume validation before emitting final metrics.

> **Samarth:**

**Q5. Should deterministic quality artifacts be separated from volatile timing
and environment data?**
Recommendation: agree. Predictions and accuracy can then be byte-compared while
timestamps, hardware and throughput remain available in the manifest.
If the other branch: whole-run byte identity is not promised; comparisons must
parse and normalize volatile fields.

> **Samarth:**

## Alternatives

- Use lm-evaluation-harness directly. It provides broad benchmark coverage but
  adds a large abstraction surface and makes the project-specific artifact and
  quantized row-delta contract harder to control. Its prompt is still useful as
  a golden reference.
- Use Hugging Face Transformers generation. This avoids vLLM but does not meet
  the serving goal and gives up the batching/runtime path intended for later
  quantized experiments.
- Score generated prose with a parser. This resembles chat usage but introduces
  invalid-answer and parser behavior that can dominate small quantization
  deltas.

## Consequences

The repository gains a modest evaluation subsystem rather than a single opaque
benchmark script. That costs more typed models and tests, but prompt, scoring,
aggregation and runtime changes become independently reviewable. vLLM and
PyArrow substantially increase the environment lock and installation size.
Exact answer-token assumptions must be validated for every tokenizer, and new
benchmark protocols require distinct names rather than overwriting this one.

## Surface Areas

- Dependency resolution and the gdevbox CUDA/Python environment.
- Local model serving and vLLM engine configuration.
- MMLU parquet schema, demonstrations and subject taxonomy.
- Prompt rendering, answer-token constraints and log probabilities.
- Quality metrics, performance measurements and JSON/JSONL artifacts.
- Offline unit tests, GPU smoke validation and benchmark documentation.

## Outcome

Pending answers and evaluator implementation. The feasibility smoke test is
complete, and the worktree declares and locks vLLM 0.29.x and PyArrow 25.x. The
full MMLU baseline remains unmeasured.
