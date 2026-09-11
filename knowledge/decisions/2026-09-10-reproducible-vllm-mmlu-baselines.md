# ADR — Reproducible vLLM MMLU baselines

| | |
|---|---|
| **Date** | 2026-09-10 |
| **Status** | Implemented |
| **Author** | Codex session |
| **Touches** | `src/inference_experiments/mmlu/`, `tests/mmlu/`, `pyproject.toml`, `uv.lock`, `README.md` |
| **Invariants** | none |
| **CONTEXT** | none |

## Original prompt

> Okay so I need your help developing a pipeline with the MMLU data that we recently downloaded okay?
>
> 1. i want you to install vllm and get a basic model of it working with the model that we downloaded below. It's important to me that you do this because i wnat to be able to easily use vllm to serve inference and i would use a script that basically gets the useful stats for mmlu that people care about so we get some baseline metrics for Qwen3-8B and that specific baseline so when we do a quantized model it looks a little different
> 2. can you scope out what that looks like, in a new worktree scope out the implementation and what it would look like and what new files oyu would need to create. create an ADR as well
>
> 1. I think it would be a good idea if we split the deployment of the Qwen3-8b on two gpus that seems a lot better, so i would make that change in the implemetnation and the adr
> 2. adr as a whole makes sense, some responses to your questions:
> - I think it makes sense to do the five-shot stuff ONLY if that is what is generally used, do other people when evaluating using MMLU use this?
> - sure, not entirely sure wwhat you mean by micro-accuracy ~ but whatever is evaluated against by other researchers who have done papers and tests using MMLU works for me here
> - what is in process batching? can you explain what that means and how it differs from http server?
> - yes the yshould be separated (timing data and evaluation data)
> - yup failed run should be partial!

## Context

The repository contains a local BF16 Qwen3-8B checkpoint under
`models/Qwen3-8B` and the pinned `cais/mmlu` parquet snapshot under
`data/mmlu/all`, but it has no inference runtime, evaluator, result schema or
serving entry point. The test split has 14,042 rows across 57 subjects and the
dev split supplies five demonstrations per subject. Consequently, an ad hoc
script could produce a number, but it would not establish whether later
quantized-model differences came from weights, prompts, decoding, aggregation
or runtime settings. Baseline quality and throughput are currently unmeasured.
A gdevbox feasibility check on 2026-09-10 installed vLLM 0.29.0. A subsequent
two-GPU tensor-parallel check loaded half of the BF16 weights (7.64 GiB) on each
RTX 4000 Ada, exposed 9.17 GiB of KV-cache memory per worker, and
deterministically generated `two GPUs work`.

## Decision

Build a typed, offline-first evaluator that owns MMLU loading, original
five-shot prompting, A–D probability scoring, aggregation and durable run
artifacts, while placing vLLM behind an inference protocol. Both the evaluator
and the separately exposed OpenAI-compatible `vllm serve` command default to
tensor parallelism across two GPUs. Treat the prompt/scoring protocol and run
manifest as the comparison contract: a future quantized run changes engine
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

- **Promises** — the evaluator is offline after environment setup; defaults to
  tensor parallelism across two GPUs; validates every input row; uses stable
  row IDs, prompts and greedy answer decoding;
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
launches the vLLM OpenAI-compatible server. Both vLLM entry points use
`tensor_parallel_size=2` by default without coupling the benchmark to a network
service.

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
original MMLU protocol, and prevents free-form answer-parser differences from
obscuring weight-quantization effects. The
[original paper](https://arxiv.org/abs/2009.03300) uses up to five fixed
same-subject dev demonstrations and compares probabilities for A through D;
five-shot MMLU remains a common published and lm-evaluation-harness setting.
If the other branch: define a zero-shot, thinking-enabled or generated-answer
protocol as a separate named benchmark; do not silently mix its results with
this baseline.

> **Samarth:** I think it makes sense to do the five-shot stuff ONLY if that is
> what is generally used, do other people when evaluating using MMLU use this?
>
> **Resolution:** Agree. The original and widely reproduced few-shot protocol
> is five-shot, so the condition is satisfied.

**Q2. Should the primary overall score be size-weighted accuracy (total correct
divided by total questions), with subject and category scores alongside it?**
Recommendation: agree. This is the original paper's weighted all-example
accuracy and EleutherAI's
[`weight_by_size` aggregate](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_mmlu.yaml).
It is also called micro accuracy, but reports will use the clearer “weighted
accuracy” label. The additional aggregations expose subject imbalance and
localized regressions.
If the other branch: make macro subject accuracy primary and label historical
or external comparisons carefully because aggregation conventions differ.

> **Samarth:** sure, not entirely sure wwhat you mean by micro-accuracy ~ but
> whatever is evaluated against by other researchers who have done papers and
> tests using MMLU works for me here
>
> **Resolution:** Agree. Use the established weighted all-question accuracy as
> the headline result and retain subject/category breakdowns.

**Q3. Should the evaluator call vLLM in-process and keep OpenAI-compatible
serving as a separate operator command?**
Recommendation: agree. In-process batching exposes token/log-probability data
and removes HTTP variability from the quality baseline, while the separate
server still supports interactive inference.
If the other branch: evaluate through the HTTP endpoint and add server lifecycle,
health checks, retries and request-order restoration to the benchmark contract.

> **Samarth:** what is in process batching? can you explain what that means and
> how it differs from http server?
>
> **Resolution:** Unanswered pending explanation. In-process batching means the
> evaluation Python process imports vLLM, loads one two-GPU engine, and hands it
> groups of prompts directly. The HTTP alternative starts `vllm serve` as a
> separate long-running process; the evaluator serializes requests over a local
> socket and must handle readiness, failures, retries and response ordering.
>
> **Samarth:** we want in processes haha for sure.
>
> **Resolution:** Agree. Quality evaluation calls the two-GPU vLLM engine
> directly; HTTP remains a separate serving interface.

**Q4. Should successful artifacts finalize atomically, with resume deferred?**
Recommendation: agree. A failed run remains visibly partial and cannot be
mistaken for a complete baseline; the 14,042-row run is small enough to rerun.
If the other branch: add checkpointed batches, a prompt/config fingerprint,
duplicate rejection and exact resume validation before emitting final metrics.

> **Samarth:** yup failed run should be partial!
>
> **Resolution:** Agree. Only completed runs receive finalized artifacts.

**Q5. Should deterministic quality artifacts be separated from volatile timing
and environment data?**
Recommendation: agree. Predictions and accuracy can then be byte-compared while
timestamps, hardware and throughput remain available in the manifest.
If the other branch: whole-run byte identity is not promised; comparisons must
parse and normalize volatile fields.

> **Samarth:** yes the yshould be separated (timing data and evaluation data)
>
> **Resolution:** Agree. Stable quality artifacts and volatile performance data
> are separate files.

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

Implemented in `src/inference_experiments/mmlu/` with two console entry points,
strict parquet validation, original five-shot prompting, constrained A-D
probability scoring, weighted/subject/category metrics, checkpoint hashing, and
atomic partial-to-complete artifacts. All five questions are answered. The
offline suite has 17 passing tests and mocks inference, CUDA and timing. A live
two-GPU run scored two `abstract_algebra` test rows at 50% weighted accuracy,
recorded 699 prompt tokens, and validated the final artifacts. The two-GPU
serving entry point also answered its model-discovery and chat-completions HTTP
requests before shutting down cleanly. The full 14,042-question baseline and
any quantized comparison remain unmeasured.
