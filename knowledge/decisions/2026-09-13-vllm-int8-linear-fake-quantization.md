# ADR — Evaluate per-channel INT8 linear-weight error through vLLM

| | |
|---|---|
| **Date** | 2026-09-13 |
| **Status** | Implemented |
| **Author** | Codex session |
| **Touches** | `src/inference_experiments/fake_quant.py`, vLLM engine configuration, benchmark run contracts, tests, and operator documentation |
| **Invariants** | none |
| **CONTEXT** | Extends the MMLU and WikiText reproducible-baseline ADRs |

## Original prompt

> I am currently working on quantizing Qwen3-8B!
>
> The first thing I want to do is "fake quantize" the linear layer.
>
> What I mean by that is:
> * When we are doing the forward pass we first qunatize the rows (per-channel quantization) to int8 for the linear layers and then we undo it by multiplying by scale and then do the BF16 multiplication
> * The main difference is that instead of literally qunatiizng the model, I want to quantize the linear layers first and then test with the MMLU/WikiText benchmarks and see what sort of affects we have does that make sense Only the linear layers, not attention, not activations only linear alyers.
>
> Can you scope out what that would look like? and then make a pr with an adr and the proposed implementation that aligns with the implementation we have for the benchmarks (which are throughVLLM)

## Context

The MMLU and WikiText evaluators already use in-process vLLM 0.29.0 and share
the local BF16 Qwen3-8B checkpoint, but only WikiText exposes vLLM's generic
`quantization` argument. vLLM 0.29.0 does not provide online INT8 weight-only
quantization for dense linear layers; its similarly named built-in method is
limited to mixture-of-experts weights. Producing a second checkpoint would add
serialization and loader-format variables to an experiment intended to isolate
rounding error.

Qwen3-8B's dense MLP projections and attention projections are vLLM
`LinearBase` layers. The attention calculation, rotary embeddings, KV cache,
normalization, token embeddings, and vocabulary projection are separate
operations. Tensor parallelism introduces one subtlety: row-parallel output and
down projections split each logical weight row across GPUs, so independently
computed shard maxima would implement per-shard rather than per-channel
quantization. The effect on full MMLU accuracy and WikiText perplexity is
currently unmeasured.

## Decision

Register `int8-fake-quant` as a vLLM general plugin so it is installed in the
driver and every tensor-parallel worker. Load the unchanged checkpoint into
ordinary BF16 linear parameters, calculate an FP32 absolute maximum for each
logical output row, quantize symmetrically to signed values in `[-127, 127]`,
and immediately dequantize into the BF16 parameter. Run vLLM's ordinary BF16
GEMM afterward. Do the deterministic round trip once after loading rather than
before every invocation: the effective forward weights and quality results are
identical, while repeated work would contaminate runtime measurements and add
large transient allocations. For a row-parallel layer, MAX-reduce row maxima
over the tensor-parallel device group before rounding.

### Touched / untouched

- **Touched** — `src/inference_experiments/fake_quant.py`: pure row fake
  quantization and the vLLM quantization plugin.
- **Touched** — MMLU configuration, CLI, engine, and artifacts: carry the same
  explicit quantization identity that WikiText already records.
- **Touched** — the serving wrapper: permit interactive runs through the same
  plugin.
- **Touched** — `tests/`: deterministic CPU coverage for rounding, independent
  scales, zero rows, externally synchronized maxima, invalid inputs, and CLI
  propagation.
- **Touched** — `README.md`: paired benchmark commands and experiment scope.
- **Untouched** — checkpoint files: fake quantization creates no derived model
  and does not modify files on disk.
- **Untouched** — MMLU prompts, answer scoring, WikiText tokenization, sliding
  windows, metrics, and comparison logic: candidate and baseline retain the
  existing evaluation contracts.
- **Untouched** — activations, embeddings, normalization, attention kernels,
  rotary embeddings, KV cache, and vocabulary head: the plugin returns a
  method only for vLLM `LinearBase` layers.

### Promises / non-promises

- **Promises** — scales are symmetric FP32 `absmax / 127`, one per logical
  output row; zero rows remain zero; dequantized parameters and GEMM inputs are
  BF16; all dense Qwen linear projections use the method; both benchmarks
  record `int8-fake-quant` in their stable configuration.
- **Promises** — tensor-parallel results use global row maxima for row-sharded
  weights and local maxima where each complete row resides on one worker.
- **Non-promises** — this does not reduce model memory, use an INT8 GEMM,
  quantize activations, or measure deployable INT8 performance.
- **Non-promises** — this does not yet establish the MMLU or WikiText quality
  delta; complete paired benchmark runs remain UNMEASURED.

### Interfaces

`fake_quantize_rows()` is the independently testable numerical primitive.
`Int8FakeQuantConfig` selects `Int8FakeQuantLinearMethod` only for vLLM linear
layers. The `vllm.general_plugins` entry point makes registration available in
all engine processes. Existing `--quantization` fields select the method;
MMLU gains that same field and persists it in `config.json`.

### Sketches

```python
def fake_quantize_rows(
    weight: torch.Tensor, *, row_absmax: torch.Tensor | None = None
) -> torch.Tensor: ...


class Int8FakeQuantLinearMethod(LinearMethodBase):
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
```

## Alternatives

- Quantize and dequantize on every forward. This is algebraically equivalent
  for immutable inference weights, but repeatedly scans roughly eight billion
  parameters and allocates full-size temporaries, making benchmark timing
  describe the experimental wrapper rather than BF16 inference.
- Write a fake-quantized checkpoint. That can use stock vLLM but duplicates the
  model and introduces checkpoint creation and serialization as additional
  experimental variables.
- Use vLLM's `int8_per_channel_weight_only`. In vLLM 0.29.0 that shorthand has
  only an MoE specification; Qwen3-8B's dense linear layers remain unchanged.
- Compute a scale independently on each tensor-parallel shard. It is simpler,
  but row-parallel layers then use two scales per logical row and no longer
  implement the requested per-channel rule.

## Consequences

The same immutable BF16 checkpoint can now produce paired BF16 and fake-INT8
quality artifacts by changing one explicit argument. The implementation adds
no persistent INT8 representation and offers no memory or throughput benefit.
Model loading performs an extra read, rounding pass, and BF16 write over every
linear weight. vLLM's plugin and internal linear-method APIs are version-bound,
so upgrading beyond the pinned 0.29 release requires a live compatibility
smoke test.

## Surface Areas

- vLLM general-plugin discovery and custom quantization registration.
- Qwen dense attention and MLP projections.
- Two-GPU tensor-parallel weight loading and collectives.
- MMLU and WikiText stable run configuration.
- OpenAI-compatible vLLM serving configuration.
- Offline numerical tests and live GPU smoke validation.

## Outcome

The plugin, benchmark wiring, focused tests, and documentation landed together.
The offline suite passed 68 tests. Live vLLM 0.29.0 validation loaded the local
Qwen3-8B checkpoint through the plugin on two RTX 4000 Ada GPUs. Initial MMLU
and WikiText integration smokes completed before the full evaluations.

The full fake-quant MMLU run completed all 14,042 test questions with 10,515
correct, 74.8825% weighted accuracy, 76.7664% subject-macro accuracy, and zero
invalid predictions. The matched BF16 baseline had 10,505 correct and 74.8113%
weighted accuracy, so the aggregate difference was +10 answers and +0.0712
percentage points. Fake-quant evaluation took 242.08 seconds at 58.01 examples
per second after 71.43 seconds of engine initialization. Its local artifacts
are stored under `artifacts/mmlu/qwen3-8b-int8-fake-quant-full`.

The full fake-quant WikiText run scored all 299,077 eligible tokens in 578
windows. Mean NLL was 2.112195, perplexity was 8.266366, and bits per token was
3.047253. The repository's paired comparison validator accepted the candidate
and BF16 contracts and confirmed the same model hash. Relative to the BF16
mean NLL of 2.113318 and perplexity of 8.275654, fake quantization changed mean
NLL by -0.001123 and perplexity by -0.009289, for a perplexity ratio of
0.998878. Scoring took 441.11 seconds at 678.01 scored tokens per second after
100.43 seconds of engine initialization. Its local artifacts are stored under
`artifacts/wikitext/qwen3-8b-int8-fake-quant-full`.

These results establish that symmetric per-output-row INT8 rounding followed
by BF16 reconstruction does not cause an aggregate quality regression for this
checkpoint under the accepted MMLU and WikiText protocols. The tiny apparent
improvements are treated as effectively neutral rather than evidence that
quantization improves the model; repeated runs would be needed to characterize
runtime-level variation. This result supports proceeding to a stored INT8
checkpoint, but does not validate its serialization, scale layout, tensor-
parallel loading, packing, INT8 kernels, memory use, or performance. Those are
separate obligations for future changes.
