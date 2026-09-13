# ADR — vLLM INT4 linear fake quantization

## Context

The repository has per-output-channel INT8 and INT4 fake-quantization plugins
for evaluating BF16 Qwen3-8B linear layers without creating a quantized
checkpoint. The matching four-bit benchmark has now been run on the current
branch.

## Decision

Register `int4-fake-quant` alongside `int8-fake-quant`. Use symmetric signed
INT4 levels from −7 through 7, reconstruct BF16 weights once after loading, and
continue to use ordinary BF16 matrix multiplication. Simulated INT4 values are
held in `torch.int8`; no packed INT4 kernel is introduced.
Both evaluation engines explicitly set vLLM's `kv_cache_dtype` to `auto`, so
the plugin cannot silently quantize the KV cache.

## Alternatives

- Replace the existing INT8 plugin: rejected because existing experiments must
  remain reproducible.
- Add packed INT4 kernels: deferred because this experiment isolates numerical
  fake-quantization effects rather than storage or kernel performance.

## Consequences

The INT4 path is directly comparable with the INT8 path and requires no model
conversion. It does not measure packed-weight memory savings or INT4 kernel
throughput. The two plugin names are represented by the typed
`FakeQuantization` enum (`INT_8_FAKE_QUANT` and `INT_4_FAKE_QUANT`) while their
vLLM-facing values remain `int8-fake-quant` and `int4-fake-quant`.

## Surface Areas

The plugin module, offline unit tests, CLI/serving documentation, benchmark
artifacts, and this ADR are updated.

## Outcome

The full INT4 variant-0 runs completed on two RTX 4000 Ada GPUs with
`kv_cache_dtype=auto`, so the KV cache remained in its normal runtime dtype.
MMLU scored 9,173 of 14,042 questions correctly, or 65.33% weighted accuracy.
The matched BF16 baseline scored 74.8113% and the INT8 fake-quant run scored
74.8825%, putting INT4 9.48 percentage points below BF16 and 9.55 points below
INT8.

WikiText scored all 299,077 eligible tokens. Mean NLL was 2.371375,
perplexity was 10.712106, and bits per token was 3.421170. Relative to the
BF16 perplexity of 8.275654, INT4 increased mean NLL by 0.258057 and had a
perplexity ratio of approximately 1.294.

Artifacts are stored under
`artifacts/mmlu/qwen3-8b-int4-fake-quant-variant0` and
`artifacts/wikitext/qwen3-8b-int4-fake-quant-variant0`.
