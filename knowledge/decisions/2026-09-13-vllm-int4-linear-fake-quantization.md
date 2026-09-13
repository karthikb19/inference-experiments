# ADR — vLLM INT4 linear fake quantization

## Context

The repository has an INT8 per-output-channel fake-quantization plugin for
evaluating BF16 Qwen3-8B linear layers without creating a quantized checkpoint.
We need the matching four-bit experiment, and benchmark execution is deferred.

## Decision

Register `int4-fake-quant` alongside `int8-fake-quant`. Use symmetric signed
INT4 levels from −7 through 7, reconstruct BF16 weights once after loading, and
continue to use ordinary BF16 matrix multiplication. Simulated INT4 values are
held in `torch.int8`; no packed INT4 kernel is introduced.

## Alternatives

- Replace the existing INT8 plugin: rejected because existing experiments must
  remain reproducible.
- Add packed INT4 kernels: deferred because this experiment isolates numerical
  fake-quantization effects rather than storage or kernel performance.

## Consequences

The INT4 path is directly comparable with the INT8 path and requires no model
conversion. It does not measure packed-weight memory savings or INT4 kernel
throughput.

## Surface Areas

The plugin module, offline unit tests, CLI/serving documentation, and this ADR
are updated. No MMLU or WikiText run is started by this change.
