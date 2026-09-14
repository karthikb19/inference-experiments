# ADR — Make INT8 QKV projection coverage an enforced contract

## Context

The `int8-fake-quant` plugin selects every vLLM `LinearBase`. In vLLM 0.29.0,
Qwen3 attention constructs WQ, WK, and WV as row ranges in one
`QKVParallelLinear`, which inherits from `LinearBase`. The requested attention
projection weights were therefore already fake-quantized, although the code
had no direct regression test or failure if a future vLLM version changed that
representation. Attention activations, projected Q/K/V tensors, attention
kernels, and the KV cache are separate and remain outside this experiment.

## Decision

Keep the existing `int8-fake-quant` experiment identity and its symmetric
per-output-row `absmax / 127` round trip. Explicitly recognize Qwen's fused QKV
projection in the INT8 configuration, reject a `.qkv_proj` prefix whose layer
is not `QKVParallelLinear`, and test both vLLM method selection and independent
scales across representative WQ, WK, and WV rows. Do not apply the new guard to
INT4 because this experiment is intentionally limited to INT8.

## Alternatives

- Add a second quantization pass for QKV weights: rejected because the current
  `LinearBase` pass already covers them and a second pass would change rounding.
- Quantize projected Q/K/V tensors: rejected because those are activations, not
  the requested projection weight parameters.
- Register a new quantization method name: rejected because the numerical INT8
  contract is unchanged and existing artifacts already describe QKV coverage.

## Consequences

Current Qwen3-8B numerics are unchanged. The configuration now fails loudly
instead of silently omitting WQ/WK/WV if the pinned vLLM representation drifts,
and focused offline tests document the fused projection layout. This still
does not reduce memory, use INT8 matrix multiplication, or quantize attention
activations. New MMLU and WikiText runs validate the enforced contract on GPU.

## Surface Areas

- `src/inference_experiments/fake_quant.py`: INT8 QKV compatibility guard.
- `tests/test_fake_quant.py`: fused QKV selection, scaling, and failure cases.
- `README.md`: exact weight-versus-activation scope.
- MMLU and WikiText artifacts: new validation runs, ignored by Git.
