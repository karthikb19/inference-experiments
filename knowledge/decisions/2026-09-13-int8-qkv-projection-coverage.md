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

## Outcome

The implementation audit confirmed that vLLM 0.29.0 constructs each Qwen3
attention block's WQ, WK, and WV weights as row ranges of one
`QKVParallelLinear`, whose inheritance path is `QKVParallelLinear` →
`ColumnParallelLinear` → `LinearBase`. The existing INT8 method therefore
already rounded these parameters once after checkpoint loading. The change
adds a compatibility guard rather than a second rounding pass, and the model
loaded successfully through that guard on two RTX 4000 Ada GPUs. The offline
suite passed 78 tests; Ruff formatting and lint checks also passed.

The full MMLU verification evaluated all 14,042 questions with 10,509 correct,
74.8398% weighted accuracy, 76.7563% subject-macro accuracy, and zero invalid
predictions. Scoring took 242.69 seconds after 69.80 seconds of initialization.
Relative to the matched BF16 baseline's 10,505 correct and 74.8113% weighted
accuracy, the candidate changed four answers net and weighted accuracy by
+0.0285 percentage points. It scored six fewer answers than the earlier INT8
run, a small run-to-run difference that does not indicate a quality regression.
The local artifact directory is
`artifacts/mmlu/qwen3-8b-int8-qkv-projection-verification`. Its configuration,
metrics, and runtime SHA-256 digests are
`d9b0ac4bb3bcb364844122b8b6a0fa69b77e675c125f34cf09a5afc14644f304`,
`8f1a1ff79e0a0b46b80f0589b8390a1ae85c9e14896efd369509df1207ef8742`,
and `5fd47704d3670e820f5aca12011d85aa35a0ecb8ac2a87070b03a8e4d1625bae`
respectively.

The current-source WikiText BF16 control and INT8 candidate each scored all
299,077 eligible tokens across 578 windows. BF16 produced mean NLL 2.113318,
perplexity 8.275654, and 3.048873 bits per token. INT8 produced mean NLL
2.112195, perplexity 8.266366, and 3.047253 bits per token, exactly reproducing
the earlier INT8 aggregate result. The strict paired validator accepted the
model, corpus, tokenizer, and token-plan identities and measured a mean-NLL
delta of -0.001123, perplexity delta of -0.009289, and perplexity ratio of
0.998878. Of 299,077 tokens, 139,918 improved, 9,623 were unchanged, and
149,536 worsened; mean absolute token-NLL change was 0.044059. The mixed
token-level movement and negligible aggregate delta support the conclusion
that this INT8 round trip does not degrade quality, not that it improves the
model.

The local WikiText directories are
`artifacts/wikitext/qwen3-8b-bf16-qkv-verification-control` and
`artifacts/wikitext/qwen3-8b-int8-qkv-projection-verification`. The paired
comparison is
`artifacts/wikitext/bf16-vs-int8-qkv-projection-verification.json`, with
SHA-256 `88c63e120d24a112da7013a4c85a1109818dd7a1be69785278d2002feb69130b`.
The BF16 configuration, metrics, and runtime digests are
`dfc1a8409c8cb6c7332d6d06dda17f9d55ceed6b11bcad6d271e23585fb4ba1a`,
`127be97c25905eaf87370d0ef4a02e1a0d3f4d229b0d111efa53bd6d09401de8`,
and `b1b7ed1756729c3ff1274d75dad788a24f4d91dc93738d5ac7389ad63b848436`.
The INT8 configuration, metrics, and runtime digests are
`49c0a4a20c980fcf95bdc8505123ce10992b9e41c166334b5f63efd1e324f335`,
`035bc0f4df516b9db38a84a6b7507c7c47fcffc24d1cf696242703c2690f3930`,
and `edb3f55bdfbd225bc395bd23ec8b1231377f634f022274b1ec7828e61cf07310`.
Raw artifacts remain ignored because the score files are large; the committed
decision record preserves their provenance, metrics, interpretation, and
locations.
