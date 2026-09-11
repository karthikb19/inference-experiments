# ADR — Reproducible vLLM WikiText-103 perplexity

| | |
|---|---|
| **Date** | 2026-09-11 |
| **Status** | Implemented |
| **Author** | Codex session |
| **Touches** | `src/inference_experiments/wikitext/`, `tests/wikitext/`, `pyproject.toml`, `README.md`, `knowledge/runbooks/`, `knowledge/reports/` |
| **Invariants** | none |
| **CONTEXT** | Extends `knowledge/decisions/2026-09-10-reproducible-vllm-mmlu-baselines.md` |

## Original prompt

> Hi!
>
> You have downloaded the wikitext stuff and it is merged into the latest main
>
> I want to do the same thing that we did for MMLU! For MMLU we defined an ADR that outlined basic information of what we were tryna achieve ~ the basic primitives/schemas that would allow us to use inprocess VLLM to test accuracy and get the evlauations and the logprobs
>
> I want to develop something similar here for wiki text and perplexity. I wasnt to see what the perplexity is for qwen3-8b-bf16 and how that gets adjusted based off of the fact we do quantization which is the ultimate goal of this exercise
>
> > can you define how we would get perplexity from this wikitext that we have available to us right now and then define the primitives for the pipeline that we would generate?
>
> Just the general schemas/commonw ays to evaluate it will suffice, we have a lot of primitives from mmlu that we can define
>
> think abt edgecases too btw, okay?

> To clarify,
>
> Use a 4,096-token scoring window and a 512-token stride. The engine limit is at least 4,097 because vLLM 0.29.0 requires one generated token when using LLM.generate; that generated token is an implementation artifact and is never scored. The first window scores positions 1:4096. Every later window carries as much left context as fits and scores only its new suffix, up to 512 tokens. The final short suffix is included.
> > we first start from 1, 4096 -> 512, 4096+512, .... correct?
>
> ===
> as for the questions
> 1. yes, agree lets follow hugging face orotocl here
> 2. yes, i agree with you
> 3. asked thsi question earlier
> 4. yes, we only wnat to use the promptlog probs, we don't necessairly care about the token that we are at in that particular moment
> 5. YES! its not that much data! we can store!
> 6. yes, lets use mean nll!
> 7. yes
> 8. yes, we discussed this with mmlu
>
> @dataclass(frozen=True)
> class PerplexityMetrics:
>     corpus_tokens: int
>     scored_tokens: int
>     total_nll: float
>     mean_nll: float
>     perplexity: float
>     bits_per_token: float
>
> ^ can you quikcly explain what bits_per_token means? and what do the other tokens mean/I agree with the long term metrics tho just an fyi - just asking about this

## Context

The repository now pins the `Salesforce/wikitext` `wikitext-103-raw-v1`
snapshot at revision `b08601e04326c79dfdd32d625aee71d232d685c3`, including
the 4,358-row test split and 3,760-row validation split
(`data/wikitext/README.md:3-10`). The only column is nullable UTF-8 `text`.
Inspection of the committed files found no nulls, but found 1,467 empty test
rows and 1,299 empty validation rows. Those rows encode document and section
spacing and must not be discarded as if they were missing examples.

There is no WikiText loader, causal-likelihood engine, perplexity metric,
comparison schema, CLI or artifact contract. The existing MMLU evaluator
already supplies useful patterns: typed inputs and outputs
(`src/inference_experiments/mmlu/models.py:29-166`), an in-process vLLM protocol,
immutable checkpoint hashing, stable quality artifacts, separate runtime data,
and atomic partial-to-complete run directories
(`src/inference_experiments/mmlu/evaluate.py:40-101`). Its classification request
and A-D scoring types cannot represent a continuous token stream or overlapping
context windows, however.

Perplexity for tokens \(x_0, \ldots, x_{N-1}\) is the exponentiated mean
negative log-likelihood over the tokens that have preceding context:

\[
\operatorname{mean\_nll} = -\frac{1}{N-1}\sum_{i=1}^{N-1}
\log p_\theta(x_i \mid x_{<i}), \qquad
\operatorname{ppl} = \exp(\operatorname{mean\_nll}).
\]

Because Qwen has a finite context window, the pipeline can only approximate
the full prefix in that equation. A strided sliding window is the common
practical method: use old tokens as unscored context and score each corpus token
exactly once. Averaging window perplexities, row perplexities, or per-window
mean losses would be wrong because differently sized windows would receive
equal weight. The aggregate must sum token negative log-likelihoods first and
divide once by the number of scored tokens. Hugging Face's
[fixed-length perplexity guide](https://huggingface.co/docs/transformers/main/perplexity)
documents the same token-weighted sliding-window method and its context-length
sensitivity.

For the local Qwen tokenizer, exact `"\n\n".join(rows)` rendering with no
special tokens produces 299,078 test tokens and 262,337 validation tokens. The
test corpus text SHA-256 is
`696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83`.
Qwen's tokenizer does not add a BOS token, so the first corpus token has no
modelled predecessor and is excluded. Before implementation, Qwen3-8B BF16 test
perplexity and its change under quantization were **UNMEASURED**.

## Decision

Build a typed, offline-first WikiText evaluator that deterministically renders
and tokenizes one corpus, plans overlapping scoring windows, requests prompt
token log-probabilities from an in-process vLLM engine, validates and preserves
the target-token scores, and computes a single token-weighted perplexity. Treat
the rendered corpus, tokenizer, ordered token IDs, target indices and window
policy as the paired-comparison contract. A quantized run may change checkpoint
and engine quantization configuration but must not change any of those
evaluation inputs. Report mean NLL as the primary comparison quantity and
perplexity as its conventional transform.

### Evaluation protocol

The accepted named protocol is
`wikitext-103-raw-qwen-token-ppl-sliding-v1`:

1. Read parquet shards in lexical filename order and rows in stored order.
   Require exactly one string `text` column and reject nulls, schema drift,
   missing shards, duplicate shard names, or an empty split.
2. Render the corpus as exact `"\n\n".join(text_rows)`. Preserve empty strings,
   whitespace, headings, existing newlines and WikiText placeholders such as
   `@-@`; do not strip, normalize, detokenize, or insert document EOS tokens.
3. Tokenize once with the BF16 Qwen3-8B tokenizer and
   `add_special_tokens=False`. Persist hashes of the rendered UTF-8 bytes,
   tokenizer files and ordered token IDs. Quantized evaluation reuses these
   token IDs rather than retokenizing with checkpoint-local defaults.
4. Use a 4,096-token scoring window and a 512-token stride. Require the stride
   to be shorter than the window so every later target has preceding context.
   The engine limit is
   at least 4,097 because vLLM 0.29.0 requires one generated token when using
   `LLM.generate`; that generated token is an implementation artifact and is
   never scored. The first window scores positions `1:4096`. Every later
   window carries as much left context as fits and scores only its new suffix,
   up to 512 tokens. The final short suffix is included.

   Using half-open ranges, the first prompt is `[0, 4096)` and scores
   `[1, 4096)`. The second prompt shifts by 512 to `[512, 4608)` but scores only
   `[4096, 4608)`; its first 3,584 tokens are context. The third prompt is
   `[1024, 5120)` and scores `[4608, 5120)`. This continues without scoring an
   overlap twice or leaving a gap.
5. Submit token IDs, not text, through one in-process batched engine with
   `SamplingParams(prompt_logprobs=0, max_tokens=1, temperature=0.0)`. vLLM
   returns the actual prompt token's log-probability even when zero alternative
   log-probabilities are requested. Its first prompt position is `None` because
   no preceding token exists; a window's first position is always context and
   never a target. This behavior is part of vLLM's
   [`PromptLogprobs` contract](https://docs.vllm.ai/en/latest/api/vllm/logprobs/).
6. Require output IDs and order to match the request; one prompt score per
   prompt position; `None` only at offset zero; the actual token ID to be
   present at every scored position; and every selected log-probability to be
   finite and non-positive. Reject rather than silently omit any target.
7. Traverse scores in ascending corpus-token order and use `math.fsum` to
   calculate `total_nll`. Set `mean_nll = total_nll / scored_tokens`,
   `perplexity = exp(mean_nll)`, and `bits_per_token = mean_nll / log(2)`.
   Refuse zero scored tokens, duplicate or missing target indices, non-finite
   aggregates, and exponential overflow.

This is a Qwen-token perplexity protocol, not the word-level perplexity used by
some classic WikiText-103 papers. Its absolute result is comparable only when
dataset rendering, tokenizer, special tokens, context length, stride and
scoring implementation match. Its main purpose here is a controlled BF16 to
quantized comparison.

### Primitives and schemas

`WikiTextRow` and `TokenizedCorpus` own source validation and deterministic
rendering. `ScoringWindow` identifies both the full context span and the target
suffix in global corpus coordinates. `LikelihoodEngine` is the only vLLM seam;
tests use a fake that returns analytically chosen scores. `WindowScore` retains
each target token's ID and log-probability so quantization differences can be
aligned and diagnosed token by token. `PerplexityMetrics` is derived only from
complete, validated window scores.

```python
@dataclass(frozen=True)
class EvaluationConfig:
    model: Path
    tokenizer: Path
    data_dir: Path
    output_dir: Path
    split: Literal["validation", "test"] = "test"
    context_length: int = 4096
    stride: int = 512
    tensor_parallel_size: int = 2
    batch_size: int = 8
    gpu_memory_utilization: float = 0.9
    quantization: str | None = None
    limit_tokens: int | None = None


@dataclass(frozen=True)
class TokenizedCorpus:
    split: Literal["validation", "test"]
    text_sha256: str
    tokenizer_sha256: str
    token_ids_sha256: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class ScoringWindow:
    window_id: str
    corpus_start: int
    corpus_end: int
    target_start: int
    target_end: int
    prompt_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class PromptTokenLogprob:
    token_id: int
    logprob: float


@dataclass(frozen=True)
class PromptLikelihood:
    window_id: str
    prompt_token_ids: tuple[int, ...]
    prompt_logprobs: tuple[PromptTokenLogprob | None, ...]


class LikelihoodEngine(Protocol):
    def score(
        self, requests: tuple[ScoringWindow, ...]
    ) -> tuple[PromptLikelihood, ...]: ...


@dataclass(frozen=True)
class WindowScore:
    window_id: str
    target_start: int
    target_end: int
    target_token_ids: tuple[int, ...]
    target_logprobs: tuple[float, ...]
    total_nll: float


@dataclass(frozen=True)
class PerplexityMetrics:
    corpus_tokens: int
    scored_tokens: int
    total_nll: float
    mean_nll: float
    perplexity: float
    bits_per_token: float


@dataclass(frozen=True)
class PerplexityComparison:
    scored_tokens: int
    baseline_mean_nll: float
    candidate_mean_nll: float
    mean_nll_delta: float
    baseline_perplexity: float
    candidate_perplexity: float
    perplexity_delta: float
    perplexity_ratio: float
```

The metric fields have the following meanings:

- `corpus_tokens` is the number of Qwen tokenizer tokens in the rendered
  corpus: 299,078 for the complete test split under this protocol.
- `scored_tokens` is the number with a defined preceding context. It is
  299,077 because the very first corpus token cannot be predicted from an
  earlier corpus token.
- `total_nll` is the sum of `-log(probability)` for all scored ground-truth
  tokens, using natural logarithms. Lower is better, but the value grows with
  corpus size.
- `mean_nll` divides `total_nll` by `scored_tokens`. It is the primary,
  corpus-length-normalized comparison; lower is better.
- `perplexity` is `exp(mean_nll)`. It is the conventional language-model
  presentation of the same result; lower is better.
- `bits_per_token` is `mean_nll / log(2)`, equivalently
  `log2(perplexity)`. It expresses average model surprise as idealized coding
  bits per Qwen token. It does not describe the token IDs' storage size and is
  not bits per byte or character. Lower is better.

The eventual comparison primitive accepts two completed manifests plus their
scores and returns `mean_nll_delta = quantized - bf16`,
`perplexity_delta = quantized - bf16`, `perplexity_ratio = quantized / bf16`,
and token-level delta summaries. It first requires identical protocol,
dataset/text/tokenizer/token-ID hashes, split, context length, stride, target
indices and scored-token count. A positive NLL delta is a regression; the exact
relationship is `perplexity_ratio = exp(mean_nll_delta)`.

### Artifacts

Follow the MMLU partial-directory lifecycle and separate stable evaluation data
from volatile runtime data:

- `config.json` — protocol, dataset revision/split and hashes, checkpoint hash,
  tokenizer and evaluator-source hashes, dtype/quantization, context/stride,
  engine limit, batch and tensor-parallel settings, token limit, seed, and Git
  commit.
- `corpus.json` — row count, UTF-8 byte count, corpus/scored-token counts and
  corpus/token-ID hashes. Token IDs need not be duplicated when their hash and
  deterministic construction are recorded.
- `scores.jsonl` — one ordered `WindowScore` per window, including target token
  IDs and log-probabilities. This is the diagnostic equivalent of MMLU's
  row-level predictions.
- `metrics.json` — only complete-run `PerplexityMetrics`; no per-window
  perplexity and no volatile data.
- `runtime.json` — initialization/evaluation seconds, scored tokens per second,
  prompt tokens per second, vLLM/Torch/CUDA/Python versions and GPU names.
- `failure.json` — exception type/message plus completed windows and scored
  tokens, retained only in the `.partial` directory.
- `comparison.json` — emitted by a separate comparison command after both input
  runs pass compatibility checks; it names both run/checkpoint hashes and
  records aggregate and token-delta statistics.

Stable traversal and JSON shape are promised. Exact floating-point byte
identity across vLLM, Torch, CUDA, GPU type, tensor parallelism or batch size is
not promised; numerical reruns use explicit tolerances. A BF16/quantized pair
should use the same software and hardware environment whenever practical.

### Touched / untouched

- **Touched** — `src/inference_experiments/wikitext/`: typed data, tokenization,
  window planning, inference, aggregation, comparison and artifact components.
- **Touched** — `tests/wikitext/`: offline unit and integration tests with tiny
  parquet fixtures, a fake tokenizer and fake likelihood engine.
- **Touched** — `pyproject.toml`: a `wikitext-perplexity` console entry point;
  no dependency is currently expected beyond the installed stack.
- **Touched** — `README.md` and `knowledge/runbooks/wikitext-perplexity.md`:
  operator commands, interpretation and comparison guidance.
- **Untouched** — `data/wikitext/wikitext-103-raw-v1/*.parquet`: immutable
  benchmark inputs remain byte-for-byte unchanged.
- **Untouched** — `models/Qwen3-8B/`: the BF16 checkpoint and tokenizer remain
  immutable external inputs.
- **Untouched** — `src/inference_experiments/mmlu/`: reuse its design patterns,
  not its benchmark-specific types; premature shared-framework extraction is
  avoided until both implementations expose proven common seams.
- **Untouched** — training data: only validation smoke runs and test headline
  runs are allowed; the train split is not needed for zero-shot perplexity.
- **Untouched** — `vllm serve`: quality scoring stays in-process and does not
  add HTTP lifecycle or transport variability.

### Promises / non-promises

- **Promises** — exact source validation and rendering; fixed Qwen tokenization;
  each eligible token scored once; token-weighted aggregation; saved target
  log-probabilities; strict BF16/quantized compatibility checks; stable target
  ordering; atomic successful artifacts; visible partial failures; deterministic,
  fast and offline unit tests; and two-GPU in-process vLLM by default.
- **Promises** — tests cover null/non-string rows, empty and one-token corpora,
  empty rows, shard ordering, final short windows, invalid context/stride,
  context limits, missing/extra/reordered responses, malformed log-probability
  shapes, missing actual-token scores, `None` in a target, NaN/infinity/positive
  scores, duplicate/gapped targets, weighted aggregation, overflow, incompatible
  comparisons, partial failure and output-directory collisions.
- **Non-promises** — this does not reproduce word-level WikiText leaderboards,
  compare models with different tokenizers, measure chat quality, insert a Qwen
  chat template, infer statistical significance from one corpus, or prove that
  a small perplexity change is user-visible.
- **Non-promises** — batching throughput is not serving latency. The 4,096/512
  policy is a named compute/quality tradeoff, not Qwen's maximum possible
  context. The quantized value remains **UNPROVEN** until a quantized checkpoint
  is evaluated.

### Interfaces

`load_wikitext_rows()`, `render_corpus()`, `tokenize_corpus()` and
`plan_scoring_windows()` are pure functions with explicit validation failures.
`LikelihoodEngine.score()` isolates vLLM and accepts pre-tokenized windows.
`score_windows()` aligns and validates engine responses. `aggregate_perplexity()`
owns the only corpus metric calculation. `compare_runs()` refuses mismatched
contracts before reading score deltas. `run_evaluation()` owns batching,
partial artifacts, finalization, timing and engine cleanup.

```python
def plan_scoring_windows(
    corpus: TokenizedCorpus, *, context_length: int, stride: int
) -> tuple[ScoringWindow, ...]: ...


def aggregate_perplexity(
    corpus_tokens: int, scores: tuple[WindowScore, ...]
) -> PerplexityMetrics: ...


def compare_runs(
    baseline: CompletedRun, candidate: CompletedRun
) -> PerplexityComparison: ...
```

## Questions

**Q1. Should exact raw-row concatenation define the corpus, including empty
rows, with the test split as the headline and validation for smoke tests?**
Recommendation: agree. It mirrors the common Hugging Face raw-WikiText recipe,
avoids heuristic article parsing, and freezes all otherwise ambiguous
whitespace. If the other branch: define document boundaries and boundary tokens
precisely, reset context at each boundary, and use a different protocol name.

> **Samarth:** yes, agree lets follow hugging face orotocl here

> **Resolution:** Agree. Preserve exact raw-row concatenation, use validation
> for smoke tests, and reserve complete test runs for headline results.

**Q2. Should the Qwen3-8B BF16 tokenizer with no BOS/EOS or chat template be
frozen for both BF16 and quantized runs?**
Recommendation: agree. Quantization changes weights/runtime representation, not
the units being predicted. If the other branch: results become separate
absolute benchmarks and the comparator must refuse a paired perplexity delta.

> **Samarth:** yes, i agree with you

> **Resolution:** Agree. Freeze the BF16 tokenizer and omit BOS, EOS and chat
> templating for every comparable run.

**Q3. Should the first protocol use a 4,096-token window and 512-token stride?**
Recommendation: agree. It supplies at least 3,584 prior tokens for later target
suffixes while requiring 578 test requests, a practical compromise for
the available two-GPU setup. If the other branch: choose and freeze another pair
before the BF16 baseline; changing either later creates a new protocol and
requires rerunning every model.

> **Samarth:** asked thsi question earlier

> **Resolution:** Agree after clarifying half-open ranges. Prompt windows shift
> by 512 tokens, while only the non-overlapping suffix is scored each time.

**Q4. Should vLLM prompt log-probabilities be the scoring backend, with the one
required generated token explicitly ignored?**
Recommendation: agree. This keeps quality evaluation on the requested
in-process vLLM path, requests only the observed token's prompt score, and
retains strict response validation. If the other branch: use a direct
Transformers forward-loss implementation, which is simpler for masking but
does not validate the inference runtime intended for quantized models.

> **Samarth:** yes, we only wnat to use the promptlog probs, we don't
> necessairly care about the token that we are at in that particular moment

> **Resolution:** Agree. Use prompt log-probabilities for the observed
> ground-truth corpus tokens. Ignore the one generated decode token completely.

**Q5. Should every target-token log-probability be retained in ordered window
artifacts?**
Recommendation: agree. The approximately 299,077 test scores are modest and
make local quantization regressions, malformed responses and aggregation
reproducible. If the other branch: retain only window sums, reducing artifact
size while giving up token-level paired diagnosis.

> **Samarth:** YES! its not that much data! we can store!

> **Resolution:** Agree. Retain every target token ID and log-probability.

**Q6. Should mean NLL delta be the primary quantization comparison, with
perplexity delta and ratio reported alongside it?**
Recommendation: agree. NLL is additive and its delta maps exactly to the
perplexity ratio, while raw perplexity differences can look disproportionate.
If the other branch: make perplexity delta primary but still retain NLL so the
comparison remains mathematically auditable.

> **Samarth:** yes, lets use mean nll!

> **Resolution:** Agree. Mean NLL is primary; perplexity delta and ratio remain
> conventional derived views.

**Q7. Should successful runs finalize atomically while failed runs retain
ordered partial scores, with resume deferred?**
Recommendation: agree. A partial run cannot masquerade as a metric, and this
corpus is small enough to rerun before adding fingerprinted resume complexity.
If the other branch: add window-level checkpointing, fsync boundaries,
configuration/token-plan fingerprint verification, duplicate rejection and
exact continuation from the first missing target before finalization.

> **Samarth:** yes

> **Resolution:** Agree. Finalize only complete runs, retain visible partial
> failures, and defer resume support.

**Q8. Should stable evaluation artifacts remain separate from volatile timing
and hardware provenance?**
Recommendation: agree. This carries forward the MMLU comparison contract and
keeps performance noise out of quality review. If the other branch: whole-run
artifacts cannot be compared without parsing and normalizing volatile fields.

> **Samarth:** yes, we discussed this with mmlu

> **Resolution:** Agree, consistent with the MMLU artifact contract. Keep
> deterministic quality outputs separate from runtime and hardware data.

## Alternatives

- Use lm-evaluation-harness's WikiText task. It offers external convention and
  word-perplexity reporting, but its document detokenization and normalization
  answer a different question than exact Qwen-token likelihood and make the
  paired token-level quantization contract less direct.
- Use disjoint 4,096-token chunks. This is faster, but most chunk-start tokens
  receive almost no context and the resulting perplexity is systematically
  worse and more boundary-sensitive.
- Score each parquet row or inferred article independently. This avoids
  cross-document context but introduces many unscored first tokens and makes
  empty-row/document parsing policy part of the result.
- Use stride one. It best approximates full available context but would require
  nearly 300,000 requests instead of 578, an unnecessary cost for a
  first paired quantization benchmark.
- Store only aggregate perplexity. This is compact but cannot distinguish an
  actual weight regression from dropped, duplicated or misaligned token scores.

## Consequences

The project gains a second evaluator with benchmark-specific data and metric
types while reusing the MMLU lifecycle patterns. A full run performs overlapping
prefill work and emits several megabytes of token scores, trading compute and
storage for a much better finite-context likelihood estimate and auditable
quantization deltas. Quantized checkpoints must be coupled to the baseline
tokenizer/token plan. Any future change to corpus rendering, special tokens,
window length, stride or target selection requires a new protocol identifier
and a new BF16 baseline.

Prompt log-probabilities exercise vLLM's prefill path rather than normal text
generation. The implementation therefore needs a small live smoke test against
an analytically cross-checked Transformers forward pass before trusting a full
GPU result. That cross-check should compare selected token log-probabilities
within an explicit tolerance; exact equality is not expected across kernels.

## Surface Areas

- WikiText parquet schema, shard ordering and exact raw-text rendering.
- Qwen tokenizer identity, special-token policy and immutable token plans.
- Finite-context window planning and target masking.
- In-process vLLM prompt log-probabilities and two-GPU engine configuration.
- Token-weighted NLL, perplexity, bits-per-token and paired quantization deltas.
- Stable score/config/metric artifacts, volatile runtime data and partial runs.
- Offline unit tests plus a bounded GPU and Transformers parity smoke test.

## Outcome

Q1–Q8 were accepted on 2026-09-11. The evaluator landed in
`src/inference_experiments/wikitext/` with strict parquet/token-plan validation,
Hugging Face-style window planning, in-process vLLM prompt scoring, retained
target log probabilities, token-weighted metrics, atomic artifacts, paired-run
comparison gates, two console commands, and an operator runbook. The offline
suite has 57 passing tests.

A nine-window live smoke scored 8,191 validation targets, and a separate
Transformers forward pass agreed with the vLLM observed-token log probabilities
within a recorded maximum absolute tolerance of 0.1. The complete two-GPU BF16
test run scored all 299,077 eligible targets across 578 windows: mean NLL
`2.113317981`, perplexity `8.275654244`, and bits/token `3.048873371`. The
completed-run loader independently reconstructed those metrics from the saved
7.5 MB score artifact. Full provenance and runtime measurements are in
`knowledge/reports/qwen3-8b-bf16-wikitext-perplexity.md`. Quantized perplexity
and paired deltas remain **UNMEASURED** pending a quantized checkpoint.
