# Qwen3-8B BF16 full MMLU baseline

The saved run `qwen3-8b-bf16-full-official-1` completed all **14,042 test questions across 57 subjects**, with **10,505 correct (74.81% example-weighted accuracy)** and **zero invalid predictions**. Equal-weight subject accuracy was **76.76%**. This is a direct-answer, five-shot baseline for the local checkpoint, not a thinking-mode result.

## What this evaluation measures

MMLU is a multiple-choice evaluation organized by subject. This repository evaluates four answer options (A–D) and groups subjects into STEM, humanities, social sciences, and other. Five-shot means the prompt includes five answered development examples from the same subject before the test question; the model weights are not updated.

The protocol `mmlu-original-5shot-nonthinking-v1` uses a plain-text subject header, those five demonstrations, and a test question ending in `Answer:`. vLLM runs at temperature zero with one output token constrained to the four answer tokens. “Nonthinking” here means no generated reasoning trace; this evaluator does not use a chat template or an explicit thinking-mode toggle. The artifact name’s “official” suffix is a local run label, not evidence of external benchmark certification or equivalence to a published Qwen score.

Example-weighted accuracy is total correct divided by total questions. Subject-macro accuracy averages the 57 subject accuracies equally, so small subjects have the same weight as large subjects.

## Accuracy

| Scope | Questions | Correct | Incorrect | Weighted accuracy | Subject-macro accuracy |
|---|---:|---:|---:|---:|---:|
| Overall | 14,042 | 10,505 | 3,537 | 74.81% | 76.76% |
| STEM | 3,153 | 2,381 | 772 | 75.52% | 73.81% |
| Humanities | 4,705 | 3,107 | 1,598 | 66.04% | 76.02% |
| Social Sciences | 3,077 | 2,582 | 495 | 83.91% | 82.83% |
| Other | 3,107 | 2,435 | 672 | 78.37% | 76.21% |

Social sciences has the highest weighted score (83.91%); humanities has the lowest (66.04%). The humanities macro score is much higher (76.02%) because its larger, lower-scoring subjects receive less weight under the macro definition. Professional law contributes 748 errors and moral scenarios contributes 409: together, 1,157 of 3,537 errors (32.71%) from 2,429 of 14,042 questions (17.30%). This is an error-count concentration, not a causal diagnosis.

Marketing is the strongest subject at 94.44%, followed by high-school microeconomics at 93.70% and government and politics at 93.26%. Professional law is lowest at 51.24%, followed by global facts at 52.00% and moral scenarios at 54.30%. Abstract algebra remains at 56/100, matching the earlier subject-only run’s aggregate result; equal accuracy alone does not establish identical predictions.

## Completion and answer diagnostics

The analysis independently checked the saved predictions against the local test parquet: all 14,042 row IDs, source indices, subjects, and expected answers match in order, with no duplicates. Reaggregating the predictions reproduces the entire metrics JSON, including all subject and category entries. Correctness flags, prompt/output token totals, finite A–D scores, and selected-token maximum scores also pass verification.

There are **130 exact maximum-score ties**, with 46 correct emitted answers. All selected answers are at the maximum; none is strictly below it. The evaluator preserves vLLM’s emitted token on a tie, as implemented in the recorded commit. Replacing that policy with the first maximum in A–D order would yield 10,503/14,042 (74.80%) on these saved scores; this is a counterfactual rescoring, not a rerun.

Of 10,284 answers with `exp(selected_logprob) >= 0.9`, 1,452 are wrong. These are selected-token probabilities, not calibrated probabilities of answer correctness. The saved scores support confidence diagnostics but do not expose the model’s reasoning or explain individual mistakes.

## Runtime

| Measurement | Value |
|---|---:|
| Engine initialization | 56.92 s |
| Evaluation loop | 242.00 s (4.03 min) |
| Initialization + evaluation | 298.93 s (4.98 min) |
| Evaluation throughput | 58.02 questions/s |
| Prompt tokens | 9,776,485 |
| Reported prompt throughput | 40,398.39 tokens/s |
| Output tokens | 14,042 |
| Reported output throughput | 58.02 tokens/s |

The evaluation timer includes batched inference, scoring, and prediction writes. Initialization is measured separately. Their sum excludes checkpoint hashing, dataset/prompt preparation, and final artifact writing/cleanup, so it is not complete command wall time. Prompt throughput counts logical prompt tokens, including repeated demonstrations; it does not establish how much computation was saved by caching. One-token answers make output throughput unsuitable as a measure of sustained long-form generation speed. This is one timing observation, not a repeated performance benchmark.

## Configuration and provenance

| Field | Recorded value |
|---|---|
| Model path | `models/Qwen3-8B` |
| Precision evidence | Local model config declares `torch_dtype: bfloat16`; runtime artifact does not record resolved execution dtype |
| Model SHA-256 | `ac81fb6641aeb3b66128a9c24602b4123a76b86f33d10653efe9813e61e39195` |
| Evaluator commit | `279d371caccd85f3822dd500444fa56f23068346` |
| Dataset revision | `c30699e8356da336a370243923dbaf21066bb9fe` |
| Split / selection | `test`; all subjects; no limit |
| Protocol | `mmlu-original-5shot-nonthinking-v1` |
| Batch size / seed | 64 / 0 |
| Maximum model length | 4,096 tokens |
| Tensor parallelism | 2 |
| GPU memory utilization setting | 0.9 |
| GPUs | 2 × NVIDIA RTX 4000 Ada Generation |
| Python / vLLM | 3.12.3 / 0.29.0 |
| Torch / CUDA version reported by Torch | 2.13.0+cu130 / 13.0 |

“BF16” describes the intended checkpoint baseline and is supported by the local checkpoint configuration. The four run artifacts alone do not independently prove the effective runtime dtype. The dataset revision and model hash above are recorded provenance; this analysis did not redownload the dataset or rehash the model weights.

## All subject results

| Subject | Questions | Correct | Incorrect | Accuracy |
|---|---:|---:|---:|---:|
| abstract_algebra | 100 | 56 | 44 | 56.00% |
| anatomy | 135 | 95 | 40 | 70.37% |
| astronomy | 152 | 138 | 14 | 90.79% |
| business_ethics | 100 | 78 | 22 | 78.00% |
| clinical_knowledge | 265 | 208 | 57 | 78.49% |
| college_biology | 144 | 125 | 19 | 86.81% |
| college_chemistry | 100 | 55 | 45 | 55.00% |
| college_computer_science | 100 | 73 | 27 | 73.00% |
| college_mathematics | 100 | 57 | 43 | 57.00% |
| college_medicine | 173 | 140 | 33 | 80.92% |
| college_physics | 102 | 67 | 35 | 65.69% |
| computer_security | 100 | 81 | 19 | 81.00% |
| conceptual_physics | 235 | 198 | 37 | 84.26% |
| econometrics | 114 | 75 | 39 | 65.79% |
| electrical_engineering | 145 | 116 | 29 | 80.00% |
| elementary_mathematics | 378 | 297 | 81 | 78.57% |
| formal_logic | 126 | 85 | 41 | 67.46% |
| global_facts | 100 | 52 | 48 | 52.00% |
| high_school_biology | 310 | 282 | 28 | 90.97% |
| high_school_chemistry | 203 | 156 | 47 | 76.85% |
| high_school_computer_science | 100 | 86 | 14 | 86.00% |
| high_school_european_history | 165 | 141 | 24 | 85.45% |
| high_school_geography | 198 | 172 | 26 | 86.87% |
| high_school_government_and_politics | 193 | 180 | 13 | 93.26% |
| high_school_macroeconomics | 390 | 318 | 72 | 81.54% |
| high_school_mathematics | 270 | 147 | 123 | 54.44% |
| high_school_microeconomics | 238 | 223 | 15 | 93.70% |
| high_school_physics | 151 | 111 | 40 | 73.51% |
| high_school_psychology | 545 | 498 | 47 | 91.38% |
| high_school_statistics | 216 | 170 | 46 | 78.70% |
| high_school_us_history | 204 | 175 | 29 | 85.78% |
| high_school_world_history | 237 | 203 | 34 | 85.65% |
| human_aging | 223 | 166 | 57 | 74.44% |
| human_sexuality | 131 | 109 | 22 | 83.21% |
| international_law | 121 | 88 | 33 | 72.73% |
| jurisprudence | 108 | 87 | 21 | 80.56% |
| logical_fallacies | 163 | 133 | 30 | 81.60% |
| machine_learning | 112 | 71 | 41 | 63.39% |
| management | 103 | 90 | 13 | 87.38% |
| marketing | 234 | 221 | 13 | 94.44% |
| medical_genetics | 100 | 83 | 17 | 83.00% |
| miscellaneous | 783 | 672 | 111 | 85.82% |
| moral_disputes | 346 | 261 | 85 | 75.43% |
| moral_scenarios | 895 | 486 | 409 | 54.30% |
| nutrition | 306 | 245 | 61 | 80.07% |
| philosophy | 311 | 242 | 69 | 77.81% |
| prehistory | 324 | 273 | 51 | 84.26% |
| professional_accounting | 282 | 164 | 118 | 58.16% |
| professional_law | 1,534 | 786 | 748 | 51.24% |
| professional_medicine | 272 | 223 | 49 | 81.99% |
| professional_psychology | 612 | 477 | 135 | 77.94% |
| public_relations | 110 | 76 | 34 | 69.09% |
| security_studies | 245 | 195 | 50 | 79.59% |
| sociology | 201 | 174 | 27 | 86.57% |
| us_foreign_policy | 100 | 85 | 15 | 85.00% |
| virology | 166 | 93 | 73 | 56.02% |
| world_religions | 171 | 147 | 24 | 85.96% |

## Interpretation and comparison limits

This completed run provides a baseline for future experiments using the same checkpoint provenance, dataset, demonstrations, answer-token scoring, and aggregation. Comparisons should report both weighted and subject-macro accuracy and inspect per-question changes, particularly ties. A fixed seed does not establish identical results across batch shapes, kernels, or repeated runs. No repeatability study, quantized comparison, thinking-mode comparison, or external leaderboard comparison was performed here.

## Evidence and reproduction

The source directory is `artifacts/mmlu/qwen3-8b-bf16-full-official-1/` in the main checkout. Artifacts are ignored by Git and are not bundled with this report, so these paths require access to that checkout. The table below identifies the exact files analyzed by content hash.

| File | SHA-256 |
|---|---|
| `config.json` | `3d9dd1835fb5b9906bfcc937af0551237e318f062cd09853242f16e7c54f85ed` |
| `metrics.json` | `aa4bdfefc37f86f8a7256d48cf8447a402250992bb8ad8eff1ea903a08110f24` |
| `runtime.json` | `7bc7e45b573d4fa65bc2627d34b0fd7f864cec1d57be42baa85bdab94805a23f` |
| `predictions.jsonl` | `9ab82e3b06083b8ea477189d466ae777181929e71fc0301e17e69f1310de7df8` |

For environment setup and a new run, see the [MMLU runbook](../runbooks/mmlu-baseline.md). Use a new output directory because the evaluator refuses to overwrite existing artifacts. Protocol and metric definitions are implemented in [prompting.py](../../src/inference_experiments/mmlu/prompting.py), [engine.py](../../src/inference_experiments/mmlu/engine.py), and [metrics.py](../../src/inference_experiments/mmlu/metrics.py). Timing boundaries are in [evaluate.py](../../src/inference_experiments/mmlu/evaluate.py).

This report analyzes existing artifacts offline in gdevbox; it does not rerun GPU inference.
