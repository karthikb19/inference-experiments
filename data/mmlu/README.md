# MMLU data

This directory contains the `all` configuration from the
[`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) Hugging Face dataset.

The files were downloaded from revision
`c30699e8356da336a370243923dbaf21066bb9fe` and include the test, validation,
development, and auxiliary-training splits.

The baseline evaluator uses the splits as follows:

- `dev` supplies the five fixed demonstrations for each of the 57 subjects;
- `test` supplies the 14,042 rows used for the final weighted accuracy;
- `validation` is available for development smoke runs with
  `mmlu-baseline --split validation`;
- `auxiliary_train` is not used by the baseline.

The evaluator reads these local parquet files directly and records the dataset
revision in every completed run configuration. It does not download data.
