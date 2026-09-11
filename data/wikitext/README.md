# WikiText data

This directory contains the `wikitext-103-raw-v1` configuration from the
[`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext)
Hugging Face dataset.

The files were downloaded from immutable revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. They include the `train`
(1,801,350 rows), `validation` (3,760 rows), and `test` (4,358 rows) splits.
Every split has one nullable string column named `text`.

The two upstream train parquet files were losslessly re-sharded in row order
into four files so that each committed file remains below GitHub's 100 MB
limit. The test and validation files are unchanged from upstream.
