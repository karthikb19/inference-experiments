# inference-experiments
doing some experiments quantizing models, training spec dec models, etc

## Download Qwen3-8B

Install the project and development dependencies with:

```bash
uv sync
```

Then download the model snapshot from Hugging Face Hub:

```bash
uv run python scripts/download_qwen3_8b.py
```

The files are stored in `models/Qwen3-8B` by default. Use `--output-dir` to
choose a different location, or `--token` for a private or gated repository.
