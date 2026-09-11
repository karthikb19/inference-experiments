"""Deterministic WikiText test helpers."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_text_shard(path: Path, texts: list[str | None]) -> None:
    """Write a tiny source-compatible WikiText parquet shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": pa.array(texts, type=pa.string())})
    pq.write_table(table, path)
