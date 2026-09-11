"""Shared deterministic MMLU test data builders."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_split(
    path: Path,
    rows: list[tuple[str, str, list[str], int]],
) -> None:
    """Write the exact repository MMLU parquet schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "question": [row[0] for row in rows],
            "subject": [row[1] for row in rows],
            "choices": [row[2] for row in rows],
            "answer": [row[3] for row in rows],
        }
    )
    pq.write_table(table, path)
