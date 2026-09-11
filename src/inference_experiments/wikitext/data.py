"""Strict WikiText loading, rendering, and tokenization."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from inference_experiments.wikitext.models import (
    DataValidationError,
    Split,
    TokenizedCorpus,
    WikiTextRow,
)


class Tokenizer(Protocol):
    """Minimal tokenizer behavior used to freeze a corpus token plan."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


def load_wikitext_rows(data_dir: Path, split: Split) -> tuple[WikiTextRow, ...]:
    """Load a split from ordered shards and validate every source row."""
    paths = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not paths:
        raise DataValidationError(f"WikiText split has no parquet shards: {split}")
    if len({path.name for path in paths}) != len(paths):
        raise DataValidationError("WikiText split has duplicate shard names")

    rows: list[WikiTextRow] = []
    for path in paths:
        table = pq.read_table(path)
        if table.column_names != ["text"]:
            raise DataValidationError(
                f"{path.name}: expected exactly one text column, got "
                f"{table.column_names}"
            )
        if not pa.types.is_string(table.schema.field("text").type):
            raise DataValidationError(f"{path.name}: text column must be string")
        for value in table.column("text").to_pylist():
            index = len(rows)
            if not isinstance(value, str):
                raise DataValidationError(f"row {index} has non-string text")
            rows.append(WikiTextRow(source_index=index, text=value))
    if not rows:
        raise DataValidationError("WikiText split contains no rows")
    return tuple(rows)


def render_corpus(rows: tuple[WikiTextRow, ...]) -> str:
    """Join raw rows without normalizing meaningful whitespace."""
    if not rows:
        raise DataValidationError("cannot render an empty WikiText split")
    for expected, row in enumerate(rows):
        if row.source_index != expected:
            raise DataValidationError("WikiText rows are not in contiguous order")
    return "\n\n".join(row.text for row in rows)


def tokenize_corpus(
    rows: tuple[WikiTextRow, ...],
    tokenizer: Tokenizer,
    tokenizer_dir: Path,
    split: Split,
    *,
    limit_tokens: int | None = None,
) -> TokenizedCorpus:
    """Create and fingerprint the exact token plan used by every model run."""
    text = render_corpus(rows)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if not all(isinstance(token_id, int) and token_id >= 0 for token_id in encoded):
        raise DataValidationError("tokenizer returned an invalid token ID")
    if limit_tokens is not None:
        encoded = encoded[:limit_tokens]
    if len(encoded) < 2:
        raise DataValidationError("WikiText corpus must contain at least two tokens")
    token_ids = tuple(encoded)
    return TokenizedCorpus(
        split=split,
        row_count=len(rows),
        utf8_bytes=len(text.encode("utf-8")),
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        tokenizer_sha256=tokenizer_sha256(tokenizer_dir),
        token_ids_sha256=token_ids_sha256(token_ids),
        token_ids=token_ids,
    )


def tokenizer_sha256(tokenizer_dir: Path) -> str:
    """Hash tokenizer inputs in stable filename order."""
    if not tokenizer_dir.is_dir():
        raise DataValidationError(
            f"tokenizer directory does not exist: {tokenizer_dir}"
        )
    exact_names = {"merges.txt", "special_tokens_map.json", "vocab.json"}
    paths = sorted(
        path
        for path in tokenizer_dir.iterdir()
        if path.is_file()
        and (path.name.startswith("tokenizer") or path.name in exact_names)
    )
    if not paths:
        raise DataValidationError("tokenizer directory has no tokenizer files")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def token_ids_sha256(token_ids: tuple[int, ...]) -> str:
    """Hash ordered non-negative token IDs with a fixed-width encoding."""
    digest = hashlib.sha256()
    for token_id in token_ids:
        if token_id < 0 or token_id >= 2**64:
            raise DataValidationError(f"token ID is outside uint64: {token_id}")
        digest.update(struct.pack(">Q", token_id))
    return digest.hexdigest()
