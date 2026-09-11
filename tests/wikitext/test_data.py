"""Tests for exact WikiText loading and token-plan construction."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from inference_experiments.wikitext.data import (
    load_wikitext_rows,
    render_corpus,
    tokenize_corpus,
)
from inference_experiments.wikitext.models import DataValidationError
from tests.wikitext.conftest import write_text_shard


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert text == "first\n\n\n\nlast"
        assert not add_special_tokens
        return [10, 11, 12, 13]


class OneTokenTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return [10]


def test_load_and_render_preserves_empty_rows_in_shard_order(tmp_path: Path) -> None:
    write_text_shard(tmp_path / "test-00001-of-00002.parquet", ["last"])
    write_text_shard(tmp_path / "test-00000-of-00002.parquet", ["first", ""])

    rows = load_wikitext_rows(tmp_path, "test")

    assert [row.source_index for row in rows] == [0, 1, 2]
    assert render_corpus(rows) == "first\n\n\n\nlast"


def test_load_rejects_null_and_schema_drift(tmp_path: Path) -> None:
    write_text_shard(tmp_path / "test-00000-of-00001.parquet", [None])
    with pytest.raises(DataValidationError, match="non-string"):
        load_wikitext_rows(tmp_path, "test")

    (tmp_path / "test-00000-of-00001.parquet").unlink()
    pq.write_table(
        pa.table({"text": ["valid"], "extra": [1]}),
        tmp_path / "test-00000-of-00001.parquet",
    )
    with pytest.raises(DataValidationError, match="exactly one text column"):
        load_wikitext_rows(tmp_path, "test")


def test_load_rejects_missing_split(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="has no parquet shards"):
        load_wikitext_rows(tmp_path, "test")


def test_tokenize_corpus_limits_and_fingerprints_tokens(tmp_path: Path) -> None:
    write_text_shard(
        tmp_path / "data" / "test-00000-of-00001.parquet", ["first", "", "last"]
    )
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}\n")
    rows = load_wikitext_rows(tmp_path / "data", "test")

    corpus = tokenize_corpus(
        rows,
        FakeTokenizer(),
        tokenizer_dir,
        "test",
        limit_tokens=3,
    )

    assert corpus.token_ids == (10, 11, 12)
    assert corpus.row_count == 3
    assert corpus.utf8_bytes == len(b"first\n\n\n\nlast")
    assert len(corpus.text_sha256) == 64
    assert len(corpus.tokenizer_sha256) == 64
    assert len(corpus.token_ids_sha256) == 64


def test_tokenize_rejects_corpus_without_a_target(tmp_path: Path) -> None:
    write_text_shard(tmp_path / "data" / "test-00000-of-00001.parquet", ["only"])
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}\n")
    rows = load_wikitext_rows(tmp_path / "data", "test")

    with pytest.raises(DataValidationError, match="at least two tokens"):
        tokenize_corpus(rows, OneTokenTokenizer(), tokenizer_dir, "test")
