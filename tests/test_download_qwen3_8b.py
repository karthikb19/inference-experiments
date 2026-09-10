from dataclasses import dataclass
from pathlib import Path

from scripts import download_qwen3_8b


@dataclass(frozen=True)
class SnapshotDownloadCall:
    repo_id: str
    local_dir: str
    revision: str | None
    token: str | None


def test_download_model_passes_repository_options_to_hugging_face(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[SnapshotDownloadCall] = []

    def fake_snapshot_download(
        *,
        repo_id: str,
        local_dir: str,
        revision: str | None,
        token: str | None,
    ) -> str:
        calls.append(
            SnapshotDownloadCall(
                repo_id=repo_id,
                local_dir=local_dir,
                revision=revision,
                token=token,
            )
        )
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(download_qwen3_8b, "snapshot_download", fake_snapshot_download)

    output_directory = tmp_path / "models" / "Qwen3-8B"
    result = download_qwen3_8b.download_model(
        output_directory=output_directory,
        repository="example/qwen",
        revision="main",
        token="test-token",
    )

    assert result == output_directory
    assert output_directory.parent.is_dir()
    assert calls == [
        SnapshotDownloadCall(
            repo_id="example/qwen",
            local_dir=str(output_directory),
            revision="main",
            token="test-token",
        )
    ]


def test_parse_args_uses_qwen3_defaults() -> None:
    options = download_qwen3_8b.parse_args([])

    assert options.repo == download_qwen3_8b.DEFAULT_REPOSITORY
    assert options.output_dir == download_qwen3_8b.DEFAULT_OUTPUT_DIRECTORY
    assert options.revision is None
    assert options.token is None
