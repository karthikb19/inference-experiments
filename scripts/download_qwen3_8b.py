"""Download the Qwen3-8B model files from Hugging Face Hub."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPOSITORY = "Qwen/Qwen3-8B"
DEFAULT_OUTPUT_DIRECTORY = Path("models/Qwen3-8B")


def download_model(
    output_directory: Path,
    repository: str = DEFAULT_REPOSITORY,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Download a model snapshot and return the local directory."""
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repository,
        local_dir=str(output_directory),
        revision=revision,
        token=token,
    )
    return output_directory


def parse_args(arguments: list[str] | None = None) -> Namespace:
    """Parse command-line options for the downloader."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory in which to store the model (default: models/Qwen3-8B).",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPOSITORY,
        help=f"Hugging Face repository (default: {DEFAULT_REPOSITORY}).",
    )
    parser.add_argument(
        "--revision",
        help="Optional branch, tag, or commit to download.",
    )
    parser.add_argument(
        "--token",
        help="Optional Hugging Face access token for private or gated repositories.",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Run the downloader CLI."""
    options = parse_args(arguments)
    output_directory = download_model(
        output_directory=options.output_dir,
        repository=options.repo,
        revision=options.revision,
        token=options.token,
    )
    print(f"Downloaded {options.repo} to {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
