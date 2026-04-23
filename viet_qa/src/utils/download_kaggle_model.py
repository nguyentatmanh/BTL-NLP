from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_MODEL_URL = "https://www.kaggle.com/models/huynguyen199/vietnamese-open-domain"
DEFAULT_MODEL_HANDLE = "huynguyen199/vietnamese-open-domain/transformers/default"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "checkpoints" / "extractive"
)
HF_MARKERS = (
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
WEIGHT_MARKERS = ("model.safetensors", "pytorch_model.bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the extractive checkpoint from Kaggle into the local checkpoint folder."
    )
    parser.add_argument(
        "--model-url",
        default=DEFAULT_MODEL_URL,
        help="Kaggle model page URL.",
    )
    parser.add_argument(
        "--model-handle",
        default=os.getenv("KAGGLE_MODEL_HANDLE"),
        help=(
            "Full Kaggle model handle in the format "
            "<owner>/<model>/<framework>/<variation>. "
            "If omitted, the script derives it from the URL or falls back to a default guess."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the Hugging Face checkpoint should be placed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the local checkpoint directory if it already exists.",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Run kagglehub.login() before downloading.",
    )
    return parser.parse_args()


def has_checkpoint(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").exists()
        and any((path / marker).exists() for marker in WEIGHT_MARKERS)
    )


def score_checkpoint_dir(path: Path) -> tuple[int, int]:
    marker_count = sum((path / marker).exists() for marker in HF_MARKERS)
    depth = len(path.parts)
    return marker_count, -depth


def find_checkpoint_dir(root: Path) -> Path | None:
    candidates: list[Path] = []

    if root.is_dir():
        if has_checkpoint(root):
            candidates.append(root)
        for candidate in root.rglob("*"):
            if candidate.is_dir() and has_checkpoint(candidate):
                candidates.append(candidate)

    if not candidates:
        return None

    return max(candidates, key=score_checkpoint_dir)


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_dir_contents(source: Path, destination: Path) -> None:
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def derive_handle(model_url: str, explicit_handle: str | None) -> tuple[str, bool]:
    if explicit_handle:
        return explicit_handle.strip("/"), False

    parsed = urlparse(model_url)
    segments = [segment for segment in parsed.path.split("/") if segment]

    try:
        models_index = segments.index("models")
    except ValueError:
        return DEFAULT_MODEL_HANDLE, True

    model_segments = segments[models_index + 1 :]
    if len(model_segments) >= 4:
        return "/".join(model_segments[:4]), False

    if len(model_segments) >= 2:
        owner, model = model_segments[:2]
        return f"{owner}/{model}/transformers/default", True

    return DEFAULT_MODEL_HANDLE, True


def print_auth_help() -> None:
    print("Kaggle authentication is required before downloading the model.")
    print("Option 1: create %USERPROFILE%\\.kaggle\\kaggle.json")
    print("Option 2: set KAGGLE_API_TOKEN in your shell")
    print("You can get the token from https://www.kaggle.com/settings")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()

    if has_checkpoint(output_dir) and not args.force:
        print(f"Checkpoint already exists at: {output_dir}")
        print("Use --force if you want to re-download and overwrite it.")
        return 0

    try:
        import kagglehub
    except ImportError:
        print("Missing dependency: kagglehub")
        print("Install it with: pip install kagglehub")
        return 1

    handle, used_fallback_guess = derive_handle(args.model_url, args.model_handle)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix="kaggle_model_", dir=str(output_dir.parent))
    )

    try:
        if args.login:
            kagglehub.login()

        print(f"Kaggle model page: {args.model_url}")
        print(f"Using Kaggle handle: {handle}")
        if used_fallback_guess:
            print(
                "The provided URL does not include framework/variation, "
                "so the script is trying the default guess: transformers/default"
            )

        downloaded_path = Path(
            kagglehub.model_download(
                handle,
                output_dir=str(staging_root),
                force_download=args.force,
            )
        )

        search_root = downloaded_path if downloaded_path.exists() else staging_root
        checkpoint_dir = find_checkpoint_dir(search_root) or find_checkpoint_dir(
            staging_root
        )

        if checkpoint_dir is None:
            print("Downloaded files were found, but no Hugging Face checkpoint was detected.")
            print(f"Inspect the staging folder manually: {staging_root}")
            return 1

        ensure_empty_dir(output_dir)
        copy_dir_contents(checkpoint_dir, output_dir)

        print(f"Checkpoint downloaded successfully to: {output_dir}")
        print("The extractive model is now ready for local inference.")
        return 0
    except Exception as exc:
        print(f"Download failed: {exc}")
        print_auth_help()
        print("If the model variation is not transformers/default, rerun with:")
        print(
            '  py .\\viet_qa\\src\\utils\\download_kaggle_model.py '
            '--model-handle "huynguyen199/vietnamese-open-domain/<framework>/<variation>" --force'
        )
        return 1
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
