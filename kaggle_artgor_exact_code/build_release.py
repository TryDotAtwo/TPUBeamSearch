"""Build a minimal Kaggle dataset from one immutable Git commit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPOSITORY = "https://github.com/TryDotAtwo/TPUBeamSearch.git"
ARTGOR_SCRIPT_VERSION = 344319112
RUNTIME_FILES = (
    "tpu_beam_search/__init__.py",
    "tpu_beam_search/artgor_exact_inference.py",
    "tpu_beam_search/artgor_staged_beam.py",
    "tpu_beam_search/sharding.py",
    "tpu_beam_search/stream1_architecture.py",
    "tpu_beam_search/stream1_embedding_experimental.py",
    "tpu_beam_search/stream1_layernorm_exact.py",
    "tpu_beam_search/stream1_layernorm_pallas.py",
    "tpu_beam_search/stream1_layernorm_reference.py",
    "tpu_beam_search/stream1_pallas.py",
    "tpu_beam_search/tpu_layout.py",
)
RUNTIME_ARCHIVE = "tpu_beam_search_runtime.zip"
TARGET_RUNTIME = {
    "python": "3.12",
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
    "libtpu": "0.0.42.1",
}


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _validate_commit(source_commit: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be a lowercase 40-hex Git SHA")
    try:
        object_type = _git("cat-file", "-t", source_commit).decode().strip()
    except subprocess.CalledProcessError as error:
        raise ValueError("source_commit does not exist in this repository") from error
    if object_type != "commit":
        raise ValueError("source_commit must name a commit object")
    return source_commit


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def build_release(output: Path | str, *, source_commit: str) -> Path:
    """Materialize only ``RUNTIME_FILES`` as stored in ``source_commit``."""

    source_commit = _validate_commit(source_commit)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("release output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)

    hashes: dict[str, str] = {}
    payloads: dict[str, bytes] = {}
    for relative in RUNTIME_FILES:
        source_path = f"src/{relative}"
        try:
            payload = _git("show", f"{source_commit}:{source_path}")
        except subprocess.CalledProcessError as error:
            raise ValueError(
                f"allowlisted file is absent from source commit: {source_path}"
            ) from error
        payloads[relative] = payload
        hashes[relative] = _sha256(payload)

    archive_path = output / RUNTIME_ARCHIVE
    with zipfile.ZipFile(archive_path, mode="w") as runtime_zip:
        for relative in RUNTIME_FILES:
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            runtime_zip.writestr(
                info,
                payloads[relative],
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )

    metadata_source = Path(__file__).with_name("dataset-metadata.json")
    metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
    metadata_bytes = (
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    _write(output / "dataset-metadata.json", metadata_bytes)

    manifest = {
        "schema_version": 1,
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": source_commit,
        "artgor_script_version": ARTGOR_SCRIPT_VERSION,
        "target_runtime": TARGET_RUNTIME,
        "dataset_metadata_sha256": _sha256(metadata_bytes),
        "runtime_archive": {
            "name": RUNTIME_ARCHIVE,
            "sha256": _sha256(archive_path.read_bytes()),
        },
        "files": hashes,
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")
    _write(output / "release_manifest.json", manifest_bytes)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = build_release(args.output, source_commit=args.source_commit)
    print(output)


if __name__ == "__main__":
    main()
