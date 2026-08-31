"""Lossless, deterministic packaging of raw follow-up evidence; stdlib only.

Default: preserve raw originals, compress >90MiB files, generate manifest.
--compress-only: create/verify archives but do not finalize the manifest.
--check: regenerate/hash-validate manifest without writes. In a fresh checkout,
ignored oversized originals may be absent; their bytes are verified by streaming
archive decompression instead. No original is silently reconstructed or edited.
--self-test: synthetic fixture tests in an isolated temporary directory.
"""

from __future__ import annotations

import tempfile
import argparse
import contextlib
import gzip
import hashlib
import json
from pathlib import Path
import unittest


PRODUCER = "package_artifacts.py"
LOG_NAME = "tpu-layernorm-arithmetic-followup.log"
CHUNK_SIZE = 1024 * 1024


def digest(stream):
    sha, size = hashlib.sha256(), 0
    while block := stream.read(CHUNK_SIZE):
        sha.update(block)
        size += len(block)
    return size, sha.hexdigest()


def raw_digest(path):
    with path.open("rb") as stream:
        return digest(stream)


def validate_archive(path, size, sha):
    with path.open("rb") as stream:
        header = stream.read(10)
    if len(header) != 10 or header[:4] != b"\x1f\x8b\x08\x00" or header[4:8] != b"\x00" * 4:
        raise ValueError(f"non-deterministic or invalid gzip header: {path.name}")
    with gzip.open(path, "rb") as stream:
        expanded = digest(stream)
    if expanded != (size, sha):
        raise ValueError(f"archive decompression SHA256/size mismatch: {path.name}")
    archive_size, archive_sha = raw_digest(path)
    if archive_size >= 90 * 1024 * 1024:
        raise ValueError(f"archive still exceeds safe publication size: {path.name}")
    return {"path": f"archives/{path.name}", "bytes": archive_size, "sha256": archive_sha,
            "decompressed_bytes": size, "decompressed_sha256": sha,
            "format": "gzip", "gzip_filename": "", "gzip_mtime": 0, "compression_level": 9}


def package(root, *, threshold=90 * 1024 * 1024, chunk_bytes=64 * 1024 * 1024,
            check=False, compress_only=False, expected_count=None):
    root = root.resolve()
    manifest_path = root / "artifact_manifest.json"
    previous_bytes = manifest_path.read_bytes() if manifest_path.exists() else None
    previous = json.loads(previous_bytes) if previous_bytes is not None else None
    if previous is not None and previous.get("producer") != PRODUCER:
        raise ValueError("refusing to overwrite an unrelated artifact_manifest.json")
    if check and previous is None:
        raise ValueError("--check requires an existing artifact_manifest.json")
    if check:
        threshold = previous["compression_threshold_bytes"]
        chunk_bytes = previous["raw_chunk_bytes"]
    if not 0 < chunk_bytes <= 64 * 1024 * 1024:
        raise ValueError("raw chunk size must be positive and at most64MiB")
    raw_root = root / "arithmetic_followup"
    actual_paths = sorted(path for path in raw_root.rglob("*") if path.is_file())
    # Generated archives live outside arithmetic_followup, so raw .trace.json.gz
    # remains evidence, not an accidentally excluded compression artifact.
    actual_paths.append(root / LOG_NAME)
    actual = {}
    for path in actual_paths:
        if not path.is_file():
            raise ValueError(f"missing required raw file: {path.name}")
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("raw file escapes the named result directory")
        size, sha = raw_digest(path)
        if size == 0:
            raise ValueError(f"zero-length raw artifact: {path.relative_to(root)}")
        actual[path.relative_to(root).as_posix()] = {"path": path.relative_to(root).as_posix(),
                                                     "bytes": size, "sha256": sha}
    declared = {entry["path"]: entry for entry in previous["files"]} if check else {}
    if check and set(actual) - set(declared):
        raise ValueError(f"unexpected raw files: {sorted(set(actual) - set(declared))}")
    for name in set(declared) - set(actual):
        entry = declared[name]
        if "archive_parts" not in entry or entry["bytes"] <= threshold:
            raise ValueError(f"missing unarchived raw file: {name}")
        actual[name] = {k: entry[k] for k in ("path", "bytes", "sha256")}
    if not any(name.startswith("arithmetic_followup/") for name in actual):
        raise ValueError("zero raw files under arithmetic_followup")
    if expected_count is not None and len(actual) != expected_count:
        raise ValueError(f"raw count {len(actual)} != expected {expected_count}")
    archives = {}
    groups = {}
    files = []
    for name, entry in sorted(actual.items()):
        entry = dict(entry)
        if entry["bytes"] > threshold:
            if entry["sha256"] not in groups:
                part_count = (entry["bytes"] + chunk_bytes - 1) // chunk_bytes
                part_names = []
                combined_sha, combined_size = hashlib.sha256(), 0
                source_path = root / name
                with source_path.open("rb") if source_path.exists() else contextlib.nullcontext(None) as source:
                    for index in range(part_count):
                        suffix = ".raw.gz" if part_count == 1 else f".part{index+1:03d}-of{part_count:03d}.raw.gz"
                        archive_name = f"archives/{entry['sha256']}{suffix}"
                        archive_path = root / archive_name
                        if source is None:
                            if not check:
                                raise ValueError(f"raw source missing: {name}")
                            with gzip.open(archive_path, "rb") as decoded:
                                data = decoded.read(chunk_bytes + 1)
                        else:
                            data = source.read(chunk_bytes)
                        expected_size = min(chunk_bytes, entry["bytes"] - index * chunk_bytes)
                        if len(data) != expected_size:
                            raise ValueError(f"wrong decompressed chunk size: {archive_name}")
                        chunk_sha = hashlib.sha256(data).hexdigest()
                        combined_sha.update(data)
                        combined_size += len(data)
                        if not archive_path.exists():
                            if check:
                                raise ValueError(f"missing archive: {archive_name}")
                            archive_path.parent.mkdir(exist_ok=True)
                            # Exclusive creation: never overwrite unrelated data.
                            with archive_path.open("xb") as destination:
                                with gzip.GzipFile(filename="", mode="wb", compresslevel=9,
                                                   fileobj=destination, mtime=0) as encoded:
                                    encoded.write(data)
                        archives[archive_name] = validate_archive(archive_path, len(data), chunk_sha)
                        archives[archive_name].update({"raw_paths": [], "part_index": index + 1,
                                                      "part_count": part_count, "raw_offset_bytes": index * chunk_bytes,
                                                      "whole_raw_sha256": entry["sha256"]})
                        part_names.append(archive_name)
                    if source is not None and source.read(1):
                        raise ValueError(f"raw source grew during packaging: {name}")
                if (combined_size, combined_sha.hexdigest()) != (entry["bytes"], entry["sha256"]):
                    raise ValueError(f"concatenated decompressed parts disagree with raw SHA256/size: {name}")
                groups[entry["sha256"]] = part_names
            entry["archive_parts"] = groups[entry["sha256"]]
            for part_name in entry["archive_parts"]:
                archives[part_name]["raw_paths"].append(name)
            entry["publication_storage"] = "lossless_archive_raw_original_ignored"
        else:
            entry["publication_storage"] = "raw"
        files.append(entry)
    # Verify all still-present originals after compression and hashing; a
    # concurrent downloader or accidental mutation cannot finalize this manifest.
    for entry in files:
        raw = root / entry["path"]
        if raw.exists() and raw_digest(raw) != (entry["bytes"], entry["sha256"]):
            raise ValueError(f"raw artifact changed during packaging: {entry['path']}")
    actual_archives = {p.relative_to(root).as_posix() for p in (root / "archives").glob("*.raw.gz")}
    if actual_archives != set(archives):
        raise ValueError("unexpected/unreferenced generated archives; no files removed automatically")
    archive_list = [archives[name] for name in sorted(archives)]
    raw_total = sum(e["bytes"] for e in files)
    oversized = [e for e in files if "archive_parts" in e]
    result = {
        "schema_version": 1, "producer": PRODUCER,
        "raw_scope": "arithmetic_followup/** plus tpu-layernorm-arithmetic-followup.log; generated archives excluded",
        "compression_threshold_bytes": threshold,
        "raw_chunk_bytes": chunk_bytes,
        "archive_reconstruction": "For each raw file, gzip-decompress archive_parts in the recorded order and concatenate the resulting bytes; verify the raw file bytes and SHA256. Both identical raw paths map to the same ordered parts.",
        "raw_file_count": len(files),
        "arithmetic_followup_file_count": sum(e["path"].startswith("arithmetic_followup/") for e in files),
        "top_level_kaggle_log_count": 1,
        "raw_bytes": raw_total,
        "oversized_raw_file_count": len(oversized),
        "archive_count": len(archives),
        "archive_bytes": sum(a["bytes"] for a in archive_list),
        "publication_file_count": len(files) - len(oversized) + len(archives),
        "publication_bytes": raw_total - sum(e["bytes"] for e in oversized) + sum(a["bytes"] for a in archive_list),
        "hash_scope": "All sizes and SHA256 values are computed from downloaded local bytes, not remote API assertions.",
        "source_listing": {
            "remote_names_verification": "The accompanying report records the separately performed Kaggle filename-set check; this script does not query Kaggle or claim remote hashes/sizes.",
            "arithmetic_followup_local_names": sum(e["path"].startswith("arithmetic_followup/") for e in files),
            "separate_top_level_log": LOG_NAME,
        },
        "checks": {"no_zero_length_raw_files": True, "raw_bytes_unchanged": True,
                   "archive_decompressed_sha256_and_size_match": True,
                   "deterministic_gzip_headers": True, "archives_below_90MiB": True},
        "files": files, "archives": archive_list,
    }
    rendered = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if check:
        if rendered != previous_bytes:
            raise ValueError("regenerated manifest differs: raw files/archive contents changed")
    elif not compress_only:
        manifest_path.write_bytes(rendered)
    return result


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tpu-artifact-package-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "arithmetic_followup").mkdir()
        (self.root / "arithmetic_followup" / "a.txt").write_bytes(b"raw\r\n" * 64)
        (self.root / "arithmetic_followup" / "b.txt").write_bytes(b"raw\r\n" * 64)
        (self.root / "tpu-layernorm-arithmetic-followup.log").write_bytes(b"COMPLETE\n")

    def test_roundtrip_deduplicates_and_checks_with_ignored_original_missing(self):
        result = package(self.root, threshold=32, expected_count=3)
        self.assertEqual(result.get("raw_file_count"), 3)
        self.assertEqual(result["archive_count"], 1)
        self.assertEqual(result["oversized_raw_file_count"], 2)
        original = (self.root / "arithmetic_followup" / "a.txt").read_bytes()
        self.assertEqual(original, b"raw\r\n" * 64)
        (self.root / "arithmetic_followup" / "a.txt").unlink()
        self.assertEqual(package(self.root, check=True), result)

    def test_zero_length_raw_rejected(self):
        (self.root / "arithmetic_followup" / "empty.txt").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "zero-length"):
            package(self.root, threshold=32)

    def test_chunked_roundtrip_preserves_part_order_and_raw_hash(self):
        result = package(self.root, threshold=32, chunk_bytes=64, expected_count=3)
        self.assertEqual(result["archive_count"], 5)
        paths = result["files"][0]["archive_parts"]
        self.assertEqual(len(paths), 5)
        self.assertEqual(b"".join(gzip.decompress((self.root / path).read_bytes()) for path in paths), b"raw\r\n" * 64)
        (self.root / "arithmetic_followup" / "a.txt").unlink()
        (self.root / "arithmetic_followup" / "b.txt").unlink()
        self.assertEqual(package(self.root, check=True), result)

    def test_preexisting_bad_archive_not_overwritten(self):
        import hashlib
        sha = hashlib.sha256(b"raw\r\n" * 64).hexdigest()
        (self.root / "archives").mkdir()
        path = self.root / "archives" / f"{sha}.raw.gz"
        path.write_bytes(b"unrelated file")
        with self.assertRaises((ValueError, OSError, EOFError)):
            package(self.root, threshold=32)
        self.assertEqual(path.read_bytes(), b"unrelated file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--compress-only", action="store_true")
    parser.add_argument("--expected-raw-count", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    else:
        if args.check and args.compress_only:
            parser.error("--check and --compress-only are mutually exclusive")
        result = package(Path(__file__).resolve().parent, check=args.check,
                         compress_only=args.compress_only, expected_count=args.expected_raw_count)
        print(f"{'CHECKED' if args.check else 'PACKAGED'}: {result['raw_file_count']} raw files "
              f"({result['arithmetic_followup_file_count']} arithmetic outputs + separate Kaggle log), "
              f"{result['raw_bytes']} raw bytes; {result['oversized_raw_file_count']} oversized raws -> "
              f"{result['archive_count']} archives/{result['archive_bytes']} bytes; "
              f"publication {result['publication_file_count']} files/{result['publication_bytes']} bytes")
