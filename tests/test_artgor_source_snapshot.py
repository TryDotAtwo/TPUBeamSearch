import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "third_party" / "artgor_cube555_v344319112"


def test_artgor_snapshot_is_the_frozen_script_version():
    manifest = json.loads((SNAPSHOT / "manifest.json").read_text())
    assert manifest["script_version_id"] == 344319112
    assert manifest["source_url"].endswith("scriptVersionId=344319112")
    expected = {
        "cayleypy-cube555-tpu-beam-q.ipynb": (
            "c74613a9fa400b391aca49bb128a2f6d3b0465e8e7cb933abc9b126a317e0e0b"
        ),
        "jax_model.py": (
            "6d00da89ce45cf84167db20780e30f676cde3ae756d376c8e05a7e0dcf98e46e"
        ),
        "jax_beam_spmd_v_only.py": (
            "aaa0dbe16fd82a0f2bc08f1216f4e87c8a2a63c855f5d7012b6c18d8b57d42cb"
        ),
    }
    assert manifest["sha256"] == expected
    for name, digest in expected.items():
        assert hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest() == digest
