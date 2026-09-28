import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_private_boundary_launcher_pins_clean_source_and_isolates_output():
    from kaggle_beam_response_boundary.run import COMMIT_SHA, CHECKOUT, OUTPUT
    source = (ROOT / 'kaggle_beam_response_boundary' / 'run.py').read_text()
    metadata = json.loads((ROOT / 'kaggle_beam_response_boundary' /
                           'kernel-metadata.json').read_text())
    assert COMMIT_SHA == '1c1e9d82948ee7e2f01b119a9e57517a38076c2c'
    assert CHECKOUT.as_posix().startswith('/tmp/')
    assert OUTPUT.as_posix().startswith('/kaggle/working/')
    assert 'benchmarks.beam_response_boundary_probe' in source
    assert "actual != COMMIT_SHA or dirty" in source
    assert metadata['is_private'] and metadata['enable_tpu']
    assert not metadata['enable_gpu']
