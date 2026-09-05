import os
import subprocess
import numpy as np
import pytest
from test_beam_hash import oracle


@pytest.mark.skipif(not os.environ.get('BEAM_ROUTE_ORACLE'),reason='route oracle not configured')
def test_original_cpp_hash_routing_accepts_arbitrary_metadata_hashes():
    words = np.random.default_rng(731).integers(0,2**32,(17,4),dtype=np.uint32)
    words[0],words[1] = 0,0xffffffff
    tokens = [len(words),8,3,*words.ravel()]
    result = subprocess.run([os.environ['BEAM_ROUTE_ORACLE'],'route'],
        input=' '.join(map(str,tokens)),text=True,capture_output=True,timeout=30)
    assert result.returncode == 0, result.stderr
    expected = []
    for w in words:
        lo,hi = int(w[0])|(int(w[1]) << 32),int(w[2])|(int(w[3]) << 32)
        expected.append([oracle(lo,hi,0x243f6a8885a308d3)%8,
                         oracle(lo,hi,0x13198a2e03707344)%3])
    actual = [list(map(int,line.split())) for line in result.stdout.splitlines()]
    assert actual == expected
