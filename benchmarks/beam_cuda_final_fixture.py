"""Load immutable CUDA final-materialization fixtures without regenerating inputs."""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def load_cases(root):
    root = Path(root)
    report = json.loads((root / 'report.json').read_text())
    if report.get('all_exact') is not True:
        raise ValueError('CUDA oracle must pass first')
    cases, seen = [], set()
    for row in report['cases']:
        count, mode = row['count'], row['mode']
        if not isinstance(count, int) or not 0 <= count <= 1 << 20 or mode not in ('local', 'remote'):
            raise ValueError('invalid fixture identity')
        name = f'count{count}_{mode}'
        if name in seen or row.get('exact') is not True or row.get('returncode') != 0:
            raise ValueError('duplicate or failed CUDA case')
        seen.add(name)
        blob = (root / f'{name}.bin').read_bytes()
        cuda = (root / f'{name}.out').read_bytes()
        if hashlib.sha256(blob).hexdigest() != row['input_sha256']:
            raise ValueError('input SHA mismatch')
        if hashlib.sha256(cuda).hexdigest() != row['output_sha256']:
            raise ValueError('CUDA SHA mismatch')
        if len(blob) < 32 or blob[:8] != b'TFIN0001':
            raise ValueError('invalid fixture header')
        parents_count, n, moves, state_len, width, target = struct.unpack_from('<6I', blob, 8)
        if not 0 < parents_count <= 1 << 20 or n != count or target != count:
            raise ValueError('invalid fixture counts')
        if (moves, state_len, width) != (24, 120, 128):
            raise ValueError('unsupported fixture geometry')
        if len(blob) != 32 + parents_count * width + count * 16 + moves * width or len(cuda) != count * width:
            raise ValueError('fixture byte length mismatch')
        cursor = 32
        parents = np.frombuffer(blob, np.uint8, parents_count * width, cursor).reshape(parents_count, width).copy()
        cursor += parents_count * width
        raw = np.frombuffer(blob, '<u4', count * 4, cursor).reshape(count, 4)
        cursor += count * 16
        generators = np.frombuffer(blob, np.uint8, moves * width, cursor).reshape(moves, width).astype(np.int32)
        capacity = max(128, (count + 127) // 128 * 128)
        requests = np.zeros((4, capacity), np.uint32)
        requests[:, :count] = raw.T
        expected = np.zeros((capacity, width), np.uint8)
        expected[:count] = np.frombuffer(cuda, np.uint8).reshape(count, width)
        cases.append(dict(name=name, input_sha256=row['input_sha256'], cuda_sha256=row['output_sha256'],
                          inputs=(parents, generators, requests, np.array([count], np.uint32),
                                  np.array([target], np.uint32)), expected=expected))
    if not cases:
        raise ValueError('empty CUDA evidence')
    return cases
