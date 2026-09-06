import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('case', ['valid','empty','duplicate','missing','extra','overflow'])
def test_exact_target_coverage(case):
    from tpu_beam_search.beam_final_target_coverage import pallas_final_target_coverage
    targets = np.zeros((1,128),np.uint32)
    valid = np.zeros_like(targets)
    targets[0,:3] = [2,0,1]
    valid[0,:3] = 1
    count = 3
    if case == 'empty':
        count = 0
        valid[:] = 0
    if case == 'duplicate': targets[0,2] = 0
    if case == 'missing': valid[0,2] = 0
    if case == 'extra': valid[0,3] = 1
    if case == 'overflow': count = 129
    reasons = pallas_final_target_coverage(jnp.asarray(targets),jnp.asarray(valid),
        jnp.array([count],jnp.uint32),interpret=True)
    assert bool(np.asarray(reasons).any()) == (case not in ('valid','empty'))


def test_cross_tile_sparse_validity_ignores_padding_and_detects_duplicate():
    from tpu_beam_search.beam_final_target_coverage import pallas_final_target_coverage
    rng = np.random.default_rng(614)
    slots = rng.choice(256,129,replace=False)
    targets = np.full((1,256),0xffffffff,np.uint32)
    targets[0,slots] = rng.permutation(129).astype(np.uint32)
    valid = np.zeros((1,256),np.uint32)
    valid[0,slots] = 7  # Any nonzero validity is true.
    count = jnp.array([129],jnp.uint32)
    reasons = pallas_final_target_coverage(jnp.asarray(targets),jnp.asarray(valid),count,interpret=True)
    assert not np.asarray(reasons).any()
    targets[0,slots[-1]] = targets[0,slots[0]]
    reasons = pallas_final_target_coverage(jnp.asarray(targets),jnp.asarray(valid),count,interpret=True)
    assert np.asarray(reasons).any()
