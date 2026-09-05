import importlib.util
import numpy as np
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize('rank',[0,1,2,3])
def test_s4_gate_fixture_matches_reserved_job_with_empty_and_nonempty_ranks(rank):
    assert importlib.util.find_spec('benchmarks.beam_s4_probe') is not None
    from benchmarks.beam_s4_probe import fixtures
    from tpu_beam_search.beam_s4_commit import pallas_run_reserved_s4
    inputs,expected = fixtures()
    assert all(x.shape[0] == 8 for x in (*inputs,*expected))
    actual = pallas_run_reserved_s4(*(jnp.asarray(x[rank]) for x in inputs),
                                   bins=128,interpret=True)
    for got,want in zip(actual,expected,strict=True):
        np.testing.assert_array_equal(got,want[rank])
    assert expected[3][rank,3,0] == (inputs[3][rank,3,0]^1)
    assert expected[3][rank,2,0] == 0
