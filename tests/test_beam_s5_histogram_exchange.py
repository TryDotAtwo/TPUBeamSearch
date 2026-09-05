import importlib.util
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu


def test_single_rank_histogram_exchange_retains_both_planes_and_tiles():
    assert importlib.util.find_spec('tpu_beam_search.beam_s5_histogram_exchange') is not None
    from tpu_beam_search.beam_s5_histogram_exchange import make_s5_histogram_call
    values = np.arange(512,dtype=np.uint32).reshape(2,256)
    values[0,0] = 0xffffffff
    got = make_s5_histogram_call(SimpleNamespace(size=1),width=256,
        interpret=pltpu.InterpretParams(detect_races=True))(jnp.asarray(values))
    np.testing.assert_array_equal(got,values)


def test_eight_rank_histogram_exchange_traces_full_pair_width():
    from tpu_beam_search.beam_s5_histogram_exchange import make_s5_histogram_call
    fn = make_s5_histogram_call(SimpleNamespace(size=8),width=256)
    traced = jax.make_jaxpr(fn,axis_env=[('core',8)])(
        jax.ShapeDtypeStruct((2,256),jnp.uint32))
    assert tuple(x.shape for x in traced.out_avals) == ((2,256),)
