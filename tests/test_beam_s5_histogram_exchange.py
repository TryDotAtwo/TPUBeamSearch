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


def test_raw_wire_diagnostic_preserves_single_rank_payload():
    from tpu_beam_search.beam_s5_histogram_exchange import make_s5_histogram_call
    values = np.arange(512,dtype=np.uint32).reshape(2,256)
    fn = make_s5_histogram_call(SimpleNamespace(size=1),width=256,return_wire=True,
        interpret=pltpu.InterpretParams(detect_races=True))
    np.testing.assert_array_equal(fn(jnp.asarray(values)),values)


def test_own_only_diagnostic_has_no_remote_axis_requirement():
    from tpu_beam_search.beam_s5_histogram_exchange import make_s5_histogram_call
    values = np.arange(512,dtype=np.uint32).reshape(2,256)
    fn = make_s5_histogram_call(SimpleNamespace(size=8),width=256,own_only=True,
        interpret=pltpu.InterpretParams(detect_races=True))
    np.testing.assert_array_equal(fn(jnp.asarray(values)),values)


def test_full_shape_local_replication_initializes_every_pair_without_remote():
    from tpu_beam_search.beam_s5_histogram_exchange import make_s5_histogram_call
    values = np.arange(512,dtype=np.uint32).reshape(2,256)+17
    fn = make_s5_histogram_call(SimpleNamespace(size=8),width=256,local_replicate=True,
        interpret=pltpu.InterpretParams(detect_races=True))
    np.testing.assert_array_equal(fn(jnp.asarray(values)),np.tile(values,(8,1)))


def test_initialized_wire_traces_remote_full_output():
    from tpu_beam_search.beam_s5_histogram_exchange import make_s5_histogram_call
    fn = make_s5_histogram_call(SimpleNamespace(size=8),width=256,
        return_wire=True,initialize_wire=True)
    traced = jax.make_jaxpr(fn,axis_env=[('core',8)])(
        jax.ShapeDtypeStruct((2,256),jnp.uint32))
    assert tuple(x.shape for x in traced.out_avals) == ((16,256),)


def test_explicit_hbm_wire_has_output_memory_constraint():
    from tpu_beam_search.beam_s5_histogram_exchange import make_s5_histogram_call
    fn = make_s5_histogram_call(SimpleNamespace(size=8),width=256,
        return_wire=True,explicit_hbm_output=True)
    traced = jax.make_jaxpr(fn,axis_env=[('core',8)])(
        jax.ShapeDtypeStruct((2,256),jnp.uint32))
    assert tuple(x.shape for x in traced.out_avals) == ((16,256),)
    call = next(e for e in traced.jaxpr.eqns if e.primitive.name == 'pallas_call')
    assert call.params['out_avals'][0].memory_space == pltpu.HBM
