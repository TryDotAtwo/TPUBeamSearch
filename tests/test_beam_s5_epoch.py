from types import SimpleNamespace
import numpy as np
import jax.numpy as jnp
import jax
from jax.experimental.pallas import tpu as pltpu


def test_serialized_epoch_no_request_then_forced_publication():
    from tpu_beam_search.beam_s5_epoch import make_s5_epoch_call
    fn = make_s5_epoch_call(SimpleNamespace(size=1),bins=128,period=3,
        interpret=pltpu.InterpretParams(detect_races=True))
    hist = np.zeros((2,128),np.uint32)
    hist[0,2],hist[1,5] = 2,4
    hist_b = np.zeros_like(hist)
    active_hist = np.zeros((1,128),np.uint32)
    a,b = np.zeros((2,128),np.uint32),np.zeros((2,128),np.uint32)
    active = np.zeros((1,128),np.uint32)
    beam = np.zeros((2,128),np.uint32)
    beam[0,0] = 3
    state = np.zeros((4,128),np.uint32)
    state[0,0] = 2
    args = tuple(map(jnp.asarray,(hist,hist_b,active_hist,a,b,active,beam,state)))
    unchanged = fn(*args,jnp.array([0],jnp.uint32))
    for got,want in zip(unchanged,(a,b,active,state),strict=True):
        np.testing.assert_array_equal(got,want)
    updated = fn(*args,jnp.array([1],jnp.uint32))
    assert int(updated[2][0,0]) == 1
    np.testing.assert_array_equal(np.asarray(updated[1])[:,0],[5,1])
    np.testing.assert_array_equal(np.asarray(updated[3])[:,0],[0,1,0,0])
    hist[0,2] = 3
    again = fn(jnp.asarray(hist),jnp.asarray(hist_b),jnp.asarray(active_hist),
        *updated[:3],jnp.asarray(beam),updated[3],jnp.array([1],jnp.uint32))
    assert int(again[2][0,0]) == 0
    np.testing.assert_array_equal(np.asarray(again[0])[:,0],[2,1])
    np.testing.assert_array_equal(np.asarray(again[3])[:,0],[0,2,0,0])


def test_epoch_reads_selected_histogram_version_and_never_relaxes_threshold():
    from tpu_beam_search.beam_s5_epoch import make_s5_epoch_call
    fn = make_s5_epoch_call(SimpleNamespace(size=1),bins=128,period=1,
        interpret=pltpu.InterpretParams(detect_races=True))
    hist_a,hist_b = np.zeros((2,128),np.uint32),np.zeros((2,128),np.uint32)
    # Combining both selected shards requires a low-word carry for beam2**32.
    hist_a[:,20] = 0xffffffff
    hist_b[:,3] = 0xffffffff
    versions = np.zeros((1,128),np.uint32)
    beam = np.zeros((2,128),np.uint32)
    beam[1,0] = 1
    slots = (jnp.zeros((2,128),jnp.uint32),jnp.zeros((2,128),jnp.uint32),
             jnp.zeros((1,128),jnp.uint32))
    state = jnp.zeros((4,128),jnp.uint32)
    for epoch,(selected,expected) in enumerate(((0,20),(1,3),(0,3))):
        versions[0,:2] = selected
        result = fn(jnp.asarray(hist_a),jnp.asarray(hist_b),jnp.asarray(versions),
            *slots,jnp.asarray(beam),state,jnp.array([1],jnp.uint32))
        slots,state = result[:3],result[3]
        active = int(slots[2][0,0])
        np.testing.assert_array_equal(np.asarray(slots[active])[:,0],[expected,1])
        assert int(state[1,0]) == epoch+1


def test_eight_rank_epoch_traces_one_conditional_and_expected_outputs():
    from tpu_beam_search.beam_s5_epoch import make_s5_epoch_call
    fn = make_s5_epoch_call(SimpleNamespace(size=8),bins=128,period=3)
    shapes = ((2,128),(2,128),(1,128),(2,128),(2,128),(1,128),(2,128),(4,128),(1,))
    traced = jax.make_jaxpr(fn,axis_env=[('core',8)])(
        *(jax.ShapeDtypeStruct(shape,jnp.uint32) for shape in shapes))
    assert tuple(x.shape for x in traced.out_avals) == ((2,128),(2,128),(1,128),(4,128))
    assert sum(e.primitive.name == 'cond' for e in traced.jaxpr.eqns) == 1
def test_epoch_forwards_explicit_hbm_constraint(monkeypatch):
    from types import SimpleNamespace
    import tpu_beam_search.beam_s5_epoch as module
    captured=[]
    original=module.make_s5_histogram_call
    def capture(*args,**kwargs):
        captured.append(kwargs)
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'make_s5_histogram_call',capture)
    module.make_s5_epoch_call(SimpleNamespace(size=1),bins=128,period=3,
        explicit_hbm_output=True)
    assert captured[0]['explicit_hbm_output'] is True
