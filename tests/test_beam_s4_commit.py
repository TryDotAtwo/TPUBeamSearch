import importlib.util
import numpy as np
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu


def test_s4_commit_writes_records_and_inactive_hist_before_releasing_control():
    assert importlib.util.find_spec('tpu_beam_search.beam_s4_commit') is not None
    from tpu_beam_search.beam_s4_commit import pallas_commit_s4
    resident = np.full((8,256),99,np.uint32)
    records = np.arange(8*256,dtype=np.uint32).reshape(8,256)
    a,b = np.full((1,384),11,np.uint32),np.full((1,384),22,np.uint32)
    histogram = np.arange(384,dtype=np.uint32)[None]
    ctrl = np.zeros((4,128),np.uint32)
    ctrl[:,0] = (128,32,1,0)
    count = np.zeros((1,128),np.uint32)
    count[0,0] = 120
    actual = pallas_commit_s4(*map(jnp.asarray,(resident,a,b,ctrl,records,histogram,count)),
        interpret=pltpu.InterpretParams(detect_races=True))
    ctrl[:,0] = (120,0,0,1)
    for got,want in zip(actual,(records,a,histogram,ctrl)):
        np.testing.assert_array_equal(got,want)
