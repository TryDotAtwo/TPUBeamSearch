import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from tpu_beam_search.beam_transport import pallas_pack_candidates


@pytest.mark.parametrize('buffers', [2, 3])
@pytest.mark.parametrize('pipelined', [False, True])
def test_pallas_pack_preserves_all_bits_across_pipeline_wrap(buffers, pipelined):
    words = (np.arange(8 * 640, dtype=np.uint32).reshape(8, 640) + np.uint32(0x80000000))
    fields = [jnp.asarray(words[:4]), jnp.asarray(words[4:6]),
              jnp.asarray(words[6:7]), jnp.asarray(words[7:8])]
    # Supply simulated target geometry; this is NOT execution on a TPU.
    mesh = jax.sharding.AbstractMesh((), (), abstract_device=jax.sharding.AbstractDevice(
        device_kind='TPU v5 lite', num_cores=1, platform='tpu'))
    with jax.sharding.use_abstract_mesh(mesh):
        result = pallas_pack_candidates(*fields, buffer_count=buffers,
                                       pipelined=pipelined,
                                       interpret=pltpu.InterpretParams(detect_races=True))
    np.testing.assert_array_equal(result, words)


def test_pack_rejects_invalid_geometry_and_dtype():
    fields = [jnp.zeros((rows, 256), jnp.uint32) for rows in (4, 2, 1, 1)]
    for kwargs in ({'tile_candidates': 0}, {'tile_candidates': 129}, {'buffer_count': 1}):
        with pytest.raises(ValueError):
            pallas_pack_candidates(*fields, **kwargs, interpret=True)
    with pytest.raises(ValueError):
        pallas_pack_candidates(fields[0].astype(jnp.float32), *fields[1:], interpret=True)
    with pytest.raises(ValueError):
        pallas_pack_candidates(fields[0][:, :128], *fields[1:], interpret=True)
