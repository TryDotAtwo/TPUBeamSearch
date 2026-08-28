import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from tpu_beam_search.sharding import make_sharded_inference


def test_sharded_inference_accepts_outputs_without_manual_vma_metadata():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("core",))
    weights = (jnp.array([10, 20], dtype=jnp.int32),)
    compiled = make_sharded_inference(
        lambda states, dynamic_weights: states + dynamic_weights[0],
        mesh=mesh,
        weights_example=weights,
    )
    states = jax.device_put(
        jnp.array([[1, 2], [3, 4]], dtype=jnp.int32),
        NamedSharding(mesh, P("core", None)),
    )
    replicated_weights = jax.tree.map(
        lambda value: jax.device_put(value, NamedSharding(mesh, P())), weights
    )

    actual = compiled(states, replicated_weights)

    np.testing.assert_array_equal(actual, [[11, 22], [13, 24]])
