import hashlib
import numpy as np


def test_bounded_gate_counts_bitwise_mismatches_and_hashes_across_chunks():
    import benchmarks.artgor_prefix_gate as gate
    reference = np.zeros((5,2), np.float32)
    candidate = reference.copy()
    candidate[3,1] = -0.
    candidate[4,0] = 1.
    report = gate.compare_prefix(reference, candidate, chunk_rows=2)
    assert report['mismatch_count'] == 2
    assert report['numeric_mismatch_count'] == 1
    assert report['signed_zero_mismatch_count'] == 1
    assert report['first_mismatch'] == [3,1]
    assert report['reference_sha256'] == hashlib.sha256(reference.tobytes()).hexdigest()
    assert report['candidate_sha256'] == hashlib.sha256(candidate.tobytes()).hexdigest()
    assert report['finite'] and not report['exact']
    assert gate.compare_prefix(reference, reference, chunk_rows=2)['exact']


def test_gate_rejects_equal_nonfinite_tensors():
    import benchmarks.artgor_prefix_gate as gate
    values = np.full((2,2), np.inf, np.float32)
    report = gate.compare_prefix(values, values)
    assert report['mismatch_count'] == 0
    assert not report['finite'] and not report['exact']


def test_composed_prefix_uses_state_embedding_and_projects_correct_channels():
    import jax.numpy as jnp
    import benchmarks.artgor_prefix_gate as gate
    params = dict(encoding='embedding', state_size=1, num_classes=2,
                  embed=jnp.array([[0.],[1.]]), head_w=jnp.zeros((128,30)), head_b=jnp.zeros(30),
                  input_stack=[dict(lin_w=jnp.array([[1.]*64+[-1.]*64]), lin_b=jnp.zeros(128),
                                    ln_gamma=jnp.ones(128), ln_beta=jnp.zeros(128))], res_blocks=[])
    architecture = gate.Stream1Architecture.from_artgor_params(params, STATE_STORAGE_LEN=1)
    weights = gate.layernorm_stream1_weights_from_artgor_params(params, architecture)
    packed = gate.prepare_pallas_exact_weights(weights, architecture)
    states = jnp.array([[0]]*64+[[1]]*64, jnp.uint8)
    expected = np.zeros((128,128), np.float32)
    expected[64:,:64] = 1.
    for order in gate.ORDERS:
        actual = gate.pallas_prefix(states, packed, architecture, order=order, interpret=True)
        np.testing.assert_array_equal(np.asarray(actual,np.float32),expected)
