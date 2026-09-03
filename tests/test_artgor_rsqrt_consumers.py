import numpy as np
import jax.numpy as jnp
import pytest


def test_consumer_collector_preserves_device_order_before_host_broadcast():
    from benchmarks.artgor_rsqrt_consumers import collect_consumer
    values = np.arange(16, dtype=np.float32)
    seen = []
    def operation(x):
        seen.append(x.copy())
        return x + 100
    result = collect_consumer(values, operation, devices=2, chunk_rows=2, width=3)
    np.testing.assert_array_equal(seen[0], [0, 1, 8, 9])
    np.testing.assert_array_equal(result, np.repeat((values + 100)[:, None], 3, axis=1))


def test_consumer_collector_keeps_matrix_columns():
    from benchmarks.artgor_rsqrt_consumers import collect_consumer
    values = np.arange(48, dtype=np.float32).reshape(16, 3)
    result = collect_consumer(values, lambda x: x + 2, devices=2, chunk_rows=2, width=3)
    np.testing.assert_array_equal(result, values + 2)


@pytest.mark.parametrize('shape', [(256,), (128, 128)])
@pytest.mark.parametrize('engine', ['jax', 'pallas'])
@pytest.mark.parametrize('arithmetic', ['fp32', 'bf16_expression'])
def test_consumer_preserves_shape_and_uses_epsilon(shape, engine, arithmetic):
    from benchmarks.artgor_rsqrt_consumers import consume_variance
    # sqrt(3+1)=2: omitting epsilon would change every element.
    result = consume_variance(jnp.full(shape, 3, jnp.float32), engine=engine,
                              arithmetic=arithmetic, epsilon=1, interpret=True)
    assert result.shape == shape and result.dtype == jnp.bfloat16
    np.testing.assert_array_equal(result, np.full(shape, 0.5))


@pytest.mark.parametrize('engine', ['jax', 'pallas'])
def test_bf16_expression_is_distinct_from_fp32_consumer(engine):
    from benchmarks.artgor_rsqrt_consumers import consume_variance
    values = jnp.full((128,), 1.004, jnp.float32)
    # A precision witness uses cancellation: FP32 1.004 - 1 > 0,
    # whereas the BF16 expression rounds its input before the addition.
    fp = consume_variance(values, engine=engine, arithmetic='fp32', epsilon=-1, interpret=True)
    bf = consume_variance(values, engine=engine, arithmetic='bf16_expression', epsilon=-1, interpret=True)
    assert np.all(np.asarray(fp) != np.asarray(bf))
