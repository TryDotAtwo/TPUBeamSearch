# Numerical and consumer contracts

Checked: 2026-08-31. Use when replacing operators, changing fusion/precision,
or deciding whether outputs remain suitable for their consumer.

## Separate boundary error from candidate error

Compare monolithic JAX, separately jitted JAX blocks, and the exact compiled
JAX suffix used after a replaced prefix. Compare the hybrid against that
same-suffix control, including a zero-replacement path. JIT transformations
can change floating-point outputs: [JAX numerical FAQ](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs).

Complete CPU boundary-control example; requires JAX and NumPy. It makes no
TPU-performance claim and need not produce nonzero differences:

```python
import jax
import jax.numpy as jnp
import numpy as np

def block(x, w, b):
    return jax.nn.relu(x @ w + b)

step = jax.jit(block)
suffix = jax.jit(lambda h, w, b: block(block(h, w, b), w, b))
whole = jax.jit(lambda x, w, b:
                block(block(block(x, w, b), w, b), w, b))

with jax.default_device(jax.devices("cpu")[0]):
    rng = np.random.default_rng(4)
    x, w, b = [jnp.asarray(rng.normal(size=s), jnp.bfloat16)
               for s in [(4, 8), (8, 8), (8,)]]
    h = step(x, w, b)
    segmented = step(step(h, w, b), w, b)
    control = suffix(h, w, b)
    monolithic = whole(x, w, b)
    hybrid_zero = suffix(h, w, b)  # no replaced prefix yet
    for name, a, ref in [("boundary", control, segmented),
                         ("whole", control, monolithic),
                         ("zero", hybrid_zero, control)]:
        a, ref = [np.asarray(t.block_until_ready(), np.float32)
                  for t in (a, ref)]
        assert np.isfinite(a).all() and np.isfinite(ref).all()
        print(name, np.max(np.abs(a - ref)), np.array_equal(a, ref))
```

For attribution, cross JAX/Pallas Dense and LayerNorm independently. BF16
dot-result rounding before bias and FP32 accumulated bias before BF16 casting
are different expression contracts. The project's CPU witness does not prove
either caused the measured TPU discrepancy; target A/B and lowered-code
inspection remain necessary. See the [pinned audit](https://github.com/TryDotAtwo/TPUBeamSearch/blob/f17eedff869f2cb23535c99b63ae024c6aa602cc/docs/research/2026-08-31-tpu-coding-research.md).

## Validate the actual decision

Use the consumer's min/max direction, action order, inverse-move and validity
masks. Row argmin/argmax alone does not establish flattened global top-K
agreement. Compare selected identities, overlap/order, ties, best/second-best
and K/K+1 margins, invalid-slot leakage and valid count. When fewer than K
valid candidates exist, preserve that count; sentinel-filled slots remain
invalid. Choose task-specific acceptance and downstream replay checks.

Test reachable legal scrambles/recorded frontiers separately from random
categorical stress inputs; valid dtype/range does not prove reachability.
Track finite values, max/mean absolute error, RMSE and task metrics. Matching
aggregates against an oracle do not establish pairwise equality: compare the
two tensors directly. See [case studies](case-studies.md) for evidence scope.
