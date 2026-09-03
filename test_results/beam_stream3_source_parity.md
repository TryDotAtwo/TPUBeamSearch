# Stream3 original-source differential gate

2026-09-03. Adapter links the original `D:/100XH100/src/stream3.cpp` read-only.
SHA256: `584d37e0c96d581174359932a2fe04df76cbfdfca12b44b0a5df75c3aec1d2b5`.
No changes to that checkout or to production TPU primitives.

## Verification

Before implementation the new Stream3 test failed: adapter returned code 3
instead of a result because the Stream3 protocol did not exist. After adding
the adapter and rebuilding against the original source:

```
cmd /c tests\build_beam_source_oracle.cmd
$env:BEAM_SOURCE_ORACLE = (Resolve-Path .local/beam_source_oracle.exe).Path
python -m pytest tests/test_beam_source_parity.py -q
5 passed in 18.57s
```

This targeted run includes previous hash/goal and Stream4 tests plus three
Stream3 fixtures. The full repository suite was not rerun for this test-only
change.

- 127 inputs / capacity 128, eight owners, local rank 3, threshold 3.
- Same records, one owner, maximum uint32 threshold (padding must remain invalid).
- Empty input, eight owners, threshold zero.

Inputs include Hash128 high words, parents above 2^48, distinct move IDs and
descending original payload IDs. One deliberate equal-score duplicate pair
must choose parent 2^48+1, not the smaller parent. Compare all eight metadata
words, exact local/remote order, per-peer counts and prefix offsets.

## Evidence boundary

The C++ oracle executes on CPU. Pallas dedup/routing uses CPU interpretation.
The initial test partitions survivors on the host to match the source split
contract. It is now extended to compare the bounded Pallas split directly with
the same C++ local/remote records, counts and offsets. Both Pallas paths still
use CPU interpretation. Therefore this establishes a reusable original-source
oracle and local primitive semantic parity, NOT CUDA execution, physical TPU
compilation, distributed exchange or throughput.

V4 remains the independent physical compile gate and is not restarted by this
change. Its pinned source does not include this test-only extension.
