# Eight-device beam primitive compile gate

Scope: compile and exact integer-output checks on eight actual TPU devices.
Not a whole-beam comparison and not a speed or overlap proof.

The bundle `benchmarks.beam_primitive_bundle` runs 11 independent cases:
four SoA pack configurations (serialized/pipelined, 2/3 input buffers), one
Hash128 owner/shard routing case, immediate hash/goal for 120/24 and 150/30,
and Stream3/4 diagnostic dedup at capacities 128/256. Replicas use identical
fixtures intentionally; every device output is compared, but these cases do not
exercise communication or distributed candidate selection.

Seed 9341; inputs and outputs have SHA256. Expected integer arrays are generated
with NumPy/Python independently of Pallas kernels. Source C++ differential tests
remain a separate local gate. Failure in any single case is recorded with phase
and traceback without suppressing later cases. JSON is saved before and after
each case; successful compiled HLO is saved. Runtime versions and all eight
device identities are recorded before compilation.

Require exactly eight physical TPU devices. Do not reinterpret on CPU if the
hardware gate fails. There is deliberately no timing-based winner selection.
Keep one active Kaggle TPU session; prior prefix V10 is COMPLETE before launch.

Next: fix any actual lowering failures with reproductions and run a separately
validated remote-DMA ring probe before integrating the beam scheduler.
