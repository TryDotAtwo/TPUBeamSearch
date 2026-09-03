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

## V2: gather fixes and eligible timing

V1 completed with five exact cases and six gather compile rejections; see
`test_results/beam_primitives_v1/report.md`. V2 changes gather forms without
changing valid candidate semantics. Test `test_beam_gather_lowering.py` catches
the original structural failure; it does not replace physical TPU validation.

After correctness, only exact compiled cases receive timings. Packing uses the
same uint32 data at 65536 candidates/device for serialized/pipelined b2/b3. Three
warmup rounds, 21 synchronized rounds alternating forward/reverse variant order;
retain every sample, median and p10/p90. Exclude compilation and placement.
The other exact cases receive diagnostic primitive latencies in a separate group;
unlike operations must not be described as comparative speedups. No end-to-end
beam throughput claim. All cases still save phase/error individually.

The larger seeded packing allocation changes the subsequent RNG stream, so V2
fixtures differ from V1; input hashes identify that difference. The A/B packing
variants within V2 have identical inputs. Runtime x64 is disabled explicitly.
