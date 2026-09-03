# V5 TPU primitive and Stream3 split gate

Source b469a6863a6ca3472c71022245d7d3ef4f86be65, launcher 1a81667.
Kaggle COMPLETE, all_exact=false. All groups report eight TPU v5 lite devices,
JAX/jaxlib 0.10.2 and libtpu 0.0.42.1. The isolated coordinator attempted all
ten groups / thirteen cases.

## Result

The same five cases are exact: four packing configurations and routing.
Hash120 repeats the native VectorLayout abort; hash150 repeats the unsupported
uint8 shape cast. Hash was intentionally unchanged pending a minimal geometry
probe.

Signed count accumulation moved every dedup case beyond V4's unsupported
uint32 reduction. All four now fail at the following operation:

```
ValueError: Cannot store scalars to VMEM
```

Both new Stream3 split cases (empty and 127 survivors) fail at the same scalar
VMEM store. Therefore neither dedup execution nor split execution is physically
validated yet. The shared failure identifies the next implementation boundary:
logical scalar counts need aligned vector-backed control storage.

## Matched packing control

Same 65536 candidates/device, 3 warmups and 21 alternating synchronized calls:

| Configuration | Median ms | Serial/pipeline ratio |
|---|---:|---:|
| serial b2 | 1.352740 | — |
| pipeline b2 | 0.659560 | 2.051x |
| serial b3 | 1.341420 | — |
| pipeline b3 | 0.633850 | 2.116x |

Routing diagnostic median is 0.515220 ms (p10 0.469320, p90 0.539370).
These are complete primitive calls, not full beam/inference speed or a profiler
proof of DMA overlap.

## V6 change

Dedup count and Stream3 local count become padded uint32 `[1,128]` control
planes. Element `[0,0]` is the logical count and the remaining lanes are zero.
Kernels write the whole plane with vector `where`; no scalar VMEM store remains.
Stream3 peer counts/offsets were already padded planes. This is an explicit TPU
storage adaptation; the logical count and all decisions are unchanged.
