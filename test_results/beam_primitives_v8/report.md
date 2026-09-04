# V8 bitonic predicate fix gate

Source `f8deda43ba8f4e4ed2558de16ee8c90536b7cd32`, launcher `87f408c`.
All ten isolated groups / thirteen cases were attempted on eight TPU v5 lite
devices with JAX/jaxlib 0.10.2 and libtpu 0.0.42.1. The bundle is not all-exact
because both known hash failures and four dedup compile failures remain.

The predicate fix is physically effective. Both complete Stream3 split cases
(`count=0` and `count=127`) now compile and execute exactly, including local and
remote metadata, counts and offsets. This is the first physical exact result
for the bounded split path; it is not remote DMA or HBM-scale evidence.

Dedup advances past the V7 `select_n` boundary. At capacity 128, both modes now
fail at the unsigned `maximum(indices, 1)` used to address the previous hash:
Mosaic cannot legalize `arith.maxui` for the selected vector layout. At capacity
256, both modes fail earlier in the bitonic partner gather because Mosaic does
not implement multiple source vregs along the gather dimension. These are two
distinct next boundaries and require separate probes/implementations.

Five packing/routing controls remain exact. Hash120 retains its isolated native
`VectorLayout::join` abort and hash150 retains the unsupported uint8 reshape.
No isolated timing is extrapolated to whole beam or inference.
