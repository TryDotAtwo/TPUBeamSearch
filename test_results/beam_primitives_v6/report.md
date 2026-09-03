# V6 vector-backed control plane gate

Source 9f2d083aa51bfe240ebaae3243d0fff20112dbc9, launcher 9572eeb.
Kaggle COMPLETE, all_exact=false. All ten groups / thirteen cases were attempted
on eight TPU v5 lite devices with JAX/jaxlib 0.10.2 and libtpu 0.0.42.1.

Five controls remain exact: four pack configurations and routing. Hash120 and
hash150 retain their explicitly unresolved layout failures.

V6 removes scalar VMEM stores. All four dedup and both split cases advance to a
new common compiler error:

```
Mosaic failed to compile TPU kernel: Invalid input layout
at select_n/select_n
arith.extsi vector<128xi1> -> vector<128xi8>
```

This is evidence that vector-backed output storage removed the V5 boundary; it
does not validate dedup/split execution. The only newly introduced common
operation is boolean-select construction of the padded control plane. Split has
additional select operations for counts/offsets. V7 tests this single hypothesis:
construct all control planes through uint32 indicator multiplication/addition,
with no select_n output of shape [1,128]. Sorting/data selection is unchanged.

The full local V6 suite passed 559 tests before launch. Raw JSON, individual
process logs and all generated HLO for successful controls are preserved.
Packing measurements are repeated controls and do not establish beam/inference
speed or full overlap.
