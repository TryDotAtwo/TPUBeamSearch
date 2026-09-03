# Beam primitive V3: native compiler abort

Source: e9c90a9ef3f132fce651218f20b62707d741e21e.
Kaggle status: ERROR. The first five cases (four packing configurations and
routing) are exact on eight TPU devices. Hash/goal 120/24 then aborts the native
compiler in VectorLayout::join / inferElementwise:

```
layout.h:341 Check failed: arr.size() >= layout_rank(implicit_dim) (1 vs. 2)
```

The subprocess exits by SIGABRT, not a catchable Python compile exception.
Consequently hash/goal 150/30 and all four dedup cases were not attempted.
There are no V3 steady-state timings. A partial JSON `phase=placement` was stale:
the harness changed phase to compile without flushing it before compilation.

The stack establishes a compiler vector-layout failure, but does not identify
the specific source expression responsible. In particular it is not evidence
of incorrect search semantics, nor proof that V3's dedup fix fails.

Next diagnostic keeps production primitives unchanged and runs eight groups
in sequential subprocesses, never concurrent TPU clients. Packing remains one
four-variant interleaved group. Each other primitive is isolated. Raw process
logs, return codes, and partial reports survive a native abort. Compile phase
is flushed before lowering. Cross-process timings are diagnostic, not matched
A/B comparisons. Kernel COMPLETE still does not imply all_exact.
