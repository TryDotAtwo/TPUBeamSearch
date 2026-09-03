# V7 control-plane select removal gate

Source `2eb6aa7b40c9b56456c4bfc1627504904b4dc851`, launcher `6af0a1a`.
Kaggle completed all ten isolated subprocess groups on eight TPU v5 lite
devices with JAX/jaxlib 0.10.2 and libtpu 0.0.42.1. `COMPLETE` is not a
correctness pass: the bundle has `all_exact=false`.

Five controls are exact: the four matched packing variants and routing. The
pipeline packing medians are 0.629210 ms (two buffers) and 0.604160 ms (three
buffers), versus 1.315729 ms and 1.315659 ms for the corresponding serial
calls. These are isolated primitive timings only.

The V7 hypothesis is falsified. Replacing boolean selection only in the padded
count/offset control planes did not move any of the four dedup or two split
cases past the V6 boundary. Every one still fails while compiling at
`select_n/select_n`, with the same invalid i1-to-i8 vector layout extension.
Therefore the failing selector is in the shared survivor data path, not proven
to be in count-plane construction. A smaller selector probe is required before
another production edit.

The two known hash failures are unchanged: `hash_goal_120_24` aborts its
isolated native compiler subprocess in `VectorLayout::join`, and
`hash_goal_150_30` rejects the uint8 `vector<8x160xi8>` to
`vector<1280xi8>` reshape. Neither is accepted.

Raw aggregate JSON, nested per-group JSON/process logs, successful-control HLO,
and the complete Kaggle log are retained in this directory. Cross-process
latencies are not matched A/B measurements. No packing result is extrapolated
to inference or whole-beam throughput.
