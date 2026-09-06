# S5 remote/local HLO difference to investigate

Direct comparison of published V3 `wire/s5_wire.hlo.txt` and V4
`replicate/s5_replicate.hlo.txt` shows a concrete lowering difference:

- Both calls accept `u32[2,256]{1,0:T(2,128)}` and produce `[16,256]`.
- V3 communicating custom-call output is
  `u32[16,256]{1,0:T(8,128)S(1)}`. Its bitcast retains `S(1)` and the entry
  root is a copy to the default memory-space layout.
- V4 local custom-call output is `u32[16,256]{1,0:T(8,128)}`. The root is
  only a bitcast, without that extra copy.
- V3 backend config has `has_communication:true`; V4 does not.

This is an observed compiled-HLO distinction, not yet a causal diagnosis.
Do not infer which physical memory is used solely from the printed color,
or claim the root copy is faulty. Both source calls request HBM BlockSpecs.
The initialized-remote diagnostic must save HLO and check whether this
difference persists alongside actual output correctness. If necessary, a
later output-alias/control experiment should isolate memory placement from
the transfer protocol, with explicit lifetimes and immutable input.

Readiness reasoning for initialization: each rank finishes every local store
before signaling readiness to its sender for that offset. A sender waits for
the receiver's signal before writing the receiver's slot. Thus intended
protocol ordering prevents a late local initialization from overwriting the
incoming transfer; physical execution remains to be validated.
