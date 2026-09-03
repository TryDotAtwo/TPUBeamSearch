# V10: reduction geometry isolates the remaining prefix discrepancy

Kaggle `trydotatwo/tpu-artgor-pallas-prefix-gate` is COMPLETE. Source:
`ded58790b445f3f6bbf00998d4f45089819466ce`. Runtime JAX/jaxlib 0.10.2,
libtpu 0.0.42.1; eight TPU v5 lite devices. Compiled geometry HLO explicitly
contains `num_partitions=8`; generic context `active_device_count=1` is not used
as an execution/utilization denominator.

The eight cases reuse BOTH validated V4 large Dense and scalar mean buffers:
Dense SHA 1d48026ba86efbdf1bd0be5bdd124159189052bb36f99dd61893e7c97d2ebec3;
mean SHA 862a6cea2c157079bac2981dd9ca0dc2f823ae3be822fad28de300b1af1e1ff0.
Legal seed 42, 131072 states. The output remains a diagnostic prefix, not full Q.

| Reduction input | FP32/original expression | Rows per device | invstd mismatches vs V4 | Prefix mismatches vs large oracle |
|---|---|---|---:|---:|
| `[B,1024]`, reduce axis 1 | both | 16384 and 256 | 2048 | 1275 |
| `[1024,B]`, reduce axis 0 | both | 16384 and 256 | 0 | 0 |

All eight cases complete and finite. Within each geometry/arithmetic variant,
changing chunk size with identical buffers yields identical invstd. Transposed
cases reproduce reference invstd SHA
`5e045ff909b46ae76a23403ff09ca88ed876113b7448b24bba1b565fc73119b8`
and prefix SHA
`9755606bffa3d179337f5741fcd23dce5f0469d6b11ebc56c546f6e25b6cd7f0`.
Non-transposed cases instead reproduce the native Pallas invstd result.

Compiled HLO confirms distinct matrix geometries and reduction axes:
`bf16[16384,1024]{1,0:T(8,128)(2,1)}` / dimensions={1} versus
`bf16[1024,16384]{1,0:T(8,128)(2,1)}` / dimensions={0}.
This is evidence that reduction geometry can account for the observed difference
on these fixed inputs. It does not identify an exact machine-level reduction
tree or establish a general arithmetic equivalence beyond this corpus.

Both large/small attribution controls are valid and reproduce V4 mean/invstd.
The additional diagnostic variance capture still changes output and is invalid;
do not use its variance as an attribution oracle.

Important limitation: the successful producer is JAX. No all-Pallas completion
or speed claim follows. Preserve this evidence for S1 integration without
restarting standalone inference sweeps: current priority is the entire beam port.
Raw JSON, HLO and downloaded arrays are kept under artgor_reduction_geometry/.
Download completed successfully: 332 output files plus this report, about 9.8 MB;
all 165 NPZ archives pass CRC validation. Both benchmark and Kaggle logs retained.
