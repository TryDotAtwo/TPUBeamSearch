# Artgor exact inference speed-proof protocol

This protocol is frozen before the dedicated Kaggle run.  It tests one narrow
claim: the accepted `exact_split` full-Q forward is at least 1.5 times faster
than Artgor's immutable original JAX full-Q forward on all eight Kaggle TPU v5e
cores.  It does not claim that a complete beam-search depth or solver is 1.5x
faster.

## Fixed workload

- Artgor Kaggle `scriptVersionId=344319112`; model source hash
  `6d00da89ce45cf84167db20780e30f676cde3ae756d376c8e05a7e0dcf98e46e`.
- `q555_2k_BEST.pt`, BF16, 30 Q outputs per 150-sticker parent state.
- Eight TPU devices, 32,768 states/device and 262,144 states globally.
- Original JAX and exact-split calls receive the same device-resident input and
  their own already-replicated device-resident weights.
- The exact path uses prefix BM4096 and a Pallas head at
  BM256/BK1024/BN128 with late rounding.

## Cases and timing

Three independently generated legal-scramble corpora use seeds 42, 142 and
242.  Three categorical stress corpora use seeds 43, 143 and 243.  Each case
has three warmups followed by 21 paired measurements.  Invocation order
alternates AB/BA, and every returned array is synchronized before the timer
stops.  First compile-and-execute time is recorded separately and excluded
from warmed latency.  Input placement and host transfer are excluded.

The result JSON retains every raw timing pair, invocation order, input and
output hashes, runtime/package/device inventory, source/checkpoint hashes,
median throughput and compilation time.

## Frozen acceptance gate

Every one of the six cases must satisfy all of the following:

1. Original and exact outputs are elementwise and hash identical.
2. Ratio of median warmed latencies is at least 1.5x.
3. Every observed paired latency ratio is at least 1.5x.
4. The deterministic stratified paired-log-ratio bootstrap one-sided 99%
   lower bound is at least 1.5x.

The conservative `min(original) / max(exact)` envelope and a sign-test
diagnostic are reported but are not substituted for any gate.  Criteria will
not be weakened after observing the TPU result.

## Separate beam-search evidence

Component and solver timings remain separate.  Existing completed evidence is:

- exact full-Q inference: 1.588x legal and 1.621x stress;
- three-depth parity benchmark at global beam 1,048,576: 1.103x steady depth;
- matched real pid 1034/frame 0/global beam 16,777,216: original 3,346.59 s,
  exact 2,955.75 s, or 1.132x, with the same verified 116-move result.

The real-solve comparison is across separate Kaggle sessions and has one
matched case, so it is engineering evidence rather than a statistical 1.132x
solver guarantee.
