# TPU access applications — 2026-09-05

Status: TRC application submitted on 2026-09-05. The Google Forms confirmation
page displayed "Ответ записан." A copy of the answers was requested by email.
Do not resubmit: the form considers only one submission per person.
TPU Builders was also submitted on 2026-09-05; its confirmation page displayed
"Ответ записан." Neither submission constitutes acceptance or TPU access.

## Programs and blockers

- TPU Research Cloud: https://sites.research.google/trc/about/
  Official application: https://docs.google.com/forms/d/e/1FAIpQLSctqU8r5h9FNbw0eBd90OlnQLDbS4uTZbf2oaupZHsKnhwYYg/viewform
  Submitted through the applicant's authorized Google session, with Israel
  and independent developer affiliation. Required terms, AI principles and
  privacy acknowledgements were accepted after explicit user confirmation.
  The form says TRC is oversubscribed and this is interest in future placement;
  submission does not establish an allocation or approval.
  FAQ: https://sites.research.google/trc/faq/
  TPU allocation is temporary; ancillary GCP services can incur charges.
- TPU Builders: public program invitation hosted by Kyoto University:
  https://www.med.kyoto-u.ac.jp/wp/wp-content/uploads/2026/04/2026-Google-TPU-Builder-_-AI-GDE-Program.pdf
  Offers TPU/cloud resources, engineering contact and community, with public
  technical contributions expected. The live form explicitly includes
  developers and engineers building open-source tools, not only academics.
  The short URL returned HTTP 403; the direct URL was extracted from the PDF:
  https://docs.google.com/forms/d/e/1FAIpQLScll-hW9u_cjnmNu02Bk_Ja2sQGOMuTijNrN3TMZgH5iVSJAA/viewform
  Submitted with JAX, active Kaggle TPU use, open-source contributions and
  interest in the Builders Program. No Google referral was claimed; optional
  mailing-list consent was declined. The application discloses the TRC
  submission and asks to coordinate allocations to avoid duplication.
- Google Cloud research credits: https://edu.google.com/intl/ALL_us/programs/credits/research/
  Academic affiliation/role eligibility must be checked before applying.
  Do not represent an independent project as an institution or startup.

## Shared proposal draft

Title: TPUBeamSearch: reproducible Pallas inference and distributed neural-guided search

Public repository: https://github.com/TryDotAtwo/TPUBeamSearch

We develop an open-source JAX/Pallas implementation of neural-guided beam
search for combinatorial puzzles, focusing on numerical reproducibility,
TPU memory layouts and device-to-device communication. Our current work
combines residual MLP inference with discrete operations including sorting,
deduplication, routing and bounded remote-DMA rings.

The repository documents an exact hybrid inference path for the Artgor-family
embedding/LayerNorm ResMLP, with a measured 1.58–1.63x inference speedup over
the original model on eight Kaggle TPU v5 lite devices at local batches 16K
and 32K. This is an inference result, not a claim of end-to-end beam speedup.
See test_results/kaggle_final_residual_ab_v1/report.md. The project also
publishes compiler/layout diagnostics, correctness tests and pinned launchers.

We request reliable TPU access to reduce delays between compiler experiments
and physical-device validation. The next objectives are scalable HBM sorting
and cross-tile deduplication, integration of distributed search stages, and
multi-depth replay comparisons against the source GPU architecture. We will
publish code, reproducible measurements, useful HLO/profile artifacts and
technical reports, subject to third-party licenses and removal of credentials.

Proposed initial resource request (draft, not an existing grant): one small
multi-device TPU allocation supporting eight-way execution for a 30-day pilot,
with approximately 100 active allocation-hours. Adapt the topology and units
to the hardware offered; eight devices on one TPU generation are not assumed
equivalent to eight chips on another. Release idle capacity. Larger allocations
would be requested only after validating the small configuration.

Deliverables: correctness-gated primitive benchmarks; an integrated replay
harness; measured inference and search-stage performance; documentation of
layout/compiler issues with minimal reproductions. End-to-end performance
improvement remains a research objective.

## Applicant information required

The user supplied contact email, Israel, independent developer affiliation,
and authorized their Google account. Personal email and response-edit links
are deliberately omitted from this public project record. No billing was
activated and no cloud resources were provisioned by these submissions.
