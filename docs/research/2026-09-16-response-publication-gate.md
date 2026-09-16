# Response routing to publication gate

The current `beam_response_epoch_fixture.py` generates arbitrary 128-byte wire
records. `expected_epoch` checks byte-preserving routing and source-major order.
It does not assign valid destination-local frontier indices at bytes120..123.
Consequently its eleven cases cannot be reused unchanged as a successful
coverage/publication corpus, even if all33 routed epochs are exact.

The next integration corpus must assign each live record a unique index within
its destination before permuting records and bytes together. Compute expected
destination counts from original requests, independently of device grouping.
Encode the index little-endian into the response field; preserve independent
state bytes and deliberately poisoned padding. Keep original source/parent/move
metadata for the corresponding history record; destination is not source.

After each real response epoch, retain actual private output and valid counts.
Accumulate transport errors across every epoch, including empty final epochs.
After the final epoch, run production unpack, whole-depth coverage and
collective error agreement. The good case must cover each destination index
exactly once across all senders and epochs. Separate cases duplicate a target
across epochs, omit one, exceed destination count, and inject a late error.

History must be routed to the same balanced destination and decoded after
completed transfers. Only a successful common decision after response and
history completion permits publication. Failed depths must retain the previous
frontier and history; retry must not observe partial append state. Frontier
padding must be zero before publication. Scratch aliases cannot be reused until
all producers and consumers have finished.

Acceptance requires actual eight-device execution of this composition. Host
oracle and simulated links can validate the fixture and local contracts first,
but cannot establish distributed drains, publication or overlap. This gate is
still smaller than the required full multi-depth GPU/TPU replay and does not
replace it. No new remote session is launched while V4 is queued/running.
