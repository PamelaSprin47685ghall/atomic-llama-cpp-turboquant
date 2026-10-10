# Counter memory-domain review

The first native counter utility retained an old allocation heuristic choosing
the last non-host-visible DEVICE_LOCAL type. That can select AMD device-coherent
uncached types instead of the backend's first ordinary device-local VRAM type.
Its memory flags were not printed. Therefore `traffic-v2/` values are retained
but are NOT admitted as current production VRAM-traffic evidence.

The next counter run explicitly excludes DEVICE_COHERENT/DEVICE_UNCACHED AMD
types, prints all types and selected flags, and uses the first ordinary pure-VRAM
type for model buffers. No model backend or driver property is changed. Results
go to `traffic-cached/`. Also check exact dispatched wave count against the
driver-derived Waves metric; discrepancy must not be silently corrected by an
assumed scale factor. Byte counters have a separate scope and need their own
interpretation. This correction tightens evidence eligibility, not a tolerance.
