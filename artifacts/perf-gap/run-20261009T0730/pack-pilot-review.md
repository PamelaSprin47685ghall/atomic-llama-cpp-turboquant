# First pack pilot: not qualified

The first candidate pilot completed normally, A66.5733/B68.0023 tok/s, but A's
measured body changed from the established176-token text to158 tokens. B retained
the176-token body. No matched-trajectory or new speed qualification is claimed.

The first implementation placed the new single-workgroup loop behind a runtime
branch in the shared pack shader. Even with legacy400 groups, that changed the
baseline shader compilation. Numerical byte fixtures on the single-workgroup path
passed, but that does not prove why A changed. Do not declare a unique pack race,
buffer-lifetime or floating-point cause from text alone.

Revision: generate a separate TP5_LATE_PACK_SINGLE SPIR-V; the default pack source
body is restored byte-for-byte at preprocessing, and only opt-in selects the new
module plus one workgroup. Repeat the shader bytes tests and model confirmation
with the isolated revision. The earlier pilot and its output hashes stay retained.
