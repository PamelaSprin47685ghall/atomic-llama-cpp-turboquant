# Scope

This C package composes installed TQCalibration and public A GGUF/GGML APIs. It owns conversion policy, temporary payload storage, tensor metadata, and publication. It does not include private A headers or depend on a sibling C package.

Use A's installed `ggml` package and its public `ggml::ggml-base` target. The pinned package exposes the public header path as `GGML_INCLUDE_DIR`. Do not add a llama or backend link dependency for metadata conversion.

## Contracts

- Preserve expert-major raw sums. Let A normalize by exact sample counts. Reject F32 overflow, nonzero underflow, and inexact counts.
- Keep file parsing in B and export policy in C. A partial-range option must preserve the original counters and every rejection reason it overrides.
- Global-mean rows are estimates. Record their IDs and policy. Never label the replacement count as an observation.
- Use public A metadata emission and verify actual tensor offsets/sizes against the streamed payload.
- Keep memory bounded by one layer plus explicit metadata and conversion buffers. Do not retain all calibration layers for convenience.
- Publish only complete output and never overwrite an existing entry. Make caught I/O errors terminal and clean only owned private files.
- Test actual old-writer interoperability and actual A consumption separately from synthetic format fixtures. Do not infer model quality from those tests.

The supported publication scope is a caller-controlled POSIX directory on a filesystem with hard links. Crash durability, signal-time cleanup, and hostile directory mutation are not provided. Rebuild C against the pinned A revision.
