# Scope

This B package reads WXQCAL03 files with C++17 and the standard library. Do not introduce an A dependency, a model name, GGUF policy, or a callback to another service.

## Ownership and validation

- The caller owns the input stream and must keep it alive. The reader owns its parse position and failure state. Return owned layer values.
- Check byte arithmetic, dimensions, and budgets before allocating arrays. Check skipped covariance data as real input, including its final bytes.
- Keep routing, activation, and hidden sample populations distinct. Do not equate independently sampled counters.
- Preserve progress fields. Completeness and imputation decisions belong to the consumer.
- Make reader failures terminal. Do not expose a partially parsed layer.
- Use format and ownership tests without A. Qualify production-writer compatibility separately with the unchanged writer, and label fixture data honestly.
