# Scope

This package is a C participant. It owns one model/context run, caller batch arrays, token IDs, and output publication. It calls only installed public llama and GGML APIs. It does not include A private headers or depend on another extension package.

## Contracts

- Keep the TP5LOG2 byte format compatible with the documented comparator. Do not append metadata to an existing format without a version and a reader migration.
- Save logits after consuming the declared teacher-forced token. Sampling must not change the recorded input sequence.
- Bound input, token, vocabulary, batch, and output sizes. Check dimensions before context allocation and check every row for finite values before writing it.
- Synchronize every decode before reusing caller batch arrays, including calls that request no output or report failure.
- Publish a complete new file without replacing any existing entry. The writer owns only its temporary. Keep the existing output intact on every failure.
- Use real model runs to qualify inference behavior. Writer fixtures test the format and filesystem contract only.

## Platform limits

The writer uses POSIX directory descriptors and no-clobber hard links. The caller controls the output directory. Windows, hostile directory replacement, crash durability, and signal-time cleanup are outside the current contract. Keep these limits explicit in the package documentation.
