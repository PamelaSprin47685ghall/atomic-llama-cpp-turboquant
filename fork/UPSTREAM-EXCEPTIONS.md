# Exact upstream exceptions

A is the commit recorded in `upstream.json`, plus the explicitly reviewed files
in `upstream-exceptions.json`. B modules stay independent of A; C uses installed
public interfaces. An A exception is coupled work whose continued presence must
be justified. The manifest does not turn that work into an independent B module.

The manifest has exactly this schema; placeholder hashes are not accepted:

```json
{
  "version": 1,
  "upstream_commit": "full upstream commit ID matching upstream.json",
  "files": [{
    "path": "include/llama.h",
    "sha256": "64 lowercase hexadecimal digits of the reviewed file bytes",
    "mode": "100644",
    "consumers": ["fork/batch-compose"],
    "reason": "Describe the public behavior required by the named consumer.",
    "removal_condition": "State the upstream API or migration that removes this exception."
  }]
}
```

Use `100755` only for executable regular files. Paths are repository relative and
canonical. Deletions, symlink overlays, duplicate paths, unused entries, and A
files with `skip-worktree` or `assume-unchanged` are rejected. The visible changed
A path set must equal the manifest set; each live file and any changed staged
version must match its digest and mode. Fork files remain outside this A gate.
Ignored build products are not source overlays; the manifest must not be used as
an automatic enrollment mechanism for whatever happens to be in the worktree.

After a deliberate A change, review the complete changed file, its named
consumers, and the removal condition, then update its digest. Neither the build
script nor isolation runner regenerates or accepts a hash for you. Run
`python3 fork/check-upstream.py` before committing. The Git repository itself and
its configuration are trusted inputs; this guard is a development boundary gate,
not a sandbox for hostile build scripts.

## Claims and executable evidence

- **AXIOM:** the full upstream commit and the manifest entries have been reviewed.
  A digest proves identity with those recorded bytes, not functional correctness.
- **LEMMA:** the gate rejects undeclared changes and mismatched file content or
  modes, including an unreviewed staged version hidden by a reviewed worktree.
  `tests/test-upstream-boundary.py` exercises real temporary Git repositories.
- **SPEC:** default `tests/isolation.py` archives the pinned upstream commit,
  applies only the bytes captured by successful verification, builds A and its
  public seam tests, then builds three B and four C projects independently using
  installed package exports. Intermediate source/build copies and original
  install prefixes are removed before relocated consumers run. Package/help
  checks prove build and loading boundaries; use `--model /absolute/model.gguf`
  to additionally run the real batch/phase comparison and teacher-forced logits
  collection after relocation. The logits check also consumes the installed
  batch program to check the same submitted prompt tokens, final maximum logit,
  and selected token within 1e-5. This independent check does not obtain a second
  full-vocabulary row; repeat tapes and scalar prefix/suffix controls compare all
  vocabulary values.

The current 18 entries comprise 17 upstream code files and one fork CI workflow.
The workflow lives under `.github/workflows/` because GitHub requires that path;
it is fork build infrastructure, not an inference-engine dependency. The Meta
async offset fix adds no API and is justified by independent failing get/set
regressions. Recurrent lifecycle qualification adds tests without changing A.

`--a-prefix` is a convenience for repeated package work. The receipt labels it
`reused_unverified`: the reused A package's source commit, overlay content, and
build provenance have not been established by that run. Its public seam and
consumer tests still run. Only the default archive/build path proves the stated
source construction. Separate source copies are not an operating system sandbox.
