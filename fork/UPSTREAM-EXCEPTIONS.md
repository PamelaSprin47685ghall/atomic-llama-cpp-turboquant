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
