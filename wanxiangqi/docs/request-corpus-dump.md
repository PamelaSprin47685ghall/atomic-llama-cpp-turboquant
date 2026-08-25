# Request token dump (`--run-dump`)

`llama-server --run-dump DIR` records the canonical token IDs of every model
request while the server continues serving normally. Chat requests are dumped
after chat-template expansion and tokenization, so the corpus is the same token
stream the model receives rather than the original HTTP JSON.

For multimodal requests, only real text tokens are stored. Internal
`LLAMA_TOKEN_NULL` positions used to reserve media cells are omitted.

The in-memory representation is a prefix trie. On disk, the trie uses a parent
representation and fixed-size little-endian binary records split into shards.
Node `0` is an implicit root and is never written.

## Files

- `format.json`: machine-readable format summary.
- `nodes-000000.bin`, `nodes-000001.bin`, ...: trie node shards.
- `requests-000000.bin`, `requests-000001.bin`, ...: one terminal/leaf record
  for every request sample, including duplicate requests.

Each binary shard starts with a 24-byte header:

| Offset | Type | Meaning |
| --- | --- | --- |
| 0 | 8 bytes | magic (`CLTNOD01` or `CLTREQ01`) |
| 8 | `u32` | format version (`1`) |
| 12 | `u32` | record size |
| 16 | `u64` | first global node/request ID in this shard |

Node records are 16 bytes:

| Offset | Type | Meaning |
| --- | --- | --- |
| 0 | `u64` | parent node ID |
| 8 | `i32` | token ID |
| 12 | `u32` | reserved, currently zero |

Node IDs are implicit: `shard.start_id + record_index`. Reconstruct a request
by following `parent_id` from its leaf back to node `0`, collecting token IDs,
then reversing the sequence.

Request records are 24 bytes:

| Offset | Type | Meaning |
| --- | --- | --- |
| 0 | `u64` | terminal/leaf node ID |
| 8 | `u32` | token count |
| 12 | `u32` | request kind: `1=completion`, `2=infill`, `3=embedding`, `4=rerank` |
| 16 | `u64` | Unix timestamp in milliseconds |

The default shard limit is 1,000,000 records per file and can be changed with
`--request-token-dump-shard N`. Existing shards are loaded on startup to rebuild
the in-memory trie, then new requests continue appending to the same corpus.

Use a separate dump directory for each tokenizer/model vocabulary, and do not
point multiple `llama-server` processes at the same directory concurrently.
The on-disk corpus stores token IDs, not tokenizer metadata or an inter-process
write lock.

The persistence order is nodes first, request leaf second. Therefore an
interrupted write can at worst leave unreachable/orphan trie nodes; a committed
request record never intentionally points at a node that was not flushed first.
On restart, an incomplete final record in the last node/request shard is
truncated automatically; partial records in older shards are treated as corpus
corruption and abort dump initialization.
