# TQCorpus

TQCorpus stores request token sequences in a shared prefix trie. Each request
keeps its own record, including duplicate requests. It requires C++17 and the
standard library. It has no llama.cpp, ggml, model, or GPU dependency.

The consumer supplies token ids and a vocabulary size. The size checks the token
range. It does not prove that a corpus and a model use the same tokenizer. The
consumer must preserve and compare model or tokenizer provenance.

## Build and install

    cmake -S fork/corpus -B build-corpus -DBUILD_TESTING=ON
    cmake --build build-corpus
    ctest --test-dir build-corpus --output-on-failure
    cmake --install build-corpus --prefix /path/to/prefix

A separate consumer uses find_package(TQCorpus CONFIG REQUIRED) and links
TQCorpus::corpus. Include <tq/corpus/trie.h>.

## Contract

Construct tq::corpus::token_trie with a positive vocabulary size. append accepts
a nonempty vector of valid int32 token ids and returns the request leaf id.
Repeated prefixes share nodes. Repeated requests keep separate request records.

Node ids start at 1. Node 0 is the implicit root and has no token. node(0) throws;
children(0) returns root children and path_to(0) returns an empty vector. Each
query returns an owned value. Copies and separate instances do not share state.

node_count counts stored token nodes. request_count counts request records.
leaf_positions sums full request lengths, including repeated prefixes and
duplicate requests. Requests must fit the format's uint32 token count.

save requires a new target directory with an existing parent directory. It
creates the target directory and refuses every existing target, including an
empty directory. It writes nodes-000000.bin and requests-000000.bin. It does not
provide crash-atomic publication or filesystem durability. If a process stops
during a save, remove or inspect the incomplete new directory before retrying.
Ordinary write failures remove the files that this save created.

## Existing corpus format

The reader and writer use the token records from the existing request-token
prefix trie:

| Item | Layout |
| --- | --- |
| Header | 8-byte magic, uint32 version 1, uint32 record size, 8 reserved bytes |
| Nodes file | CLTNOD01; 16-byte records: uint64 parent, int32 token, 4 reserved bytes |
| Requests file | CLTREQ01; 24-byte records: uint64 leaf, uint32 token count, 12 metadata bytes |

All integers use little-endian byte order. The reader ignores reserved and
metadata bytes. The writer sets these bytes to zero. It preserves token paths,
node ids, request order, and duplicate requests; it does not preserve capture
metadata.

The reader rejects wrong magic, version, stride, incomplete records, invalid
tokens, nondecreasing parents, missing leaves, and incorrect request depths.
The parent rule rejects cycles without recursion. Parsing and path traversal
are iterative.
