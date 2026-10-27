# TQRerot ownership

This directory is an independent C++17 library. It must build and install
without llama.cpp, GGML, Relay, a model, a device SDK, or repository helpers.
The installed target is `TQRerot::rerot`.

Keep the dependency direction flat. This library owns logical DAGs, stage
state, frontier transactions, and immutable reader views. It does not call A,
Relay, a callback supplied by either, or a backend. C calls the participants
and supplies real completion facts.

Keep physical KV ownership, tensor storage, recurrent state, sampler state,
tokens, templates, and transport outside this directory. Never use a pen or
physical row as a logical stage identity.

Preserve token-by-token whiteboard visibility. Each running stage advances
one input token per frontier. All peer increments become visible together
at the next frontier. FRAME setup is an explicit transaction and is not a
reason to delay an already running peer's BODY until completion.

Reject invalid or stale input before changing owned state. A failed compute
enters fail-stop. A drain acknowledgement is an input from C, not a device
fence implemented here. Do not hide a partial publication behind a counter.

Do not add model or hardware claims to the unit-test results. Changes to
visibility, stage lifetime, phase arithmetic, or publication need independent
counterexamples. Retain the exact global-softmax and stored-key oracle.
