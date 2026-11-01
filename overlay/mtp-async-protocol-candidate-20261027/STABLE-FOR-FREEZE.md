STABLE SNAPSHOT for freeze approval — merge-dualwin-20261031

XWIN removal (this round): the GGML_TP5_DW_XWIN_DEVICE gate, dw_xwin_device.h, the per-window
rank/observation state and the TP5_DW_XWIN_DEVICE receipt are gone from runtime.cpp; the
tail_epoch provider member, its reader, its publish point, the ABI-3 bump and ABI_VERSION_MIN are
gone from relay.hip / ggml-relay-impl.h / ggml-backend-meta.cpp; ggml_backend_relay_tail_epoch
and its public declaration are gone; ggml/include/ggml-xstream.h is deleted. relay.hip,
ggml-relay-impl.h, ggml/include/ggml-relay.h, ggml-backend.cpp and ggml-cuda.cu are now
BYTE-IDENTICAL to HEAD at the end of that round, and the relay provider ABI was 2 again. The
rank-event round below changes relay.hip and ggml-relay-impl.h again and moves the ABI to 3;
ggml/include/ggml-relay.h, ggml-backend.cpp and ggml-cuda.cu are still byte-identical to HEAD
(ggml-backend-impl.h carries only a pre-existing comment noting that the first-child probes are
declared in the public header).

Kept (real consumers): ggml_backend_meta_first_child_device / ..._first_child_backend, used by
the GGML_TP5_DW_BACKEND_EVENT leg. The event-creation fallback to the first child that implements
event_new is inline in runtime.cpp over ggml_backend_meta_n_backends / ..._simple_backend, so no
new Meta symbol and no public-header change is introduced.

Rank events: RETRACTED. An intermediate revision added a provider vtable member `rank_events`
(ABI 3) exposing each rank lane's HIP event, and the overlay runtime completed a PEER window's
events on the host. It was withdrawn: ggml_backend_event_t is a struct whose first member is a
device pointer (that is what ggml_backend_event_synchronize dereferences), so casting a raw
hipEvent_t into it is a type confusion that crashes on first use; and blocking one window on the
other's epoch is the same-phase lockstep the dual-window proof forbids. Relay provider ABI is 2
again, ggml-relay-impl.h is byte-identical to HEAD, relay.hip is the busy-wait stage with
`++p.status.cpu_reductions` unconditional, and the Meta rank-event accessors are gone.

The DEVICE_WAIT gate is now the runtime's switch for a SAME-WINDOW barrier:
llama_tp5_output_copy_record creates real ggml events per contributing child (all five ranks of
this window for a split output) and llama_tp5_output_event_wait completes them after the relay
drain; no barrier -> stock llama_synchronize stays, counted. Kept: the BACKEND_EVENT leg with its
skip_no_child / skip_no_device / skip_no_iface attribution. The event-creation fallback asks the
Meta implementation for the first child that can create an event
(ggml_backend_meta_first_event_child_backend, declared in ggml-backend.h), because struct
ggml_backend_device is incomplete outside libggml-base: runtime.cpp never inspects a device
interface. wait_skip_unrecorded is removed - it could never be nonzero.

Still unproven, NOT claimed complete: asynchronous send, busy-wait behaviour, provider
independence and real GPU overlap. Two windows share one physical GPU, so any overlap figure
must be read against that contention.

Source hashes (sha256)

dc619c7367f64dcf3fd683639c0f1183e6c78dc8139f897a0363ab673d05230e  ggml/include/ggml-backend.h
9280ad49fe2a904d3e0dbf2557234aa22da790b3894ab1fb21626cbbcfc7ea8e  ggml/src/ggml-backend-impl.h
ac04263e76f735f049e4d3c662eef11e817310317feb978d80e0488d3cd34b10  ggml/src/ggml-backend-meta.cpp
527c06d6540832d1a4e186b89e93bfe915a6d6fe3062304c741a6c03aa0845ec  ggml/src/ggml-relay-impl.h
1badd280b9a826d47182182a0058c9a0a4cdda362c2e9f8ee44b0f414fe3d389  ggml/src/ggml-cuda/relay.hip
c0aad9efd9eb1167b1240da95b674e7d28997f16e4964dc56fdc5b87c10f3510  include/llama.h
71a68d6d413f3ee4b37edd01e2ff47843ac16a14d5d917229ec169e2bd9a5b81  src/llama-context.cpp
ecc6e599122ebee240223a9f1be9d1e8822dfb5039cd66ebd1356628486e9f00  fork/tp5-compose/src/runtime.cpp
1e02a19c89b9031847abef57fd1835fdf0d8417c2518e27921f69241b1dcab6b  docs/backend/DUALWIN-COMPOSE-EVENT.md
ae1eae90d84dbe3a52d9a8a92c8e2a3a640953df2d02ce12163acafcbf52bfbf  overlay/mtp-async-protocol-candidate-20261027/runtime.cpp
223d5d07965889a952a4c8455d5cd9245b924212b4b536331933a936ff088d99  overlay/mtp-async-protocol-candidate-20261027/runtime.patch
ac04263e76f735f049e4d3c662eef11e817310317feb978d80e0488d3cd34b10  overlay/mtp-async-protocol-candidate-20261027/ggml-backend-meta.cpp
bec1cabecb4febaca69b9f5775313a12e271273147ac6ce65b4a82a2efd32ceb  overlay/mtp-async-protocol-candidate-20261027/meta.patch
daf10a4d333861acd2ba38ef74f200bbe520ebd04dc10f724d1ee44615f4c055  overlay/mtp-async-protocol-candidate-20261027/dw_device_wait.h
857eefd189ec6a338cf79cb58c979e9cad46d68358e9bda0dc27dcb706de6204  overlay/mtp-async-protocol-candidate-20261027/dw-thru.h
1badd280b9a826d47182182a0058c9a0a4cdda362c2e9f8ee44b0f414fe3d389  overlay/mtp-async-protocol-candidate-20261027/hip-relay/relay.hip
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  overlay/mtp-async-protocol-candidate-20261027/hip-relay/relay.patch
dc619c7367f64dcf3fd683639c0f1183e6c78dc8139f897a0363ab673d05230e  overlay/mtp-async-protocol-candidate-20261027/headers/ggml-backend.h
654a568ec586e3926eb13f2b6511b65f52b0b33c0b49d2d227f5c89240e5250d  overlay/mtp-async-protocol-candidate-20261027/headers/ggml-backend.patch

PROVENANCE.md / hip-relay/PROVENANCE.md / headers/README.md carry their own hashes and are
regenerated after this file.

Patch chains (reproducible from real pre-images; Devops verifies)
  base/runtime.cpp           + runtime.patch -> runtime.cpp
  base/ggml-backend-meta.cpp + meta.patch    -> ggml-backend-meta.cpp
  base/runtime.ldl.cpp       + runtime.ldl.patch -> base/runtime.cpp            (built source)
  base/ggml-backend-meta.ldl.cpp + meta.ldl.patch -> base/ggml-backend-meta.cpp (built source)
  hip-relay/relay.orig.hip   + relay.ldl.patch -> relay.ldl.hip                (busy-wait stage)
  hip-relay/relay.ldl.hip    + relay.patch     -> relay.hip                    (delivered)
  headers/base/ggml-backend.h + headers/ggml-backend.patch -> headers/ggml-backend.h
  headers/base/llama.h        + headers/llama.patch        -> headers/llama.h

Not compiled, not tested, no GPU. Devops owns all verification.
