# Event Ownership and Mapping

## Ownership

- B owns in-memory event values only: `LeaseEvent`, `PlanEvent`, and
  `PlanKey`. These types are trivially copyable and do no I/O.
- WS0 owns the external JSON schemas: `alloc.schema.json`,
  `plan.schema.json`, and `leg.schema.json`.
- The C adapter writes the external JSON. B does not write JSON.
- B does not define `LegManifest`. The leg manifest belongs to the WS0 and C
  boundary. If C needs a value type, its adapter defines one.
- `LeaseEvent` is not the WS0 `AllocEvent`. The WS0 record observes uprobe and
  kprobe data and uses a different class, action, and window vocabulary.

## LeaseEvent to WS0 AllocEvent

| B LeaseEvent | WS0 AllocEvent | Adapter rule |
| --- | --- | --- |
| `ts_ns` | `ts_ns` | Same monotonic nanosecond clock. |
| `leg_id` (u32) | `leg_id` (string) | Map the numeric id to the leg label. |
| `seq` | none | B ordering only. Drop. |
| `lifetime` | none | B lifecycle scope. Drop or keep in a side record. |
| `reuse_class` | `class` | Not equivalent. B values are frame, staging, cold. WS0 values are kv, relay, graph, weight, scratch, slot, other. The adapter decides any mapping. |
| `site` (u32) | `site` (string) | Map the site id to a symbol or binary offset. |
| `bytes` | `bytes` | Same value. |
| `slot` | `slot` | Same value. |
| `slot_gen` | `slot_gen` | Same value. |
| `arena_gen` | `arena_gen` | Same value. |
| `window` (u32) | `window` (string) | Not equivalent. B holds an arena window number. WS0 holds idle, steady, startstop, degraded, or unknown. The adapter decides the sampling label. |
| `request` | `request` | Same value: 0, 1, or null. |
| `action` | `action` | See the action table below. |
| `from`, `to` | none | B state transition. Drop or keep in a side record. |

Lease action mapping:

| B LeaseAction | WS0 action |
| --- | --- |
| `lease` | `alloc` |
| `release` | `free` |
| `cancel` | `free` |
| `quarantine` | `free` (the B fault detail has no WS0 field) |
| `drain` | none, omit |
| `advance_window` | none, omit |

## PlanEvent to WS0 PlanEvent

| B PlanEvent | WS0 PlanEvent | Adapter rule |
| --- | --- | --- |
| `ts_ns` | `ts_ns` | Same clock. |
| `leg_id` (u32) | `leg_id` (string) | Map the numeric id to the leg label. |
| `seq` | none | B ordering only. Drop. |
| `plan_key` (PlanKey) | `plan_key` (string) | Use `PlanKey::describe()` or format `PlanKey::hash()` as 16 lowercase hex digits. |
| `kind` | `action` | See the kind table below. |
| `graph_exec_create` | `action` or `graph_exec_create` | Use one convention per capture: the action value or the counter. |
| `graph_exec_destroy` | `action` or `destroy` | Use one convention per capture: the action value or the counter. |
| `relay_uid` (u64) | `relay_uid` (string or null) | Format the id as text. |
| `arena_gen` | none | Join with the alloc events, or drop. |

Plan kind mapping:

| B PlanKind | WS0 action |
| --- | --- |
| `hit` | `hit` |
| `miss` | `miss` |
| `evict` | `evict` |
| `flush` | `flush` |
| `insert` | `other` |
| `replace` | `other` |
| `pin` | `other` |
| `unpin` | `other` |

WS0 may express hit and miss through the action value or through the counters.
The adapter uses exactly one convention per capture.

## LegManifest

B has no `LegManifest`. The WS0 `leg.schema.json` defines the external record.
C owns any value type and serializer it needs for that record.
