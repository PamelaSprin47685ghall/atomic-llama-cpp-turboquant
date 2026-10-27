"""Check one output slot with two readers and two generations.

This finite CPU model checks slot ownership. It does not model HIP,
C++ atomics, caches, kernel scheduling, or native completion.
"""

from collections import deque
import json


def inspect(use_input_credit):
    # State: payload epoch, reader cursors, done mask, reused flag.
    start = (1, (0, 0), 0, False)
    pending = deque([(start, [])])
    seen = {start}
    unsafe_edges = 0
    first_error = None
    complete_reuse_states = 0

    while pending:
        state, trace = pending.popleft()
        epoch, cursors, done, reused = state
        if reused and done == 3:
            complete_reuse_states += 1

        changes = []
        for reader in range(2):
            if cursors[reader] < 2:
                action = f"reader {reader} reads word {cursors[reader]}"
                if epoch != 1:
                    unsafe_edges += 1
                    if first_error is None:
                        first_error = trace + [action]
                else:
                    next_cursors = list(cursors)
                    next_cursors[reader] += 1
                    changes.append(
                        ((epoch, tuple(next_cursors), done, reused), action)
                    )
            elif not done & (1 << reader):
                changes.append(
                    (
                        (epoch, cursors, done | (1 << reader), reused),
                        f"reader {reader} releases output",
                    )
                )

        may_reuse = use_input_credit or done == 3
        if not reused and may_reuse:
            changes.append(
                ((2, cursors, done, True), "writer stores epoch 2")
            )

        for next_state, action in changes:
            if next_state not in seen:
                seen.add(next_state)
                pending.append((next_state, trace + [action]))

    return {
        "states": len(seen),
        "unsafe_read_edges": unsafe_edges,
        "complete_reuse_states": complete_reuse_states,
        "first_counterexample": first_error,
    }


def main():
    broken = inspect(use_input_credit=True)
    fixed = inspect(use_input_credit=False)
    assert broken["unsafe_read_edges"] > 0
    assert fixed["unsafe_read_edges"] == 0
    assert fixed["complete_reuse_states"] > 0

    # A separate torn-row witness.
    old_row = [101, 102]
    consumed = [old_row[0]]
    new_row = [201, 202]
    consumed.append(new_row[1])
    assert consumed == [101, 202]
    assert consumed != old_row

    print(json.dumps(
        {"broken": broken, "fixed": fixed, "torn_row": consumed},
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Run without -O or -OO; assertions are required")
    main()
