#!/usr/bin/env python3
"""Offline DAG logic reference from AGENTS.md volume 09.

Proves cycle-preferred Kahn views, ready-set activation, unique node order,
and fixed-frame close/open composition. Does not test a model or backend.
"""
from heapq import heappop, heappush
from itertools import permutations


def topo(priority, edges):
    priority = tuple(priority)
    if len(set(priority)) != len(priority):
        raise ValueError("duplicate node")
    rank = {node: i for i, node in enumerate(priority)}
    following = {node: [] for node in priority}
    degree = {node: 0 for node in priority}
    seen = set()
    for before, after in edges:
        if before not in rank or after not in rank:
            raise ValueError("unknown endpoint")
        if (before, after) in seen:
            raise ValueError("duplicate edge")
        seen.add((before, after))
        following[before].append(after)
        degree[after] += 1
    heap = []
    for node in priority:
        if degree[node] == 0:
            heappush(heap, rank[node])
    result = []
    while heap:
        node = priority[heappop(heap)]
        result.append(node)
        for after in following[node]:
            degree[after] -= 1
            if degree[after] == 0:
                heappush(heap, rank[after])
    return tuple(result) if len(result) == len(priority) else None


def ready(nodes, edges, sealed):
    predecessors = {node: set() for node in nodes}
    for before, after in edges:
        predecessors[after].add(before)
    return tuple(node for node in nodes
                 if node not in sealed and predecessors[node] <= sealed)


def reader_view(plan_order, edges, started, reader):
    started = set(started)
    if reader not in started:
        raise ValueError("reader not started")
    if any(after in started and before not in started
           for before, after in edges):
        raise ValueError("started set misses a predecessor")
    visible_edges = tuple((a, b) for a, b in edges
                          if a in started and b in started)
    if any(a == reader for a, b in visible_edges):
        raise ValueError("active reader has an already-started successor")
    index = plan_order.index(reader)
    cycle = plan_order[index + 1:] + plan_order[:index] + (reader,)
    priority = tuple(node for node in cycle if node in started)
    return topo(priority, visible_edges)


def make_frame(node, body):
    return (("end", node), ("call", node), ("result", node),
            ("start", node), ("body", body))


def walk_frames(tokens):
    # P supplies exactly one open reasoning region.
    phase = "reasoning"
    call_id = None
    for kind, value in tokens:
        if kind == "end" and phase == "reasoning":
            phase = "content"
        elif kind == "call" and phase == "content":
            call_id = value
            phase = "waiting_tool"
        elif kind == "result" and phase == "waiting_tool" and value == call_id:
            call_id = None
            phase = "assistant_start"
        elif kind == "start" and phase == "assistant_start":
            phase = "reasoning"
        elif kind == "body" and phase == "reasoning":
            pass
        else:
            return False, phase
    return True, phase


def main():
    flat = (1, 2, 3)
    assert reader_view(flat, (), flat, 1) == (2, 3, 1)
    assert reader_view(flat, (), flat, 2) == (3, 1, 2)
    assert reader_view(flat, (), flat, 3) == (1, 2, 3)

    overlap_edges = ((1, 3),)
    assert ready(flat, overlap_edges, set()) == (1, 2)
    # Node 2 is still active. This set includes both continuing and new work.
    assert ready(flat, overlap_edges, {1}) == (2, 3)
    assert reader_view(flat, overlap_edges, flat, 2) == (1, 3, 2)
    assert reader_view(flat, overlap_edges, flat, 3) == (1, 2, 3)

    diamond = ((1, 2), (1, 3), (2, 4), (3, 4))
    assert ready((1, 2, 3, 4), diamond, {1, 2}) == (3,)
    assert ready((1, 2, 3, 4), diamond, {1, 2, 3}) == (4,)
    assert topo((1, 2, 3, 4), diamond) == (1, 2, 3, 4)

    nodes = (1, 2, 3, 4)
    possible = tuple((a, b) for a in nodes for b in nodes if a != b)
    dag_count = state_count = view_count = 0
    for mask in range(1 << len(possible)):
        edges = tuple(edge for bit, edge in enumerate(possible)
                      if mask & (1 << bit))
        if topo(nodes, edges) is None:
            continue
        dag_count += 1
        for sealed_mask in range(1 << len(nodes)):
            sealed = {node for bit, node in enumerate(nodes)
                      if sealed_mask & (1 << bit)}
            if any(b in sealed and a not in sealed for a, b in edges):
                continue
            active = ready(nodes, edges, sealed)
            if not active:
                continue
            started = sealed | set(active)
            state_count += 1
            for reader in active:
                view = reader_view(nodes, edges, started, reader)
                assert view is not None and view[-1] == reader
                assert len(view) == len(set(view)) == len(started)
                assert set(view) == started
                position = {node: i for i, node in enumerate(view)}
                assert all(position[a] < position[b] for a, b in edges
                           if a in started and b in started)
                assert set(active) - {reader} <= set(view[:-1])
                view_count += 1

    frame_count = 0
    for order in permutations(nodes):
        for prefix_mask in range(1 << len(nodes)):
            tokens = ()
            for node in order:
                body = (node, "long" if prefix_mask & (1 << (node - 1)) else "short")
                tokens += make_frame(node, body)
            assert walk_frames(tokens) == (True, "reasoning")
            frame_count += 1
    double_close = make_frame(1, "R1") + (("end", 1),) + make_frame(2, "R2")
    assert not walk_frames(double_close)[0]
    final = make_frame(1, "R1") + make_frame(2, "R2")
    final += make_frame(0, "synthesis") + (("end", 0),)
    assert walk_frames(final) == (True, "content")

    clock_nodes = tuple((t, lane) for t in range(5) for lane in (1, 2, 3))
    clock_edges = tuple(((t, a), (t + 1, b)) for t in range(4)
                        for a in (1, 2, 3) for b in (1, 2, 3))
    assert topo(clock_nodes, clock_edges) is not None
    assert topo((1, 2), ((1, 2), (2, 1))) is None
    assert topo(("0.plan", "child", "0.synthesize"),
                (("0.plan", "child"), ("child", "0.synthesize"))) is not None
    assert (dag_count, state_count, view_count, frame_count) == (543, 3007, 3904, 384)
    print("DAGs:", dag_count)
    print("Legal live states:", state_count)
    print("Reader views:", view_count)
    print("Fixed-frame compositions:", frame_count)
    print("All pure logical checks passed; no model/backend tested.")


if __name__ == "__main__":
    main()
