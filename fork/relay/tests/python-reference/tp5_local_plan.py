"""Check a five-rank synthetic GLU layout on the CPU; no GPU qualification."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


MAX_U64 = (1 << 64) - 1
MAX_INPUT_BYTES = 65536
INPUT_SCHEMA = "tp5.local_glu.v1"
PLANNER_REVISION = "local-glu-cpu-1"
RANKS = 5
FIELDS = {
    "schema", "ranks", "hidden", "intermediate", "active_rows",
    "block_elements", "block_bytes", "storage_layout", "strategy",
    "output_layout",
}


class PlanError(ValueError):
    def __init__(self, decision, code, field, detail):
        super().__init__(detail)
        self.decision = decision
        self.code = code
        self.field = field
        self.detail = detail

    def report(self):
        return {
            "schema": "tp5.local_glu.result.v1",
            "decision": self.decision,
            "scope": "synthetic_local_storage_geometry_only",
            "gpu_used": False,
            "error": {"code": self.code, "field": self.field,
                      "detail": self.detail},
        }


def invalid(code, field, detail):
    raise PlanError("invalid", code, field, detail)


def infeasible(code, field, detail):
    raise PlanError("infeasible", code, field, detail)


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            invalid("DUPLICATE_KEY", key, "JSON object repeats this key")
        value[key] = item
    return value


def json_integer(text):
    if len(text.lstrip("-")) > 20:
        invalid("INTEGER_TOO_LONG", "$", "integer exceeds the input limit")
    return int(text)


def no_float(text):
    invalid("FLOAT_NOT_ALLOWED", "$", "this schema accepts integer counts only")


def no_constant(text):
    invalid("NONFINITE_JSON", "$", "nonstandard JSON numeric constant")


def load_spec(data):
    if len(data) > MAX_INPUT_BYTES:
        invalid("INPUT_TOO_LARGE", "$", "input exceeds 65536 bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        invalid("INVALID_UTF8", "$", "input is not UTF-8")
    try:
        value = json.loads(
            text, object_pairs_hook=unique_object,
            parse_int=json_integer, parse_float=no_float,
            parse_constant=no_constant,
        )
    except (json.JSONDecodeError, RecursionError) as error:
        invalid("JSON_SYNTAX", "$", str(error))
    return validate_spec(value)


def validate_spec(value):
    if type(value) is not dict:
        invalid("WRONG_TYPE", "$", "top-level value must be an object")
    if any(type(key) is not str for key in value):
        invalid("WRONG_TYPE", "$", "object field names must be strings")
    unknown, missing = set(value) - FIELDS, FIELDS - set(value)
    if unknown:
        invalid("UNKNOWN_FIELD", sorted(unknown)[0], "unknown schema field")
    if missing:
        invalid("MISSING_FIELD", sorted(missing)[0], "required field is absent")
    if type(value["schema"]) is not str or value["schema"] != INPUT_SCHEMA:
        invalid("SCHEMA_MISMATCH", "schema", "unsupported input schema")
    for name in ("ranks", "hidden", "intermediate", "active_rows",
                 "block_elements", "block_bytes"):
        item = value[name]
        if type(item) is not int:
            invalid("WRONG_TYPE", name, "expected an integer, excluding bool")
        if not 1 <= item <= MAX_U64:
            invalid("OUT_OF_RANGE", name, "expected 1 through UINT64_MAX")
    if value["ranks"] != RANKS:
        invalid("RANK_COUNT", "ranks", "this teaching planner requires five ranks")
    enums = {
        "storage_layout": {"independent_blocks_no_padding"},
        "strategy": {"gather_intermediate", "split_intermediate"},
        "output_layout": {"sharded", "replicated"},
    }
    for name, allowed in enums.items():
        if type(value[name]) is not str or value[name] not in allowed:
            invalid("INVALID_ENUM", name, "unsupported value for this schema")
    return {key: value[key] for key in sorted(FIELDS)}


def product(field, *values):
    result = 1
    for value in values:
        if result and value > MAX_U64 // result:
            invalid("SIZE_OVERFLOW", field, "product exceeds UINT64_MAX")
        result *= value
    return result


def total(field, values):
    result = 0
    for value in values:
        if value > MAX_U64 - result:
            invalid("SIZE_OVERFLOW", field, "sum exceeds UINT64_MAX")
        result += value
    return result


def partition(size, unit, field):
    if size % unit:
        infeasible("PARTITION_ALIGNMENT", field, "axis does not contain whole units")
    units = size // unit
    if units < RANKS:
        infeasible("INSUFFICIENT_UNITS", field,
                   "five nonempty shards require at least five indivisible units")
    base, extra = divmod(units, RANKS)
    result = []
    begin = 0
    for rank in range(RANKS):
        end = begin + (base + (rank < extra)) * unit
        result.append([begin, end])
        begin = end
    return result


def row_bytes(spec, features, field):
    block = spec["block_elements"]
    if features % block:
        infeasible("ROW_BLOCK_ALIGNMENT", field,
                   "a row must contain whole blocks with no padding")
    return product(field, features // block, spec["block_bytes"])


def collective(step, value, operation, distribution, features, rows,
               producer_ranges, consumer_ranges):
    sent = [product("producer_bytes", end - begin, rows, 4)
            for begin, end in producer_ranges]
    received = [product("consumer_bytes", end - begin, rows, 4)
                for begin, end in consumer_ranges]
    return {
        "step": step, "value": value, "operation": operation,
        "distribution": distribution, "features": features,
        "active_rows": rows, "wire": "f32", "wire_element_bytes": 4,
        "wire_order": "active_row_then_local_feature",
        "range_unit": "feature_index_per_active_row",
        "producer_ranges": producer_ranges,
        "consumer_ranges": consumer_ranges,
        "producer_bytes": sent, "consumer_bytes": received,
        "endpoint_send_bytes": total("endpoint_send_bytes", sent),
        "endpoint_receive_bytes": total("endpoint_receive_bytes", received),
        "assembled_value_bytes": product("assembled_value_bytes", features, rows, 4),
    }


def compile_plan(spec):
    # Revalidate direct callers, including bool and floating-point counts.
    spec = validate_spec(spec)
    h, f, rows = spec["hidden"], spec["intermediate"], spec["active_rows"]
    hrow = row_bytes(spec, h, "gate_up_row")
    frow = row_bytes(spec, f, "down_row")
    strategy, layout = spec["strategy"], spec["output_layout"]
    unit = 1 if strategy == "gather_intermediate" else spec["block_elements"]
    middle = partition(f, unit, "intermediate")
    hidden = None
    if strategy == "gather_intermediate" or layout == "sharded":
        hidden = partition(h, 1, "output_hidden")
    consumers = hidden if layout == "sharded" else [[0, h] for _ in range(RANKS)]
    ranks = []
    for rank, (begin, end) in enumerate(middle):
        width = end - begin
        gate_up_bytes = product("gate_up_weight_bytes", 2, width, hrow)
        gate_up_macs = product("gate_up_macs", 2, rows, h, width)
        if strategy == "gather_intermediate":
            out_begin, out_end = hidden[rank]
            down_input = [0, f]
            down_output = [out_begin, out_end]
            down_bytes = product("down_weight_bytes", out_end - out_begin, frow)
            down_macs = product("down_macs", rows, out_end - out_begin, f)
            kind = "complete_output_rows"
        else:
            down_input, down_output = [begin, end], [0, h]
            down_bytes = product("down_weight_bytes", h,
                                 row_bytes(spec, width, "down_input_slice"))
            down_macs = product("down_macs", rows, h, width)
            kind = "full_hidden_partial_sum"
        ranks.append({
            "rank": rank, "gate_up_output_range": [begin, end],
            "down_input_range": down_input, "down_output_range": down_output,
            "down_value_kind": kind,
            "gate_up_weight_bytes": gate_up_bytes, "down_weight_bytes": down_bytes,
            "source_weight_bytes": total("rank_weight_bytes", [gate_up_bytes, down_bytes]),
            "linear_macs": total("rank_linear_macs", [gate_up_macs, down_macs]),
        })
    if strategy == "gather_intermediate":
        transfers = [collective(0, "middle_activation", "gather", "broadcast",
                                f, rows, middle, [[0, f] for _ in range(RANKS)])]
        if layout == "replicated":
            transfers.append(collective(1, "output_hidden", "gather", "broadcast",
                                        h, rows, hidden, consumers))
    else:
        transfers = [collective(0, "output_hidden", "sum",
                                "scatter" if layout == "sharded" else "broadcast",
                                h, rows, [[0, h] for _ in range(RANKS)], consumers)]
    source_bytes = total("source_weight_bytes",
                         [product("gate_up_total", 2, f, hrow),
                          product("down_total", h, frow)])
    expected_macs = product("linear_macs", 3, rows, h, f)
    if total("rank_weight_bytes", [r["source_weight_bytes"] for r in ranks]) != source_bytes:
        raise RuntimeError("internal weight coverage error")
    if total("rank_linear_macs", [r["linear_macs"] for r in ranks]) != expected_macs:
        raise RuntimeError("internal work coverage error")
    canonical = json.dumps({"revision": PLANNER_REVISION, "spec": spec},
                           sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "schema": "tp5.local_glu.result.v1", "decision": "feasible",
        "scope": "synthetic_local_storage_geometry_only", "gpu_used": False,
        "planner_revision": PLANNER_REVISION,
        "fingerprint_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "spec": spec,
        "weight_shapes_ggml_ne": {"gate": [h, f], "up": [h, f], "down": [f, h]},
        "input_boundary": {
            "layout": "replicated", "delivery_is_external": True,
            "bytes_per_rank": product("input_bytes_per_rank", rows, h, 4),
        },
        "output_boundary": {"layout": layout, "rank_feature_ranges": consumers},
        "ranks": ranks, "collectives": transfers,
        "totals": {
            "source_weight_bytes": source_bytes, "linear_macs": expected_macs,
            "endpoint_send_bytes": total("all_endpoint_send_bytes",
                                          [x["endpoint_send_bytes"] for x in transfers]),
            "endpoint_receive_bytes": total("all_endpoint_receive_bytes",
                                             [x["endpoint_receive_bytes"] for x in transfers]),
        },
        "limits": [
            "Independent per-row blocks include all block metadata and have no row padding.",
            "Source bytes exclude repacking, allocator overhead, workspace and model state.",
            "Endpoint bytes are logical transfers, not PCIe or DRAM counters.",
            "Kernel row tails, quantized arithmetic and the real model are not qualified.",
            "No transport, busy wait, synchronization, cancellation or GPU work is executed.",
        ],
    }


def example_spec(**changes):
    spec = {
        "schema": INPUT_SCHEMA, "ranks": 5, "hidden": 2048,
        "intermediate": 512, "active_rows": 2, "block_elements": 256,
        "block_bytes": 144, "storage_layout": "independent_blocks_no_padding",
        "strategy": "gather_intermediate", "output_layout": "sharded",
    }
    spec.update(changes)
    return spec


def check(condition, detail):
    if not condition:
        raise AssertionError(detail)


def result_text(result):
    # Escaped output also handles unpaired surrogates inside rejected JSON keys.
    return json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False)


def expect_error(action, decision, code):
    try:
        action()
    except PlanError as error:
        check((error.decision, error.code) == (decision, code),
              f"unexpected rejection: {error.decision}/{error.code}")
        return
    raise AssertionError("invalid or infeasible input was accepted")


def dense_fixture(h, f, rows):
    x = [[(((t + 1) * (i + 2)) % 9 - 4) * 0.05 for i in range(h)] for t in range(rows)]
    gate = [[((j * 5 + i * 3) % 13 - 6) * 0.03 for i in range(h)] for j in range(f)]
    up = [[((j * 7 + i * 2) % 11 - 5) * 0.02 for i in range(h)] for j in range(f)]
    down = [[((i * 4 + j * 3) % 17 - 8) * 0.04 for j in range(f)] for i in range(h)]
    return x, gate, up, down


def dense_reference(x, gate, up, down):
    result = []
    for row in x:
        middle = []
        for g, u in zip(gate, up):
            gv = sum(a * b for a, b in zip(g, row))
            uv = sum(a * b for a, b in zip(u, row))
            middle.append(gv / (1.0 + math.exp(-gv)) * uv)
        result.append([sum(a * b for a, b in zip(weight, middle)) for weight in down])
    return result


def dense_from_plan(plan, x, gate, up, down):
    h, f = plan["spec"]["hidden"], plan["spec"]["intermediate"]
    result = [[0.0] * h for _ in x]
    fragments = []
    for rank in plan["ranks"]:
        begin, end = rank["gate_up_output_range"]
        local = []
        for row in x:
            values = []
            for feature in range(begin, end):
                g = sum(gate[feature][i] * row[i] for i in range(h))
                u = sum(up[feature][i] * row[i] for i in range(h))
                values.append(g / (1.0 + math.exp(-g)) * u)
            local.append(values)
        fragments.append(local)
    if plan["spec"]["strategy"] == "gather_intermediate":
        joined = [[0.0] * f for _ in x]
        for rank, local in zip(plan["ranks"], fragments):
            begin, end = rank["gate_up_output_range"]
            for row_index in range(len(x)):
                joined[row_index][begin:end] = local[row_index]
        for rank in plan["ranks"]:
            begin, end = rank["down_output_range"]
            for row_index in range(len(x)):
                for feature in range(begin, end):
                    result[row_index][feature] = sum(
                        down[feature][i] * joined[row_index][i] for i in range(f))
    else:
        for rank, local in zip(plan["ranks"], fragments):
            begin, end = rank["down_input_range"]
            for row_index in range(len(x)):
                for feature in range(h):
                    partial = sum(down[feature][i] * local[row_index][i - begin]
                                  for i in range(begin, end))
                    result[row_index][feature] += partial
    return result


def dense_error(actual, expected, rows, features):
    check(type(actual) is list and type(expected) is list, "dense output must be a matrix")
    check(len(actual) == rows and len(expected) == rows, "dense row count differs")
    largest = 0.0
    for row in range(rows):
        check(type(actual[row]) is list and type(expected[row]) is list,
              "dense row must be a list")
        check(len(actual[row]) == features and len(expected[row]) == features,
              "dense feature count differs")
        for feature in range(features):
            left, right = actual[row][feature], expected[row][feature]
            check(type(left) in (int, float) and type(right) in (int, float),
                  "dense value is not numeric")
            check(math.isfinite(left) and math.isfinite(right), "nonfinite dense value")
            largest = max(largest, abs(left - right))
    return largest


def expect_check_failure(action):
    try:
        action()
    except AssertionError:
        return
    raise AssertionError("corrupt dense output was accepted")


def self_test():
    passed = []
    base = example_spec()
    plan = compile_plan(base)
    check([r["gate_up_output_range"] for r in plan["ranks"]] ==
          [[0, 103], [103, 206], [206, 308], [308, 410], [410, 512]], "wrong middle cover")
    check([r["down_output_range"] for r in plan["ranks"]] ==
          [[0, 410], [410, 820], [820, 1230], [1230, 1639], [1639, 2048]], "wrong hidden cover")
    check([r["source_weight_bytes"] for r in plan["ranks"]] ==
          [355392, 355392, 353088, 352800, 352800], "wrong source weight accounting")
    check(plan["totals"] == {"source_weight_bytes": 1769472, "linear_macs": 6291456,
                             "endpoint_send_bytes": 4096, "endpoint_receive_bytes": 20480},
          "wrong baseline totals")
    passed.append("five nonempty unaligned output-row shards preserve complete blocks")
    replicated = compile_plan(example_spec(output_layout="replicated"))
    check(len(replicated["collectives"]) == 2, "missing final hidden gather")
    check(replicated["totals"]["endpoint_send_bytes"] == 20480 and
          replicated["totals"]["endpoint_receive_bytes"] == 102400, "missing boundary traffic")
    passed.append("replicated output includes its final communication")
    expect_error(lambda: compile_plan(example_spec(strategy="split_intermediate")),
                 "infeasible", "INSUFFICIENT_UNITS")
    passed.append("two quantization blocks cannot become five nonempty input shards")
    for layout, received in (("sharded", 16384), ("replicated", 81920)):
        paired = compile_plan(example_spec(intermediate=1280, strategy="split_intermediate",
                                          output_layout=layout))
        check(paired["totals"]["source_weight_bytes"] == 4423680, "wrong paired source bytes")
        check(paired["totals"]["endpoint_send_bytes"] == 81920, "wrong sum input bytes")
        check(paired["totals"]["endpoint_receive_bytes"] == received, "wrong sum output bytes")
    passed.append("aligned paired partition handles scatter and broadcast distinctly")
    expect_error(lambda: compile_plan(example_spec(hidden=2049)), "infeasible", "ROW_BLOCK_ALIGNMENT")
    passed.append("no-padding storage rejects an incomplete source row block")
    raw = json.dumps(base).encode("utf-8")
    expect_error(lambda: load_spec(raw[:-1] + b',"hidden":2048}'), "invalid", "DUPLICATE_KEY")
    passed.append("duplicate keys cannot silently override the geometry")
    for value, code in ((True, "WRONG_TYPE"), (0, "OUT_OF_RANGE"), (-1, "OUT_OF_RANGE"),
                        (1 << 64, "OUT_OF_RANGE")):
        expect_error(lambda v=value: compile_plan(example_spec(active_rows=v)), "invalid", code)
    expect_error(lambda: load_spec(raw.replace(b'"active_rows": 2', b'"active_rows": 2.0')),
                 "invalid", "FLOAT_NOT_ALLOWED")
    passed.append("counts reject bool, float, zero, negative and uint64 overflow")
    for token in (b"NaN", b"Infinity", b"-Infinity"):
        expect_error(lambda v=token: load_spec(raw.replace(b'"active_rows": 2', b'"active_rows": ' + v)),
                     "invalid", "NONFINITE_JSON")
    expect_error(lambda: load_spec(raw.replace(b'"active_rows": 2', b'"active_rows": 1e999')),
                 "invalid", "FLOAT_NOT_ALLOWED")
    passed.append("nonfinite and exponent-form values do not enter integer arithmetic")
    expect_error(lambda: compile_plan(dict(base, extra=1)), "invalid", "UNKNOWN_FIELD")
    missing = dict(base)
    del missing["output_layout"]
    expect_error(lambda: compile_plan(missing), "invalid", "MISSING_FIELD")
    expect_error(lambda: load_spec(b"[]"), "invalid", "WRONG_TYPE")
    passed.append("schema rejects unknown, missing and wrong top-level fields")
    for suffix, expected_code in ((b',"\\ud800":1}', "UNKNOWN_FIELD"),
                                  (b',"\\ud800":1,"\\ud800":2}', "DUPLICATE_KEY")):
        try:
            load_spec(raw[:-1] + suffix)
        except PlanError as error:
            check(error.code == expected_code, "wrong surrogate-key rejection")
            encoded = result_text(error.report()).encode("utf-8")
            check(json.loads(encoded)["decision"] == "invalid", "invalid error JSON")
        else:
            raise AssertionError("unknown surrogate key was accepted")
    passed.append("rejected Unicode keys remain serializable as structured JSON errors")
    expect_error(lambda: load_spec(b" " * (MAX_INPUT_BYTES + 1)), "invalid", "INPUT_TOO_LARGE")
    expect_error(lambda: load_spec(b"\xff"), "invalid", "INVALID_UTF8")
    expect_error(lambda: load_spec(b"{} trailing"), "invalid", "JSON_SYNTAX")
    passed.append("byte limit, UTF-8 and trailing data are enforced before planning")
    expect_error(lambda: compile_plan(example_spec(hidden=1 << 63, intermediate=5,
                                                   block_elements=1)), "invalid", "SIZE_OVERFLOW")
    passed.append("derived byte and work counts cannot overflow uint64")
    alternate = load_spec(json.dumps(base, sort_keys=True, indent=2).encode("utf-8"))
    check(compile_plan(alternate)["fingerprint_sha256"] == plan["fingerprint_sha256"], "unstable canonical hash")
    check(replicated["fingerprint_sha256"] != plan["fingerprint_sha256"], "boundary omitted from hash")
    passed.append("fingerprint ignores JSON spacing but includes output semantics")
    max_abs = 0.0
    x, gate, up, down = dense_fixture(10, 15, 2)
    reference = dense_reference(x, gate, up, down)
    for strategy in ("gather_intermediate", "split_intermediate"):
        for layout in ("sharded", "replicated"):
            small = compile_plan(example_spec(hidden=10, intermediate=15, block_elements=1,
                                              block_bytes=4, strategy=strategy, output_layout=layout))
            actual = dense_from_plan(small, x, gate, up, down)
            delta = dense_error(actual, reference, 2, 10)
            check(math.isfinite(delta) and delta <= 1e-12, "dense GLU oracle disagrees")
            max_abs = max(max_abs, delta)
    passed.append("both partition algebras match an independent small dense GLU oracle")
    nonfinite = [row[:] for row in reference]
    nonfinite[0][1] = float("nan")
    short_row = [row[:] for row in reference]
    short_row[0] = short_row[0][:-1]
    for corrupt in (nonfinite, reference[:-1], short_row):
        expect_check_failure(lambda value=corrupt: dense_error(value, reference, 2, 10))
    expect_check_failure(lambda: dense_error(reference, nonfinite, 2, 10))
    passed.append("dense validation rejects NaN in either arm and every shape truncation")
    ranges = partition(15, 1, "fixture")
    correct = [-1] * 30
    wrong = [-1] * 30
    for begin, end in ranges:
        local = [row * 1000 + feature for row in range(2) for feature in range(begin, end)]
        for row in range(2):
            for local_index, feature in enumerate(range(begin, end)):
                correct[row * 15 + feature] = local[row * (end - begin) + local_index]
        wrong[begin:begin + len(local)] = local
    expected = [row * 1000 + feature for row in range(2) for feature in range(15)]
    check(correct == expected and wrong != expected, "two-row gather did not expose flat-copy bug")
    passed.append("two active rows require row-strided placement when gathering fragments")
    return {
        "schema": "tp5.local_glu.tests.v1", "passed": True, "checks": len(passed),
        "scope": "cpu_symbolic_plan_and_small_dense_arithmetic", "gpu_used": False,
        "dense_max_abs": max_abs, "checks_passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path)
    group.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        code = 0
    else:
        try:
            try:
                with args.input.open("rb") as source:
                    data = source.read(MAX_INPUT_BYTES + 1)
            except OSError as error:
                invalid("INPUT_IO", "$", str(error))
            result = compile_plan(load_spec(data))
            code = 0
        except PlanError as error:
            result = error.report()
            code = 3 if error.decision == "infeasible" else 2
    print(result_text(result))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
