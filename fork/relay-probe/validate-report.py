#!/usr/bin/env python3
"""Strict evidence validator for TQRelayProbe reports and bench files.

Guide 53.4: a qualification result is trustworthy only if its terminal state
is internally consistent. This script is the machine-checkable version of
that contract, meant for the operator on the real machine: after a run,
validate the report (and bench file, if produced) before shipping them back.
It checks structure, types, and every cross-field invariant the probe writer
guarantees; it does not interpret platform facts.

Usage:
  validate-report.py REPORT [BENCH]     exit 0: both valid; 1: invalid; 2: usage
  validate-report.py --self-test        verify the validator itself, no files

The validator is strict on purpose: it knows the exact field set of
schema=1 as written by src/evidence.cpp, rejects duplicate keys, non-finite
or non-integer numbers, trailing data, and unknown fields. A future writer
that adds fields must bump the schema or extend this validator; that
drift must be loud, because evidence identity is part of the deliverable.
"""

import json
import sys

REPORT_TARGET = "tp5-mapped-relay"
BENCH_TARGET = "tp5-relay-bench"
BENCH_MEASUREMENTS = {
    "host_publish_to_device_ack": "ns",
    "device_produce_to_host_ready": "ns",
    "five_stream_relay_cycle": "ns",
    "host_five_way_fold": "bytes_per_second",
}

# Case registry order of src/evidence.cpp. Names are part of the evidence
# record: renumbering or renaming changes what an old report can claim, so
# this list only grows at the end.
REGISTRY = [
    "device_identity",
    "mapped_host_access",
    "cpu_to_gpu_publish",
    "gpu_to_cpu_publish",
    "producer_delay_ordering",
    "ticket_semantics",
    "multi_group_completion",
    "abort_stop",
    "host_system_rmw_u64_add",
    "host_system_rmw_u64_cas",
    "five_rank_cross_device",
    "multi_consumer_output_credit",
    "delayed_consumer_ordering",
    "two_bank_rotation",
    "varying_live_bytes",
    "five_stream_relay_cycle",
    "offset_addressing",
    "multi_group_ordering",
    "native_sync_deadline",
    "same_stream_inverse_dependency",
    "dual_window_independent_progress",
    "service_loop_independent_progress",
    "nonfinite_reduction_gate",
    "host_wait_deadline_no_reuse",
    "five_rank_indexing",
]

# Every healthy device record carries every case except the five-rank one,
# which exists only when exactly five devices ran it (guide 31.7).
BASE_CASES = [name for name in REGISTRY if name != "five_rank_cross_device"]

# The two system-RMW bits are independent capability observations, not
# qualification gates (probe AGENTS.md LEMMA 2); every other case must pass
# on every device for qualification=pass. This mirrors run_probe's gate.
MANDATORY_CASES = [
    name for name in BASE_CASES
    if name not in ("host_system_rmw_u64_add", "host_system_rmw_u64_cas")
]

OUTCOMES = {"pass", "fail", "unsupported", "not_run"}


class EvidenceError(ValueError):
    pass


def fail(message):
    raise EvidenceError(message)


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            fail("duplicate key %r" % key)
        value[key] = item
    return value


def reject_constant(text):
    fail("non-finite JSON number %r" % text)


def load_strict(path):
    with open(path, "rb") as source:
        data = source.read()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        fail("not UTF-8")
    return load_strict_text(text)


def require_dict(value, where):
    if type(value) is not dict:
        fail("%s must be an object" % where)
    for key in value:
        if type(key) is not str:
            fail("%s has a non-string field name" % where)
        require_str(key, where + " field name")


def require_int(value, where):
    if type(value) is not int or type(value) is bool:
        fail("%s must be an integer, not a bool or float" % where)
    if not 0 <= value <= (1 << 64) - 1:
        fail("%s must be a uint64" % where)


def require_str(value, where, nonempty=False):
    if type(value) is not str:
        fail("%s must be a string" % where)
    if nonempty and not value:
        fail("%s must not be empty" % where)
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        fail("%s must contain valid Unicode scalar values" % where)


def require_keys(value, expected, where):
    actual = set(value.keys())
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        message = "%s has the wrong field set" % where
        if missing:
            message += " (missing %s)" % ", ".join(missing)
        if extra:
            message += " (unknown %s)" % ", ".join(extra)
        fail(message)


def validate_device(device, index, device_count):
    require_dict(device, "devices[%d]" % index)
    require_keys(device, [
        "rank", "pci_bdf", "reported_name", "gcn_arch_name",
        "total_vram_bytes", "can_map_host_memory", "cases",
    ], "devices[%d]" % index)
    require_int(device["rank"], "devices[%d].rank" % index)
    if device["rank"] != index:
        fail("devices[%d].rank=%r must equal its index" % (index, device["rank"]))
    require_str(device["pci_bdf"], "devices[%d].pci_bdf" % index)
    require_str(device["reported_name"], "devices[%d].reported_name" % index)
    require_str(device["gcn_arch_name"], "devices[%d].gcn_arch_name" % index)
    require_int(device["total_vram_bytes"], "devices[%d].total_vram_bytes" % index)
    if type(device["can_map_host_memory"]) is not bool:
        fail("devices[%d].can_map_host_memory must be a bool" % index)
    cases = device["cases"]
    require_dict(cases, "devices[%d].cases" % index)
    expected = set(BASE_CASES)
    if device_count == 5:
        expected.add("five_rank_cross_device")
    actual = set(cases.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        message = "devices[%d].cases has the wrong case set" % index
        if missing:
            message += " (missing %s)" % ", ".join(missing)
        if extra:
            message += " (unknown %s)" % ", ".join(extra)
        fail(message)
    for name, entry in cases.items():
        require_dict(entry, "devices[%d].cases.%s" % (index, name))
        require_keys(entry, ["result", "detail"], "devices[%d].cases.%s" % (index, name))
        require_str(entry["result"], "devices[%d].cases.%s.result" % (index, name))
        if entry["result"] not in OUTCOMES:
            fail("devices[%d].cases.%s has an invalid result %r" % (index, name, entry["result"]))
        require_str(entry["detail"], "devices[%d].cases.%s.detail" % (index, name), nonempty=True)
    identity = cases["device_identity"]["result"]
    if identity == "pass":
        if not device["pci_bdf"] or not device["reported_name"] or not device["gcn_arch_name"]:
            fail("devices[%d] passed device_identity but lacks identity fields" % index)
        if device["total_vram_bytes"] == 0:
            fail("devices[%d].total_vram_bytes must be nonzero for a healthy device" % index)
    return cases


def expected_qualification(devices, five_rank_status):
    if not devices:
        return "fail"
    for index, device in enumerate(devices):
        for name in MANDATORY_CASES:
            if device["cases"][name]["result"] != "pass":
                return "fail"
    if len(devices) == 5 and five_rank_status != "pass":
        return "fail"
    return "pass"


def validate_report(value):
    require_dict(value, "report")
    require_keys(value, [
        "schema", "target", "build_id", "hip_runtime_version",
        "hip_runtime_version_string", "compiler_version", "host_arch",
        "control_word_bits", "memory_flags_requested", "hot_reuse_rounds",
        "devices", "five_rank_status", "five_rank_detail", "qualification",
        "qualification_detail", "limits",
    ], "report")
    require_int(value["schema"], "schema")
    if value["schema"] != 1:
        fail("schema must be 1")
    require_str(value["target"], "target", nonempty=True)
    if value["target"] != REPORT_TARGET:
        fail("target must be %r" % REPORT_TARGET)
    require_str(value["build_id"], "build_id", nonempty=True)
    require_int(value["hip_runtime_version"], "hip_runtime_version")
    require_str(value["hip_runtime_version_string"], "hip_runtime_version_string", nonempty=True)
    require_str(value["compiler_version"], "compiler_version", nonempty=True)
    require_str(value["host_arch"], "host_arch", nonempty=True)
    require_int(value["control_word_bits"], "control_word_bits")
    if value["control_word_bits"] != 64:
        fail("control_word_bits must be 64")
    require_str(value["memory_flags_requested"], "memory_flags_requested", nonempty=True)
    require_int(value["hot_reuse_rounds"], "hot_reuse_rounds")
    if not 1 <= value["hot_reuse_rounds"] <= (1 << 32) - 1:
        fail("hot_reuse_rounds must be a positive uint32")
    devices = value["devices"]
    if type(devices) is not list:
        fail("devices must be an array")
    device_count = len(devices)
    parsed = [validate_device(device, index, device_count)
              for index, device in enumerate(devices)]

    status = value["five_rank_status"]
    require_str(status, "five_rank_status", nonempty=True)
    require_str(value["five_rank_detail"], "five_rank_detail")
    if device_count > 0 and not value["five_rank_detail"]:
        fail("five_rank_detail must not be empty when devices ran")
    if device_count == 5:
        if status not in ("pass", "fail"):
            fail("five_rank_status must be pass or fail when five devices ran")
        for index, cases in enumerate(parsed):
            result = cases["five_rank_cross_device"]["result"]
            wanted = "pass" if status == "pass" else "fail"
            if result != wanted:
                fail("devices[%d].five_rank_cross_device=%r disagrees with five_rank_status=%r"
                     % (index, result, status))
    else:
        if status != "not_run":
            fail("five_rank_status must be not_run when %d devices ran" % device_count)

    qualification = value["qualification"]
    require_str(qualification, "qualification", nonempty=True)
    if qualification not in OUTCOMES:
        fail("qualification has an invalid result %r" % qualification)
    require_str(value["qualification_detail"], "qualification_detail", nonempty=True)
    expected = expected_qualification(devices, status)
    if qualification != expected:
        fail("qualification=%r but the case results require %r" % (qualification, expected))

    limits = value["limits"]
    if type(limits) is not list:
        fail("limits must be an array")
    for index, limit in enumerate(limits):
        require_str(limit, "limits[%d]" % index, nonempty=True)


def validate_bench(value):
    require_dict(value, "bench")
    require_keys(value, [
        "schema", "target", "build_id", "hip_runtime_version_string",
        "device_ordinal", "device_name", "gcn_arch_name", "rounds",
        "measurements", "limits",
    ], "bench")
    require_int(value["schema"], "schema")
    if value["schema"] != 1:
        fail("schema must be 1")
    require_str(value["target"], "target", nonempty=True)
    if value["target"] != BENCH_TARGET:
        fail("target must be %r" % BENCH_TARGET)
    require_str(value["build_id"], "build_id", nonempty=True)
    require_str(value["hip_runtime_version_string"], "hip_runtime_version_string", nonempty=True)
    require_int(value["device_ordinal"], "device_ordinal")
    require_str(value["device_name"], "device_name", nonempty=True)
    require_str(value["gcn_arch_name"], "gcn_arch_name", nonempty=True)
    require_int(value["rounds"], "rounds")
    if not 1 <= value["rounds"] <= (1 << 32) - 1:
        fail("rounds must be a positive uint32")
    measurements = value["measurements"]
    if type(measurements) is not list or not measurements:
        fail("measurements must be a nonempty array")
    seen = set()
    for index, measurement in enumerate(measurements):
        where = "measurements[%d]" % index
        require_dict(measurement, where)
        require_keys(measurement, ["name", "unit", "samples", "min", "median", "p90", "max"], where)
        require_str(measurement["name"], where + ".name", nonempty=True)
        if measurement["name"] not in BENCH_MEASUREMENTS:
            fail("%s has an unknown measurement name" % where)
        if measurement["name"] in seen:
            fail("%s repeats measurement name %r" % (where, measurement["name"]))
        seen.add(measurement["name"])
        require_str(measurement["unit"], where + ".unit", nonempty=True)
        if measurement["unit"] != BENCH_MEASUREMENTS[measurement["name"]]:
            fail("%s.unit disagrees with its measurement name" % where)
        for field in ("samples", "min", "median", "p90", "max"):
            require_int(measurement[field], where + "." + field)
        expected_samples = 32 if measurement["name"] == "host_five_way_fold" else value["rounds"]
        if measurement["samples"] != expected_samples:
            fail("%s.samples must be %d" % (where, expected_samples))
        if not (measurement["min"] <= measurement["median"] <= measurement["p90"] <= measurement["max"]):
            fail("%s must satisfy min <= median <= p90 <= max" % where)
    if seen != set(BENCH_MEASUREMENTS):
        fail("bench is missing required measurements")
    limits = value["limits"]
    if type(limits) is not list:
        fail("limits must be an array")
    for index, limit in enumerate(limits):
        require_str(limit, "limits[%d]" % index, nonempty=True)


def validate_pair(report, bench):
    """Check the shared identity fields; this is not proof of same-run provenance."""
    ordinal = bench["device_ordinal"]
    if ordinal >= len(report["devices"]):
        fail("bench device_ordinal is absent from the report")
    device = report["devices"][ordinal]
    if bench["device_name"] != device["reported_name"]:
        fail("bench device_name disagrees with the report")
    if bench["gcn_arch_name"] != device["gcn_arch_name"]:
        fail("bench gcn_arch_name disagrees with the report")
    if bench["rounds"] != report["hot_reuse_rounds"]:
        fail("bench rounds disagree with the report")
    if bench["hip_runtime_version_string"] != report["hip_runtime_version_string"]:
        fail("bench HIP version disagrees with the report")


# -------- self-test ---------------------------------------------------------

def good_report(device_count=1, qualification=None):
    if device_count == 5:
        status, status_detail = "pass", "five devices each wrote a disjoint word"
    elif device_count == 0:
        status, status_detail = "not_run", ""
    else:
        status, status_detail = "not_run", "requires exactly five devices; %d present" % device_count
    report = {
        "schema": 1,
        "target": REPORT_TARGET,
        "build_id": "tq-relay-probe",
        "hip_runtime_version": 70000204,
        "hip_runtime_version_string": "7.2.4",
        "compiler_version": "amd clang",
        "host_arch": "x86_64",
        "control_word_bits": 64,
        "memory_flags_requested": "mapped portable",
        "hot_reuse_rounds": 2000,
        "devices": [],
        "five_rank_status": status,
        "five_rank_detail": status_detail,
        "qualification": "pass",
        "qualification_detail": "all mandatory cases passed",
        "limits": ["probe scope"],
    }
    for index in range(device_count):
        cases = {}
        for name in BASE_CASES:
            if name == "device_identity":
                result, detail = "pass", "mapped_host_memory_supported=1"
            elif name == "host_system_rmw_u64_add":
                result, detail = "fail", "produced a wrong value (448 instead of 7)"
            else:
                result, detail = "pass", "detail"
            cases[name] = {"result": result, "detail": detail}
        if device_count == 5:
            cases["five_rank_cross_device"] = {"result": "pass", "detail": "rank %d of 5 wrote" % index}
        report["devices"].append({
            "rank": index,
            "pci_bdf": "0000:0%d:00.0" % index,
            "reported_name": "AMD Radeon RX 6800",
            "gcn_arch_name": "gfx1030",
            "total_vram_bytes": 1 << 34,
            "can_map_host_memory": True,
            "cases": cases,
        })
    if qualification is not None:
        report["qualification"] = qualification
    return report


def good_bench():
    return {
        "schema": 1,
        "target": BENCH_TARGET,
        "build_id": "tq-relay-probe-bench",
        "hip_runtime_version_string": "7.2.4",
        "device_ordinal": 0,
        "device_name": "AMD Radeon RX 6800",
        "gcn_arch_name": "gfx1030",
        "rounds": 2000,
        "measurements": [
            {"name": name, "unit": unit,
             "samples": 32 if name == "host_five_way_fold" else 2000,
             "min": 100, "median": 200, "p90": 300, "max": 400}
            for name, unit in BENCH_MEASUREMENTS.items()
        ],
        "limits": ["observations, not gates"],
    }


def expect_invalid(label, mutate):
    value = mutate()
    try:
        if label.startswith("bench:"):
            validate_bench(value)
        else:
            validate_report(value)
    except EvidenceError:
        return
    raise SystemExit("self-test failed: %s was accepted" % label)


def expect_invalid_text(label, text, kind):
    try:
        value = load_strict_text(text)
    except EvidenceError:
        return
    try:
        if kind == "bench":
            validate_bench(value)
        else:
            validate_report(value)
    except EvidenceError:
        return
    raise SystemExit("self-test failed: %s was accepted" % label)


def load_strict_text(text):
    try:
        return json.loads(text, object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
    except EvidenceError:
        raise
    except (ValueError, RecursionError) as error:
        fail("invalid JSON: %s" % error)


def mutation_fuzz():
    """Deterministic structural fuzz over the report and bench validators.

    Every mutation of a valid document must end in a clean outcome: accepted
    as still valid, or rejected with the validator's own EvidenceError
    (never an accidental TypeError/KeyError/RecursionError from a malformed
    tree). The seed is fixed so a failure is reproducible; the count is
    bounded so --self-test stays a fast check.
    """
    import random
    rng = random.Random(20261019)

    def mutate(value):
        kind = type(value)
        if kind is dict and value:
            key = rng.choice(list(value))
            return {k: (mutate(v) if k == key else v) for k, v in value.items()}
        if kind is list and value:
            index = rng.randrange(len(value))
            result = list(value)
            result[index] = mutate(result[index])
            return result
        if kind is bool:
            return rng.choice([True, False])
        if kind is int:
            return rng.choice([0, 1, -1, 2**31 - 1, 2**63 - 1, 2**64, rng.randrange(10**6)])
        if kind is str:
            return rng.choice(["", "x", "\x00", "\ud800", "a" * 200, str(rng.randrange(10**12))])
        return rng.choice([None, 0, "", [], {}])

    documents = (
        ("report", good_report(1), validate_report),
        ("bench", good_bench(), validate_bench),
    )
    for label, document, check in documents:
        for _ in range(500):
            candidate = mutate(document)
            try:
                check(candidate)
            except EvidenceError:
                pass  # clean rejection is the expected outcome
            # Any other exception propagates and fails the self-test.


def self_test():
    validate_report(good_report(1))
    validate_report(good_report(5))
    validate_bench(good_bench())
    validate_pair(good_report(1), good_bench())
    # No-device report (device count 0 or enumeration failure) is a real
    # writer output and must validate.
    validate_report(good_report(0, qualification="fail"))

    def with_case_change(index, name, mutation):
        def mutate():
            report = good_report(1)
            cases = report["devices"][index]["cases"]
            entry = cases[name]
            mutated = dict(entry)
            mutation(mutated)
            cases[name] = mutated
            return report
        return mutate

    expect_invalid("mandatory case fail without qualification fail",
                   with_case_change(0, "cpu_to_gpu_publish", lambda e: e.update({"result": "fail"})))

    def unknown_case():
        report = good_report(1)
        report["devices"][0]["cases"]["invented_case"] = {"result": "pass", "detail": "x"}
        return report
    expect_invalid("unknown case id", unknown_case)

    for nested in (False, True):
        report = good_report(1)
        target = report["devices"][0]["cases"] if nested else report
        target["\ud800"] = 1
        try:
            validate_report(report)
        except EvidenceError as error:
            str(error).encode("utf-8")
        else:
            raise AssertionError("invalid Unicode field name was accepted")

    def wrong_qualification_field():
        return good_report(1, qualification="fail")
    expect_invalid("qualification field disagreeing with case results", wrong_qualification_field)

    def five_without_rank_case():
        report = good_report(5)
        del report["devices"][0]["cases"]["five_rank_cross_device"]
        return report
    expect_invalid("five devices missing the five-rank case", five_without_rank_case)

    def five_rank_not_run():
        report = good_report(5)
        report["five_rank_status"] = "not_run"
        return report
    expect_invalid("five devices with five_rank not_run", five_rank_not_run)

    def five_rank_fail():
        report = good_report(5)
        report["five_rank_status"] = "fail"
        report["qualification"] = "fail"
        report["qualification_detail"] = "a rank failed"
        for device in report["devices"]:
            device["cases"]["five_rank_cross_device"] = {
                "result": "fail",
                "detail": "rank 2 drain failed: an error",
            }
        return report
    validate_report(five_rank_fail())

    def five_rank_disagree():
        report = five_rank_fail()
        report["devices"][1]["cases"]["five_rank_cross_device"]["result"] = "pass"
        return report
    expect_invalid("five-rank per-device record disagreeing with five_rank_status", five_rank_disagree)

    def empty_detail_with_devices():
        report = good_report(1)
        report["five_rank_detail"] = ""
        return report
    expect_invalid("five_rank_detail empty while devices ran", empty_detail_with_devices)

    def no_device_pass():
        return good_report(0, qualification="pass")
    expect_invalid("qualification pass with no devices", no_device_pass)

    def single_device_rank_case():
        report = good_report(1)
        report["devices"][0]["cases"]["five_rank_cross_device"] = {"result": "pass", "detail": "x"}
        return report
    expect_invalid("five-rank case present with one device", single_device_rank_case)

    def wrong_rank():
        report = good_report(1)
        report["devices"][0]["rank"] = 3
        return report
    expect_invalid("device rank not equal to index", wrong_rank)

    def rounds_out_of_range():
        report = good_report(1)
        report["hot_reuse_rounds"] = 1 << 32
        return report
    expect_invalid("report rounds outside the CLI uint32 domain", rounds_out_of_range)

    def malformed_bench(change):
        bench = good_bench()
        change(bench)
        return bench

    expect_invalid("bench:missing required measurements",
                   lambda: malformed_bench(lambda b: b["measurements"].pop()))
    expect_invalid("bench:unknown measurement name",
                   lambda: malformed_bench(lambda b: b["measurements"][0].update({"name": "invented"})))
    expect_invalid("bench:known but incorrect unit",
                   lambda: malformed_bench(lambda b: b["measurements"][0].update({"unit": "bytes_per_second"})))
    expect_invalid("bench:latency samples disagree with rounds",
                   lambda: malformed_bench(lambda b: b["measurements"][0].update({"samples": 1})))
    expect_invalid("bench:fold must have 32 samples",
                   lambda: malformed_bench(lambda b: b["measurements"][-1].update({"samples": 2000})))
    expect_invalid("bench:rounds outside the CLI uint32 domain",
                   lambda: malformed_bench(lambda b: b.update({"rounds": 1 << 32})))

    for field, changed in (("device_ordinal", 1), ("device_name", "another GPU"),
                           ("gcn_arch_name", "gfx1100"), ("rounds", 1),
                           ("hip_runtime_version_string", "another runtime")):
        bench = good_bench()
        bench[field] = changed
        try:
            validate_pair(good_report(1), bench)
        except EvidenceError:
            pass
        else:
            raise SystemExit("self-test failed: mismatched bench %s was accepted" % field)
    try:
        validate_pair(good_report(0, qualification="fail"), good_bench())
    except EvidenceError:
        pass
    else:
        raise SystemExit("self-test failed: bench paired with no devices was accepted")

    def missing_pci():
        report = good_report(1)
        report["devices"][0]["pci_bdf"] = ""
        return report
    expect_invalid("healthy device without pci_bdf", missing_pci)

    def bad_qualification_enum():
        report = good_report(1)
        report["qualification"] = "maybe"
        return report
    expect_invalid("qualification outside the outcome enum", bad_qualification_enum)

    # Raw-text mutations exercise the strict loader itself.
    base_text = json.dumps(good_report(1), indent=1)
    expect_invalid_text("duplicate JSON key",
                        base_text.replace('"schema": 1', '"schema": 1,\n "schema": 1'),
                        "report")
    expect_invalid_text("non-finite number",
                        base_text.replace('"total_vram_bytes": 17179869184',
                                          '"total_vram_bytes": NaN'),
                        "report")
    expect_invalid_text("bool where an integer belongs",
                        base_text.replace('"hot_reuse_rounds": 2000',
                                          '"hot_reuse_rounds": true'),
                        "report")
    expect_invalid_text("trailing data", base_text + " {}", "report")
    expect_invalid_text("unpaired Unicode surrogate",
                        base_text.replace('"amd clang"', '"\\ud800"'), "report")
    expect_invalid_text("JSON nesting exceeds parser depth", "[" * 10000 + "]" * 10000, "report")
    expect_invalid_text("oversized integer", base_text.replace('"schema": 1', '"schema": ' + "9" * 5000),
                        "report")
    bench_text = json.dumps(good_bench())
    expect_invalid_text("bench min above median",
                        bench_text.replace('"min": 100', '"min": 250'),
                        "bench")
    expect_invalid_text("bench unknown unit",
                        bench_text.replace('"unit": "ns"', '"unit": "furlongs"'),
                        "bench")
    expect_invalid_text("bench duplicate measurement name",
                        bench_text.replace('"device_produce_to_host_ready"',
                                           '"host_publish_to_device_ack"'), "bench")
    mutation_fuzz()
    print("PASS: validator self-test")


def main():
    args = sys.argv[1:]
    if args == ["--self-test"]:
        self_test()
        return 0
    if not 1 <= len(args) <= 2:
        print("usage: validate-report.py REPORT [BENCH]", file=sys.stderr)
        print("       validate-report.py --self-test", file=sys.stderr)
        return 2
    try:
        report = load_strict(args[0])
        validate_report(report)
        print("valid: %s" % args[0])
    except (EvidenceError, OSError) as error:
        print("invalid: %s: %s" % (args[0], error))
        return 1
    if len(args) == 2:
        try:
            bench = load_strict(args[1])
            validate_bench(bench)
            validate_pair(report, bench)
            print("valid: %s" % args[1])
        except (EvidenceError, OSError) as error:
            print("invalid: %s: %s" % (args[1], error))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
