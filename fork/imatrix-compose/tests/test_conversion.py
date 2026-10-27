#!/usr/bin/env python3
"""Exercise the real converter and inspect its GGUF output without NumPy."""

import argparse
import json
from pathlib import Path
import resource
import signal
import struct
import subprocess
import tempfile


def calibration(*, decoded=9, total=9, complete=1, positions=4, expert_counts=(2, 1), hidden_counts=(2, 1),
                expert_sum2=(5.0, 8.0, 9.0, 4.0), hidden_sum2=(4.0, 8.0, 12.0, 3.0, 6.0, 9.0)):
    output = bytearray(b"WXQCAL03")
    output += struct.pack("<6I3QB7x", 1, 0, 0, 0, 1, 7, decoded, total, 2, complete)
    output += struct.pack("<4I3Q", 3, 2, 2, 3, positions, 9, 4)
    output += struct.pack("<2Q2d", 9, 6, 10, 20)
    output += struct.pack("<2Q", *expert_counts)
    first_moments = (-1.0, 2.0, 3.0, -2.0)
    first_moments = tuple(value if expert_counts[i // 2] else 0.0 for i, value in enumerate(first_moments))
    output += struct.pack("<4d4d", *first_moments, *expert_sum2)
    output += struct.pack("<2Q6d", *hidden_counts, *hidden_sum2)
    output += struct.pack("<4d", 10, -3, -3, 20)
    return bytes(output)


def read_gguf(path):
    data = path.read_bytes()
    position = 0

    def scalar(fmt):
        nonlocal position
        count = struct.calcsize("<" + fmt)
        if position + count > len(data):
            raise AssertionError("truncated GGUF")
        result = struct.unpack_from("<" + fmt, data, position)[0]
        position += count
        return result

    def string():
        nonlocal position
        size = scalar("Q")
        if size > 1024 * 1024 or position + size > len(data):
            raise AssertionError("invalid GGUF string")
        value = data[position:position + size].decode("utf-8")
        position += size
        return value

    def value(kind):
        formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if kind in formats:
            return scalar(formats[kind])
        if kind == 8:
            return string()
        if kind == 9:
            subtype, count = scalar("I"), scalar("Q")
            if count > 65536 or subtype == 9:
                raise AssertionError("invalid GGUF array")
            return [value(subtype) for _ in range(count)]
        raise AssertionError("invalid GGUF value type")

    if data[:4] != b"GGUF":
        raise AssertionError("not GGUF")
    position = 4
    if scalar("I") != 3:
        raise AssertionError("unexpected GGUF version")
    tensors, pairs = scalar("Q"), scalar("Q")
    if tensors > 100 or pairs > 100:
        raise AssertionError("unexpected test output dimensions")
    metadata, types = {}, {}
    for _ in range(pairs):
        key, kind = string(), scalar("I")
        if key in metadata:
            raise AssertionError("duplicate GGUF key")
        metadata[key], types[key] = value(kind), kind
    directory = {}
    for _ in range(tensors):
        name, dimensions = string(), scalar("I")
        if name in directory or not 1 <= dimensions <= 4:
            raise AssertionError("invalid tensor directory")
        shape = tuple(scalar("Q") for _ in range(dimensions))
        kind, offset = scalar("I"), scalar("Q")
        directory[name] = (shape, kind, offset)
    alignment = metadata.get("general.alignment", 32)
    start = (position + alignment - 1) // alignment * alignment
    output = {}
    for name, (shape, kind, offset) in directory.items():
        if kind != 0 or offset % alignment:
            raise AssertionError("tensor is not aligned F32")
        count = 1
        for dimension in shape:
            count *= dimension
        if count > 100000 or start + offset + count * 4 > len(data):
            raise AssertionError("invalid tensor extent")
        output[name] = (shape, struct.unpack_from(f"<{count}f", data, start + offset))
    return metadata, types, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    binary = str(args.binary.resolve())
    with tempfile.TemporaryDirectory(prefix="tq-imatrix-test-") as temporary:
        root = Path(temporary)
        source, output = root / "source.bin", root / "imatrix.gguf"
        source.write_bytes(calibration())
        original = source.read_bytes()
        base = ["--input", str(source), "--output", str(output), "--dataset", "format-fixture"]

        def run(extra=(), expected=0, arguments=None, **settings):
            result = subprocess.run([binary, *(base if arguments is None else arguments), *extra],
                                    capture_output=True, text=True, timeout=30, **settings)
            if result.returncode != expected:
                raise AssertionError((result.returncode, expected, result.stdout, result.stderr))
            if list(root.glob(".tq-imatrix-*.tmp")):
                raise AssertionError("private files remain")
            return result

        run(arguments=["--help"])
        run(arguments=[], expected=2)
        for extra in (("--bad",), ("--unseen", "silent"), ("--max-output-bytes", "0"),
                      ("--max-layer-bytes", "-1"), ("--max-input-bytes", "1x"),
                      ("--max-output-bytes", "18446744073709551616"), ("--dataset", "line\nbreak")):
            run(extra, 2)
        summary = json.loads(run().stdout)
        metadata, types, tensors = read_gguf(output)
        if summary["layers"] != 1 or summary["tensors"] != 6 or summary["output_bytes"] != output.stat().st_size:
            raise AssertionError("summary disagrees with output")
        if types["imatrix.chunk_count"] != 4 or metadata["imatrix.chunk_count"] != 9 or metadata["imatrix.chunk_size"] != 1:
            raise AssertionError("imatrix chunk metadata changed")
        if metadata["imatrix.datasets"] != ["format-fixture"] or metadata["tq.calibration.partial_accepted"]:
            raise AssertionError("dataset or selected-range metadata changed")
        for projection in ("gate", "up"):
            prefix = f"blk.3.ffn_{projection}_exps.weight"
            if tensors[prefix + ".in_sum2"] != ((2, 2), (5.0, 8.0, 9.0, 4.0)):
                raise AssertionError("input square sums changed")
            if tensors[prefix + ".counts"] != ((1, 2), (2.0, 1.0)):
                raise AssertionError("per-expert counts changed")
        if tensors["blk.3.ffn_down_exps.weight.in_sum2"] != ((3, 2), (4.0, 8.0, 12.0, 3.0, 6.0, 9.0)):
            raise AssertionError("hidden square sums changed")
        saved = output.read_bytes()
        run(expected=1)
        if output.read_bytes() != saved or source.read_bytes() != original:
            raise AssertionError("existing file changed")
        output.unlink()
        output.symlink_to(root / "absent")
        run(expected=1)
        if not output.is_symlink():
            raise AssertionError("output symlink changed")
        output.unlink()

        for blob in (original[:63], original[:-1], original + b"tail",
                     calibration(expert_sum2=(float("nan"), 8, 9, 4)),
                     calibration(expert_sum2=(-1, 8, 9, 4)),
                     calibration(expert_sum2=(1e300, 8, 9, 4)),
                     calibration(expert_sum2=(1e-300, 8, 9, 4)),
                     calibration(decoded=16777216, total=16777216, positions=16777216,
                                 expert_counts=(16777216, 1), expert_sum2=(1e-40, 8, 9, 4)),
                     calibration(decoded=16777217, total=16777217, positions=16777217, expert_counts=(16777217, 1)),
                     calibration(decoded=2147483648, total=2147483648)):
            source.write_bytes(blob)
            run(expected=2)
            if output.exists():
                raise AssertionError("invalid calibration published output")
        source.write_bytes(original)
        for extra in (("--max-layer-bytes", str(len(original) - 65)), ("--max-input-bytes", str(len(original) - 1)), ("--max-output-bytes", "1")):
            run(extra, 2)
        for values, reason in (({"complete": 0}, "collector_reported_incomplete"),
                               ({"total": 10}, "decoded_below_declared_total"),
                               ({"total": 8}, "decoded_above_declared_total"),
                               ({"total": 0}, "declared_total_is_zero")):
            source.write_bytes(calibration(**values))
            run(expected=2)
            run(("--allow-partial",))
            metadata, _, _ = read_gguf(output)
            if reason not in metadata["tq.calibration.partial_reasons"] or not metadata["tq.calibration.partial_accepted"]:
                raise AssertionError("partial range was concealed")
            output.unlink()

        source.write_bytes(calibration(expert_counts=(2, 0), hidden_counts=(2, 0),
                                      expert_sum2=(5, 8, 0, 0), hidden_sum2=(4, 8, 12, 0, 0, 0)))
        run(expected=2)
        run(("--unseen", "global-mean"))
        metadata, _, tensors = read_gguf(output)
        if tensors["blk.3.ffn_gate_exps.weight.in_sum2"][1] != (5.0, 8.0, 2.5, 4.0):
            raise AssertionError("global mean is not weighted by observed counts")
        if tensors["blk.3.ffn_down_exps.weight.in_sum2"][1] != (4.0, 8.0, 12.0, 2.0, 4.0, 6.0):
            raise AssertionError("hidden mean is wrong")
        if metadata["tq.calibration.imputed_input_ids"] != [(3 << 32) | 1] or metadata["tq.calibration.imputed_rows_are_measurements"]:
            raise AssertionError("imputation provenance is wrong")
        output.unlink()

        source.write_bytes(original)
        def limit_file_size():
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (256, 256))
        run(expected=1, preexec_fn=limit_file_size)
        if output.exists():
            raise AssertionError("write failure published output")

        first = subprocess.Popen([binary, *base], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        second = subprocess.Popen([binary, *base], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        first.communicate(timeout=30)
        second.communicate(timeout=30)
        if sorted((first.returncode, second.returncode)) != [0, 1]:
            raise AssertionError("no-clobber race did not have exactly one winner")
        read_gguf(output)
        output.unlink()
        if Path("/dev/full").exists():
            with open("/dev/full", "wb") as sink:
                result = subprocess.run([binary, *base], stdout=sink, stderr=subprocess.PIPE, timeout=30)
            if result.returncode != 1 or not output.exists():
                raise AssertionError("summary failure contract changed")
            read_gguf(output)
        if source.read_bytes() != original or list(root.glob(".tq-imatrix-*.tmp")):
            raise AssertionError("source or private-file lifecycle changed")
    print("imatrix conversion: all checks passed")


if __name__ == "__main__":
    main()
