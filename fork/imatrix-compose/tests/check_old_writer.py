#!/usr/bin/env python3
"""Check the unchanged legacy writer against the real installed converter."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from test_conversion import read_gguf


def sha256_argument(value):
    if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise argparse.ArgumentTypeError("expected a 64-character SHA-256 digest recorded at producer build time")
    return value.lower()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--writer-revision", required=True, help="old commit used to build the fixture producer")
    parser.add_argument("--writer-source-sha256", type=sha256_argument, required=True)
    parser.add_argument("--writer-header-sha256", type=sha256_argument, required=True)
    args = parser.parse_args()
    producer = args.producer.resolve(strict=True)
    converter = args.converter.resolve(strict=True)
    producer_sha256 = file_sha256(producer)
    converter_sha256 = file_sha256(converter)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = args.output_dir.resolve() / "old-writer.bin"
    output = args.output_dir.resolve() / "imatrix.gguf"
    producer_command = [str(producer), str(source)]
    producer_result = subprocess.run(producer_command, text=True, capture_output=True, check=True, timeout=30)
    source_bytes = source.read_bytes()
    converter_command = [str(converter), "--input", str(source), "--output", str(output),
                         "--dataset", "unchanged-old-writer-format-fixture"]
    result = subprocess.run(converter_command, text=True, capture_output=True, timeout=30, check=True)
    metadata, _, tensors = read_gguf(output)
    summary = json.loads(result.stdout)
    if source.read_bytes() != source_bytes or summary["layers"] != 2 or len(tensors) != 12:
        raise AssertionError("legacy source identity or layer count changed")
    for layer in (3, 7):
        for projection in ("gate", "up"):
            name = f"blk.{layer}.ffn_{projection}_exps.weight"
            if tensors[name + ".in_sum2"] != ((2, 2), (5.0, 8.0, 9.0, 4.0)):
                raise AssertionError("legacy expert sums changed")
            if tensors[name + ".counts"] != ((1, 2), (2.0, 1.0)):
                raise AssertionError("legacy counts changed")
        if tensors[f"blk.{layer}.ffn_down_exps.weight.in_sum2"] != ((3, 2), (4.0, 8.0, 12.0, 3.0, 6.0, 9.0)):
            raise AssertionError("legacy hidden sums changed")
        if tensors[f"blk.{layer}.ffn_down_exps.weight.counts"] != ((1, 2), (2.0, 1.0)):
            raise AssertionError("legacy hidden counts changed")
    if (metadata["tq.calibration.flags"] != 7 or metadata["tq.calibration.partial_accepted"]
            or metadata["imatrix.chunk_count"] != 9 or metadata["imatrix.chunk_size"] != 1):
        raise AssertionError("legacy header metadata changed")
    if file_sha256(producer) != producer_sha256 or file_sha256(converter) != converter_sha256:
        raise AssertionError("producer or converter binary changed during qualification")
    print(json.dumps({"pass": True, "scope": "unchanged writer format interoperability; no model quality claim",
                      "producer_build_inputs": {
                          "provenance": "caller-supplied revision and source digests recorded at build time",
                          "revision": args.writer_revision,
                          "sources": {"wanxiangqi/lib/calib-dump.cpp": args.writer_source_sha256,
                                      "wanxiangqi/lib/calib-dump.h": args.writer_header_sha256}},
                      "producer": {"command": producer_command, "sha256": producer_sha256,
                                   "stdout": producer_result.stdout, "stderr": producer_result.stderr},
                      "converter": {"command": converter_command, "sha256": converter_sha256,
                                    "stderr": result.stderr},
                      "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                      "output_sha256": file_sha256(output), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
