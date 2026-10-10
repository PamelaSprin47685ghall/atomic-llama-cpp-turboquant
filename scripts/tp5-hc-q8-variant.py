#!/usr/bin/env python3
"""Create an explicit HC-Q8 derivative, preserving every non-HC tensor byte.

Never edits source GGUFs. Only floating HC down/up/inject matrices are changed.
This changes model arithmetic and needs separate quality/performance tests.
Uses the current ggml CPU quantizer, not an independent Q8 implementation.
"""
import argparse
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct

spec = importlib.util.spec_from_file_location("gguf_directory", Path(__file__).with_name("tp5-gguf-metadata.py"))
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)

PATTERN = re.compile(r"^(?:blk\.\d+\.hc_(?:attn|ffn)_(?:down|up|inject)|output_hc_(?:down|up))\.weight$")


def digest_range(f, offset, length):
    f.seek(offset)
    h = hashlib.sha256()
    while length:
        b = f.read(min(length, 8 << 20))
        if not b:
            raise ValueError("Short payload while hashing")
        h.update(b)
        length -= len(b)
    return h.hexdigest()


def copy_range(src, dst, length):
    while length:
        try:
            n = os.copy_file_range(src.fileno(), dst.fileno(), min(length, 64 << 20))
        except OSError:
            b = src.read(min(length, 8 << 20))
            if not b:
                raise ValueError("Short tensor payload")
            n = dst.write(b)
        if n <= 0:
            raise ValueError("Incomplete tensor copy")
        length -= n


def quantize(helper, raw, dtype, shape):
    n = math.prod(shape)
    if dtype not in (0, 1, 30) or shape[0] % 32 or len(raw) != n * (4 if dtype == 0 else 2):
        raise ValueError("Invalid floating HC tensor")
    out = ctypes.create_string_buffer(n // 32 * 34)
    max_abs, relative_l2 = ctypes.c_double(), ctypes.c_double()
    result = helper.tp5_hc_quantize(raw, dtype, n // shape[0], shape[0], out,
                                    ctypes.byref(max_abs), ctypes.byref(relative_l2))
    if result:
        raise ValueError("Native finite/shape/Q8 size check failed")
    return out.raw, {"max_abs": max_abs.value, "relative_l2": relative_l2.value}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--lib", type=Path, required=True)
    ap.add_argument("--helper", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()
    source = a.source.resolve()
    match = re.fullmatch(r"(.*)-00001-of-(\d{5})\.gguf", source.name)
    if not match:
        ap.error("Expected first shard of a split GGUF")
    paths = sorted(source.parent.glob(match.group(1) + "-?????-of-" + match.group(2) + ".gguf"))
    if len(paths) != int(match.group(2)):
        ap.error("Incomplete shard set")
    a.out_dir.mkdir(parents=True, exist_ok=False)
    lib = ctypes.CDLL(str(a.lib.resolve()))
    helper = ctypes.CDLL(str(a.helper.resolve()))
    helper.tp5_hc_quantize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
                                      ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    helper.tp5_hc_quantize.restype = ctypes.c_int
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                       ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    lib.ggml_row_size.restype = ctypes.c_size_t
    report = {"status": "IN_PROGRESS", "arithmetic_changed": True,
              "scope": "Only floating HC down/up/inject weights converted to Q8_0; source files are untouched.",
              "files": [], "changed_tensors": [], "unchanged_tensors": []}
    def save():
        a.report.write_text(json.dumps(report, indent=2) + "\n")
    save()
    with open("/var/tmp/tp5-gpu.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for src_path in paths:
            info = reader.directory(src_path)
            align = int(info['metadata'].get('general.alignment', 32))
            selected = {t['name'] for t in info['tensors'] if PATTERN.fullmatch(t['name']) and t['raw_type'] in (0, 1, 30)}
            # Keep the original split suffix and a single consistent new prefix.
            suffix = re.search(r"(-\d{5}-of-\d{5}\.gguf)$", src_path.name).group(1)
            dst_path = a.out_dir / ("Q2-HCQ8" + suffix)
            with src_path.open("rb", buffering=0) as src:
                header = bytearray(src.read(info['payload_offset']))
                cursor = 0
                for t in info['tensors']:
                    old_type = 47 if t['raw_type'] == 42 else t['raw_type']
                    old_bytes = lib.ggml_row_size(old_type, t['shape'][0]) * math.prod(t['shape'][1:])
                    if (old_bytes + align - 1) // align * align != t['storage_extent_bytes']:
                        raise ValueError(f"Unsupported source extent: {t['name']}")
                    struct.pack_into("<Q", header, t['header_offset_position'], cursor)
                    if t['name'] in selected:
                        struct.pack_into("<I", header, t['header_type_position'], 8)
                        n = t['elements'] // 32 * 34
                    else:
                        # Extent includes the original alignment padding.
                        n = t['storage_extent_bytes']
                    t['new_offset'] = cursor
                    t['new_extent'] = (n + align - 1) // align * align
                    cursor += t['new_extent']
                with dst_path.open("wb", buffering=0) as dst:
                    dst.write(header)
                    for t in info['tensors']:
                        if dst.tell() != info['payload_offset'] + t['new_offset']:
                            raise ValueError("Non-contiguous output tensor directory")
                        src.seek(info['payload_offset'] + t['offset'])
                        if t['name'] in selected:
                            elem_size = 4 if t['raw_type'] == 0 else 2
                            raw = src.read(t['elements'] * elem_size)
                            encoded, stats = quantize(helper, raw, t['raw_type'], t['shape'])
                            dst.write(encoded)
                            dst.write(b'\0' * (t['new_extent'] - len(encoded)))
                            report['changed_tensors'].append({'name': t['name'], 'source_type': t['raw_type'], 'new_type': 8,
                                'source_sha256': hashlib.sha256(raw).hexdigest(), 'new_sha256': hashlib.sha256(encoded).hexdigest(),
                                'shape': t['shape'], **stats})
                        else:
                            copy_range(src, dst, t['storage_extent_bytes'])
                # Verify every unchanged tensor's raw bytes, not just its shape.
                with dst_path.open("rb", buffering=0) as dst:
                    for t in info['tensors']:
                        if t['name'] in selected:
                            continue
                        old = digest_range(src, info['payload_offset'] + t['offset'], t['storage_extent_bytes'])
                        new = digest_range(dst, info['payload_offset'] + t['new_offset'], t['storage_extent_bytes'])
                        if old != new:
                            raise ValueError(f"Unchanged tensor mismatch: {t['name']}")
                        report['unchanged_tensors'].append({'name': t['name'], 'sha256': old, 'bytes': t['storage_extent_bytes']})
            reader.directory(dst_path)
            report['files'].append({'source': str(src_path), 'output': str(dst_path.resolve()),
                                    'source_bytes': src_path.stat().st_size, 'output_bytes': dst_path.stat().st_size})
            save()
            print(f"Verified {dst_path.name}: converted={len(selected)} unchanged={len(info['tensors'])-len(selected)}", flush=True)
    report['status'] = 'CONVERTED_AND_UNCHANGED_TENSOR_BYTES_VERIFIED_NOT_QUALITY_QUALIFIED'
    save()


if __name__ == '__main__':
    main()
