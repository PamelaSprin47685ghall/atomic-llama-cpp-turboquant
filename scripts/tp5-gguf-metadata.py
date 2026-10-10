#!/usr/bin/env python3
"""Read only GGUF metadata/tensor directory, never map or modify model payload.

Raw type IDs are retained (important for cross-fork Q2_0 numbering). Tensor
storage extents include at most alignment padding and are not VRAM traffic.
"""
import argparse
import json
import math
from pathlib import Path
import struct


def directory(path):
    with path.open("rb") as f:
        def scalar(fmt):
            size = struct.calcsize("<" + fmt)
            b = f.read(size)
            if len(b) != size:
                raise ValueError("Truncated GGUF")
            return struct.unpack("<" + fmt, b)[0]

        def string():
            n = scalar("Q")
            if n > 100_000_000:
                raise ValueError("Unreasonable metadata string size")
            data = f.read(n)
            if len(data) != n:
                raise ValueError("Truncated metadata string")
            return data.decode("utf-8", errors="replace")

        def value(t, retain=True):
            formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
            if t in formats:
                return scalar(formats[t])
            if t == 8:
                return string()
            if t == 9:
                sub, n = scalar("I"), scalar("Q")
                if n > 10_000_000:
                    raise ValueError("Unreasonable metadata array size")
                vals = []
                for _ in range(n):
                    v = value(sub, retain and n < 1024)
                    if retain and n < 1024:
                        vals.append(v)
                return vals if retain and n < 1024 else {"array_type": sub, "count": n}
            raise ValueError(f"Unknown GGUF metadata type {t}")

        if f.read(4) != b"GGUF":
            raise ValueError(f"Not GGUF: {path}")
        version, nt, nk = scalar("I"), scalar("Q"), scalar("Q")
        if version not in (2, 3):
            raise ValueError(f"Unsupported version: {version}")
        meta = {}
        for _ in range(nk):
            key, t = string(), scalar("I")
            meta[key] = value(t, not key.startswith("tokenizer.ggml."))
        tensors = []
        for _ in range(nt):
            name, nd = string(), scalar("I")
            if nd > 4:
                raise ValueError(f"Invalid dims {nd}")
            shape = [scalar("Q") for _ in range(nd)]
            type_position = f.tell()
            raw_type = scalar("I")
            offset_position = f.tell()
            tensors.append({"name": name, "shape": shape, "elements": math.prod(shape), "raw_type": raw_type, "offset": scalar("Q"),
                            "header_type_position": type_position, "header_offset_position": offset_position})
        align = int(meta.get("general.alignment", 32))
        start = (f.tell() + align - 1) // align * align
        ordered = sorted(tensors, key=lambda t: t["offset"])
        for i, t in enumerate(ordered):
            end = ordered[i + 1]["offset"] if i + 1 < len(ordered) else path.stat().st_size - start
            t["storage_extent_bytes"] = end - t["offset"]
        return {"path": str(path.resolve()), "file_bytes": path.stat().st_size, "version": version,
                "payload_offset": start, "metadata": meta, "tensors": tensors}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    records = [directory(path) for path in a.files]
    a.out.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    for r in records:
        print(r["path"], "tensors", len(r["tensors"]), "header_bytes", r["payload_offset"])
