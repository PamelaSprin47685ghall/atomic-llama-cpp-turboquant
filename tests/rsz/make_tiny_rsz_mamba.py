#!/usr/bin/env python3
"""Write a tiny deterministic Mamba GGUF for the rs_z union-binder CPU test.

The model is a real (tiny) LLM_ARCH_MAMBA checkpoint: two blocks, F32 weights,
a six-token llama tokenizer. It is small enough (< 32 KiB) to load in a unit
test and it exercises the recurrent-state graph (`build_rs`) that carries the
`union_rs_zero_index` / `union_rs_zero_row` / `union_rs_copy_dst` roots.

No third-party modules: the GGUF container is written directly so the test
does not depend on numpy or gguf-py being installed.

usage: python3 tests/rsz/make_tiny_rsz_mamba.py out.gguf
"""

import random
import struct
import sys

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# GGUF metadata value types
T_UINT32 = 4
T_FLOAT32 = 6
T_BOOL = 7
T_STRING = 8
T_ARRAY = 9

# ggml tensor types
GGML_TYPE_F32 = 0

# GGUF token types (llama_vocab)
TOKEN_TYPE_NORMAL = 1
TOKEN_TYPE_UNKNOWN = 2
TOKEN_TYPE_CONTROL = 3
TOKEN_TYPE_BYTE = 6

ALIGNMENT = 32


def align(n, a=ALIGNMENT):
    return (n + a - 1) // a * a


class Writer:
    def __init__(self):
        self.kv = []

    def add(self, key, value):
        self.kv.append((key, value))

    def _string(self, s):
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def _value(self, v):
        if isinstance(v, bool):
            return struct.pack("<I", T_BOOL) + struct.pack("<B", 1 if v else 0)
        if isinstance(v, int):
            return struct.pack("<I", T_UINT32) + struct.pack("<I", v)
        if isinstance(v, float):
            return struct.pack("<I", T_FLOAT32) + struct.pack("<f", v)
        if isinstance(v, str):
            return struct.pack("<I", T_STRING) + self._string(v)
        if isinstance(v, list):
            if not v:
                raise ValueError("empty arrays are not used here")
            if isinstance(v[0], str):
                body = b"".join(self._string(x) for x in v)
                return struct.pack("<I", T_ARRAY) + struct.pack("<I", T_STRING) + struct.pack("<Q", len(v)) + body
            if isinstance(v[0], float):
                body = b"".join(struct.pack("<f", x) for x in v)
                return struct.pack("<I", T_ARRAY) + struct.pack("<I", T_FLOAT32) + struct.pack("<Q", len(v)) + body
            if isinstance(v[0], int):
                body = b"".join(struct.pack("<i", x) for x in v)
                return struct.pack("<I", T_ARRAY) + struct.pack("<I", 5) + struct.pack("<Q", len(v)) + body
        raise TypeError(f"unsupported metadata value: {type(v)}")

    def write(self, path, tensors):
        # tensors: list of (name, dims, data bytes); dims are ggml order (ne0 first)
        offsets = []
        off = 0
        for _, _, data in tensors:
            off = align(off)
            offsets.append(off)
            off += len(data)


        with open(path, "wb") as f:
            f.write(GGUF_MAGIC)
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(tensors)))
            f.write(struct.pack("<Q", len(self.kv)))
            for key, value in self.kv:
                f.write(self._string(key))
                f.write(self._value(value))
            for (name, dims, _), tensor_off in zip(tensors, offsets):
                f.write(self._string(name))
                f.write(struct.pack("<I", len(dims)))
                for d in dims:
                    f.write(struct.pack("<Q", d))
                f.write(struct.pack("<I", GGML_TYPE_F32))
                f.write(struct.pack("<Q", tensor_off))
            # GGUF tensor data starts at the aligned position after the tensor
            # infos, not at the total tensor-data size. The old code used the
            # data size here, so a spec-compliant loader read the zero padding
            # for the leading tensors (token_embd/output_norm/output) and the
            # tiny model produced all-zero logits.
            data_start = align(f.tell())
            f.write(b"\x00" * (data_start - f.tell()))
            for (_, _, data), tensor_off in zip(tensors, offsets):
                f.write(b"\x00" * (tensor_off - (f.tell() - data_start)))
                f.write(data)
            f.write(b"\x00" * (align(f.tell()) - f.tell()))


def f32_tensor(rng, dims, scale=0.1, value=None):
    n = 1
    for d in dims:
        n *= d
    if value is None:
        vals = [rng.gauss(0.0, scale) for _ in range(n)]
    else:
        vals = [value] * n
    return struct.pack("<%df" % n, *vals)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = sys.argv[1]

    n_embd = 8
    n_layer = 2
    d_inner = 2 * n_embd
    d_state = 4
    dt_rank = 2
    d_conv = 4

    rng = random.Random(20261103)

    w = Writer()
    w.add("general.architecture", "mamba")
    w.add("general.name", "tiny-rsz-mamba")
    w.add("mamba.context_length", 64)
    w.add("mamba.embedding_length", n_embd)
    w.add("mamba.block_count", n_layer)
    w.add("mamba.attention.layer_norm_rms_epsilon", 1e-5)
    w.add("mamba.ssm.conv_kernel", d_conv)
    w.add("mamba.ssm.inner_size", d_inner)
    w.add("mamba.ssm.state_size", d_state)
    w.add("mamba.ssm.time_step_rank", dt_rank)
    w.add("mamba.ssm.dt_b_c_rms", False)

    # A llama/SPM vocabulary needs the 256 byte-fallback tokens for the
    # tokenizer to accept arbitrary text; the test itself feeds token ids
    # directly, but this keeps the checkpoint usable by stock tools too.
    tokens = ["<unk>", "<s>", "</s>"]
    token_types = [TOKEN_TYPE_UNKNOWN, TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL]
    tokens += ["<0x%02X>" % i for i in range(256)]
    token_types += [TOKEN_TYPE_BYTE] * 256
    tokens += ["a", "b", "c"]
    token_types += [TOKEN_TYPE_NORMAL] * 3
    n_vocab = len(tokens)
    token_a = tokens.index("a")
    w.add("tokenizer.ggml.model", "llama")
    w.add("tokenizer.ggml.tokens", tokens)
    w.add("tokenizer.ggml.scores", [0.0] * len(tokens))
    w.add("tokenizer.ggml.token_type", token_types)

    tensors = []

    def add(name, dims, scale=0.1, value=None):
        tensors.append((name, list(dims), f32_tensor(rng, dims, scale, value)))

    add("token_embd.weight", (n_embd, n_vocab), scale=0.5)
    add("output_norm.weight", (n_embd,), value=1.0)
    add("output.weight", (n_embd, n_vocab), scale=0.5)

    for il in range(n_layer):
        p = f"blk.{il}."
        add(p + "attn_norm.weight", (n_embd,), value=1.0)
        add(p + "ssm_in.weight", (n_embd, 2 * d_inner), scale=0.1)
        add(p + "ssm_conv1d.weight", (d_conv, d_inner), scale=0.2)
        add(p + "ssm_conv1d.bias", (d_inner,), value=0.0)
        add(p + "ssm_x.weight", (d_inner, dt_rank + 2 * d_state), scale=0.1)
        add(p + "ssm_dt.weight", (dt_rank, d_inner), scale=0.1)
        add(p + "ssm_dt.bias", (d_inner,), value=0.1)
        # A must be negative for a stable scan; D is a positive skip.
        a = [-rng.uniform(0.5, 2.0) for _ in range(d_state * d_inner)]
        tensors.append((p + "ssm_a", [d_state, d_inner], struct.pack("<%df" % len(a), *a)))
        add(p + "ssm_d", (d_inner,), value=1.0)
        add(p + "ssm_out.weight", (d_inner, n_embd), scale=0.1)

    w.write(out, tensors)
    print(f"wrote {out}: arch=mamba layers={n_layer} n_embd={n_embd} d_inner={d_inner} "
          f"d_state={d_state} dt_rank={dt_rank} d_conv={d_conv} vocab={n_vocab} tensors={len(tensors)}")
    print(f"token ids: a={token_a} b={token_a + 1} c={token_a + 2} n_vocab={n_vocab}")


if __name__ == "__main__":
    main()
