#!/usr/bin/env python3
# Tiny deterministic Llama-arch GGUF generator for the NKV union-binder driver.
#
# Real production loader requirements, verified against this tree:
#   - src/models/llama.cpp load_arch_hparams: llama.vocab_size (uint32) sets
#     n_vocab; llama.attention.layer_norm_rms_epsilon is REQUIRED.
#   - src/llama-model.cpp load: llama.context_length / embedding_length /
#     block_count / feed_forward_length / attention.head_count required;
#     head_count_kv optional (defaults to head_count); rope dims default to
#     n_embd / n_head.
#   - src/llama-vocab.cpp:1939: tokenizer.ggml.model = "no_vocab" =>
#     LLAMA_VOCAB_TYPE_NONE (token ids are fed directly to llama_decode; the
#     driver never tokenizes text).
#   - Token type constraints of the reader-view path (src/llama-context.cpp
#     attention_views_unsupported_reason): text RoPE llama arch, K/V cache
#     GGML_TYPE_F32 (type requested via llama_context_params.type_k/type_v).
#
# Deterministic: fixed seed, pure numpy float32 writes.
import pathlib
import sys

import numpy as np

WORK = pathlib.Path(__file__).resolve().parent.parent.parent
GGUF_PY = WORK / "lead-fa-lower-20261030" / "source" / "gguf-py"
if GGUF_PY.is_dir():
    sys.path.insert(0, str(GGUF_PY))
else:
    sys.stderr.write(f"note: {GGUF_PY} not found, relying on ambient gguf module\n")

from gguf import GGUFWriter  # noqa: E402  (gguf-py from the source tree)

VOCAB = 64
EMBD = 64
FF = 128
LAYERS = 2
HEADS = 4
KV_HEADS = 4
HEAD_DIM = EMBD // HEADS  # 16 -> n_rot 16, text RoPE (LLAMA_ROPE_TYPE_NORM)
CTX = 512  # 512 cells so get_n_kv() spans two 256-wide stock buckets
SEED = 20261030

OUT = pathlib.Path(__file__).resolve().parent / "nkv-driver-tiny.gguf"


def gen() -> None:
    rng = np.random.default_rng(SEED)

    def rnd(rows, cols, scale):
        return rng.normal(0.0, scale, size=(rows, cols)).astype(np.float32)

    writer = GGUFWriter(str(OUT), "llama")
    writer.add_string("general.name", "nkv-union-driver-tiny")
    writer.add_uint32("llama.vocab_size", VOCAB)
    writer.add_uint32("llama.context_length", CTX)
    writer.add_uint32("llama.embedding_length", EMBD)
    writer.add_uint32("llama.block_count", LAYERS)
    writer.add_uint32("llama.feed_forward_length", FF)
    writer.add_uint32("llama.attention.head_count", HEADS)
    writer.add_uint32("llama.attention.head_count_kv", KV_HEADS)
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-5)
    writer.add_float32("llama.rope.freq_base", 10000.0)
    writer.add_string("tokenizer.ggml.model", "no_vocab")

    # numpy shape is (ne1, ne0): gguf-py writes shape reversed (gguf_writer.py:268)
    tensors = [
        ("token_embd.weight", rnd(VOCAB, EMBD, 0.05)),
        ("output_norm.weight", np.ones(EMBD, dtype=np.float32)),
        ("output.weight", rnd(VOCAB, EMBD, 0.05)),
    ]
    for i in range(LAYERS):
        tensors += [
            (f"blk.{i}.attn_norm.weight", np.ones(EMBD, dtype=np.float32)),
            (f"blk.{i}.attn_q.weight", rnd(HEADS * HEAD_DIM, EMBD, 0.02)),
            (f"blk.{i}.attn_k.weight", rnd(KV_HEADS * HEAD_DIM, EMBD, 0.02)),
            (f"blk.{i}.attn_v.weight", rnd(KV_HEADS * HEAD_DIM, EMBD, 0.02)),
            (f"blk.{i}.attn_output.weight", rnd(EMBD, HEADS * HEAD_DIM, 0.02)),
            (f"blk.{i}.ffn_norm.weight", np.ones(EMBD, dtype=np.float32)),
            (f"blk.{i}.ffn_gate.weight", rnd(FF, EMBD, 0.02)),
            (f"blk.{i}.ffn_up.weight", rnd(FF, EMBD, 0.02)),
            (f"blk.{i}.ffn_down.weight", rnd(EMBD, FF, 0.02)),

        ]

    for name, arr in tensors:
        writer.add_tensor(name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    gen()
