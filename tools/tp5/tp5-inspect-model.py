#!/usr/bin/env python3
# TP5 manifest generator: reads GGUF headers only (no weight data, no GPU, no model loader).
#
# Usage:
#   tools/tp5/tp5-inspect-model.py --model <dir-or-file> [--ranks 5] [--context 131072]
#                                  [--kv-dtype f16|f32|q8_0|turbo4] [--output tp5-manifest.json]
#
# Acceptance contract (TP5.md T01):
#   - header-only: never opens tensor data sections beyond seeking metadata
#   - emits per-tensor shard layout, per-rank memory budget, role tables
#   - exits non-zero on unsupported/illegal slicing with a structured error

import argparse
import json
import os
import re
import struct
import sys

GGUF_MAGIC = b'GGUF'

VTYPE = {0: ('B', 1), 1: ('b', 1), 2: ('H', 2), 3: ('h', 2), 4: ('I', 4),
         5: ('i', 4), 6: ('f', 4), 7: ('?', 1), 8: ('str', None), 9: ('arr', None),
         10: ('Q', 8), 11: ('q', 8), 12: ('d', 8)}

# This fork's ggml_type numbering (ggml/include/ggml.h); keep in sync with the enum there.
GGML_TYPE = {
    0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0',
    9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K',
    15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S',
    20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S', 23: 'IQ4_XS', 24: 'I8', 25: 'I16',
    26: 'I32', 27: 'I64', 28: 'F64', 29: 'IQ1_M', 30: 'BF16', 34: 'TQ1_0',
    35: 'TQ2_0', 39: 'MXFP4', 40: 'NVFP4', 41: 'Q1_0', 42: 'TURBO2_0',
    43: 'TURBO3_0', 44: 'TURBO4_0', 45: 'TQ3_1S', 46: 'TQ4_1S', 47: 'Q2_0',
}

# (type_size bytes per block, blck_size elements) for every type the loader can meet.
TYPE_TRAITS = {
    'F32': (4, 1), 'F16': (2, 1), 'BF16': (2, 1), 'F64': (8, 1),
    'I8': (1, 1), 'I16': (2, 1), 'I32': (4, 1), 'I64': (8, 1),
    'Q4_0': (18, 32), 'Q5_0': (22, 32), 'Q8_0': (34, 32),
    'Q4_1': (24, 32), 'Q5_1': (26, 32), 'Q8_1': (36, 32),
    'Q2_K': (84, 256), 'Q3_K': (110, 256), 'Q4_K': (144, 256),
    'Q5_K': (176, 256), 'Q6_K': (210, 256), 'Q8_K': (292, 256),
    'IQ2_XXS': (92, 256), 'IQ2_XS': (96, 256), 'IQ3_XXS': (98, 256),
    'IQ1_S': (78, 256), 'IQ4_NL': (18, 32), 'IQ3_S': (110, 256),
    'IQ2_S': (84, 256), 'IQ4_XS': (138, 256), 'IQ1_M': (79, 256),
    'TQ1_0': (44, 64), 'TQ2_0': (82, 64),
}


class Tp5Error(Exception):
    def __init__(self, code, **fields):
        self.code = code
        self.fields = fields
        super().__init__(f"{code}: " + json.dumps(fields, default=str))


def u32(f):
    return struct.unpack('<I', f.read(4))[0]


def u64(f):
    return struct.unpack('<Q', f.read(8))[0]


def rstr(f):
    n = u64(f)
    return f.read(n).decode('utf-8')


def rvalue(f):
    t = u32(f)
    fmt, sz = VTYPE[t]
    if fmt == 'str':
        return rstr(f)
    if fmt == 'arr':
        et = u32(f)
        n = u64(f)
        efmt, esz = VTYPE[et]
        if efmt == 'str':
            return [rstr(f) for _ in range(n)]
        return list(struct.unpack('<' + efmt * n, f.read(esz * n)))
    return struct.unpack(fmt, f.read(sz))[0]


def parse_gguf(path):
    """Read header + tensor directory. Never touches the data section."""
    with open(path, 'rb') as f:
        if f.read(4) != GGUF_MAGIC:
            raise Tp5Error('TP5_E_NOT_GGUF', file=path)
        version = u32(f)
        n_tensors = u64(f)
        n_kv = u64(f)
        kvs = {}
        for _ in range(n_kv):
            k = rstr(f)
            kvs[k] = rvalue(f)
        tensors = []
        for _ in range(n_tensors):
            name = rstr(f)
            n_dims = u32(f)
            dims = [u64(f) for _ in range(n_dims)]
            ttype = u32(f)
            off = u64(f)
            tensors.append({'name': name, 'ne': dims, 'type_id': ttype, 'offset': off})
        return version, kvs, tensors, f.tell()


def row_size_bytes(type_name, ne0):
    ts, bs = TYPE_TRAITS[type_name]
    if ne0 % bs != 0:
        raise Tp5Error('TP5_E_QUANT_ROW', tensor_type=type_name, ne0=ne0,
                       blck_size=bs, reason='row length is not a whole number of quant blocks')
    return ts * ne0 // bs


def tensor_nbytes(type_name, ne):
    total = row_size_bytes(type_name, ne[0])
    for n in ne[1:]:
        total *= n
    return total


def discover_shards(path):
    if os.path.isfile(path):
        return [path]
    m = re.match(r'(.+)-\d+-of-(\d+)\.gguf$', '')
    files = sorted(fn for fn in os.listdir(path) if fn.endswith('.gguf'))
    if not files:
        raise Tp5Error('TP5_E_NO_GGUF', path=path)
    # order by split.no
    shards = []
    for fn in files:
        _, kvs, _, _ = None, None, None, None
        shards.append((os.path.join(path, fn), fn))
    # read split.no for correct ordering
    ordered = []
    for full, fn in shards:
        _, kvs, _, _ = parse_gguf(full)
        no = kvs.get('split.no', 0)
        ordered.append((no, full))
    ordered.sort()
    return [p for _, p in ordered]


class Plan:
    """TP5 A-F plan for qwen4exp: role tables, per-rank slices, budgets."""

    def __init__(self, kvs, ranks):
        a = 'qwen4exp.'
        self.ranks = ranks
        self.H = kvs[a + 'embedding_length']
        self.L = kvs[a + 'block_count']
        self.C = kvs[a + 'hyper_connection.count']
        self.R = kvs[a + 'hyper_connection.low_rank']
        self.E = kvs[a + 'expert_count']
        self.K = kvs[a + 'expert_used_count']
        self.F = kvs[a + 'expert_feed_forward_length']
        self.Fs = kvs.get(a + 'expert_shared_feed_forward_length', self.F)
        self.Nq = kvs[a + 'attention.head_count']
        self.Nkv = kvs[a + 'attention.head_count_kv']
        self.da = kvs[a + 'attention.key_length']
        self.Nk = kvs[a + 'ssm.group_count']
        self.Nv = kvs[a + 'ssm.time_step_rank']
        self.ds = kvs[a + 'ssm.state_size']
        self.Ni = kvs[a + 'attention.indexer.head_count']
        self.di = kvs[a + 'attention.indexer.key_length']
        self.n_vocab = kvs[a + 'vocab_size'] if a + 'vocab_size' in kvs else None
        self.ctx = kvs.get(a + 'context_length', 0)
        self.full_interval = kvs.get(a + 'full_attention_interval', 4)
        self.ple_layers = kvs.get(a + 'ple.layers', [])
        self.ple_ngram = kvs.get(a + 'ple.ngram_size', 0)
        self.ple_heads = kvs.get(a + 'ple.heads_per_ngram', 0)

        # per-layer type: is_recr unless (i+1) % interval == 0
        self.is_recr = [not ((i + 1) % self.full_interval == 0) for i in range(self.L)]
        self.n_full = sum(1 for r in self.is_recr if not r)

        # QSA role table (TP5.md §7.1): Q roles [5,5,5,5,4], KV instances [1,1,2,1,1]
        q_per_role = self._split_heads(self.Nq, ranks)          # [5,5,5,5,4]
        kv_role = [1] * ranks
        kv_role[2] = 2                                           # bridge role
        self.q_role_counts = q_per_role
        self.kv_role_counts = kv_role

        # rotated KV instances per physical rank over n_full layers (TP5.md §4.4)
        self.kv_instances = [0] * ranks
        for o in range(self.n_full):
            for role, cnt in enumerate(kv_role):
                self.kv_instances[(role + o) % ranks] += cnt

        # GDN V/state heads per rank: [10,10,10,10,8] for Nv=48, ranks=5
        self.gdn_v_heads = self._split_heads(self.Nv, ranks)
        # GDN V/state heads per rank: quantization-aware front-loaded split.
        # TP5.md 8.1: [0,10),[10,20),[20,30),[30,40),[40,48) for Nv=48/P=5.
        # A balanced [10,10,10,9,9] would give 9*ds=1152 rows, 1152 % 256 != 0,
        # an illegal half-block cut of the Q5_K ssm_out input axis.
        self.gdn_v_heads = self._split_heads_quant(
            self.Nv, ranks, unit=self.ds, blck=256)
        # GDN Q/K heads per rank follow the modulo map; prearrangement duplicates
        self.gdn_qk_instances = sum(self.gdn_v_heads)  # 48 Q/K head instances after prearrangement

    @staticmethod
    def _split_heads(n, ranks):
        base = n // ranks
        rem = n % ranks
        return [base + (1 if i < rem else 0) for i in range(ranks)]

    @staticmethod
    def _split_heads_quant(n, ranks, unit, blck):
        """Front-loaded split where every rank's head count keeps quant blocks
        whole: each rank's slice length * unit must be a multiple of blck.
        For Nv=48, ds=128, blck=256: step = 256/128 = 2 heads -> [10,10,10,10,8].
        Raises when n itself or the total cannot be legally distributed."""
        import math
        if blck % unit != 0:
            # unit does not divide blck: only multiples of blck/unit heads are
            # guaranteed legal; if that is not an integer every boundary at a
            # multiple of lcm(unit, blck)/unit is. Use the general rule:
            lcm = unit * blck // math.gcd(unit, blck)
            step = lcm // unit
        else:
            step = blck // unit
        if step <= 1:
            return Plan._split_heads(n, ranks)
        if n % step != 0:
            raise Tp5Error('TP5_E_HEAD_SPLIT', n=n, ranks=ranks, unit=unit,
                           blck=blck, reason='total head count is not a whole number of legal steps')
        # front-load: give each of the first ranks ceil(n/ranks) rounded up to a
        # whole step, last rank takes the remainder (also a whole step).
        target = math.ceil(n / ranks)
        per = math.ceil(target / step) * step
        counts = [per] * (ranks - 1) + [n - per * (ranks - 1)]
        if counts[-1] <= 0 or counts[-1] % step != 0:
            # try one smaller step multiple
            per -= step
            if per <= 0:
                raise Tp5Error('TP5_E_HEAD_SPLIT', n=n, ranks=ranks, unit=unit,
                               blck=blck, reason='no legal front-loaded split exists')
            counts = [per] * (ranks - 1) + [n - per * (ranks - 1)]
            if counts[-1] <= 0 or counts[-1] % step != 0:
                raise Tp5Error('TP5_E_HEAD_SPLIT', n=n, ranks=ranks, unit=unit,
                               blck=blck, counts=counts,
                               reason='no legal front-loaded split exists')
        return counts

    def tensor_role(self, name):
        """Return (semantic, split_kind, axis, per_rank_len_fn or None)."""
        m = re.match(r'blk\.(\d+)\.(.+)', name)
        il = int(m.group(1)) if m else None
        base = m.group(2) if m else name

        # --- HC: fully replicated (A-F) ---
        if base.startswith('hc_'):
            return ('hc', 'mirrored', None, None)
        if name.startswith('output_hc_'):
            return ('hc', 'mirrored', None, None)

        # --- MoE experts: intra-expert channel split ---
        if base == 'ffn_gate_exps.weight' or base == 'ffn_up_exps.weight':
            return ('moe_gate_up', 'split_axis1_rows', 1, lambda r: self._rows_per_rank(self.F, r))
        if base == 'ffn_down_exps.weight':
            return ('moe_down', 'split_axis0_channels', 0, lambda r: self._chan_per_rank(self.F, r))
        if base in ('ffn_gate_inp.weight',):
            return ('moe_router', 'mirrored', None, None)

        # --- shared expert: same channel split ---
        if base in ('ffn_gate_shexp.weight', 'ffn_up_shexp.weight'):
            return ('shexp_gate_up', 'split_axis1_rows', 1, lambda r: self._rows_per_rank(self.Fs, r))
        if base == 'ffn_down_shexp.weight':
            return ('shexp_down', 'split_axis0_channels', 0, lambda r: self._chan_per_rank(self.Fs, r))
        if base == 'ffn_gate_inp_shexp.weight':
            return ('shexp_router', 'mirrored', None, None)

        # --- QSA attention ---
        if il is not None and not self.is_recr[il]:
            if base == 'attn_q.weight':
                return ('qsa_q_gate', 'split_axis1_head_rows', 1,
                        lambda r: 2 * self.da * self.q_role_counts[r])
            if base in ('attn_k.weight', 'attn_v.weight'):
                return ('qsa_kv', 'split_axis1_head_rows_replicated', 1,
                        lambda r: self.da * self.kv_role_counts[r])
            if base == 'attn_output.weight':
                return ('qsa_out', 'split_axis0_head_cols', 0,
                        lambda r: self.da * self.q_role_counts[r])
            if base.startswith('indexer.'):
                return ('indexer', 'mirrored', None, None)
            if base in ('attn_q_norm.weight', 'attn_k_norm.weight'):
                return ('qsa_norm', 'mirrored', None, None)
        elif il is not None and self.is_recr[il]:
            # GDN
            if base == 'attn_qkv.weight':
                return ('gdn_qkv', 'split_axis1_vhead_rows', 1, lambda r: self._gdn_qkv_rows(r))
            if base == 'attn_gate.weight':
                return ('gdn_gate', 'split_axis1_vhead_rows', 1, lambda r: self.ds * self.gdn_v_heads[r])
            if base == 'ssm_out.weight':
                return ('gdn_out', 'split_axis0_vhead_cols', 0, lambda r: self.ds * self.gdn_v_heads[r])
            if base in ('ssm_beta.weight', 'ssm_alpha.weight'):
                return ('gdn_scalar_proj', 'split_axis1_vhead', 1, lambda r: self.gdn_v_heads[r])
            if base in ('ssm_a', 'ssm_dt.bias', 'ssm_norm.weight'):
                return ('gdn_state_param', 'mirrored', None, None)
            if base == 'ssm_conv1d.weight':
                return ('gdn_conv', 'split_axis1_vhead_rows', 1, lambda r: self._gdn_conv_rows(r))
        # PLE (layer is recurrent by loader invariant)
        if base.startswith('ple_') or base == 'ple_conv1d.weight':
            return ('ple', 'mirrored', None, None)

        # --- global ---
        if name == 'token_embd.weight':
            return ('token_embd', 'mirrored', None, None)
        if name == 'output.weight':
            return ('lm_head', 'split_axis1_vocab_rows', 1, lambda r: self._rows_per_rank(self.n_vocab, r))
        if name == 'per_layer_token_embd.weight':
            return ('ple_table', 'cpu_resident', None, None)
        return ('other', 'mirrored', None, None)

    def _rows_per_rank(self, n_rows, r):
        return self._split_heads(n_rows, self.ranks)[r]

    def _chan_per_rank(self, n_chan, r):
        # contiguous channel ranges [128i, 128(i+1))
        assert n_chan % self.ranks == 0
        return n_chan // self.ranks

    def _gdn_qkv_rows(self, r):
        # prearranged layout: per local V head h -> Q/K head (global V head + offset) % Nk, plus V rows
        n_v_rows = self.ds * self.gdn_v_heads[r]
        n_qk_rows = 2 * self.ds * self.gdn_qk_per_rank(r)
        return n_qk_rows + n_v_rows

    def gdn_qk_per_rank(self, r):
        return self.gdn_v_heads[r]

    def _gdn_conv_rows(self, r):
        # conv channels = 2*ds*Nk + ds*Nv prearranged per local V head
        return self._gdn_qkv_rows(r)

    def budget(self, tensors, ctx, kv_type_name='F16'):
        """Per-rank static weight budget + KV + GDN state + comm buffers."""
        per_rank = [0] * self.ranks
        detail = []
        for t in tensors:
            tname = GGML_TYPE.get(t['type_id'])
            if tname is None:
                raise Tp5Error('TP5_E_UNKNOWN_TYPE', tensor=t['name'], type_id=t['type_id'])
            semantic, kind, axis, len_fn = self.tensor_role(t['name'])
            full_bytes = tensor_nbytes(tname, t['ne'])
            if kind.startswith('cpu_'):
                detail.append((t['name'], semantic, 'cpu', 0, full_bytes))
                continue
            if kind == 'mirrored':
                for r in range(self.ranks):
                    per_rank[r] += full_bytes
                detail.append((t['name'], semantic, 'mirror', full_bytes, full_bytes))
            else:
                axis_len = t['ne'][axis] if axis is not None and axis < len(t['ne']) else None
                for r in range(self.ranks):
                    local = len_fn(r)
                    # rows on axis1: local row count; channels on axis0: local channels
                    if axis == 1:
                        b = row_size_bytes(tname, t['ne'][0]) * local
                        for n in t['ne'][2:]:
                            b *= n
                    else:
                        b = tensor_nbytes(tname, [local] + list(t['ne'][1:]))
                    per_rank[r] += b
                detail.append((t['name'], semantic, kind, full_bytes, None))
        # KV cache: n_full layers * ctx * 2(KV) * kv_instances[r] * da * bytes
        ts, bs = TYPE_TRAITS[kv_type_name]
        el_bytes = ts / bs
        kv_bytes = [self.n_full * ctx * 2 * self.kv_instances[r] * self.da * el_bytes for r in range(self.ranks)]
        # GDN state: 36 layers * Nv_local * ds*ds * 4B (F32)
        gdn_bytes = [self.L - self.n_full, ] * self.ranks  # placeholder replaced below
        gdn_bytes = [sum(1 for x in self.is_recr if x) * self.gdn_v_heads[r] * self.ds * self.ds * 4 for r in range(self.ranks)]
        # indexer cache: n_full * ctx * di * 2B (F16), replicated
        idx_bytes = [self.n_full * ctx * self.di * 2] * self.ranks
        # comm inbox: 96 stages * 5 senders * H * 4B (FP32 wire, decode)
        comm_bytes = [(2 * self.L) * self.ranks * self.H * 4] * self.ranks
        return per_rank, kv_bytes, gdn_bytes, idx_bytes, comm_bytes, detail


def main():
    ap = argparse.ArgumentParser(description='TP5 header-only GGUF manifest generator')
    ap.add_argument('--model', required=True, help='GGUF file or directory of shards')
    ap.add_argument('--ranks', type=int, default=5)
    ap.add_argument('--context', type=int, default=131072)
    ap.add_argument('--kv-dtype', default='F16', choices=sorted(TYPE_TRAITS))
    ap.add_argument('--output', default='tp5-manifest.json')
    ap.add_argument('--source-commit', default=None)
    args = ap.parse_args()

    shards = discover_shards(args.model)
    kvs = None
    tensors = []
    for shard in shards:
        _, k, ts, _ = parse_gguf(shard)
        if kvs is None:
            kvs = k
        else:
            # merge: keep first shard's general metadata
            for kk, vv in k.items():
                if kk not in kvs:
                    kvs[kk] = vv
        tensors.extend(ts)

    arch = kvs.get('general.architecture')
    if arch != 'qwen4exp':
        raise Tp5Error('TP5_E_ARCH', architecture=arch,
                       reason='this plan generator only covers qwen4exp')

    plan = Plan(kvs, args.ranks)

    # vocab from tokenizer metadata
    if plan.n_vocab is None:
        plan.n_vocab = kvs.get('tokenizer.ggml.tokens_count', 0) or len(
            kvs.get('tokenizer.ggml.tokens', []))

    per_rank, kv_bytes, gdn_bytes, idx_bytes, comm_bytes, detail = plan.budget(
        tensors, args.context, args.kv_dtype)

    GiB = 1024 ** 3
    MiB = 1024 ** 2

    manifest = {
        'schema_version': 1,
        'source_commit': args.source_commit,
        'plan': 'qwen4exp_tp5_af',
        'model_identity': {
            'name': kvs.get('general.name'),
            'architecture': arch,
            'quantized_by': kvs.get('general.quantized_by'),
            'file_type': kvs.get('general.file_type'),
            'shards': len(shards),
            'total_bytes': sum(os.path.getsize(s) for s in shards),
        },
        'actual_model_validated': True,
        'ranks': args.ranks,
        'hparams': {
            'H': plan.H, 'L': plan.L, 'C': plan.C, 'R': plan.R,
            'E': plan.E, 'K': plan.K, 'F': plan.F, 'Fs': plan.Fs,
            'Nq': plan.Nq, 'Nkv': plan.Nkv, 'da': plan.da,
            'Nk': plan.Nk, 'Nv': plan.Nv, 'ds': plan.ds,
            'Ni': plan.Ni, 'di': plan.di, 'vocab': plan.n_vocab,
            'context_length': plan.ctx,
            'n_full_attn': plan.n_full, 'n_gdn': plan.L - plan.n_full,
            'ple_layers': plan.ple_layers,
        },
        'hc': {'layout': 'mirrored', 'compute': 'af', 'reference_available': True},
        'moe': {'partition': 'intra_expert', 'channels_per_rank': plan.F // plan.ranks},
        'attention': {
            'q_role_counts': plan.q_role_counts,
            'kv_role_counts': plan.kv_role_counts,
            'kv_instances_rotated': plan.kv_instances,
            'rotation': 'full_attention_ordinal_mod_5',
        },
        'gdn': {
            'state_head_counts': plan.gdn_v_heads,
            'qk_mapping': 'global_v_head_mod_global_qk_count',
            'qk_instances_after_prearrangement': plan.gdn_qk_instances,
        },
        'collective': {
            'decode_algorithm': 'mesh_sum',
            'wire_type': 'f32',
            'accumulation_type': 'f32',
            'expected_main_events': 2 * plan.L,
        },
        'memory_budget_per_rank': {
            'weights_static': [round(b / GiB, 4) for b in per_rank],
            'kv_cache': [round(b / GiB, 4) for b in kv_bytes],
            'gdn_state': [round(b / MiB, 2) for b in gdn_bytes],
            'indexer_cache': [round(b / MiB, 2) for b in idx_bytes],
            'comm_inbox': [round(b / MiB, 2) for b in comm_bytes],
            'total_static': [round((per_rank[r] + kv_bytes[r] + gdn_bytes[r] + idx_bytes[r] + comm_bytes[r]) / GiB, 4)
                             for r in range(args.ranks)],
        },
        'tensor_layouts': [
            {'name': n, 'semantic': s, 'kind': k, 'full_bytes': fb if fb else None,
             'per_rank_bytes': prb if prb else None}
            for (n, s, k, fb, prb) in detail
        ],
        'unsupported_features': [
            'multi-request batching', 'MTP/speculative decode', 'LoRA', 'custom control vectors',
            'XKV/RERoT', 'FlashPrefill sparse kernel', 'multimodal',
        ],
    }

    # quantization slice legality checks (TP5.md §6.3)
    errors = []
    for t in tensors:
        tname = GGML_TYPE.get(t['type_id'])
        semantic, kind, axis, len_fn = plan.tensor_role(t['name'])
        if kind == 'split_axis0_channels':
            blck = TYPE_TRAITS[tname][1]
            per = len_fn(0)
            if per % blck != 0:
                errors.append({'tensor': t['name'], 'type': tname, 'axis': 0,
                               'requested': per, 'blck': blck})
    if errors:
        raise Tp5Error('TP5_E_QUANT_SLICE', action='reject_before_weight_upload',
                       violations=errors)

    with open(args.output, 'w') as f:
        json.dump(manifest, f, indent=1)
    print(f'wrote {args.output}: {len(tensors)} tensors, ranks={args.ranks}, ctx={args.context}')
    for r in range(args.ranks):
        tot = manifest['memory_budget_per_rank']['total_static'][r]
        print(f'  rank {r}: weights={manifest["memory_budget_per_rank"]["weights_static"][r]:.3f} GiB'
              f'  kv={manifest["memory_budget_per_rank"]["kv_cache"][r]:.3f} GiB'
              f'  total={tot:.3f} GiB')


if __name__ == '__main__':
    try:
        main()
    except Tp5Error as e:
        print(f'{e.code}:', json.dumps(e.fields, default=str), file=sys.stderr)
        sys.exit(2)
