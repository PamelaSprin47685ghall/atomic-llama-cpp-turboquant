#!/usr/bin/env python3
"""Bitwise comparator for padmax_cpu_probe records.

Record layout (little-endian):
    uint32 n_tokens, vocab, pos_after, step, n_outputs, seq_id
    float32 logits[n_outputs * vocab]

The header must match exactly (same step schedule and same recorded row);
the logits payload must match byte for byte.  A single differing float is a
failure and is reported with its record index, step, position, sequence and
float index.
"""
import struct
import sys

HEADER = struct.Struct('<IIIIII')


def read_records(path):
    records = []
    with open(path, 'rb') as f:
        while True:
            header = f.read(HEADER.size)
            if not header:
                break
            if len(header) != HEADER.size:
                raise ValueError(f'{path}: truncated header in record {len(records)}')
            n_tokens, vocab, pos_after, step, n_outputs, seq_id = HEADER.unpack(header)
            if n_tokens == 0 or vocab == 0 or vocab > 10_000_000 or n_outputs == 0 or n_outputs > n_tokens:
                raise ValueError(f'{path}: invalid header {n_tokens}, {vocab}, {n_outputs} '
                                 f'in record {len(records)}')
            payload = f.read(n_outputs * vocab * 4)
            if len(payload) != n_outputs * vocab * 4:
                raise ValueError(f'{path}: truncated logits in record {len(records)}')
            records.append((n_tokens, vocab, pos_after, step, n_outputs, seq_id, payload))
    return records


def main():
    if len(sys.argv) != 3:
        print(f'usage: {sys.argv[0]} LEFT.bin RIGHT.bin', file=sys.stderr)
        return 2
    try:
        left = read_records(sys.argv[1])
        right = read_records(sys.argv[2])
    except (OSError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2

    if not left or not right:
        print('FAIL no records to compare')
        return 1

    if len(left) != len(right):
        print(f'FAIL record_count left={len(left)} right={len(right)}')
        return 1

    total = 0
    for i, (a, b) in enumerate(zip(left, right)):
        if a[:6] != b[:6]:
            print(f'FAIL record={i} header left={a[:6]} right={b[:6]}')
            return 1
        vocab = a[1]
        n_outputs = a[4]
        total += n_outputs * vocab
        if a[6] != b[6]:
            for j in range(n_outputs * vocab):
                x = a[6][j * 4:j * 4 + 4]
                y = b[6][j * 4:j * 4 + 4]
                if x != y:
                    print(f'FAIL record={i} step={a[3]} pos={a[2]} seq={a[5]} '
                          f'nout={n_outputs} float_index={j} '
                          f'left=0x{struct.unpack("<I", x)[0]:08x} '
                          f'right=0x{struct.unpack("<I", y)[0]:08x}')
                    return 1

    print(f'PASS records={len(left)} float32_values={total} bitwise_equal')
    return 0


if __name__ == '__main__':
    sys.exit(main())
