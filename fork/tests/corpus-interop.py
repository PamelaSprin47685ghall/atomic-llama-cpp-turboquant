#!/usr/bin/env python3
"""Run the archived production corpus writer/reader against installed TQCorpus."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='new fixture/build directory')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    prefix = args.prefix.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    legacy = '2777543b2358c88cd1c2000f7fa342b764a0e213'
    records = {}
    # These are unmodified, versioned test participants, not an imitation of the
    # format. Only their public header depends on llama_token's installed type.
    for name in ('server-corpus-dump.h', 'server-corpus-dump.cpp'):
        source = 'wanxiangqi/server/' + name
        data = subprocess.check_output(['git', '-C', str(root), 'show', legacy + ':' + source])
        (output / name).write_bytes(data)
        records[source] = hashlib.sha256(data).hexdigest()
    binary = output / 'corpus-production-interop'
    command = ['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
               str(root / 'fork/tests/corpus-production-interop.cpp'),
               str(output / 'server-corpus-dump.cpp'), '-I' + str(output),
               '-I' + str(prefix / 'include'), str(prefix / 'lib/libtq_corpus.a'), '-o', str(binary)]
    compiled = subprocess.run(command, capture_output=True, text=True)
    (output / 'compile.log').write_text(compiled.stdout + compiled.stderr)
    compiled.check_returncode()
    completed = subprocess.run([str(binary), str(output / 'fixtures')], capture_output=True, text=True)
    (output / 'interop.log').write_text(completed.stdout + completed.stderr)
    completed.check_returncode()
    result = {'status': 'passed', 'legacy_commit': legacy, 'unmodified_sources_sha256': records,
              'production_writer_and_reader': True, 'byte_round_trip': True,
              'kinds_timestamps_empty_and_prefix_requests': True, 'reader_changed_files': False}
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(completed.stdout, end='')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
