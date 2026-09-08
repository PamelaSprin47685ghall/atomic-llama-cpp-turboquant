#!/usr/bin/env python3
"""Harness-only checks: no server, model, network or GPU is needed."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    'rerot_semantic_smoke', Path(__file__).resolve().parents[1] / 'scripts/rerot-semantic-smoke.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeHarnessTests(unittest.TestCase):
    def test_auto_context_is_preserved(self):
        self.assertEqual(smoke.resolve_context(None, 'auto'), 131072)
        self.assertEqual(smoke.resolve_context(262144, 'auto'), 262144)

    def test_manual_default_fits_its_capacity(self):
        self.assertEqual(smoke.resolve_context(None, '8192'), 8192)
        self.assertEqual(smoke.resolve_context(None, '262144'), 131072)
        self.assertEqual(smoke.resolve_context(4096, '8192'), 4096)

    def test_invalid_explicit_capacity_is_not_silently_rewritten(self):
        for context, capacity in [(131072, '8192'), (0, 'auto'), (-1, '8192'),
                                  (None, '0'), (None, '-1'), (None, 'oops')]:
            with self.subTest(context=context, capacity=capacity):
                with self.assertRaises(ValueError):
                    smoke.resolve_context(context, capacity)

    def test_fingerprint_is_content_based(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'libllama.so.0.0.stale-build-number'
            path.write_bytes(b'first artifact')
            first = smoke.sha256_file(path)
            path.write_bytes(b'different machine code')
            self.assertEqual(first, hashlib.sha256(b'first artifact').hexdigest())
            self.assertNotEqual(first, smoke.sha256_file(path))


if __name__ == '__main__':
    unittest.main()
