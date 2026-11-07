#!/usr/bin/env python3
"""Regression fixtures for the CPU logits comparator."""
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

COMPARATOR = pathlib.Path(__file__).with_name("compare_records.py")
HEADER = struct.Struct("<IIIIII")


class CompareRecordsTest(unittest.TestCase):
    def compare(self, left, right):
        with tempfile.TemporaryDirectory() as directory:
            left_path = pathlib.Path(directory) / "left.bin"
            right_path = pathlib.Path(directory) / "right.bin"
            left_path.write_bytes(left)
            right_path.write_bytes(right)
            return subprocess.run(
                [sys.executable, str(COMPARATOR), str(left_path), str(right_path)],
                capture_output=True, text=True, check=False,
            )

    def test_empty_files_are_rejected(self):
        self.assertNotEqual(self.compare(b"", b"").returncode, 0)

    def test_zero_output_record_is_rejected(self):
        record = HEADER.pack(1, 512, 1, 0, 0, 0)
        result = self.compare(record, record)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid header", result.stderr)

    def test_missing_logits_are_rejected(self):
        record = HEADER.pack(1, 1, 1, 0, 1, 0)
        self.assertNotEqual(self.compare(record, record).returncode, 0)

    def test_identical_logits_pass(self):
        record = HEADER.pack(1, 1, 1, 0, 1, 0) + struct.pack("<f", 1.0)
        self.assertEqual(self.compare(record, record).returncode, 0)

    def test_different_logits_fail(self):
        header = HEADER.pack(1, 1, 1, 0, 1, 0)
        result = self.compare(header + struct.pack("<f", 1.0), header + struct.pack("<f", 2.0))
        self.assertEqual(result.returncode, 1)


if __name__ == "__main__":
    unittest.main()
