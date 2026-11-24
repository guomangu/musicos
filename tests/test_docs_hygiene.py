#!/usr/bin/env python3
"""v3.0 regression tests — P2 doc hygiene: runbook <= 6000 chars, zero hardcoded /home/gamo paths."""
from __future__ import annotations

import os
import unittest

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDocsHygiene(unittest.TestCase):
    def test_agent_runbook_under_6000_chars(self):
        path = os.path.join(ROOT, "docs", "AGENT_RUNBOOK.md")
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertLessEqual(len(content), 6000, f"AGENT_RUNBOOK.md = {len(content)} cars > 6000")

    def test_no_hardcoded_home_paths(self):
        bad = []
        for sub in ("docs", "scripts", "schemas"):
            base = os.path.join(ROOT, sub)
            for root, dirnames, fns in os.walk(base):
                dirnames[:] = [d for d in dirnames if d != "__pycache__"]
                for fn in fns:
                    if fn.endswith((".pyc", ".png", ".wav", ".npz", ".db")):
                        continue
                    p = os.path.join(root, fn)
                    try:
                        with open(p, encoding="utf-8", errors="ignore") as fh:
                            if "/home/gamo" in fh.read():
                                bad.append(p)
                    except Exception:
                        continue
        self.assertEqual(bad, [], f"chemins /home/gamo encore présents: {bad}")

    def test_reference_archive_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(ROOT, "docs", "REFERENCE_ARCHIVE.md")))


if __name__ == "__main__":
    unittest.main()
