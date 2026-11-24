#!/usr/bin/env python3
"""v3.1 TEST-06/SSOT-01 — transactional autofix: tier-1 regression => rollback + blacklist,
oscillation A<->B => rollback, valid tier-2 improvement => accepted, full history journal."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "mcp-servers", "audio-analyzer")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from autofix_transaction import apply_with_rollback, classify_tiers, evaluate_patch, load_history


def _report(clipped=0, tp=-2.0, le_corr=0.9, msafety=0, tuning=2.0, glitches=0, clash=0.0,
            lufs=-11.0, low_end=30.0, air=5.0, lra=5.0, sharp=1.5, corridor=0.9):
    return {
        "clipped": clipped, "true_peak_dbtp": tp, "low_end_corr": le_corr,
        "musical_safety": {"errors": msafety},
        "foundation_tuning_ct": tuning, "micro_glitches": glitches,
        "harmonic_audio_clash_ratio": clash,
        "lufs": lufs, "low_end_share": low_end, "air_share": air, "lra_lu": lra,
        "sharpness": sharp, "corridor_score": corridor,
    }


class TestTierClassification(unittest.TestCase):
    def test_tier1_pass_and_fail(self):
        ok = classify_tiers(_report())
        self.assertTrue(ok["tier1_pass"])
        self.assertEqual(ok["tier1_failures"], [])
        bad = classify_tiers(_report(tp=-0.5))  # true_peak > ceiling(-1.2)+0.2 = -1.0
        self.assertFalse(bad["tier1_pass"])
        self.assertTrue(any("true_peak" in f for f in bad["tier1_failures"]))
        bad2 = classify_tiers(_report(msafety=2))
        self.assertIn("musical_safety_errors>0", bad2["tier1_failures"])


class TestRollback(unittest.TestCase):
    def test_tier1_regression_rolls_back_and_blacklists(self):
        before = _report(air=2.0)
        after = _report(air=6.0, clipped=3)  # improves air but clips
        d = evaluate_patch(before, after)
        self.assertEqual(d["verdict"], "ROLLED_BACK")
        self.assertIn("clipped", d["reason"])

    def test_oscillation_rolls_back(self):
        a = _report(lufs=-13.0, low_end=52.0)
        b = _report(lufs=-11.0, low_end=30.0)
        # A->B improves tier2, but B->A also "improves" relative to B? simulate A->B then B->A
        d1 = evaluate_patch(a, b)
        self.assertEqual(d1["verdict"], "ACCEPTED")  # real progress
        d2 = evaluate_patch(b, a)  # going back degrades composite -> rollback
        self.assertEqual(d2["verdict"], "ROLLED_BACK")

    def test_valid_patch_accepted_and_journaled(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = _report(lufs=-13.0)
            after = _report(lufs=-11.2)
            state = {"applied": False}

            def apply_fn():
                state["applied"] = True
                return {"ok": True}

            def eval_fn():
                return after

            res = apply_with_rollback(tmp, {"code": "ADJUST_MASTER_GAIN", "target": "mixer/0"}, apply_fn, eval_fn, before)
            self.assertEqual(res["verdict"], "ACCEPTED")
            self.assertTrue(state["applied"])
            hist = load_history(tmp)
            self.assertEqual(len(hist["attempts"]), 1)
            self.assertEqual(hist["attempts"][0]["verdict"], "ACCEPTED")
            self.assertEqual(hist["blacklisted_actions"], [])

            # now a clipping action: rollback + restore called + blacklisted
            restored = {"v": False}

            def snap_fn():
                return {"mixer": "snapshot"}

            def restore_fn(snap):
                restored["v"] = True
                restored["snap"] = snap

            res2 = apply_with_rollback(
                tmp, {"code": "BOOST_AIR", "target": "BUS Drums"},
                lambda: {"ok": True}, lambda: _report(air=9.0, clipped=5),
                before, snapshot_fn=snap_fn, restore_fn=restore_fn,
            )
            self.assertEqual(res2["verdict"], "ROLLED_BACK")
            self.assertTrue(restored["v"])
            hist2 = load_history(tmp)
            self.assertEqual(len(hist2["blacklisted_actions"]), 1)
            self.assertEqual(hist2["blacklisted_actions"][0]["code"], "BOOST_AIR")
            # blacklisted action is now skipped without applying
            called = {"n": 0}
            res3 = apply_with_rollback(
                tmp, {"code": "BOOST_AIR", "target": "BUS Drums"},
                lambda: called.__setitem__("n", called["n"] + 1), lambda: before, before,
            )
            self.assertEqual(res3["verdict"], "SKIPPED_BLACKLISTED")
            self.assertEqual(called["n"], 0)
            # history file exists on disk
            self.assertTrue(os.path.isfile(os.path.join(tmp, "renders", "autofix_history.json")))


if __name__ == "__main__":
    unittest.main()
