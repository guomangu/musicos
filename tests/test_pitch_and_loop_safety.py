#!/usr/bin/env python3
"""v3.0 regression tests — Axe 1: pitch & loop safety (octave lock, S_max, loop key-fit/BPM, kick, bells)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "mcp-servers", "audio-analyzer"), os.path.join(ROOT, "mcp-servers", "audio-index")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np


def _bp(tracks, key="C minor", bpm=132.0):
    return {"key": key, "bpm": bpm, "harmonic_core": {"key": "C", "mode": "aeolian"}, "tracks": tracks}


def _sample_track(**kw):
    t = {"name": "Lead", "role": "lead_melody", "source_type": "sample",
         "sample_path": "samples/lead.wav", "mixer_track": 3, "root_note": 60}
    t.update(kw)
    return t


def _catalog(tmp, entries):
    path = os.path.join(tmp, "samples_catalog.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"samples": entries}, fh)
    return path


class TestStrictRootLock(unittest.TestCase):
    def test_octave_error_is_blocking(self):
        from blueprint_safety import blueprint_musical_safety_issues
        with tempfile.TemporaryDirectory() as tmp:
            cat = _catalog(tmp, [{"filepath": os.path.join(tmp, "samples", "a.wav"), "filename": "a.wav",
                                  "midi_root": 69, "playable_chromatically": True, "is_loop": False, "hard_reject": False}])
            for bad in (60, 81, 57):  # 9 st, +12 st, -12 st away from the true A4=69
                bp = _bp([_sample_track(root_note=bad, sample_path="samples/a.wav")])
                codes = {i["code"] for i in blueprint_musical_safety_issues(bp, catalog_path=cat)}
                self.assertIn("root_note_mismatch", codes, f"root_note={bad} must block (varispeed octave error)")
            bp = _bp([_sample_track(root_note=69, sample_path="samples/a.wav")])
            self.assertNotIn("root_note_mismatch", {i["code"] for i in blueprint_musical_safety_issues(bp, catalog_path=cat)})

    def test_midi_root_octave_sanity(self):
        from audio_metrics import validate_midi_root_octave
        # centroid 2.9 kHz + no energy < 150 Hz -> cannot be a C2 (36): locked on a partial
        fixed, was = validate_midi_root_octave(36, 2900.0, 0.001)
        self.assertTrue(was)
        self.assertEqual(fixed, 48)
        # real low fundamental stays untouched
        fixed2, was2 = validate_midi_root_octave(36, 120.0, 0.6)
        self.assertFalse(was2)
        self.assertEqual(fixed2, 36)


class TestMaxSemitoneShift(unittest.TestCase):
    def test_shift_caps_and_multizone_decision(self):
        from blueprint_safety import decide_shift_action, max_semitone_shift, SAMPLE_SHIFT_CAP_ST
        self.assertEqual(max_semitone_shift(60, [60, 64, 67, 72]), 12)
        self.assertEqual(decide_shift_action(4, 12, "acoustic"), "ok")
        self.assertEqual(decide_shift_action(6, 16, "acoustic"), "multizone")  # 16 st ambitus <= 18
        self.assertEqual(decide_shift_action(6, 24, "acoustic"), "swap")       # ambitus too wide
        self.assertEqual(decide_shift_action(8, 18, "synth"), "multizone")     # synth cap 7 st
        self.assertEqual(SAMPLE_SHIFT_CAP_ST, {"acoustic": 5, "synth": 7})

    def test_blueprint_shift_violation_error(self):
        from blueprint_safety import blueprint_musical_safety_issues
        bp = _bp([_sample_track(root_note=72, sample_path="samples/piano.wav")])  # piano C5
        # piano played from C5 down to C3 = -24 st -> exceeds acoustic cap 5
        issues = blueprint_musical_safety_issues(bp, notes_by_track={"Lead": [48, 50, 52, 55, 72]})
        codes = {i["code"] for i in issues}
        self.assertIn("max_semitone_shift_exceeded", codes)
        iss = next(i for i in issues if i["code"] == "max_semitone_shift_exceeded")
        self.assertEqual(iss["s_max_st"], 24)
        self.assertEqual(iss["category"], "synth")  # no catalog entry -> default synth cap


class TestLoopSafety(unittest.TestCase):
    def test_loop_typing_hpss_hnr(self):
        from blueprint_safety import classify_loop_nature
        self.assertEqual(classify_loop_nature(0.10, 1.5), "percussive_loop")
        self.assertEqual(classify_loop_nature(0.10, 8.0), "melodic_loop")   # tonal despite percussive energy
        self.assertEqual(classify_loop_nature(0.60, 1.0), "melodic_loop")

    def test_melodic_loop_key_fit_reject_and_repitch(self):
        from blueprint_safety import best_loop_transposition, chroma_out_of_scale_ratio, blueprint_musical_safety_issues
        c_minor = {0, 2, 3, 5, 7, 8, 10}
        # E major loop: chroma concentrated on E, G#, B -> pc {4, 8, 11}
        chroma_e = [0.0] * 12
        for pc in (4, 8, 11):
            chroma_e[pc] = 1.0 / 3.0
        r_out = chroma_out_of_scale_ratio(chroma_e, c_minor)
        self.assertGreater(r_out, 0.22)
        delta, r_after = best_loop_transposition(chroma_e, c_minor)
        self.assertLessEqual(r_after, 0.15)
        self.assertIn(delta, (-2, -1, 1, 2))
        # blueprint-level: melodic loop with out-of-scale chroma -> error or repitch warning
        with tempfile.TemporaryDirectory() as tmp:
            cat = _catalog(tmp, [{"filepath": os.path.join(tmp, "samples", "loop_e.wav"), "filename": "loop_e.wav",
                                  "midi_root": 64, "playable_chromatically": True, "is_loop": True,
                                  "loop_nature": "melodic_loop", "chroma_cqt": chroma_e, "bpm": 132.0, "hard_reject": False}])
            bp = _bp([_sample_track(name="PadLoop", role="pad_atmosphere", sample_path="samples/loop_e.wav",
                                    root_note=64, is_loop=True, bpm=132.0)])
            issues = blueprint_musical_safety_issues(bp, catalog_path=cat)
            codes = {i["code"] for i in issues}
            self.assertTrue(codes & {"loop_key_mismatch", "loop_repitch_recommended"})
            # percussive loop in the same wrong key -> exempt
            cat2 = _catalog(tmp, [{"filepath": os.path.join(tmp, "samples", "drums.wav"), "filename": "drums.wav",
                                   "midi_root": None, "playable_chromatically": False, "is_loop": True,
                                   "loop_nature": "percussive_loop", "chroma_cqt": chroma_e, "bpm": 132.0, "hard_reject": False}])
            bp2 = _bp([_sample_track(name="Drums", role="kick", sample_path="samples/drums.wav",
                                     root_note=36, is_loop=True, bpm=132.0)])
            self.assertEqual([i for i in blueprint_musical_safety_issues(bp2, catalog_path=cat2)
                              if i["code"].startswith("loop_key")], [])

    def test_bpm_lock(self):
        from blueprint_safety import bpm_compatibility
        self.assertTrue(bpm_compatibility(132.0, 132.0)["ok"])
        self.assertFalse(bpm_compatibility(132.0, 132.0)["needs_stretch"])
        c = bpm_compatibility(144.0, 140.0)
        self.assertTrue(c["ok"])
        self.assertTrue(c["needs_stretch"])  # residual 4 BPM > 0.05 -> mandatory exact stretch
        self.assertFalse(bpm_compatibility(145.0, 110.0)["ok"])   # r_eff 1.318 > 1.14
        self.assertTrue(bpm_compatibility(90.0, 180.0)["ok"])      # double tempo allowed
        self.assertFalse(bpm_compatibility(90.0, 180.0)["needs_stretch"])


class TestKickAndBellGuards(unittest.TestCase):
    def test_kick_tuning_guard(self):
        from sample_prep_lib import analyze_kick_tail, apply_tail_gate, kick_tuning_guard
        sr = 44100
        t = np.arange(int(0.5 * sr)) / sr
        # dry-ish click + resonant tail at ~50 Hz (G1) in C major (tonic 0, fifth 7)
        kick_g = 0.9 * np.exp(-t / 0.004) * np.sin(2 * np.pi * 2000 * t) + 0.5 * np.exp(-t / 0.09) * np.sin(2 * np.pi * 50.0 * t)
        res = kick_tuning_guard(kick_g, sr, tonic_pc=0)
        self.assertEqual(res["action"], "fine_tune_ct")  # 50 Hz ~ G1 is within 2 st of the fifth
        self.assertLessEqual(res["dist_st"], 2.0)
        # out-of-key resonant tail at ~38.9 Hz (Eb1, pc 3) -> 3 st from {C,G} -> tail gate
        kick_bb = 0.9 * np.exp(-t / 0.004) * np.sin(2 * np.pi * 2000 * t) + 0.5 * np.exp(-t / 0.09) * np.sin(2 * np.pi * 38.9 * t)
        res2 = kick_tuning_guard(kick_bb, sr, tonic_pc=0)
        self.assertEqual(res2["action"], "tail_gate_95ms")
        self.assertGreater(res2["dist_st"], 2.0)
        gated = apply_tail_gate(kick_bb, sr, gate_ms=95.0)
        tail_after = analyze_kick_tail(gated, sr, start_ms=120.0, end_ms=180.0)
        self.assertLess(tail_after["tail_rms_dbfs"], -40.0)

    def test_bell_inharmonicity_and_confinement(self):
        from audio_metrics import estimate_inharmonicity
        from blueprint_safety import allowed_degrees_for_sample
        sr = 44100
        t = np.arange(int(1.0 * sr)) / sr
        bell = 0.25 * np.sin(2 * np.pi * 440 * t) + 1.0 * np.sin(2 * np.pi * 440 * 2.76 * t) + 0.7 * np.sin(2 * np.pi * 440 * 5.40 * t)
        inh = estimate_inharmonicity(bell, sr, 440.0)
        self.assertGreater(inh, 0.15)
        self.assertEqual(allowed_degrees_for_sample({"sample_nature": "inharmonic_bell_or_fx"}), [0, 7, 12])
        self.assertEqual(allowed_degrees_for_sample({"inharmonicity": inh}), [0, 7, 12])
        self.assertIsNone(allowed_degrees_for_sample({"inharmonicity": 0.02, "sample_nature": "mono_oneshot"}))


if __name__ == "__main__":
    unittest.main()
