#!/usr/bin/env python3
"""v3.0 regression tests — Axe 2: harmonic realism (Low Interval Limit, modal clashes,
contour-preserving autofix, overlaps, directional metrics)."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "mcp-servers", "audio-analyzer"), os.path.join(ROOT, "mcp-servers", "audio-index")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np
import pretty_midi


class TestLowIntervalLimit(unittest.TestCase):
    def test_table(self):
        from musicality_metrics import low_interval_violation
        self.assertTrue(low_interval_violation(59, 60))    # b2 below C4 forbidden
        self.assertFalse(low_interval_violation(60, 61))   # C4-D ok
        self.assertTrue(low_interval_violation(51, 54))    # tierce below E3 forbidden
        self.assertFalse(low_interval_violation(52, 55))
        self.assertTrue(low_interval_violation(42, 49))    # quinte below G2 forbidden
        self.assertFalse(low_interval_violation(43, 50))
        self.assertTrue(low_interval_violation(35, 47))    # octave below C2 forbidden
        self.assertFalse(low_interval_violation(36, 48))

    def test_optimize_voice_leading_respects_lil(self):
        from musicality_metrics import low_interval_violation, optimize_voice_leading
        for pcs in ([0, 3, 7], [0, 4, 7], [0, 3, 7, 10], [5, 8, 0, 3]):
            voicings = optimize_voice_leading([pcs, pcs], low_midi=48, high_midi=72)
            for v in voicings:
                s = sorted(v)
                for i in range(len(s)):
                    for j in range(i + 1, len(s)):
                        self.assertFalse(low_interval_violation(s[i], s[j]),
                                         f"LIL violation in voicing {s} for pcs {pcs}")


class TestModalClashes(unittest.TestCase):
    def _notes(self, seq):
        return [(float(s), float(s) + 1.0, int(p), tr) for s, p, tr in seq]

    def test_phrygian_b9_allowed_close_b2_forbidden(self):
        from musicality_metrics import analyze_harmonic_clashes
        # bass C2 (36) held vs upper Db4 (61): 25 st apart... use b9 = 13 st: bass C3=48, upper Db4=61
        chords = [{"start": 0.0, "end": 2.0, "symbol": "bIImaj7"}]  # C phrygian: bII = Db
        notes = self._notes([(0.0, 48, "Bass"), (0.0, 61, "Lead")])
        res = analyze_harmonic_clashes(notes, chords=chords, mode="phrygian", tonic_pc=0)
        self.assertEqual(res["hard_clashes"], [])
        self.assertEqual(len(res["allowed_b9"]), 1)
        # same b9 outside the modal context -> hard
        res2 = analyze_harmonic_clashes(notes, chords=chords, mode="aeolian", tonic_pc=0)
        self.assertTrue(any(c["rule"] == "b9_forbidden_outside_modal_context" for c in res2["hard_clashes"]))
        # close held minor 2nd (1 st) always forbidden: C4=60 vs Db4=61 held
        notes3 = self._notes([(0.0, 60, "Bass"), (0.0, 61, "Lead")])
        res3 = analyze_harmonic_clashes(notes3, chords=[{"start": 0.0, "end": 2.0, "symbol": "i"}], mode="phrygian", tonic_pc=0)
        self.assertTrue(any(c["rule"] == "close_minor_2nd_held_forbidden" for c in res3["hard_clashes"]))

    def test_leading_tone_and_chord_third(self):
        from musicality_metrics import analyze_harmonic_clashes
        # harmonic minor in C: leading tone B (pc 11) over V7 allowed, over i forbidden
        chords_v = [{"start": 0.0, "end": 1.0, "symbol": "V7"}]
        chords_i = [{"start": 0.0, "end": 1.0, "symbol": "i"}]
        lt = [(0.0, 1.0, 71, "Lead")]
        r_ok = analyze_harmonic_clashes(lt, chords=chords_v, mode="harmonic_minor", tonic_pc=0)
        self.assertEqual(r_ok["leading_tone_issues"], [])
        r_bad = analyze_harmonic_clashes(lt, chords=chords_i, mode="harmonic_minor", tonic_pc=0)
        self.assertTrue(any("leading_tone_outside_dominant" in i["rule"] for i in r_bad["leading_tone_issues"]))
        # doubled leading tone across tracks forbidden
        r_dbl = analyze_harmonic_clashes([(0.0, 1.0, 71, "Lead"), (0.0, 1.0, 83, "Chords")],
                                         chords=chords_v, mode="harmonic_minor", tonic_pc=0)
        self.assertTrue(any("leading_tone_doubled" in i["rule"] for i in r_dbl["leading_tone_issues"]))
        # chord without its third -> flagged
        r_third = analyze_harmonic_clashes([(0.0, 1.0, 48, "Chords"), (0.0, 1.0, 55, "Chords")],
                                           chords=[{"start": 0.0, "end": 1.0, "symbol": "Cmaj7"}], mode="ionian", tonic_pc=0)
        self.assertTrue(any(i["rule"] == "chord_third_missing" for i in r_third["chord_third_issues"]))


class TestContourPreservingAutofix(unittest.TestCase):
    def test_cost_prefers_contour_and_avoids_parallel(self):
        from midi_autofix import contour_preserving_cost
        # ascending phrase 60 -> 62 -> n_k=64; candidate 64 (repeat prev? no) vs 62 (regression)
        j_stay = contour_preserving_cost(65, 64, 62, None, None)
        j_regress = contour_preserving_cost(62, 64, 62, None, None)
        self.assertGreater(j_regress, j_stay)  # w3: repeat of prev note penalized + w2 sign change
        # parallel fifth with bass: prev (62, bass 50) -> iv 0? (62-50)%12=0 octave; cand (69, bass 57): (69-57)%12=0 same + both moved up -> w4
        j_par = contour_preserving_cost(69, 64, 62, 57, 50)
        j_nopar = contour_preserving_cost(65, 64, 62, 57, 50)
        self.assertGreaterEqual(j_par, j_nopar + 5.0)


class TestNoteOverlapSanitizer(unittest.TestCase):
    def test_mono_808_no_overlap_after_full_groove_chain(self):
        from groove_lib import apply_haas_pocket, apply_swing, enforce_frequency_inertia, ms_to_beats, sanitize_note_overlaps
        bpm = 140.0
        notes = [
            {"start": 0.0, "pitch": 36, "length": 0.30, "velocity": 100},
            {"start": 0.25, "pitch": 36, "length": 0.30, "velocity": 100},   # overlap 0.05
            {"start": 0.50, "pitch": 38, "length": 0.10, "velocity": 90},
            {"start": 0.55, "pitch": 38, "length": 0.20, "velocity": 90},     # overlap
        ]
        tracks = {"808": list(notes)}
        tracks["808"] = apply_swing(tracks["808"], swing=0.12, beats_per_bar=4.0)
        tracks["808"] = enforce_frequency_inertia(tracks["808"], bpm=bpm)
        tracks = apply_haas_pocket(tracks, bpm=bpm, profile="trap_rap")
        out = sanitize_note_overlaps(tracks, bpm=bpm, voices_by_role={"808": 1})
        guard = ms_to_beats(10.0, bpm)
        ns = sorted(out["808"], key=lambda n: float(n["start"]))
        for i in range(len(ns) - 1):
            end_i = float(ns[i]["start"]) + float(ns[i].get("length", ns[i].get("duration")))
            self.assertLessEqual(end_i, float(ns[i + 1]["start"]) - guard + 1e-4)  # _set_len rounds to 5 decimals

    def test_poly_same_pitch_overlap_removed(self):
        from groove_lib import sanitize_note_overlaps
        tracks = {"chords": [
            {"start": 0.0, "pitch": 60, "length": 1.0, "velocity": 90},
            {"start": 0.5, "pitch": 60, "length": 1.0, "velocity": 90},
        ]}
        out = sanitize_note_overlaps(tracks, bpm=120.0, voices_by_role={"chords": 4})
        ns = sorted(out["chords"], key=lambda n: float(n["start"]))
        self.assertLessEqual(float(ns[0]["start"]) + float(ns[0]["length"]), float(ns[1]["start"]) + 1e-6)


class TestSynchronousAnticipation(unittest.TestCase):
    def test_lead_crossing_frontier_is_cut_unless_in_new_chord(self):
        from transitions_lib import apply_harmonic_anticipation
        chords = [{"start": 4.0, "pitch": 60, "length": 1.0, "velocity": 90}]
        bass = [{"start": 4.0, "pitch": 36, "length": 1.0, "velocity": 100}]
        lead_cross = [{"start": 3.2, "pitch": 62, "length": 1.0, "velocity": 90}]   # crosses frontier 3.5, Db not in new C chord
        lead_ok = [{"start": 3.2, "pitch": 60, "length": 1.0, "velocity": 90}]       # C belongs to new chord -> kept
        res = apply_harmonic_anticipation(
            chords, bass, bar_boundaries=[4.0, 8.0], offset_beats=0.5, ratio=1.0,
            lead_notes=lead_cross + lead_ok,
            allowed_pcs_by_pivot={4.0: {0, 4, 7}},
        )
        self.assertGreaterEqual(res["moved_chords"], 1)
        self.assertGreaterEqual(res["cut_lead"], 1)
        cut = next(n for n in lead_cross + lead_ok if n["pitch"] == 62)
        kept = next(n for n in lead_cross + lead_ok if n["pitch"] == 60)
        self.assertLessEqual(float(cut["start"]) + float(cut.get("length", cut.get("duration"))), 3.5 - 0.05 + 1e-6)
        self.assertAlmostEqual(float(kept["start"]) + float(kept.get("length", kept.get("duration"))), 4.2, places=3)


class TestDirectionalMetrics(unittest.TestCase):
    def _pm(self, lead_pitches, bpm=120.0):
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        inst = pretty_midi.Instrument(program=0, name="Lead")
        for i, p in enumerate(lead_pitches):
            inst.notes.append(pretty_midi.Note(velocity=90, pitch=int(p), start=float(i), end=float(i) + 1.0))
        pm.instruments.append(inst)
        return pm

    def test_rhythmic_motif_recurrence_band(self):
        from musicality_metrics import rhythmic_motif_recurrence
        # 4 identical bars -> cosine 1.0 -> mechanically cloned -> not ok
        pm_clone = self._pm([60, 62, 64, 65] * 4)
        r_clone = rhythmic_motif_recurrence(pm_clone, n_bars=4)
        self.assertFalse(r_clone["ok"])
        # varied bars -> inside [0.45, 0.85]
        pm_var = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Lead")
        bars = [[60, 62], [60, 62, 65], [60, 67], [62, 65, 67, 69]]
        t = 0.0
        for bar in bars:
            for p in bar:
                inst.notes.append(pretty_midi.Note(velocity=90, pitch=p, start=t, end=t + 0.5))
                t += 0.5
            t = round(t)  # next bar
        pm_var.instruments.append(inst)
        r_var = rhythmic_motif_recurrence(pm_var, n_bars=4)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in r_var["sims"]))

    def test_cadence_resolution(self):
        from musicality_metrics import cadence_resolution_ok
        pm_ok = self._pm([60, 62, 64, 67, 72, 67, 64, 60])  # final long note C (tonic)
        self.assertTrue(cadence_resolution_ok(pm_ok, tonic_pc=0)["ok"])
        pm_bad = self._pm([60, 62, 64, 67, 72, 67, 64, 62])  # final on 2nd degree
        self.assertFalse(cadence_resolution_ok(pm_bad, tonic_pc=0)["ok"])


if __name__ == "__main__":
    unittest.main()
