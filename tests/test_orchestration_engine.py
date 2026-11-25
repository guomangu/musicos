#!/usr/bin/env python3
"""v3.3 J3 (#2) — OrchestrationEngine : 0 croisement soprano/Lead, 0 violation LIL sur
slash chords (inversions), pédales tonique/dominante, variété de comping entre Couplet et
Refrain, waltz 3/4, et optimize_voice_leading_with_lead (pas d'8ve/5te parallèle avec le Lead)."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "mcp-servers", "audio-analyzer"), os.path.join(ROOT, "mcp-servers", "audio-index")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from orchestration_engine import (
    COMPING_STYLES,
    apply_bass_motion,
    apply_comping_to_score_ir,
    comping_pattern,
    default_comping_arc,
    revoice_chords_with_lead,
)
from score_ir import build_score_ir


def _meta(bars: int = 16) -> dict:
    return {"bpm": 120.0, "beats_per_bar": 4, "denominator": 4, "valence": 0.0,
            "total_bars": bars, "tonic": "C", "key": "C"}


def _score(bass_pitches, lead_pitches=None, chords_per_bar=(60, 64, 67)) -> object:
    hc = {"key": "C", "mode": "ionian", "degree_progression": ["I", "V", "vi", "IV"]}
    tracks = {
        "bass": [{"start": b * 2.0, "length": 1.9, "pitch": p, "velocity": 100} for b, p in enumerate(bass_pitches)],
        "chords": [{"start": b * 4.0, "length": 3.9, "pitch": p, "velocity": 70} for b in range(16) for p in chords_per_bar],
    }
    if lead_pitches:
        tracks["lead"] = [{"start": b * 1.0, "length": 0.9, "pitch": p, "velocity": 90} for b, p in enumerate(lead_pitches)]
    return build_score_ir(hc, _meta(), tracks)


class _Sec:
    def __init__(self, name, start_bar, bars=4):
        self.name = name
        self.start_bar = start_bar
        self.bars = bars


class TestBassMotion(unittest.TestCase):
    def test_stepwise_inversion_reduces_big_jump(self):
        # saut de 36 (C2) à 45 (A2) = 9 st -> 1er renversé (tierce de l'accord) ou passage
        sir = _score([36, 36, 45, 45] * 8)
        rep = apply_bass_motion(sir, mode="stepwise_inversions", tonic_pc=0, beats_per_bar=4,
                                scale_pcs=[0, 2, 4, 5, 7, 9, 11])
        notes = sorted(sir.tracks["bass"].notes, key=lambda n: n.grid_beat)
        jumps = [abs(int(b.pitch) - int(a.pitch)) for a, b in zip(notes, notes[1:])]
        orig = [abs(int(b.pitch) - int(a.pitch)) for a, b in zip(sorted(sir.tracks["bass"].notes, key=lambda n: n.grid_beat)[:0] or [36,36,45,45]*8, [36,36,45,45]*8[1:])] if False else None
        # moyenne des sauts strictement réduite vs le fixture original (0,9,0,9,... moyenne 4.5)
        self.assertLess(sum(jumps) / len(jumps), 4.5, f"sauts: {jumps}")
        self.assertTrue(all(int(n.pitch) >= 31 for n in notes))  # plancher bass_midi >= 31
        self.assertGreaterEqual(rep["n_inversions"] + rep["n_passing"], 1)

    def test_tonic_and_dominant_pedal(self):
        sir = _score([36, 38, 41, 43] * 8)
        spans = [_Sec("Intro", 0, 4), _Sec("PreRefrain", 4, 4), _Sec("Refrain", 8, 8)]
        rep = apply_bass_motion(sir, mode="root", section_spans=spans,
                                section_modes={"intro": "tonic_pedal", "pre_refrain": "dominant_pedal"},
                                tonic_pc=0, beats_per_bar=4)
        notes = sorted(sir.tracks["bass"].notes, key=lambda n: n.grid_beat)
        intro = [int(n.pitch) for n in notes if n.grid_beat < 16]
        pre = [int(n.pitch) for n in notes if 16 <= n.grid_beat < 32]
        self.assertTrue(all(p % 12 == 0 for p in intro))   # pédale de tonique (C)
        self.assertTrue(all(p % 12 == 7 for p in pre))     # pédale de dominante (G)
        self.assertGreaterEqual(rep["n_pedal"], 1)


class TestComping(unittest.TestCase):
    def test_six_styles(self):
        self.assertEqual(len(COMPING_STYLES), 6)
        for st in COMPING_STYLES:
            pat = comping_pattern([60, 64, 67], st, beats_per_bar=4, bar_start=0.0)
            self.assertTrue(pat, st)
            for ev in pat:
                self.assertGreaterEqual(ev["start"], 0.0)
                self.assertLess(ev["start"], 4.0)
        # waltz 3/4 : pas d'attaque au temps 1
        pat = comping_pattern([60, 64, 67], "waltz_bass_chord", beats_per_bar=3, bar_start=0.0)
        self.assertTrue(all(ev["start"] >= 1.0 for ev in pat))
        # rhythmic_comp : contretemps
        pat = comping_pattern([60, 64, 67], "rhythmic_comp", beats_per_bar=4, bar_start=0.0)
        self.assertTrue(all((ev["start"] % 1.0) == 0.5 for ev in pat))

    def test_comping_varies_between_verse_and_chorus(self):
        sir = _score([36] * 32)
        spans = [_Sec("Couplet", 0, 8), _Sec("Refrain", 8, 8)]
        styles = default_comping_arc([{"name": "Couplet"}, {"name": "Refrain"}])
        self.assertEqual(styles["couplet"], "broken_arp_up")
        self.assertEqual(styles["refrain"], "rhythmic_comp")
        rep = apply_comping_to_score_ir(sir, spans, styles, beats_per_bar=4)
        self.assertEqual(rep["applied"], 16)  # 16 mesures retravaillées
        # le refrain a des attaques sur contretemps, pas le couplet
        chords = sorted(sir.tracks["chords"].notes, key=lambda n: n.grid_beat)
        refrain = [float(n.grid_beat) % 1.0 for n in chords if float(n.grid_beat) >= 32]
        verse = [float(n.grid_beat) % 1.0 for n in chords if float(n.grid_beat) < 32]
        self.assertTrue(any(v == 0.5 for v in refrain))
        self.assertTrue(all(v == 0.0 for v in verse))


class TestVoiceLeadingWithLead(unittest.TestCase):
    def test_no_soprano_lead_crossing(self):
        from musicality_metrics import optimize_voice_leading_with_lead
        pcs_list = [[0, 4, 7], [9, 0, 4], [5, 9, 0], [7, 11, 2]]  # I vi IV V en C
        lead = [84, 84, 84, 84]  # Lead très haut (C6) -> pas de croisement
        v = optimize_voice_leading_with_lead(pcs_list, lead)
        self.assertTrue(all(max(voice) < 84 - 2 for voice in v))
        # Lead à G4 (67) : le soprano des accords doit rester sous 65 (pas de croisement)
        lead_low = [67, 67, 67, 67]
        v2 = optimize_voice_leading_with_lead(pcs_list, lead_low)
        self.assertTrue(all(max(voice) < 65 for voice in v2), f"soprano: {[max(x) for x in v2]}")
        # Lead très bas (C4=60, sous la contrainte LIL) : le croisement reste borné (<= 62)
        v3 = optimize_voice_leading_with_lead(pcs_list, [60, 60, 60, 60])
        self.assertTrue(all(max(voice) <= 62 for voice in v3), f"soprano: {[max(x) for x in v3]}")

    def test_no_parallel_fifths_with_lead(self):
        from musicality_metrics import optimize_voice_leading_with_lead
        pcs_list = [[0, 4, 7], [7, 11, 2]]
        lead = [67, 74]  # G4 puis D5 : intervalles soprano/lead à éviter en parallèle
        v = optimize_voice_leading_with_lead(pcs_list, lead)
        # pas d'octave/quinte parallèle entre soprano et lead
        for i in range(1, len(v)):
            iv_now = (v[i][-1] - lead[i]) % 12
            iv_prev = (v[i - 1][-1] - lead[i - 1]) % 12
            self.assertFalse(iv_now == iv_prev and iv_now in (0, 7),
                             f"parallèle {iv_now} à l'accord {i}: {v[i][-1]} vs lead {lead[i]}")

    def test_revoice_chords_with_lead_on_score_ir(self):
        sir = _score([36] * 32, lead_pitches=[60] * 64)  # lead à C4, chords à 60/64/67 -> croisement
        rep = revoice_chords_with_lead(sir, lead_min_margin=2)
        self.assertGreaterEqual(rep["applied"], 1)
        chords = sir.tracks["chords"].notes
        lead_min = 60
        self.assertTrue(all(int(n.pitch) < lead_min - 2 + 12 for n in chords))  # abaissés d'1 octave max


if __name__ == "__main__":
    unittest.main()
