#!/usr/bin/env python3
"""Unit tests for musicality_metrics.py and audit_midi_file integration."""

import os
import random
import tempfile
import unittest

import pretty_midi

from deep_diagnostics import audit_midi_file
from musicality_metrics import evaluate_midi_musicality, optimize_voice_leading


class TestMusicalityMetrics(unittest.TestCase):
    def test_random_erratic_midi_fails(self):
        random.seed(7)
        pm = pretty_midi.PrettyMIDI(initial_tempo=140.0)
        bass = pretty_midi.Instrument(program=33, name="808 Bass")
        chords = pretty_midi.Instrument(program=0, name="Chords")
        lead = pretty_midi.Instrument(program=80, name="Lead")
        counter = pretty_midi.Instrument(program=11, name="Pluck Counter")

        e_minor_pc = [4, 6, 7, 9, 11, 0, 2]
        # Jumping root-position chords (> 2.5 st)
        roots = [48, 60, 48, 62]
        for bar, r in enumerate(roots):
            t0 = bar * (4.0 * 60.0 / 140.0)
            t1 = (bar + 1) * (4.0 * 60.0 / 140.0) - 0.05
            bass.notes.append(pretty_midi.Note(100, r - 12, t0, t1))
            for iv in (0, 3, 7):
                chords.notes.append(pretty_midi.Note(90, r + iv, t0, t1))

        # Random wild leaps with zero rest & overlapping counter in same octave
        t = 0.0
        step = 60.0 / 140.0 * 0.5
        for i in range(32):
            p = 60 + random.choice([0, 11, 2, 19, 1, 14, 7, 23])
            while p % 12 not in e_minor_pc:
                p += 1
            lead.notes.append(pretty_midi.Note(100, p, t, t + step * 0.98))
            counter.notes.append(pretty_midi.Note(90, p + 2, t, t + step * 0.9))
            t += step

        pm.instruments.extend([bass, chords, lead, counter])
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            path = tmp.name
        try:
            pm.write(path)
            res = evaluate_midi_musicality(path)
            self.assertEqual(res["musicality_verdict"], "FAIL", res)
            self.assertTrue(len(res["failures"]) >= 2)
        finally:
            os.remove(path)

    def test_structured_top_down_sketch_passes(self):
        # Em7 -> Cmaj7 -> Am7 -> Bm7 in smooth voice leading
        chord_pcs = [
            [4, 7, 11, 2],  # Em7
            [0, 4, 7, 11],  # Cmaj7
            [9, 0, 4, 7],   # Am7
            [11, 2, 6, 9],  # Bm7
        ]
        voicings = optimize_voice_leading(chord_pcs, low_midi=48, high_midi=67)
        bass_roots = [40, 36, 45, 35]  # E2, C2, A2, B1 (creates natural contrary motion!)

        bpm = 140.0
        beat_s = 60.0 / bpm
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        bass = pretty_midi.Instrument(program=33, name="Bass")
        chords = pretty_midi.Instrument(program=0, name="Chords")
        lead = pretty_midi.Instrument(program=0, name="Lead")

        for bar in range(4):
            b0 = bar * 4.0
            for p in voicings[bar]:
                chords.notes.append(pretty_midi.Note(85, p, b0 * beat_s, (b0 + 3.75) * beat_s))
            bass.notes.append(pretty_midi.Note(105, bass_roots[bar], b0 * beat_s, (b0 + 3.5) * beat_s))

        # A - A' - A'' - B stepwise singable melody with ~35% rest and chord tones on strong beats
        melody_beats = [
            # Bar 1 (A on Em7: E, G, B, D) -> E4(64) G4(67) F#4(66) G4(67) [rest 2.5..4.0]
            (0.0, 0.5, 64), (0.5, 0.5, 67), (1.0, 0.5, 66), (2.0, 0.75, 67),
            # Bar 2 (A' on Cmaj7: C, E, G, B) -> G4(67) A4(69) G4(67) E4(64) [contrary to bass C2 down]
            (4.0, 0.5, 67), (4.5, 0.5, 69), (5.0, 0.5, 67), (6.0, 0.75, 64),
            # Bar 3 (A'' on Am7: A, C, E, G -> Apex C5=72) -> E4(64) G4(67) C5(72) A4(69) [leap +5 then step -3, A4 in Am7!]
            (8.0, 0.5, 64), (8.5, 0.5, 67), (9.0, 0.75, 72), (10.0, 0.75, 69),
            # Bar 4 (B cadence on Bm7: B, D, F#, A) -> B4(71) A4(69) F#4(66) F#4(66) [chord tone F#4 on beat 2!]
            (12.0, 0.5, 71), (12.5, 0.5, 69), (13.0, 0.5, 66), (14.0, 0.75, 66),
        ]
        for st, ln, p in melody_beats:
            lead.notes.append(pretty_midi.Note(95, p, st * beat_s, (st + ln) * beat_s))

        pm.instruments.extend([bass, chords, lead])
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            path = tmp.name
        try:
            pm.write(path)
            res = audit_midi_file(path, expected_key="E minor", max_pitch=84, strict_musicality=True)
            self.assertEqual(res["musicality_verdict"], "PASS", res)
            self.assertTrue(res["gate_passed"], res)
            self.assertLessEqual(res["musicality_metrics"]["voice_leading_avg_semitones"], 2.5)
        finally:
            os.remove(path)

    def test_dark_tense_mode_relaxes_appoggiaturas_and_diatonic_clashes(self):
        # F minor (F, G, Ab, Bb, C, Db, Eb): Fm -> DbMaj7 -> Bbm7 -> Fm
        chord_pcs = [
            [5, 8, 0],      # Fm
            [1, 5, 8, 0],   # Dbmaj7
            [10, 1, 5, 8],  # Bbm7
            [5, 8, 0],      # Fm
        ]
        voicings = optimize_voice_leading(chord_pcs, low_midi=48, high_midi=65)
        bass_roots = [41, 37, 46, 41]  # F2, Db2, Bb2, F2

        bpm = 132.0
        beat_s = 60.0 / bpm
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        bass = pretty_midi.Instrument(program=33, name="Bass")
        chords = pretty_midi.Instrument(program=0, name="Chords")
        lead = pretty_midi.Instrument(program=0, name="Lead")

        for bar in range(4):
            b0 = bar * 4.0
            for p in voicings[bar]:
                chords.notes.append(pretty_midi.Note(82, p, b0 * beat_s, (b0 + 3.5) * beat_s))
            bass.notes.append(pretty_midi.Note(100, bass_roots[bar], b0 * beat_s, (b0 + 3.25) * beat_s))

        # Descending dark motif in MIDI 63..72 with 2 strong-beat appoggiaturas out of 7 (5/7 = 0.714 >= 0.65, < 0.80)
        melody_beats = [
            # Bar 1 (Fm: F, Ab, C): G4(67, appoggiatura!) -> F4(65) -> Ab4(68) -> F4(65)
            (0.0, 0.5, 67), (0.5, 0.5, 65), (1.5, 0.5, 68), (2.0, 0.75, 65),
            # Bar 2 (Dbmaj7: Db, F, Ab, C): Ab4(68) -> F4(65) -> F4(65)
            (4.0, 0.75, 68), (5.0, 0.5, 65), (6.0, 0.75, 65),
            # Bar 3 (Bbm7: Bb, Db, F, Ab): C5(72, appoggiatura!) -> Bb4(70) -> Ab4(68) -> F4(65)
            (8.0, 0.5, 72), (8.5, 0.5, 70), (9.0, 0.5, 68), (10.0, 0.75, 65),
            # Bar 4 (Fm: F, Ab, C): F4(65) -> Eb4(63) -> F4(65) [rest 14.0..16.0]
            (12.0, 0.75, 65), (13.0, 0.5, 63), (13.5, 0.75, 65),
        ]
        for st, ln, p in melody_beats:
            lead.notes.append(pretty_midi.Note(90, p, st * beat_s, (st + ln) * beat_s))

        pm.instruments.extend([bass, chords, lead])
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            path = tmp.name
        try:
            pm.write(path)
            # Strict consonant mode MUST fail due to strong_beat_chord_tone_ratio (0.714 < 0.80)
            res_cons = audit_midi_file(path, expected_key="F minor", max_pitch=72, strict_musicality=True, valence=0.2, tension=0.2)
            self.assertFalse(res_cons["gate_passed"], res_cons)

            # Dark/tense mode MUST pass (strong_beat_ratio >= 0.65)
            res_dark = audit_midi_file(path, expected_key="F minor", max_pitch=72, strict_musicality=True, valence=-0.7, tension=0.6)
            self.assertTrue(res_dark["gate_passed"], res_dark)
            self.assertEqual(res_dark["too_high_count"], 0)
            self.assertGreaterEqual(res_dark["musicality_metrics"]["strong_beat_chord_tone_ratio"], 0.65)

            # Enforcing max_pitch=70 MUST fail because C5=72 > 70
            res_high = audit_midi_file(path, expected_key="F minor", max_pitch=70, strict_musicality=True, valence=-0.7, tension=0.6)
            self.assertFalse(res_high["gate_passed"], res_high)
            self.assertGreater(res_high["too_high_count"], 0)
        finally:
            os.remove(path)

    def test_dark_mode_allows_up_to_2_diatonic_clashes_and_rejects_excess_or_nondiatonic(self):
        chord_pcs = [[5, 8, 0], [1, 5, 8, 0], [10, 1, 5, 8], [5, 8, 0]]
        voicings = optimize_voice_leading(chord_pcs, low_midi=48, high_midi=65)
        bass_roots = [41, 37, 46, 41]
        bpm = 132.0
        beat_s = 60.0 / bpm

        def build_with_clashes(clash_notes):
            pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
            bass = pretty_midi.Instrument(program=33, name="Bass")
            chords = pretty_midi.Instrument(program=0, name="Chords")
            lead = pretty_midi.Instrument(program=0, name="Lead")
            for bar in range(4):
                b0 = bar * 4.0
                for p in voicings[bar]:
                    chords.notes.append(pretty_midi.Note(82, p, b0 * beat_s, (b0 + 3.5) * beat_s))
                bass.notes.append(pretty_midi.Note(100, bass_roots[bar], b0 * beat_s, (b0 + 3.25) * beat_s))
            for st, dur, p in clash_notes:
                lead.notes.append(pretty_midi.Note(90, p, st * beat_s, (st + dur) * beat_s))
            pm.instruments.extend([bass, chords, lead])
            return pm

        # Find C in bar 0 voicing -> Db (13 st or 1 st above C) is diatonic in F minor
        c_p = [p for p in voicings[0] if p % 12 == 0][0]
        db_clash = c_p + 13 if c_p + 13 <= 74 else c_p + 1
        c_hi = db_clash - 1  # C5(72) if db_clash=73

        # 2 diatonic clashes in bar 0 (stepwise C5 -> Db5 -> C5 -> Db5 -> C5 -> Bb4 -> Ab4) -> PASS in dark mode, FAIL in consonant
        notes_2_diatonic = [
            (0.0, 0.5, c_hi), (0.5, 0.4, db_clash), (1.0, 0.5, c_hi), (1.5, 0.4, db_clash), (2.0, 0.75, c_hi),
            (4.0, 0.75, 68), (5.0, 0.5, 65), (6.0, 0.75, 65),
            (8.0, 0.75, 65), (9.0, 0.5, 68), (10.0, 0.75, 65),
            (12.0, 0.75, 65), (13.0, 0.75, 68),
        ]
        # 3 diatonic clashes in same 4-bar block -> FAIL even in dark mode
        notes_3_diatonic = notes_2_diatonic + [(2.8, 0.4, db_clash)]
        # 1 non-diatonic clash (A natural = 69 against Ab3 = 56 -> diff = 13 st) -> FAIL even in dark mode
        ab_p = [p for p in voicings[0] if p % 12 == 8][0]
        a_clash = ab_p + 13 if ab_p + 13 <= 74 else ab_p + 1
        notes_nondiatonic = [
            (0.0, 0.5, 65), (0.5, 0.4, a_clash), (1.0, 0.5, 65), (2.0, 0.75, 65),
            (4.0, 0.75, 68), (5.0, 0.5, 65), (6.0, 0.75, 65),
            (8.0, 0.75, 65), (9.0, 0.5, 68), (10.0, 0.75, 65),
            (12.0, 0.75, 65), (13.0, 0.75, 68),
        ]

        for label, nlist in [("2_diat", notes_2_diatonic), ("3_diat", notes_3_diatonic), ("non_diat", notes_nondiatonic)]:
            pm = build_with_clashes(nlist)
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
                path = tmp.name
            try:
                pm.write(path)
                res_dark = audit_midi_file(path, expected_key="F minor", max_pitch=74, strict_musicality=True, valence=-0.7, tension=0.7)
                if label == "2_diat":
                    self.assertTrue(res_dark["gate_passed"], res_dark)
                    self.assertEqual(len(res_dark["minor_2nd_clashes"]), 2)
                    self.assertEqual(len(res_dark["blocking_minor_2nd_clashes"]), 0)
                    # Consonant mode must still reject the 2 clashes
                    res_cons = audit_midi_file(path, expected_key="F minor", max_pitch=74, strict_musicality=True, valence=0.0, tension=0.2)
                    self.assertFalse(res_cons["gate_passed"], res_cons)
                else:
                    self.assertFalse(res_dark["gate_passed"], f"{label} should fail: {res_dark}")
                    self.assertGreaterEqual(len(res_dark["blocking_minor_2nd_clashes"]), 1)
            finally:
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
