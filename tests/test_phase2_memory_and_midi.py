#!/usr/bin/env python3
"""
Phase 2 unit & performance tests:
  - Unified SQLite memory (`samples.db`): < 25ms on 5,000 rows without CLAP,
    warning='clap_index_missing_fallback_dsp', 0 PyTorch/CLAP imports, hybrid filters.
  - Multi-meter MIDI Gate (`4/4`, `3/4`, `6/8`, `5/4`, `free_ambient`, missing bass/lead).
  - Deterministic MIDI Auto-Fix (`scripts/midi_autofix.py` + `verify_midi_gate(..., auto_fix=True)`).
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest

import pretty_midi

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "mcp-servers", "audio-analyzer"), os.path.join(ROOT, "mcp-servers", "audio-index")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from deep_diagnostics import audit_midi_file
from indexer import init_db
from midi_autofix import autofix_midi
from musicality_metrics import evaluate_midi_musicality
from select_sample import select_best_from_db as cli_select_best_samples
from studio_lib import verify_midi_gate


def _add_note(inst: pretty_midi.Instrument, pitch: int, start_q: float, dur_q: float, vel: int = 90, bpm: float = 120.0) -> None:
    sec_per_q = 60.0 / bpm
    inst.notes.append(
        pretty_midi.Note(
            velocity=vel,
            pitch=pitch,
            start=start_q * sec_per_q,
            end=(start_q + dur_q) * sec_per_q,
        )
    )


def _create_5k_sqlite_db(db_path: str) -> None:
    conn = init_db(db_path)
    sample_rows = []
    for i in range(5000):
        cat = "kick" if i % 5 == 0 else ("lead" if i % 5 == 1 else "pad")
        bpm = 120.0 if i % 2 == 0 else 140.0
        cent = 180.0 + (i % 50) * 80.0
        sample_rows.append(
            (
                f"/tmp/fake_pack/{cat}_{i:04d}.wav",
                f"{cat}_{i:04d}.wav",
                "fake_pack",
                f"{cat}_{i:04d}.wav",
                ".wav",
                1700000000.0,
                65536,
                cat,
                0,
                0.8,
                44100,
                1,
                -1.0,
                -14.0,
                bpm,
                "C minor" if i % 3 == 0 else None,
                0.9,
                cent,
                "dark" if cent < 1200 else "bright",
                4.0,
                "punchy",
                cat,
                None,
                None,
            )
        )
    conn.executemany(
        """
        INSERT INTO samples (
            path, filename, pack_name, rel_path, ext, file_mtime, file_size,
            category, is_loop, duration_sec, sample_rate, channels, peak_db, rms_db,
            bpm, musical_key, key_confidence, spectral_centroid_hz, brightness_band,
            attack_ms, decay_profile, tags, clap_embedding, clap_updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        sample_rows,
    )
    conn.commit()
    ids = [r[0] for r in conn.execute("SELECT id FROM samples ORDER BY id").fetchall()]
    dsp_rows = []
    for idx, sid in enumerate(ids):
        playable = 1 if idx % 5 in (1, 2) else 0
        cent = 180.0 + (idx % 50) * 80.0
        r_harsh = 0.05 + (idx % 10) * 0.03
        lm_corr = 0.92 if idx % 4 != 0 else 0.30
        b_sub = 0.45 if idx % 5 == 0 else 0.08
        b_bass = 0.35 if idx % 5 == 0 else 0.15
        dsp_rows.append(
            (
                sid,
                "tonal_oneshot" if playable else "perc_oneshot",
                playable,
                65.4 if playable else None,
                36 if idx % 5 == 0 else 60,
                0,
                "C4" if idx % 3 == 0 else "G3",
                cent,
                r_harsh,
                -22.0,
                0.08,
                0.05,
                0.75 if playable else 0.25,
                0.15,
                18.0,
                lm_corr,
                0.96,
                11.0,
                4.5,
                -1.0,
                0.8,
                0.1,
                1,
                0.0,
                0,
                0,
                "[]",
                b_sub,
                b_bass,
                0.10,
                0.06,
                0.03,
                0.01,
                None,
                "[]",
                1700000000.0,
            )
        )
    conn.executemany(
        """
        INSERT INTO sample_dsp_metrics (
            sample_id, sample_nature, playable_chromatically, f0_hz, recommended_root_midi,
            recommended_fine_tune_ct, detected_note, centroid_hz, r_harsh, peak_harsh_db,
            roughness, sfm, r_harm, spectral_flux, hnr_db, low_mid_corr, phase_corr,
            crest_factor_db, attack_ms, peak_dbfs, duration_s, f0_std_semitones,
            onset_count, polyphony_ratio, needs_fade, hard_reject, reject_reasons,
            band_sub, band_bass, band_lowmid, band_mid, band_harsh, band_air,
            bark_24_json, inharmonic_warnings_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        dsp_rows,
    )
    conn.commit()
    conn.close()


class TestPhase2MemoryAndMidi(unittest.TestCase):
    def test_sqlite_5k_select_best_samples_fast_no_clap_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "samples_5k.db")
            _create_5k_sqlite_db(db_path)

            # Warm up SQLite OS page cache once, then measure median latency across 5 calls
            _ = cli_select_best_samples(
                role="kick",
                db_path=db_path,
                target_key="C minor",
                top_k=5,
                max_centroid_hz=1200.0,
                max_r_harsh=0.25,
                min_low_mid_corr=0.70,
                bpm_range=(118.0, 142.0),
                valence=-0.6,
            )
            timings_ms = []
            res = {}
            for _ in range(5):
                t0 = time.perf_counter()
                res = cli_select_best_samples(
                    role="kick",
                    db_path=db_path,
                    target_key="C minor",
                    top_k=5,
                    max_centroid_hz=1200.0,
                    max_r_harsh=0.25,
                    min_low_mid_corr=0.70,
                    bpm_range=(118.0, 142.0),
                    valence=-0.6,
                )
                timings_ms.append((time.perf_counter() - t0) * 1000.0)
            median_ms = sorted(timings_ms)[len(timings_ms) // 2]
            self.assertLess(median_ms, 25.0, f"5k query took {median_ms:.2f} ms (limit < 25 ms)")
            self.assertEqual(res.get("warning"), "clap_index_missing_fallback_dsp")
            self.assertEqual(res["query"]["backend"], "sqlite")
            self.assertGreater(len(res["recommendations"]), 0)
            for rec in res["recommendations"]:
                self.assertLessEqual(rec["dsp_summary"]["spectral_centroid_hz"], 1200.0)
                self.assertLessEqual(rec["dsp_summary"]["r_harsh"], 0.25)

            # Also verify playable_chromatically SQL filter works
            res_tonal = cli_select_best_samples(
                role="lead_melody",
                db_path=db_path,
                target_key="C minor",
                top_k=5,
                playable_chromatically=True,
                max_centroid_hz=2000.0,
            )
            self.assertGreater(len(res_tonal["recommendations"]), 0)
            self.assertTrue(all(r["dsp_summary"]["playable_chromatically"] for r in res_tonal["recommendations"]))

    def test_audio_index_mcp_tools_use_sqlite_without_clap_hang(self) -> None:
        import audio_index_server as ais

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "samples_mcp.db")
            _create_5k_sqlite_db(db_path)

            t0 = time.perf_counter()
            raw = ais.select_best_samples(
                role="kick",
                target_key="C minor",
                db_path=db_path,
                top_k=5,
                max_centroid_hz=1200.0,
                max_r_harsh=0.25,
                min_low_mid_corr=0.70,
                bpm_min=115.0,
                bpm_max=145.0,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            payload = json.loads(raw)
            self.assertEqual(payload.get("warning"), "clap_index_missing_fallback_dsp")
            self.assertGreater(len(payload.get("recommendations", [])), 0)
            self.assertLess(elapsed_ms, 250.0)

            search_raw = ais.search_samples(
                category="kick",
                playable_chromatically=False,
                max_centroid_hz=1000.0,
                limit=5,
                db_path=db_path,
            )
            search_payload = json.loads(search_raw)
            self.assertGreater(search_payload["count"], 0)
            self.assertIn("results", search_payload)
            self.assertEqual(search_payload.get("warning"), "clap_index_missing_fallback_dsp")

    def test_multimeter_3_4_and_6_8_and_5_4_pass(self) -> None:
        """
        Verify strong beats are evaluated accurately per meter:
          - 3/4: beat 0 strong, beats 1 & 2 passing (breaks old beats[::2] bug on beat 2)
          - 6/8: eighths 0 & 3 (0.0 & 1.5 qn) strong, passing tones on 1, 2, 4, 5
          - 5/4: beats 0 & 3 strong, passing tones on beats 1, 2, 4
        """
        with tempfile.TemporaryDirectory() as tmp:
            # 1) 3/4 waltz (4 bars of 3 quarters = 12 quarters): C minor: Cm -> Fm -> G -> Cm
            pm_34 = pretty_midi.PrettyMIDI(initial_tempo=120.0)
            pm_34.time_signature_changes.append(pretty_midi.TimeSignature(3, 4, 0.0))
            lead_34 = pretty_midi.Instrument(program=0, name="Lead Synth")
            chords_34 = pretty_midi.Instrument(program=89, name="Chords Pad")
            bass_34 = pretty_midi.Instrument(program=38, name="Bass")

            bars_34 = [
                ([48, 51, 55, 60], 36, [(0.0, 0.75, 67), (1.0, 0.75, 65), (2.0, 0.5, 62)]),  # G4 strong, F/D passing (beat 2 non-chord!)
                ([53, 56, 60], 41, [(3.0, 0.75, 60), (4.0, 0.75, 62)]),                     # C4 strong, D4 passing + rest on beat 2
                ([55, 58, 62], 43, [(6.0, 0.75, 62), (7.0, 0.75, 60), (8.0, 0.5, 62)]),     # D4 strong, C4 passing (no clash!)
                ([48, 51, 55, 60], 36, [(9.0, 1.5, 63)]),                                   # Eb4 strong + rest
            ]
            for bar_idx, (c_pitches, b_pitch, mel) in enumerate(bars_34):
                b0 = bar_idx * 3.0
                for cp in c_pitches:
                    _add_note(chords_34, cp, b0, 2.8, vel=70)
                _add_note(bass_34, b_pitch, b0, 2.5, vel=90)
                for s_q, d_q, p in mel:
                    _add_note(lead_34, p, s_q, d_q, vel=95)
            pm_34.instruments.extend([lead_34, chords_34, bass_34])
            p34 = os.path.join(tmp, "waltz_3_4.mid")
            pm_34.write(p34)

            r34 = audit_midi_file(p34, expected_key="C minor", time_signature=(3, 4))
            self.assertEqual(r34["time_signature"], "3/4")
            self.assertGreaterEqual(r34["musicality_metrics"]["strong_beat_chord_tone_ratio"], 0.99, r34)
            self.assertTrue(r34["gate_passed"], r34)

            # 2) 6/8 compound duple (bar = 3.0 qn; strong beats at 0.0 and 1.5 qn)
            # Notice weak eighths (1, 2, 4, 5 = 0.5, 1.0, 2.0, 2.5 qn) hold non-chord stepwise passing tones!
            # Old code checked beats=[0, 0.75, 1.5, ...]? Even worse without TS or with 4/4 fallback 1.0 qn would fail.
            pm_68 = pretty_midi.PrettyMIDI(initial_tempo=120.0)
            pm_68.time_signature_changes.append(pretty_midi.TimeSignature(6, 8, 0.0))
            lead_68 = pretty_midi.Instrument(program=0, name="Lead Bell")
            chords_68 = pretty_midi.Instrument(program=89, name="Chords")
            bass_68 = pretty_midi.Instrument(program=38, name="Bass")

            # Cm -> Ab -> Fm -> Cm (C aeolian) with non-clashing passing tones on weak eighths
            prog_68 = [
                ([48, 51, 55, 60], 36, [(0.0, 0.4, 67), (0.5, 0.4, 65), (1.0, 0.4, 63), (1.5, 0.6, 60)]),
                ([56, 60, 63], 44, [(3.0, 0.4, 60), (3.5, 0.4, 60), (4.5, 0.8, 63)]),
                ([53, 56, 60], 41, [(6.0, 0.4, 65), (6.5, 0.4, 67), (7.5, 0.7, 68)]),
                ([48, 51, 55, 60], 36, [(9.0, 0.8, 67), (10.5, 0.8, 63)]),
            ]
            for idx, (c_p, b_p, mel) in enumerate(prog_68):
                b0 = idx * 3.0
                for cp in c_p:
                    _add_note(chords_68, cp, b0, 2.8, vel=68)
                _add_note(bass_68, b_p, b0, 1.4, vel=90)
                _add_note(bass_68, b_p, b0 + 1.5, 1.2, vel=84)
                for sq, dq, p in mel:
                    _add_note(lead_68, p, sq, dq, vel=92)
            pm_68.instruments.extend([lead_68, chords_68, bass_68])
            p68 = os.path.join(tmp, "compound_6_8.mid")
            pm_68.write(p68)

            r68 = audit_midi_file(p68, expected_key="C minor")
            self.assertEqual(r68["time_signature"], "6/8")
            self.assertGreaterEqual(r68["musicality_metrics"]["strong_beat_chord_tone_ratio"], 0.75, r68)
            self.assertTrue(r68["gate_passed"], r68)

            # 3) 5/4 meter (bar = 5.0 qn; strong beats at 0.0 & 3.0 qn)
            pm_54 = pretty_midi.PrettyMIDI(initial_tempo=120.0)
            pm_54.time_signature_changes.append(pretty_midi.TimeSignature(5, 4, 0.0))
            lead_54 = pretty_midi.Instrument(program=0, name="Lead Synth")
            chords_54 = pretty_midi.Instrument(program=89, name="Chords")
            bass_54 = pretty_midi.Instrument(program=38, name="Bass")
            prog_54 = [
                ([48, 51, 55, 60], 36, [(0.0, 0.8, 67), (1.0, 0.8, 65), (2.0, 0.8, 62), (3.0, 1.0, 63)]),  # D on beat 2 passing, Eb on beat 3 chord tone!
                ([53, 56, 60], 41, [(5.0, 0.8, 65), (6.0, 0.8, 67), (8.0, 1.0, 68)]),
                ([55, 58, 62], 43, [(10.0, 0.8, 67), (11.0, 0.8, 65), (13.0, 1.0, 62)]),
                ([48, 51, 55, 60], 36, [(15.0, 2.0, 60)]),
            ]
            for idx, (c_p, b_p, mel) in enumerate(prog_54):
                b0 = idx * 5.0
                for cp in c_p:
                    _add_note(chords_54, cp, b0, 4.6, vel=68)
                _add_note(bass_54, b_p, b0, 2.8, vel=90)
                _add_note(bass_54, b_p, b0 + 3.0, 1.6, vel=82)
                for sq, dq, p in mel:
                    _add_note(lead_54, p, sq, dq, vel=92)
            pm_54.instruments.extend([lead_54, chords_54, bass_54])
            p54 = os.path.join(tmp, "meter_5_4.mid")
            pm_54.write(p54)

            r54 = audit_midi_file(p54, expected_key="C minor", time_signature=(5, 4))
            self.assertEqual(r54["time_signature"], "5/4")
            self.assertGreaterEqual(r54["musicality_metrics"]["strong_beat_chord_tone_ratio"], 0.99, r54)
            self.assertTrue(r54["gate_passed"], r54)

    def test_free_ambient_and_missing_bass_or_lead(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Pad-only free_ambient (no lead, no bass, continuous warm triads)
            pm_amb = pretty_midi.PrettyMIDI(initial_tempo=72.0)
            pm_amb.time_signature_changes.append(pretty_midi.TimeSignature(3, 4, 0.0))
            pad = pretty_midi.Instrument(program=89, name="Strings Pad")
            for idx, chord in enumerate(([57, 60, 64], [55, 59, 64], [53, 57, 60], [52, 55, 59])):
                for p in chord:
                    _add_note(pad, p, idx * 3.0, 2.9, vel=72, bpm=72.0)
            pm_amb.instruments.append(pad)
            p_amb = os.path.join(tmp, "ambient_pad_only.mid")
            pm_amb.write(p_amb)

            r_amb = audit_midi_file(p_amb, expected_key="A minor", phrase_plan="free_ambient")
            self.assertIsNone(r_amb["musicality_metrics"]["contrary_or_oblique_ratio"])
            self.assertEqual(r_amb["musicality_verdict"], "PASS", r_amb)
            self.assertTrue(r_amb["gate_passed"], r_amb)

            # Lead + Chords without Bass -> contrary_or_oblique_ratio must be None and gate PASS
            pm_nb = pretty_midi.PrettyMIDI(initial_tempo=100.0)
            lead_nb = pretty_midi.Instrument(program=0, name="Lead Flute")
            chords_nb = pretty_midi.Instrument(program=48, name="Chords Strings")
            for idx, (c_p, mel) in enumerate(
                [
                    ([57, 60, 64], [(0.0, 0.8, 69), (1.0, 0.8, 71), (2.0, 0.8, 72)]),
                    ([53, 57, 60], [(4.0, 0.9, 69), (5.0, 0.9, 67)]),
                    ([55, 59, 62], [(8.0, 0.8, 67), (9.0, 0.8, 65), (10.0, 0.8, 64)]),
                    ([57, 60, 64], [(12.0, 2.2, 64)]),
                ]
            ):
                for p in c_p:
                    _add_note(chords_nb, p, idx * 4.0, 3.8, vel=65, bpm=100.0)
                for sq, dq, mp in mel:
                    _add_note(lead_nb, mp, sq, dq, vel=90, bpm=100.0)
            pm_nb.instruments.extend([lead_nb, chords_nb])
            p_nb = os.path.join(tmp, "no_bass.mid")
            pm_nb.write(p_nb)

            r_nb = evaluate_midi_musicality(p_nb)
            self.assertIsNone(r_nb["metrics"]["contrary_or_oblique_ratio"])
            self.assertEqual(r_nb["musicality_verdict"], "PASS", r_nb)

    def test_midi_autofix_repairs_all_five_rules_and_reports_structural_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Build a broken C minor MIDI violating:
            # 1) out_of_scale (F#4=66), 2) too_high (> 74: G5=79), 3) too_low (< 31: C1=24),
            # 4) sub_collisions (chords voiced down in C2-G2 octave < 48),
            # 5) strong_beat_chord_tone_ratio (F4/D4 shifted <=2 st from chord tones E/C on beats 0 & 2)
            # While keeping valid rest_ratio, stepwise contour, and contrary bass!
            pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
            lead = pretty_midi.Instrument(program=0, name="Lead Keys")
            chords = pretty_midi.Instrument(program=4, name="Chords EP")
            bass = pretty_midi.Instrument(program=38, name="808 Sub")

            # Bar 0 (0..4 qn): Cm (bass C1=24 too low!, chords C2-Eb2-G2 < 48!)
            # Lead plays: beat 0 D4(62, non-chord 1st from Eb4!), beat 1 Eb4(63), beat 2 F#4(66, out_of_scale!), beat 3 F4(65)
            for p in (36, 39, 43):
                _add_note(chords, p, 0.0, 3.8, vel=68)
            _add_note(bass, 24, 0.0, 3.8, vel=95)  # C1=24 < 31 -> autofix +12 to C2=36!
            _add_note(lead, 62, 0.0, 0.8, vel=92)  # beat 0 strong -> D4 (1st from Eb4) -> should snap!
            _add_note(lead, 63, 1.0, 0.8, vel=90)
            _add_note(lead, 66, 2.0, 0.8, vel=90)  # F#4 out of scale! -> snaps to F4/G4
            _add_note(lead, 65, 3.0, 0.7, vel=88)

            # Bar 1 (4..8 qn): Ab (chords Ab1-C2-Eb2 < 48!), bass Ab2=44 (UP from C2=36 while lead steps DOWN!)
            for p in (32, 36, 39):
                _add_note(chords, p, 4.0, 3.8, vel=68)
            _add_note(bass, 44, 4.0, 3.8, vel=92)
            _add_note(lead, 79, 4.0, 0.9, vel=95)  # G5=79 too high (>74) -> -12 to G4=67, then strong-beat snaps -> Ab4=68
            _add_note(lead, 67, 5.0, 0.8, vel=90)
            _add_note(lead, 65, 6.0, 0.8, vel=88)  # F4 -> snaps to Eb4(63) or passes

            # Bar 2 (8..12 qn): Fm (chords F2-Ab2-C3 < 48), bass F2=41 (DOWN from Ab2=44 while lead steps UP!)
            for p in (41, 44, 48):
                _add_note(chords, p, 8.0, 3.8, vel=68)
            _add_note(bass, 41, 8.0, 3.8, vel=92)
            _add_note(lead, 65, 8.0, 0.9, vel=92)  # F4 chord tone
            _add_note(lead, 67, 9.0, 0.8, vel=90)
            _add_note(lead, 68, 10.0, 0.9, vel=92)  # Ab4 chord tone
            _add_note(lead, 70, 11.0, 0.7, vel=88)

            # Bar 3 (12..16 qn): Cm, bass C2=36 (DOWN from F2=41 while lead stays/steps)
            for p in (36, 39, 43):
                _add_note(chords, p, 12.0, 3.8, vel=68)
            _add_note(bass, 36, 12.0, 3.8, vel=92)
            _add_note(lead, 72, 12.0, 0.9, vel=94)  # C5 chord tone
            _add_note(lead, 70, 13.0, 0.8, vel=90)
            _add_note(lead, 67, 14.0, 0.9, vel=92)  # G4 chord tone

            pm.instruments.extend([lead, chords, bass])
            broken_path = os.path.join(tmp, "broken.mid")
            fixed_path = os.path.join(tmp, "fixed.mid")
            pm.write(broken_path)

            with self.assertRaises(RuntimeError):
                verify_midi_gate(broken_path, expected_key="C minor", min_pitch=31, max_pitch=74, auto_fix=False)

            res = verify_midi_gate(
                broken_path,
                expected_key="C minor",
                min_pitch=31,
                max_pitch=74,
                auto_fix=True,
                fixed_out_path=fixed_path,
            )
            self.assertTrue(os.path.isfile(fixed_path))
            self.assertTrue(res["gate_passed"], res)
            counts = res["fix_report"]["counts_by_rule"]
            self.assertGreater(counts["out_of_scale"], 0)
            self.assertGreater(counts["too_high_count"], 0)
            self.assertGreater(counts["too_low_count"], 0)
            self.assertGreater(counts["sub_collisions"], 0)

            # Structural failure check: zero-rest rapid lead must still fail after autofix
            pm_struct = pretty_midi.PrettyMIDI(initial_tempo=120.0)
            l_struct = pretty_midi.Instrument(program=0, name="Lead Keys")
            c_struct = pretty_midi.Instrument(program=4, name="Chords Pad")
            b_struct = pretty_midi.Instrument(program=38, name="Bass")
            for p in (48, 51, 55):
                _add_note(c_struct, p, 0.0, 8.0, vel=68)
            _add_note(b_struct, 36, 0.0, 8.0, vel=90)
            for step in range(32):  # zero rests across all 8 quarter notes -> rest_ratio=0.0!
                _add_note(l_struct, 60 if step % 2 == 0 else 63, step * 0.25, 0.25, vel=90)
            pm_struct.instruments.extend([l_struct, c_struct, b_struct])
            p_struct = os.path.join(tmp, "structural_fail.mid")
            pm_struct.write(p_struct)

            rep = autofix_midi(p_struct, expected_key="C minor", min_pitch=31, max_pitch=74)
            self.assertFalse(rep["gate_passed"])
            self.assertTrue(any("melody_rest_ratio" in s for s in rep["remaining_structural_failures"]))


if __name__ == "__main__":
    unittest.main()
