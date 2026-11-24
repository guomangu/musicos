#!/usr/bin/env python3
"""
Phase 6 CI Tests — Multi-Scale H x V Synchronization & Token-Lean Architecture.

Deterministic synthetic audio & MIDI tests (0 external assets, 0 PyTorch/GPU).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
import shutil

import numpy as np
import soundfile as sf

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
ANALYZER_DIR = os.path.join(ROOT, "mcp-servers", "audio-analyzer")
for _p in (ROOT, SCRIPTS_DIR, ANALYZER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from artifact_cache import (
    compute_e1_track_hash,
    compute_e2_vertical_hash,
    compute_e3_seam_hash,
    get_cached_verdict,
    set_cached_verdict,
)
from loop_first_runner import LoopFirstStateMachine
from multiscale_pyramid import build_multiscale_pyramid
from target_fingerprint import extract_target_fingerprint, validate_multi_scale_contract


def _synth_ref_wav(path: str, *, sr: int = 44100, dur_s: float = 6.0, dark: bool = False) -> None:
    t = np.linspace(0.0, dur_s, int(sr * dur_s), endpoint=False)
    # Rich low-end + mid + optional high harmonics + stereo decorrelation above 1kHz
    sub = 0.35 * np.sin(2.0 * np.pi * 55.0 * t)
    mid = 0.16 * np.sin(2.0 * np.pi * 220.0 * t) + 0.10 * np.sin(2.0 * np.pi * 440.0 * t)
    hi_amp = 0.02 if dark else 0.14
    hi_l = hi_amp * np.sin(2.0 * np.pi * 6500.0 * t)
    hi_r = hi_amp * np.sin(2.0 * np.pi * 6500.0 * t + 1.1)
    # Pulsed envelope so onset density & breathing ratio are non-trivial
    env = 0.35 + 0.65 * np.maximum(0.0, np.sin(2.0 * np.pi * 2.0 * t))
    l = (sub + mid) * env + hi_l * env
    r = (sub + mid) * env + hi_r * env
    sf.write(path, np.column_stack([l, r]).astype(np.float32), sr, subtype="PCM_24")


class TestPhase6Sprint1Foundations(unittest.TestCase):
    def test_target_fingerprint_fallback_and_reference_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proj_fb = os.path.join(tmp, "proj_fb")
            os.makedirs(proj_fb, exist_ok=True)
            bp = {
                "title": "DarkTest",
                "genre": "dark trap",
                "mix_profile": "trap_rap",
                "has_drums": True,
                "emotion_vector": {"valence": -0.7, "tension": 0.6, "energy": 0.75, "depth": 0.7},
                "bpm": 140,
                "time_signature": {"numerator": 4, "denominator": 4},
                "key": "E phrygian",
                "harmonic_core": {
                    "key": "E",
                    "mode": "phrygian",
                    "progression_bars": 4,
                    "chord_progression": [
                        {"bar": 1, "symbol": "Em", "bass_midi": 40, "voicing_midi": [52, 55, 59]}
                    ],
                    "motif_cell": {"intervals_from_tonic": [0, 1, 3], "rhythm_beats": [0.5, 0.5, 1.0], "apex_midi": 72},
                    "phrase_plan": "A-A'-A''-B",
                },
                "structure": [
                    {"section": "Intro", "start_bar": 1, "bars": 4, "energy": 0.3},
                    {"section": "Drop", "start_bar": 5, "bars": 8, "energy": 0.9},
                    {"section": "Outro", "start_bar": 13, "bars": 4, "energy": 0.2},
                ],
                "tracks": [{"name": "Kick", "role": "kick", "source_type": "gridline_native", "mixer_track": 1}],
            }
            with open(os.path.join(proj_fb, "song_blueprint.json"), "w", encoding="utf-8") as fh:
                json.dump(bp, fh)

            t0 = time.perf_counter()
            res_fb = extract_target_fingerprint([], "trap_rap", bp["emotion_vector"], project_dir=proj_fb)
            dt_fb = time.perf_counter() - t0
            self.assertLess(dt_fb, 1.0)
            self.assertEqual(res_fb["source_mode"], "profile_fallback")
            self.assertTrue(os.path.isfile(res_fb["contract_path"]))
            self.assertTrue(os.path.isfile(res_fb["profile_path"]))
            c_fb = res_fb["multi_scale_contract"]
            validate_multi_scale_contract(c_fb)
            self.assertEqual(len(c_fb["s5_context"]["bark24_target"]), 24)
            self.assertEqual(len(c_fb["s4_form"]["sections_energy_arc"]), 3)
            self.assertEqual(len(c_fb["s3_syntax"]["tension_curve_4bar"]), 4)
            self.assertNotIn("torch", sys.modules)

            # Reference WAV override
            proj_ref = os.path.join(tmp, "proj_ref")
            os.makedirs(proj_ref, exist_ok=True)
            ref_wav = os.path.join(tmp, "ref_bright.wav")
            _synth_ref_wav(ref_wav, dark=False)

            t1 = time.perf_counter()
            res_ref = extract_target_fingerprint([ref_wav], "trap_rap", {"valence": 0.2, "tension": 0.4, "energy": 0.8}, project_dir=proj_ref)
            dt_ref = time.perf_counter() - t1
            self.assertLess(dt_ref, 2.0)
            self.assertEqual(res_ref["source_mode"], "reference_audio")
            c_ref = res_ref["multi_scale_contract"]
            validate_multi_scale_contract(c_ref)
            self.assertGreater(c_ref["s5_context"]["stereo_width_by_band"]["air"], c_ref["s5_context"]["stereo_width_by_band"]["sub"])
            self.assertNotEqual(c_ref["s5_context"]["bark24_target"], c_fb["s5_context"]["bark24_target"])

    def test_artifact_cache_hit_miss_speed_and_layer_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proj_dir = os.path.join(tmp, "cache_proj")
            os.makedirs(proj_dir, exist_ok=True)
            samp_path = os.path.join(proj_dir, "kick.wav")
            _synth_ref_wav(samp_path, dur_s=1.0)

            sm = LoopFirstStateMachine(proj_dir)
            self.assertIn("artifact_hashes", sm.state)
            self.assertIn("multiscale_scores", sm.state)

            # E1 cache hit/miss
            h1 = compute_e1_track_hash(sample_path=samp_path, bpm=132.0, role="kick", project_dir=proj_dir)
            set_cached_verdict(proj_dir, "e1:Kick", h1, {"verdict": "PASS", "score": 0.95})

            t0 = time.perf_counter()
            hit = get_cached_verdict(proj_dir, "e1:Kick", h1)
            hit_ms = (time.perf_counter() - t0) * 1000.0
            self.assertIsNotNone(hit)
            self.assertTrue(hit["cache_hit"])
            self.assertLess(hit_ms, 5.0, f"Cache hit took {hit_ms:.2f}ms (> 5ms limit)")

            # Change BPM -> cache miss
            h2 = compute_e1_track_hash(sample_path=samp_path, bpm=140.0, role="kick", project_dir=proj_dir)
            self.assertNotEqual(h1, h2)
            self.assertIsNone(get_cached_verdict(proj_dir, "e1:Kick", h2))

            # E2 layer-isolated cache test on synthetic .gridline JSON
            core_gl = os.path.join(proj_dir, "core_loop.gridline")
            gl_data = {
                "tempo": 132,
                "timeSigNum": 4,
                "timeSigDen": 4,
                "ppq": 96,
                "channels": [
                    {"id": 1, "name": "Kick", "mixerTrack": 1, "gain": 0.8},
                    {"id": 2, "name": "Lead", "mixerTrack": 2, "gain": 0.7},
                ],
                "mixer": [{"id": 0, "name": "Master"}, {"id": 1, "volume": 0.8}, {"id": 2, "volume": 0.8}],
                "patterns": [{"id": 1, "bars": 4, "notes": {"1": [{"start": 0, "pitch": 60}], "2": [{"start": 48, "pitch": 67}]}}],
                "playlist": [{"pattern": 1, "start": 0, "track": 0}],
            }
            with open(core_gl, "w", encoding="utf-8") as fh:
                json.dump(gl_data, fh)

            hv1_before = compute_e2_vertical_hash("V1_LOW_END", project_path=core_gl)
            hv4_before = compute_e2_vertical_hash("V4_LEAD", project_path=core_gl)

            # Mutate ONLY Lead gain -> V1_LOW_END hash MUST stay identical, V4_LEAD MUST change
            gl_data["channels"][1]["gain"] = 0.9
            with open(core_gl, "w", encoding="utf-8") as fh:
                json.dump(gl_data, fh)
            hv1_after = compute_e2_vertical_hash("V1_LOW_END", project_path=core_gl)
            hv4_after = compute_e2_vertical_hash("V4_LEAD", project_path=core_gl)
            self.assertEqual(hv1_before, hv1_after)
            self.assertNotEqual(hv4_before, hv4_after)

            # E3 seam hash test
            hs1 = compute_e3_seam_hash("Intro -> Drop", section_a_notes=[1, 2], section_b_notes=[3, 4], transitions_autos=[])
            hs2 = compute_e3_seam_hash("Intro -> Drop", section_a_notes=[1, 2], section_b_notes=[3, 5], transitions_autos=[])
            self.assertNotEqual(hs1, hs2)

    def test_multiscale_pyramid_budget_sorting_and_cascade_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # 1) Verbose deviations + S1 blocking -> must block S4 & S5 to None, cap <= 3, <= 1500 chars
            devs = [
                {
                    "scale": "S5_context",
                    "axis": "V",
                    "location": "master",
                    "metric": "spectral_match=0.41",
                    "diagnosis": "Very verbose diagnosis " * 12,
                    "next_action": "MATCH_SPECTRAL_PROFILE",
                    "blocking": True,
                    "severity": 0.9,
                },
                {
                    "scale": "S4_form",
                    "axis": "H",
                    "location": "bars:9-16",
                    "metric": "ssm_8bar=0.97",
                    "diagnosis": "Cloned 8 bar loop " * 10,
                    "next_action": "MUTATE_BLOCK_DENSITY_OR_LPF(bars='9-16')",
                    "blocking": True,
                    "severity": 0.95,
                },
                {
                    "scale": "S3_syntax",
                    "axis": "H",
                    "location": "track:Lead",
                    "metric": "motif_dist=0.02",
                    "diagnosis": "Robotic clone " * 10,
                    "next_action": "FIX_MOTIF_CLONE",
                    "blocking": True,
                    "severity": 0.7,
                },
                {
                    "scale": "S2_acoustic",
                    "axis": "H",
                    "location": "track:Snare",
                    "metric": "lag_ms=+14.0",
                    "diagnosis": "Sample silence offset " * 10,
                    "next_action": "TRIM_SAMPLE_START_MS(track='Snare', ms=14.0)",
                    "blocking": True,
                    "severity": 0.8,
                },
                {
                    "scale": "S1_physical",
                    "axis": "V",
                    "location": "tracks:kick+808, bars:1-4",
                    "metric": "transient_phase_corr=-0.28 (min:+0.25)",
                    "diagnosis": "Annulation de phase sous 120 Hz sur attaque Kick/808 " * 6,
                    "next_action": "INVERT_808_POLARITY(track='808')",
                    "blocking": True,
                    "severity": 1.0,
                },
            ]
            pyr = build_multiscale_pyramid(
                stage="E2_V1_LOW_END",
                scale_matrix={
                    "S1_physical": {"V": 0.32, "H": 0.92},
                    "S2_acoustic": {"V": 0.88, "H": 0.58},
                    "S3_syntax": {"V": 0.90, "H": 0.50},
                    "S4_form": {"V": 0.70, "H": 0.40},
                    "S5_context": {"V": 0.45, "H": None},
                },
                deviations=devs,
                autofix_applied=["SHIFT_BASS_MICRO_MS(delta_ms=-4.2)", "FIX_A", "FIX_B", "FIX_C"],
                project_dir=tmp,
                full_details={"raw_ssm": [[1.0, 0.9], [0.9, 1.0]]},
            )
            serialized = json.dumps(pyr)
            self.assertLessEqual(len(serialized), 1500, f"Compact pyramid too large: {len(serialized)} chars")
            self.assertEqual(pyr["blocking_scale"], "S1_physical")
            self.assertEqual(pyr["verdict"], "FIX")
            self.assertIsNone(pyr["scale_matrix"]["S4_form"]["V"])
            self.assertIsNone(pyr["scale_matrix"]["S4_form"]["H"])
            self.assertIsNone(pyr["scale_matrix"]["S5_context"]["V"])
            self.assertIsNone(pyr["scale_matrix"]["S5_context"]["H"])
            self.assertLessEqual(len(pyr["top_deviations"]), 3)
            self.assertEqual(pyr["top_deviations"][0]["scale"], "S1_physical")
            self.assertTrue(all(d["scale"] not in ("S4_form", "S5_context") for d in pyr["top_deviations"]))
            self.assertTrue(os.path.isfile(os.path.join(tmp, pyr["full_report_path"])))

            # 2) Clean PASS case
            pyr_ok = build_multiscale_pyramid(
                stage="E4_FINAL",
                scale_matrix={
                    "S1_physical": {"V": 0.92, "H": 0.88},
                    "S2_acoustic": {"V": 0.90, "H": 0.86},
                    "S3_syntax": {"V": 0.89, "H": 0.84},
                    "S4_form": {"V": 0.85, "H": 0.87},
                    "S5_context": {"V": 0.83, "H": 0.80},
                },
                deviations=[],
                project_dir=tmp,
            )
            self.assertEqual(pyr_ok["verdict"], "PASS")
            self.assertIsNone(pyr_ok["blocking_scale"])
            self.assertEqual(pyr_ok["top_deviations"], [])
            self.assertLessEqual(len(json.dumps(pyr_ok)), 1500)


class TestPhase6Sprint2SyntaxS3(unittest.TestCase):
    def _build_4bar_pm(
        self,
        *,
        bpm: float = 128.0,
        chords_prog: list[list[int]] | None = None,
        bass_roots: list[int] | None = None,
        lead_by_bar: list[list[tuple[float, float, int]]] | None = None,
        chords_offset_q: float = 0.0,
        time_sig: tuple[int, int] = (4, 4),
    ):
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        if time_sig != (4, 4):
            pm.time_signature_changes.append(pretty_midi.TimeSignature(time_sig[0], time_sig[1], 0.0))
        q_s = 60.0 / bpm
        bar_q = float(time_sig[0]) * (4.0 / float(time_sig[1]))
        kick = pretty_midi.Instrument(program=0, is_drum=True, name="Kick")
        bass = pretty_midi.Instrument(program=33, name="808 Bass")
        chords = pretty_midi.Instrument(program=0, name="Chords Pad")
        lead = pretty_midi.Instrument(program=80, name="Lead")

        c_prog = chords_prog or [
            [52, 55, 59],        # Em
            [48, 52, 55, 59],    # Cmaj7
            [57, 60, 64],        # Am
            [59, 62, 66, 69],    # Bm7
        ]
        b_list = bass_roots or [40, 36, 45, 35]
        l_bars = lead_by_bar or [
            [(0.0, 0.6, 64), (0.75, 0.5, 67), (1.5, 0.6, 66), (2.25, 0.7, 67)],
            [(0.0, 0.6, 67), (1.0, 0.6, 69), (2.0, 0.6, 67), (2.75, 0.6, 64)],
            [(0.0, 0.5, 64), (0.5, 0.5, 67), (1.25, 0.75, 72), (2.25, 0.75, 69)],
            [(0.0, 0.6, 71), (0.75, 0.5, 69), (1.5, 0.75, 66), (2.5, 0.75, 66)],
        ]

        for b_idx in range(4):
            b0 = b_idx * bar_q
            kick.notes.append(pretty_midi.Note(110, 36, b0 * q_s, (b0 + 0.3) * q_s))
            bass.notes.append(pretty_midi.Note(100, b_list[b_idx], b0 * q_s, (b0 + bar_q * 0.8) * q_s))
            c_st = max(0.0, b0 + chords_offset_q)
            for p in c_prog[b_idx]:
                chords.notes.append(pretty_midi.Note(78, p, c_st * q_s, (b0 + bar_q * 0.35) * q_s))
                chords.notes.append(pretty_midi.Note(74, p, (c_st + 1.5) * q_s, (b0 + bar_q - 0.15) * q_s))
            for sq, dq, lp in l_bars[b_idx]:
                lead.notes.append(pretty_midi.Note(96, lp, (b0 + sq) * q_s, (b0 + sq + dq) * q_s))

        pm.instruments.extend([kick, bass, chords, lead])
        return pm

    def test_s3_motif_clone_no_cohesion_valid_and_autofix(self) -> None:
        from deep_diagnostics import audit_midi_file
        from midi_autofix import autofix_midi, fix_chords_lead_overlap
        from musicality_metrics import (
            compute_harmonic_tension_curve,
            compute_motif_cohesion,
            compute_rhythmic_interlocking,
            evaluate_midi_musicality,
        )

        with tempfile.TemporaryDirectory() as tmp:
            # 1) Robotic clone: identical melody in bars 1, 2, 3, 4 on static Em chord
            bar_a = [(0.0, 0.6, 64), (0.75, 0.5, 67), (1.5, 0.6, 66), (2.25, 0.7, 64)]
            pm_clone = self._build_4bar_pm(
                chords_prog=[[52, 55, 59]] * 4,
                bass_roots=[40, 40, 40, 40],
                lead_by_bar=[bar_a, bar_a, bar_a, bar_a],
                chords_offset_q=0.0,
            )
            p_clone = os.path.join(tmp, "clone.mid")
            pm_clone.write(p_clone)

            mc_clone = compute_motif_cohesion(p_clone)
            self.assertEqual(mc_clone["verdict"], "FAIL")
            self.assertEqual(mc_clone["issue"], "ROBOTIC_LOOP_CLONE")

            ht_flat = compute_harmonic_tension_curve(p_clone, target_tension=0.35)
            self.assertEqual(ht_flat["verdict"], "FAIL")
            self.assertTrue(any("harmonic_tension_flat" in s for s in ht_flat["issues"]))

            ri_clash = compute_rhythmic_interlocking(p_clone)
            self.assertEqual(ri_clash["verdict"], "FAIL")
            self.assertGreater(ri_clash["chords_lead_attack_overlap"], 0.30)

            # Fix motif clone + chords/lead overlap via midi_autofix
            pm_clone_good_harm = self._build_4bar_pm(
                lead_by_bar=[bar_a, bar_a, bar_a, [(0.0, 0.6, 71), (0.75, 0.5, 69), (1.5, 0.75, 66), (2.5, 0.75, 66)]],
                chords_offset_q=0.0,
            )
            p_clone_gh = os.path.join(tmp, "clone_gh.mid")
            p_fixed = os.path.join(tmp, "clone_fixed.mid")
            pm_clone_good_harm.write(p_clone_gh)

            fix_rep = autofix_midi(
                p_clone_gh,
                expected_key="E minor",
                valence=0.1,
                tension=0.4,
                out_path=p_fixed,
                fix_s3=True,
                strict_s3=True,
            )
            self.assertGreater(fix_rep["counts_by_rule"]["fix_motif_clone"], 0)
            self.assertGreater(fix_rep["counts_by_rule"]["fix_chords_lead_overlap"], 0)
            self.assertTrue(fix_rep["gate_passed"], fix_rep)
            mc_post = compute_motif_cohesion(p_fixed)
            self.assertEqual(mc_post["verdict"], "PASS", mc_post)
            self.assertIsNone(mc_post["issue"])

            # 2) Erratic melody -> NO_MOTIF_COHESION
            pm_erratic = self._build_4bar_pm(
                lead_by_bar=[
                    [(0.0, 0.4, 64), (0.5, 0.4, 67)],
                    [(0.25, 0.25, 72), (1.0, 0.25, 60), (1.75, 0.25, 71), (2.5, 0.25, 62), (3.25, 0.25, 74)],
                    [(0.75, 0.25, 60), (1.25, 0.25, 74), (2.25, 0.25, 61), (3.0, 0.25, 72), (3.5, 0.25, 63)],
                    [(0.0, 1.0, 64)],
                ],
                chords_offset_q=0.5,
            )
            p_err = os.path.join(tmp, "erratic.mid")
            pm_erratic.write(p_err)
            mc_err = compute_motif_cohesion(p_err)
            self.assertEqual(mc_err["verdict"], "FAIL")
            self.assertEqual(mc_err["issue"], "NO_MOTIF_COHESION")

            # 3) Clean A-A'-A''-B phrase with natural Lead/Chords interlocking -> strict_s3 PASS
            pm_ok = self._build_4bar_pm(
                lead_by_bar=[
                    [(0.5, 0.5, 64), (1.0, 0.5, 67), (1.5, 0.5, 66), (2.5, 0.75, 67)],
                    [(0.0, 0.6, 67), (0.75, 0.5, 69), (1.75, 0.5, 67), (2.5, 0.75, 64)],
                    [(0.5, 0.5, 64), (1.0, 0.5, 67), (1.75, 0.6, 72), (2.5, 0.75, 69)],
                    [(0.5, 0.5, 71), (1.0, 0.5, 69), (2.0, 0.75, 66), (2.75, 0.75, 66)],
                ],
                chords_offset_q=0.0,
            )
            p_ok = os.path.join(tmp, "valid_s3.mid")
            pm_ok.write(p_ok)
            res_ok = evaluate_midi_musicality(p_ok, valence=0.1, tension=0.4, strict_s3=True)
            self.assertEqual(res_ok["musicality_verdict"], "PASS", res_ok)
            self.assertIn("s3_diagnostics", res_ok)
            self.assertEqual(res_ok["s3_diagnostics"]["motif_cohesion"]["verdict"], "PASS")
            self.assertEqual(res_ok["s3_diagnostics"]["harmonic_tension"]["verdict"], "PASS")
            self.assertEqual(res_ok["s3_diagnostics"]["rhythmic_interlocking"]["verdict"], "PASS")

            # Also verify fix_chords_lead_overlap standalone preserves strong_beat_chord_tone_ratio
            pm_overlap = self._build_4bar_pm(chords_offset_q=0.0)
            changed = fix_chords_lead_overlap(pm_overlap, time_signature=(4, 4), max_overlap=0.25)
            self.assertGreater(changed, 0)
            p_ov = os.path.join(tmp, "ov_fixed.mid")
            pm_overlap.write(p_ov)
            aud_ov = audit_midi_file(p_ov, expected_key="E minor", valence=0.1, tension=0.4, strict_s3=True)
            self.assertTrue(aud_ov["gate_passed"], aud_ov)


class TestPhase6Sprint3AcousticS1S2(unittest.TestCase):
    def test_s1_v_kick_bass_polarity_inversion_and_micro_shift(self) -> None:
        from deep_diagnostics import analyze_kick_bass_transient_phase
        from sample_prep_lib import build_sample_from_spec

        sr = 44100
        dur_s = 1.2
        t = np.linspace(0.0, dur_s, int(sr * dur_s), endpoint=False)
        # 2 kick attacks at t=0.1s and t=0.6s (55 Hz transient core + click)
        kick = np.zeros_like(t)
        bass_inv = np.zeros_like(t)
        bass_lag = np.zeros_like(t)
        for t0 in (0.10, 0.60):
            mask = (t >= t0) & (t < t0 + 0.25)
            dt = t[mask] - t0
            pulse = 0.70 * np.sin(2.0 * np.pi * 55.0 * dt) * np.exp(-dt * 10.0)
            click = 0.25 * np.sin(2.0 * np.pi * 1600.0 * dt) * np.exp(-dt * 120.0)
            kick[mask] += pulse + click
            bass_inv[mask] += -0.65 * np.sin(2.0 * np.pi * 55.0 * dt) * np.exp(-dt * 3.0)
            # Delay bass by +5 ms (90 degrees at 55 Hz -> phase diff ~ +5ms)
            mask_lag = (t >= t0 + 0.005) & (t < t0 + 0.255)
            dt_lag = t[mask_lag] - (t0 + 0.005)
            bass_lag[mask_lag] += 0.65 * np.sin(2.0 * np.pi * 55.0 * dt_lag) * np.exp(-dt_lag * 3.0)

        with tempfile.TemporaryDirectory() as tmp:
            p_kick = os.path.join(tmp, "kick.wav")
            p_bass_inv = os.path.join(tmp, "bass_inv.wav")
            p_bass_fixed = os.path.join(tmp, "bass_fixed.wav")
            p_bass_lag = os.path.join(tmp, "bass_lag.wav")
            sf.write(p_kick, np.column_stack([kick, kick]).astype(np.float32), sr, subtype="PCM_24")
            sf.write(p_bass_inv, np.column_stack([bass_inv, bass_inv]).astype(np.float32), sr, subtype="PCM_24")
            sf.write(p_bass_lag, np.column_stack([bass_lag, bass_lag]).astype(np.float32), sr, subtype="PCM_24")

            # 1) Opposite polarity -> r_0 < -0.5 -> INVERT_808_POLARITY
            res_inv = analyze_kick_bass_transient_phase(p_kick, p_bass_inv, sr=sr)
            self.assertEqual(res_inv["verdict"], "FIX", res_inv)
            self.assertLess(res_inv["r_0"], -0.5)
            self.assertEqual(res_inv["next_action"]["code"], "INVERT_808_POLARITY")

            # Fix with sample_prep_lib invert_phase=True -> PASS
            build_sample_from_spec(
                tmp,
                tmp,
                {"src": "bass_inv.wav", "out": "bass_fixed.wav", "invert_phase": True, "fade_in_ms": 0.5},
            )
            res_ok = analyze_kick_bass_transient_phase(p_kick, p_bass_fixed, sr=sr)
            self.assertEqual(res_ok["verdict"], "PASS", res_ok)
            self.assertGreater(res_ok["r_0"], 0.70)

            # 2) +5 ms lag -> r_0 < 0.25 -> SHIFT_BASS_MICRO_MS -> apply shift -> PASS
            res_lag = analyze_kick_bass_transient_phase(p_kick, p_bass_lag, sr=sr)
            self.assertEqual(res_lag["verdict"], "FIX", res_lag)
            self.assertEqual(res_lag["next_action"]["code"], "SHIFT_BASS_MICRO_MS")
            shift_ms = float(res_lag["next_action"]["params"]["delta_ms"])
            self.assertAlmostEqual(abs(shift_ms), 5.0, delta=2.0)

            # Apply negative shift (advance bass by -shift_ms = trim 5ms)
            shift_samples = int(round((-shift_ms) * sr / 1000.0))
            bass_aligned = np.roll(bass_lag, -shift_samples)
            res_aligned = analyze_kick_bass_transient_phase(kick, bass_aligned, sr=sr)
            self.assertEqual(res_aligned["verdict"], "PASS", res_aligned)

    def test_s1_h_sustained_spectral_flux_static_vs_modulated(self) -> None:
        from deep_diagnostics import analyze_sustained_spectral_flux

        sr = 44100
        dur_s = 1.6
        t = np.linspace(0.0, dur_s, int(sr * dur_s), endpoint=False)
        env = np.minimum(1.0, t / 0.03) * np.minimum(1.0, (dur_s - t) / 0.05)

        # Static pure organ (constant harmonic balance) -> STATIC_TIMBRE
        y_static = (0.5 * np.sin(2.0 * np.pi * 220.0 * t) + 0.25 * np.sin(2.0 * np.pi * 440.0 * t)) * env
        res_st = analyze_sustained_spectral_flux(y_static, sr=sr, role="pad")
        self.assertEqual(res_st["warning"], "STATIC_TIMBRE", res_st)
        self.assertLess(res_st["spectral_flux_norm"], 0.015)
        self.assertEqual(res_st["next_action"]["code"], "ADD_SUBTLE_MODULATION_OR_LPF_MOVEMENT")

        # Modulated pad (evolving harmonic filter sweep / LFO) -> PASS
        lfo = 0.5 * (1.0 + np.sin(2.0 * np.pi * 2.2 * t))
        y_mod = (
            0.40 * np.sin(2.0 * np.pi * 220.0 * t)
            + 0.35 * lfo * np.sin(2.0 * np.pi * 880.0 * t)
            + 0.25 * (1.0 - lfo) * np.sin(2.0 * np.pi * 1540.0 * t)
        ) * env
        res_mod = analyze_sustained_spectral_flux(y_mod, sr=sr, role="pad")
        self.assertEqual(res_mod["verdict"], "PASS", res_mod)
        self.assertIsNone(res_mod["warning"])
        self.assertGreaterEqual(res_mod["spectral_flux_norm"], 0.015)

    def test_s2_h_leading_silence_trim_and_s2_v_transient_stacking(self) -> None:
        from deep_diagnostics import analyze_audio_pocket_alignment, analyze_transient_stacking
        from sample_prep_lib import build_sample_from_spec

        sr = 44100
        bpm = 120.0  # 1 beat = 0.5 s, 1/16 = 0.125 s
        dur_s = 2.0
        n_tot = int(sr * dur_s)
        t = np.linspace(0.0, dur_s, n_tot, endpoint=False)

        with tempfile.TemporaryDirectory() as tmp:
            # 1) S2/H: 20 ms leading silence on a snare/pluck sample played at t=0.0 -> TRIM_SAMPLE_START_MS
            hit = np.zeros(int(0.45 * sr), dtype=np.float32)
            t_hit = np.linspace(0.0, 0.40, int(0.40 * sr), endpoint=False)
            sil_n = int(0.020 * sr)
            sig_hit = 0.7 * np.sin(2.0 * np.pi * 320.0 * t_hit) * np.exp(-t_hit * 18.0)
            hit[sil_n : sil_n + len(sig_hit)] = sig_hit
            p_raw = os.path.join(tmp, "snare_20ms.wav")
            p_trim = os.path.join(tmp, "snare_trimmed.wav")
            sf.write(p_raw, np.column_stack([hit, hit]), sr, subtype="PCM_24")

            res_lag = analyze_audio_pocket_alignment(
                {"Snare": p_raw},
                {"Snare": [0.0]},
                bpm=bpm,
                haas_targets_ms={"snare": 0.0},
                tolerance_ms=7.0,
                track_is_sampler={"Snare": True},
            )
            self.assertEqual(res_lag["verdict"], "FIX", res_lag)
            self.assertEqual(res_lag["next_action"]["code"], "TRIM_SAMPLE_START_MS")
            meas_ms = float(res_lag["roles"]["Snare"]["median_lag_ms"])
            self.assertAlmostEqual(meas_ms, 20.0, delta=2.0)

            # Trim measured silence via sample_prep_lib -> PASS
            build_sample_from_spec(
                tmp,
                tmp,
                {"src": "snare_20ms.wav", "out": "snare_trimmed.wav", "trim_start_ms": float(res_lag["next_action"]["params"]["ms"]), "fade_in_ms": 0.5},
            )
            res_ok = analyze_audio_pocket_alignment(
                {"Snare": p_trim},
                {"Snare": [0.0]},
                bpm=bpm,
                haas_targets_ms={"snare": 0.0},
                tolerance_ms=7.0,
                track_is_sampler={"Snare": True},
            )
            self.assertEqual(res_ok["verdict"], "PASS", res_ok)

            # 2) S2/V: 3 non-percussive stems hitting simultaneously in 1-4 kHz -> STAGGER_ATTACK_OR_STRUM
            def _make_pulse_track(onset_times: list[float], freq_hz: float) -> np.ndarray:
                y = np.zeros_like(t)
                for to in onset_times:
                    i0 = int(to * sr)
                    ln = min(n_tot - i0, int(0.09 * sr))
                    if ln > 16:
                        dt = np.arange(ln) / float(sr)
                        y[i0 : i0 + ln] += 0.55 * np.sin(2.0 * np.pi * freq_hz * dt) * np.exp(-dt * 28.0)
                return y

            onsets_same = [0.0, 0.5, 1.0, 1.5]
            stems_col = {
                "Bass": _make_pulse_track(onsets_same, 1100.0),
                "Chords": _make_pulse_track(onsets_same, 1650.0),
                "Lead": _make_pulse_track(onsets_same, 2400.0),
            }
            res_col = analyze_transient_stacking(stems_col, bpm=bpm, sr=sr, max_simultaneous=2)
            self.assertEqual(res_col["verdict"], "FIX", res_col)
            self.assertEqual(res_col["next_action"]["code"], "STAGGER_ATTACK_OR_STRUM")
            self.assertGreater(res_col["collision_ratio"], 0.15)

            # Staggered onsets across 1/16 steps -> PASS
            stems_stag = {
                "Bass": _make_pulse_track([0.0, 0.5, 1.0, 1.5], 1100.0),
                "Chords": _make_pulse_track([0.125, 0.625, 1.125, 1.625], 1650.0),
                "Lead": _make_pulse_track([0.25, 0.75, 1.25, 1.75], 2400.0),
            }
            res_stag = analyze_transient_stacking(stems_stag, bpm=bpm, sr=sr, max_simultaneous=2)
            self.assertEqual(res_stag["verdict"], "PASS", res_stag)
            self.assertEqual(res_stag["collision_steps"], 0)


class TestPhase6Sprint4MacroS4AndCatalogS5(unittest.TestCase):
    def test_s4_ssm_clone_vs_variation_and_checkerboard_novelty(self) -> None:
        from macro_structure_audit import analyze_macro_structure

        sr = 44100
        bpm = 120.0  # 1 bar = 2.0 s
        bar_s = 2.0

        def _make_bar(root_hz: float, *, brighten: float = 1.0, dense_hats: bool = False) -> np.ndarray:
            n = int(sr * bar_s)
            t = np.arange(n, dtype=np.float64) / float(sr)
            y = np.zeros(n, dtype=np.float64)
            # Kick + bass + harmonic fifth
            for b_idx in range(4):
                i0 = int(b_idx * 0.5 * sr)
                lk = min(n - i0, int(0.18 * sr))
                tk = np.arange(lk, dtype=np.float64) / float(sr)
                y[i0 : i0 + lk] += 0.45 * np.sin(2.0 * np.pi * (55.0 + 70.0 * np.exp(-tk * 35.0)) * tk) * np.exp(-tk * 12.0)
            y += 0.18 * np.sin(2.0 * np.pi * root_hz * t)
            y += (0.14 * brighten) * np.sin(2.0 * np.pi * (root_hz * 1.5) * t)
            y += (0.10 * brighten) * np.sin(2.0 * np.pi * (root_hz * 3.0) * t)
            if dense_hats:
                rng = np.random.default_rng(17)
                noise = rng.standard_normal(int(0.03 * sr))
                for st in range(16):
                    i0 = int(st * 0.125 * sr)
                    ln = min(n - i0, len(noise))
                    y[i0 : i0 + ln] += 0.16 * noise[:ln] * np.exp(-np.arange(ln) / (0.008 * sr))
            return np.column_stack([y, y])

        with tempfile.TemporaryDirectory() as tmp:
            # 1) 16-bar exact 8-bar clone -> FAIL (> 0.94) + MUTATE_BLOCK_DENSITY_OR_LPF
            bar_a = _make_bar(220.0, brighten=0.6, dense_hats=False)
            clone_song = np.vstack([bar_a] * 16)
            p_clone = os.path.join(tmp, "cloned_16bar.wav")
            sf.write(p_clone, np.clip(clone_song, -0.95, 0.95), sr, subtype="PCM_24")

            res_clone = analyze_macro_structure(
                p_clone,
                bpm=bpm,
                beats_per_bar=4,
                sections=[
                    {"section": "Couplet", "start_bar": 1, "bars": 8},
                    {"section": "Refrain", "start_bar": 9, "bars": 8},
                ],
                project_dir=tmp,
            )
            self.assertEqual(res_clone["clones_8bar"]["verdict"], "FAIL", res_clone["clones_8bar"])
            self.assertGreater(res_clone["clones_8bar"]["max_similarity"], 0.94)
            self.assertEqual(res_clone["clones_8bar"]["next_action"]["code"], "MUTATE_BLOCK_DENSITY_OR_LPF")
            # Missing transition peak across identical halves -> REINFORCE_SECTION_TRANSITION
            self.assertEqual(res_clone["novelty"]["verdict"], "FIX", res_clone["novelty"])
            self.assertTrue(os.path.isfile(res_clone["full_report_path"]))

            # 2) 16-bar evolved form: Intro/Couplet (bars 1-8) -> Drop/Refrain (bars 9-16) with LPF + harmony lift + hats
            sec1 = np.vstack([_make_bar(220.0, brighten=0.15, dense_hats=False) for _ in range(8)])
            sec2 = np.vstack([_make_bar(293.66, brighten=1.6, dense_hats=True) for _ in range(8)])
            evolved = np.vstack([sec1, sec2])
            evolved[len(sec1) :, 0] *= 1.05
            evolved[len(sec1) :, 1] *= 0.90  # wider M/S delta on section 2
            p_evo = os.path.join(tmp, "evolved_16bar.wav")
            sf.write(p_evo, np.clip(evolved, -0.95, 0.95), sr, subtype="PCM_24")

            res_evo = analyze_macro_structure(
                p_evo,
                bpm=bpm,
                beats_per_bar=4,
                sections=[
                    {"section": "Intro", "start_bar": 1, "bars": 8},
                    {"section": "Drop", "start_bar": 9, "bars": 8},
                ],
                project_dir=tmp,
            )
            self.assertEqual(res_evo["clones_8bar"]["verdict"], "PASS", res_evo["clones_8bar"])
            self.assertEqual(res_evo["novelty"]["verdict"], "PASS", res_evo["novelty"])
            detected_bar = res_evo["novelty"]["boundaries"][0]["detected_peak_bar"]
            self.assertIsNotNone(detected_bar)
            self.assertLessEqual(abs(int(detected_bar) - 9), 1)

    def test_s4_5d_arc_classical_ambient_and_cognitive_ceiling_stems(self) -> None:
        from macro_structure_audit import analyze_macro_structure

        sr = 44100
        bpm = 120.0  # 2.0 s/bar
        bar_n = int(sr * 2.0)

        def _ambient_block(n_bars: int, harm_weights: tuple[float, float, float], width_pan: float, swell_hz: float) -> np.ndarray:
            tot = n_bars * bar_n
            t = np.arange(tot, dtype=np.float64) / float(sr)
            # Slow bowed modulation so onset density & centroid evolve organically without drum hits
            env = 0.72 + 0.25 * np.sin(2.0 * np.pi * swell_hz * t)
            c1 = harm_weights[0] * np.sin(2.0 * np.pi * 220.0 * t)
            c2 = harm_weights[1] * np.sin(2.0 * np.pi * 660.0 * t)
            c3 = harm_weights[2] * np.sin(2.0 * np.pi * 1980.0 * t)
            mono = 0.22 * env * (c1 + c2 + c3)
            # Inject soft pizzicato swells at slow rate in climax
            if swell_hz > 1.2:
                for p in np.arange(0.0, n_bars * 2.0, 0.35):
                    i0 = int(p * sr)
                    ln = min(tot - i0, int(0.08 * sr))
                    if ln > 16:
                        dt = np.arange(ln) / float(sr)
                        mono[i0 : i0 + ln] += 0.14 * np.sin(2.0 * np.pi * 1320.0 * dt) * np.exp(-dt * 24.0)
            left = mono * (1.0 + width_pan)
            right = np.roll(mono, int(width_pan * 180)) * (1.0 - 0.5 * width_pan)
            return np.column_stack([left, right])

        # 12 bars drumless classical_ambient evolving via spectral brightness, width, and onset density
        b_intro = _ambient_block(4, (1.0, 0.05, 0.0), 0.0, 0.3)
        b_climax = _ambient_block(4, (0.5, 0.85, 0.75), 0.45, 1.8)
        b_outro = _ambient_block(4, (0.95, 0.08, 0.0), 0.05, 0.25)
        full_amb = np.vstack([b_intro, b_climax, b_outro])

        with tempfile.TemporaryDirectory() as tmp:
            p_amb = os.path.join(tmp, "ambient_master.wav")
            sf.write(p_amb, np.clip(full_amb, -0.95, 0.95), sr, subtype="PCM_24")

            # Stems for cognitive ceiling test:
            # Case A: Clean respect of quotas (Intro <= 2 active, Refrain <= 5 active)
            # Case B: Intro overloaded with 4 loud stems (> -28 dBFS) across all 4 bars -> PRUNE_TRACK_IN_SECTION
            stems_bad = os.path.join(tmp, "stems_bad")
            stems_good = os.path.join(tmp, "stems_good")
            os.makedirs(stems_bad, exist_ok=True)
            os.makedirs(stems_good, exist_ok=True)

            t_full = np.arange(len(full_amb), dtype=np.float64) / float(sr)
            for idx, freq in enumerate((110.0, 220.0, 330.0, 440.0, 550.0)):
                sig_bad = 0.20 * np.sin(2.0 * np.pi * freq * t_full)
                sf.write(os.path.join(stems_bad, f"track_{idx+1}.wav"), np.column_stack([sig_bad, sig_bad]), sr, subtype="PCM_24")

                sig_good = sig_bad.copy()
                if idx >= 2:
                    # Mute layers 3..5 during Intro (bars 1-4) and Outro (bars 9-12)
                    sig_good[: 4 * bar_n] = 0.0
                    sig_good[8 * bar_n :] = 0.0
                sf.write(os.path.join(stems_good, f"track_{idx+1}.wav"), np.column_stack([sig_good, sig_good]), sr, subtype="PCM_24")

            secs = [
                {"section": "Intro", "start_bar": 1, "bars": 4},
                {"section": "Climax", "start_bar": 5, "bars": 4},
                {"section": "Outro", "start_bar": 9, "bars": 4},
            ]
            res_amb_good = analyze_macro_structure(
                p_amb,
                bpm=bpm,
                beats_per_bar=4,
                sections=secs,
                stems_dir_or_map=stems_good,
                has_drums=False,
                project_dir=tmp,
            )
            self.assertEqual(res_amb_good["arc_5d"]["verdict"], "PASS", res_amb_good["arc_5d"])
            self.assertGreater(res_amb_good["arc_5d"]["arc_contrast_span"], 0.20)
            self.assertEqual(res_amb_good["cognitive_ceiling"]["verdict"], "PASS", res_amb_good["cognitive_ceiling"])

            res_amb_bad = analyze_macro_structure(
                p_amb,
                bpm=bpm,
                beats_per_bar=4,
                sections=secs,
                stems_dir_or_map=stems_bad,
                has_drums=False,
                project_dir=tmp,
            )
            self.assertEqual(res_amb_bad["cognitive_ceiling"]["verdict"], "FAIL", res_amb_bad["cognitive_ceiling"])
            self.assertEqual(res_amb_bad["cognitive_ceiling"]["next_action"]["code"], "PRUNE_TRACK_IN_SECTION")

    def test_s5_ep_catalog_coherence_camelot_and_diversity(self) -> None:
        from catalog_coherence import audit_catalog_coherence, build_master_fingerprint, key_to_camelot

        self.assertEqual(key_to_camelot("C minor"), "5A")
        self.assertEqual(key_to_camelot("G minor"), "6A")
        self.assertEqual(key_to_camelot("D minor"), "7A")

        rng = np.random.default_rng(7)
        bark_base = np.linspace(0.10, 0.01, 24)
        bark_base = (bark_base / np.sum(bark_base)).tolist()

        with tempfile.TemporaryDirectory() as tmp:
            # Build 3 valid EP tracks intentionally out of Camelot order: 7A, 5A, 6A -> optimal order: 5A -> 6A -> 7A (or reverse)
            specs = [
                ("track_alpha", 128.0, "D minor", ["i", "VI", "III", "VII"], -11.2, -4.1),
                ("track_beta", 122.0, "C minor", ["i", "iv", "v", "i"], -11.8, -4.3),
                ("track_gamma", 136.0, "G minor", ["i", "VII", "VI", "V"], -11.0, -3.9),
            ]
            for nm, bpm, key, prog, lufs, slope in specs:
                pdir = os.path.join(tmp, nm, "renders")
                os.makedirs(pdir, exist_ok=True)
                bv = np.clip(np.asarray(bark_base) + rng.uniform(-0.002, 0.002, 24), 0.0005, 1.0)
                bv = (bv / np.sum(bv)).tolist()
                fp = build_master_fingerprint(
                    project_name=nm,
                    project_dir=os.path.join(tmp, nm),
                    mix_profile="trap_rap",
                    bpm=bpm,
                    key=key,
                    degree_progression=prog,
                    integrated_lufs=lufs,
                    true_peak_dbtp=-1.2,
                    crest_factor_db=9.5,
                    pink_slope_db_per_oct=slope,
                    bark24=bv,
                    duration_s=140.0,
                )
                with open(os.path.join(pdir, "master_fingerprint.json"), "w", encoding="utf-8") as fh:
                    json.dump(fp, fh)

            rep = audit_catalog_coherence(tmp, mode="ep")
            self.assertEqual(rep["verdict"], "PASS", rep)
            self.assertTrue(rep["sound_homogeneity"]["passed"])
            self.assertTrue(rep["compositional_diversity"]["passed"])
            self.assertEqual(rep["transition_compatibility"]["optimal_compatible_edges"], "2/2")
            self.assertEqual(rep["recommended_tracklist"][1], "track_gamma")  # 6A bridges 5A and 7A

            # Collision test: same BPM±4, same key, same progression + LUFS spread > 1.5 dB
            fp_dup = build_master_fingerprint(
                project_name="track_clone",
                project_dir=os.path.join(tmp, "track_clone"),
                mix_profile="trap_rap",
                bpm=124.0,  # within ±4 of 122.0
                key="C minor",
                degree_progression=["i", "iv", "v", "i"],
                integrated_lufs=-14.5,  # spread now > 3 dB
                true_peak_dbtp=-1.2,
                crest_factor_db=10.0,
                pink_slope_db_per_oct=-4.2,
                bark24=bark_base,
                duration_s=140.0,
            )
            os.makedirs(os.path.join(tmp, "track_clone", "renders"), exist_ok=True)
            with open(os.path.join(tmp, "track_clone", "renders", "master_fingerprint.json"), "w", encoding="utf-8") as fh:
                json.dump(fp_dup, fh)

            rep_bad = audit_catalog_coherence(tmp, mode="ep")
            self.assertEqual(rep_bad["verdict"], "FIX", rep_bad)
            self.assertFalse(rep_bad["sound_homogeneity"]["passed"])
            self.assertFalse(rep_bad["compositional_diversity"]["passed"])
            self.assertEqual(len(rep_bad["compositional_diversity"]["collisions"]), 1)


class TestPhase6Sprint5AutofixPipelineAndPerformance(unittest.TestCase):
    @unittest.skipUnless(shutil.which("gridline"), "binaire gridline requis")
    def test_multiscale_autofix_dispatcher_s1_to_s4_and_guardrails(self) -> None:
        import shutil
        import pretty_midi
        from deep_diagnostics import analyze_kick_bass_transient_phase, audit_midi_file
        from loop_first_runner import LoopFirstStateMachine
        from multiscale_autofix import dispatch_multiscale_autofix

        sr = 44100
        dur_s = 1.5
        t = np.arange(int(sr * dur_s), dtype=np.float64) / float(sr)

        with tempfile.TemporaryDirectory() as tmp:
            # 1) S1/V + S2/H sample-level fixes: inverted 808 + 20ms trimmed snare in samples/
            samples_dir = os.path.join(tmp, "samples")
            os.makedirs(samples_dir, exist_ok=True)
            k_sig = np.zeros_like(t)
            b_sig = np.zeros_like(t)
            for bt in (0.0, 0.5, 1.0):
                i0 = int(bt * sr)
                ln = min(len(t) - i0, int(0.18 * sr))
                tk = np.arange(ln) / float(sr)
                wave = 0.65 * np.sin(2.0 * np.pi * 55.0 * tk) * np.exp(-tk * 10.0)
                k_sig[i0 : i0 + ln] += wave
                b_sig[i0 : i0 + ln] -= wave  # Out of phase
            p_kick = os.path.join(samples_dir, "kick.wav")
            p_808 = os.path.join(samples_dir, "808_sub.wav")
            p_snare = os.path.join(samples_dir, "snare_trim.wav")
            sn_raw = np.zeros(int(0.4 * sr), dtype=np.float64)
            tsn = np.arange(int(0.35 * sr)) / float(sr)
            sn_raw[int(0.020 * sr) : int(0.020 * sr) + len(tsn)] = 0.6 * np.sin(2.0 * np.pi * 300.0 * tsn) * np.exp(-tsn * 18.0)

            sf.write(p_kick, np.column_stack([k_sig, k_sig]), sr, subtype="PCM_24")
            sf.write(p_808, np.column_stack([b_sig, b_sig]), sr, subtype="PCM_24")
            sf.write(p_snare, np.column_stack([sn_raw, sn_raw]), sr, subtype="PCM_24")
            with open(os.path.join(samples_dir, "manifest.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "samples": [
                            {"src": "kick.wav", "out": "kick.wav"},
                            {"src": "808_sub.wav", "out": "808_sub.wav"},
                            {"src": "snare_trim.wav", "out": "snare_trim.wav"},
                        ]
                    },
                    fh,
                )

            # S3 MIDI with cloned 4-bar motif + chords-lead overlap
            p_mid = os.path.join(tmp, "master_sketch.mid")
            pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
            lead = pretty_midi.Instrument(program=80, name="Lead")
            chords = pretty_midi.Instrument(program=88, name="Chords")
            bass = pretty_midi.Instrument(program=38, name="Bass")
            for bar in range(4):
                b0 = bar * 2.0
                for j, p in enumerate((60, 62, 63, 67)):
                    lead.notes.append(pretty_midi.Note(95, p, b0 + j * 0.5, b0 + j * 0.5 + 0.35))
                for p in (48, 55, 60):
                    chords.notes.append(pretty_midi.Note(72, p, b0, b0 + 1.8))
                bass.notes.append(pretty_midi.Note(105, 36, b0, b0 + 1.5))
            pm.instruments.extend([lead, chords, bass])
            pm.write(p_mid)

            # Also test Gridline-backed SHIFT_BASS_MICRO_MS, STAGGER_ATTACK_OR_STRUM, MUTATE_BLOCK_DENSITY_OR_LPF, APPLY_OUTRO_FADE
            src_gl = os.path.join(ROOT, "projects", "validation_fm_dark", "projet.gridline")
            dst_gl = os.path.join(tmp, "projet.gridline")
            shutil.copy2(src_gl, dst_gl)
            with open(os.path.join(tmp, "song_blueprint.json"), "w", encoding="utf-8") as fh:
                json.dump({"mix_profile": "trap_rap", "bpm": 120.0, "harmonic_core": {"key": "C minor"}}, fh)

            fix_rep = dispatch_multiscale_autofix(
                tmp,
                [
                    {"code": "INVERT_808_POLARITY", "params": {"track": "808", "sample_path": p_808}},
                    {"code": "TRIM_SAMPLE_START_MS", "params": {"track": "snare_trim", "ms": 20.0, "sample_path": p_snare}},
                    {"code": "FIX_MOTIF_CLONE", "params": {}},
                    {"code": "FIX_CHORDS_LEAD_OVERLAP", "params": {}},
                    {"code": "SHIFT_BASS_MICRO_MS", "params": {"delta_ms": 6.0}},
                    {"code": "STAGGER_ATTACK_OR_STRUM", "params": {"track": "Chords"}},
                    {"code": "MUTATE_BLOCK_DENSITY_OR_LPF", "params": {"bars": "9-16"}},
                    {"code": "APPLY_OUTRO_FADE", "params": {}},
                ],
                gridline_path=dst_gl,
            )
            applied_codes = {a["code"] for a in fix_rep["applied"]}
            for req in (
                "INVERT_808_POLARITY",
                "TRIM_SAMPLE_START_MS",
                "FIX_MOTIF_CLONE",
                "SHIFT_BASS_MICRO_MS",
                "STAGGER_ATTACK_OR_STRUM",
                "MUTATE_BLOCK_DENSITY_OR_LPF",
                "APPLY_OUTRO_FADE",
            ):
                self.assertIn(req, applied_codes, fix_rep)

            # Verify polarity & MIDI were genuinely fixed
            kb_after = analyze_kick_bass_transient_phase(p_kick, p_808)
            self.assertEqual(kb_after["verdict"], "PASS", kb_after)
            midi_after = audit_midi_file(p_mid, expected_key="C minor", strict_s3=True)
            s3_co = (midi_after.get("s3_diagnostics") or {}).get("motif_cohesion") or {}
            self.assertNotEqual(s3_co.get("reason"), "ROBOTIC_LOOP_CLONE")

            # Verify user_samples_locked guardrail blocks TRIM_SAMPLE_START_MS
            locked_dir = os.path.join(tmp, "Samples")
            os.makedirs(locked_dir, exist_ok=True)
            sf.write(os.path.join(locked_dir, "user_loop.wav"), np.column_stack([k_sig, k_sig]), sr, subtype="PCM_24")
            sm = LoopFirstStateMachine(tmp)
            self.assertTrue(sm.state["user_samples_locked"])
            lock_rep = dispatch_multiscale_autofix(
                tmp, [{"code": "TRIM_SAMPLE_START_MS", "params": {"track": "snare_trim", "ms": 12.0, "sample_path": p_snare}}]
            )
            self.assertEqual(lock_rep["applied_count"], 0)
            self.assertTrue(lock_rep["escalate_to_llm"])

    @unittest.skipUnless(shutil.which("gridline"), "binaire gridline requis")
    def test_mcp_tools_cli_contract_override_and_2min_cpu_benchmark(self) -> None:
        import shutil
        import subprocess
        import pretty_midi
        import audio_analyzer_server as aas
        from render_and_audit import render_and_evaluate

        sr = 44100
        # Build a bright clean stereo 120-second (2 minute) track + stems + 32-bar MIDI
        dur_s = 120.0
        n_tot = int(sr * dur_s)
        bpm = 128.0
        bar_s = 4.0 * 60.0 / bpm
        n_bars = int(dur_s / bar_s)
        t = np.arange(n_tot, dtype=np.float32) / float(sr)

        # Modulated bass & pads + evolving brightness per 16-bar block
        kick = np.zeros(n_tot, dtype=np.float32)
        for bt in np.arange(0.0, dur_s - 0.25, 60.0 / bpm):
            i0 = int(bt * sr)
            ln = min(n_tot - i0, int(0.16 * sr))
            tk = np.arange(ln, dtype=np.float32) / float(sr)
            kick[i0 : i0 + ln] += 0.45 * np.sin(2.0 * np.pi * (55.0 + 60.0 * np.exp(-tk * 35.0)) * tk) * np.exp(-tk * 14.0)

        lfo = 1.0 + 0.35 * np.sin(2.0 * np.pi * 0.4 * t)
        bright_env = np.clip((t / dur_s), 0.1, 1.0)
        bass = 0.22 * np.sin(2.0 * np.pi * 55.0 * t) * lfo + 0.08 * np.sin(2.0 * np.pi * 110.0 * t)
        chords = 0.14 * np.sin(2.0 * np.pi * 220.0 * t) * lfo + 0.12 * bright_env * np.sin(2.0 * np.pi * 660.0 * t)
        lead = 0.10 * bright_env * np.sin(2.0 * np.pi * 1320.0 * t + 0.5 * np.sin(2.0 * np.pi * 3.0 * t))
        master_l = np.clip(kick + bass + chords + lead * 1.05, -0.92, 0.92)
        master_r = np.clip(kick + bass + 0.85 * chords + np.roll(lead, 64) * 0.90, -0.92, 0.92)

        with tempfile.TemporaryDirectory() as tmp:
            renders_dir = os.path.join(tmp, "renders")
            stems_dir = os.path.join(renders_dir, "stems")
            os.makedirs(stems_dir, exist_ok=True)
            p_master = os.path.join(renders_dir, "master.wav")
            sf.write(p_master, np.column_stack([master_l, master_r]), sr, subtype="PCM_24")
            sf.write(os.path.join(stems_dir, "Kick.wav"), np.column_stack([kick, kick]), sr, subtype="PCM_24")
            sf.write(os.path.join(stems_dir, "808_Bass.wav"), np.column_stack([bass, bass]), sr, subtype="PCM_24")
            sf.write(os.path.join(stems_dir, "Chords.wav"), np.column_stack([chords, chords]), sr, subtype="PCM_24")
            sf.write(os.path.join(stems_dir, "Lead.wav"), np.column_stack([lead, lead]), sr, subtype="PCM_24")

            # Minimal varied MIDI sketch (8 bars)
            p_mid = os.path.join(tmp, "master_sketch.mid")
            pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
            m_lead = pretty_midi.Instrument(program=80, name="Lead")
            m_chords = pretty_midi.Instrument(program=88, name="Chords")
            m_bass = pretty_midi.Instrument(program=38, name="Bass")
            beat_s = 60.0 / bpm
            for bar in range(8):
                b0 = bar * 4 * beat_s
                m_bass.notes.append(pretty_midi.Note(100, 36 if bar % 2 == 0 else 41, b0, b0 + 3 * beat_s))
                for p in (48, 55, 60):
                    m_chords.notes.append(pretty_midi.Note(72, p, b0, b0 + 3.5 * beat_s))
                offs = [0.25, 1.0, 2.25, 3.0] if bar % 2 == 0 else [0.5, 1.5, 2.5, 3.5]
                pitches = [60, 63, 65, 67] if bar < 4 else [67, 65, 63, 62]
                for step_off, pitch in zip(offs, pitches):
                    st = b0 + step_off * beat_s
                    m_lead.notes.append(pretty_midi.Note(95, pitch, st, st + 0.4 * beat_s))
            pm.instruments.extend([m_lead, m_chords, m_bass])
            pm.write(p_mid)

            with open(os.path.join(tmp, "song_blueprint.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "mix_profile": "trap_rap",
                        "has_drums": True,
                        "bpm": bpm,
                        "time_signature": {"numerator": 4, "denominator": 4},
                        "harmonic_core": {"key": "C minor", "degree_progression": ["i", "VI", "III", "VII"]},
                        "structure": [
                            {"section": "Intro", "start_bar": 1, "bars": 8, "energy": 0.30},
                            {"section": "Couplet", "start_bar": 9, "bars": 16, "energy": 0.65},
                            {"section": "Refrain", "start_bar": 25, "bars": 24, "energy": 0.90},
                            {"section": "Outro", "start_bar": 49, "bars": max(4, n_bars - 48), "energy": 0.25},
                        ],
                    },
                    fh,
                )

            # 1) MCP extract_target_fingerprint_tool using reference WAV slice
            ref_wav = os.path.join(tmp, "ref_snippet.wav")
            sf.write(ref_wav, np.column_stack([master_l[: sr * 4], master_r[: sr * 4]]), sr, subtype="PCM_24")
            fp_json = aas.extract_target_fingerprint_tool(
                audio_paths=[ref_wav],
                mix_profile="trap_rap",
                project_dir=tmp,
            )
            self.assertLessEqual(len(fp_json), 1500)
            fp_parsed = json.loads(fp_json)
            self.assertEqual(fp_parsed["source_mode"], "reference_audio")
            self.assertTrue(os.path.isfile(fp_parsed["contract_path"]))

            # 2) CPU Benchmark (< 4.0 s on 2-min master + stems + MIDI) via MCP audit_multiscale_hv
            t_start = time.perf_counter()
            hv_json = aas.audit_multiscale_hv(project_dir=tmp, stage="ALL", skip_render=True)
            cpu_elapsed = time.perf_counter() - t_start
            self.assertLess(cpu_elapsed, 4.0, f"audit_multiscale_hv exceeded 4.0s CPU budget on 2-min track: {cpu_elapsed:.2f}s")
            self.assertLessEqual(len(hv_json), 1500, f"MCP payload too big: {len(hv_json)} chars")
            hv_obj = json.loads(hv_json)
            self.assertLessEqual(len(hv_obj["top_deviations"]), 3)
            rep_p = hv_obj["full_report_path"]
            rep_abs = rep_p if os.path.isabs(rep_p) else os.path.join(tmp, rep_p)
            self.assertTrue(os.path.isfile(rep_abs), f"Missing full diagnostics report: {rep_abs}")
            with open(os.path.join(tmp, "pipeline_state.json"), encoding="utf-8") as fh:
                st_saved = json.load(fh)
            self.assertIn("multiscale_scores", st_saved)

            # 3) CLI run_autonomous_pipeline.py --json-compact stdout purity check
            proc = subprocess.run(
                [
                    os.path.join(ROOT, ".venv", "bin", "python"),
                    os.path.join(SCRIPTS_DIR, "run_autonomous_pipeline.py"),
                    "--project",
                    tmp,
                    "--stage",
                    "E4",
                    "--skip-render",
                    "--auto-fix",
                    "--max-autofix-passes",
                    "2",
                    "--json-compact",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            out_s = proc.stdout.strip()
            self.assertLessEqual(len(out_s), 1500, f"CLI compact stdout exceeded 1500 chars: {len(out_s)}")
            cli_obj = json.loads(out_s)
            self.assertEqual(cli_obj["stage"], "E4")

            # 4) Verify render_and_evaluate consumes multi_scale_contract.json threshold overrides
            src_gl = os.path.join(ROOT, "projects", "validation_fm_dark", "projet.gridline")
            gl_proj = os.path.join(tmp, "projet.gridline")
            shutil.copy2(src_gl, gl_proj)
            ev_rep = render_and_evaluate(
                project=gl_proj,
                out_wav=os.path.join(tmp, "renders", "override_test.wav"),
                profile="trap_rap",
                range_bars="1-4",
                bpm=bpm,
                use_neural=False,
                contract_path=fp_parsed["contract_path"],
            )
            self.assertTrue(ev_rep["contract_used"])
            self.assertEqual(ev_rep["effective_thresholds"]["target_lufs"], fp_parsed["target_lufs"])
            self.assertEqual(ev_rep["effective_thresholds"]["low_mids_max"], fp_parsed["low_mids_max"])


if __name__ == "__main__":
    unittest.main()
