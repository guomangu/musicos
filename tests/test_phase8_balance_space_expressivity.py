#!/usr/bin/env python3
"""Phase 8 Q1..Q7 tests: gain staging, bounded stem hierarchy, 3D FX buses,
stereo audit, external pro references, rich transitions, expressive timbres,
cache/pyramid integration and CPU/token budgets."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import soundfile as sf

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "mcp-servers", "audio-analyzer")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gl_dsl  # noqa: E402
import song_builder  # noqa: E402
from artifact_cache import compute_e1_track_hash, compute_e2_vertical_hash  # noqa: E402
from bootstrap_golden_refs import build_spectral_profiles  # noqa: E402
from deep_diagnostics import analyze_stem_hierarchy, analyze_stereo_image  # noqa: E402
from gain_staging_lib import (  # noqa: E402
    ROLE_TARGET_OFFSETS_DB,
    compute_gain_staging_plan,
    measure_active_loudness,
)
from loop_first_runner import LoopFirstStateMachine  # noqa: E402
from macro_structure_audit import audit_section_track_audibility  # noqa: E402
from multiscale_pyramid import build_multiscale_pyramid  # noqa: E402
from studio_lib import GridlineMCP  # noqa: E402
from taste_memory import record_validated_taste  # noqa: E402
from target_fingerprint import extract_target_fingerprint  # noqa: E402
from transitions_lib import (  # noqa: E402
    add_transition_fx,
    apply_harmonic_anticipation,
    generate_drum_fill,
)

SR = 44100


def _tone(freq: float, dur_s: float, amp: float = 0.2, decay: float = 0.0) -> np.ndarray:
    n = int(SR * dur_s)
    t = np.arange(n, dtype=np.float64) / SR
    y = amp * np.sin(2.0 * np.pi * freq * t)
    if decay > 0:
        y *= np.exp(-t * decay)
    return y


class TestPhase8BalanceSpaceExpressivity(unittest.TestCase):
    # ------------------------------------------------------------------ Q1
    def test_q1_active_loudness_and_gain_staging(self) -> None:
        # Sparse pluck (high crest factor) and sustained 808 (low crest factor)
        pluck = np.zeros(int(SR * 2.0))
        for k in range(4):
            i0 = int(k * 0.5 * SR)
            pluck[i0 : i0 + int(0.08 * SR)] = _tone(880.0, 0.08, amp=0.42, decay=55.0)
        bass = _tone(55.0, 2.0, amp=0.42)

        pluck_m = measure_active_loudness(pluck, SR)
        bass_m = measure_active_loudness(bass, SR)
        self.assertFalse(pluck_m["is_effectively_silent"])
        self.assertFalse(bass_m["is_effectively_silent"])
        self.assertGreater(pluck_m["crest_factor_db"], bass_m["crest_factor_db"] + 5.0)

        plan = compute_gain_staging_plan(
            {
                "Lead": {"role": "lead", "active_lufs": pluck_m["active_lufs"], "peak_dbfs": pluck_m["peak_dbfs"], "current_volume": 0.78},
                "Bass": {"role": "bass_sub", "active_lufs": bass_m["active_lufs"], "peak_dbfs": bass_m["peak_dbfs"], "current_volume": 0.62},
            },
            mix_profile="trap_rap",
            has_drums=True,
        )
        lead_t = plan["tracks"]["Lead"]
        bass_t = plan["tracks"]["Bass"]
        for t_plan in (lead_t, bass_t):
            self.assertLessEqual(t_plan["predicted_peak_dbfs"], -1.99)
            self.assertGreaterEqual(t_plan["recommended_volume"], 0.10)
            self.assertLessEqual(t_plan["recommended_volume"], 1.15)
            self.assertLessEqual(abs(t_plan["predicted_active_lufs"] - t_plan["target_active_lufs"]), 1.0, t_plan)
        self.assertEqual(lead_t["target_active_lufs"] - bass_t["target_active_lufs"], 0.0)
        self.assertEqual(lead_t["role"], "lead")
        self.assertEqual(bass_t["role"], "bass_sub")

    # ------------------------------------------------------------------ Q2
    def test_q2_bounded_stem_hierarchy_and_autofader(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stems = os.path.join(td, "stems")
            os.makedirs(stems)
            n = int(SR * 8.0)
            t = np.arange(n) / SR
            kick = np.zeros((n, 2))
            for b in range(0, n, int(0.5 * SR)):
                ln = min(n - b, int(0.12 * SR))
                tt = np.arange(ln) / SR
                sig = 0.8 * np.sin(2 * np.pi * (52 + 70 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 18)
                kick[b : b + ln, 0] += sig
                kick[b : b + ln, 1] += sig
            bass = np.column_stack([_tone(55.0, 8.0, 0.25, 0.0)] * 2)
            lead = np.column_stack([_tone(660.0, 8.0, 0.22, 0.0)] * 2)
            chords = np.column_stack([_tone(440.0, 8.0, 0.22 * 10 ** (-14 / 20), 0.0)] * 2)
            chords_quiet = np.column_stack([_tone(440.0, 8.0, 0.22 * 10 ** (-28 / 20), 0.0)] * 2)
            sf.write(os.path.join(stems, "track - 01 Kick.wav"), kick, SR, subtype="PCM_24")
            sf.write(os.path.join(stems, "track - 02 808 Bass.wav"), bass, SR, subtype="PCM_24")
            sf.write(os.path.join(stems, "track - 03 Lead.wav"), lead, SR, subtype="PCM_24")
            sf.write(os.path.join(stems, "track - 04 Chords.wav"), chords, SR, subtype="PCM_24")

            res = analyze_stem_hierarchy(stems_dir=stems, has_drums=True, mix_profile="trap_rap")
            self.assertFalse(res["checks"]["bounded_hierarchy_ok"])
            recs = [r for r in res.get("recommendations", []) if r.get("metric") == "lead_vs_harmony_db"]
            self.assertTrue(recs, res)
            rec = recs[0]
            self.assertEqual(rec["action"], "ADJUST_TRACK_FADER")
            self.assertGreaterEqual(rec["delta_db"], 1.5)
            self.assertLessEqual(rec["delta_db"], 4.0)
            diff = float(res["metrics"]["pairwise_balance_db"]["lead_vs_harmony_db"])
            self.assertGreater(diff, 9.5)

            # Two bounded passes converge inside the bilateral +2.5..+9.5 dB window
            sim = diff
            for _ in range(2):
                sim -= rec["delta_db"]  # chords fader boost reduces lead-vs-harmony
            self.assertGreaterEqual(sim, 2.5)
            self.assertLessEqual(sim, 9.5)

            # Section audibility floor: chords 18 dB under the section mix -> violation
            plan = {
                "meta": {"bpm": 120.0, "time_signature": {"numerator": 4, "denominator": 4}},
                "sections": [{"name": "Drop", "start_bar": 0, "bars": 2, "active_roles": ["kick", "chords"]}],
            }
            sess = np.column_stack([kick[:, 0] + bass[:, 0] + lead[:, 0] + chords_quiet[:, 0]] * 2)
            aud = audit_section_track_audibility(
                {"01 Kick": (kick, SR), "02 808 Bass": (bass, SR), "03 Lead": (lead, SR), "04 Chords": (chords_quiet, SR)},
                plan["sections"],
                bar_s=2.0,
                bpm=120.0,
                beats_per_bar=4,
                mix_data=(sess, SR),
            )
            self.assertEqual(aud["verdict"], "FIX", aud)
            floor = [v for v in aud["violations"] if v["code"] == "TRACK_UNDER_AUDIBLE_FLOOR"]
            self.assertTrue(floor, aud)
            self.assertGreaterEqual(floor[0]["delta_db"], 1.5)
            self.assertLessEqual(floor[0]["delta_db"], 4.0)

            # E2 fault isolation: a locked track cannot be edited
            sm = LoopFirstStateMachine(td, has_drums=True)
            sm.state["next_mandatory_step"] = {"forbidden_tracks": ["Kick"]}
            with self.assertRaises(PermissionError):
                sm.validate_edit_target("Kick")
            self.assertTrue(sm.validate_edit_target("Chords")["allowed"])

    # ------------------------------------------------------------------ Q3
    @unittest.skipUnless(shutil.which("gridline"), "binaire gridline requis")
    def test_q3_fx_buses_and_stereo_audits(self) -> None:
        dly = gl_dsl.delay_send_fx(bpm=140.0, valence=0.0)
        self.assertEqual(dly["Dry"], 0.0)
        self.assertEqual(dly["Time"], 8)  # dotted eighth
        self.assertAlmostEqual(dly["Feedback"], 0.26, places=3)
        hall = gl_dsl.reverb_hall(decay_sec=2.0, valence=-0.5)
        self.assertEqual(hall["Model"], 0)
        self.assertEqual(hall["Dry"], 0)
        self.assertEqual(hall["Dry (full)"], 0)
        self.assertEqual(hall["High cut (full)"], 45)  # dark hall 4500 Hz

        with tempfile.TemporaryDirectory() as td:
            proj = os.path.join(td, "buses.gridline")
            meta = {"bpm": 120.0, "beats_per_bar": 4, "denominator": 4, "mix_profile": "trap_rap", "valence": -0.4}
            chans = [song_builder.ChannelSpec("Pad", "pad", 1, "music", plugin="gl.warmsynth", role="pad", valence=-0.4, hall_send=0.2)]
            song_builder.build_gridline_project(
                proj_path=proj,
                export_dir=os.path.join(td, "exports"),
                samples_dir=td,
                meta=meta,
                sections=[("Main", 0, 1, "main")],
                channels=chans,
                tracks={"pad": [{"start": 0.0, "length": 3.0, "pitch": 60, "velocity": 90}]},
                markers=[],
                min_pattern_lufs=-60.0,
            )
            mcp = GridlineMCP(proj, td)
            try:
                mix = mcp.call("get_mixer", {})
                by_idx = {t["mixer"]: t for t in mix["tracks"]}
                self.assertEqual(by_idx[21]["name"], "FX Delay")
                self.assertEqual(by_idx[22]["name"], "FX Hall")
                self.assertIn("gl.delay", [e.get("plugin") for e in by_idx[21]["effects"]])
                self.assertIn("gl.reverb", [e.get("plugin") for e in by_idx[22]["effects"]])
                send_targets = {s.get("to") for s in by_idx[1].get("sends", []) if isinstance(s, dict)}
                self.assertIn(22, send_targets)
            finally:
                mcp.close()

        t = np.arange(int(SR * 5)) / SR
        mono = 0.2 * np.sin(2 * np.pi * 440 * t)
        mono_res = analyze_stereo_image(np.column_stack([mono, mono]), SR, "trap_rap")
        self.assertIn("WIDEN_HARMONY_OR_PERC", mono_res["codes"])
        left = 0.28 * np.sin(2 * np.pi * 440 * t)
        right = 0.20 * np.sin(2 * np.pi * 440 * t)
        pan_res = analyze_stereo_image(np.column_stack([left, right]), SR, "trap_rap")
        self.assertGreater(abs(pan_res["stereo_balance_db"]), 1.2)
        self.assertIn("FIX_STEREO_BALANCE", pan_res["codes"])
        rng = np.random.default_rng(3)
        noise = rng.standard_normal(len(t)) * 0.05
        wide = analyze_stereo_image(np.column_stack([mono + noise, mono - noise]), SR, "trap_rap")
        self.assertTrue(wide["width_ok"], wide)

    # ------------------------------------------------------------------ Q4
    def test_q4_external_refs_and_taste_guard(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bark_ext = [1.0 / 24.0] * 24
            bark_taste = [0.0] * 24
            bark_taste[6] = 1.0
            profiles = {
                "version": 1,
                "profiles": {
                    "trap_rap": {
                        "mix_profile": "trap_rap",
                        "source_type": "measured_external",
                        "provenance": "external_pro_reference",
                        "source_count": 2,
                        "pink_slope_mean": -4.0,
                        "pink_slope_std": 0.4,
                        "bark_24_mean": bark_ext,
                        "bark_24_std": [0.01] * 24,
                        "lufs_integrated": -9.5,
                    }
                },
            }
            prof_path = os.path.join(td, "spectral_profiles.json")
            with open(prof_path, "w", encoding="utf-8") as fh:
                json.dump(profiles, fh)
            taste = {
                "version": 1,
                "entries": [{
                    "project_id": "p1",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "mix_profile": "trap_rap",
                    "emotion_vector": {"valence": 0.0, "tension": 0.3, "energy": 0.7, "depth": 0.5},
                    "weight": 2.0,
                    "bark24_mean": bark_taste,
                    "pink_slope_db_per_oct": -2.0,
                }],
            }
            taste_path = os.path.join(td, "taste.json")
            with open(taste_path, "w", encoding="utf-8") as fh:
                json.dump(taste, fh)

            res = extract_target_fingerprint(
                mix_profile="trap_rap",
                emotion_vector={"valence": 0.0, "tension": 0.3, "energy": 0.7, "depth": 0.5},
                profiles_path=prof_path,
                taste_memory_path=taste_path,
            )
            self.assertEqual(res["source_mode"], "external_pro_reference+taste_memory")
            bark = np.asarray(res["multi_scale_contract"]["s5_context"]["bark24_target"], dtype=np.float64)
            self.assertAlmostEqual(float(np.sum(bark)), 1.0, places=4)
            # 30% taste max: bin 6 cannot exceed ~0.3 after normalisation
            self.assertLess(float(bark[6]), 0.35)
            self.assertGreater(float(bark[6]), 0.1)

            fp = os.path.join(td, "fp.json")
            with open(fp, "w", encoding="utf-8") as fh:
                json.dump({"project_id": "p1", "mix_profile": "trap_rap"}, fh)
            rejected = record_validated_taste(
                td, fingerprint_path=fp, taste_path=taste_path, pq_score=6.2,
            )
            self.assertFalse(rejected["recorded"])
            accepted = record_validated_taste(
                td, fingerprint_path=fp, taste_path=taste_path, pq_score=6.2, approved_by_user=True,
            )
            self.assertTrue(accepted["recorded"])

            # External bootstrap marks provenance and exposes the extra metrics
            with tempfile.TemporaryDirectory() as ext_td:
                ext_root = os.path.join(ext_td, "external_masters")
                os.makedirs(os.path.join(ext_root, "trap_rap"))
                ext_wav = os.path.join(ext_root, "trap_rap", "master.wav")
                sf.write(ext_wav, np.column_stack([_tone(220.0, 5.0, 0.3)] * 2), SR, subtype="PCM_24")
                out_prof = os.path.join(ext_td, "profiles.json")
                boot = build_spectral_profiles(
                    output_path=out_prof,
                    min_pq=None,
                    external_dir=ext_root,
                )
                self.assertIn("trap_rap", boot["external_pro_profiles"])
                with open(out_prof, encoding="utf-8") as fh:
                    saved = json.load(fh)
                entry = saved["profiles"]["trap_rap"]
                self.assertEqual(entry["provenance"], "external_pro_reference")
                for key in ("lufs_integrated", "lra_lu", "crest_factor_db", "mid_high_side_to_mid_db", "band_ratios"):
                    self.assertIn(key, entry)

    # ------------------------------------------------------------------ Q5
    def test_q5_musical_transitions_and_dark_register_split(self) -> None:
        tracks = {
            "Kick": [{"start": 6.0, "length": 0.25, "pitch": 36, "velocity": 100},
                     {"start": 7.75, "length": 0.25, "pitch": 36, "velocity": 100}],
            "Snare": [],
            "Chords": [{"start": 7.0, "length": 1.0, "pitch": 64, "velocity": 80},
                       {"start": 8.0, "length": 1.0, "pitch": 67, "velocity": 80}],
            "Bass": [{"start": 7.0, "length": 1.0, "pitch": 40, "velocity": 90},
                     {"start": 8.0, "length": 1.0, "pitch": 43, "velocity": 90}],
            "Lead": [{"start": 8.0, "length": 1.0, "pitch": 74, "velocity": 95}],
        }
        fill = generate_drum_fill(tracks, 8.0, bpb=4.0, profile="trap_rap", bpm=120.0)
        self.assertEqual(fill["kind"], "snare_roll")
        self.assertGreaterEqual(fill["added_notes"], 8)
        fill_notes = [n for n in tracks["Snare"] if 6.0 <= n["start"] < 8.0]
        self.assertEqual(fill_notes[0]["velocity"], 55)
        self.assertEqual(fill_notes[-1]["velocity"], 112)
        self.assertEqual(fill_notes[-1]["pitch"], 34)  # -4 st on the last hit
        self.assertFalse(any(7.5 <= n["start"] < 8.0 for n in tracks["Kick"]))

        autos: list[dict] = []
        fx = add_transition_fx(tracks, autos, 8.0, fx_track_idx=11, fx_channel_idx=3)
        self.assertTrue(fx["riser"])
        self.assertTrue(fx["impact"])
        self.assertTrue(any(n["start"] == 8.0 and n["pitch"] == 48 for n in tracks["FX"]))
        riser_vol = next(a for a in autos if a["name"] == "Riser Volume")
        self.assertEqual(riser_vol["points"][0]["value"], 0.30)
        self.assertEqual(riser_vol["points"][-1]["value"], 0.0)

        ch = [{"start": 8.0, "length": 1.0, "pitch": 64, "velocity": 80}]
        bs = [{"start": 8.0, "length": 1.0, "pitch": 40, "velocity": 90}]
        prev_ch = [{"start": 6.0, "length": 2.0, "pitch": 60, "velocity": 70}]
        prev_bs = [{"start": 6.0, "length": 2.0, "pitch": 36, "velocity": 80}]
        antic = apply_harmonic_anticipation(prev_ch + ch, prev_bs + bs, [8.0], offset_beats=0.5, ratio=1.0)
        self.assertEqual(antic["moved_chords"], 1)
        self.assertEqual(antic["moved_bass"], 1)
        all_notes = prev_ch + ch + prev_bs + bs
        self.assertTrue(all(n["start"] >= 0.0 for n in all_notes))
        for a, b in ((prev_ch[0], ch[0]), (prev_bs[0], bs[0])):
            self.assertLessEqual(a["start"] + a["length"], b["start"] + 1e-4)

        split = song_builder._apply_dark_register_split(
            {"Chords": [{"start": 0.0, "length": 1.0, "pitch": 64, "velocity": 80},
                        {"start": 1.0, "length": 1.0, "pitch": 67, "velocity": 80}],
             "Lead": [{"start": 0.0, "length": 1.0, "pitch": 72, "velocity": 90}]},
            -0.7,
            chords_keys=["Chords"],
            lead_keys=["Lead"],
        )
        self.assertTrue(split["applied"])
        self.assertGreaterEqual(split["gap_st"], 5.0)

    # ------------------------------------------------------------------ Q6
    @unittest.skipUnless(shutil.which("gridline"), "binaire gridline requis")
    def test_q6_expressive_synth_and_multizone(self) -> None:
        catalog = json.loads(
            subprocess.run(["gridline", "plugin-params"], capture_output=True, text=True, check=True, timeout=60).stdout
        )
        by_type = {p["type"]: {q["name"]: q for q in p["params"]} for p in catalog if p.get("type")}
        for plugin in gl_dsl.SYNTH_REGISTRY:
            for role in ("pad", "chords", "lead", "pluck", "bell", "bass"):
                voice = gl_dsl.synth_voice(plugin, role, valence=-0.3, expressive=True)
                params = by_type[plugin]
                for name, value in voice.channel_params.items():
                    self.assertIn(name, params, f"{plugin}/{role}: unknown parameter {name}")
                    spec = params[name]
                    if spec.get("choices") is not None:
                        self.assertGreaterEqual(int(value), 0)
                        self.assertLess(int(value), len(spec["choices"]), f"{plugin}/{role}/{name}={value}")
                    else:
                        self.assertGreaterEqual(float(value), float(spec["min"]) - 1e-6, f"{plugin}/{role}/{name}={value}")
                        self.assertLessEqual(float(value), float(spec["max"]) + 1e-6, f"{plugin}/{role}/{name}={value}")
                self.assertGreaterEqual(voice.makeup_gain_db, 0.0)
                self.assertLessEqual(voice.makeup_gain_db, 2.5)

        # Headless MCP acceptance for representative expressive synths
        with tempfile.TemporaryDirectory() as td:
            sample = os.path.join(td, "note.wav")
            sf.write(sample, _tone(220.0, 1.0, 0.3), SR, subtype="PCM_24")
            proj = os.path.join(td, "expressive.gridline")
            meta = {"bpm": 120.0, "beats_per_bar": 4, "denominator": 4, "mix_profile": "pop_rnb", "valence": 0.2}
            chans = [
                song_builder.ChannelSpec("Chords", "chords", 1, "music", plugin="gl.analog", role="chords", valence=0.2),
                song_builder.ChannelSpec("Lead", "lead", 2, "music", plugin="gl.fm", role="lead", valence=0.2),
                song_builder.ChannelSpec("Keys", "keys", 3, "music", plugin="gl.keys", role="chords", valence=0.2),
                song_builder.ChannelSpec("Sampler", "lead", 4, "music", plugin="gl.sampler", sample_file=sample, root_note=57, role="lead"),
            ]
            tracks = {
                "chords": [{"start": 0.0, "length": 2.0, "pitch": 60, "velocity": 80}],
                "lead": [{"start": 0.0, "length": 1.0, "pitch": 67, "velocity": 90}],
                "keys": [{"start": 0.0, "length": 2.0, "pitch": 64, "velocity": 85}],
            }
            # keys and sampler share track keys with lead/chords; keep one note set per track key
            tracks["lead"].append({"start": 2.0, "length": 1.0, "pitch": 69, "velocity": 70})
            song_builder.build_gridline_project(
                proj_path=proj,
                export_dir=os.path.join(td, "exports"),
                samples_dir=td,
                meta=meta,
                sections=[("Main", 0, 1, "main")],
                channels=chans,
                tracks=tracks,
                markers=[],
                min_pattern_lufs=-60.0,
            )
            self.assertTrue(os.path.isfile(proj))

        zones = song_builder.split_notes_by_pitch_zones(
            [{"start": 0.0, "length": 1.0, "pitch": p, "velocity": 90} for p in range(48, 73)]
        )
        self.assertEqual(set(zones), {"zone_1", "zone_2", "zone_3"})
        for i, key in enumerate(("zone_1", "zone_2", "zone_3"), start=1):
            lo, hi = song_builder.DEFAULT_PITCH_ZONES[i - 1]
            center = (lo + hi) / 2.0
            self.assertTrue(zones[key])
            for ev in zones[key]:
                self.assertGreaterEqual(ev["pitch"], lo)
                self.assertLessEqual(ev["pitch"], hi)
                self.assertLessEqual(abs(ev["pitch"] - center), 4.0)

    # ------------------------------------------------------------------ Q7
    def test_q7_performance_and_token_budget(self) -> None:
        n = int(SR * 120.0)
        rng = np.random.default_rng(9)
        t = np.arange(n) / SR
        base = 0.12 * np.sin(2 * np.pi * 110 * t) + 0.05 * rng.standard_normal(n)
        data = np.column_stack([base, np.roll(base, 137) * 0.9])
        t0 = time.perf_counter()
        img = analyze_stereo_image(data, SR, "trap_rap")
        active = measure_active_loudness(data[:, 0], SR)
        elapsed = time.perf_counter() - t0
        self.assertIn("stereo_balance_db", img)
        self.assertIn("active_lufs", active)
        self.assertLess(elapsed, 1.5, f"Phase 8 stereo/active DSP overhead {elapsed:.2f}s > 1.5s")

        deviations = [
            {"scale": "S2_acoustic", "metric": "stereo_balance=+2.4dB", "diagnosis": "panning asymetrique",
             "next_action": "FIX_STEREO_BALANCE(balance=+2.4dB)", "severity": 0.9},
            {"scale": "S2_acoustic", "metric": "mid_high_width=-19dB", "diagnosis": "mix trop etroit",
             "next_action": "WIDEN_HARMONY_OR_PERC(width=-19dB)", "severity": 0.8},
            {"scale": "S3_syntax", "metric": "lead_vs_harmony=+13dB", "diagnosis": "harmonie inaudible",
             "next_action": "ADJUST_TRACK_FADER(track=Chords,delta_db=+4.0)", "severity": 0.75},
            {"scale": "S3_syntax", "metric": "hihat=+2dB", "diagnosis": "hats trop forts",
             "next_action": "ADJUST_TRACK_FADER(track=HiHat,delta_db=-2.0)", "severity": 0.4},
            {"scale": "S5_context", "metric": "taste_drift", "diagnosis": "derive legere",
             "next_action": "REVIEW", "severity": 0.2},
        ]
        with tempfile.TemporaryDirectory() as td:
            compact = build_multiscale_pyramid(
                stage="E4",
                scale_matrix={
                    "S1_physical": {"V": 0.8, "H": 0.8},
                    "S2_acoustic": {"V": 0.5, "H": 0.7},
                    "S3_syntax": {"V": 0.6, "H": 0.7},
                    "S4_form": {"V": 0.8, "H": 0.8},
                    "S5_context": {"V": 0.9, "H": 0.9},
                },
                deviations=deviations,
                project_dir=td,
            )
        dumped = json.dumps(compact, ensure_ascii=False)
        self.assertLessEqual(len(dumped), 1500)
        codes = {d.get("code") for d in compact["top_deviations"]}
        self.assertIn("FIX_STEREO_BALANCE", codes)
        self.assertIn("WIDEN_HARMONY_OR_PERC", codes)
        self.assertIn("ADJUST_TRACK_FADER", codes)

        # Cache keys invalidate when Q1/Q3/Q6 fields change
        base = dict(sample_path="", notes_slice=[{"start": 0, "pitch": 60}], bpm=120.0, role="lead")
        h0 = compute_e1_track_hash(**base, delay_send=0.0, hall_send=0.0, expressive=True)
        h1 = compute_e1_track_hash(**base, delay_send=0.1, hall_send=0.0, expressive=True)
        h2 = compute_e1_track_hash(**base, delay_send=0.0, hall_send=0.0, expressive=False)
        h3 = compute_e1_track_hash(**base, delay_send=0.0, hall_send=0.0, expressive=True,
                                   gain_staging_overrides={"Lead": {"volume_trim_db": 1.5}})
        self.assertEqual(len({h0, h1, h2, h3}), 4)
        e20 = compute_e2_vertical_hash("V4_LEAD", profile="trap_rap", gain_staging_overrides={}, expressive=True)
        e21 = compute_e2_vertical_hash("V4_LEAD", profile="trap_rap", gain_staging_overrides={"Lead": {"volume_trim_db": 1.5}}, expressive=True)
        self.assertNotEqual(e20, e21)


if __name__ == "__main__":
    unittest.main()
