#!/usr/bin/env python3
"""
Phase 7 Expressivity, Psychoacoustics & Parametric Autonomy Validation Suite (P1..P6).

Covers:
  - P1: Declarative `schemas/arrangement_plan.schema.json`, `compile_arrangement_plan`,
        and zero-token local mutations (`PRUNE_TRACK_IN_SECTION`,
        `REINFORCE_SECTION_TRANSITION`, `MUTATE_BLOCK_DENSITY_OR_LPF`).
  - P2: `phrasing_engine.py` (4-bar velocity arch + LPF automations, cadential bass/lead
        ornaments with `detect_minor_2nd_clashes` guard, micro-pocket offsets).
  - P3: `deep_diagnostics.py` + `render_and_audit.py` + `multiscale_autofix.py`
        (`APPLY_DYNAMIC_MIRROR_EQ`, `RELAX_BUS_COMPRESSION`, `BOOST_BASS_HARMONICS` E2E).
  - P4: `taste_memory.py` + `target_fingerprint.py` (`record_validated_taste`,
        `query_taste_prior`, `"source": "user_taste_memory"` override in 0 tokens).
  - P5: `--checkpoint-e2` & `--resume-from-checkpoint` (`AWAITING_E2_CHECKPOINT`,
        `renders/e2_checkpoint_15s.wav` at -14 LUFS, <= 300 tokens 5-line card, cache reuse).
  - P6: `vocal_arrangement_sync.py` (`.ustx` + WAV phrase extraction, breathing-window
        counter muting, `BUS Music` `1850 Hz` Band 5 formant ducking automation, `SYNC_VOCAL_ARRANGEMENT`).
  - Budget: P1..P6 cumulative CPU <= 2.5 s on 2-min stereo track, 0 PyTorch / 0 GPU.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import shutil

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
ANALYZER_DIR = os.path.join(ROOT, "mcp-servers", "audio-analyzer")
for _d in (SCRIPTS_DIR, ANALYZER_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from deep_diagnostics import (
    analyze_bass_harmonic_translation,
    analyze_dynamic_bark_masking,
    compute_ebu_r128_stats,
    compute_lra_ebu,
)
from gl_dsl import eq_gain_to_auto
from loop_first_runner import LoopFirstStateMachine
from multiscale_autofix import dispatch_multiscale_autofix
from phrasing_engine import apply_instrumental_prosody
from render_and_audit import _should_record_taste
from song_builder import compile_arrangement_plan, load_arrangement_plan
from studio_lib import GridlineMCP
from target_fingerprint import extract_target_fingerprint
from taste_memory import MAX_TASTE_ENTRIES, query_taste_prior, record_validated_taste
from vocal_arrangement_sync import (
    build_formant_ducking_automation,
    extract_vocal_phrases,
    mute_counter_during_vocals,
)


def _sample_arrangement_plan() -> dict:
    # 4 sections: Intro (4), Couplet1 (16), Refrain1 (8), Outro (4) -> 32 bars
    # Includes counter track so P1/P6 interactions can be exercised cleanly.
    bass_bar = [
        {"start": 0.0, "duration": 2.0, "pitch": 36, "velocity": 100},
        {"start": 2.0, "duration": 1.5, "pitch": 39, "velocity": 94},
    ]
    chords_bar = [
        {"start": 0.0, "duration": 3.5, "pitch": p, "velocity": 76}
        for p in (48, 51, 55)
    ]
    lead_bar = [
        {"start": 0.5, "duration": 0.75, "pitch": 60, "velocity": 92},
        {"start": 1.5, "duration": 0.75, "pitch": 63, "velocity": 96},
        {"start": 2.5, "duration": 0.75, "pitch": 67, "velocity": 98},
    ]
    counter_bar = [
        {"start": 0.25, "duration": 0.5, "pitch": 67, "velocity": 80},
        {"start": 2.25, "duration": 0.5, "pitch": 65, "velocity": 82},
    ]
    kick_bar = [
        {"start": 0.0, "duration": 0.5, "pitch": 36, "velocity": 110},
        {"start": 2.5, "duration": 0.5, "pitch": 36, "velocity": 104},
    ]
    hat_bar = [{"start": i * 0.5, "duration": 0.25, "pitch": 42, "velocity": 78} for i in range(8)]
    return {
        "version": 1,
        "meta": {
            "title": "Phase7TestTrack",
            "genre": "dark_trap",
            "bpm": 132.0,
            "time_signature": {"numerator": 4, "denominator": 4},
            "key": "C minor",
            "mode": "natural_minor",
            "mix_profile": "trap_rap",
            "has_drums": True,
            "valence": -0.5,
            "tension": 0.65,
            "energy": 0.75,
            "depth": 0.60,
        },
        "harmonic_core": {"key": "C minor", "mode": "natural_minor", "degree_progression": ["i", "VI", "III", "VII"]},
        "motifs": {
            "kick": {"role": "drums", "bars": 1, "loop": True, "notes": kick_bar},
            "hihat": {"role": "drums", "bars": 1, "loop": True, "notes": hat_bar},
            "bass": {"role": "bass", "bars": 1, "loop": True, "notes": bass_bar},
            "chords": {"role": "chords", "bars": 1, "loop": True, "notes": chords_bar},
            "lead": {"role": "lead", "bars": 1, "loop": True, "notes": lead_bar},
            "counter": {"role": "counter", "bars": 1, "loop": True, "notes": counter_bar},
        },
        "sections": [
            {"name": "Intro", "kind": "intro", "start_bar": 0, "bars": 4, "active_roles": ["chords", "bass"]},
            {
                "name": "Couplet1",
                "kind": "verse",
                "start_bar": 4,
                "bars": 16,
                "active_roles": ["kick", "hihat", "bass", "chords", "counter"],
            },
            {
                "name": "Refrain1",
                "kind": "chorus",
                "start_bar": 20,
                "bars": 8,
                "active_roles": ["kick", "hihat", "bass", "chords", "lead", "counter"],
            },
            {"name": "Outro", "kind": "outro", "start_bar": 28, "bars": 4, "active_roles": ["chords", "bass"]},
        ],
    }


class TestPhase7ExpressivityAutonomy(unittest.TestCase):
    @unittest.skipUnless(shutil.which("gridline"), "binaire gridline requis")
    def test_p1_arrangement_plan_schema_compile_and_autofix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan_p = os.path.join(tmp, "arrangement_plan.json")
            out_gl = os.path.join(tmp, "projet.gridline")
            with open(plan_p, "w", encoding="utf-8") as fh:
                json.dump(_sample_arrangement_plan(), fh, indent=2)

            plan, _meta = load_arrangement_plan(plan_p, validate=True)
            self.assertEqual(len(plan["sections"]), 4)

            comp_res = compile_arrangement_plan(plan_p, tmp, out_gl, verify_gate=False)
            self.assertTrue(os.path.isfile(comp_res["gridline_path"]))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "master_sketch.mid")))

            # Dispatch 3 P1 autofix actions -> mutate arrangement_plan.json locally & recompile
            fix_res = dispatch_multiscale_autofix(
                tmp,
                [
                    "PRUNE_TRACK_IN_SECTION(Couplet1, counter)",
                    "REINFORCE_SECTION_TRANSITION(Refrain1)",
                    "MUTATE_BLOCK_DENSITY_OR_LPF(bars=13-20)",
                ],
                gridline_path=out_gl,
                respect_guardrails=False,
            )
            self.assertEqual(fix_res["applied_count"], 3, fix_res)

            mutated, _ = load_arrangement_plan(plan_p, validate=True)
            couplet_secs = [s for s in mutated["sections"] if "Couplet1" in s["name"]]
            self.assertGreaterEqual(len(couplet_secs), 2, "Expected 16-bar block split into 8+8 halves")
            for cs in couplet_secs:
                self.assertNotIn("counter", cs["active_roles"])
            ref_sec = next(s for s in mutated["sections"] if s["name"] == "Refrain1")
            self.assertIn("filter_sweep", ref_sec.get("transitions") or {})

    def test_p2_instrumental_prosody_dynamics_lpf_and_b2_safety(self) -> None:
        # 8 bars Couplet (0..32) + 8 bars Refrain (32..64)
        tracks = {
            "chords": [
                {"start": b * 4.0, "duration": 3.5, "pitch": p, "velocity": 78}
                for b in range(16)
                for p in (48, 51, 55)
            ],
            "bass": [
                {"start": b * 4.0, "duration": 2.0, "pitch": 36 if (b % 2 == 0) else 41, "velocity": 100}
                for b in range(16)
            ],
            "lead": [
                {"start": b * 4.0 + 0.5, "duration": 1.0, "pitch": 67 if (b % 4 == 2) else 63, "velocity": 92}
                for b in range(16)
            ],
        }
        sections = [("Couplet1", 0, 8, "verse"), ("Refrain1", 8, 8, "chorus")]
        ev = {"valence": -0.5, "tension": 0.65, "energy": 0.72, "depth": 0.6}
        hc = {"key": "C minor", "mode": "natural_minor"}

        out_tracks, autos = apply_instrumental_prosody(
            tracks,
            sections,
            hc,
            ev,
            bpm=132.0,
            time_signature=(4, 4),
            track_idx_start=15,
            channel_mixer_map={"chords": 3, "lead": 4},
        )
        # 1) 4-bar velocity curve rises into bar 3 (0-indexed bar 2) relative to bar 1 (bar 0)
        ch_couplet = [n for n in out_tracks["chords"] if n["start"] < 16.0]
        v_bar0 = max(n["velocity"] for n in ch_couplet if n["start"] < 4.0)
        v_bar2 = max(n["velocity"] for n in ch_couplet if 8.0 <= n["start"] < 12.0)
        self.assertGreater(v_bar2, v_bar0)

        # 2) LPF breathing automation emitted on chords/lead
        self.assertGreaterEqual(len(autos), 1)
        self.assertTrue(any("Cutoff" in a["param"] or "Band" in a["param"] for a in autos))

        # 3) Cadential approach notes added on bass & lead
        self.assertGreater(len(out_tracks["bass"]), len(tracks["bass"]))
        self.assertGreater(len(out_tracks["lead"]), len(tracks["lead"]))

        # 4) Micro-pocket offset: Couplet (> 0) > Refrain (< 0 or earlier relative to bar grid)
        l_couplet = min(n["start"] % 1.0 for n in out_tracks["lead"] if abs((n["start"] % 4.0) - 0.5) < 0.08 and n["start"] < 32.0)
        l_refrain = min(n["start"] % 1.0 for n in out_tracks["lead"] if abs((n["start"] % 4.0) - 0.5) < 0.08 and n["start"] >= 32.0)
        self.assertGreater(l_couplet, l_refrain)

    @unittest.skipUnless(shutil.which("gridline"), "binaire gridline requis")
    def test_p3_bark_masking_lra_and_bass_translation_headless_autofix(self) -> None:
        sr = 44100
        # Check LRA match against pyloudnorm
        t = np.arange(int(18.0 * sr), dtype=np.float64) / sr
        env = 0.08 + 0.55 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 6.0)) ** 2
        sig = env * (0.6 * np.sin(2 * np.pi * 50.0 * t) + 0.2 * np.sin(2 * np.pi * 600.0 * t))
        stereo = np.column_stack([sig, 0.9 * sig]).astype(np.float32)

        lra_fast = compute_lra_ebu(stereo, sr)
        lra_ref = float(pyln.Meter(sr).loudness_range(stereo))
        self.assertLess(abs(lra_fast - lra_ref), 0.15)

        # Dynamic Bark masking trigger check
        tb = np.arange(int(2.0 * sr)) / sr
        y_lead = (0.12 * np.sin(2 * np.pi * 2100.0 * tb)).astype(np.float32)
        y_harm = (0.20 * np.sin(2 * np.pi * 2100.0 * tb) + 0.05 * np.sin(2 * np.pi * 400.0 * tb)).astype(np.float32)
        bm = analyze_dynamic_bark_masking((y_lead, sr), (y_harm, sr))
        self.assertEqual(bm["verdict"], "FIX")
        self.assertEqual(bm["next_action"]["code"], "APPLY_DYNAMIC_MIRROR_EQ")

        # Headless Gridline E2E test for BOOST_BASS_HARMONICS, APPLY_DYNAMIC_MIRROR_EQ, RELAX_BUS_COMPRESSION
        with tempfile.TemporaryDirectory() as tmp:
            gl_path = os.path.join(tmp, "p3_e2e.gridline")
            mcp = GridlineMCP("/usr/share/gridline/assets/templates/Basics/Empty with mixer tracks.gridline", tmp)
            try:
                mcp.call("set_song", {"title": "P3E2E", "tempo": 120.0})
                mcp.call("add_channel", {"name": "808 Sub", "mixer": 1, "plugin": "gl.analog"})
                mcp.call("edit_mixer_track", {"mixer": 1, "name": "808 Sub"})
                mcp.call("set_parameters", {"changes": [
                    {"path": "channel/1/Filter mode", "value": 1},
                    {"path": "channel/1/Cutoff", "value": 65.0},
                ]})
                mcp.call("edit_pattern", {"pattern": 1, "name": "Sub", "bars": 2})
                mcp.call("set_notes", {"pattern": 1, "channel": 1, "notes": [
                    {"pitch": 30, "start": 0.0, "length": 4.0, "velocity": 115},
                    {"pitch": 30, "start": 4.0, "length": 4.0, "velocity": 115},
                ]})
                mcp.call("add_clip", {"pattern": 1, "track": 1, "start": 0.0})
                mcp.call("add_channel", {"name": "Chords Pad", "mixer": 2, "plugin": "gl.analog"})
                mcp.call("edit_mixer_track", {"mixer": 2, "name": "Chords Pad"})
                bm_res = mcp.call("add_mixer_track", {"name": "BUS Music"})
                bm_idx = int(bm_res.get("mixer", 17))
                mcp.call("add_plugin", {"mixer": bm_idx, "plugin": "gl.comp"})
                mcp.call("add_plugin", {"mixer": 0, "plugin": "gl.limiter"})
                mcp.call("save_project", {"path": gl_path})
            finally:
                mcp.close()

            mcp = GridlineMCP(gl_path, tmp)
            try:
                r1 = mcp.call("render_audio", {"path": "pre.wav"})
            finally:
                mcp.close()
            w1 = r1.get("path") or os.path.join(tmp, "pre.wav")
            if not os.path.isfile(w1):
                w1 = os.path.join(tmp, "pre.wav")
            bt1 = analyze_bass_harmonic_translation(w1)
            self.assertTrue(bt1["requires_translation_boost"], bt1)

            fix_res = dispatch_multiscale_autofix(
                tmp,
                [
                    {"next_action": f"BOOST_BASS_HARMONICS({bt1['f0_hz']})", "action_payload": bt1["next_action"]},
                    "APPLY_DYNAMIC_MIRROR_EQ(2150.0)",
                    "RELAX_BUS_COMPRESSION",
                ],
                gridline_path=gl_path,
                respect_guardrails=False,
            )
            self.assertEqual(fix_res["applied_count"], 3, fix_res)

            mcp = GridlineMCP(gl_path, tmp)
            try:
                r2 = mcp.call("render_audio", {"path": "post.wav"})
            finally:
                mcp.close()
            w2 = r2.get("path") or os.path.join(tmp, "post.wav")
            if not os.path.isfile(w2):
                w2 = os.path.join(tmp, "post.wav")
            bt2 = analyze_bass_harmonic_translation(w2)
            self.assertFalse(bt2["requires_translation_boost"], bt2)
            self.assertGreater(bt2["r_trans_db"], -14.0)

    def test_p4_taste_memory_prior_and_contract_override(self) -> None:
        self.assertFalse(_should_record_taste("/tmp/temp_test_proj", "", False))
        self.assertTrue(_should_record_taste("/tmp/temp_test_proj", "", True))

        with tempfile.TemporaryDirectory() as tmp:
            taste_p = os.path.join(tmp, "user_taste_profile.json")
            pa = os.path.join(tmp, "song_a")
            pb = os.path.join(tmp, "song_b")
            for pdir, name, slope, lufs, ev, app in [
                (pa, "song_a", -5.2, -9.8, {"valence": -0.8, "tension": 0.8, "energy": 0.9, "depth": 0.7}, True),
                (pb, "song_b", -3.4, -13.2, {"valence": 0.4, "tension": 0.2, "energy": 0.35, "depth": 0.3}, False),
            ]:
                os.makedirs(os.path.join(pdir, "renders"), exist_ok=True)
                with open(os.path.join(pdir, "song_blueprint.json"), "w", encoding="utf-8") as fh:
                    json.dump({"mix_profile": "trap_rap", "bpm": 134, "emotion_vector": ev}, fh)
                with open(os.path.join(pdir, "renders", "master_fingerprint.json"), "w", encoding="utf-8") as fh:
                    json.dump({
                        "project_name": name, "mix_profile": "trap_rap", "bpm": 134,
                        "pink_slope_db_per_oct": slope, "crest_factor_db": 8.0,
                        "integrated_lufs": lufs, "bark24": [round(1.0 / 24.0, 6)] * 24,
                    }, fh)
                rec = record_validated_taste(pdir, approved_by_user=app, taste_path=taste_p)
                self.assertTrue(rec["recorded"])

            prior = query_taste_prior("trap_rap", {"valence": -0.75, "tension": 0.78, "energy": 0.88, "depth": 0.68}, taste_path=taste_p)
            self.assertIsNotNone(prior)
            self.assertLess(prior["pink_slope_db_per_oct"], -4.5)

            p_new = os.path.join(tmp, "new_song")
            os.makedirs(p_new, exist_ok=True)
            res = extract_target_fingerprint(
                audio_paths=[],
                mix_profile="trap_rap",
                emotion_vector={"valence": -0.75, "tension": 0.78, "energy": 0.88, "depth": 0.68},
                project_dir=p_new,
                taste_memory_path=taste_p,
            )
            self.assertEqual(res["source_mode"], "user_taste_memory")
            with open(res["contract_path"], encoding="utf-8") as fh:
                contract = json.load(fh)
            self.assertEqual(contract["source"], "user_taste_memory")

    @unittest.skipUnless(shutil.which("gridline"), "binaire gridline requis")
    def test_p5_checkpoint_e2_and_cache_resume(self) -> None:
        sr = 44100
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "renders"), exist_ok=True)
            with open(os.path.join(tmp, "song_blueprint.json"), "w", encoding="utf-8") as fh:
                json.dump({
                    "mix_profile": "trap_rap",
                    "bpm": 132,
                    "time_signature": {"numerator": 4, "denominator": 4},
                    "harmonic_core": {"key": "C minor", "degree_progression": ["i", "VI", "III", "VII"]},
                    "tracks": [{"name": "Kick"}, {"name": "808"}, {"name": "Chords"}, {"name": "Lead"}],
                }, fh)
            t = np.arange(int(7.27 * sr), dtype=np.float32) / sr
            sig = 0.08 * (np.sin(2 * np.pi * 60.0 * t) + 0.5 * np.sin(2 * np.pi * 440.0 * t))
            sf.write(os.path.join(tmp, "renders", "final_mix.wav"), np.column_stack([sig, sig]), sr)

            sm = LoopFirstStateMachine(tmp, e2_checkpoint=True)
            sm.set_artifact_hash("E2_V4_LEAD", "sha_dummy_ok", {"verdict": "PASS"})

            proc = subprocess.run([
                os.path.join(ROOT, ".venv", "bin", "python"),
                os.path.join(SCRIPTS_DIR, "run_autonomous_pipeline.py"),
                "--project", tmp, "--stage", "ALL", "--skip-render", "--checkpoint-e2", "--json-compact",
            ], capture_output=True, text=True, check=True)
            out_s = proc.stdout.strip()
            self.assertLessEqual(len(out_s), 950, f"Exceeds 300-token limit: {out_s}")
            chk = json.loads(out_s)
            self.assertEqual(chk["status"], "AWAITING_E2_CHECKPOINT")
            self.assertEqual(len(chk["summary"]), 5)
            chk_wav = os.path.join(tmp, chk["checkpoint_wav"])
            self.assertTrue(os.path.isfile(chk_wav))
            d_c, sr_c = sf.read(chk_wav, dtype="float32")
            self.assertGreaterEqual(len(d_c) / sr_c, 12.0)
            self.assertLessEqual(len(d_c) / sr_c, 15.05)
            lufs_c = compute_ebu_r128_stats(d_c, sr_c)["integrated_lufs"]
            self.assertLessEqual(abs(lufs_c - (-14.0)), 1.2)

            proc2 = subprocess.run([
                os.path.join(ROOT, ".venv", "bin", "python"),
                os.path.join(SCRIPTS_DIR, "run_autonomous_pipeline.py"),
                "--project", tmp, "--stage", "ALL", "--skip-render", "--resume-from-checkpoint", "--json-compact",
            ], capture_output=True, text=True, check=True)
            pyr = json.loads(proc2.stdout.strip())
            self.assertIn("verdict", pyr)
            with open(os.path.join(tmp, "pipeline_state.json"), encoding="utf-8") as fh:
                st = json.load(fh)
            self.assertTrue(st["e2_checkpoint_approved"])

    @unittest.skipUnless(shutil.which("gridline"), "binaire gridline requis")
    def test_p6_vocal_arrangement_sync_ustx_wav_and_autofix(self) -> None:
        sr = 44100
        with tempfile.TemporaryDirectory() as tmp:
            ustx_p = os.path.join(tmp, "vocals.ustx")
            ustx_doc = {
                "resolution": 480,
                "tempos": [{"position": 0, "bpm": 120.0}],
                "voice_parts": [{
                    "position": 0,
                    "notes": [
                        {"position": 0, "duration": 960, "lyric": "oh"},       # beats 0..2
                        {"position": 960, "duration": 960, "lyric": "la"},     # beats 2..4 (merged < 250ms)
                        {"position": 2880, "duration": 1440, "lyric": "hey"},  # beats 6..9 (gap 4..6 = 2 beats)
                    ],
                }],
            }
            with open(ustx_p, "w", encoding="utf-8") as fh:
                yaml.safe_dump(ustx_doc, fh)

            # Audio fallback check (50ms RMS > -35dBFS, merges gap < 250ms)
            vox_wav = os.path.join(tmp, "samples", "vocals.wav")
            os.makedirs(os.path.dirname(vox_wav), exist_ok=True)
            t = np.arange(int(5.0 * sr), dtype=np.float32) / sr
            y_v = np.zeros_like(t)
            y_v[(t >= 0.0) & (t < 1.0)] = 0.25 * np.sin(2 * np.pi * 220.0 * t[(t >= 0.0) & (t < 1.0)])
            y_v[(t >= 1.1) & (t < 2.0)] = 0.25 * np.sin(2 * np.pi * 220.0 * t[(t >= 1.1) & (t < 2.0)])
            y_v[(t >= 3.0) & (t < 4.5)] = 0.25 * np.sin(2 * np.pi * 220.0 * t[(t >= 3.0) & (t < 4.5)])
            sf.write(vox_wav, y_v, sr)

            phrases_ustx = extract_vocal_phrases(ustx_path=ustx_p, audio_path=vox_wav, bpm=120.0)
            self.assertEqual(len(phrases_ustx), 2)
            self.assertAlmostEqual(phrases_ustx[0]["start_beat"], 0.0)
            self.assertAlmostEqual(phrases_ustx[0]["end_beat"], 4.0)

            phrases_wav = extract_vocal_phrases(ustx_path="", audio_path=vox_wav, bpm=120.0)
            self.assertEqual(len(phrases_wav), 2)

            tracks = {
                "counter": [
                    (1.0, 0.5, 67, 88),   # overlaps beat 0..4 -> muted
                    (4.5, 0.75, 65, 86),  # inside breathing gap 4..6 (>= 0.5 beat) -> KEPT
                    (7.0, 0.5, 63, 84),   # overlaps beat 6..9 -> muted
                ],
                "lead": [(1.0, 0.25, 72, 80)],
            }
            synced = mute_counter_during_vocals(tracks, phrases_ustx)
            self.assertEqual(len(synced["counter"]), 1)
            self.assertAlmostEqual(synced["counter"][0][0], 4.5)
            self.assertLess(synced["lead"][0][3], 80)

            auto = build_formant_ducking_automation(phrases_ustx, bpm=120.0, mixer_idx=19, slot_idx=2, band_idx=5)
            self.assertEqual(auto["mixer"], 19)
            self.assertEqual(auto["param"], "Band 5 gain")
            vals = [p["value"] for p in auto["points"]]
            self.assertTrue(any(abs(v - eq_gain_to_auto(-3.5)) < 0.01 for v in vals))

            # Full arrangement plan compilation + SYNC_VOCAL_ARRANGEMENT dispatcher
            plan_p = os.path.join(tmp, "arrangement_plan.json")
            out_gl = os.path.join(tmp, "vocal_sync.gridline")
            with open(plan_p, "w", encoding="utf-8") as fh:
                json.dump(_sample_arrangement_plan(), fh)
            compile_arrangement_plan(plan_p, tmp, out_gl, verify_gate=False)
            fix_v = dispatch_multiscale_autofix(tmp, ["SYNC_VOCAL_ARRANGEMENT"], gridline_path=out_gl, respect_guardrails=False)
            self.assertEqual(fix_v["applied_count"], 1, fix_v)

    def test_p1_to_p6_cumulative_cpu_budget_on_2min_track(self) -> None:
        sr = 44100
        n = int(120.0 * sr)
        t = np.arange(n, dtype=np.float32) / sr
        lead = (0.18 * np.sin(2 * np.pi * 1400.0 * t)).astype(np.float32)
        harm = (0.18 * np.sin(2 * np.pi * 450.0 * t)).astype(np.float32)
        bass = (0.28 * np.sin(2 * np.pi * 48.0 * t)).astype(np.float32)
        master = np.column_stack([lead + harm + bass, lead + harm + bass]).astype(np.float32)

        # Warm pyln filter init once outside hot loop
        _ = compute_ebu_r128_stats(master[:sr], sr)

        with tempfile.TemporaryDirectory() as tmp:
            taste_p = os.path.join(tmp, "taste.json")
            p_old = os.path.join(tmp, "old")
            os.makedirs(os.path.join(p_old, "renders"), exist_ok=True)
            with open(os.path.join(p_old, "song_blueprint.json"), "w", encoding="utf-8") as fh:
                json.dump({"mix_profile": "trap_rap", "bpm": 132, "emotion_vector": {"valence": -0.5, "tension": 0.65, "energy": 0.75, "depth": 0.6}}, fh)
            with open(os.path.join(p_old, "renders", "master_fingerprint.json"), "w", encoding="utf-8") as fh:
                json.dump({"project_name": "old", "mix_profile": "trap_rap", "bpm": 132, "pink_slope_db_per_oct": -4.2, "bark24": [1 / 24.0] * 24}, fh)
            record_validated_taste(p_old, taste_path=taste_p)

            t0 = time.perf_counter()
            # P2 Prosody (64 bars = 2 min at 128 BPM)
            tr_p2 = {
                "chords": [{"start": float(b * 4), "duration": 3.5, "pitch": p, "velocity": 80} for b in range(64) for p in (48, 51, 55)],
                "bass": [{"start": float(b * 4), "duration": 2.0, "pitch": 36, "velocity": 100} for b in range(64)],
                "lead": [{"start": float(b * 4 + 0.5), "duration": 1.0, "pitch": 67, "velocity": 95} for b in range(64)],
            }
            apply_instrumental_prosody(tr_p2, [("V1", 0, 32, "verse"), ("C1", 32, 32, "chorus")], {"key": "C minor", "mode": "natural_minor"}, {"valence": -0.5, "tension": 0.6, "energy": 0.7, "depth": 0.6}, bpm=128.0)
            # P3 Bark Masking + Bass Translation + Single-pass EBU R128 LRA
            analyze_bass_harmonic_translation((bass, sr), sr=sr)
            analyze_dynamic_bark_masking((lead, sr), (harm, sr), sr=sr)
            compute_ebu_r128_stats(master, sr)
            # P4 Taste query
            query_taste_prior("trap_rap", {"valence": -0.5, "tension": 0.65, "energy": 0.75, "depth": 0.6}, taste_path=taste_p)
            # P6 Vocal sync
            ph = [{"start_beat": float(k * 8), "end_beat": float(k * 8 + 4)} for k in range(16)]
            mute_counter_during_vocals({"counter": [(float(k * 2), 0.5, 65, 80) for k in range(64)]}, ph)
            build_formant_ducking_automation(ph, bpm=128.0)
            elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, 2.5, f"P1..P6 overhead exceeded 2.5s budget: {elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main()
