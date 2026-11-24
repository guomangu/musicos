#!/usr/bin/env python3
"""
Phase 5 CI Tests — Loop-First Pipeline (Phases A, B, C, D).

Covers:
  - Phase A: semantic_anchors.json + .npz, sample_semantic_cards, timbre_card,
    < 50 ms select_sample + propose_palettes on 5k SQLite DB with 0 PyTorch,
    index_project_samples.py & strict user-sample exclusivity invariant.
  - Phase B: audit_track_soundcheck (silent, +18 ct detuned, bright saw @ valence=-0.7,
    clean pass, real headless Gridline soundcheck), export_palette_preview.py,
    MultimodalAudioListener contract parity.
  - Phase C: V3 audio bass/harmony chroma clash detection, song_builder.build_core_loop_project,
    headless vertical_tester (V1..V4) on trap_rap and classical_ambient (has_drums=False skipping V2).
  - Phase D: audit_horizontal_seam rejecting flat transition & abrupt outro then passing
    with contrast + beat4_dropout + outro_fade, and LoopFirstStateMachine guardrails
    (user_samples_locked, forbidden_tracks, retry escalation to SWAP_SAMPLE).
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
for _p in (
    ROOT,
    os.path.join(ROOT, "scripts"),
    os.path.join(ROOT, "mcp-servers", "audio-analyzer"),
    os.path.join(ROOT, "mcp-servers", "audio-index"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import audio_analyzer_server
import audio_index_server
from deep_diagnostics import analyze_vertical_stage
from export_palette_preview import export_palette_preview
from horizontal_tester import audit_outro_wav, audit_seam_wav
from index_project_samples import index_project_samples
from indexer import DEFAULT_ANCHORS_NPZ, init_db, load_semantic_anchors, populate_semantic_cards
from listener_backend import MultimodalAudioListener
from loop_first_runner import LoopFirstStateMachine
from palette_builder import propose_palettes
from select_sample import select_best_from_db
from song_builder import ChannelSpec, build_core_loop_project, eq_kick_clean, eq_music_pocket
from soundcheck_lib import audit_track_soundcheck
from vertical_tester import run_all_vertical_stages, run_vertical_stage


def _synth_tone(freq_hz: float, dur_s: float, sr: int = 44100, amp: float = 0.35, harmonics: int = 2) -> np.ndarray:
    t = np.linspace(0.0, dur_s, int(sr * dur_s), endpoint=False)
    env = np.minimum(1.0, t / 0.015) * np.exp(-t * (1.2 / max(0.4, dur_s)))
    y = np.zeros_like(t)
    for k in range(1, harmonics + 1):
        y += (1.0 / (k**1.4)) * np.sin(2.0 * np.pi * freq_hz * k * t)
    y = y / (np.max(np.abs(y)) + 1e-9) * amp * env
    return np.column_stack([y, y]).astype(np.float32)


def _create_5k_db_with_cards(db_path: str) -> None:
    conn = init_db(db_path)
    cats = ("kick", "808", "snare", "hat", "piano", "synth")
    s_rows = []
    for i in range(5000):
        cat = cats[i % len(cats)]
        cent = 120.0 if cat in ("kick", "808") else (1400.0 + (i % 20) * 60.0 if cat in ("piano", "synth") else 5500.0)
        s_rows.append((
            f"/tmp/fake_5k/{cat}_{i:04d}.wav",
            f"{cat}_{i:04d}.wav",
            "fake_5k",
            f"{cat}_{i:04d}.wav",
            ".wav",
            1700000000.0,
            65536,
            cat,
            0,
            0.7,
            44100,
            2,
            -1.0,
            -14.0,
            132.0,
            "C minor",
            0.9,
            cent,
            "dark" if cent < 1500 else "bright",
            8.0 if cat in ("kick", "snare", "hat") else 35.0,
            "punchy" if cat in ("kick", "snare", "hat") else "medium",
            f"{cat} warm dark",
            None,
            None,
        ))
    conn.executemany(
        """
        INSERT INTO samples (
            path, filename, pack_name, rel_path, ext, file_mtime, file_size,
            category, is_loop, duration_sec, sample_rate, channels, peak_db, rms_db,
            bpm, musical_key, key_confidence, spectral_centroid_hz, brightness_band,
            attack_ms, decay_profile, tags, clap_embedding, clap_updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        s_rows,
    )
    conn.commit()
    ids = [r[0] for r in conn.execute("SELECT id, category FROM samples ORDER BY id").fetchall()]
    d_rows = []
    for idx, sid in enumerate(ids):
        cat = cats[idx % len(cats)]
        playable = 1 if cat in ("808", "piano", "synth") else 0
        if cat == "kick":
            bands = (0.45, 0.40, 0.08, 0.04, 0.02, 0.01)
            att = 5.0
            rh = 0.25
        elif cat == "808":
            bands = (0.55, 0.35, 0.06, 0.02, 0.01, 0.01)
            att = 14.0
            rh = 0.88
        elif cat == "snare":
            bands = (0.01, 0.12, 0.28, 0.32, 0.20, 0.07)
            att = 6.0
            rh = 0.20
        elif cat == "hat":
            bands = (0.0, 0.01, 0.03, 0.08, 0.32, 0.56)
            att = 4.0
            rh = 0.12
        elif cat == "piano":
            bands = (0.02, 0.18, 0.38, 0.30, 0.09, 0.03)
            att = 26.0
            rh = 0.86
        else:
            bands = (0.01, 0.08, 0.24, 0.44, 0.17, 0.06)
            att = 65.0
            rh = 0.84
        cent = 120.0 if cat in ("kick", "808") else (1400.0 if cat in ("piano", "synth") else 5500.0)
        d_rows.append((
            sid,
            "mono_oneshot" if playable else "percussion",
            playable,
            261.63 if playable else None,
            60 if playable else 36,
            0,
            "C4" if playable else None,
            cent,
            0.08,
            -18.0,
            0.05,
            0.04 if playable else 0.25,
            rh,
            0.12,
            18.0,
            0.90,
            0.96,
            12.0,
            att,
            -1.0,
            0.7,
            0.02,
            1,
            0.05,
            0,
            0,
            "[]",
            bands[0], bands[1], bands[2], bands[3], bands[4], bands[5],
            None,
            "[]",
            1700000000.0,
        ))
    conn.executemany(
        """
        INSERT INTO sample_dsp_metrics (
            sample_id, sample_nature, playable_chromatically, f0_hz,
            recommended_root_midi, recommended_fine_tune_ct, detected_note,
            centroid_hz, r_harsh, peak_harsh_db, roughness, sfm, r_harm,
            spectral_flux, hnr_db, low_mid_corr, phase_corr, crest_factor_db,
            attack_ms, peak_dbfs, duration_s, f0_std_semitones, onset_count,
            polyphony_ratio, needs_fade, hard_reject, reject_reasons,
            band_sub, band_bass, band_lowmid, band_mid, band_harsh, band_air,
            bark_24_json, inharmonic_warnings_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        d_rows,
    )
    conn.commit()
    populate_semantic_cards(conn, force=True)
    conn.close()


class TestPhase5LoopFirst(unittest.TestCase):
    def test_phase_a_semantic_memory_palettes_speed_and_exclusivity(self) -> None:
        # 1) Verify pre-encoded semantic anchors exist & have 60-80 descriptors across 6 axes
        anchors = load_semantic_anchors(DEFAULT_ANCHORS_NPZ)
        self.assertIsNotNone(anchors)
        assert anchors is not None
        self.assertGreaterEqual(len(anchors["ids"]), 60)
        self.assertLessEqual(len(anchors["ids"]), 85)
        self.assertEqual(len(set(anchors["axes"])), 6)
        self.assertEqual(anchors["embeddings_f16"].dtype, np.float16)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 2) 5k-row SQLite DB: select_sample + propose_palettes < 50 ms each, 0 torch
            db_path = os.path.join(tmpdir, "test_5k.db")
            _create_5k_db_with_cards(db_path)

            t0 = time.perf_counter()
            sel = select_best_from_db(
                db_path=db_path,
                role="lead_melody",
                target_key="C minor",
                top_k=5,
                with_clap=True,
                valence=-0.6,
            )
            ms_sel = (time.perf_counter() - t0) * 1000.0
            self.assertLess(ms_sel, 50.0, f"select_sample too slow on 5k DB: {ms_sel:.1f}ms")
            self.assertTrue(sel["recommendations"])
            self.assertIn("timbre_card", sel["recommendations"][0])
            self.assertLessEqual(len(sel["recommendations"][0]["timbre_card"]), 140)
            self.assertNotIn("torch", sys.modules)

            t1 = time.perf_counter()
            pal = propose_palettes(
                mix_profile="trap_rap",
                emotion_vector={"valence": -0.6, "tension": 0.5, "energy": 0.7, "depth": 0.4},
                target_key="C minor",
                num_palettes=3,
                db_path=db_path,
            )
            ms_pal = (time.perf_counter() - t1) * 1000.0
            self.assertLess(ms_pal, 50.0, f"propose_palettes too slow on 5k DB: {ms_pal:.1f}ms")
            self.assertEqual(pal["verdict"], "PASS")
            self.assertEqual(len(pal["palettes"]), 3)
            self.assertFalse(pal["user_samples_locked"])
            self.assertNotIn("torch", sys.modules)

            # 3) Project-local Samples/ exclusivity & melodic_sequence prep recipe suggestion
            proj_dir = os.path.join(tmpdir, "user_proj")
            user_samples = os.path.join(proj_dir, "Samples")
            os.makedirs(user_samples, exist_ok=True)
            # Single clean C4 note
            sf.write(os.path.join(user_samples, "user_keys_C4.wav"), _synth_tone(261.63, 0.9, harmonics=3), 44100)
            # Kick-like low thump
            t_k = np.linspace(0.0, 0.35, int(44100 * 0.35), endpoint=False)
            f_sw = 52.0 + 95.0 * np.exp(-t_k * 28.0)
            y_k = np.sin(2.0 * np.pi * np.cumsum(f_sw) / 44100.0) * np.exp(-t_k * 10.0) * 0.7
            sf.write(os.path.join(user_samples, "user_kick.wav"), np.column_stack([y_k, y_k]).astype(np.float32), 44100)
            # Multi-note melodic sequence (3 distinct notes -> should propose slice_s recipes!)
            seq = np.vstack([
                _synth_tone(261.63, 0.45, harmonics=3),
                _synth_tone(311.13, 0.45, harmonics=3),
                _synth_tone(392.00, 0.45, harmonics=3),
            ])
            sf.write(os.path.join(user_samples, "user_melody_loop.wav"), seq, 44100)

            cat = index_project_samples(proj_dir, with_clap=False, write_catalog=True)
            self.assertTrue(os.path.isfile(os.path.join(proj_dir, "samples_catalog.json")))
            self.assertTrue(cat["user_samples_locked"])
            self.assertEqual(cat["sample_count"], 3)
            loop_entry = next(s for s in cat["samples"] if s["filename"] == "user_melody_loop.wav")
            self.assertFalse(loop_entry["playable_chromatically"])
            if loop_entry["sample_nature"] == "melodic_sequence":
                self.assertTrue(any("slice_s" in r["sample_prep_spec"] for r in loop_entry["suggested_prep_recipes"]))

            # Exclusivity check: propose_palettes MUST NOT import any external sample from db_path
            pal_locked = propose_palettes(
                mix_profile="trap_rap",
                emotion_vector={"valence": -0.4, "tension": 0.4, "energy": 0.6, "depth": 0.4},
                target_key="C minor",
                roles=["kick", "chord_keys", "lead_melody"],
                project_samples_dir=proj_dir,
                allow_gl_synths=True,
                db_path=db_path,
            )
            self.assertTrue(pal_locked["user_samples_locked"])
            for p in pal_locked["palettes"]:
                for slot in p["slots"]:
                    if slot["filepath"]:
                        self.assertTrue(
                            os.path.abspath(slot["filepath"]).startswith(os.path.abspath(proj_dir)),
                            f"External sample leaked into user-locked palette: {slot['filepath']}",
                        )

            # Verify MCP wrappers exist and return valid JSON
            mcp_cat = json.loads(audio_index_server.index_project_samples(project_dir=proj_dir, with_clap=False))
            self.assertEqual(mcp_cat["sample_count"], 3)
            mcp_pal = json.loads(audio_index_server.propose_palettes(project_samples_dir=proj_dir, db_path=db_path))
            self.assertEqual(len(mcp_pal["palettes"]), 3)

    @unittest.skipUnless(shutil.which("gridline"), "binaire gridline requis")
    def test_phase_b_soundcheck_synthetic_headless_preview_and_listener(self) -> None:
        sr = 44100
        t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
        env = np.minimum(1.0, t / 0.02) * np.exp(-t * 1.4)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1) Silent track -> SWAP / CHECK_ROUTING_OR_SWAP
            p_sil = os.path.join(tmpdir, "silent.wav")
            sf.write(p_sil, np.zeros((sr * 2, 2), dtype=np.float32), sr)
            r_sil = audit_track_soundcheck(wav_path=p_sil, role="lead_melody", track_name="Silent")
            self.assertEqual(r_sil["verdict"], "SWAP")
            self.assertEqual(r_sil["next_action"]["code"], "CHECK_ROUTING_OR_SWAP")

            # 2) +18 ct detuned A4 -> FIX / ADJUST_FINE_TUNE (-18 ct)
            f_det = 440.0 * (2.0 ** (18.0 / 1200.0))
            y_det = (0.35 * np.sin(2 * np.pi * f_det * t) + 0.10 * np.sin(4 * np.pi * f_det * t)) * env
            p_det = os.path.join(tmpdir, "detuned.wav")
            sf.write(p_det, np.column_stack([y_det, y_det]).astype(np.float32), sr)
            r_det = audit_track_soundcheck(wav_path=p_det, role="lead_melody", track_name="Detuned", expected_midis=[69])
            self.assertEqual(r_det["verdict"], "FIX")
            self.assertEqual(r_det["next_action"]["code"], "ADJUST_FINE_TUNE")
            self.assertAlmostEqual(r_det["next_action"]["params"]["fine_tune_delta_ct"], -18, delta=2)

            # 3) Unfiltered bright saw for valence = -0.7 -> FIX / APPLY_LPF
            saw = sum((1.0 / k) * np.sin(2 * np.pi * 440.0 * k * t) for k in range(1, 28))
            saw = saw / np.max(np.abs(saw)) * 0.45 * env
            p_saw = os.path.join(tmpdir, "saw.wav")
            sf.write(p_saw, np.column_stack([saw, saw]).astype(np.float32), sr)
            r_saw = audit_track_soundcheck(
                wav_path=p_saw, role="lead_melody", track_name="BrightSaw", valence=-0.7, expected_midis=[69]
            )
            self.assertEqual(r_saw["verdict"], "FIX")
            self.assertIn("APPLY_LPF", r_saw["next_action"]["code"])
            self.assertIn("cutoff_hz", r_saw["next_action"]["params"])

            # 4) Real headless Gridline soundcheck render (gl.keys chords)
            r_gl = audit_track_soundcheck(
                project_dir=tmpdir,
                track_spec={"name": "Keys_Main", "role": "chord_keys", "plugin": "gl.keys", "volume": 0.78},
                bpm=124.0,
                key="C minor",
                valence=-0.4,
            )
            self.assertEqual(r_gl["verdict"], "PASS", f"Expected PASS on gl.keys soundcheck, got {r_gl}")
            self.assertTrue(os.path.isfile(r_gl["wav_path"]))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, ".tmp_soundcheck")))
            self.assertTrue(os.path.isfile(r_gl["diagnostics_path"]))

            # 5) Palette preview export
            p_clean = os.path.join(tmpdir, "clean_bass.wav")
            sf.write(p_clean, _synth_tone(65.41, 3.0, harmonics=2), sr)
            prev = export_palette_preview(
                [("Bass", p_clean), ("Keys", r_gl["wav_path"])],
                os.path.join(tmpdir, "renders", "palette_preview.wav"),
            )
            self.assertTrue(os.path.isfile(prev["preview_wav"]))
            self.assertTrue(os.path.isfile(prev["index_txt"]))
            self.assertIn("0:00 Bass", prev["index_line"])
            self.assertIn("0:03 Keys", prev["index_line"])

            # 6) MultimodalAudioListener contract parity via callable hook
            mm = MultimodalAudioListener(
                callable_fn=lambda _w, _p, _ts, _dsp: {
                    "verdict": "PASS",
                    "score": 0.92,
                    "human_diagnosis": ["Timbre chaud cohérent avec l'intention."],
                    "next_action": {"code": "KEEP_TRACK", "params": {}},
                }
            )
            r_mm = mm.evaluate_track(p_clean, {"name": "Bass", "role": "bass_sub"}, {"checks": {}, "metrics": {}})
            self.assertEqual(r_mm["verdict"], "PASS")
            self.assertEqual(r_mm["backend"], "multimodal_audio")

            # 7) MCP wrapper smoke test
            mcp_sc = json.loads(audio_analyzer_server.audit_track_soundcheck(wav_path=p_sil, role="lead_melody"))
            self.assertEqual(mcp_sc["verdict"], "SWAP")

    @unittest.skipUnless(shutil.which("gridline"), "binaire gridline requis")
    def test_phase_c_vertical_incremental_headless_trap_and_ambient(self) -> None:
        sr = 44100
        t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1) Synthetic V3 bass-vs-harmony chroma clash test (C2 bass vs C#3/F3/G#3 chords)
            bass_c2 = 0.35 * np.sin(2 * np.pi * 65.41 * t)
            clash_chords = 0.20 * (
                np.sin(2 * np.pi * 277.18 * t) + np.sin(2 * np.pi * 349.23 * t) + np.sin(2 * np.pi * 415.30 * t)
            )
            y_clash = np.column_stack([bass_c2 + clash_chords, bass_c2 + clash_chords]).astype(np.float32)
            p_clash = os.path.join(tmpdir, "v3_clash.wav")
            sf.write(p_clash, y_clash, sr)
            r_cl = analyze_vertical_stage(p_clash, "V3_HARMONY", profile="trap_rap")
            self.assertFalse(r_cl["checks"]["chroma_consonance_ok"])
            self.assertEqual(r_cl["next_action"]["code"], "FIX_HARMONY_CHROMA_CLASH")

            # Consonant C2 bass + C3/Eb3/G3 chords
            ok_chords = 0.16 * (
                np.sin(2 * np.pi * 261.63 * t) + np.sin(2 * np.pi * 311.13 * t) + np.sin(2 * np.pi * 392.00 * t)
            )
            y_cons = np.column_stack([bass_c2 + ok_chords, bass_c2 + ok_chords]).astype(np.float32)
            p_cons = os.path.join(tmpdir, "v3_ok.wav")
            sf.write(p_cons, y_cons, sr)
            r_ok = analyze_vertical_stage(p_cons, "V3_HARMONY", profile="trap_rap")
            self.assertTrue(r_ok["checks"]["chroma_consonance_ok"])

            # 2) Headless trap_rap core_loop.gridline -> V1 -> V2 -> V3 -> V4
            trap_dir = os.path.join(tmpdir, "trap_proj")
            os.makedirs(trap_dir, exist_ok=True)
            bp_trap = {
                "title": "TrapCore",
                "genre": "trap",
                "mix_profile": "trap_rap",
                "has_drums": True,
                "emotion_vector": {"valence": -0.5, "tension": 0.4, "energy": 0.7, "depth": 0.4},
                "bpm": 132,
                "time_signature": {"numerator": 4, "denominator": 4},
                "key": "C minor",
            }
            with open(os.path.join(trap_dir, "song_blueprint.json"), "w", encoding="utf-8") as fh:
                json.dump(bp_trap, fh)

            meta_trap = {"bpm": 132.0, "beats_per_bar": 4, "denominator": 4, "mix_profile": "trap_rap", "valence": -0.5}
            sections_trap = [("Intro", 0, 4, "intro"), ("Drop", 4, 4, "drop")]
            channels_trap = [
                ChannelSpec("Kick", "kick", mixer=1, bus="drums", plugin="gl.analog", role="bass", volume=0.72, sidechain_buses=[(18, 1.0)]),
                ChannelSpec("808 Bass", "bass", mixer=2, bus="bass", plugin="gl.analog", role="bass", volume=0.72),
                ChannelSpec("Snare", "snare", mixer=3, bus="drums", plugin="gl.pluck", role="pluck", volume=0.58),
                ChannelSpec("Chords", "chords", mixer=4, bus="music", plugin="gl.keys", role="chords", volume=0.55),
                ChannelSpec("Lead", "lead", mixer=5, bus="music", plugin="gl.pluck", role="pluck", volume=0.78),
            ]
            tracks_trap: dict[str, list[dict]] = {"kick": [], "bass": [], "snare": [], "chords": [], "lead": []}
            for bar in range(4, 8):
                b0 = float(bar * 4)
                tracks_trap["kick"].extend([
                    {"start": b0 + 0.0, "length": 0.25, "pitch": 36, "velocity": 105},
                    {"start": b0 + 2.5, "length": 0.25, "pitch": 36, "velocity": 98},
                ])
                tracks_trap["bass"].append({"start": b0 + 0.25, "length": 3.25, "pitch": 36 if bar % 2 == 0 else 43, "velocity": 92})
                tracks_trap["snare"].extend([
                    {"start": b0 + 1.0, "length": 0.2, "pitch": 62, "velocity": 90},
                    {"start": b0 + 3.0, "length": 0.2, "pitch": 62, "velocity": 96},
                ])
                for p in (48, 51, 55):
                    tracks_trap["chords"].append({"start": b0, "length": 3.5, "pitch": p, "velocity": 68})
                tracks_trap["lead"].extend([
                    {"start": b0 + 0.5, "length": 0.75, "pitch": 67, "velocity": 94},
                    {"start": b0 + 2.0, "length": 1.0, "pitch": 63, "velocity": 90},
                ])

            core_trap = os.path.join(trap_dir, "core_loop.gridline")
            cinfo = build_core_loop_project(
                proj_path=core_trap,
                export_dir=os.path.join(trap_dir, "exports"),
                samples_dir=os.path.join(trap_dir, "samples"),
                meta=meta_trap,
                sections=sections_trap,
                channels=channels_trap,
                tracks=tracks_trap,
                max_bars=4,
                insert_eq_params=eq_kick_clean(1) + eq_music_pocket(4),
                limiter_gain_db=4.0,
                limiter_ceiling_db=-2.2,
            )
            self.assertEqual(cinfo["source_section"], "Drop")
            self.assertEqual(cinfo["bars"], 4)

            v_all = run_all_vertical_stages(core_trap, profile="trap_rap", stop_on_fail=False, render_stems_for_stage=True)
            self.assertEqual(set(v_all["stages"].keys()), {"V1_LOW_END", "V2_GROOVE", "V3_HARMONY", "V4_LEAD"})
            self.assertEqual(v_all["stages"]["V3_HARMONY"]["target_layer_tracks"], ["Chords"])
            self.assertIn("Kick", v_all["stages"]["V3_HARMONY"]["forbidden_tracks"])
            self.assertIn("808 Bass", v_all["stages"]["V3_HARMONY"]["forbidden_tracks"])
            self.assertIn("Snare", v_all["stages"]["V3_HARMONY"]["forbidden_tracks"])
            self.assertTrue(os.path.isfile(v_all["stages"]["V4_LEAD"]["diagnostics_path"]))

            # 3) Headless classical_ambient (has_drums=False) -> skips V2_GROOVE cleanly!
            amb_dir = os.path.join(tmpdir, "amb_proj")
            os.makedirs(amb_dir, exist_ok=True)
            bp_amb = {
                "title": "AmbCore",
                "genre": "classical",
                "mix_profile": "classical_ambient",
                "has_drums": False,
                "emotion_vector": {"valence": -0.3, "tension": 0.2, "energy": 0.3, "depth": 0.8},
                "bpm": 88,
                "time_signature": {"numerator": 4, "denominator": 4},
                "key": "D minor",
            }
            with open(os.path.join(amb_dir, "song_blueprint.json"), "w", encoding="utf-8") as fh:
                json.dump(bp_amb, fh)

            meta_amb = {"bpm": 88.0, "beats_per_bar": 4, "denominator": 4, "mix_profile": "classical_ambient", "valence": -0.3}
            channels_amb = [
                ChannelSpec("Cello Bass", "bass", mixer=1, bus="bass", plugin="gl.string", role="pad", volume=0.65),
                ChannelSpec("Pad Strings", "chords", mixer=2, bus="music", plugin="gl.string", role="pad", volume=0.58),
                ChannelSpec("Lead Piano", "lead", mixer=3, bus="music", plugin="gl.keys", role="lead", volume=0.78),
            ]
            tracks_amb: dict[str, list[dict]] = {"bass": [], "chords": [], "lead": []}
            for bar in range(4):
                b0 = float(bar * 4)
                tracks_amb["bass"].append({"start": b0, "length": 3.8, "pitch": 38, "velocity": 78})
                for p in (50, 53, 57):
                    tracks_amb["chords"].append({"start": b0, "length": 3.8, "pitch": p, "velocity": 68})
                tracks_amb["lead"].append({"start": b0 + 1.0, "length": 2.0, "pitch": 69 if bar % 2 == 0 else 65, "velocity": 86})

            core_amb = os.path.join(amb_dir, "core_loop.gridline")
            build_core_loop_project(
                proj_path=core_amb,
                export_dir=os.path.join(amb_dir, "exports"),
                samples_dir=os.path.join(amb_dir, "samples"),
                meta=meta_amb,
                sections=[("Climax", 0, 4, "climax")],
                channels=channels_amb,
                tracks=tracks_amb,
                max_bars=4,
                limiter_gain_db=2.0,
                limiter_ceiling_db=-2.2,
            )
            r_v2 = run_vertical_stage(core_amb, "V2_GROOVE", profile="classical_ambient")
            self.assertEqual(r_v2["verdict"], "PASS")
            self.assertTrue(r_v2["skipped"])

            mcp_v2 = json.loads(audio_analyzer_server.audit_vertical_step(
                stage="V2_GROOVE", project_path=core_amb, profile="classical_ambient"
            ))
            self.assertTrue(mcp_v2["skipped"])

    def test_phase_d_horizontal_seams_and_state_machine_guardrails(self) -> None:
        sr = 44100
        bpm = 120.0  # 1 beat = 0.5 s, 1 bar (4/4) = 2.0 s, 4 bars = 8.0 s
        n_total = int(sr * 8.0)
        t = np.linspace(0.0, 8.0, n_total, endpoint=False)
        sig = 0.25 * np.sin(2.0 * np.pi * 220.0 * t) + 0.12 * np.sin(2.0 * np.pi * 440.0 * t)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1) Flat Intro -> Drop transition (ΔLUFS ~ 0 dB, no dropout/sweep) -> MUST FAIL (FIX)
            p_flat = os.path.join(tmpdir, "seam_flat.wav")
            sf.write(p_flat, np.column_stack([sig, sig]).astype(np.float32), sr)
            r_flat = audit_seam_wav(p_flat, from_section="Intro", to_section="Drop", bpm=bpm, beats_per_bar=4)
            self.assertEqual(r_flat["verdict"], "FIX")
            self.assertFalse(r_flat["checks"]["energy_contrast_ok"])
            self.assertFalse(r_flat["checks"]["transition_event_ok"])

            # 2) Proper Intro -> Drop transition (contrast +4.5 dB + beat4 dropout on beat 8 [3.5s..4.0s]) -> PASS
            y_good = sig.copy()
            # Intro (0..4s) quieter, Drop (4..8s) +5 dB louder
            y_good[: int(4.0 * sr)] *= 0.55
            y_good[int(4.0 * sr) :] *= 1.05
            # Beat 4 dropout on last beat before Drop (3.5s..4.0s) with smooth 10ms fade edges
            i_d0, i_d1 = int(3.5 * sr), int(4.0 * sr)
            fade = int(0.01 * sr)
            y_good[i_d0 : i_d0 + fade] *= np.linspace(1.0, 0.05, fade)
            y_good[i_d0 + fade : i_d1 - fade] *= 0.05
            y_good[i_d1 - fade : i_d1] *= np.linspace(0.05, 1.0, fade)
            p_good = os.path.join(tmpdir, "seam_good.wav")
            sf.write(p_good, np.column_stack([y_good, y_good]).astype(np.float32), sr)
            r_good = audit_seam_wav(p_good, from_section="Intro", to_section="Drop", bpm=bpm, beats_per_bar=4)
            self.assertEqual(r_good["verdict"], "PASS", f"Expected PASS on good seam, got {r_good}")
            self.assertTrue(r_good["metrics"]["dropout_detected"])
            self.assertGreaterEqual(r_good["metrics"]["delta_lufs"], 2.5)

            # 3) Abruptly cut Outro -> FIX / APPLY_OUTRO_FADE
            p_abrupt = os.path.join(tmpdir, "outro_abrupt.wav")
            sf.write(p_abrupt, np.column_stack([sig, sig]).astype(np.float32), sr)
            r_abrupt = audit_outro_wav(p_abrupt)
            self.assertEqual(r_abrupt["verdict"], "FIX")
            self.assertEqual(r_abrupt["next_action"]["code"], "APPLY_OUTRO_FADE")

            # 4) Properly faded Outro (smooth logarithmic fade over last 4 s -> < -50 dBFS in last 1s) -> PASS
            y_fade = sig.copy()
            f0 = int(4.0 * sr)
            alpha = np.linspace(0.0, 1.0, n_total - f0)
            y_fade[f0:] *= np.power(10.0, (-3.2 * alpha))
            y_fade[-int(0.5 * sr) :] *= np.linspace(1.0, 0.0, int(0.5 * sr))
            p_faded = os.path.join(tmpdir, "outro_faded.wav")
            sf.write(p_faded, np.column_stack([y_fade, y_fade]).astype(np.float32), sr)
            r_faded = audit_outro_wav(p_faded)
            self.assertEqual(r_faded["verdict"], "PASS", f"Expected PASS on faded outro, got {r_faded}")

            # MCP wrapper smoke test
            mcp_outro = json.loads(audio_analyzer_server.audit_horizontal_seam(wav_path=p_faded, is_outro=True))
            self.assertEqual(mcp_outro["verdict"], "PASS")

            # 5) LoopFirstStateMachine: user_samples_locked, forbidden_tracks, retry escalation to SWAP_SAMPLE
            proj_dir = os.path.join(tmpdir, "runner_proj")
            os.makedirs(os.path.join(proj_dir, "Samples"), exist_ok=True)
            sf.write(os.path.join(proj_dir, "Samples", "kick.wav"), _synth_tone(60.0, 0.3), sr)

            sm = LoopFirstStateMachine(proj_dir, max_attempts=2)
            self.assertTrue(sm.state["user_samples_locked"])
            self.assertEqual(sm.state["stage"], "E0_PALETTE")

            sm.select_palette(
                "palette_A",
                tracks=["Kick", "808", "Snare", "Chords", "Lead"],
                track_stage_map={
                    "Kick": "V1_LOW_END",
                    "808": "V1_LOW_END",
                    "Snare": "V2_GROOVE",
                    "Chords": "V3_HARMONY",
                    "Lead": "V4_LEAD",
                },
            )
            self.assertEqual(sm.state["stage"], "E1_SOUNDCHECK")
            for tr in ("Kick", "808", "Snare", "Chords", "Lead"):
                sm.record_soundcheck(tr, "PASS")
            self.assertEqual(sm.state["stage"], "E2_VERTICAL")
            self.assertEqual(sm.state["substage"], "V1_LOW_END")

            sm.record_vertical("V1_LOW_END", "PASS")
            sm.record_vertical("V2_GROOVE", "PASS")
            self.assertEqual(sm.state["substage"], "V3_HARMONY")

            # Fault-isolation guardrail: Kick, 808, Snare MUST be forbidden during V3_HARMONY
            with self.assertRaises(PermissionError):
                sm.validate_edit_target("Kick")
            self.assertTrue(sm.validate_edit_target("Chords")["allowed"])

            # Fail V3_HARMONY twice -> must escalate to SWAP_SAMPLE
            sm.record_vertical("V3_HARMONY", "FAIL", blocker="low_mids 31% > 24%")
            self.assertIn("HPF_180Hz", sm.state["next_mandatory_step"]["allowed_actions"])
            sm.record_vertical("V3_HARMONY", "FAIL", blocker="low_mids still high")
            self.assertEqual(sm.state["next_mandatory_step"]["allowed_actions"], ["SWAP_SAMPLE"])

            # Pass V3 + V4 -> locks core_loop & transitions to E3_HORIZONTAL
            sm.record_vertical("V3_HARMONY", "PASS")
            sm.record_vertical("V4_LEAD", "PASS")
            self.assertTrue(sm.state["core_loop_locked"])
            self.assertEqual(sm.state["stage"], "E3_HORIZONTAL")

            sm.record_horizontal("Intro -> Drop", "PASS")
            sm.record_horizontal("Outro", "PASS")
            self.assertEqual(sm.state["stage"], "E4_FINAL_GATE")

            sm.record_final_gate(True)
            self.assertEqual(sm.state["stage"], "DONE")
            self.assertTrue(os.path.isfile(os.path.join(proj_dir, "pipeline_state.json")))


if __name__ == "__main__":
    unittest.main()
