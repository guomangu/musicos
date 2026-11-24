#!/usr/bin/env python3
"""Regression tests for the quality/portability improvements:
- token-based sample category classification (no substring false positives)
- loop BPM filtering in palette building + blueprint_tracks carrying root_note/fine_tune_ct
- root_note propagation from song_blueprint.json into ChannelSpec (gl.sampler)
- notchable (narrow) vs tonal (wide) 1/3-octave resonances
- swing application, sparse-note humanization, BPM-capped Haas pocket
- portable ROOT / bundled Gridline template fallback, installable requirements pins
- auto_fetch hard-disabled when user samples are locked (Invariant #1)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "mcp-servers", "audio-analyzer"), os.path.join(ROOT, "mcp-servers", "audio-index")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import indexer  # noqa: E402
from groove_lib import apply_anti_machine_gun, apply_haas_pocket, apply_swing, ms_to_beats  # noqa: E402
from palette_builder import loop_bpm_penalty, palette_slots_to_blueprint_tracks, propose_palettes  # noqa: E402


class TestTokenCategoryClassification(unittest.TestCase):
    def test_no_substring_false_positives(self):
        cat, _, is_loop = indexer._infer_category_and_pack("pack/x.wav", "my chat sample.wav", 0.5)
        self.assertEqual(cat, "other")  # "chat" must NOT match "hat"
        cat, _, _ = indexer._infer_category_and_pack("pack/x.wav", "scalplover.wav", 0.5)
        self.assertEqual(cat, "other")  # "scalplover" must NOT match "cp"

    def test_real_keywords_still_match(self):
        self.assertEqual(indexer._infer_category_and_pack("p/x.wav", "Snare_Roll_002.wav", 0.5)[0], "snare")
        self.assertEqual(indexer._infer_category_and_pack("p/x.wav", "open_hat.wav", 0.5)[0], "open_hat")
        self.assertEqual(indexer._infer_category_and_pack("p/x.wav", "Kick_909.wav", 0.5)[0], "kick")
        cat, _, _ = indexer._infer_category_and_pack("p/x.wav", "808s folder.wav", 0.5)
        self.assertEqual(cat, "808")  # plural "808s" singularized to "808"

    def test_loop_detection_by_token_not_substring(self):
        _, _, is_loop = indexer._infer_category_and_pack("p/x.wav", "pad_full.wav", 3.0)
        self.assertTrue(is_loop)
        _, _, is_loop = indexer._infer_category_and_pack("p/x.wav", "oneshot_hit.wav", 0.4)
        self.assertFalse(is_loop)


class TestLoopBpmFiltering(unittest.TestCase):
    def test_penalty_bands(self):
        loop140 = {"is_loop": True, "bpm": 140.0}
        oneshot = {"is_loop": False, "bpm": 140.0}
        unknown_loop = {"is_loop": True, "bpm": None}
        self.assertEqual(loop_bpm_penalty(oneshot, 90.0), 0.0)
        self.assertEqual(loop_bpm_penalty(unknown_loop, 90.0), 0.0)
        # v3.0 lock: exact tempo -> 0; any residual > 0.05 BPM needs an exact time-stretch (0.15)
        self.assertEqual(loop_bpm_penalty({"is_loop": True, "bpm": 90.0}, 90.0), 0.0)
        self.assertEqual(loop_bpm_penalty({"is_loop": True, "bpm": 92.0}, 90.0), 0.15)
        self.assertEqual(loop_bpm_penalty({"is_loop": True, "bpm": 100.0}, 90.0), 0.15)  # r_eff=0.9 in [0.88,1.14]
        self.assertGreaterEqual(loop_bpm_penalty(loop140, 90.0), 10.0)  # r_eff=0.643 out of range -> excluded
        self.assertGreaterEqual(loop_bpm_penalty({"is_loop": True, "bpm": 110.0}, 145.0), 10.0)  # r_eff=1.318 -> excluded
        # half/double tempo allowed: 180 BPM loop in a 90 BPM project -> ratio 0.5, no stretch
        self.assertEqual(loop_bpm_penalty({"is_loop": True, "bpm": 180.0}, 90.0), 0.0)

    def test_propose_palettes_excludes_wrong_bpm_loop_and_emits_root_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = {
                "project_dir": tmp,
                "user_samples_locked": True,
                "samples": [
                    {
                        "filepath": os.path.join(tmp, "loop_140.wav"),
                        "filename": "loop_140.wav",
                        "hard_reject": False,
                        "playable_chromatically": True,
                        "compatible_roles": ["lead_melody"],
                        "role_scores": {"lead_melody": 0.95},
                        "comfort_score": 0.9,
                        "centroid_hz": 2000.0,
                        "attack_ms": 10.0,
                        "midi_root": 60,
                        "recommended_gl_fine_tune_ct": 0,
                        "bark_24": None,
                        "timbre_card": "loop 140 bpm",
                        "bpm": 140.0,
                        "is_loop": True,
                    },
                    {
                        "filepath": os.path.join(tmp, "good_oneshot.wav"),
                        "filename": "good_oneshot.wav",
                        "hard_reject": False,
                        "playable_chromatically": True,
                        "compatible_roles": ["lead_melody"],
                        "role_scores": {"lead_melody": 0.80},
                        "comfort_score": 0.85,
                        "centroid_hz": 1800.0,
                        "attack_ms": 8.0,
                        "midi_root": 57,
                        "recommended_gl_fine_tune_ct": -13,
                        "bark_24": None,
                        "timbre_card": "clean one-shot A3",
                        "bpm": None,
                        "is_loop": False,
                    },
                ],
            }
            with open(os.path.join(tmp, "samples_catalog.json"), "w", encoding="utf-8") as fh:
                json.dump(catalog, fh)
            res = propose_palettes(
                mix_profile="trap_rap",
                target_key="A minor",
                roles=["lead_melody"],
                project_samples_dir=tmp,
                allow_gl_synths=False,
                num_palettes=1,
                bpm=90.0,
                bpm_tolerance=6.0,
            )
        self.assertEqual(res["bpm"], 90.0)
        best = res["palettes"][0]
        self.assertEqual(best["missing_roles"], [])
        slot = best["slots"][0]
        self.assertTrue(slot["filepath"].endswith("good_oneshot.wav"), f"wrong-BPM loop should be excluded: {slot}")
        self.assertEqual(slot["midi_root"], 57)
        bt = best["blueprint_tracks"][0]
        self.assertEqual(bt["root_note"], 57)
        self.assertEqual(bt["fine_tune_ct"], -13)
        self.assertEqual(bt["source_type"], "sample")

    def test_blueprint_tracks_helper_maps_sample_fields(self):
        chosen = [{
            "role": "kick", "source_type": "project_sample", "filepath": "/x/kick.wav",
            "plugin": "gl.sampler", "midi_root": 36, "recommended_fine_tune_ct": 4,
        }]
        tr = palette_slots_to_blueprint_tracks(chosen)[0]
        self.assertEqual(tr["root_note"], 36)
        self.assertEqual(tr["fine_tune_ct"], 4)
        self.assertEqual(tr["sample_path"], "/x/kick.wav")
        self.assertEqual(tr["source_type"], "sample")


class TestRootNotePropagation(unittest.TestCase):
    def test_blueprint_root_note_reaches_channel_spec(self):
        import song_builder

        with tempfile.TemporaryDirectory() as tmp:
            bp = {
                "tracks": [
                    {
                        "name": "Lead", "role": "lead_melody", "source_type": "sample",
                        "sample_path": "lead.wav", "mixer_track": 5,
                        "root_note": 57, "fine_tune_ct": -13,
                    }
                ]
            }
            with open(os.path.join(tmp, "song_blueprint.json"), "w", encoding="utf-8") as fh:
                json.dump(bp, fh)
            chans = song_builder._default_channels_for_plan({}, ["lead_melody"], tmp, valence=0.0)
        self.assertEqual(len(chans), 1)
        self.assertEqual(chans[0].plugin, "gl.sampler")
        self.assertEqual(chans[0].root_note, 57)
        self.assertEqual(chans[0].fine_tune, -13)

    def test_schema_accepts_root_note_and_fine_tune_ct(self):
        import jsonschema

        with open(os.path.join(ROOT, "schemas", "song_blueprint.schema.json"), encoding="utf-8") as fh:
            schema = json.load(fh)
        bp = json.load(open(os.path.join(ROOT, "projects", "validation_fm_dark", "song_blueprint.json"), encoding="utf-8"))
        bp = dict(bp)
        bp["tracks"] = [{
            "name": "Lead", "role": "lead_melody", "source_type": "sample",
            "sample_path": "x.wav", "mixer_track": 3, "root_note": 57, "fine_tune_ct": -13,
        }]
        jsonschema.validate(bp, schema)  # must not raise


class TestResonanceClassification(unittest.TestCase):
    def test_narrow_strong_peak_is_notchable_wide_plateau_is_not(self):
        from deep_diagnostics import classify_resonances

        bands = [
            {"freq_hz": 250.0, "local_excess_db": 6.5, "is_local_peak": True, "peak_width_bands": 1},
            {"freq_hz": 1000.0, "local_excess_db": 7.2, "is_local_peak": True, "peak_width_bands": 4},
            {"freq_hz": 3000.0, "local_excess_db": 5.4, "is_local_peak": True, "peak_width_bands": 1},
            {"freq_hz": 5000.0, "local_excess_db": 1.0, "is_local_peak": False, "peak_width_bands": 1},
        ]
        notchable, candidates = classify_resonances(bands)
        self.assertEqual([b["freq_hz"] for b in notchable], [250.0])
        self.assertEqual([b["freq_hz"] for b in candidates], [250.0, 1000.0, 3000.0])


class TestGrooveHumanization(unittest.TestCase):
    def test_swing_shifts_only_offbeat_eighths(self):
        notes = [
            {"start": 0.0, "length": 0.25, "pitch": 60, "velocity": 100},
            {"start": 0.5, "length": 0.25, "pitch": 62, "velocity": 100},
            {"start": 1.0, "length": 0.25, "pitch": 64, "velocity": 100},
            {"start": 1.5, "length": 0.25, "pitch": 65, "velocity": 100},
            {"start": 1.25, "length": 0.1, "pitch": 67, "velocity": 100},  # syncopated: untouched
        ]
        out = apply_swing(notes, swing=0.2, beats_per_bar=4.0)
        self.assertEqual(out[0]["start"], 0.0)
        self.assertAlmostEqual(out[1]["start"], 0.5 + 0.2 * 0.25, places=5)
        self.assertEqual(out[2]["start"], 1.0)
        self.assertAlmostEqual(out[3]["start"], 1.5 + 0.2 * 0.25, places=5)
        self.assertEqual(out[4]["start"], 1.25)

    def test_humanize_sparse_accents_without_jitter(self):
        sparse = [
            {"start": 0.0, "length": 1.0, "pitch": 60, "velocity": 100},
            {"start": 0.5, "length": 1.0, "pitch": 64, "velocity": 100},
        ]
        out = apply_anti_machine_gun(sparse, beats_per_bar=4.0, humanize_sparse=True)
        self.assertEqual(out[0]["velocity"], 104)
        self.assertEqual(out[1]["velocity"], 96)
        # default behavior unchanged: sparse notes untouched
        out2 = apply_anti_machine_gun([dict(n) for n in sparse], beats_per_bar=4.0)
        self.assertEqual([n["velocity"] for n in out2], [100, 100])

    def test_haas_pocket_capped_at_5_percent_of_beat(self):
        notes = [{"start": 0.0, "length": 0.5, "pitch": 36, "velocity": 100}]
        out = apply_haas_pocket({"808": notes}, bpm=300.0, profile="trap_rap")
        shift_ms = out["808"][0]["start"] * 60000.0 / 300.0  # start is in beats
        self.assertAlmostEqual(shift_ms, 0.05 * (60000.0 / 300.0), places=3)  # capped at 10 ms


class TestPortabilityAndPackaging(unittest.TestCase):
    def test_bundled_template_fallback_exists_and_resolves(self):
        import studio_lib

        tpl = studio_lib.resolve_empty_template()
        self.assertTrue(os.path.isfile(tpl), f"template introuvable: {tpl}")
        bundled = os.path.join(ROOT, "assets", "empty_with_mixer_tracks.gridline")
        self.assertTrue(os.path.isfile(bundled))
        with open(bundled, encoding="utf-8") as fh:
            d = json.load(fh)
        self.assertGreaterEqual(len(d["mixer"]), 17)  # Master + Insert 1..16

    def test_requirements_essentia_pin_is_installable(self):
        with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as fh:
            pin_lines = [ln.strip() for ln in fh if ln.strip().startswith("essentia")]
        self.assertEqual(len(pin_lines), 1)
        self.assertIn("essentia>=2.1b6.dev1389", pin_lines[0])
        self.assertNotIn("dev1438", pin_lines[0])

    def test_auto_fetch_disabled_when_user_samples_locked(self):
        from indexer import init_db
        from select_sample import select_best_from_db

        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "samples.db")
            conn = init_db(db)
            conn.close()
            res = select_best_from_db(db_path=db, role="kick", auto_fetch=True, user_samples_locked=True, with_clap=False)
        self.assertNotIn("online_fallback", res)




class StubMCP:
    """Minimal GridlineMCP stub for in-session audition tests (no gridline binary)."""

    def __init__(self, channels: list[dict], lufs_by_channel: dict[int, float]):
        self._channels = channels
        self._lufs = lufs_by_channel
        self._solo: int | None = None
        self.edits: list[dict] = []

    def call(self, tool: str, args: dict):
        if tool == "list_channels":
            return {"channels": self._channels}
        if tool == "get_channel":
            ch = int(args["channel"])
            return next((c for c in self._channels if int(c.get("channel", 0)) == ch), {"name": f"ch{ch}"})
        if tool == "edit_channel":
            self.edits.append(dict(args))
            return {}
        if tool == "save_project":
            return {}
        raise RuntimeError(f"unexpected tool {tool}")

    def solo_channel(self, channel: int, channels=None):
        self._solo = int(channel)
        return [int(c["channel"]) for c in self._channels if int(c["channel"]) != int(channel)]

    def unsolo(self, muted, keep_solo=None):
        self._solo = keep_solo

    def analyze_loudness(self, scope="pattern", **kw):
        return {"lufs": self._lufs.get(self._solo, -30.0), "silent": False}


class TestInSessionAudition(unittest.TestCase):
    def test_balance_tracks_in_session_trims_toward_role_targets(self):
        from gain_staging_lib import balance_tracks_in_session

        channels = [
            {"channel": 1, "name": "Kick", "muted": False},
            {"channel": 2, "name": "HiHat", "muted": False},
        ]
        mcp = StubMCP(channels, {1: -16.0, 2: -8.0})  # hat way too loud vs target ~-26
        tracks = [
            {"name": "Kick", "channel": 1, "pattern": 1, "role": "kick", "volume": 0.76},
            {"name": "HiHat", "channel": 2, "pattern": 2, "role": "hihat_perc", "volume": 0.52},
        ]
        report = balance_tracks_in_session(mcp, tracks, profile="trap_rap", has_drums=True)
        self.assertTrue(report["balanced"])
        hat = next(t for t in report["tracks"] if t["name"] == "HiHat")
        self.assertEqual(hat["status"], "TRIMMED")
        self.assertLess(hat["volume_after"], hat["volume_before"])  # too loud -> trimmed down
        self.assertLessEqual(abs(hat["delta_db"]), 6.0)  # bounded
        kick = next(t for t in report["tracks"] if t["name"] == "Kick")
        self.assertEqual(kick["status"], "OK")  # already at anchor target

    def test_vertical_session_hierarchy_verdict(self):
        from vertical_tester import analyze_vertical_stage_session

        channels = [
            {"channel": 1, "name": "Kick"},
            {"channel": 2, "name": "Bass"},
            {"channel": 3, "name": "Chords"},
            {"channel": 4, "name": "Lead"},
        ]
        mcp = StubMCP(channels, {})
        lufs = {1: -15.0, 2: -18.0, 3: -24.0, 4: -17.0}  # lead 7 dB above harmony

        def fn(ch, pattern=1):
            return {"lufs": lufs[ch], "silent": False}

        res = analyze_vertical_stage_session(mcp, "V4_LEAD", active_channels=[1, 2, 3, 4], loudness_fn=fn)
        self.assertEqual(res["verdict"], "PASS")
        self.assertTrue(res["checks"]["lead_hierarchy_ok"])
        # now make harmony too loud vs lead -> FIX
        lufs[3] = -14.0
        res2 = analyze_vertical_stage_session(mcp, "V4_LEAD", active_channels=[1, 2, 3, 4], loudness_fn=fn)
        self.assertEqual(res2["verdict"], "FIX")
        self.assertIn("lead_hierarchy_ok", res2["failed_checks"])

    def test_fast_soundcheck_dsp_mode_no_wav(self):
        import soundfile as sf
        import numpy as np
        from soundcheck_lib import audit_track_soundcheck_fast

        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "a4.wav")
            sr = 44100
            t = np.arange(int(0.6 * sr)) / sr
            sf.write(wav, 0.5 * np.sin(2 * np.pi * 440.0 * t), sr)
            res = audit_track_soundcheck_fast(
                track_spec={"name": "Lead", "role": "lead_melody", "plugin": "gl.sampler", "sample_file": wav},
                audition="dsp",
            )
        self.assertEqual(res["verdict"], "PASS")
        self.assertTrue(res["fast"])
        self.assertEqual(res["metrics"].get("detected_note"), "A")
        self.assertNotIn("wav_path", res)  # no wav rendered


class TestNaturalSoundAndGates(unittest.TestCase):
    def test_genre_gates_have_low_end_max(self):
        import render_and_audit

        for prof, gate in render_and_audit.GENRE_GATES.items():
            self.assertIn("low_end_max", gate, prof)
            self.assertGreater(gate["low_end_max"], 20.0)

    def test_nominal_balance_evens_default_faders(self):
        import song_builder
        from gain_staging_lib import apply_nominal_balance

        chans = [
            song_builder.ChannelSpec("Kick", "kick", 1, "drums", plugin="gl.kick", volume=0.76),
            song_builder.ChannelSpec("HiHat", "hihat_perc", 2, "drums", plugin="gl.hat", volume=0.52),
            song_builder.ChannelSpec("Lead", "lead_melody", 3, "music", plugin="gl.pluck", volume=0.74),
        ]
        meta = {"mix_profile": "trap_rap", "has_drums": True}
        out = apply_nominal_balance(chans, meta)
        by = {c.name: c.volume for c in out}
        self.assertAlmostEqual(by["Kick"], 0.76, places=2)  # anchor untouched
        self.assertLess(by["HiHat"], 0.52)  # hats pulled down toward their role target
        self.assertLessEqual(abs(by["Lead"] - 0.74), 0.5)




class TestPitchAndChordGuards(unittest.TestCase):
    """Guards against 'LeadBell dissonant' / 'ChordKey etrange' errors."""

    @staticmethod
    def _analyze(y, sr=44100):
        import numpy as np
        from audio_metrics import analyze_pitch_and_purity

        y2 = np.stack([y, y], axis=1).astype(np.float32)
        bands = {"sub": 0.0, "bass": 0.0, "low_mid": 0.1, "mid": 0.4, "harsh": 0.3, "air": 0.2}
        return analyze_pitch_and_purity(y2, y.astype(np.float32), sr, bands, r_harm=0.6)

    def test_bell_with_strong_inharmonic_partials_keeps_true_root(self):
        import numpy as np

        sr = 44100
        t = np.arange(int(1.2 * sr)) / sr
        bell = 0.25 * np.sin(2 * np.pi * 440 * t) + 1.0 * np.sin(2 * np.pi * 440 * 2.76 * t) + 0.7 * np.sin(2 * np.pi * 440 * 5.40 * t)
        r = self._analyze(bell, sr)
        self.assertEqual(r["midi_root"], 69)  # A4, not the 2.76x partial (D#)
        self.assertTrue(r["playable_chromatically"])  # single stable pitch bell stays playable

    def test_chord_and_dyad_are_not_playable_chromatically(self):
        import numpy as np

        sr = 44100
        t = np.arange(int(1.2 * sr)) / sr
        chord = 0.4 * np.sin(2 * np.pi * 261.63 * t) + 0.4 * np.sin(2 * np.pi * 329.63 * t) + 0.4 * np.sin(2 * np.pi * 392.0 * t)
        r = self._analyze(chord, sr)
        self.assertEqual(r["sample_nature"], "poly_chord_or_dyad")
        self.assertFalse(r["playable_chromatically"])
        dyad = 0.5 * np.sin(2 * np.pi * 261.63 * t) + 0.3 * np.sin(2 * np.pi * 523.25 * t) + 0.25 * np.sin(2 * np.pi * 466.16 * t)
        r2 = self._analyze(dyad, sr)
        self.assertFalse(r2["playable_chromatically"])

    def test_sine_oneshot_detected_and_playable(self):
        import numpy as np

        sr = 44100
        t = np.arange(int(0.8 * sr)) / sr
        r = self._analyze(0.5 * np.sin(2 * np.pi * 440 * t), sr)
        self.assertEqual(r["midi_root"], 69)
        self.assertTrue(r["playable_chromatically"])

    def test_key_fit_penalty(self):
        from audio_metrics import key_fit_penalty, key_scale_pcs

        self.assertEqual(key_scale_pcs("F minor"), {5, 7, 8, 10, 0, 1, 3})
        self.assertEqual(key_fit_penalty(65, "F minor", "lead_melody"), 0.0)  # F in scale
        self.assertEqual(key_fit_penalty(36, "C major", "kick"), 0.0)  # percussive: ignored
        self.assertLessEqual(key_fit_penalty(66, "C major", "lead_melody"), 0.03)  # 1 st away max in 12-TET


class TestPresenceFloorsAndBlueprintWiring(unittest.TestCase):
    def test_role_targets_respect_presence_floors(self):
        from gain_staging_lib import ROLE_PRESENCE_FLOOR_DB, _effective_profile_offsets

        for prof in ("trap_rap", "pop_rnb", "lofi_chill", "classical_ambient"):
            _, _, offsets = _effective_profile_offsets(prof, prof != "classical_ambient", None)
            for role, fl in ROLE_PRESENCE_FLOOR_DB.items():
                self.assertGreaterEqual(offsets[role][1], fl, f"{prof}/{role}")

    def test_high_centroid_lift_in_session_balance(self):
        from gain_staging_lib import balance_tracks_in_session

        channels = [{"channel": 1, "name": "LeadBell", "muted": False}]
        mcp = StubMCP(channels, {1: -30.0})
        tracks = [{"name": "LeadBell", "channel": 1, "pattern": 1, "role": "lead", "volume": 0.74, "centroid_hz": 5200.0}]
        rep = balance_tracks_in_session(mcp, tracks, profile="trap_rap", has_drums=True)
        t = rep["tracks"][0]
        # trap_rap lead offset is -1.0 dB (above the -3 floor) + high-centroid +2 dB
        # => target = -16 + (-1.0 + 2.0) = -15.0 LUFS
        self.assertEqual(t["target_lufs"], -15.0)
        self.assertEqual(t["status"], "TRIMMED")  # -30 measured -> big boost, bounded
        self.assertLessEqual(t["delta_db"], 6.0)

    def test_apply_palette_to_blueprint_writes_root_note(self):
        import json as _json
        from loop_first_runner import LoopFirstStateMachine

        with tempfile.TemporaryDirectory() as tmp:
            bp = {"title": "t", "tracks": [{"name": "Lead", "role": "lead_melody", "source_type": "gridline_native", "plugin": "gl.pluck", "mixer_track": 3}]}
            with open(os.path.join(tmp, "song_blueprint.json"), "w", encoding="utf-8") as fh:
                _json.dump(bp, fh)
            sm = LoopFirstStateMachine(tmp)
            palette = {"blueprint_tracks": [
                {"name": "bell_lead_d5.wav", "role": "lead_melody", "source_type": "sample", "sample_path": "samples/bell_lead_d5.wav", "mixer_track": 3, "root_note": 63, "fine_tune_ct": -8},
            ]}
            res = sm.apply_palette_to_blueprint(palette)
            self.assertEqual(res["tracks_merged"], 1)
            bp2 = _json.load(open(os.path.join(tmp, "song_blueprint.json"), encoding="utf-8"))
            tr = bp2["tracks"][0]
            self.assertEqual(tr["source_type"], "sample")
            self.assertEqual(tr["root_note"], 63)
            self.assertEqual(tr["fine_tune_ct"], -8)

    def test_schema_requires_root_note_for_sample_tracks(self):
        import jsonschema

        with open(os.path.join(ROOT, "schemas", "song_blueprint.schema.json"), encoding="utf-8") as fh:
            schema = json.load(fh)
        with open(os.path.join(ROOT, "projects", "validation_fm_dark", "song_blueprint.json"), encoding="utf-8") as fh:
            base = json.load(fh)
        bad = dict(base)
        bad["tracks"] = [{"name": "X", "role": "lead_melody", "source_type": "sample", "sample_path": "x.wav", "mixer_track": 2}]  # no root_note
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)
        jsonschema.validate(base, schema)  # example blueprint valid


class TestBlueprintMusicalSafety(unittest.TestCase):
    """Project-wide blueprint audit: root_note contract, catalog cross-check, key-fit."""

    @staticmethod
    def _bp(tracks, key="C major", mode="ionian", hc_key="C"):
        return {
            "key": key,
            "harmonic_core": {"key": hc_key, "mode": mode},
            "tracks": tracks,
        }

    def _sample_track(self, **kw):
        t = {"name": "Lead", "role": "lead", "source_type": "sample",
             "sample_path": "samples/lead.wav", "mixer_track": 3, "root_note": 60}
        t.update(kw)
        return t

    def test_key_scale_pcs_mode_aware(self):
        from blueprint_safety import blueprint_key_scale_pcs, key_str_scale_pcs

        self.assertEqual(key_str_scale_pcs("F minor"), {5, 7, 8, 10, 0, 1, 3})
        self.assertEqual(key_str_scale_pcs("C"), {0, 2, 4, 5, 7, 9, 11})
        # top-level key without quality -> harmonic_core.mode decides
        self.assertEqual(blueprint_key_scale_pcs(self._bp([], key="F", hc_key="F", mode="aeolian")), {5, 7, 8, 10, 0, 1, 3})
        self.assertEqual(blueprint_key_scale_pcs(self._bp([], key="C", hc_key="C", mode="dorian")), {0, 2, 3, 5, 7, 9, 10})
        # explicit quality in top-level key wins over mode
        self.assertEqual(blueprint_key_scale_pcs(self._bp([], key="C minor", hc_key="C", mode="dorian")), {0, 2, 3, 5, 7, 8, 10})

    def test_missing_root_note_and_path_are_errors(self):
        from blueprint_safety import blueprint_musical_safety_issues

        bp = self._bp([{"name": "ChordKey", "role": "chords", "source_type": "sample", "mixer_track": 4}])
        codes = {i["code"] for i in blueprint_musical_safety_issues(bp)}
        self.assertIn("sample_missing_path", codes)
        self.assertIn("sample_missing_root_note", codes)
        self.assertTrue(all(i["severity"] == "error" for i in blueprint_musical_safety_issues(bp)))

    def test_in_scale_sample_track_is_clean(self):
        from blueprint_safety import blueprint_musical_safety_issues

        bp = self._bp([self._sample_track(root_note=60)])  # C in C major
        self.assertEqual(blueprint_musical_safety_issues(bp), [])

    def test_percussive_and_loop_roles_exempt_from_key_fit(self):
        from blueprint_safety import blueprint_musical_safety_issues

        bp = self._bp([
            self._sample_track(name="Kick", role="kick", root_note=36, sample_path="samples/kick.wav"),
            self._sample_track(name="Loop", role="pad", root_note=61, sample_path="samples/loop.wav", is_loop=True),
        ])
        self.assertEqual(blueprint_musical_safety_issues(bp), [])

    def test_root_note_pc_mismatch_against_catalog_is_error_with_fix(self):
        import json as _json
        from blueprint_safety import blueprint_musical_safety_issues

        with tempfile.TemporaryDirectory() as tmp:
            cat = {"samples": [{
                "filepath": os.path.join(tmp, "samples", "bell_a4.wav"),
                "filename": "bell_a4.wav",
                "midi_root": 69,
                "playable_chromatically": True,
                "sample_nature": "mono_oneshot",
                "is_loop": False,
                "hard_reject": False,
            }]}
            cat_path = os.path.join(tmp, "samples_catalog.json")
            with open(cat_path, "w", encoding="utf-8") as fh:
                _json.dump(cat, fh)
            # blueprint declares root_note=60 (C4 default) but the sample is really A4 (69)
            bp = self._bp([self._sample_track(name="LeadBell", role="lead", root_note=60,
                                              sample_path="samples/bell_a4.wav")])
            issues = blueprint_musical_safety_issues(bp, catalog_path=cat_path)
            mism = [i for i in issues if i["code"] == "root_note_mismatch"]
            self.assertEqual(len(mism), 1)
            self.assertEqual(mism[0]["severity"], "error")
            self.assertEqual(mism[0]["detected_midi_root"], 69)
            self.assertEqual(mism[0]["recommended_root_note"], 69)  # strict lock: exact midi_root
            self.assertEqual(mism[0]["delta_st"], 9)
            # v3.0: even a pure octave error (+12 st) is blocking (varispeed chipmunk/smudge)
            bp2 = self._bp([self._sample_track(root_note=81, sample_path="samples/bell_a4.wav")])
            issues2 = blueprint_musical_safety_issues(bp2, catalog_path=cat_path)
            mism2 = [i for i in issues2 if i["code"] == "root_note_mismatch"]
            self.assertEqual(len(mism2), 1)
            self.assertEqual(mism2[0]["delta_st"], -12)
            # exact match -> no issue
            bp3 = self._bp([self._sample_track(root_note=69, sample_path="samples/bell_a4.wav")])
            self.assertNotIn("root_note_mismatch", {i["code"] for i in blueprint_musical_safety_issues(bp3, catalog_path=cat_path)})

    def test_non_playable_chord_sample_in_melodic_role_is_error(self):
        import json as _json
        from blueprint_safety import blueprint_musical_safety_issues

        with tempfile.TemporaryDirectory() as tmp:
            cat = {"samples": [{
                "filepath": os.path.join(tmp, "samples", "chord.wav"),
                "filename": "chord.wav",
                "midi_root": 60,
                "playable_chromatically": False,
                "sample_nature": "poly_chord_or_dyad",
                "is_loop": False,
                "hard_reject": False,
            }]}
            cat_path = os.path.join(tmp, "samples_catalog.json")
            with open(cat_path, "w", encoding="utf-8") as fh:
                _json.dump(cat, fh)
            bp = self._bp([self._sample_track(name="ChordKey", role="chords", root_note=60,
                                              sample_path="samples/chord.wav")])
            codes = {i["code"] for i in blueprint_musical_safety_issues(bp, catalog_path=cat_path)}
            self.assertIn("non_playable_sample_in_melodic_role", codes)
            # same chord sample on a percussive role is fine
            bp2 = self._bp([self._sample_track(name="Kick", role="kick", root_note=36,
                                               sample_path="samples/chord.wav")])
            self.assertNotIn("non_playable_sample_in_melodic_role",
                             {i["code"] for i in blueprint_musical_safety_issues(bp2, catalog_path=cat_path)})

    def test_summarize_and_runner_wiring(self):
        import json as _json
        from blueprint_safety import summarize_musical_safety
        from loop_first_runner import LoopFirstStateMachine

        with tempfile.TemporaryDirectory() as tmp:
            bp = self._bp([self._sample_track(name="Lead", role="lead", root_note=60)])
            with open(os.path.join(tmp, "song_blueprint.json"), "w", encoding="utf-8") as fh:
                _json.dump(bp, fh)
            sm = LoopFirstStateMachine(tmp)
            palette = {"blueprint_tracks": [
                {"name": "Lead", "role": "lead", "source_type": "sample",
                 "sample_path": "samples/lead.wav", "mixer_track": 3, "root_note": 60},
            ]}
            res = sm.apply_palette_to_blueprint(palette)
            self.assertIn("musical_safety", res)
            self.assertTrue(res["musical_safety"]["ok"])
            self.assertEqual(res["musical_safety"]["errors"], 0)
            # a dissonant root (pc mismatch vs catalog) surfaces as errors > 0
            cat = {"samples": [{"filepath": os.path.join(tmp, "samples", "lead.wav"), "filename": "lead.wav",
                                "midi_root": 63, "playable_chromatically": True, "sample_nature": "mono_oneshot",
                                "is_loop": False, "hard_reject": False}]}
            with open(os.path.join(tmp, "samples_catalog.json"), "w", encoding="utf-8") as fh:
                _json.dump(cat, fh)
            summ = summarize_musical_safety(bp, catalog_path=os.path.join(tmp, "samples_catalog.json"))
            self.assertFalse(summ["ok"])
            self.assertGreater(summ["errors"], 0)
            self.assertIn("root_note_mismatch", {i["code"] for i in summ["issues"]})

    def test_load_blueprint_exposes_musical_safety(self):
        from song_builder import load_blueprint

        bp_path = os.path.join(ROOT, "projects", "validation_fm_dark", "song_blueprint.json")
        _bp, meta = load_blueprint(bp_path, validate=True)
        self.assertIn("musical_safety", meta)
        self.assertTrue(meta["musical_safety"]["checked"])


if __name__ == "__main__":
    unittest.main()
