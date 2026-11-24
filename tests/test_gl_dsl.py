#!/usr/bin/env python3
"""Unit and headless integration tests for `scripts/gl_dsl.py` and `scripts/song_builder.py`."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
import warnings

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import gl_dsl  # noqa: E402
import song_builder  # noqa: E402
from studio_lib import GridlineMCP  # noqa: E402


class TestGlDslEffects(unittest.TestCase):
    def test_eq_band_kinds_and_calibrated_widths(self):
        expected_types = {
            "hpf": 1,
            "bandpass": 2,
            "lpf": 3,
            "notch": 4,
            "low_shelf": 5,
            "peak": 6,
            "high_shelf": 7,
        }
        for kind, tid in expected_types.items():
            d = gl_dsl.eq_band(2, kind, 1000.0, gain_db=-3.5, q=1.4)
            self.assertEqual(d["Band 2 type"], tid)
            self.assertEqual(d["Band 2 freq"], 1000.0)
            for k in d:
                self.assertNotIn("Q (legacy)", k)
            if tid in (5, 6, 7):
                self.assertEqual(d["Band 2 gain"], -3.5)
            else:
                self.assertNotIn("Band 2 gain", d)
            if tid in (2, 6):
                self.assertAlmostEqual(d["Band 2 width"], 0.39, places=4)
            else:
                self.assertNotIn("Band 2 width", d)

        # Calibrated anchor points
        self.assertAlmostEqual(gl_dsl.q_to_width(1.2), 0.45, places=4)
        self.assertAlmostEqual(gl_dsl.q_to_width(1.4), 0.39, places=4)
        self.assertAlmostEqual(gl_dsl.q_to_width(1.6), 0.35, places=4)
        self.assertAlmostEqual(gl_dsl.q_to_width(2.8), 0.22, places=4)

        # Monotonicity and bounds
        qs = [0.2, 0.5, 0.8, 1.2, 1.4, 1.6, 2.8, 5.0, 10.0, 25.0]
        ws = [gl_dsl.q_to_width(q) for q in qs]
        for w in ws:
            self.assertGreaterEqual(w, 0.05)
            self.assertLessEqual(w, 0.95)
        for a, b in zip(ws[:-1], ws[1:]):
            self.assertGreaterEqual(a, b)

    def test_eq_band_validation(self):
        with self.assertRaises(ValueError):
            gl_dsl.eq_band(0, "peak", 1000)
        with self.assertRaises(ValueError):
            gl_dsl.eq_band(8, "peak", 1000)
        with self.assertRaises(ValueError):
            gl_dsl.eq_band(1, "bad_kind", 1000)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            gl_dsl.q_to_width(0.0)

    def test_reverb_room_units_and_bounds(self):
        rv = gl_dsl.reverb_room(decay_sec=0.4, hpf_hz=400.0, lpf_hz=6800.0, wet_pct=40.0, er_pct=12.0)
        self.assertEqual(rv["Model"], 1)
        self.assertEqual(rv["Dry"], 0)
        self.assertEqual(rv["Dry (full)"], 0)
        self.assertEqual(rv["Decay (full)"], 4)
        self.assertEqual(rv["Decay time"], 0.4)
        self.assertEqual(rv["High cut (full)"], 68)
        self.assertEqual(rv["Wet (full)"], 51)
        self.assertEqual(rv["Early reflections (full)"], 15)

        rv_hi = gl_dsl.reverb_room(decay_sec=99.0, lpf_hz=50000.0, wet_pct=200.0, er_pct=-5.0)
        self.assertEqual(rv_hi["Decay (full)"], 200)
        self.assertEqual(rv_hi["High cut (full)"], 221)
        self.assertEqual(rv_hi["Wet (full)"], 160)
        self.assertEqual(rv_hi["Early reflections (full)"], 0)

        rv_lo = gl_dsl.reverb_room(decay_sec=0.01, lpf_hz=100.0)
        self.assertEqual(rv_lo["Decay (full)"], 1)
        self.assertEqual(rv_lo["High cut (full)"], 5)

    def test_limiter_master_and_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lim = gl_dsl.limiter_master(gain_db=6.0, ceiling_db=-2.2)
            self.assertEqual(len(w), 0)
        self.assertEqual(lim["Attack (look-ahead)"], 0.002)
        self.assertEqual(lim["0.2 dB safety"], 1)
        self.assertEqual(lim["Ceiling"], -2.2)
        self.assertEqual(lim["Gain"], 6.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lim_hot = gl_dsl.limiter_master(gain_db=2.0, ceiling_db=-1.2)
            self.assertGreaterEqual(len(w), 1)
            self.assertEqual(lim_hot["Ceiling"], -1.2)

    def test_sidechain_and_automation_scaling(self):
        sc = gl_dsl.sidechain_send(1, 18, 0.72)
        self.assertTrue(sc["sidechain"])
        self.assertTrue(sc["preFader"])
        self.assertEqual(sc["mixer"], 1)
        self.assertEqual(sc["to"], 18)
        self.assertEqual(sc["level"], 0.72)

        self.assertAlmostEqual(gl_dsl.mixer_gain_to_auto(0.0), 0.0)
        self.assertAlmostEqual(gl_dsl.mixer_gain_to_auto(0.80), 0.64)
        self.assertAlmostEqual(gl_dsl.mixer_gain_to_auto(1.0), 0.80)
        self.assertAlmostEqual(gl_dsl.mixer_gain_to_auto(1.25), 1.0)
        self.assertAlmostEqual(gl_dsl.mixer_gain_to_auto(2.0), 1.0)

        self.assertAlmostEqual(gl_dsl.db_to_auto(0.0), 0.80)
        self.assertLess(gl_dsl.db_to_auto(-2.5), gl_dsl.db_to_auto(0.0))
        self.assertGreater(gl_dsl.db_to_auto(1.0), gl_dsl.db_to_auto(0.0))
        self.assertEqual(gl_dsl.db_to_auto(-96.0), 0.0)
        self.assertLessEqual(gl_dsl.db_to_auto(20.0), 1.0)


class TestGlDslSynths(unittest.TestCase):
    def test_all_registered_synths_adsr_and_cpu_bounds(self):
        all_plugins = [
            "gl.analog", "gl.wavetable", "gl.warmsynth", "gl.dcosynth",
            "gl.twinva", "gl.sawswarm", "gl.string", "gl.pluck",
            "gl.fm", "gl.matrixfm", "gl.keys", "gl.fingerbass", "gl.sampler",
        ]
        for pl in all_plugins:
            v = gl_dsl.synth_voice(pl, "pad", valence=-0.5, polyphony=32, unison=8)
            # v3.0 anti brick-wall: source LPF floor 6.5 kHz for pads/chords (config/mix_profiles.json)
            self.assertLessEqual(v.cutoff_hz, 9000.0)
            self.assertGreaterEqual(v.cutoff_hz, 6500.0)
            self.assertLessEqual(v.unison, 2)
            expected_poly_cap = 16 if pl == "gl.keys" else 12
            self.assertLessEqual(v.polyphony, expected_poly_cap)

        # Specific plugin unit checks
        v_ana = gl_dsl.synth_voice("gl.analog", "pad", attack_ms=250.0, release_ms=600.0, valence=0.0)
        self.assertAlmostEqual(v_ana.channel_params["Amp attack"], 0.25, places=3)
        self.assertAlmostEqual(v_ana.channel_params["Amp release"], 0.60, places=3)
        self.assertGreaterEqual(v_ana.channel_params["Cutoff"], 6500.0)
        self.assertLessEqual(v_ana.channel_params["Cutoff"], 9000.0)
        self.assertFalse(v_ana.needs_insert_lpf)

        v_warm = gl_dsl.synth_voice("gl.warmsynth", "chords", valence=-0.8)
        self.assertEqual(v_warm.channel_params["Filter"], 1)
        self.assertGreater(v_warm.channel_params["Cutoff"], 0.0)
        self.assertLess(v_warm.channel_params["Cutoff"], 1.0)
        self.assertGreater(v_warm.channel_params["Release"], 0.0)

        v_dco = gl_dsl.synth_voice("gl.dcosynth", "lead", valence=0.5)
        self.assertEqual(v_dco.channel_params["VCA"], 0)
        # v3.0: lead floor 9 kHz
        self.assertGreaterEqual(v_dco.cutoff_hz, 9000.0)
        self.assertLessEqual(v_dco.cutoff_hz, 12500.0)

        v_saw = gl_dsl.synth_voice("gl.sawswarm", "chords", polyphony=24, unison=4)
        self.assertLessEqual(v_saw.channel_params["Polyphony"], 2)  # enum index <= 2 ('12')
        self.assertLessEqual(v_saw.channel_params["Unison"], 1)     # enum index <= 1 ('2 voices')

        for no_lpf_pl in ("gl.fm", "gl.matrixfm", "gl.keys"):
            v_nl = gl_dsl.synth_voice(no_lpf_pl, "bell", valence=-0.4)
            self.assertTrue(v_nl.needs_insert_lpf)
            tuples = v_nl.to_insert_eq_tuples(mixer=6, slot=1)
            tdict = dict(tuples)
            self.assertEqual(tdict["mixer/6/slot/1/Quality"], 0)
            self.assertEqual(tdict["mixer/6/slot/1/Band 7 type"], 3)
            self.assertEqual(tdict["mixer/6/slot/1/Band 7 freq"], v_nl.cutoff_hz)


class TestSongBuilderCompatAndIntegration(unittest.TestCase):
    def test_song_builder_legacy_eq_helpers_preserved(self):
        pocket = song_builder.eq_music_pocket(3)
        d_pocket = dict(pocket)
        self.assertEqual(d_pocket["mixer/3/slot/1/Band 1 type"], 1)
        self.assertEqual(d_pocket["mixer/3/slot/1/Band 1 freq"], 175.0)
        self.assertEqual(d_pocket["mixer/3/slot/1/Band 2 type"], 6)
        self.assertEqual(d_pocket["mixer/3/slot/1/Band 2 width"], 0.45)

        kick_eq = song_builder.eq_kick_clean(1)
        d_kick = dict(kick_eq)
        self.assertEqual(d_kick["mixer/1/slot/1/Band 2 type"], 6)
        self.assertEqual(d_kick["mixer/1/slot/1/Band 2 gain"], -9.5)

        ins = song_builder.eq_insert(4, [(1, 1, 250), (4, 6, 1500, -2.5, 0.45), (7, 3, 5500)])
        d_ins = dict(ins)
        self.assertEqual(d_ins["mixer/4/slot/1/Band 4 type"], 6)
        self.assertEqual(d_ins["mixer/4/slot/1/Band 4 width"], 0.45)

        # Dual-mode eq_band works in both semantic and legacy signatures
        sem = song_builder.eq_band(2, "peak", 3200, -3.0, 1.4)
        self.assertIsInstance(sem, dict)
        self.assertEqual(sem["Band 2 type"], 6)
        leg = song_builder.eq_band(4, 2, 6, 1500, -2.5, 0.45)
        self.assertIsInstance(leg, list)
        self.assertEqual(dict(leg)["mixer/4/slot/1/Band 2 width"], 0.45)

    @unittest.skipUnless(shutil.which("gridline"), "binaire gridline requis")
    def test_headless_build_with_synth_voice_auto_insert_eq(self):
        with tempfile.TemporaryDirectory() as td:
            proj = os.path.join(td, "test_dsl.gridline")
            meta = {"bpm": 120.0, "beats_per_bar": 4, "denominator": 4, "mix_profile": "pop_rnb", "valence": -0.4}
            sections = [("Main", 0, 2, "main")]
            channels = [
                song_builder.ChannelSpec("Pad", "pad", 1, "music", plugin="gl.warmsynth", role="pad", valence=-0.4),
                song_builder.ChannelSpec("Bell", "bell", 2, "music", plugin="gl.fm", role="bell", valence=-0.4),
            ]
            tracks = {
                "pad": [{"start": 0.0, "length": 3.5, "pitch": 60, "velocity": 85}],
                "bell": [{"start": 0.0, "length": 1.0, "pitch": 67, "velocity": 90}],
            }
            song_builder.build_gridline_project(
                proj_path=proj,
                export_dir=os.path.join(td, "exports"),
                samples_dir=td,
                meta=meta,
                sections=sections,
                channels=channels,
                tracks=tracks,
                markers=[{"name": "Main", "position": 0.0}],
                min_pattern_lufs=-40.0,
            )
            self.assertTrue(os.path.isfile(proj))
            # Verify gl.eq LPF was automatically attached to mixer 2 (gl.fm)
            mcp = GridlineMCP(proj, td)
            try:
                mix = mcp.call("get_mixer", {})
                m2 = [t for t in mix["tracks"] if t["mixer"] == 2][0]
                effects = [s.get("plugin") for s in m2.get("effects", [])]
                self.assertIn("gl.eq", effects)
                pars = {p["name"]: p["value"] for p in mcp.call("get_parameters", {"mixer": 2, "slot": 1, "limit": 200})["params"]}
                self.assertEqual(pars["Quality"], 0)
                self.assertEqual(pars["Band 7 type"], 3)
            finally:
                mcp.close()


if __name__ == "__main__":
    unittest.main()
