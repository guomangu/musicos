#!/usr/bin/env python3
"""v3.0 regression tests — Axe 3: mix coherence (single source config/mix_profiles.json,
DAG analytic solver, notch caps, stem matrix, tuning invariants)."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "mcp-servers", "audio-analyzer"), os.path.join(ROOT, "mcp-servers", "audio-index")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np


class TestSingleSourceOfTruth(unittest.TestCase):
    def test_config_profiles_match_genre_gates(self):
        import mix_config
        from render_and_audit import GENRE_GATES
        cfg_profiles = set(mix_config.profile_names())
        self.assertEqual(cfg_profiles, set(GENRE_GATES.keys()))
        # spot-check: trap_rap low_end_max identical in both
        self.assertEqual(mix_config.profile_settings("trap_rap")["low_end_max_pct"], GENRE_GATES["trap_rap"]["low_end_max"])
        self.assertEqual(mix_config.profile_settings("trap_rap")["target_lufs"], GENRE_GATES["trap_rap"]["default_lufs"])

    def test_global_invariants(self):
        import mix_config
        self.assertEqual(mix_config.get_invariant("master_limiter_ceiling_dbtp"), -1.2)
        self.assertEqual(mix_config.get_invariant("master_limiter_max_gain_db"), 5.5)
        self.assertEqual(mix_config.notch_caps(), (2, 3))
        self.assertEqual(mix_config.source_lpf_floor_hz("chords"), 6500.0)
        self.assertEqual(mix_config.source_lpf_floor_hz("lead"), 9000.0)
        t = mix_config.tuning_invariants()
        self.assertEqual(t["foundation_max_ct"], 6.0)
        self.assertEqual(t["expressive_peak_ct"], 12.0)

    def test_limiter_defaults_resolve_from_config(self):
        import mix_config
        self.assertEqual(mix_config.resolve_limiter_params(None, None), (-1.2, 5.5))
        # explicit legacy values are respected
        self.assertEqual(mix_config.resolve_limiter_params(-2.5, 9.6), (-2.5, 9.6))

    def test_source_lpf_floors_applied(self):
        from gl_dsl import _default_cutoff_hz
        self.assertGreaterEqual(_default_cutoff_hz("chords", -0.8), 6500.0)
        self.assertGreaterEqual(_default_cutoff_hz("lead", -0.8), 9000.0)
        self.assertGreaterEqual(_default_cutoff_hz("pad", 0.5), 6500.0)


class TestAnalyticGainSolver(unittest.TestCase):
    def test_trims_toward_targets_with_caps_and_floors(self):
        from gain_staging_lib import solve_analytic_gain_trims
        measured = {"kick": -20.0, "bass": -26.0, "lead": -24.0, "chords": -25.0}
        offsets = {"kick": 0.0, "bass": -2.0, "lead": -1.0, "chords": -4.0}
        floors = {"lead": -3.0, "chords": -10.0}
        res = solve_analytic_gain_trims(measured, offsets, anchor_lufs=-16.0,
                                        presence_floors_db=floors)
        for role, d in res["trims_db"].items():
            self.assertLessEqual(abs(d), 6.0)
        # lead measured -24, target -17 -> +7 clamped to +6, but floor -3 -> -24+6=-18 >= -16-3=-19 ok
        self.assertEqual(res["trims_db"]["lead"], 6.0)
        # bass: target -18, measured -26 -> +8 clamped to +6
        self.assertEqual(res["trims_db"]["bass"], 6.0)
        # kick: target -16, measured -20 -> +4.0
        self.assertEqual(res["trims_db"]["kick"], 4.0)

    def test_low_end_constraint_pulls_bass_down(self):
        from gain_staging_lib import solve_analytic_gain_trims
        res = solve_analytic_gain_trims(
            measured_lufs={"kick": -18.0, "bass": -17.0, "lead": -22.0},
            target_offsets_db={"kick": 0.0, "bass": -2.0, "lead": -1.0},
            anchor_lufs=-16.0,
            low_end_share_pct=52.0,
            low_end_max_pct=42.0,
            low_share_weight_by_role={"kick": 0.30, "bass": 0.25, "lead": 0.0},
        )
        # bass must have been pulled below its target trim
        self.assertLess(res["trims_db"]["bass"], 1.0)
        self.assertLessEqual(res["low_end_share_after_pct"], 52.0)


class TestStemMatrixEvaluator(unittest.TestCase):
    def test_synthetic_stems_single_render(self):
        from stem_matrix_evaluator import evaluate_stem_matrix
        sr = 44100
        t = np.arange(int(2.0 * sr)) / sr
        rng = np.random.default_rng(7)
        stems = {
            "stem_1_kick": 0.8 * np.exp(-t / 0.02) * np.sin(2 * np.pi * 55 * t),
            "stem_2_bass": 0.5 * np.sin(2 * np.pi * 55 * t),
            "stem_3_chords": 0.3 * (np.sin(2 * np.pi * 261.6 * t) + np.sin(2 * np.pi * 329.6 * t)),
            "stem_4_lead": 0.3 * np.sin(2 * np.pi * 523.25 * t),
        }
        stems = {k: np.stack([v, v], axis=1) for k, v in stems.items()}
        rep = evaluate_stem_matrix(stems, sr, profile="trap_rap")
        self.assertEqual(rep["gridline_renders_used"], 1)
        self.assertEqual(rep["n_stems"], 4)
        for stage in ("V1_low_end", "V2_groove", "V3_harmony", "V4_lead"):
            self.assertTrue(rep["vertical_stages"][stage]["present"])
        self.assertTrue(all(rep["per_stem"][n]["audible"] for n in stems))
        self.assertTrue(rep["ok"])

    def test_silent_stem_is_error(self):
        from stem_matrix_evaluator import evaluate_stem_matrix
        sr = 44100
        stems = {"stem_1_kick": np.zeros((sr, 2)), "stem_2_lead": np.zeros((sr, 2)) + 1e-9 * np.random.default_rng(1).standard_normal((sr, 2))}
        rep = evaluate_stem_matrix(stems, sr)
        self.assertFalse(rep["ok"])
        self.assertIn("stem_silent", {i["code"] for i in rep["issues"]})


class TestCanonicalPipelineOrder(unittest.TestCase):
    def test_compile_song_pipeline_ends_with_sanitize(self):
        from compile_song import CANONICAL_MIDI_PIPELINE
        self.assertTrue(CANONICAL_MIDI_PIPELINE[-1].startswith("7.sanitize_note_overlaps"))
        self.assertTrue(any("LowIntervalLimit" in s for s in CANONICAL_MIDI_PIPELINE))
        self.assertTrue(any("contour_preserving" in s for s in CANONICAL_MIDI_PIPELINE))


if __name__ == "__main__":
    unittest.main()
