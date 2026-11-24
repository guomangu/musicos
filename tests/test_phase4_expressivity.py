#!/usr/bin/env python3
"""Unit tests for Phase 4 expressivity helpers (`groove_lib`, `transitions_lib`) and song_builder integration."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import song_builder as sb
from gl_dsl import db_to_auto
from groove_lib import (
    apply_anti_machine_gun,
    apply_haas_pocket,
    apply_micro_strum,
    enforce_frequency_inertia,
    ms_to_beats,
)
from transitions_lib import beat4_dropout, filter_sweep, outro_fade, section_gain_arc


class TestPhase4Expressivity(unittest.TestCase):
    def test_reexports_from_song_builder(self) -> None:
        for attr in (
            "apply_micro_strum",
            "apply_haas_pocket",
            "apply_anti_machine_gun",
            "enforce_frequency_inertia",
            "beat4_dropout",
            "filter_sweep",
            "section_gain_arc",
            "outro_fade",
        ):
            self.assertTrue(hasattr(sb, attr), f"Missing re-export {attr} in song_builder")

    def test_micro_strum_staggers_and_softens_inner_voices(self) -> None:
        chord = [
            {"start": 4.0, "length": 2.0, "pitch": 48, "velocity": 80},
            {"start": 4.0, "length": 2.0, "pitch": 55, "velocity": 80},
            {"start": 4.0, "length": 2.0, "pitch": 60, "velocity": 80},
        ]
        out = apply_micro_strum(chord, strum_ms=20.0, bpm=120.0, inner_vel_drop=10)
        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(out[0]["start"], 4.0, places=5)
        expected_spread = ms_to_beats(20.0, 120.0)
        self.assertAlmostEqual(out[-1]["start"] - out[0]["start"], expected_spread, places=4)
        self.assertEqual(out[0]["velocity"], 80)
        self.assertEqual(out[1]["velocity"], 70)
        self.assertEqual(out[2]["velocity"], 80)

    def test_haas_pocket_and_anti_machine_gun_and_sub_inertia(self) -> None:
        tracks = {
            "kick": [{"start": 0.0, "length": 0.5, "pitch": 60, "velocity": 110}],
            "hihat": [
                {"start": i * 0.25, "length": 0.2, "pitch": 60, "velocity": 90}
                for i in range(8)
            ],
            "808": [
                {"start": 0.0, "length": 0.125, "pitch": 36, "velocity": 100},
                {"start": 0.125, "length": 0.125, "pitch": 36, "velocity": 100},
            ],
        }
        pocketed = apply_haas_pocket(tracks, bpm=120.0, profile="trap_rap")
        self.assertEqual(pocketed["kick"][0]["start"], 0.0)
        self.assertGreater(pocketed["hihat"][0]["start"], 0.0)
        self.assertGreater(pocketed["808"][0]["start"], pocketed["hihat"][0]["start"])

        hats_hum = apply_anti_machine_gun(tracks["hihat"], subdivision_thresh_beats=0.25, vel_jitter_pct=5.0, seed=7)
        vels = [n["velocity"] for n in hats_hum]
        self.assertGreater(len(set(vels)), 1)
        self.assertGreater(hats_hum[0]["velocity"], hats_hum[1]["velocity"])
        # Reproducible with same seed
        hats_hum_2 = apply_anti_machine_gun(tracks["hihat"], subdivision_thresh_beats=0.25, vel_jitter_pct=5.0, seed=7)
        self.assertEqual(hats_hum, hats_hum_2)

        sub_clean = enforce_frequency_inertia(tracks["808"], bpm=120.0, max_sub_rate_beats=0.5)
        self.assertEqual(len(sub_clean), 1)
        self.assertGreaterEqual(sub_clean[0]["length"], 0.5)

    def test_transitions_primitives(self) -> None:
        tracks = {
            "kick": [
                {"start": 12.0, "length": 0.5, "pitch": 60, "velocity": 110},
                {"start": 15.0, "length": 0.5, "pitch": 60, "velocity": 110},
                {"start": 16.0, "length": 0.5, "pitch": 60, "velocity": 110},
            ],
            "bass": [
                {"start": 12.0, "length": 4.0, "pitch": 36, "velocity": 100},
                {"start": 16.0, "length": 4.0, "pitch": 36, "velocity": 100},
            ],
            "lead": [
                {"start": 15.0, "length": 1.0, "pitch": 67, "velocity": 95},
            ],
        }
        autos: list[dict] = []
        section_gain_arc(
            autos,
            bus_mixer_idx=19,
            sections_plan=[
                ("Intro", 1, 2),
                ("Couplet", 3, 4),
                ("Climax", 5, 6),
            ],
            beats_per_bar=4.0,
            track_idx=9,
        )
        self.assertEqual(len(autos), 1)
        vals = [p["value"] for p in autos[0]["points"]]
        self.assertAlmostEqual(vals[0], db_to_auto(-2.5), places=3)
        self.assertAlmostEqual(vals[2], db_to_auto(0.0), places=3)
        self.assertAlmostEqual(vals[-1], db_to_auto(1.0), places=3)

        rep = beat4_dropout(tracks, autos, bar_idx=3, beats_per_bar=4, fx_room_mixer=20)
        self.assertEqual(rep["removed_notes"], 1)  # kick at beat 15.0 removed
        self.assertEqual(rep["clipped_notes"], 1)  # bass starting at 12.0 clipped to 15.0
        self.assertEqual([n["start"] for n in tracks["kick"]], [12.0, 16.0])
        self.assertAlmostEqual(tracks["bass"][0]["length"], 3.0)
        self.assertEqual(len(tracks["lead"]), 1)  # lead untouched!

        filter_sweep(autos, mixer_idx=19, slot_idx=2, start_beat=8.0, end_beat=16.0, start_hz=450.0, end_hz=12000.0, curve="exp")
        outro_fade(autos, master_or_bus_idx=19, start_beat=16.0, end_beat=24.0)
        self.assertEqual(autos[0]["points"][-1]["value"], 0.0)
        self.assertGreaterEqual(len(autos), 3)


if __name__ == "__main__":
    unittest.main()
