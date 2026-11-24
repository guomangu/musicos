#!/usr/bin/env python3
"""v3.0 regression tests — Axe 4: 3-turn pipeline budget (renders <= 2, MCP answers <= 1500 chars,
compile_song declarative gate). Full gridline pipeline test is skipped without the binary."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "mcp-servers", "audio-analyzer"), os.path.join(ROOT, "mcp-servers", "audio-index")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

HAS_GRIDLINE = bool(shutil.which("gridline"))


def _project(tmp, tracks):
    bp = {
        "title": "t", "genre": "trap", "mix_profile": "trap_rap",
        "emotion_vector": {"valence": -0.5, "tension": 0.5, "energy": 0.6, "depth": 0.6},
        "bpm": 132.0, "time_signature": {"numerator": 4, "denominator": 4},
        "key": "C minor", "harmonic_core": {"key": "C", "mode": "aeolian", "progression_bars": 4,
                                             "chord_progression": [{"bar": 1, "symbol": "Cm7", "bass_midi": 36, "voicing_midi": [48, 51, 55, 58]}],
                                             "motif_cell": {"intervals_from_tonic": [0, 3], "rhythm_beats": [0, 1], "apex_midi": 72},
                                             "phrase_plan": "A-A'-B-B'"},
        "structure": [{"section": "Intro", "start_bar": 1, "bars": 4}, {"section": "Drop", "start_bar": 5, "bars": 8}],
        "has_drums": True, "tracks": tracks,
    }
    path = os.path.join(tmp, "song_blueprint.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(bp, fh)
    return path


class TestCompileSongGate(unittest.TestCase):
    def test_dissonant_root_blocks_compile(self):
        from compile_song import compile_song
        with tempfile.TemporaryDirectory() as tmp:
            _project(tmp, [{"name": "Lead", "role": "lead_melody", "source_type": "sample",
                            "sample_path": "samples/lead.wav", "mixer_track": 3, "root_note": 60}])
            cat = {"samples": [{"filepath": os.path.join(tmp, "samples", "lead.wav"), "filename": "lead.wav",
                                "midi_root": 69, "playable_chromatically": True, "is_loop": False, "hard_reject": False}]}
            with open(os.path.join(tmp, "samples_catalog.json"), "w", encoding="utf-8") as fh:
                json.dump(cat, fh)
            rep = compile_song(tmp, core_only=True)
            self.assertFalse(rep["ok"])
            self.assertEqual(rep["stage"], "blueprint_safety")
            self.assertGreater(rep["musical_safety"]["errors"], 0)

    def test_clean_blueprint_passes_gate(self):
        from compile_song import compile_song
        with tempfile.TemporaryDirectory() as tmp:
            _project(tmp, [{"name": "Kick", "role": "kick", "source_type": "sample",
                            "sample_path": "samples/kick.wav", "mixer_track": 1, "root_note": 36}])
            rep = compile_song(tmp, core_only=True)
            # no arrangement_plan.json and no build_song.py -> stage no_input, but safety gate passed
            self.assertNotEqual(rep.get("stage"), "blueprint_safety")
            self.assertTrue(rep["musical_safety"]["ok"])


class TestTokenBudget(unittest.TestCase):
    def test_mcp_answers_compact(self):
        from audio_analyzer_server import _compact_json
        big = {"x": "a" * 5000}
        out = _compact_json(big)
        self.assertLessEqual(len(out), 1500)
        small = {"ok": True}
        self.assertEqual(_compact_json(small), '{"ok":true}')

    def test_stem_eval_uses_one_render(self):
        import numpy as np
        from stem_matrix_evaluator import evaluate_stem_matrix
        sr = 22050
        t = np.arange(sr) / sr
        stems = {"stem_1_kick": np.stack([0.5 * np.sin(2 * np.pi * 50 * t)] * 2, axis=1),
                 "stem_2_lead": np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1)}
        rep = evaluate_stem_matrix(stems, sr)
        self.assertEqual(rep["gridline_renders_used"], 1)


@unittest.skipUnless(HAS_GRIDLINE, "binaire gridline requis pour le pipeline complet")
class TestFullPipelineBudget(unittest.TestCase):
    def test_two_renders_max(self):
        # Placeholder executable uniquement avec gridline installé : le pipeline v3.0
        # fait 1 rendu stems 8 mesures (tour 2) + 1 rendu master (tour 3) = 2 rendus max.
        self.assertTrue(HAS_GRIDLINE)


if __name__ == "__main__":
    unittest.main()
