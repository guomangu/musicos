#!/usr/bin/env python3
"""v3.3 J2 (#5) — déterminisme strict par `seed` : 2 compilations avec le même seed
produisent un ScoreIR/blueprint strictement identiques (hash égaux) ; un seed différent
change le jitter de groove. `derive_seed` est stable (SHA-256 title|key|bpm)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "mcp-servers", "audio-analyzer"), os.path.join(ROOT, "mcp-servers", "audio-index")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from determinism import derive_seed, resolve_seed, seed_all
from score_ir import build_score_ir, apply_groove_to_score_ir


def _meta(bars: int = 16) -> dict:
    return {"bpm": 132.0, "beats_per_bar": 4, "denominator": 4, "valence": -0.5,
            "total_bars": bars, "tonic": "C", "key": "C"}


def _dense_score() -> object:
    hc = {"key": "C", "mode": "aeolian", "degree_progression": ["i", "VI", "III", "VII"]}
    # doubles croches denses -> le jitter anti-machine-gun (seed) s'applique
    hats = [{"start": b * 0.25, "length": 0.2, "pitch": 42, "velocity": 80} for b in range(64 * 4)]
    return build_score_ir(hc, _meta(), {"hihat": hats})


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class TestDeriveSeed(unittest.TestCase):
    def test_stable_and_sensitive(self):
        s1 = derive_seed("Morceau", "C minor", 132.0)
        s2 = derive_seed("Morceau", "C minor", 132.0)
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, derive_seed("Autre", "C minor", 132.0))
        self.assertNotEqual(s1, derive_seed("Morceau", "C minor", 140.0))
        self.assertTrue(0 <= s1 < 2**31)

    def test_resolve_seed_prefers_explicit(self):
        self.assertEqual(resolve_seed({"seed": 777, "title": "x", "key": "C", "bpm": 120}), 777)
        bp = {"title": "x", "key": "C", "bpm": 120}
        self.assertEqual(resolve_seed(bp), derive_seed("x", "C", 120.0))

    def test_seed_all_returns_dedicated_rngs(self):
        r1, g1 = seed_all(42)
        r2, g2 = seed_all(42)
        self.assertEqual([r1.random() for _ in range(3)], [r2.random() for _ in range(3)])
        self.assertTrue((g1.random(3) == g2.random(3)).all())


class TestGrooveDeterminism(unittest.TestCase):
    def test_same_seed_same_scoreir(self):
        s = _dense_score()
        import copy
        s2 = copy.deepcopy(s)
        apply_groove_to_score_ir(s, swing=0.08, bpm=132.0, profile="trap_rap", seed=123)
        apply_groove_to_score_ir(s2, swing=0.08, bpm=132.0, profile="trap_rap", seed=123)
        self.assertEqual(_hash(s.to_dict() if hasattr(s, "to_dict") else s.__dict__),
                         _hash(s2.to_dict() if hasattr(s2, "to_dict") else s2.__dict__))

    def test_different_seed_different_jitter(self):
        s = _dense_score()
        import copy
        s2 = copy.deepcopy(s)
        apply_groove_to_score_ir(s, swing=0.08, bpm=132.0, profile="trap_rap", seed=1)
        apply_groove_to_score_ir(s2, swing=0.08, bpm=132.0, profile="trap_rap", seed=999)
        h1 = _hash(s.to_dict() if hasattr(s, "to_dict") else s.__dict__)
        h2 = _hash(s2.to_dict() if hasattr(s2, "to_dict") else s2.__dict__)
        self.assertNotEqual(h1, h2)


class TestCompileDeterminism(unittest.TestCase):
    def _bp(self) -> dict:
        return {
            "title": "Det", "genre": "trap", "mix_profile": "trap_rap",
            "emotion_vector": {"valence": -0.5, "tension": 0.5, "energy": 0.6, "depth": 0.6},
            "bpm": 132.0, "time_signature": {"numerator": 4, "denominator": 4},
            "key": "C minor",
            "harmonic_core": {"key": "C", "mode": "aeolian", "progression_bars": 4,
                              "chord_progression": [{"bar": 1, "symbol": "Cm7", "bass_midi": 36, "voicing_midi": [48, 51, 55, 58]}],
                              "motif_cell": {"intervals_from_tonic": [0, 3], "rhythm_beats": [0, 1], "apex_midi": 72},
                              "phrase_plan": "A-A'-B-B'"},
            "structure": [{"section": "Intro", "start_bar": 1, "bars": 4}, {"section": "Drop", "start_bar": 5, "bars": 8}],
            "has_drums": True,
            "tracks": [{"name": "Kick", "role": "kick", "source_type": "gridline_native", "plugin": "gl.kick", "mixer_track": 1}],
        }

    def test_compile_twice_same_seed_identical_enrichment(self):
        from compile_song import compile_song
        hashes = []
        seeds = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                with open(os.path.join(tmp, "song_blueprint.json"), "w", encoding="utf-8") as fh:
                    json.dump(self._bp(), fh)
                rep = compile_song(tmp, core_only=True)
                seeds.append(rep.get("seed"))
                with open(os.path.join(tmp, "song_blueprint.json"), encoding="utf-8") as fh:
                    hashes.append(_hash(json.load(fh)))
        self.assertEqual(seeds[0], seeds[1])
        self.assertEqual(hashes[0], hashes[1])  # enricher 100 % déterministe

    def test_explicit_seed_overrides_derived(self):
        from compile_song import compile_song
        bp = self._bp()
        bp["seed"] = 424242
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "song_blueprint.json"), "w", encoding="utf-8") as fh:
                json.dump(bp, fh)
            rep = compile_song(tmp, core_only=True)
            self.assertEqual(rep.get("seed"), 424242)


if __name__ == "__main__":
    unittest.main()
