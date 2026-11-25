#!/usr/bin/env python3
"""v3.3 J5 (#5) — scénarios C/D dans `studio_prepare_session`, macro-outil
`studio_refine_track` (Tour 3b, delta chirurgical + rollback transactionnel + preview 24 s +
feedback_tags dans la mémoire de goût), nettoyage des temporaires, PRAGMA user_version,
rôle `remix_stem` dans le ScoreIR."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "mcp-servers", "audio-analyzer"), os.path.join(ROOT, "mcp-servers", "audio-index")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np


def _blueprint() -> dict:
    return {
        "title": "RefineTest", "genre": "trap", "mix_profile": "trap_rap", "seed": 42,
        "emotion_vector": {"valence": -0.5, "tension": 0.5, "energy": 0.6, "depth": 0.6},
        "bpm": 140.0, "time_signature": {"numerator": 4, "denominator": 4},
        "key": "C minor",
        "harmonic_core": {"key": "C", "mode": "aeolian", "progression_bars": 4,
                          "chord_progression": [{"bar": 1, "symbol": "Cm7", "bass_midi": 36, "voicing_midi": [48, 51, 55, 58]}],
                          "motif_cell": {"intervals_from_tonic": [0, 3], "rhythm_beats": [0, 1], "apex_midi": 72},
                          "phrase_plan": "A-A'-B-B'"},
        "structure": [{"section": "Intro", "start_bar": 1, "bars": 4},
                      {"section": "Refrain", "start_bar": 5, "bars": 8},
                      {"section": "Outro", "start_bar": 13, "bars": 4}],
        "has_drums": True,
        "tracks": [
            {"name": "808", "role": "bass", "source_type": "gridline_native", "plugin": "gl.warmsynth", "mixer_track": 2, "volume_db": 0.0},
            {"name": "Lead", "role": "lead_melody", "source_type": "gridline_native", "plugin": "gl.pluck", "mixer_track": 6},
        ],
    }


def _plan() -> dict:
    return {
        "version": 1,
        "meta": {"title": "RefineTest", "genre": "dark_trap", "bpm": 140.0,
                 "time_signature": {"numerator": 4, "denominator": 4}, "key": "C minor",
                 "mode": "natural_minor", "mix_profile": "trap_rap", "has_drums": True,
                 "valence": -0.5, "tension": 0.65, "energy": 0.75, "depth": 0.60},
        "harmonic_core": {"key": "C minor", "mode": "natural_minor", "degree_progression": ["i", "VI", "III", "VII"]},
        "motifs": {"bass": {"role": "bass", "bars": 1, "loop": True,
                            "notes": [{"start": 0.0, "duration": 2.0, "pitch": 36, "velocity": 100}]}},
        "sections": [
            {"name": "Intro", "kind": "intro", "start_bar": 0, "bars": 4, "active_roles": ["bass"], "energy": 0.3},
            {"name": "Refrain", "kind": "chorus", "start_bar": 4, "bars": 8, "active_roles": ["bass", "lead"], "energy": 0.8},
        ],
    }


class TestPrepareSessionScenariosCD(unittest.TestCase):
    def test_scenario_c_remix_analysis(self):
        import soundfile as sf
        from audio_analyzer_server import studio_prepare_session
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "source.wav")
            t = np.arange(int(6 * 22050)) / 22050
            sf.write(src, 0.3 * np.sin(2 * np.pi * 110.0 * t), 22050)  # La 110 Hz
            out = studio_prepare_session(tmp, "remix", remix_source_path=src)
            rep = json.loads(out)
            self.assertTrue(rep["ok"])
            self.assertEqual(rep["scenario"], "C")
            self.assertIsNotNone(rep["remix_analysis"])
            self.assertIn("bpm", rep["remix_analysis"])
            self.assertIn("suggested_harmonic_core", rep["remix_analysis"])
            self.assertIn("degree_progression", rep["remix_analysis"]["suggested_harmonic_core"])
            self.assertLessEqual(len(out), 1500)

    def test_scenario_d_lyrics_plan(self):
        from audio_analyzer_server import studio_prepare_session
        with tempfile.TemporaryDirectory() as tmp:
            out = studio_prepare_session(tmp, "chant", lyrics_spec={
                "text": "un deux trois quatre cinq six sept huit neuf dix onze douze",
                "language": "fr", "vocal_range_midi": [50, 74]})
            rep = json.loads(out)
            self.assertTrue(rep["ok"])
            self.assertEqual(rep["scenario"], "D")
            lp = rep["lyrics_plan"]
            self.assertGreater(lp["n_syllables"], 0)
            self.assertEqual(lp["vocal_range_midi"], [50, 74])
            self.assertIn("breath_windows_s", lp)
            self.assertLessEqual(len(out), 1500)

    def test_scenario_ab_unchanged(self):
        from audio_analyzer_server import studio_prepare_session
        with tempfile.TemporaryDirectory() as tmp:
            rep = json.loads(studio_prepare_session(tmp, "trap sombre"))
            self.assertEqual(rep["scenario"], "A/B")
            self.assertIsNone(rep["remix_analysis"])
            self.assertIsNone(rep["lyrics_plan"])


class TestRefineTrack(unittest.TestCase):
    def _project(self, tmp: str) -> None:
        with open(os.path.join(tmp, "song_blueprint.json"), "w", encoding="utf-8") as fh:
            json.dump(_blueprint(), fh)
        with open(os.path.join(tmp, "arrangement_plan.json"), "w", encoding="utf-8") as fh:
            json.dump(_plan(), fh)

    def test_delta_applied_with_rollback_and_preview(self):
        import soundfile as sf
        from audio_analyzer_server import studio_refine_track
        with tempfile.TemporaryDirectory() as tmp:
            self._project(tmp)
            os.makedirs(os.path.join(tmp, "renders"))
            t = np.arange(int(30 * 22050)) / 22050
            sf.write(os.path.join(tmp, "renders", "master.wav"), 0.3 * np.sin(2 * np.pi * 220 * t), 22050)
            delta = {
                "seed_lock": True,
                "mix_adjustments_db": {"808": -1.5, "lead": 1.0},
                "section_overrides": {"Refrain": {"energy_delta": 0.15, "comping_style": "rhythmic_comp", "brightness_shift": "brighter"}},
                "swap_role": None,
                "feedback_tags": ["great_melody", "softer_808"],
            }
            out = studio_refine_track(tmp, delta)
            rep = json.loads(out)
            self.assertTrue(rep["ok"], rep)
            self.assertEqual(rep["tour"], "3b")
            self.assertTrue(rep["seed_lock"])
            self.assertTrue(any("mix:808-1.5dB" in a for a in rep["applied"]), rep["applied"])
            self.assertTrue(any("comping:Refrain=rhythmic_comp" in a for a in rep["applied"]))
            self.assertTrue(rep["taste_recorded"])
            self.assertIsNotNone(rep["preview_highlights"])
            self.assertAlmostEqual(rep["preview_highlights"]["duration_s"], 24.0, delta=0.5)
            # le delta a bien été écrit dans les fichiers
            with open(os.path.join(tmp, "song_blueprint.json"), encoding="utf-8") as fh:
                bp = json.load(fh)
            self.assertEqual(bp["seed"], 42)  # seed_lock : jamais touché
            bass = next(t for t in bp["tracks"] if t["name"] == "808")
            self.assertAlmostEqual(bass["volume_db"], -1.5, places=2)
            with open(os.path.join(tmp, "arrangement_plan.json"), encoding="utf-8") as fh:
                plan = json.load(fh)
            refrain = next(s for s in plan["sections"] if s["name"] == "Refrain")
            self.assertEqual(refrain["comping_style"], "rhythmic_comp")
            self.assertAlmostEqual(refrain["energy"], 0.95, places=2)
            # preview wav existe
            self.assertTrue(os.path.isfile(os.path.join(tmp, "renders", "preview_highlights.wav")))
            self.assertLessEqual(len(out), 1500)

    def test_rollback_on_invalid_blueprint(self):
        from audio_analyzer_server import studio_refine_track
        with tempfile.TemporaryDirectory() as tmp:
            self._project(tmp)
            # delta qui casse le schéma : on simule via un mix_adjustments sur piste inconnue (pas de crash)
            out = studio_refine_track(tmp, {"mix_adjustments_db": {"inexistant": -3.0}})
            rep = json.loads(out)
            self.assertTrue(rep["ok"])  # aucun crash, verdict transactionnel présent
            self.assertIn("verdict", rep)


class TestCleanupAndSchemaVersion(unittest.TestCase):
    def test_cleanup_temp_dirs(self):
        from audio_analyzer_server import cleanup_temp_dirs
        with tempfile.TemporaryDirectory() as tmp:
            for d in (".tmp_soundcheck", ".tmp_probe", os.path.join("renders", "stems_core")):
                os.makedirs(os.path.join(tmp, d), exist_ok=True)
            rep = cleanup_temp_dirs(tmp, keep_stems=False)
            self.assertEqual(sorted(rep["removed"]), [".tmp_probe", ".tmp_soundcheck", "renders/stems_core"])
            self.assertFalse(os.path.isdir(os.path.join(tmp, ".tmp_probe")))

    def test_samples_db_user_version(self):
        from indexer import DB_SCHEMA_VERSION, init_db
        with tempfile.TemporaryDirectory() as tmp:
            conn = init_db(os.path.join(tmp, "samples.db"))
            v = int(conn.execute("PRAGMA user_version").fetchone()[0])
            conn.close()
            self.assertGreaterEqual(v, DB_SCHEMA_VERSION)


class TestRemixStemRole(unittest.TestCase):
    def test_classify_role_remix_stem(self):
        from score_ir import classify_role
        self.assertEqual(classify_role("remix_stem"), "remix_stem")
        self.assertEqual(classify_role("Remix Vox"), "remix_stem")
        self.assertEqual(classify_role("vocals"), "vocals")
        with open(os.path.join(ROOT, "schemas", "score_ir.schema.json"), encoding="utf-8") as fh:
            schema = json.load(fh)
        # le schéma accepte le rôle remix_stem (recherche récursive de l'enum "role")
        found: list[list[str]] = []

        def _walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == "role" and isinstance(v, dict) and isinstance(v.get("enum"), list):
                        found.append(list(v["enum"]))
                    _walk(v)
            elif isinstance(o, list):
                for it in o:
                    _walk(it)

        _walk(schema)
        self.assertTrue(any("remix_stem" in r for r in found), found)


class TestPreviewHighlights(unittest.TestCase):
    def test_24s_three_segments(self):
        import soundfile as sf
        from preview_lib import build_preview_highlights
        with tempfile.TemporaryDirectory() as tmp:
            master = os.path.join(tmp, "master.wav")
            t = np.arange(int(40 * 22050)) / 22050
            sf.write(master, 0.3 * np.sin(2 * np.pi * 330 * t), 22050)
            rep = build_preview_highlights(master)
            self.assertTrue(rep["ok"])
            self.assertAlmostEqual(rep["duration_s"], 24.0, delta=0.05)
            self.assertEqual(rep["labels"], ["verse", "chorus", "bridge_outro"])
            data, sr = sf.read(rep["out"])
            self.assertLessEqual(float(np.max(np.abs(data))), 1.0)  # pas de clip


if __name__ == "__main__":
    unittest.main()
