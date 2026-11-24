#!/usr/bin/env python3
"""v3.2 #3 — FullCoverageCoreValidator : pré-vol symbolique 100 % du morceau + sonde 12 mesures.

Vérifie sur des ScoreIR synthétiques que :
  (a) un morceau varié (arc d'énergie contrasté, outro résolu, <= 8 pistes/mesure) passe ;
  (b) un clone de blocs 8 mesures (similarité > 0.94) est flaggé `clone_8bar` ;
  (c) > 8 pistes simultanées dans une mesure est flaggé `cognitive_load` ;
  (d) un arc d'énergie plat est flaggé `flat_energy_arc` ;
  (e) un outro sans tonique est flaggé `outro_resolution` ;
  (f) `build_probe_plan` produit bien 8 mesures noyau + 1 silence + 4 mesures hors-noyau
      avec au moins une piste absente du noyau.
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
ANALYZER_DIR = os.path.join(ROOT, "mcp-servers", "audio-analyzer")
for _d in (SCRIPTS_DIR, ANALYZER_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from full_coverage_validator import build_probe_plan, validate_full_score_ir  # noqa: E402
from score_ir import NoteEventIR, ScoreIR, build_score_ir, harmonic_grid_from_core  # noqa: E402

BPM = 120.0


def _meta(bars: int = 32) -> dict:
    return {"bpm": BPM, "beats_per_bar": 4, "denominator": 4, "valence": 0.0,
            "total_bars": bars, "tonic": "C", "key": "C"}


def _good_tracks(bars: int = 32) -> dict:
    """Morceau varié : 4 sections de 8 mesures, énergies différentes, outro sur C."""
    tracks = {"chords": [], "bass": [], "lead": [], "kick": [], "snare": [], "hat": []}
    hc = {"key": "C", "mode": "aeolian", "degree_progression": ["i", "VI", "III", "VII"], "progression_bars": 2}
    grid = harmonic_grid_from_core(hc, bpm=BPM, beats_per_bar=4, total_bars=bars)
    for seg in grid:
        s, e = float(seg["start_beat"]), float(seg["end_beat"])
        for pc in seg["chord_pcs"]:
            tracks["chords"].append({"start": s, "length": e - s, "pitch": 60 + pc, "velocity": 70})
        bp = 36 + int(seg["bass_pc"])
        t = s
        while t < e - 1e-6:
            tracks["bass"].append({"start": round(t, 5), "length": 0.9, "pitch": bp, "velocity": 100})
            t += 1.0
    scale = grid[0]["scale_pcs"]
    # lead : densité différente par section de 8 mesures (intro clairsemé, refrain dense)
    for bar in range(bars):
        s = bar * 4.0
        density = 1 if bar < 8 else (2 if bar < 16 else (4 if bar < 24 else 2))
        for k in range(density):
            pc = scale[(bar + k) % len(scale)]
            tracks["lead"].append({"start": round(s + k * 4.0 / density, 5), "length": 0.4,
                                   "pitch": 72 + pc, "velocity": 60 + 10 * (bar // 8)})
    # drums : intro = kick seul, puis + snare/hat
    for bar in range(bars):
        s = bar * 4.0
        tracks["kick"].append({"start": s, "length": 0.4, "pitch": 36, "velocity": 110})
        if bar >= 8:
            tracks["snare"].append({"start": s + 2.0, "length": 0.3, "pitch": 38, "velocity": 100})
            for k in range(4):
                tracks["hat"].append({"start": s + k, "length": 0.2, "pitch": 42, "velocity": 70})
    # outro : basse sur la tonique dans les 2 dernières mesures
    last_end = bars * 4.0
    tracks["bass"].append({"start": last_end - 8.0, "length": 8.0, "pitch": 36, "velocity": 90})
    return tracks


class TestFullCoverageValidator(unittest.TestCase):
    def test_good_song_passes(self):
        tracks = _good_tracks(32)
        score = build_score_ir({"key": "C", "mode": "aeolian", "degree_progression": ["i", "VI", "III", "VII"]},
                               _meta(32), tracks)
        sections = [("Intro", 0, 8, "intro"), ("Verse", 8, 8, "verse"), ("Chorus", 16, 8, "chorus"), ("Outro", 24, 8, "outro")]
        rep = validate_full_score_ir(score, sections=sections, declared_transition_beats=[32.0, 64.0, 96.0])
        self.assertTrue(rep["passed"], msg=str(rep["issues"]))
        self.assertEqual(rep["metrics"]["clone_blocks"], [])
        self.assertEqual(rep["metrics"]["overload_bars"], [])
        self.assertTrue(rep["metrics"]["outro_resolved"])

    def test_clone_8bar_detected(self):
        tracks = _good_tracks(16)
        # cloner le bloc 0..32 beats sur 32..64 : recopier toutes les notes +32 beats
        cloned = {k: list(v) for k, v in tracks.items()}
        for k, notes in tracks.items():
            for n in notes:
                if float(n["start"]) < 32.0:
                    m = dict(n)
                    m["start"] = float(n["start"]) + 32.0
                    cloned[k].append(m)
        score = build_score_ir({"key": "C", "mode": "aeolian", "degree_progression": ["i", "VI", "III", "VII"]},
                               _meta(32), cloned)
        rep = validate_full_score_ir(score)
        codes = [i["code"] for i in rep["issues"]]
        self.assertIn("clone_8bar", codes)
        self.assertFalse(rep["passed"])
        self.assertTrue(rep["metrics"]["clone_blocks"])

    def test_cognitive_load_detected(self):
        tracks = _good_tracks(8)
        for i in range(10):
            tracks[f"extra{i}"] = [{"start": 0.0, "length": 4.0, "pitch": 50 + i, "velocity": 80}]
        score = build_score_ir({"key": "C", "mode": "aeolian"}, _meta(8), tracks)
        rep = validate_full_score_ir(score)
        self.assertIn("cognitive_load", [i["code"] for i in rep["issues"]])
        self.assertIn(1, rep["metrics"]["overload_bars"])

    def test_flat_energy_arc_detected(self):
        # lead identique à chaque bloc de 4 mesures, drums identiques -> arc plat
        tracks = {"lead": [], "kick": []}
        for bar in range(16):
            tracks["lead"].append({"start": bar * 4.0, "length": 1.0, "pitch": 72, "velocity": 90})
            tracks["kick"].append({"start": bar * 4.0, "length": 0.4, "pitch": 36, "velocity": 110})
        score = build_score_ir({"key": "C", "mode": "aeolian"}, _meta(16), tracks)
        rep = validate_full_score_ir(score)
        self.assertIn("flat_energy_arc", [i["code"] for i in rep["issues"]])

    def test_outro_resolution_detected(self):
        # progression se terminant sur la tonique i (dernier segment root_pc = 0)
        hc = {"key": "C", "mode": "aeolian", "degree_progression": ["i", "VI", "III", "i"]}
        tracks = _good_tracks(16)
        # outro non résolu : aucune note de pc 0 (tonique) dans les 2 dernières mesures
        tracks["bass"] = [n for n in tracks["bass"] if not (float(n["start"]) >= 56.0 and int(n["pitch"]) % 12 == 0)]
        tracks["chords"] = [n for n in tracks["chords"] if not (float(n["start"]) >= 56.0 and int(n["pitch"]) % 12 == 0)]
        tracks["chords"].append({"start": 56.0, "length": 8.0, "pitch": 65, "velocity": 80})  # F, pas C
        score = build_score_ir(hc, _meta(16), tracks)
        rep = validate_full_score_ir(score)
        self.assertIn("outro_resolution", [i["code"] for i in rep["issues"]])
        self.assertFalse(rep["metrics"]["outro_resolved"])

    def test_build_probe_plan(self):
        plan = {
            "meta": {"bpm": 142, "time_signature": {"numerator": 4, "denominator": 4}, "key": "C", "mode": "aeolian"},
            "harmonic_core": {"key": "C", "mode": "aeolian", "degree_progression": ["i", "VI", "III", "VII"]},
            "motifs": {"chords": [], "bass": [], "lead": [], "kick": []},
            "sections": [
                {"name": "Intro", "start_bar": 0, "bars": 4, "kind": "intro", "active_roles": ["chords", "kick"]},
                {"name": "Chorus", "start_bar": 4, "bars": 8, "kind": "chorus", "active_roles": ["chords", "bass", "lead", "kick"]},
                {"name": "Bridge", "start_bar": 12, "bars": 8, "kind": "bridge", "active_roles": ["pad", "counter", "chords", "bass"]},
            ],
        }
        probe = build_probe_plan(plan, core_bars=8, probe_bars=4)
        secs = probe["sections"]
        self.assertEqual(len(secs), 3)
        self.assertEqual(secs[0]["bars"], 8)          # noyau 8 mesures
        self.assertEqual(secs[1]["bars"], 1)           # silence séparateur
        self.assertEqual(secs[1]["active_roles"], [])
        self.assertEqual(secs[2]["bars"], 4)           # tranche hors-noyau
        self.assertEqual(secs[2]["start_bar"], 9)
        p = probe["probe"]
        self.assertEqual(p["total_bars"], 13)
        self.assertEqual(p["core_section"], "Chorus")
        self.assertEqual(p["probe_section"], "Bridge")  # Bridge a pad+counter absents du noyau
        core_roles = set(secs[0]["active_roles"])
        probe_roles = set(secs[2]["active_roles"])
        self.assertTrue(probe_roles - core_roles)  # au moins une piste hors-noyau sondée


if __name__ == "__main__":
    unittest.main()
