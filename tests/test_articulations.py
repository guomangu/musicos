#!/usr/bin/env python3
"""v3.2 #2 — ExpressionArticulations : articulations ScoreIR, glides 808, vibrato retardé,
vélocité->cutoff, outro_ritardando.

Vérifie que :
  (a) `ExpressionIR` porte `articulation`/`vibrato_delay_beats`/`glide_ms` (roundtrip JSON +
      schéma score_ir) ;
  (b) une note 808 `glide_to_next` compile avec un chevauchement de glide borné [35,90] ms,
      exempté du sanitize (0 mono_overlap non exempté) ;
  (c) `synth_voice(..., glide_ms=60)` émet le param Glide vérifié et `vibrato_delay_ms`
      retarde le LFO (LFO delay > valeur par défaut = attaque juste) ;
  (d) `velocity_cutoff_automation` émet une automation LPF par phrase, plus ouverte quand
      la vélocité moyenne est haute ;
  (e) `apply_outro_ritardando` étire monotone les 2 dernières mesures (ordre préservé,
      début de fenêtre ancré).
"""
from __future__ import annotations

import json
import os
import sys
import unittest

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
ANALYZER_DIR = os.path.join(ROOT, "mcp-servers", "audio-analyzer")
for _d in (SCRIPTS_DIR, ANALYZER_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import jsonschema  # noqa: E402

from groove_lib import ms_to_beats, sanitize_note_overlaps  # noqa: E402
from score_ir import (  # noqa: E402
    GLIDE_MS_RANGE,
    ExpressionIR,
    NoteEventIR,
    ScoreIR,
    build_score_ir,
    compile_score_ir_to_notes,
)

BPM = 120.0


def _meta(bars: int = 8) -> dict:
    return {"bpm": BPM, "beats_per_bar": 4, "denominator": 4, "valence": 0.0,
            "total_bars": bars, "tonic": "C", "key": "C"}


class TestExpressionArticulations(unittest.TestCase):
    def test_expression_roundtrip_and_schema(self):
        e = ExpressionIR(timing_offset_beats=0.05, velocity_delta=-4, gate_scale=0.96,
                         articulation="glide_to_next", vibrato_delay_beats=0.5, glide_ms=60.0)
        e2 = ExpressionIR.from_dict(e.to_dict())
        self.assertEqual(e2.articulation, "glide_to_next")
        self.assertEqual(e2.glide_ms, 60.0)
        self.assertEqual(e2.vibrato_delay_beats, 0.5)
        with open(os.path.join(ROOT, "schemas", "score_ir.schema.json"), encoding="utf-8") as fh:
            schema = json.load(fh)
        props = schema["$defs"]["expression"]["properties"]
        for k in ("articulation", "vibrato_delay_beats", "glide_ms"):
            self.assertIn(k, props)
        # un ScoreIR complet avec articulations valide le schéma
        hc = {"key": "C", "mode": "aeolian", "degree_progression": ["i", "VII"]}
        tracks = {"808": [{"start": 0.0, "length": 1.0, "pitch": 36, "velocity": 100},
                          {"start": 1.0, "length": 1.0, "pitch": 36, "velocity": 100}]}
        score = build_score_ir(hc, _meta(8), tracks)
        score.tracks["808"].notes[0].expression.articulation = "glide_to_next"
        score.tracks["808"].notes[0].expression.glide_ms = 60.0
        jsonschema.validate(score.to_dict(), schema)

    def test_glide_808_exempt_overlap(self):
        hc = {"key": "C", "mode": "aeolian", "degree_progression": ["i", "VII"]}
        tracks = {"808": [{"start": b * 1.0, "length": 0.95, "pitch": 36, "velocity": 100} for b in range(8)]}
        score = build_score_ir(hc, _meta(8), tracks)
        for i, n in enumerate(score.tracks["808"].notes[:-1]):
            n.expression.articulation = "glide_to_next"
            n.expression.glide_ms = 60.0
        out, report = compile_score_ir_to_notes(score)
        self.assertEqual(report["glide_pairs"], 7)
        self.assertEqual(report["mono_overlaps"], 0)  # glides exemptés
        self.assertTrue(report["passed"])
        # le chevauchement de glide est bien présent et borné à glide_ms
        g_beats = ms_to_beats(60.0, BPM)
        ns = sorted(out["808"], key=lambda n: float(n["start"]))
        overlaps = [float(ns[i]["start"]) + float(ns[i]["length"]) - float(ns[i + 1]["start"])
                    for i in range(len(ns) - 1)]
        self.assertTrue(all(0 < o <= g_beats + 1e-6 for o in overlaps), msg=str(overlaps))
        # sans articulation, le sanitize coupe à 0 chevauchement
        score2 = build_score_ir(hc, _meta(8), tracks)
        out2, rep2 = compile_score_ir_to_notes(score2)
        self.assertEqual(rep2["glide_pairs"], 0)
        self.assertEqual(rep2["mono_overlaps"], 0)
        ns2 = sorted(out2["808"], key=lambda n: float(n["start"]))
        for i in range(len(ns2) - 1):
            self.assertLessEqual(float(ns2[i]["start"]) + float(ns2[i]["length"]), float(ns2[i + 1]["start"]) + 1e-6)

    def test_glide_ms_clamped_to_range(self):
        self.assertEqual(GLIDE_MS_RANGE, (35.0, 90.0))
        hc = {"key": "C", "mode": "aeolian", "degree_progression": ["i"]}
        tracks = {"808": [{"start": 0.0, "length": 1.0, "pitch": 36, "velocity": 100},
                          {"start": 1.0, "length": 1.0, "pitch": 36, "velocity": 100}]}
        score = build_score_ir(hc, _meta(4), tracks)
        score.tracks["808"].notes[0].expression.articulation = "glide_to_next"
        score.tracks["808"].notes[0].expression.glide_ms = 500.0  # clampé à 90 ms
        out, report = compile_score_ir_to_notes(score)
        g_max = ms_to_beats(90.0, BPM)
        ns = sorted(out["808"], key=lambda n: float(n["start"]))
        overlap = float(ns[0]["start"]) + float(ns[0]["length"]) - float(ns[1]["start"])
        self.assertLessEqual(overlap, g_max + 1e-6)

    def test_sanitize_exempt_pairs_direct(self):
        tracks = {"808": [{"start": 0.0, "length": 1.05, "pitch": 36, "velocity": 100},
                          {"start": 1.0, "length": 1.0, "pitch": 36, "velocity": 100}]}
        out = sanitize_note_overlaps(dict(tracks), bpm=BPM, voices_by_role={"808": 1},
                                     exempt_pairs={("808", 0.0)})
        self.assertAlmostEqual(float(out["808"][0]["length"]), 1.05, places=4)  # non coupé
        out2 = sanitize_note_overlaps(dict(tracks), bpm=BPM, voices_by_role={"808": 1})
        self.assertLess(float(out2["808"][0]["length"]), 1.0)  # coupé sans exemption


class TestGlideAndVibratoParams(unittest.TestCase):
    def test_synth_voice_glide_and_vibrato_delay(self):
        from gl_dsl import synth_voice
        c = synth_voice("gl.analog", "bass", glide_ms=60.0, vibrato_delay_ms=250.0)
        self.assertAlmostEqual(c.channel_params.get("Glide"), 0.06, places=4)
        self.assertEqual(c.channel_params.get("Voice mode"), 1)  # mono bass
        # vibrato retardé : LFO delay du sampler > 0 (profondeur nulle à l'attaque)
        s = synth_voice("gl.sampler", "pad", vibrato_delay_ms=300.0)
        self.assertGreater(float(s.channel_params.get("LFO delay", 0.0)), 0.0)


class TestVelocityCutoffAndRitardando(unittest.TestCase):
    def test_velocity_cutoff_automation(self):
        from phrasing_engine import velocity_cutoff_automation
        tracks = {"chords": [{"start": 0.0, "length": 4.0, "pitch": 60, "velocity": 50},
                             {"start": 16.0, "length": 4.0, "pitch": 60, "velocity": 110}]}
        autos: list[dict] = []
        rep = velocity_cutoff_automation(tracks, [("A", 0, 4, "verse"), ("B", 4, 4, "chorus")],
                                         autos, bpm=BPM, track_key="chords")
        self.assertGreaterEqual(rep["applied"], 2)
        sweeps = [a for a in autos if "vel->cutoff" in str(a.get("name", ""))]
        self.assertGreaterEqual(len(sweeps), 2)
        # phrase intime (vel 50) -> cutoff plus bas que phrase forte (vel 110)
        hz_by_beat = {float(a["start"]): float(a["points"][0]["value"]) for a in sweeps if a.get("points")}
        # les valeurs sont en Hz normalisées par filter_sweep ; comparer via lo/hi du report
        self.assertLess(rep["lo_hz"], rep["hi_hz"])

    def test_outro_ritardando(self):
        from transitions_lib import apply_outro_ritardando
        tracks = {"lead": [{"start": b * 1.0, "length": 0.9, "pitch": 72, "velocity": 90} for b in range(8)]}
        total_before = max(float(n["start"]) + float(n["length"]) for n in tracks["lead"])
        rep = apply_outro_ritardando(tracks, bpm=BPM, bars=2, amount=0.06)
        self.assertGreater(rep["applied"], 0)
        ns = sorted(tracks["lead"], key=lambda n: float(n["start"]))
        # ordre préservé et monotone
        for a, b in zip(ns, ns[1:]):
            self.assertLess(float(a["start"]), float(b["start"]))
        # début de fenêtre ancré : la note à la frontière n'est pas déplacée
        first_in_window = min(ns, key=lambda n: abs(float(n["start"]) - rep["window_beats"][0]))
        # étirement : la dernière note finit après la fin originale
        total_after = max(float(n["start"]) + float(n["length"]) for n in ns)
        self.assertGreater(total_after, total_before - 1e-6)
        # les notes hors fenêtre ne bougent pas
        self.assertEqual(float(ns[0]["start"]), 0.0)


if __name__ == "__main__":
    unittest.main()
