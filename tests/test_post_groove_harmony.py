#!/usr/bin/env python3
"""v3.1 IR-02 — tests du ScoreIR et du clamp de frontière harmonique post-groove.

Vérifie sur 6 modes (dont Phrygien i-bII-i-bVI) avec swing=0.25 + strum + Haas :
  (a) le ScoreIR préserve les rôles exacts (évaluation sans `_infer_roles`) ;
  (b) aucune note chords/808 ne bave sur un accord suivant incompatible post-groove ;
  (c) zéro chevauchement sur la piste monophonique 808 ;
  (d) le clamp de frontière coupe strictement à chord_end - guard_ms (témoin positif b2) ;
  (e) `evaluate_score_ir` / `autofix_score_ir` détectent puis réparent un clash b2 injecté ;
  (f) la Low Interval Limit est respectée sur les blocs d'accords ;
  (g) le ScoreIR sérialisé valide `schemas/score_ir.schema.json`.
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

from score_ir import (  # noqa: E402
    LOW_INTERVAL_LIMIT,
    ScoreIR,
    apply_groove_to_score_ir,
    autofix_score_ir,
    build_score_ir,
    compile_score_ir_to_notes,
    evaluate_score_ir,
    harmonic_grid_from_core,
    ms_to_beats,
)

BPM = 142.0
BPB = 4
BARS = 8

MODES = {
    "phrygian": ["i", "bII", "i", "bVI"],
    "aeolian": ["i", "VI", "III", "VII"],
    "dorian": ["i", "bVII", "IV", "i"],
    "ionian": ["I", "IV", "V", "vi"],
    "harmonic_minor": ["i", "VI", "V7", "i"],
    "mixolydian": ["I", "bVII", "IV", "I"],
}


def _meta() -> dict:
    return {
        "bpm": BPM,
        "beats_per_bar": BPB,
        "denominator": 4,
        "valence": -0.75,
        "total_bars": BARS,
        "tonic": "C",
        "key": "C",
    }


def make_tracks(mode: str, prog: list[str]) -> tuple[dict, list[dict], dict]:
    hc = {"key": "C", "mode": mode, "degree_progression": prog, "progression_bars": 1}
    grid = harmonic_grid_from_core(hc, bpm=BPM, beats_per_bar=BPB, total_bars=BARS)
    scale_pcs = grid[0]["scale_pcs"]
    tracks = {"chords": [], "808": [], "lead": []}
    for seg in grid:
        s, e = float(seg["start_beat"]), float(seg["end_beat"])
        for pc in seg["chord_pcs"]:
            tracks["chords"].append({"start": s, "length": e - s, "pitch": 60 + pc, "velocity": 80})
        bass_pitch = 36 + int(seg["bass_pc"])
        t = s
        while t < e - 1e-6:
            tracks["808"].append({"start": round(t, 5), "length": 0.9, "pitch": bass_pitch, "velocity": 100})
            t += 1.0
        t = s
        i = 0
        while t < e - 1e-6:
            pc = scale_pcs[i % len(scale_pcs)]
            tracks["lead"].append({"start": round(t, 5), "length": 0.45, "pitch": 72 + pc, "velocity": 90})
            t += 0.5
            i += 1
    return hc, grid, tracks


class TestScoreIRPostGrooveHarmony(unittest.TestCase):
    def test_roles_preserved_without_infer_roles(self):
        hc, grid, tracks = make_tracks("phrygian", MODES["phrygian"])
        score = build_score_ir(hc, _meta(), tracks)
        self.assertEqual(score.tracks["808"].role, "bass")
        self.assertTrue(score.tracks["808"].monophonic)
        self.assertEqual(score.tracks["chords"].role, "chords")
        self.assertFalse(score.tracks["chords"].monophonic)
        self.assertEqual(score.tracks["lead"].role, "lead")
        res = evaluate_score_ir(score)
        self.assertEqual(res["roles_source"], "score_ir")
        self.assertTrue(res["passed"], msg=str(res["errors"][:5]))

    def test_schema_validation(self):
        hc, grid, tracks = make_tracks("phrygian", MODES["phrygian"])
        score = build_score_ir(hc, _meta(), tracks)
        with open(os.path.join(ROOT, "schemas", "score_ir.schema.json"), encoding="utf-8") as fh:
            schema = json.load(fh)
        jsonschema.validate(score.to_dict(), schema)
        roundtrip = ScoreIR.from_dict(score.to_dict())
        jsonschema.validate(roundtrip.to_dict(), schema)
        self.assertEqual(len(roundtrip.harmonic_grid), len(grid))

    def test_six_modes_zero_clash_after_groove(self):
        for mode, prog in MODES.items():
            with self.subTest(mode=mode):
                hc, grid, tracks = make_tracks(mode, prog)
                score = build_score_ir(hc, _meta(), tracks)
                counts = apply_groove_to_score_ir(score, swing=0.25, bpm=BPM, profile="trap_rap", beats_per_bar=BPB)
                self.assertGreater(counts["swing"], 0)
                self.assertGreater(counts["strum"], 0)
                self.assertGreater(counts["haas"], 0)
                out_tracks, report = compile_score_ir_to_notes(score)
                self.assertTrue(report["passed"], msg=f"{mode}: {report}")
                self.assertEqual(report["boundary_bleeds"], 0, msg=mode)
                self.assertEqual(report["b2_boundary_clashes"], 0, msg=mode)
                self.assertEqual(report["mono_overlaps"], 0, msg=mode)
                # 0 chevauchement 808 vérifié sur les notes compilées
                ns = sorted(out_tracks["808"], key=lambda n: float(n["start"]))
                for i in range(len(ns) - 1):
                    self.assertLessEqual(
                        float(ns[i]["start"]) + float(ns[i]["length"]),
                        float(ns[i + 1]["start"]) + 1e-6,
                        msg=f"{mode} overlap 808 @{ns[i]['start']}",
                    )

    def test_phrygian_bII_boundary_clamp_positive_control(self):
        # En C phrygien : i = {0,3,7} puis bII = {1,5,8} (basse Db=pc 1).
        # Une note tenue de pc 0 (C) franchissant la frontière vers bII crée un b2 -> clamp.
        hc, grid, tracks = make_tracks("phrygian", MODES["phrygian"])
        tracks["chords"].append({"start": 3.0, "length": 2.0, "pitch": 60, "velocity": 80})  # C tenu sur la frontière beat 4
        score = build_score_ir(hc, _meta(), tracks)
        # (e) evaluate détecte le clash b2 avant compilation
        pre = evaluate_score_ir(score)
        b2 = [e for e in pre["errors"] if e.get("kind") == "b2_boundary_clash"]
        self.assertTrue(b2, msg="témoin positif: le b2 de frontière doit être détecté")
        # autofix le répare (clamp via gate_scale)
        score, fix = autofix_score_ir(score)
        self.assertTrue(any(c["kind"] == "boundary_clamp" for c in fix["changes"]))
        post = evaluate_score_ir(score)
        self.assertFalse([e for e in post["errors"] if e.get("kind") in ("b2_boundary_clash", "boundary_clash")])
        # (d) post-groove + compile : la note est strictement bornée à chord_end - guard
        apply_groove_to_score_ir(score, swing=0.25, bpm=BPM, profile="trap_rap", beats_per_bar=BPB)
        out_tracks, report = compile_score_ir_to_notes(score)
        guard_beats = ms_to_beats(20.0, BPM)
        seg0_end = float(grid[0]["end_beat"])
        witness = [n for n in out_tracks["chords"] if abs(float(n["start"]) - 3.0) < 0.2 and int(n["pitch"]) == 60]
        self.assertTrue(witness)
        for n in witness:
            self.assertLessEqual(float(n["start"]) + float(n["length"]), seg0_end - guard_beats + 1e-4)
        self.assertTrue(report["passed"], msg=str(report))

    def test_anticipation_exempt_from_clamp(self):
        hc, grid, tracks = make_tracks("phrygian", MODES["phrygian"])
        score = build_score_ir(hc, _meta(), tracks)
        # note d'anticipation marquée : démarre dans le segment 0, appartient au bII suivant
        from score_ir import NoteEventIR, ExpressionIR
        score.tracks["chords"].notes.append(NoteEventIR(
            grid_beat=3.5, duration_beats=1.0, pitch=61, velocity=80,  # Db = pc 1 ∈ bII
            chord_idx=1, note_role="anticipation", lock_harmony=True,
            expression=ExpressionIR(timing_offset_beats=-0.5),
        ))
        out_tracks, report = compile_score_ir_to_notes(score)
        antic = [n for n in out_tracks["chords"] if int(n["pitch"]) == 61 and float(n["start"]) < 3.6]
        self.assertTrue(antic)
        # l'anticipation n'est PAS clampée à la frontière du segment 0 (exemption)
        self.assertGreater(float(antic[0]["start"]) + float(antic[0]["length"]), float(grid[0]["end_beat"]) - ms_to_beats(20.0, BPM) - 1e-6)

    def test_mono_808_overlap_sanitized(self):
        hc, grid, tracks = make_tracks("phrygian", MODES["phrygian"])
        tracks["808"][0]["length"] = 1.5  # chevauche la note suivante à +1.0 beat
        score = build_score_ir(hc, _meta(), tracks)
        apply_groove_to_score_ir(score, swing=0.25, bpm=BPM, profile="trap_rap", beats_per_bar=BPB)
        out_tracks, report = compile_score_ir_to_notes(score)
        self.assertEqual(report["mono_overlaps"], 0)
        ns = sorted(out_tracks["808"], key=lambda n: float(n["start"]))
        for i in range(len(ns) - 1):
            self.assertLessEqual(float(ns[i]["start"]) + float(ns[i]["length"]), float(ns[i + 1]["start"]) + 1e-6)

    def test_low_interval_limit_on_chords(self):
        hc, grid, tracks = make_tracks("ionian", MODES["ionian"])
        score = build_score_ir(hc, _meta(), tracks)
        res = evaluate_score_ir(score)
        lil = [e for e in res["errors"] if e.get("kind") == "low_interval_violation"]
        self.assertEqual(lil, [])
        # témoin positif : cluster grave {36, 38} (seconde sous G3=55... 38 < 60) -> violation LIL
        score.tracks["chords"].notes.append({"grid_beat": 0.0, "duration_beats": 1.0, "pitch": 36, "velocity": 80, "chord_idx": 0, "note_role": "chord_tone", "lock_harmony": True})
        score.tracks["chords"].notes.append({"grid_beat": 0.0, "duration_beats": 1.0, "pitch": 38, "velocity": 80, "chord_idx": 0, "note_role": "chord_tone", "lock_harmony": True})
        from score_ir import NoteEventIR
        score.tracks["chords"].notes = [n if isinstance(n, NoteEventIR) else NoteEventIR.from_dict(n) for n in score.tracks["chords"].notes]
        res2 = evaluate_score_ir(score)
        lil2 = [e for e in res2["errors"] if e.get("kind") == "low_interval_violation"]
        self.assertTrue(lil2)
        self.assertIn(2, LOW_INTERVAL_LIMIT)
        self.assertEqual(LOW_INTERVAL_LIMIT[2], 60)

    def test_grid_phrygian_bII_pcs(self):
        grid = harmonic_grid_from_core({"key": "C", "mode": "phrygian", "degree_progression": ["i", "bII", "i", "bVI"]}, bpm=BPM, beats_per_bar=4, total_bars=4)
        self.assertEqual(grid[0]["chord_pcs"], [0, 3, 7])
        self.assertEqual(grid[1]["chord_pcs"], [1, 5, 8])  # bII majeur (N6)
        self.assertEqual(grid[1]["bass_pc"], 1)
        self.assertIn(1, grid[0]["modal_tension_pcs"])  # b2 phrygien
        self.assertEqual(grid[3]["chord_pcs"], [0, 3, 8])  # bVI = Ab maj (tri sorted)


if __name__ == "__main__":
    unittest.main()
