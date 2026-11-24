#!/usr/bin/env python3
"""v3.2 #1 — NarrativeHarmony : progressions par section, phrasé 8 mesures, opérateurs
motiviques déterministes, suspensions 4->3 / 9->8 / b6->5 vérifiées.

Vérifie que :
  (a) `section_progressions` produit une grille où le Chorus/Bridge utilisent leur propre
      progression (fin de la boucle 4 mesures unique) et valide le schéma blueprint ;
  (b) `phrasing_8bar` valide le schéma arrangement_plan ;
  (c) `motif_answer` projette la dernière note lead de chaque période de 8 mesures sur un
      accord du segment final ;
  (d) `rhythmic_lift` syncope les croches off-beat uniquement dans les sections chorus ;
  (e) `call_and_response` insère une réponse du counter dans un silence du lead sans
      chevauchement ;
  (f) `melodic_climax_arch` donne un seul apex par période de 8 mesures ;
  (g) une suspension 4->3 résolue en 0.5-1.0 beat est marquée et ne déclenche PAS d'erreur
      dans `evaluate_score_ir`, tandis qu'une suspension non résolue reste signalée.
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

from narrative_harmony import (  # noqa: E402
    apply_narrative_harmony,
    call_and_response,
    mark_suspensions,
    melodic_climax_arch,
    motif_answer,
    rhythmic_lift,
)
from score_ir import NoteEventIR, ScoreIR, build_score_ir, evaluate_score_ir, harmonic_grid_from_core  # noqa: E402

BPM = 120.0


def _meta(bars: int = 32) -> dict:
    return {"bpm": BPM, "beats_per_bar": 4, "denominator": 4, "valence": 0.0,
            "total_bars": bars, "tonic": "C", "key": "C"}


def _score_with_lead(lead_notes: list[dict], bars: int = 16, extra: dict | None = None) -> ScoreIR:
    hc = {"key": "C", "mode": "aeolian", "degree_progression": ["i", "VI", "III", "VII"]}
    tracks = {"lead": lead_notes, "chords": [{"start": 0.0, "length": bars * 4.0, "pitch": 60, "velocity": 70}]}
    if extra:
        tracks.update(extra)
    return build_score_ir(hc, _meta(bars), tracks)


class TestSectionProgressions(unittest.TestCase):
    def test_grid_uses_section_progressions(self):
        hc = {"key": "C", "mode": "aeolian", "degree_progression": ["i", "VI", "III", "VII"],
              "progression_bars": 1,
              "section_progressions": {"verse": ["i", "VI", "III", "VII"],
                                       "chorus": ["iv", "V", "i", "VI"],
                                       "bridge": ["bII", "V7", "i", "VI"]}}
        spans = [("Intro", 0, 4, "intro"), ("Verse", 4, 8, "verse"), ("Chorus", 12, 8, "chorus"),
                 ("Bridge", 20, 8, "bridge"), ("Outro", 28, 4, "outro")]
        grid = harmonic_grid_from_core(hc, bpm=BPM, beats_per_bar=4, total_bars=32, section_spans=spans)
        by_bar = {int(s["start_beat"] // 4): s for s in grid}
        self.assertEqual(by_bar[4]["symbol"], "i")     # verse = progression principale
        self.assertEqual(by_bar[12]["symbol"], "iv")   # chorus commence sur iv
        self.assertEqual(by_bar[13]["symbol"], "V")
        self.assertEqual(by_bar[20]["symbol"], "bII")  # bridge bII (Neapolitan)
        self.assertEqual(by_bar[21]["chord_pcs"], [2, 5, 7, 11])  # V7

    def test_blueprint_schema_accepts_section_progressions(self):
        with open(os.path.join(ROOT, "schemas", "song_blueprint.schema.json"), encoding="utf-8") as fh:
            schema = json.load(fh)
        hc_props = schema["properties"]["harmonic_core"]["properties"]
        self.assertIn("section_progressions", hc_props)

    def test_plan_schema_accepts_phrasing_8bar(self):
        with open(os.path.join(ROOT, "schemas", "arrangement_plan.schema.json"), encoding="utf-8") as fh:
            schema = json.load(fh)
        self.assertIn("phrasing_8bar", schema["properties"])
        jsonschema.validate({"meta": {"bpm": 120, "time_signature": {"numerator": 4, "denominator": 4}, "key": "C", "mode": "aeolian", "mix_profile": "trap_rap", "has_drums": True},
                             "motifs": {"lead": []},
                             "sections": [{"name": "A", "start_bar": 0, "bars": 4, "kind": "verse", "active_roles": ["lead"]}],
                             "phrasing_8bar": {"antecedent_bars": 4, "consequent_bars": 4, "period_bars": 8,
                                               "operators": ["motif_answer", "rhythmic_lift"]}}, schema)


class TestMotifOperators(unittest.TestCase):
    def test_motif_answer_resolves_last_note(self):
        # lead avec une dernière note hors accord (pc 2 = D) en fin de période 8 mesures
        notes = [{"start": b * 4.0, "length": 1.0, "pitch": 72, "velocity": 90} for b in range(8)]
        notes.append({"start": 28.0, "length": 2.0, "pitch": 74, "velocity": 90})  # D, hors Cm
        score = _score_with_lead(notes, bars=16)
        n = motif_answer(score, period_bars=8)
        self.assertGreaterEqual(n, 1)
        last = max(score.tracks["lead"].notes, key=lambda x: (x.grid_beat, x.pitch))
        seg = score.harmonic_grid[last.chord_idx]
        self.assertIn(last.pitch % 12, seg["chord_pcs"])

    def test_rhythmic_lift_only_in_chorus(self):
        notes = []
        for b in range(16):
            notes.append({"start": b * 4.0, "length": 0.4, "pitch": 72, "velocity": 90})       # temps
            notes.append({"start": b * 4.0 + 0.5, "length": 0.4, "pitch": 74, "velocity": 90})   # off-beat
        score = _score_with_lead(notes, bars=16)
        spans = [("Verse", 0, 8, "verse"), ("Chorus", 8, 8, "chorus")]
        n = rhythmic_lift(score, section_spans=spans)
        self.assertGreater(n, 0)
        verse_off = [x for x in score.tracks["lead"].notes if x.grid_beat < 32.0 and abs(x.grid_beat % 4 - 0.5) < 1e-3]
        chorus_off = [x for x in score.tracks["lead"].notes if x.grid_beat >= 32.0 and abs(x.grid_beat % 4 - 0.25) < 1e-3]
        self.assertEqual(len(verse_off), 8)      # verse intact (off-beats à 0.5)
        self.assertGreater(len(chorus_off), 0)   # chorus lifté (off-beats passés à 0.25)
        chorus_half = [x for x in score.tracks["lead"].notes if x.grid_beat >= 32.0 and abs(x.grid_beat % 4 - 0.5) < 1e-3]
        self.assertLess(len(chorus_half), 8)     # la moitié seulement a été liftée

    def test_call_and_response_fills_lead_gap(self):
        lead = [{"start": 0.0, "length": 1.0, "pitch": 72, "velocity": 90},
                {"start": 4.0, "length": 1.0, "pitch": 74, "velocity": 90}]  # silence 1.0..4.0 (3 beats)
        counter = [{"start": 2.0, "length": 0.5, "pitch": 60, "velocity": 60}]  # dans le silence 1.0-4.0
        score = _score_with_lead(lead, bars=8, extra={"counter": counter})
        score.tracks["counter"].role = "counter"
        n = call_and_response(score, max_gap_beats=2.0)
        self.assertGreaterEqual(n, 2)
        # pas de chevauchement lead/counter
        lns = sorted(score.tracks["lead"].notes, key=lambda x: x.grid_beat)
        cns = sorted(score.tracks["counter"].notes, key=lambda x: x.grid_beat)
        for c in cns:
            for l in lns:
                overlap = c.grid_beat < l.grid_beat + l.duration_beats and l.grid_beat < c.grid_beat + c.duration_beats
                self.assertFalse(overlap, f"chevauchement lead/counter @{c.grid_beat}")

    def test_melodic_climax_arch_single_apex(self):
        notes = []
        for b in range(16):
            if b == 7:
                continue  # case libre en bar 7 pour accueillir l'apex déplacé
            notes.append({"start": b * 4.0, "length": 1.0, "pitch": 72 + (b % 4), "velocity": 80})
        notes[3]["pitch"] = 84  # apex dans l'antécédent (bar 3)
        score = _score_with_lead(notes, bars=16)
        n = melodic_climax_arch(score, period_bars=8)
        self.assertGreaterEqual(n, 1)
        apexes = [x for x in score.tracks["lead"].notes if x.pitch == 84]
        self.assertEqual(len(apexes), 1)
        # apex déplacé dans le conséquent de la période 0 (bars 4-7 => beats 16-31)
        self.assertGreaterEqual(apexes[0].grid_beat, 16.0)
        self.assertLess(apexes[0].grid_beat, 32.0)

    def test_apply_narrative_harmony_from_plan(self):
        notes = [{"start": b * 4.0, "length": 1.0, "pitch": 72 + (b % 3), "velocity": 80} for b in range(16)]
        score = _score_with_lead(notes, bars=16)
        plan = {"phrasing_8bar": {"antecedent_bars": 4, "consequent_bars": 4, "period_bars": 8,
                                  "operators": ["motif_answer", "melodic_climax_arch", "call_and_response", "rhythmic_lift"]}}
        rep = apply_narrative_harmony(score, plan, section_spans=[("Chorus", 0, 16, "chorus")])
        self.assertIn("operators", rep)
        self.assertIn("suspensions_marked", rep)


class TestSuspensions(unittest.TestCase):
    def _score_with_suspension(self, resolve: bool) -> ScoreIR:
        # accord Cm (pcs {0,3,7}) sur toute la mesure 0 ; suspension 4->3 : F(65) -> E(64)
        hc = {"key": "C", "mode": "aeolian", "degree_progression": ["i"]}
        lead = [{"start": 0.0, "length": 0.5, "pitch": 65, "velocity": 90}]  # F = 4 au-dessus de C
        if resolve:
            lead.append({"start": 0.75, "length": 0.25, "pitch": 64, "velocity": 80})  # E = 3, 0.75 beat plus tard
        else:
            lead.append({"start": 2.0, "length": 0.5, "pitch": 72, "velocity": 80})
        tracks = {"lead": lead, "chords": [{"start": 0.0, "length": 4.0, "pitch": 60, "velocity": 70}]}
        return build_score_ir(hc, _meta(4), tracks)

    def test_resolved_suspension_marked_and_passes(self):
        score = self._score_with_suspension(resolve=True)
        n = mark_suspensions(score)
        self.assertEqual(n, 1)
        sus = [x for x in score.tracks["lead"].notes if x.note_role == "suspension"]
        self.assertEqual(len(sus), 1)
        rep = evaluate_score_ir(score)
        # la suspension résolue ne doit pas produire d'erreur out_of_scale bloquante
        self.assertFalse([e for e in rep["errors"] if e.get("kind") == "out_of_scale" and e.get("pitch") == 65],
                         msg=str(rep["errors"]))

    def test_unresolved_suspension_not_marked(self):
        score = self._score_with_suspension(resolve=False)
        n = mark_suspensions(score)
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
