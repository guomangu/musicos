#!/usr/bin/env python3
"""v3.3 J2 (#1) — test anti-code-mort : un flux minimal active effectivement les briques
v3.2/v3.3 : enricher (seed, section_progressions, phrase_length_bars, expression_defaults,
archétypes), ScoreIR, --probe (gap dynamique), FullCoverageCoreValidator, NarrativeHarmony,
ExpressionArticulations (glides Climax), TimbralCohesion (sends DRR), MelodicAppealIndex.
Le pipeline Gridline complet est testé séparément (skip si binaire absent)."""
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


def _minimal_blueprint() -> dict:
    return {
        "title": "EnrichTest", "genre": "trap", "mix_profile": "trap_rap",
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
            {"name": "808", "role": "bass", "source_type": "gridline_native", "plugin": "gl.warmsynth", "mixer_track": 2},
            {"name": "Chords", "role": "chords", "source_type": "gridline_native", "plugin": "gl.keys", "mixer_track": 4},
            {"name": "Lead", "role": "lead_melody", "source_type": "gridline_native", "plugin": "gl.pluck", "mixer_track": 6},
            {"name": "HiHat", "role": "hihat", "source_type": "gridline_native", "plugin": "gl.sampler", "mixer_track": 8},
        ],
    }


class TestBlueprintEnricher(unittest.TestCase):
    def test_fills_only_missing_fields(self):
        from blueprint_enricher import enrich_blueprint
        bp = _minimal_blueprint()
        bp["seed"] = 1234  # fourni explicitement -> jamais écrasé
        bp["harmonic_core"]["section_progressions"] = {"verse": ["i"]}
        enriched, report = enrich_blueprint(bp, project_dir=None)
        self.assertEqual(enriched["seed"], 1234)
        self.assertEqual(enriched["harmonic_core"]["section_progressions"], {"verse": ["i"]})
        # champs omis complétés
        self.assertEqual(enriched["phrase_length_bars"], 8)  # Refrain = 8 mesures
        self.assertIn("expression_defaults", enriched)       # 808 trap_rap sans articulation
        self.assertEqual(enriched["expression_defaults"]["bass_808"]["climax_bars"], [4, 8])
        # archétypes gl.* attribués
        archs = {t["name"]: t.get("archetype") for t in enriched["tracks"] if t.get("plugin", "").startswith("gl.")}
        self.assertEqual(archs["808"], "trap_rap:bass")
        self.assertEqual(archs["Chords"], "trap_rap:chords")
        self.assertEqual(archs["Lead"], "trap_rap:lead")
        self.assertGreaterEqual(report["n_decisions"], 3)

    def test_derives_seed_and_progressions(self):
        from blueprint_enricher import enrich_blueprint, _derive_section_progressions
        from determinism import derive_seed
        bp = _minimal_blueprint()
        enriched, report = enrich_blueprint(bp)
        self.assertEqual(enriched["seed"], derive_seed("EnrichTest", "C minor", 140.0))
        sp = enriched["harmonic_core"]["section_progressions"]
        self.assertIn("bridge", sp) and self.assertIn("outro", sp)
        self.assertEqual(sp["outro"][-1], "i")  # résolution tonique
        self.assertEqual(sp["pre_chorus"][-1], "V")  # tension dominante

    def test_enrichment_report_written(self):
        from blueprint_enricher import enrich_blueprint, write_enrichment_report
        with tempfile.TemporaryDirectory() as tmp:
            enriched, report = enrich_blueprint(_minimal_blueprint(), project_dir=tmp)
            path = write_enrichment_report(tmp, report)
            self.assertTrue(os.path.isfile(path))
            with open(path, encoding="utf-8") as fh:
                on_disk = json.load(fh)
            self.assertEqual(on_disk["n_decisions"], report["n_decisions"])
            self.assertIn("decisions", on_disk)

    def test_climax_glides_injected_on_score_ir(self):
        from blueprint_enricher import enrich_blueprint, inject_climax_glides
        from score_ir import build_score_ir
        bp = _minimal_blueprint()
        enriched, _ = enrich_blueprint(bp)
        meta = {"bpm": 140.0, "beats_per_bar": 4, "denominator": 4, "valence": -0.5,
                "total_bars": 16, "tonic": "C", "key": "C"}
        hc = {"key": "C", "mode": "aeolian", "degree_progression": ["i", "VI", "III", "VII"]}
        bass = [{"start": b * 1.0, "length": 0.9, "pitch": 36, "velocity": 100} for b in range(64)]
        sir = build_score_ir(hc, meta, {"bass": bass})
        sections = [type("S", (), {"name": "Refrain", "start_bar": 4})()]  # refrain à partir de la mes. 5 (0-based 4)
        n = inject_climax_glides(sir, enriched, sections)
        self.assertGreaterEqual(n, 1)
        glides = [n for t in sir.tracks.values() for n in t.notes
                  if getattr(n, "expression", None) and getattr(n.expression, "articulation", "") == "glide_to_next"]
        self.assertGreaterEqual(len(glides), 1)
        for g in glides:
            self.assertGreaterEqual(g.expression.glide_ms, 35)
            self.assertLessEqual(g.expression.glide_ms, 90)


class TestPipelineActivationSymbolic(unittest.TestCase):
    """Sans binaire gridline : on vérifie que compile_song enclenche l'enricher + le seed,
    et que la sonde/full-coverage sont actives sur un ScoreIR compilé."""

    def test_compile_song_enriches_and_seeds(self):
        from compile_song import compile_song
        with tempfile.TemporaryDirectory() as tmp:
            bp = _minimal_blueprint()
            with open(os.path.join(tmp, "song_blueprint.json"), "w", encoding="utf-8") as fh:
                json.dump(bp, fh)
            rep = compile_song(tmp, core_only=True)
            self.assertIn("seed", rep)
            self.assertIn("enrichment", rep)
            # le blueprint écrit est enrichi et valide contre le schéma
            with open(os.path.join(tmp, "song_blueprint.json"), encoding="utf-8") as fh:
                enriched = json.load(fh)
            self.assertIn("seed", enriched)
            self.assertIn("section_progressions", enriched["harmonic_core"])
            self.assertTrue(os.path.isfile(os.path.join(tmp, "renders", "diagnostics", "enrichment_report.json")))
            import jsonschema
            with open(os.path.join(ROOT, "schemas", "song_blueprint.schema.json"), encoding="utf-8") as fh:
                jsonschema.validate(enriched, json.load(fh))

    def test_probe_and_full_coverage_active(self):
        from full_coverage_validator import build_probe_plan, validate_full_score_ir
        from score_ir import build_score_ir
        meta = {"bpm": 150.0, "beats_per_bar": 4, "denominator": 4, "valence": 0.0,
                "total_bars": 16, "tonic": "C", "key": "C"}
        hc = {"key": "C", "mode": "ionian", "degree_progression": ["I", "V", "vi", "IV"],
              "section_progressions": {"verse": ["I", "V", "vi", "IV"], "chorus": ["vi", "IV", "I", "V"],
                                       "bridge": ["ii", "V", "I", "vi"], "outro": ["V7", "I"]}}
        tracks = {
            "lead": [{"start": b * 0.5, "length": 0.45, "pitch": 72 + (b % 5), "velocity": 90} for b in range(64)],
            "bass": [{"start": b * 2.0, "length": 1.9, "pitch": 36 + (b % 4), "velocity": 100} for b in range(32)],
            "chords": [{"start": b * 4.0, "length": 3.9, "pitch": 60, "velocity": 70} for b in range(16)],
        }
        sir = build_score_ir(hc, meta, tracks, section_spans=[("verse", 0, 8), ("chorus", 8, 8)])
        fc = validate_full_score_ir(sir, sections=[("verse", 0, 8, "verse"), ("chorus", 8, 8, "chorus")])
        self.assertIn("passed", fc)
        plan = {"beats_per_bar": 4, "sections": [
            {"name": "Refrain", "kind": "chorus", "start_bar": 8, "bars": 8, "active_roles": ["lead", "bass", "chords", "hats"]},
            {"name": "Pont", "kind": "bridge", "start_bar": 12, "bars": 4, "active_roles": ["lead", "choir"]}]}
        probe = build_probe_plan(plan, hall_decay_s=2.4, bpm=150.0)
        self.assertEqual(probe["probe"]["gap_bars"], 2)  # v3.3 : tampon Hall >= 2 mes.
        # NarrativeHarmony : la grille varie par section (pas une boucle 4 mesures unique)
        grid = sir.harmonic_grid
        verse_pcs = {tuple(sorted(s.get("chord_pcs", []))) for s in grid if s.get("chord_idx", 0) < 8}
        chorus_pcs = {tuple(sorted(s.get("chord_pcs", []))) for s in grid if s.get("chord_idx", 0) >= 8}
        self.assertTrue(verse_pcs and chorus_pcs)

    def test_mai_and_wetness_bricks_active(self):
        # MelodicAppealIndex tournoi (0 rendu) + TimbralCohesion sends DRR
        from musicality_metrics import compute_melodic_appeal_index, select_best_motif_variant
        from score_ir import build_score_ir
        from song_builder import ChannelSpec, apply_wetness_sends
        meta = {"bpm": 120.0, "beats_per_bar": 4, "denominator": 4, "valence": 0.0,
                "total_bars": 16, "tonic": "C", "key": "C"}
        hc = {"key": "C", "mode": "aeolian", "degree_progression": ["i", "VI", "III", "VII"]}
        lead = [{"start": b * 1.0, "length": 0.9, "pitch": p, "velocity": 90}
                for b, p in enumerate([72, 74, 76, 79, 81, 79, 76, 74] * 8)]
        sir = build_score_ir(hc, meta, {"lead": lead})
        m = compute_melodic_appeal_index(sir)
        self.assertIn("mai", m) and self.assertGreater(m["mai"], 0.0)
        sel = select_best_motif_variant(sir, [(sir, "base")])
        self.assertIn("selected", sel)
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "samples_catalog.json"), "w", encoding="utf-8") as fh:
                json.dump({"samples": [{"path": "pad/wet.wav", "wetness_score": 0.9}]}, fh)
            ch = ChannelSpec("Pad", "pad", 5, "music", plugin="gl.sampler", sample_file="pad/wet.wav", fx_send=0.18, hall_send=0.22)
            rep = apply_wetness_sends([ch], tmp)
            self.assertLess(ch.hall_send, 0.22)  # sample réverbéré -> send Hall réduit
            self.assertIn("pad", rep)

    @unittest.skipUnless(HAS_GRIDLINE, "binaire gridline requis pour le pipeline complet")
    def test_full_pipeline_gridline(self):
        from compile_song import compile_song
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "song_blueprint.json"), "w", encoding="utf-8") as fh:
                json.dump(_minimal_blueprint(), fh)
            from tests.test_phase7_expressivity_autonomy import _sample_arrangement_plan
            with open(os.path.join(tmp, "arrangement_plan.json"), "w", encoding="utf-8") as fh:
                json.dump(_sample_arrangement_plan(), fh)
            rep = compile_song(tmp, probe=True)
            self.assertTrue(rep.get("ok"))
            self.assertIn("probe", rep)


if __name__ == "__main__":
    unittest.main()
