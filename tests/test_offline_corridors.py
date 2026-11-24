#!/usr/bin/env python3
"""v3.1 REF-04 — corridors spectraux versionnés & repli à 3 niveaux, 100 % hors-ligne.

Vérifie dans un répertoire temporaire sans aucun master externe :
  - `golden_references/spectral_corridors.json` est versionné et couvre les 7 profils
    (bark24 p10/p50/p90 de 24 floats, 8 métriques [p10,p50,p90] ordonnées, provenance) ;
  - `corridor_dead_zone_loss` : 0 dans [P10,P90], monotone et croissante hors corridor ;
  - `evaluate_against_corridors` : score 1.0 à l'intérieur, pénalité monotone dehors,
    `dark_valence_overrides` appliqués si valence < -0.2 ;
  - `compute_spectral_match` : inclut `corridor_score` quand le corridors est présent ;
  - `extract_target_fingerprint` : repli Niveau 2 (corridor versionné) puis Niveau 3
    (baseline psychoacoustique) si le JSON est absent — `reference_tier` toujours renseigné,
    0 crash hors-ligne.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
ANALYZER_DIR = os.path.join(ROOT, "mcp-servers", "audio-analyzer")
for _d in (SCRIPTS_DIR, ANALYZER_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np  # noqa: E402

from bootstrap_golden_refs import (  # noqa: E402
    CORRIDOR_METRICS,
    DEFAULT_CORRIDORS_PATH,
    PROFILES,
    compute_spectral_match,
    corridor_dead_zone_loss,
    evaluate_against_corridors,
    load_spectral_corridor_entry,
)


class TestCorridorsFile(unittest.TestCase):
    def test_versioned_corridors_cover_7_profiles(self):
        self.assertTrue(os.path.isfile(DEFAULT_CORRIDORS_PATH), "spectral_corridors.json doit être versionné")
        with open(DEFAULT_CORRIDORS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data.get("schema_version"), "3.1")
        profiles = data.get("profiles") or {}
        for prof in PROFILES:
            self.assertIn(prof, profiles)
            e = profiles[prof]
            self.assertEqual(e.get("provenance"), "calibrated_psychoacoustic_prior")
            bark = e["bark24"]
            for k in ("p10", "p50", "p90"):
                self.assertEqual(len(bark[k]), 24)
                self.assertAlmostEqual(sum(bark[k]), 1.0, places=3)
            for m in CORRIDOR_METRICS:
                c = e["metrics"][m]
                self.assertLessEqual(c["p10"], c["p50"])
                self.assertLessEqual(c["p50"], c["p90"])
            self.assertIn("dark_valence_overrides", e)


class TestDeadZoneCorridorLoss(unittest.TestCase):
    def test_zero_inside_monotone_outside(self):
        p10, p90 = -12.2, -9.8
        self.assertEqual(corridor_dead_zone_loss(-11.0, p10, p90), 0.0)
        self.assertEqual(corridor_dead_zone_loss(p10, p10, p90), 0.0)
        self.assertEqual(corridor_dead_zone_loss(p90, p10, p90), 0.0)
        d_above_far = corridor_dead_zone_loss(-7.0, p10, p90)
        self.assertGreater(d_above_far, 0.0)
        self.assertGreater(corridor_dead_zone_loss(-13.2, p10, p90), corridor_dead_zone_loss(-12.5, p10, p90))
        self.assertGreater(corridor_dead_zone_loss(-8.0, p10, p90), corridor_dead_zone_loss(-9.5, p10, p90))

    def test_corridor_score_1_inside_and_penalty_outside(self):
        entry = load_spectral_corridor_entry("trap_rap")
        self.assertIsNotNone(entry)
        inside = {m: c["p50"] for m, c in entry["metrics"].items()}
        r_in = evaluate_against_corridors(inside, entry, valence=0.0)
        self.assertEqual(r_in["corridor_score"], 1.0)
        self.assertEqual(r_in["spectral_corridor_loss"], 0.0)
        outside = {m: c["p90"] + 2.0 for m, c in entry["metrics"].items()}
        r_out = evaluate_against_corridors(outside, entry, valence=0.0)
        self.assertLess(r_out["corridor_score"], 1.0)
        self.assertGreater(r_out["spectral_corridor_loss"], 0.0)
        # monotonie : plus on s'éloigne, plus la perte croît
        outside2 = {m: c["p90"] + 6.0 for m, c in entry["metrics"].items()}
        r_out2 = evaluate_against_corridors(outside2, entry, valence=0.0)
        self.assertGreater(r_out2["spectral_corridor_loss"], r_out["spectral_corridor_loss"])

    def test_dark_valence_overrides_applied(self):
        entry = load_spectral_corridor_entry("trap_rap")
        air_p50 = entry["metrics"]["air_share"]["p50"]
        dark_p50 = entry["dark_valence_overrides"]["air_share"]["p50"]
        self.assertLess(dark_p50, air_p50)
        # air = 0.10 : dans le corridor clair (p90=0.099, perte faible) mais bien hors du
        # corridor sombre (p90=0.0572, perte forte) -> l'override sombre est bien appliqué.
        r_dark = evaluate_against_corridors({"air_share": 0.10}, entry, valence=-0.75)
        r_light = evaluate_against_corridors({"air_share": 0.10}, entry, valence=0.0)
        self.assertGreater(r_dark["spectral_corridor_loss"], 0.0)
        self.assertGreater(r_light["spectral_corridor_loss"], 0.0)
        self.assertGreater(r_dark["spectral_corridor_loss"], r_light["spectral_corridor_loss"])


class TestOfflineFallbackChain(unittest.TestCase):
    def test_compute_spectral_match_includes_corridor_score(self):
        sr = 22050
        t = np.arange(int(2.0 * sr)) / sr
        y = 0.3 * np.sin(2 * np.pi * 220.0 * t) + 0.1 * np.sin(2 * np.pi * 440.0 * t)
        r = compute_spectral_match(y, sr, "trap_rap", extra_metrics={"lufs": -11.0, "air_share": 0.055})
        self.assertIn("corridor_score", r)
        self.assertGreaterEqual(r["corridor_score"], 0.0)
        self.assertLessEqual(r["corridor_score"], 1.0)
        self.assertEqual(r.get("reference_corridor_provenance"), "calibrated_psychoacoustic_prior")

    def test_target_fingerprint_tier2_versioned_corridor(self):
        from target_fingerprint import extract_target_fingerprint
        with tempfile.TemporaryDirectory() as td:
            r = extract_target_fingerprint(
                audio_paths=[],
                mix_profile="trap_rap",
                emotion_vector={"valence": -0.75, "tension": 0.4, "energy": 0.7, "depth": 0.5},
                project_dir=td,
            )
        self.assertEqual(r["reference_tier"], 2)
        self.assertEqual(r["reference_provenance"], "calibrated_psychoacoustic_prior")
        self.assertIn("reference_tier", r["multi_scale_contract"])
        self.assertIn("reference_provenance", r["multi_scale_contract"])

    def test_target_fingerprint_tier3_when_json_absent(self):
        from target_fingerprint import extract_target_fingerprint
        with tempfile.TemporaryDirectory() as td:
            r = extract_target_fingerprint(
                audio_paths=[],
                mix_profile="edm_club",
                project_dir=td,
                profiles_path=os.path.join(td, "no_profiles.json"),
                corridors_path=os.path.join(td, "no_corridors.json"),
            )
            self.assertEqual(r["reference_tier"], 3)
            self.assertEqual(r["reference_provenance"], "psychoacoustic_baseline")
            self.assertEqual(r["multi_scale_contract"]["reference_tier"], 3)
            # 0 crash hors-ligne : le contrat est quand même écrit et valide
            self.assertTrue(os.path.isfile(os.path.join(td, "multi_scale_contract.json")))

    def test_corrupt_corridors_falls_back_safely(self):
        with tempfile.TemporaryDirectory() as td:
            bad = os.path.join(td, "corrupt.json")
            with open(bad, "w", encoding="utf-8") as fh:
                fh.write("{not json")
            self.assertIsNone(load_spectral_corridor_entry("trap_rap", bad))
            from target_fingerprint import extract_target_fingerprint
            r = extract_target_fingerprint(
                audio_paths=[], mix_profile="pop_rnb", project_dir=td,
                profiles_path=os.path.join(td, "none.json"), corridors_path=bad,
            )
        self.assertEqual(r["reference_tier"], 3)
        self.assertEqual(r["reference_provenance"], "psychoacoustic_baseline")


if __name__ == "__main__":
    unittest.main()
