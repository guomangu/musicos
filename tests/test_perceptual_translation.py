#!/usr/bin/env python3
"""v3.3 J4 (#4) — PerceptualTranslation : détection annulation mono (L=-R mid/high),
small-speaker 250 Hz-7 kHz (lead vs harmonie), pic sharpness localisé (P95 vs moyenne),
PLR climax ; seuils dans config/mix_profiles.json ; taste_memory v2 `musical_preferences`
avec prior borné <= 15 %."""
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

from deep_diagnostics import (
    analyze_mono_folddown_loss,
    analyze_small_speaker_translation,
    compute_momentary_sharpness_p95,
    run_perceptual_translation_audit,
)

SR = 22050


def _stereo(sig: np.ndarray) -> np.ndarray:
    return np.stack([sig, sig], axis=1)


class TestMonoFoldDown(unittest.TestCase):
    def test_detects_antiphase_mid_high(self):
        t = np.arange(int(6 * SR)) / SR
        base = 0.3 * np.sin(2 * np.pi * 440.0 * t)          # corrélé L/R
        wide = 0.3 * np.sin(2 * np.pi * 3000.0 * t)         # haut médium en anti-phase L = -R
        stereo = np.stack([base + wide, base - wide], axis=1)
        rep = analyze_mono_folddown_loss(stereo, None, SR)
        self.assertGreater(rep["master"], 2.5)   # perte mono détectée
        self.assertFalse(rep["passed"])

    def test_in_phase_stereo_passes(self):
        t = np.arange(int(6 * SR)) / SR
        sig = 0.3 * np.sin(2 * np.pi * 440.0 * t) + 0.1 * np.sin(2 * np.pi * 3000.0 * t)
        rep = analyze_mono_folddown_loss(_stereo(sig), None, SR)
        self.assertLessEqual(rep["master"], 2.5)
        self.assertTrue(rep["passed"])

    def test_stem_lead_loss_detected(self):
        t = np.arange(int(6 * SR)) / SR
        wide = 0.3 * np.sin(2 * np.pi * 2500.0 * t)
        lead = np.stack([wide, -wide], axis=1)  # lead 100 % anti-phase
        master = _stereo(0.2 * np.sin(2 * np.pi * 220.0 * t))
        rep = analyze_mono_folddown_loss(master, {"lead_melody": (lead, SR)}, SR)
        self.assertIsNotNone(rep["lead"])
        self.assertGreater(rep["lead"], 3.0)
        self.assertFalse(rep["passed"])


class TestSmallSpeaker(unittest.TestCase):
    def test_lead_dominates_band(self):
        t = np.arange(int(4 * SR)) / SR
        lead = 0.4 * np.sin(2 * np.pi * 1500.0 * t)  # énergie dans 250 Hz-7 kHz
        chords = 0.05 * np.sin(2 * np.pi * 300.0 * t)  # grave, quasi rien dans la bande
        stems = {"lead_melody": (lead, SR), "chords_pad": (chords, SR)}
        rep = analyze_small_speaker_translation(stems, SR, min_lead_vs_harmony_db=2.0)
        self.assertIsNotNone(rep["small_speaker_lead_vs_harmony_db"])
        self.assertGreaterEqual(rep["small_speaker_lead_vs_harmony_db"], 2.0)
        self.assertTrue(rep["passed"])

    def test_weak_lead_fails(self):
        t = np.arange(int(4 * SR)) / SR
        lead = 0.02 * np.sin(2 * np.pi * 1500.0 * t)
        chords = 0.5 * np.sin(2 * np.pi * 2000.0 * t)
        stems = {"lead_melody": (lead, SR), "chords_pad": (chords, SR)}
        rep = analyze_small_speaker_translation(stems, SR, min_lead_vs_harmony_db=2.0)
        self.assertLess(rep["small_speaker_lead_vs_harmony_db"], 2.0)
        self.assertFalse(rep["passed"])


class TestSharpnessP95(unittest.TestCase):
    def test_local_spike_raises_p95_above_mean(self):
        t = np.arange(int(12 * SR)) / SR
        base = 0.2 * np.sin(2 * np.pi * 500.0 * t)  # doux partout
        spike = base.copy()
        a = int(8 * SR)
        spike[a:a + int(2 * SR)] += 0.25 * np.sin(2 * np.pi * 5000.0 * t[: int(2 * SR)])  # pic 5 kHz sur 2 s
        r_base = compute_momentary_sharpness_p95(base, SR)
        r_spike = compute_momentary_sharpness_p95(spike, SR)
        self.assertIsNotNone(r_spike["sharpness_p95"])
        self.assertGreater(r_spike["sharpness_p95"], r_base["sharpness_p95"])
        self.assertGreater(r_spike["windows"], 0)


class TestPerceptualAuditContract(unittest.TestCase):
    def test_contract_keys_and_thresholds(self):
        t = np.arange(int(8 * SR)) / SR
        master = _stereo(0.2 * np.sin(2 * np.pi * 440.0 * t))
        stems = {"lead_melody": (_stereo(0.3 * np.sin(2 * np.pi * 1500.0 * t)), SR),
                 "chords_pad": (_stereo(0.1 * np.sin(2 * np.pi * 300.0 * t)), SR)}
        rep = run_perceptual_translation_audit(master, stems, SR, "trap_rap",
                                               sections=[{"name": "Refrain", "energy": 0.9}, {"name": "Intro", "energy": 0.3}])
        for k in ("passed", "mono_folddown_loss_db", "small_speaker_lead_vs_harmony_db",
                  "sharpness_p95", "min_short_term_plr_db", "climax_section", "recommended_actions"):
            self.assertIn(k, rep)
        self.assertEqual(rep["climax_section"], "Refrain")
        codes = {a["code"] for a in rep["recommended_actions"]}
        self.assertTrue(codes <= {"REDUCE_STEREO_CANCELLATION", "RELAX_CLIMAX_LIMITING", "TAME_PEAK_SHARPNESS", "BOOST_LEAD_SMALL_SPEAKER"})

    def test_thresholds_in_config(self):
        cfg = json.load(open(os.path.join(ROOT, "config", "mix_profiles.json"), encoding="utf-8"))
        gi = cfg["global_invariants"]
        self.assertEqual(gi["mono_folddown_loss_max_db"], 2.5)
        self.assertEqual(gi["mono_folddown_lead_max_db"], 3.0)
        self.assertEqual(gi["small_speaker_lead_vs_harmony_min_db"], 2.0)
        for prof, p in cfg["profiles"].items():
            self.assertIn("sharpness_p95_max", p)
            self.assertAlmostEqual(p["sharpness_p95_max"], max(p["sharpness_max"], p.get("sharpness_max_dark", 0)) + 0.25, places=2)
            self.assertIn("min_short_term_plr_db", p)


class TestMusicalTasteMemory(unittest.TestCase):
    def test_record_and_query_musical_prefs_bounded(self):
        from taste_memory import query_musical_taste_prior, record_validated_taste
        with tempfile.TemporaryDirectory() as tmp:
            taste_path = os.path.join(tmp, "user_taste_profile.json")
            rep = record_validated_taste(
                tmp, taste_path=taste_path,
                extra_metrics={"pq_score": 8.6, "mix_profile": "trap_rap", "bpm": 140.0,
                               "emotion_vector": {"valence": -0.4, "tension": 0.5, "energy": 0.7, "depth": 0.6},
                               "bark24_mean": [1.0 / 24] * 24},
                musical_prefs={"mode": "aeolian", "progression_id": "i-bVI-III-VII",
                               "comping_styles": ["rhythmic_comp"], "timbre_archetypes": ["trap_rap:lead"],
                               "melodic_appeal_index": 0.72, "swing": 0.08, "tempo_bpm": 140,
                               "user_tags": ["warm", "punchy_808"]},
            )
            self.assertTrue(rep["recorded"])
            self.assertIn("musical_preferences", rep["entry"])
            prior = query_musical_taste_prior({"valence": -0.4, "tension": 0.5, "energy": 0.7, "depth": 0.6},
                                              "trap_rap", taste_path=taste_path)
            self.assertIsNotNone(prior)
            self.assertLessEqual(prior["bonus_max_pct"], 15.0)
            self.assertEqual(prior["mode"], "aeolian")
            self.assertAlmostEqual(prior["melodic_appeal_index_target"], 0.72, places=3)
            tags = [t for t, _cnt in prior["user_tags"]] if prior["user_tags"] and isinstance(prior["user_tags"][0], tuple) else prior["user_tags"]
            self.assertIn("punchy_808", tags)

    def test_empty_store_returns_none(self):
        from taste_memory import query_musical_taste_prior
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(query_musical_taste_prior(None, "trap_rap", taste_path=os.path.join(tmp, "none.json")))


if __name__ == "__main__":
    unittest.main()
