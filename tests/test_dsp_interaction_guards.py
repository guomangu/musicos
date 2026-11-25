#!/usr/bin/env python3
"""v3.3 #3 (J1) — DSPInteractionGuards : 0 contamination Hall dans la sonde --probe
(gap dynamique >= 2 mesures), remap des automations, 0 flam ritardando sur boucles audio
(mode symbolique + loop_fade_beat), 0 faux positif de justesse sur glide 808 (masques
steady-state), détection du pompage Climax (min_short_term_plr_db + seuils profil)."""
from __future__ import annotations

import json
import os
import sys
import unittest

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "mcp-servers", "audio-analyzer"), os.path.join(ROOT, "mcp-servers", "audio-index")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np

from dsp_interaction_guards import (
    compute_probe_gap_bars,
    extract_steady_state_masks,
    plan_safe_ritardando,
    remap_automations_for_probe,
)


class TestProbeGapBars(unittest.TestCase):
    def test_formula_and_floor(self):
        # 150 BPM, 4/4, Hall decay 2.4 s -> ceil(2.4*1.25*150/240)=2
        self.assertEqual(compute_probe_gap_bars(2.4, 150.0, 4), 2)
        # decay 1.6 s -> ceil(1.25)=2 (1 mesure = 1.6 s à 150 BPM ne suffit pas)
        self.assertEqual(compute_probe_gap_bars(1.6, 150.0, 4), 2)
        # gros decay à haut BPM -> > 2
        self.assertGreaterEqual(compute_probe_gap_bars(4.0, 200.0, 4), 5)
        # plancher absolu 2 mesures même decay nul
        self.assertEqual(compute_probe_gap_bars(0.0, 120.0, 4), 2)

    def test_probe_plan_uses_dynamic_gap(self):
        from full_coverage_validator import build_probe_plan
        plan = {
            "beats_per_bar": 4,
            "sections": [
                {"name": "Refrain", "kind": "chorus", "start_bar": 8, "bars": 8, "active_roles": ["lead", "bass", "chords", "hats"]},
                {"name": "Pont", "kind": "bridge", "start_bar": 16, "bars": 4, "active_roles": ["lead", "bass", "chords", "vocal_choir"]},
            ],
        }
        probe = build_probe_plan(plan, hall_decay_s=2.4, bpm=150.0)
        meta = probe["probe"]
        self.assertEqual(meta["gap_bars"], 2)  # et pas 1 (contamination Hall)
        self.assertEqual(meta["total_bars"], meta["core_bars"] + 2 + meta["probe_bars"])
        secs = probe["sections"]
        self.assertEqual([s["name"] for s in secs], ["Refrain", "probe_gap", "Pont"])
        self.assertEqual(secs[1]["bars"], 2)
        self.assertEqual(secs[2]["start_bar"], secs[0]["bars"] + 2)
        # compatibilité : sans hall_decay_s/bpm -> gap 1 (comportement v3.2)
        probe_legacy = build_probe_plan(plan)
        self.assertEqual(probe_legacy["probe"]["gap_bars"], 1)

    def test_automations_remapped(self):
        from full_coverage_validator import build_probe_plan
        plan = {
            "beats_per_bar": 4,
            "sections": [
                {"name": "Refrain", "kind": "chorus", "start_bar": 8, "bars": 8, "active_roles": ["lead"]},
                {"name": "Pont", "kind": "bridge", "start_bar": 16, "bars": 4, "active_roles": ["lead", "choir"]},
            ],
            "automations": [
                {"beat": 34.0, "param": "lpf", "value": 8000},   # noyau (start_bar 8 -> beats 32..64)
                {"beat": 66.0, "param": "hall_send", "value": 0.3},  # pont (start_bar 16 -> beats 64..80)
            ],
        }
        probe = build_probe_plan(plan, hall_decay_s=2.4, bpm=150.0)
        autos = probe["automations"]
        self.assertEqual(probe["probe"]["automations_remapped"], 2)
        by_param = {a["param"]: a for a in autos}
        # noyau : start_bar 8 -> src beats [32,64) remappées depuis 0 ; beat 34 -> 2.0
        self.assertAlmostEqual(by_param["lpf"]["beat"], 2.0, places=3)
        self.assertTrue(by_param["lpf"]["_remapped"])
        # pont : src [64,80) -> dst (8 core + 2 gap) * 4 = 40 ; beat 66 -> 42.0
        self.assertAlmostEqual(by_param["hall_send"]["beat"], 42.0, places=3)
        self.assertTrue(by_param["hall_send"]["_remapped"])

    def test_remap_standalone(self):
        out = remap_automations_for_probe(
            [{"beat": 10.0}, {"beat": 40.0}, {"beat": 100.0}, {"no_beat": True}],
            [(0.0, 32.0, 0.0), (32.0, 48.0, 40.0)],
        )
        self.assertAlmostEqual(out[0]["beat"], 10.0)
        self.assertTrue(out[0]["_remapped"])
        # beat 40 dans la 2e plage [32,48) -> dst 40 + (40-32) = 48
        self.assertAlmostEqual(out[1]["beat"], 48.0)
        self.assertTrue(out[1]["_remapped"])
        # beat 100 hors de toute plage : conservé, non remappé
        self.assertAlmostEqual(out[2]["beat"], 100.0)
        self.assertFalse(out[2]["_remapped"])
        self.assertFalse(out[3]["_remapped"])


class TestSafeRitardando(unittest.TestCase):
    def _sir(self, tracks: dict[str, list[dict]]) -> dict:
        return {"tracks": {k: {"role": k, "notes": v} for k, v in tracks.items()}}

    def test_loops_force_symbolic_stretch(self):
        sir = self._sir({
            "bass": [{"start": 0.0, "duration_beats": 2.0}, {"start": 60.0, "duration_beats": 2.0}],
            "melodic_loop": [{"start": 60.0, "duration_beats": 4.0}],
        })
        plan = plan_safe_ritardando(sir, [{"track": "melodic_loop", "is_audio_loop": True}], 0.06, bars=2)
        self.assertEqual(plan["mode"], "symbolic_stretch")
        self.assertEqual(plan["loop_fade_beat"], plan["window_beats"][0])  # fade avant le ritardando
        self.assertIsNone(plan["delay_mute_beat"])
        self.assertIn("melodic_loop", plan["loops_detected"])

    def test_no_loops_allows_tempo_ramp_with_delay_mute(self):
        sir = self._sir({"bass": [{"start": 0.0, "duration_beats": 2.0}, {"start": 60.0, "duration_beats": 2.0}]})
        plan = plan_safe_ritardando(sir, [], 0.06, bars=2)
        self.assertEqual(plan["mode"], "tempo_ramp")
        self.assertAlmostEqual(plan["delay_mute_beat"], plan["window_beats"][0] - 1.0, places=3)
        self.assertIsNone(plan["loop_fade_beat"])

    def test_apply_outro_ritardando_reports_plan(self):
        from transitions_lib import apply_outro_ritardando
        tracks = {
            "bass": [{"start": b * 2.0, "length": 1.9} for b in range(32)],
            "melodic_loop": [{"start": 60.0, "length": 3.9}],
        }
        rep = apply_outro_ritardando(tracks, 120.0, bars=2, amount=0.06,
                                     tracks_meta=[{"track": "melodic_loop", "is_audio_loop": True}])
        self.assertGreater(rep["applied"], 0)
        self.assertIsNotNone(rep.get("plan"))
        self.assertEqual(rep["plan"]["mode"], "symbolic_stretch")


class TestSteadyStateMasks(unittest.TestCase):
    def test_glide_and_vibrato_windows_excluded(self):
        # piste lead : note normale [0,2) beats, note glide [2,4) avec glide_ms=60, note vibrato [4,6)
        sir = {"tracks": {"lead": {"role": "lead", "notes": [
            {"start": 0.0, "duration_beats": 2.0, "pitch": 72},
            {"start": 2.0, "duration_beats": 2.0, "pitch": 74, "expression": {"articulation": "glide_to_next", "glide_ms": 60}},
            {"start": 4.0, "duration_beats": 2.0, "pitch": 76, "expression": {"vibrato_delay_ms": 220}},
        ]}}}
        masks = extract_steady_state_masks(sir, 120.0)  # beat = 0.5 s
        lead = masks["lead"]
        # note glide [2,4) beats, glide_ms=60 -> g=0.12 beat ; fenêtre exclue en beats
        # (4 - 0.12 - 0.015, 4 + 0.025) = (3.865, 4.025) -> secondes (1.9325, 2.0125)
        glide_lo, glide_hi = 1.9325, 2.0125
        flat = []
        for a, b in lead:
            flat.extend(np.arange(a, b, 0.005))
        flat = np.array(flat)
        self.assertTrue(np.all((flat <= glide_lo + 1e-6) | (flat >= glide_hi - 1e-6)))
        # attaque vibrato de la note [4,6) beats : [2.0, 2.22] s exclue aussi
        self.assertTrue(np.all((flat <= 2.0 + 1e-6) | (flat >= 2.22 - 1e-6)))
        # le reste est couvert (note normale 0..1 s comprise)
        self.assertTrue(np.any((flat >= 0.0) & (flat < 0.99)))

    def test_masks_feed_evaluator(self):
        from stem_matrix_evaluator import evaluate_barwise_harmonic_alignment
        sr = 22050
        beat_s = 0.5
        # 2 segments de 4 beats ; stem lead avec un "trou" (glide) au segment 2
        t = np.arange(int(8 * beat_s * sr)) / sr
        lead = np.zeros_like(t)
        lead[: int(4 * beat_s * sr)] = 0.3 * np.sin(2 * np.pi * 440.0 * t[: int(4 * beat_s * sr)])
        # segment 2 : silence (fenêtre de glide exclue) -> aucun sample de biais
        grid = [
            {"chord_idx": 0, "start_beat": 0.0, "end_beat": 4.0, "root_pc": 0, "bass_pc": 0, "chord_pcs": [0, 4, 7], "scale_pcs": [0, 2, 4, 5, 7, 9, 11], "modal_tension_pcs": []},
            {"chord_idx": 1, "start_beat": 4.0, "end_beat": 8.0, "root_pc": 0, "bass_pc": 0, "chord_pcs": [0, 4, 7], "scale_pcs": [0, 2, 4, 5, 7, 9, 11], "modal_tension_pcs": []},
        ]
        stems = {"lead_melody": (lead, sr)}
        masks = {"lead_melody": [(0.05, 1.95)]}  # steady uniquement dans le segment 1
        rep = evaluate_barwise_harmonic_alignment(stems, grid, 120.0, steady_masks=masks)
        self.assertIn("glide_windows_excluded", rep)
        self.assertEqual(rep["glide_windows_excluded"], 1)  # segment 2 entièrement masqué


class TestShortTermPLR(unittest.TestCase):
    def test_detects_climax_pumping(self):
        from deep_diagnostics import compute_short_term_plr
        sr = 22050
        t = np.arange(int(12 * sr)) / sr
        # 12 s : 4 s doux + 4 s très fort mais sans clip (limiteur écrasé) + 4 s doux
        sig = np.concatenate([
            0.05 * np.sin(2 * np.pi * 220 * t[: int(4 * sr)]),
            0.9 * np.sin(2 * np.pi * 220 * t[: int(4 * sr)]),
            0.05 * np.sin(2 * np.pi * 220 * t[: int(4 * sr)]),
        ])
        rep = compute_short_term_plr(sig, sr, window_s=3.0, hop_s=1.0)
        self.assertIsNotNone(rep["min_short_term_plr_db"])
        # la fenêtre la plus forte a un PLR écrasé (crête proche du loudness) vs début doux
        self.assertLess(rep["min_short_term_plr_db"], 14.0)

    def test_thresholds_in_config(self):
        cfg = json.load(open(os.path.join(ROOT, "config", "mix_profiles.json"), encoding="utf-8"))
        self.assertIn("min_short_term_plr_db", cfg["global_invariants"])
        expected = {"edm_club": 7.0, "trap_rap": 7.5, "pop_rnb": 7.5, "rock_metal": 7.5,
                    "lofi_chill": 9.5, "jazz_acoustic": 9.5, "classical_ambient": 9.5}
        for prof, v in expected.items():
            self.assertEqual(cfg["profiles"][prof]["min_short_term_plr_db"], v)


if __name__ == "__main__":
    unittest.main()
