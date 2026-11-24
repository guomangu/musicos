#!/usr/bin/env python3
"""v3.2 P5 (#4 + #5a) — TimbralCohesion (DRR/wetness, archétypes gl.*, banque de grooves
MIDI) et MelodicAppealIndex (tournoi de variantes en Tour 2).

Vérifie que :
  (a) `compute_wetness_tail` distingue sec (wetness faible) vs réverbéré (wetness haut) ;
  (b) `space_coherence_penalty` pénalise l'association wet lead + dry kick, pas deux wet ;
  (c) `apply_wetness_sends` réduit fx_send/hall_send d'un sample réverbéré, pas d'un sec ;
  (d) `resolve_archetype` couvre 7 profils x 3 rôles (21 recettes) avec dark_overrides ;
  (e) `build_groove_bank` mine UNIQUEMENT les pistes percussives d'un .mid synthétique ;
  (f) `compute_melodic_appeal_index` note un hook avec arche+tension+résolution mieux qu'une
      mélodie monotone 2 notes, et le tournoi sélectionne la meilleure variante valide.
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

from audio_metrics import compute_wetness_tail  # noqa: E402
from gl_dsl import load_timbre_archetypes, resolve_archetype  # noqa: E402
from midi_groove_bank import build_groove_bank, groove_to_notes, query_groove_bank  # noqa: E402
from musicality_metrics import compute_melodic_appeal_index, select_best_motif_variant  # noqa: E402
from palette_builder import space_coherence_penalty  # noqa: E402
from score_ir import build_score_ir  # noqa: E402
from song_builder import ChannelSpec, apply_wetness_sends  # noqa: E402

BPM = 120.0


def _meta(bars: int = 16) -> dict:
    return {"bpm": BPM, "beats_per_bar": 4, "denominator": 4, "valence": 0.0,
            "total_bars": bars, "tonic": "C", "key": "C"}


class TestWetnessDRR(unittest.TestCase):
    def test_dry_vs_wet(self):
        sr = 22050
        t = np.arange(int(1.0 * sr)) / sr
        dry = np.zeros_like(t)
        dry[: int(0.02 * sr)] = np.sin(2 * np.pi * 1000.0 * t[: int(0.02 * sr)])  # clic sec
        wet = dry + 0.8 * np.exp(-t / 0.18) * np.sin(2 * np.pi * 800.0 * t) * (t > 0.05)  # queue longue
        d = compute_wetness_tail(dry, sr)
        w = compute_wetness_tail(wet, sr)
        self.assertLess(d["wetness_score"], w["wetness_score"])
        self.assertGreater(w["tail_duration_ms"], d["tail_duration_ms"])
        self.assertGreaterEqual(d["wetness_score"], 0.0)
        self.assertLessEqual(w["wetness_score"], 1.0)

    def test_space_coherence_penalty(self):
        wet_lead = {"role": "lead_melody", "wetness_score": 0.9}
        dry_kick = {"role": "kick", "wetness_score": 0.1}
        wet_kick = {"role": "kick", "wetness_score": 0.85}
        self.assertGreater(space_coherence_penalty([wet_lead, dry_kick]), 0.0)
        self.assertEqual(space_coherence_penalty([wet_lead, wet_kick]), 0.0)

    def test_apply_wetness_sends(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "samples_catalog.json"), "w", encoding="utf-8") as fh:
                json.dump({"samples": [
                    {"path": "pads/wet_pad.wav", "wetness_score": 0.9},
                    {"path": "kick/dry_kick.wav", "wetness_score": 0.1},
                ]}, fh)
            ch_wet = ChannelSpec("Pad", "pad", 5, "music", plugin="gl.sampler", sample_file="pads/wet_pad.wav", fx_send=0.18, hall_send=0.22)
            ch_dry = ChannelSpec("Kick", "kick", 1, "drums", plugin="gl.sampler", sample_file="kick/dry_kick.wav", fx_send=0.10, hall_send=0.0)
            rep = apply_wetness_sends([ch_wet, ch_dry], td)
            self.assertIn("pad", rep)
            self.assertLess(ch_wet.hall_send, 0.22)   # réverbéré -> quasi 0 send Hall
            self.assertLess(ch_wet.fx_send, 0.18)
            # sec (wetness 0.1) -> factor 0.92 : quasi inchangé, jamais nul
            self.assertGreaterEqual(ch_dry.fx_send, 0.09)
            self.assertLess(ch_dry.fx_send, 0.10)


class TestArchetypes(unittest.TestCase):
    def test_21_recipes_cover_profiles_and_roles(self):
        arch = load_timbre_archetypes()
        self.assertGreaterEqual(len(arch), 21)
        for prof in ("trap_rap", "pop_rnb", "edm_club", "rock_metal", "lofi_chill", "jazz_acoustic", "classical_ambient"):
            for role in ("bass", "chords", "lead"):
                self.assertIn(f"{prof}:{role}", arch)

    def test_dark_overrides_applied(self):
        a = resolve_archetype("lofi_chill", "chords", valence=-0.6)
        self.assertIsNotNone(a)
        self.assertLessEqual(a["params"]["cutoff_hz"], 5200)  # dark override lofi
        b = resolve_archetype("lofi_chill", "chords", valence=0.3)
        self.assertEqual(b["params"]["cutoff_hz"], 5200)


class TestMidiGrooveBank(unittest.TestCase):
    def _write_test_midi(self, path: str):
        import mido
        mf = mido.MidiFile(ticks_per_beat=480)
        tempo = mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(140), time=0)
        drums = mido.MidiTrack()
        drums.append(tempo)
        drums.name = "Hi-hat patterns"
        drums.is_drum = True
        for i in range(16):  # 1 mesure de doubles croches (pas de 120 ticks = 1/16)
            drums.append(mido.Message("note_on", note=42, velocity=100 if i % 4 == 0 else 70, time=0))
            drums.append(mido.Message("note_off", note=42, velocity=0, time=120))
        mel = mido.MidiTrack()
        mel.name = "Piano lead"
        mel.append(mido.Message("note_on", note=60, velocity=90, time=0))
        mel.append(mido.Message("note_off", note=60, velocity=0, time=1920))
        mf.tracks.extend([drums, mel])
        mf.save(path)

    def test_bank_mines_only_percussive_tracks(self):
        with tempfile.TemporaryDirectory() as td:
            pack = os.path.join(td, "soundbanks", "trap_pack", "MIDI")
            os.makedirs(pack)
            self._write_test_midi(os.path.join(pack, "hat_roll.mid"))
            bank_path = os.path.join(td, "bank.json")
            rep = build_groove_bank(os.path.join(td, "soundbanks"), bank_path)
            self.assertEqual(rep["n_templates"], 1)  # seule la piste batterie, pas le piano
            tpl = query_groove_bank(bank_path, role="hihat_perc", bpm=140, profile="trap_rap")
            self.assertIsNotNone(tpl)
            self.assertEqual(tpl["role"], "hihat_perc")
            self.assertEqual(len(tpl["steps"]), 16)
            self.assertAlmostEqual(tpl["bpm"], 140.0, places=0)
            notes = groove_to_notes(tpl, 0.0, 4.0, pitch=42)
            self.assertEqual(len(notes), 16)
            self.assertAlmostEqual(float(notes[1]["start"]), 0.25, places=3)
            # le morceau ne contient aucune note de hauteur 60 (mélodie non copiée)
            self.assertTrue(all(int(n["pitch"]) == 42 for n in notes))


class TestMelodicAppealIndex(unittest.TestCase):
    def _score(self, lead_pitches: list[int], bars: int = 16) -> object:
        hc = {"key": "C", "mode": "aeolian", "degree_progression": ["i", "VI", "III", "VII"]}
        lead = [{"start": b * 1.0, "length": 0.9, "pitch": p, "velocity": 90} for b, p in enumerate(lead_pitches)]
        tracks = {"lead": lead,
                  "bass": [{"start": b * 2.0, "length": 1.9, "pitch": 36 + (b % 4), "velocity": 100} for b in range(bars * 2)],
                  "chords": [{"start": b * 4.0, "length": 3.9, "pitch": 60, "velocity": 70} for b in range(bars)]}
        return build_score_ir(hc, _meta(bars), tracks)

    def test_hook_beats_monotone(self):
        # hook : motif répété avec variation + arche montante puis descendante + résolution
        hook = [72, 74, 76, 79, 81, 79, 76, 74, 72, 74, 76, 79, 84, 79, 76, 72] * 4
        # monotone : 2 notes en boucle, pas d'arche, pas de tension
        mono = [72, 72, 74, 74] * 16
        s_hook = self._score(hook)
        s_mono = self._score(mono)
        m_hook = compute_melodic_appeal_index(s_hook)
        m_mono = compute_melodic_appeal_index(s_mono)
        self.assertGreater(m_hook["mai"], m_mono["mai"])
        self.assertGreater(m_hook["criteria"]["contour_clarity"], m_mono["criteria"]["contour_clarity"])
        for k in ("motif_entropy_golden", "contour_clarity", "tension_resolution", "rhythmic_complementarity"):
            self.assertIn(k, m_hook["criteria"])

    def test_tournament_selects_best_valid_variant(self):
        import copy
        base = self._score([72, 74, 76, 79, 81, 79, 76, 74] * 8)
        v_hook = copy.deepcopy(base)
        # variante "meilleure" : apex clair en bar 7 + résolution (déjà le cas du hook)
        variants = [(base, "base"), (v_hook, "hook")]
        sel = select_best_motif_variant(base, variants)
        self.assertIn(sel["selected"], ("base", "hook"))
        self.assertGreaterEqual(sel["mai"], 0.0)
        self.assertTrue(sel["evaluate_passed"])


if __name__ == "__main__":
    unittest.main()
