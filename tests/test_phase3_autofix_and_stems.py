#!/usr/bin/env python3
"""
Phase 3 unit & integration tests (Axe 3 §3.2 & Axe 4):
  - `analyze_stem_hierarchy`: Lead vs Harmony 800Hz-3.5kHz, Kick/Bass <150Hz transient overlap, Pink slope 60Hz-12kHz.
  - `bootstrap_golden_refs` / `spectral_match_score`: 24-Bark + pink slope matching (0 PyTorch).
  - `render_and_audit.py --auto-fix --max-passes 2`: deliberately detuned `.gridline` (low LUFS + 1/3-oct resonance)
    repairs automatically via `GridlineMCP` + `gl_dsl` and passes Quality Gate in <= 2 passes,
    writing `renders/autofix_history.json`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "mcp-servers", "audio-analyzer")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bootstrap_golden_refs import DEFAULT_PROFILES_PATH, compute_spectral_match
from deep_diagnostics import analyze_stem_hierarchy, compute_pink_slope_db_per_oct
from gl_dsl import eq_band, eq_params
from studio_lib import GridlineMCP


class TestPhase3AutofixAndStems(unittest.TestCase):
    def test_pink_slope_and_synthetic_stem_hierarchy(self) -> None:
        sr = 44100
        dur = 4.0
        n = int(sr * dur)
        t = np.arange(n, dtype=np.float64) / sr

        # 1) Exact pink noise (1/f power -> slope ~ -3.01 dB/oct)
        rng = np.random.default_rng(42)
        white = rng.standard_normal(n)
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        scale = np.ones_like(freqs)
        scale[freqs > 0] = 1.0 / np.sqrt(freqs[freqs > 0])
        pink = np.fft.irfft(spec * scale, n=n)
        pink /= max(1e-9, float(np.max(np.abs(pink)))) * 1.4
        slope = compute_pink_slope_db_per_oct(pink, sr)
        self.assertGreaterEqual(slope, -3.6)
        self.assertLessEqual(slope, -2.4)

        # 2) Synthetic separated stems:
        #    - Lead active 50% of time with strong 1.5 kHz sine (+10 dB over Harmony 1.2 kHz sine)
        #    - Kick 80 Hz burst every 0.5 s; Bass 55 Hz either DUCKED (neg corr) or COLLIDING (pos corr)
        with tempfile.TemporaryDirectory() as tmp:
            stems_good = os.path.join(tmp, "good")
            os.makedirs(stems_good)

            env_lead = np.where((t % 1.0) < 0.5, 1.0, 0.0)
            lead = 0.25 * env_lead * np.sin(2 * np.pi * 1500.0 * t)
            harm = 0.04 * np.sin(2 * np.pi * 1200.0 * t) + 0.08 * np.sin(2 * np.pi * 300.0 * t)

            kick = np.zeros(n, dtype=np.float64)
            bass_ducked = np.full(n, 0.22, dtype=np.float64) * np.sin(2 * np.pi * 55.0 * t)
            for beat_t in np.arange(0.0, dur, 0.5):
                i0 = int(beat_t * sr)
                l_burst = min(n - i0, int(0.09 * sr))
                tt = np.arange(l_burst, dtype=np.float64) / sr
                kick[i0 : i0 + l_burst] = 0.55 * np.exp(-tt / 0.025) * np.sin(2 * np.pi * 65.0 * tt)
                duck_len = min(n - i0, int(0.10 * sr))
                bass_ducked[i0 : i0 + duck_len] *= 0.05

            master_mix = 0.45 * (lead + harm + kick + bass_ducked) + 0.20 * pink
            sf.write(os.path.join(stems_good, "track.wav"), np.column_stack([master_mix, master_mix]), sr)
            sf.write(os.path.join(stems_good, "track - 01 Kick.wav"), np.column_stack([kick, kick]), sr)
            sf.write(os.path.join(stems_good, "track - 18 BUS Bass.wav"), np.column_stack([bass_ducked, bass_ducked]), sr)
            sf.write(os.path.join(stems_good, "track - 06 Chords Pad.wav"), np.column_stack([harm, harm]), sr)
            sf.write(os.path.join(stems_good, "track - 07 Lead Keys.wav"), np.column_stack([lead, lead]), sr)

            res = analyze_stem_hierarchy(stems_dir=stems_good, has_drums=True)
            m = res["metrics"]
            self.assertGreater(m["lead_vs_harmony_db"], 6.0)
            self.assertLess(m["kick_bass_transient_overlap"], 0.1)
            self.assertTrue(res["checks"]["lead_hierarchy_ok"])
            self.assertTrue(res["checks"]["kick_bass_separation_ok"])

            # Verify spectral_match_score is fast & in [0, 1]
            sm = compute_spectral_match(master_mix, sr, profile="trap_rap", profiles_path=DEFAULT_PROFILES_PATH)
            self.assertGreaterEqual(sm["spectral_match_score"], 0.0)
            self.assertLessEqual(sm["spectral_match_score"], 1.0)
            if os.path.isfile(DEFAULT_PROFILES_PATH):
                self.assertEqual(sm["reference_source"], "measured")
            else:
                # golden_references/ est gitignore : profil analytique par défaut sur un clone frais
                self.assertEqual(sm["reference_source"], "analytic_default")

    @unittest.skipUnless(
        os.path.isdir(os.path.join(ROOT, "projects", "validation_fm_dark", "renders", "stems_bus")),
        "projects/*/renders/ est gitignore : absent sur un clone frais",
    )
    def test_stem_hierarchy_on_real_validation_stems(self) -> None:
        real_dir = os.path.join(ROOT, "projects", "validation_fm_dark", "renders", "stems_bus")
        self.assertTrue(os.path.isdir(real_dir))
        res = analyze_stem_hierarchy(stems_dir=real_dir, has_drums=True)
        self.assertIn("lead_vs_harmony_db", res["metrics"])
        self.assertIn("kick_bass_transient_overlap", res["metrics"])
        self.assertLess(res["metrics"]["kick_bass_transient_overlap"], 0.0)

    @unittest.skipUnless(shutil.which("gridline"), "binaire gridline requis")
    def test_render_and_audit_autofix_fixes_broken_mix_in_max_2_passes(self) -> None:
        src_dir = os.path.join(ROOT, "projects", "validation_fm_dark")
        src_proj = os.path.join(src_dir, "projet.gridline")
        src_bp = os.path.join(src_dir, "song_blueprint.json")
        self.assertTrue(os.path.isfile(src_proj))

        with tempfile.TemporaryDirectory() as tmp:
            work_proj = os.path.join(tmp, "projet.gridline")
            work_bp = os.path.join(tmp, "song_blueprint.json")
            renders_dir = os.path.join(tmp, "renders")
            os.makedirs(renders_dir, exist_ok=True)
            shutil.copy2(src_proj, work_proj)
            if os.path.isfile(src_bp):
                shutil.copy2(src_bp, work_bp)

            # Deliberately break the mix on chorus range 5-12:
            # 1) Master gl.limiter Gain too low (-3.0 dB -> well below -11 LUFS)
            # 2) BUS Music slot 2 Band 4 huge narrow resonance spike +16.0 dB @ 1600 Hz (width=0.22)
            mcp = GridlineMCP(work_proj, tmp)
            try:
                changes = [
                    ("mixer/0/slot/3/Gain", -3.0),
                    *eq_params(19, [eq_band(4, "peak", 1600.0, 16.0, q=2.8)], slot=2),
                ]
                mcp.set_params(changes)
                mcp.call("save_project", {})
            finally:
                mcp.close()

            out_wav = os.path.join(renders_dir, "autofix_test.wav")
            cmd = [
                sys.executable,
                os.path.join(ROOT, "scripts", "render_and_audit.py"),
                work_proj,
                "-o",
                out_wav,
                "--range",
                "5-12",
                "--profile",
                "trap_rap",
                "--bpm",
                "132",
                "--no-neural",
                "--auto-fix",
                "--max-passes",
                "2",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
            self.assertEqual(proc.returncode, 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
            summary = json.loads(proc.stdout)

            hist_path = os.path.join(renders_dir, "autofix_history.json")
            self.assertTrue(os.path.isfile(hist_path), f"Missing autofix_history.json in {renders_dir}")
            with open(hist_path, encoding="utf-8") as fh:
                hist = json.load(fh)

            self.assertGreaterEqual(len(hist["passes"]), 2, "Expected initial failed pass + at least 1 autofix pass")
            self.assertFalse(hist["passes"][0]["quality_gate"]["passed"], "Initial broken mix must fail Quality Gate")
            self.assertTrue(summary["quality_gate"]["passed"], f"Auto-fix failed to pass in <=2 passes:\n{json.dumps(hist, indent=2)}")
            self.assertLessEqual(summary["autofix"]["passes_run"], 2)


if __name__ == "__main__":
    unittest.main()
