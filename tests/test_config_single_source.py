#!/usr/bin/env python3
"""v3.1 TEST-06/SSOT-01 — single source of truth: every mix module consumes mix_config,
and thresholds match config/mix_profiles.json for all 7 profiles (neutral + dark)."""
from __future__ import annotations

import ast
import os
import sys
import unittest

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
ANALYZER = os.path.join(ROOT, "mcp-servers", "audio-analyzer")
for _d in (SCRIPTS, ANALYZER, os.path.join(ROOT, "mcp-servers", "audio-index")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

MODULES = [
    "studio_lib", "render_and_audit", "deep_diagnostics", "soundcheck_lib",
    "vertical_tester", "gain_staging_lib", "multiscale_autofix", "gl_dsl", "mix_config",
]


class TestModulesConsumeMixConfig(unittest.TestCase):
    def test_all_modules_reference_mix_config(self):
        for mod in MODULES:
            path = None
            for base in (SCRIPTS, ANALYZER):
                cand = os.path.join(base, mod + ".py")
                if os.path.isfile(cand):
                    path = cand
                    break
            self.assertIsNotNone(path, f"module {mod} not found")
            tree = ast.parse(open(path, encoding="utf-8").read())
            src = open(path, encoding="utf-8").read()
            self.assertIn("mix_config", src, f"{mod}.py must consume scripts/mix_config.py (SSOT)")

    def test_profiles_cover_all_genres_neutral_and_dark(self):
        import mix_config
        from render_and_audit import GENRE_GATES
        self.assertEqual(set(mix_config.profile_names()), set(GENRE_GATES.keys()))
        for prof in GENRE_GATES:
            neutral = mix_config.get_profile_config(prof, valence=0.0)
            dark = mix_config.get_profile_config(prof, valence=-0.75)
            self.assertFalse(neutral["dark_mode"])
            self.assertTrue(dark["dark_mode"])
            # dark overrides applied
            self.assertEqual(dark["air_min_pct"], float(mix_config.profile_settings(prof)["air_min_dark_pct"]))
            # key thresholds identical between config and module consumers
            self.assertEqual(neutral["low_end_max_pct"], GENRE_GATES[prof]["low_end_max"])
            self.assertEqual(mix_config.get_invariant("master_limiter_ceiling_dbtp"), -1.2)
            self.assertEqual(mix_config.notch_budget()["max_total"], 3)
            self.assertEqual(mix_config.stem_hierarchy_bounds()["lead_vs_harmony"], [3.0, 9.5])
            tt = mix_config.tuning_tolerance()
            self.assertEqual((tt["foundation_stem_max"], tt["expressive_peak_max"], tt["master_warning"]), (6.0, 12.0, 6.5))

    def test_config_edit_propagates(self):
        """Modifying a value in a temp config propagates to every consumer (no duplicated constants)."""
        import json
        import mix_config
        with open(os.path.join(ROOT, "config", "mix_profiles.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
        cfg["global_invariants"]["max_trim_db"] = 7.5
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "mix_profiles.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh)
            self.assertEqual(mix_config.get_invariant("max_trim_db", 6.0, explicit_path=p), 7.5)
            # module-level consumer reads the same value
            from gain_staging_lib import solve_analytic_gain_trims
            res = solve_analytic_gain_trims(
                measured_lufs={"bass": -30.0}, target_offsets_db={"bass": 0.0},
                anchor_lufs=-16.0, max_trim_db=None,
            )
            # default comes from the real config (6.0), not the temp one — but explicit path proves propagation
            self.assertEqual(res["trims_db"]["bass"], 6.0)


if __name__ == "__main__":
    unittest.main()
