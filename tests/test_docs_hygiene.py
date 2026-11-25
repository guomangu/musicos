#!/usr/bin/env python3
"""v3.0 regression tests — P2 doc hygiene: runbook <= 6000 chars, zero hardcoded /home/gamo paths."""
from __future__ import annotations

import os
import unittest

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDocsHygiene(unittest.TestCase):
    def test_agent_runbook_under_6000_chars(self):
        path = os.path.join(ROOT, "docs", "AGENT_RUNBOOK.md")
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertLessEqual(len(content), 6000, f"AGENT_RUNBOOK.md = {len(content)} cars > 6000")

    def test_no_hardcoded_home_paths(self):
        bad = []
        for sub in ("docs", "scripts", "schemas"):
            base = os.path.join(ROOT, sub)
            for root, dirnames, fns in os.walk(base):
                dirnames[:] = [d for d in dirnames if d != "__pycache__"]
                for fn in fns:
                    if fn.endswith((".pyc", ".png", ".wav", ".npz", ".db")):
                        continue
                    p = os.path.join(root, fn)
                    try:
                        with open(p, encoding="utf-8", errors="ignore") as fh:
                            if "/home/gamo" in fh.read():
                                bad.append(p)
                    except Exception:
                        continue
        self.assertEqual(bad, [], f"chemins /home/gamo encore présents: {bad}")

    def test_reference_archive_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(ROOT, "docs", "REFERENCE_ARCHIVE.md")))


if __name__ == "__main__":
    unittest.main()


class TestDocsCoherenceCI(unittest.TestCase):
    """v3.2 (#5b) — 3 assertions CI bloquantes de cohérence documentaire."""

    def _docs_files(self):
        base = os.path.join(ROOT, "docs")
        return [os.path.join(base, fn) for fn in sorted(os.listdir(base)) if fn.endswith(".md")]

    def test_no_broken_internal_md_links(self):
        import re as _re
        broken = []
        link_re = _re.compile(r"\]\(([^)\s]+\.md)(?:#[^)]*)?\)")
        for p in self._docs_files():
            with open(p, encoding="utf-8") as fh:
                text = fh.read()
            for m in link_re.finditer(text):
                target = m.group(1)
                if target.startswith(("http://", "https://")):
                    continue
                resolved = os.path.normpath(os.path.join(os.path.dirname(p), target))
                if not os.path.isfile(resolved):
                    broken.append(f"{os.path.basename(p)} -> {target}")
        # liens depuis la racine (README.md) vers docs/
        root_readme = os.path.join(ROOT, "README.md")
        if os.path.isfile(root_readme):
            with open(root_readme, encoding="utf-8") as fh:
                text = fh.read()
            for m in link_re.finditer(text):
                target = m.group(1)
                if target.startswith(("http://", "https://")):
                    continue
                resolved = os.path.normpath(os.path.join(ROOT, target))
                if target.endswith(".md") and not os.path.isfile(resolved):
                    broken.append(f"README.md -> {target}")
        self.assertEqual(broken, [], f"liens Markdown internes cassés : {broken}")

    def test_all_scripts_indexed_in_sommaire(self):
        sommaire_path = os.path.join(ROOT, "docs", "00_SOMMAIRE.md")
        with open(sommaire_path, encoding="utf-8") as fh:
            sommaire = fh.read()
        scripts_dir = os.path.join(ROOT, "scripts")
        missing = []
        for fn in sorted(os.listdir(scripts_dir)):
            if fn.endswith(".py"):
                if fn not in sommaire:
                    missing.append(fn)
        self.assertEqual(missing, [], f"scripts non indexés dans 00_SOMMAIRE.md : {missing}")

    def test_no_numeric_divergence_vs_mix_config(self):
        import json as _json
        cfg_path = os.path.join(ROOT, "config", "mix_profiles.json")
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = _json.load(fh)
        inv = cfg.get("invariants", cfg)
        ceiling = float(inv.get("master_limiter_ceiling_dbtp", -1.2))

        def norm(text: str) -> str:
            return text.replace(",", ".").lower()

        normative_files = [
            os.path.join(ROOT, "docs", "AGENT_RUNBOOK.md"),
            os.path.join(ROOT, "docs", "05_MIXAGE_QUALITY_GATE_ET_APPRENTISSAGE.md"),
            os.path.join(ROOT, "docs", "01_SYSTEME_STACK_ET_GUIDE.md"),
            os.path.join(ROOT, "docs", "02_COMPOSITION_EMOTIONNELLE_ET_THEORIE.md"),
            os.path.join(ROOT, "docs", "04_ANALYSE_SONORE_SAMPLES_APIS_ET_VGM.md"),
            os.path.join(ROOT, "docs", "AUDIT_CARENCES_ET_SOLUTIONS_V3_2.md"),
        ]
        # valeurs périmées interdites dans les docs normatives (anciennes contradictions)
        forbidden = [
            "-2.6..-2.1", "ceiling=-2.6", "-1.5 dbtp", "ceiling -1.5 db",
            "0.06–0.10", "0.06-0.10", "mix <= 0.05", "mix <= 0.06",
            "2.5 khz` et `5.5 khz", "4.0 khz` et `7.5 khz", "lpf strict entre",
            "cutoff <= 6 khz", "lead_vs_harmony_db >= 2.0", "02_gestion_projets_et_samples.md",
        ]
        for p in normative_files:
            self.assertTrue(os.path.isfile(p), f"fichier normatif manquant : {p}")
            with open(p, encoding="utf-8") as fh:
                t = norm(fh.read())
            for f in forbidden:
                self.assertNotIn(norm(f), t, f"valeur périmée '{f}' encore présente dans {os.path.basename(p)}")
        # la valeur canonique du plafond limiteur doit être citée dans le runbook et docs/05
        for p in (normative_files[0], normative_files[1]):
            with open(p, encoding="utf-8") as fh:
                t = norm(fh.read())
            self.assertIn(f"{ceiling} dbtp", t, f"plafond limiteur {ceiling} dBTP non cité dans {os.path.basename(p)}")
        # planchers LPF sources canoniques cités
        with open(normative_files[1], encoding="utf-8") as fh:
            t = norm(fh.read())
        self.assertIn("6.5 khz", t)
        self.assertIn("9 khz", t)
        self.assertIn("config/mix_profiles.json", t)


class TestDocsCoherenceCIv33(unittest.TestCase):
    """v3.3 J0 — CI documentaire renforcée (doc audit v3.3 §0) :
    0 divergence numérique dans TOUS les tableaux de docs/05_*, runbook aligné v3.3,
    0 numérotation dupliquée / non croissante, 0 tableau à colonnes décalées."""

    def _docs_files(self):
        base = os.path.join(ROOT, "docs")
        return [os.path.join(base, fn) for fn in sorted(os.listdir(base)) if fn.endswith(".md")]

    def test_no_numeric_divergence_in_all_05_tables(self):
        forbidden = [
            "soft `8 %`", "saturation douce `10 %`",
            ">= +2 db", "[+2.5,+9.5]", "[+2.5, +9.5]",
            "lpf synthé `5–7 khz`", "lpf synthé `5-7 khz`",
        ]
        bad = []
        for p in self._docs_files():
            if not os.path.basename(p).startswith("05_"):
                continue
            with open(p, encoding="utf-8") as fh:
                for i, ln in enumerate(fh, 1):
                    if not ln.strip().startswith("|"):
                        continue
                    low = ln.replace(",", ".").lower()
                    for f in forbidden:
                        if f.replace(",", ".").lower() in low:
                            bad.append(f"{os.path.basename(p)}:{i}: {f}")
        self.assertEqual(bad, [], f"valeurs divergentes dans les tableaux docs/05_* : {bad}")

    def test_runbook_synced_with_v33(self):
        p = os.path.join(ROOT, "docs", "AGENT_RUNBOOK.md")
        with open(p, encoding="utf-8") as fh:
            t = fh.read()
        self.assertIn("# AGENT RUNBOOK v3.3", t, "en-tête du runbook doit être v3.3")
        for tok in ("--probe", "ScoreIR", "section_progressions", "NarrativeHarmony", "OrchestrationEngine",
                    "glide_to_next", "config/timbre_archetypes.json", "midi_groove_bank",
                    "MelodicAppealIndex", "FullCoverageCoreValidator", "blueprint_enricher",
                    "studio_refine_track", "preview_highlights.wav", "seed", "4->3"):
            self.assertIn(tok, t, f"runbook v3.3 ne mentionne pas {tok!r}")
        self.assertLessEqual(len(t), 6000, f"runbook = {len(t)} cars > 6000")

    def test_no_duplicated_numbering_or_misaligned_tables(self):
        import re as _re
        num_re = _re.compile(r"^\s*(\d+)[.)]\s+\S")
        pipe_re = _re.compile(r"(?<!\\)\|")
        problems = []
        for p in self._docs_files():
            with open(p, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            run: list[tuple[int, int]] = []
            tbl: list[tuple[int, int]] = []

            def flush_run():
                if run:
                    nums = [n for _, n in run]
                    if len(set(nums)) != len(nums) or nums != sorted(nums):
                        problems.append(f"{os.path.basename(p)}: numérotation {nums} lignes {[i for i, _ in run]}")

            def flush_tbl():
                if tbl:
                    counts = {c for _, c in tbl}
                    if len(counts) > 1:
                        problems.append(f"{os.path.basename(p)}: tableau lignes {[i for i, _ in tbl]} pipes={sorted(counts)}")

            for i, ln in enumerate(lines, 1):
                m = num_re.match(ln)
                if m:
                    flush_tbl(); tbl = []
                    run.append((i, int(m.group(1))))
                elif ln.strip().startswith("|"):
                    flush_run(); run = []
                    tbl.append((i, len(pipe_re.findall(ln))))
                else:
                    flush_run(); run = []
                    flush_tbl(); tbl = []
            flush_run(); flush_tbl()
        self.assertEqual(problems, [], f"problèmes de structure Markdown dans docs/ : {problems}")
