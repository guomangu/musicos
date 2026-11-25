#!/usr/bin/env python3
"""v3.3 hotfix — phase d'indexation des samples (install.sh étape 8/8) :
  (a) `_iter_media_files` remonte désormais les rejets AVEC raison (dotfile, .wav.zpa,
      bad_magic_bytes, too_small, symlink_duplicate) au lieu de les laisser invisibles ;
  (b) `index_directory` rapporte `scanned` / `rejected` / `rejected_reasons` en plus des
      compteurs historiques (total/indexed/skipped) — `skipped` = déjà en base (incrémental) ;
  (c) `verify_install.validate_active_blueprint` ne plante plus quand
      `projects/active/song_blueprint.json` est absent (installation fraîche) et valide
      strictement quand il est présent.
"""
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


def _write_wav(path: str, freq: float = 440.0, sr: int = 22050, dur: float = 0.1) -> None:
    import soundfile as sf
    t = np.arange(int(dur * sr)) / sr
    sf.write(path, 0.4 * np.sin(2 * np.pi * freq * t), sr)


def _build_samples_tree(root: str) -> None:
    os.makedirs(os.path.join(root, "sfz"), exist_ok=True)
    _write_wav(os.path.join(root, "kick.wav"), 55.0)
    _write_wav(os.path.join(root, "sfz", "inner.wav"), 220.0)          # répertoire sfz exclu
    with open(os.path.join(root, "loop.mid"), "wb") as fh:               # MIDI accepté tel quel
        fh.write(b"MThd\x00\x00\x00\x06\x00\x01\x01\xe0MTrk\x00\x00\x00\x04\x00\xff\x2f\x00")
    with open(os.path.join(root, "fake.wav"), "w", encoding="utf-8") as fh:  # > 1 Ko mais pas de magic audio
        fh.write("not audio at all " * 200)
    with open(os.path.join(root, "tiny.wav"), "wb") as fh:               # < 1 Ko
        fh.write(b"RIFF")
    with open(os.path.join(root, "pack.wav.zpa"), "w", encoding="utf-8") as fh:  # suffixe composite
        fh.write("PK" * 600)
    with open(os.path.join(root, ".DS_Store"), "wb") as fh:              # parasite OS (dotfile)
        fh.write(b"\x00" * 64)
    with open(os.path.join(root, "readme.txt"), "w", encoding="utf-8") as fh:  # non média : ignoré
        fh.write("hello")


class TestIterMediaFilesReporting(unittest.TestCase):
    def test_accepted_and_rejected_with_reasons(self):
        from indexer import _iter_media_files
        with tempfile.TemporaryDirectory() as tmp:
            _build_samples_tree(tmp)
            files, rejected = _iter_media_files(tmp)
            names = sorted(os.path.basename(f) for f, _rel in files)
            self.assertEqual(names, ["kick.wav", "loop.mid"])  # sfz/ exclu, readme.txt ignoré
            rnames = {os.path.basename(r["path"]): r["reason"] for r in rejected}
            self.assertIn("fake.wav", rnames)
            self.assertEqual(rnames["fake.wav"], "bad_magic_bytes")
            self.assertTrue(rnames["tiny.wav"].startswith("too_small"))
            self.assertEqual(rnames["pack.wav.zpa"], "non_pcm_suffix:.zpa")
            self.assertEqual(rnames[".DS_Store"], "dotfile")
            self.assertNotIn("inner.wav", rnames)  # sfz/ pruned, pas compté


class TestIndexDirectoryReport(unittest.TestCase):
    def test_report_keys_and_incremental_skip(self):
        from indexer import index_directory
        with tempfile.TemporaryDirectory() as tmp:
            samples = os.path.join(tmp, "samples")
            os.makedirs(samples)
            _build_samples_tree(samples)
            db = os.path.join(tmp, "samples.db")
            r1 = index_directory(samples, db, force=False, with_clap=False, with_dsp=False)
            for k in ("total", "indexed", "skipped", "errors", "scanned", "rejected", "rejected_reasons"):
                self.assertIn(k, r1)
            self.assertEqual(r1["indexed"], 2)          # kick.wav + loop.mid
            self.assertEqual(r1["skipped"], 0)
            self.assertEqual(r1["rejected"], 4)          # fake/tiny/.zpa/.DS_Store
            self.assertEqual(r1["scanned"], 6)           # 2 acceptés + 4 rejetés
            self.assertIn("bad_magic_bytes", r1["rejected_reasons"])
            # 2e passage incrémental : tout est déjà en base -> skipped, indexed=0
            r2 = index_directory(samples, db, force=False, with_clap=False, with_dsp=False)
            self.assertEqual(r2["indexed"], 0)
            self.assertEqual(r2["skipped"], 2)
            self.assertEqual(r2["errors"], 0)


class TestVerifyInstallBlueprintGuard(unittest.TestCase):
    def test_missing_active_blueprint_is_a_clean_skip(self):
        from verify_install import validate_active_blueprint
        with tempfile.TemporaryDirectory() as tmp:
            ok, msg = validate_active_blueprint(tmp)  # pas de projects/active -> pas d'exception
            self.assertTrue(ok)
            self.assertIn("SKIP", msg)

    def test_invalid_blueprint_fails_loudly(self):
        import jsonschema
        from verify_install import validate_active_blueprint
        schema_src = os.path.join(ROOT, "schemas", "song_blueprint.schema.json")
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "projects", "active"))
            bp = os.path.join(tmp, "projects", "active", "song_blueprint.json")
            with open(bp, "w", encoding="utf-8") as fh:
                json.dump({"title": "x"}, fh)  # champs required manquants
            with self.assertRaises(jsonschema.ValidationError):
                validate_active_blueprint(tmp, schema_path=schema_src)

    def test_valid_blueprint_passes(self):
        from verify_install import validate_active_blueprint
        schema_src = os.path.join(ROOT, "schemas", "song_blueprint.schema.json")
        src_bp = os.path.join(ROOT, "projects", "validation_fm_dark", "song_blueprint.json")
        if not os.path.isfile(src_bp):
            self.skipTest("fixture projet absent")
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "projects", "active"))
            with open(src_bp, encoding="utf-8") as fh:
                bp = json.load(fh)
            dst = os.path.join(tmp, "projects", "active", "song_blueprint.json")
            with open(dst, "w", encoding="utf-8") as fh:
                json.dump(bp, fh)
            ok, msg = validate_active_blueprint(tmp, schema_path=schema_src)
            self.assertTrue(ok)
            self.assertIn("OK", msg)


if __name__ == "__main__":
    unittest.main()
