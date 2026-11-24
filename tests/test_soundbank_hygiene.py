#!/usr/bin/env python3
"""v3.1 TEST-06/ASSET-05 — soundbank hygiene: trap files rejected, license guard, exact midi_root."""
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


def _write_wav(path, freq, sr=22050, dur=0.8):
    import soundfile as sf
    t = np.arange(int(dur * sr)) / sr
    sf.write(path, 0.4 * np.sin(2 * np.pi * freq * t), sr)


class TestFileFilters(unittest.TestCase):
    def test_extension_whitelist_and_parasites(self):
        from asset_hygiene import is_decodable_audio_file, is_audio_file_safe
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "kick.wav")
            _write_wav(good, 55.0)
            self.assertEqual(is_audio_file_safe(good), (True, "ok"))
            traps = {
                "NM - Deep 808.wav.zpa": "non_pcm_suffix",
                "Dro Pad.fst": "non_pcm_suffix",
                "proj.als": "non_pcm_suffix",
                "bank.sf2": "non_pcm_suffix",
                ".DS_Store": "os_metadata",
                "Thumbs.db": "os_metadata",
                "Icon\r": "os_metadata",
                "notes.txt": "ext_not_whitelisted",
            }
            for name, reason in traps.items():
                p = os.path.join(tmp, name)
                with open(p, "wb") as fh:
                    fh.write(b"RIFF" + b"\x00" * 2048)
                ok, why = is_decodable_audio_file(p)
                self.assertFalse(ok)
                self.assertIn(reason, why)
            # apple double + __MACOSX + tiny file
            mac = os.path.join(tmp, "__MACOSX", "._kick.wav")
            os.makedirs(os.path.dirname(mac))
            with open(mac, "wb") as fh:
                fh.write(b"RIFF" + b"\x00" * 4096)
            ok, why = is_decodable_audio_file(mac)
            self.assertFalse(ok)
            self.assertIn(why, ("apple_double", "bad_dir_segment"))
            mac2 = os.path.join(tmp, "__MACOSX", "kick.wav")
            with open(mac2, "wb") as fh:
                fh.write(b"RIFF" + b"\x00" * 4096)
            ok2, why2 = is_decodable_audio_file(mac2)
            self.assertFalse(ok2)
            self.assertIn("bad_dir_segment", why2)
            tiny = os.path.join(tmp, "tiny.wav")
            with open(tiny, "wb") as fh:
                fh.write(b"RIFF\x00\x00\x00\x00WAVE")
            ok, why = is_audio_file_safe(tiny)
            self.assertFalse(ok)  # < 1024 octets
            # valid wav but corrupted magic
            bad = os.path.join(tmp, "corrupt.wav")
            with open(bad, "wb") as fh:
                fh.write(b"XXXX" + b"\x00" * 4096)
            ok, why = is_audio_file_safe(bad)
            self.assertFalse(ok)
            self.assertEqual(why, "bad_magic_bytes")


class TestIndexingHygiene(unittest.TestCase):
    def test_only_valid_wavs_indexed_and_roots_exact(self):
        from index_project_samples import index_project_samples
        with tempfile.TemporaryDirectory() as tmp:
            samples = os.path.join(tmp, "Samples")
            os.makedirs(samples)
            _write_wav(os.path.join(samples, "sub_c2.wav"), 65.41)   # C2 = MIDI 36
            _write_wav(os.path.join(samples, "lead_a4.wav"), 440.0)  # A4 = MIDI 69
            # trap files
            for name, blob in {
                "NM - Deep 808.wav.zpa": b"PK\x03\x04" + b"\x00" * 4096,
                "pad.fst": b"CcnK" + b"\x00" * 4096,
            }.items():
                with open(os.path.join(samples, name), "wb") as fh:
                    fh.write(blob)
            os.makedirs(os.path.join(samples, "__MACOSX"))
            with open(os.path.join(samples, "__MACOSX", "._sub_c2.wav"), "wb") as fh:
                fh.write(b"RIFF" + b"\x00" * 4096)
            with open(os.path.join(samples, ".DS_Store"), "wb") as fh:
                fh.write(b"\x00" * 4096)
            with open(os.path.join(samples, "Icon\r"), "wb") as fh:
                fh.write(b"")
            catalog = index_project_samples(tmp, with_clap=False, write_catalog=True)
            by_name = {os.path.basename(e["filepath"]): e for e in catalog["samples"]}
            # seuls les 2 WAV valides de Samples/ racine sont indexés (le pack restreint est tagué)
            self.assertIn("sub_c2.wav", by_name)
            self.assertIn("lead_a4.wav", by_name)
            self.assertEqual(by_name["sub_c2.wav"]["midi_root"], 36)
            self.assertEqual(by_name["lead_a4.wav"]["midi_root"], 69)
            self.assertNotIn("NM - Deep 808.wav.zpa", by_name)
            self.assertNotIn("pad.fst", by_name)
            self.assertGreaterEqual(catalog.get("skipped_count", 0), 4)
            # aucun tag de licence (responsabilité utilisateur), aucun crash
            self.assertNotIn("license_status", by_name["sub_c2.wav"])
            self.assertEqual(catalog.get("error_count", 1), 0)


if __name__ == "__main__":
    unittest.main()
