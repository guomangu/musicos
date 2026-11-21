"""Indexeur sémantique et spectral des presets natifs Gridline (gl.*) dans samples.db."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from typing import Any

import essentia.standard as es
import numpy as np

from indexer import DEFAULT_DB_PATH, _brightness_band, _load_mono_audio, init_db

SKIP_PLUGINS = {
    "gl.midiout", "gl.sampler", "gl.slicer", "gl.granular", "gl.pads", "gl.synth",
    "gl.multisampler", "gl.samplekeys", "gl.scratch", "gl.splinescratch", "gl.platter",
    "gl.seltmann", "gl.grainscan", "gl.envctl", "gl.keyctl", "gl.cvctl", "gl.cvin",
    "gl.patcher", "gl.tracker", "gl.imagesynth",
}
SKIP_CATEGORIES = {"MIDI", "Sampler", "Control", "Utility", "Automation"}


def init_presets_table(conn: sqlite3.Connection) -> None:
    """Crée la table synth_presets dans samples.db si absente."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS synth_presets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plugin_id TEXT NOT NULL,
            plugin_name TEXT NOT NULL,
            preset_name TEXT NOT NULL,
            category TEXT NOT NULL,
            rms_db REAL,
            spectral_centroid_hz REAL,
            brightness_band TEXT,
            inharmonicity REAL,
            tags TEXT NOT NULL,
            clap_embedding BLOB,
            updated_at REAL NOT NULL,
            UNIQUE(plugin_id, preset_name)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_presets_cat ON synth_presets(category)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_presets_bright ON synth_presets(brightness_band)")
    conn.commit()


class _HeadlessGridline:
    """Client JSON-RPC minimal sur une instance unique de `gridline mcp`."""

    def __init__(self, project_path: str, export_dir: str) -> None:
        self.proc = subprocess.Popen(
            ["gridline", "mcp", project_path, "--export-dir", export_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._id = 1
        self._send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "preset_indexer", "version": "1"}}})
        self._read_response(1)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, obj: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _read_response(self, target_id: int) -> dict[str, Any]:
        assert self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("gridline mcp fermé de manière inattendue")
            msg = json.loads(line)
            if msg.get("id") == target_id:
                if "error" in msg:
                    raise RuntimeError(str(msg["error"]))
                return msg

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._id += 1
        req_id = self._id
        self._send({"jsonrpc": "2.0", "id": req_id, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
        resp = self._read_response(req_id)
        content = resp.get("result", {}).get("content", [])
        if content and "text" in content[0]:
            try:
                return json.loads(content[0]["text"])
            except Exception:
                return {"text": content[0]["text"]}
        return resp.get("result", {})

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def _test_notes_for(plugin_id: str, category: str, preset_name: str) -> list[dict[str, Any]]:
    low_kw = ("bass", "sub", "acid", "808")
    p_txt = f"{plugin_id} {preset_name}".lower()
    if category.lower() == "drums" or plugin_id in {"gl.kick", "gl.snare", "gl.hat", "gl.clap"}:
        return [{"start": 0.0, "pitch": 60, "length": 1.0, "velocity": 100}]
    if any(k in p_txt for k in low_kw):
        return [
            {"start": 0.0, "pitch": 36, "length": 2.0, "velocity": 105},
            {"start": 2.5, "pitch": 43, "length": 1.0, "velocity": 95},
        ]
    return [
        {"start": 0.0, "pitch": 48, "length": 2.5, "velocity": 92},
        {"start": 0.0, "pitch": 60, "length": 2.5, "velocity": 92},
        {"start": 0.0, "pitch": 63, "length": 2.5, "velocity": 92},
        {"start": 0.0, "pitch": 67, "length": 2.5, "velocity": 92},
    ]


def _compute_preset_clap(conn: sqlite3.Connection, audio_map: dict[tuple[str, str], np.ndarray]) -> int:
    if not audio_map:
        return 0
    import torch
    from transformers import ClapModel, ClapProcessor

    model_id = "laion/clap-htsat-unfused"
    with contextlib.redirect_stdout(sys.stderr):
        try:
            proc = ClapProcessor.from_pretrained(model_id, local_files_only=True)
            mod = ClapModel.from_pretrained(model_id, local_files_only=True)
        except Exception:
            proc = ClapProcessor.from_pretrained(model_id)
            mod = ClapModel.from_pretrained(model_id)
        mod.eval()

    keys = list(audio_map.keys())
    updated = 0
    for start in range(0, len(keys), 24):
        chunk_keys = keys[start : start + 24]
        audios = [audio_map[k] for k in chunk_keys]
        inputs = proc(audio=audios, sampling_rate=48000, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = mod.get_audio_features(**inputs)
            if not isinstance(feats, torch.Tensor):
                feats = getattr(feats, "pooler_output", getattr(feats, "last_hidden_state", feats[0]))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        vecs = feats.cpu().numpy().astype(np.float32)
        for (pid, pname), vec in zip(chunk_keys, vecs):
            conn.execute(
                "UPDATE synth_presets SET clap_embedding = ? WHERE plugin_id = ? AND preset_name = ?",
                (vec.tobytes(), pid, pname),
            )
            updated += 1
        conn.commit()
    return updated


def index_gridline_presets(
    db_path: str = DEFAULT_DB_PATH,
    with_clap: bool = True,
    force: bool = False,
) -> dict[str, int]:
    """Rend et indexe les presets d'usine des synthés natifs gl.* dans synth_presets."""
    conn = init_db(db_path)
    init_presets_table(conn)
    existing = {
        (r["plugin_id"], r["preset_name"]): r["clap_embedding"] is not None
        for r in conn.execute("SELECT plugin_id, preset_name, clap_embedding FROM synth_presets")
    }

    indexed = 0
    skipped = 0
    errors = 0
    clap_queue: dict[tuple[str, str], np.ndarray] = {}

    with tempfile.TemporaryDirectory() as td:
        proj = os.path.join(td, "preset_probe.gridline")
        subprocess.run(["gridline", "demo", proj], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        gl = _HeadlessGridline(proj, td)
        try:
            # Vide les canaux initiaux et fixe une mesure à 140 BPM (~1.7s) pour des rendus rapides
            chans = gl.call("list_channels", {})
            n_ch = len(chans.get("channels", []))
            for _ in range(n_ch):
                gl.call("remove_channel", {"channel": 1})
            gl.call("set_song", {"tempo": 140.0})
            gl.call("edit_pattern", {"pattern": 1, "bars": 1})

            cat_resp = gl.call("list_plugin_catalog", {"kind": "instrument"})
            plugins = cat_resp.get("plugins", [])
            win = es.Windowing(type="hann")
            spec_algo = es.Spectrum()
            cent_algo = es.Centroid(range=22050.0)
            peaks_algo = es.SpectralPeaks(orderBy="frequency", maxFrequency=12000.0, sampleRate=44100.0)
            inharm_algo = es.Inharmonicity()

            for plug in plugins:
                pid = plug.get("type", "")
                pname_plug = plug.get("name", pid)
                category = plug.get("category", "Synth")
                presets = plug.get("presets") or plug.get("factoryPresets") or []
                if not pid.startswith("gl.") or pid in SKIP_PLUGINS or category in SKIP_CATEGORIES or not presets:
                    continue

                for idx_p, preset_name in enumerate(presets):
                    has_row = (pid, preset_name) in existing
                    if not force and has_row and (not with_clap or existing[(pid, preset_name)]):
                        skipped += 1
                        continue
                    wav_name = f"p_{pid.replace('.', '_')}_{idx_p}.wav"
                    wav_path = os.path.join(td, wav_name)
                    try:
                        gl.call("add_channel", {"plugin": pid, "preset": preset_name, "mixer": 0})
                        notes = _test_notes_for(pid, category, preset_name)
                        gl.call("set_notes", {"pattern": 1, "channel": 1, "notes": notes})
                        gl.call(
                            "render_audio",
                            {
                                "path": wav_name,
                                "scope": "pattern",
                                "pattern": 1,
                                "sampleRate": 44100,
                                "bits": 16,
                                "maxSeconds": 2,
                                "wait": True,
                            },
                        )
                        gl.call("remove_channel", {"channel": 1})

                        y44, _, _, _ = _load_mono_audio(wav_path, target_sr=44100, max_seconds=3.0)
                        rms = float(np.sqrt(np.mean(y44**2)))
                        if rms < 1e-6:
                            continue
                        rms_db = round(20.0 * float(np.log10(max(rms, 1e-12))), 2)

                        # Centroïde pondéré par trames actives
                        cents = []
                        inharms = []
                        hop = 2048
                        for pos in range(0, max(1, len(y44) - 4096), hop):
                            frame = y44[pos : pos + 4096]
                            if float(np.max(np.abs(frame))) < 0.01:
                                continue
                            mag = spec_algo(win(frame))
                            cents.append(float(cent_algo(mag)))
                            try:
                                freqs, mags = peaks_algo(mag)
                                if len(freqs) >= 2 and freqs[0] > 30.0:
                                    inharms.append(float(inharm_algo(freqs, mags)))
                            except Exception:
                                pass
                        centroid = round(float(np.median(cents)) if cents else 1000.0, 1)
                        inharm = round(float(np.median(inharms)) if inharms else 0.0, 4)
                        bright = _brightness_band(centroid)
                        tags = f"{pid} {pname_plug} {preset_name} {category} {bright}".lower()

                        conn.execute(
                            """
                            INSERT INTO synth_presets (
                                plugin_id, plugin_name, preset_name, category, rms_db,
                                spectral_centroid_hz, brightness_band, inharmonicity, tags, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(plugin_id, preset_name) DO UPDATE SET
                                plugin_name=excluded.plugin_name,
                                category=excluded.category,
                                rms_db=excluded.rms_db,
                                spectral_centroid_hz=excluded.spectral_centroid_hz,
                                brightness_band=excluded.brightness_band,
                                inharmonicity=excluded.inharmonicity,
                                tags=excluded.tags,
                                updated_at=excluded.updated_at
                            """,
                            (pid, pname_plug, preset_name, category, rms_db, centroid, bright, inharm, tags, time.time()),
                        )
                        indexed += 1

                        if with_clap:
                            y48, _, _, _ = _load_mono_audio(wav_path, target_sr=48000, max_seconds=3.0)
                            if len(y48) > 64:
                                clap_queue[(pid, preset_name)] = y48
                    except Exception:
                        errors += 1
                        try:
                            gl.call("remove_channel", {"channel": 1})
                        except Exception:
                            pass
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                conn.commit()
        finally:
            gl.close()

    clap_done = _compute_preset_clap(conn, clap_queue) if with_clap else 0
    total = conn.execute("SELECT COUNT(*) FROM synth_presets").fetchone()[0]
    conn.close()
    return {"total_presets": total, "indexed": indexed, "skipped": skipped, "errors": errors, "clap_embedded": clap_done}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexeur de presets natifs Gridline (gl.*)")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--with-clap", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(index_gridline_presets(db_path=args.db_path, with_clap=args.with_clap, force=args.force))
