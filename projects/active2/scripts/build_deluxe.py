"""Mise à niveau complète et enrichissement musical de Obsidian Midnight (36 mesures, ~61.7s @ 140 BPM)."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import time
from typing import Any

PROJ_PATH = "/home/gamo/Bureau/musicos/projects/active2/projet.gridline"
EXPORT_DIR = "/home/gamo/Bureau/musicos/projects/active2/renders"
SFX_ROOT = "/home/gamo/Bureau/musicos/samples"
SFZ_PIANO = "/home/gamo/Bureau/musicos/soundbanks/sfz/headroom-piano/Headroom Piano (NoFX).sfz"

SAMPLES = {
    "rimshot": f"{SFX_ROOT}/TriSamples_-_808_Trapstep_Pack_Vol_2/TriSamples - 808 Trapstep Pack Vol 2/Percussion/Percussion Rimshot 004.wav",
    "riser": f"{SFX_ROOT}/CHRISSA SJE & NIKOS - INFINITE (SOUND PACK)/05. Drum Kit ∞/06. FX_Percs/04. transition - dream counting.wav",
    "crash": f"{SFX_ROOT}/TriSamples_-_808_Trapstep_Pack_Vol_2/TriSamples - 808 Trapstep Pack Vol 2/Cymbals/Cymbal Crash 001 Electronic.wav",
    "chant": f"{SFX_ROOT}/TriSamples_-_808_Trapstep_Pack_Vol_2/TriSamples - 808 Trapstep Pack Vol 2/Vocals & Chants/Chant Bounce 001.wav",
}


class GridlineClient:
    def __init__(self, proj: str, export_dir: str) -> None:
        self.proc = subprocess.Popen(
            ["gridline", "mcp", proj, "--export-dir", export_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._id = 1
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "deluxe", "version": "1.0"}},
                }
            )
            + "\n"
        )
        self.proc.stdin.flush()
        self._wait_id(1)
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        self.proc.stdin.flush()

    def _wait_id(self, tid: int) -> dict[str, Any]:
        assert self.proc.stdout
        while True:
            line = self.proc.stdout.readline()
            if not line:
                err = self.proc.stderr.read() if self.proc.stderr else ""
                raise RuntimeError(f"Gridline MCP EOF: {err}")
            msg = json.loads(line)
            if msg.get("id") == tid:
                if "error" in msg:
                    raise RuntimeError(f"MCP error: {msg['error']}")
                return msg

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self.proc.stdin
        for attempt in range(4):
            self._id += 1
            tid = self._id
            self.proc.stdin.write(
                json.dumps({"jsonrpc": "2.0", "id": tid, "method": "tools/call", "params": {"name": name, "arguments": arguments or {}}})
                + "\n"
            )
            self.proc.stdin.flush()
            try:
                resp = self._wait_id(tid)
                content = resp["result"].get("content", [])
                if content and "text" in content[0]:
                    try:
                        return json.loads(content[0]["text"])
                    except Exception:
                        return {"text": content[0]["text"]}
                return resp["result"]
            except RuntimeError as exc:
                if "Rate limit" in str(exc) and attempt < 3:
                    time.sleep(0.75)
                    continue
                raise
        raise RuntimeError("Unreachable")

    def batch_ops(self, label: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
        return self.call("batch", {"calls": calls})

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=15)


def set_params(gl: GridlineClient, prefix: str, pairs: list[tuple[str | int, Any]]) -> None:
    gl.call("set_parameters", {"changes": [{"path": f"{prefix}/{p}", "value": v} for p, v in pairs]})


def set_eq_params(
    gl: GridlineClient,
    mixer: int,
    slot: int,
    bands: dict[int, tuple[int, float, float, float]],
) -> None:
    """Configure jusqu'à 7 bandes gl.eq (index -> (type, freq, gain, width))."""
    items: list[tuple[str | int, Any]] = [(67, 1)]  # Quality: High (decramped)
    width_params = {1: 53, 2: 55, 3: 57, 4: 59, 5: 61, 6: 63, 7: 65}
    en_params = {1: 38, 2: 40, 3: 42, 4: 44, 5: 46, 6: 48, 7: 50}
    for b_idx, (b_type, freq, gain, width) in bands.items():
        base = (b_idx - 1) * 5
        items.append((base + 1, b_type))
        items.append((base + 2, freq))
        items.append((base + 3, gain))
        items.append((width_params[b_idx], width))
        items.append((en_params[b_idx], 0 if b_type == 0 else 1))
    set_params(gl, f"mixer/{mixer}/slot/{slot}", items)


def human_notes(raw: list[tuple[float, int, float, int]], rng: random.Random, jitter_time: float = 0.008) -> list[dict[str, Any]]:
    out = []
    for st, pt, ln, vl in raw:
        st_h = max(0.0, round(st + (rng.uniform(-jitter_time, jitter_time) if st > 0 else 0.0), 4))
        vl_h = max(25, min(122, vl + rng.randint(-5, 5)))
        out.append({"start": st_h, "pitch": pt, "length": round(ln, 4), "velocity": vl_h})
    return out


def build_harmony_strings(mode: str = "main", rng: random.Random | None = None) -> list[dict[str, Any]]:
    """Accords chauds Polar Strings alignés sur la 808 Cm -> Cm7 -> Ab -> Eb/G -> Fm -> Gm sans aucun frottement."""
    r = rng or random.Random(42)
    chords = [
        # (start, len, pitches, vel) — pas de Eb4 contre D4 ni de G contre Ab simultané
        (0.0, 3.85, [55, 60, 63, 67], 82),      # Cm
        (4.0, 3.85, [55, 58, 63, 67], 80),      # Cm7/Eb
        (8.0, 2.85, [56, 60, 63], 82),          # Ab
        (11.0, 0.90, [55, 58, 63], 78),         # Eb/G
        (12.0, 1.90, [53, 56, 60], 82),         # Fm (sans Eb4 pour laisser passer D/F)
        (14.0, 1.85, [55, 58, 62], 80),         # Gm
    ]
    raw = []
    for st, ln, ps, vl in chords:
        if mode == "sparse" and st in (11.0,):
            continue
        if mode == "outro" and st >= 12.0:
            if st == 12.0:
                ps = [53, 56, 60]
                ln = 1.9
            else:
                ps = [55, 60, 63, 67]
                ln = 1.9
        for p in ps:
            v = vl - 6 if mode == "sparse" else (vl - 10 if mode == "outro" else vl)
            if mode == "climax" and p == ps[-1]:
                raw.append((st, p + 12, ln, max(56, v - 6)))
            raw.append((st, p, ln, v))
    return human_notes(raw, r, jitter_time=0.01)


def build_piano_notes(kind: str, rng: random.Random) -> list[dict[str, Any]]:
    """Parties de Grand Piano SFZ expressives (0 clash de 2nde/9e mineure, >= F3 pendant les drops avec 808)."""
    raw: list[tuple[float, int, float, int]] = []
    if kind == "intro":
        left = [(0.0, 48, 3.7, 74), (0.0, 55, 3.7, 70), (4.0, 48, 3.7, 72), (4.0, 58, 3.7, 68),
                (8.0, 51, 3.0, 74), (8.0, 56, 3.0, 70), (12.0, 53, 1.8, 74), (14.0, 55, 1.8, 72)]
        mel = [
            (0.0, 60, 0.9, 82), (0.75, 63, 0.9, 80), (1.5, 67, 1.4, 90), (3.0, 74, 0.9, 86),
            (4.0, 72, 1.4, 88), (5.5, 70, 0.9, 82), (6.5, 67, 1.3, 80),
            (8.0, 68, 0.9, 88), (8.75, 67, 0.8, 82), (9.5, 63, 1.3, 84), (11.0, 70, 0.9, 80),
            (12.0, 65, 0.9, 86), (12.75, 63, 0.9, 80), (13.5, 62, 1.4, 84), (15.0, 67, 0.9, 88),
        ]
        raw = left + mel
    elif kind in ("hook_a", "hook_b"):
        # Accords main gauche >= 53 (F3) pour éviter toute collision avec la 808
        chords = [
            (0.0, [55, 60, 63], 1.3, 84),
            (1.5, [55, 60, 63], 0.9, 76),
            (4.0, [55, 58, 63], 1.3, 84),
            (5.5, [55, 58, 63], 1.0, 76),
            (8.0, [56, 60, 63], 1.3, 86),
            (9.5, [56, 60, 63], 0.9, 78),
            (11.0, [55, 58, 63], 0.9, 78),
            (12.0, [53, 56, 60], 1.3, 86),
            (14.0, [55, 58, 62], 1.4, 84),
        ]
        for st, ps, ln, vl in chords:
            for p in ps:
                raw.append((st, p, ln, vl))
        # Mélodie chantante sans Ab sur G (bar 1-2) ni D4 contre Eb4
        mel = [
            (0.0, 67, 0.7, 98), (0.75, 70, 0.7, 90), (1.5, 67, 0.9, 96), (2.5, 63, 0.75, 88), (3.25, 67, 0.7, 86),
            (4.0, 72, 0.9, 100), (5.0, 70, 0.7, 90), (5.75, 67, 0.9, 92), (7.0, 63, 0.85, 88),
            (8.0, 68, 0.75, 96), (8.75, 72, 0.7, 88), (9.5, 63, 0.9, 92), (10.5, 60, 0.45, 84), (11.0, 70, 0.9, 92),
            (12.0, 65, 0.75, 96), (12.75, 63, 0.7, 88), (13.5, 65, 0.45, 90), (14.5, 67, 0.7, 94), (15.25, 74, 0.65, 98),
        ]
        raw.extend(mel)
        if kind == "hook_b":
            octaves = [
                (0.0, 79, 0.65, 86), (4.0, 75, 0.85, 84), (8.0, 80, 0.7, 88),
                (14.5, 79, 0.35, 90), (15.0, 77, 0.35, 88), (15.35, 74, 0.35, 86), (15.7, 70, 0.28, 88),
            ]
            raw.extend(octaves)
    elif kind == "sparse":
        raw = [
            (0.0, 55, 1.5, 80), (0.0, 60, 1.5, 88), (0.0, 63, 1.5, 84), (1.5, 67, 1.2, 82),
            (4.0, 58, 1.5, 80), (6.5, 67, 1.0, 82),
            (8.0, 56, 1.5, 80), (8.0, 68, 1.5, 88), (9.5, 63, 1.2, 82),
            (12.0, 53, 1.4, 80), (12.0, 65, 1.4, 86), (14.0, 55, 1.4, 80), (14.0, 62, 1.4, 86),
        ]
    elif kind == "pont":
        raw = [
            (0.0, 48, 3.6, 78), (0.0, 60, 1.2, 86), (1.0, 67, 0.9, 92), (2.0, 74, 0.9, 96), (3.0, 75, 0.9, 90),
            (4.0, 55, 3.6, 78), (4.0, 72, 1.1, 92), (5.5, 70, 0.9, 88), (6.5, 67, 1.2, 84),
            (8.0, 56, 2.8, 80), (8.0, 68, 0.9, 94), (9.0, 75, 0.9, 96), (10.0, 79, 0.9, 98), (11.0, 75, 0.9, 90),
            (12.0, 53, 1.9, 80), (12.0, 65, 0.9, 90), (13.0, 68, 0.9, 88),
            (14.0, 55, 1.9, 82), (14.0, 74, 0.8, 96), (15.0, 77, 0.45, 92), (15.5, 79, 0.45, 96),
        ]
    else:  # outro
        raw = [
            (0.0, 48, 3.6, 78), (0.0, 60, 3.5, 84), (0.5, 63, 3.0, 80), (1.0, 67, 2.5, 86), (1.5, 74, 2.0, 88),
            (4.0, 55, 3.6, 72), (4.0, 72, 1.4, 82), (5.5, 70, 1.4, 76),
            (8.0, 51, 3.5, 70), (8.0, 68, 1.4, 78), (9.5, 67, 1.8, 72),
            (12.0, 48, 3.8, 68), (12.0, 60, 3.8, 70), (12.0, 63, 3.8, 66), (12.0, 67, 3.8, 70), (12.0, 74, 3.8, 68),
        ]
    return human_notes(raw, rng, jitter_time=0.01)


def build_nylon_pluck(mode: str, rng: random.Random) -> list[dict[str, Any]]:
    """Contre-arpège acoustique strictement en Do mineur (0 note hors gamme ni clash)."""
    patterns_per_bar = [
        [60, 67, 63, 72, 67, 63, 75, 67],          # Bar 1 Cm
        [60, 67, 63, 70, 67, 63, 75, 70],          # Bar 2 Cm7
        [60, 68, 63, 72, 68, 63, 67, 70],          # Bar 3 Ab -> Eb/G
        [60, 65, 68, 65, 62, 67, 70, 74],          # Bar 4 Fm -> Gm7
    ]
    cm_echo = {75: 70, 74: 67, 72: 67, 70: 65, 68: 63, 67: 63, 65: 60}
    raw: list[tuple[float, int, float, int]] = []
    for b_idx, seq in enumerate(patterns_per_bar):
        for step, pitch in enumerate(seq):
            if mode == "verse" and step in (2, 6):
                continue
            t = b_idx * 4.0 + step * 0.5
            vel = 96 if step in (0, 3, 6) else 78
            raw.append((t, pitch, 0.42, vel))
            if mode == "climax" and step in (3, 7):
                raw.append((t + 0.25, cm_echo.get(pitch, 60), 0.22, 82))
    return human_notes(raw, rng, jitter_time=0.006)


def build_tine_rhodes(mode: str, rng: random.Random) -> list[dict[str, Any]]:
    """Couche Rhodes électrique douce (Classic tine) sur contre-temps et pont."""
    if mode == "pont":
        raw = [
            (0.0, 60, 1.8, 68), (0.0, 63, 1.8, 66), (0.0, 67, 1.8, 70),
            (4.0, 58, 1.8, 68), (4.0, 63, 1.8, 66), (4.0, 67, 1.8, 70),
            (8.0, 60, 2.2, 72), (8.0, 63, 2.2, 68), (8.0, 68, 2.2, 74),
            (12.0, 60, 1.8, 72), (12.0, 65, 1.8, 74), (14.0, 62, 1.8, 74), (14.0, 67, 1.8, 76),
        ]
    else:
        # Stabs doux sur contre-temps 2.5 et 6.5 (termine avant beat 11.0 Eb/G pour 0 clash)
        raw = [
            (2.5, 67, 0.9, 68), (2.5, 70, 0.9, 64),
            (6.5, 63, 0.9, 66), (6.5, 67, 0.9, 70),
            (10.5, 63, 0.42, 68), (10.5, 68, 0.42, 72),
            (14.5, 65, 0.9, 70), (14.5, 70, 0.9, 72),
        ]
    return human_notes(raw, rng, jitter_time=0.008)


def main() -> None:
    rng = random.Random(2026)
    gl = GridlineClient(PROJ_PATH, EXPORT_DIR)

    # Vérifie l'état initial
    chans = gl.call("list_channels")["channels"]
    print("Canaux initiaux :", len(chans))

    # 1. Répare Ch 7 (Polar Strings) et Ch 8 (Grand Piano SFZ) + gain staging final Pass 6 (PQ max, corr>=0.7, TP<=-1.1)
    gl.batch_ops(
        "Upgrade Ch 7 Polar Strings & Ch 8 Grand Piano SFZ",
        [
            {"tool": "edit_channel", "arguments": {"channel": 7, "name": "Polar Strings", "plugin": "gl.dcosynth", "preset": "Polar Strings", "volume": 0.92, "pan": 0.0, "color": "#5C7AD6"}},
            {"tool": "edit_channel", "arguments": {"channel": 8, "name": "Grand Piano", "plugin": "gl.multisampler", "volume": 0.94, "pan": 0.03, "color": "#D4AE4A"}},
        ],
    )
    gl.call("patch", {"path": "channel/8", "set": {"plugin": {"data": SFZ_PIANO}}})
    set_params(gl, "channel/8", [(4, 48), (8, 0.78), (9, 1), (14, 0.48)])

    # 2. Calibre Mixer 1..12 et 0 : clarté maximale sur XM6 + true peak <= -1.1 dBTP + LUFS ~ -11.3
    mx_map = {t["mixer"]: t for t in gl.call("get_mixer")["tracks"]}
    if len(mx_map.get(7, {}).get("effects", [])) >= 2:
        gl.call("remove_plugin", {"mixer": 7, "slot": 2})
    if len(mx_map.get(8, {}).get("effects", [])) >= 2:
        gl.call("remove_plugin", {"mixer": 8, "slot": 2})

    gl.call("edit_mixer_track", {"mixer": 1, "volume": 0.98})       # Kick punchy
    gl.call("edit_mixer_track", {"mixer": 2, "volume": 0.84})       # 808 Bass propre & profonde
    gl.call("edit_mixer_track", {"mixer": 3, "volume": 0.98})       # Trap Clap
    gl.call("edit_mixer_track", {"mixer": 4, "volume": 0.88})       # Bounce Snare
    gl.call("edit_mixer_track", {"mixer": 5, "volume": 1.12})       # Hi-Hat Closed brillants
    gl.call("edit_mixer_track", {"mixer": 6, "volume": 0.82})       # Open Hat
    gl.call("edit_mixer_track", {"mixer": 7, "name": "Polar Strings", "volume": 1.01})
    gl.call("edit_mixer_track", {"mixer": 8, "name": "Grand Piano", "volume": 1.04, "pan": 0.03})
    gl.call("edit_mixer_track", {"mixer": 10, "volume": 0.85})      # BUS Drums
    gl.call("edit_mixer_track", {"mixer": 11, "volume": 1.17})      # BUS Music
    gl.call("patch", {"path": "mixer/7", "set": {"stereoSep": 0.55}})  # Resserre la phase bas-médiums des Polar Strings (>0.7 corr)
    gl.call("set_send", {"mixer": 1, "to": 11, "level": 0.22, "sidechain": True})
    gl.call("set_send", {"mixer": 7, "to": 12, "level": 0.03})
    gl.call("set_send", {"mixer": 8, "to": 12, "level": 0.04})

    # EQ Hi-Hat Closed (Mixer 5) pour de l'air propre (4-6% cible)
    set_eq_params(gl, 5, 1, {1: (1, 500.0, 0.0, 0.65), 6: (7, 8200.0, 4.0, 0.60)})
    # EQ Polar Strings (Mixer 7) : coupe-bas 240 Hz + nettoyage 480/620 Hz
    set_eq_params(gl, 7, 1, {
        1: (1, 240.0, 0.0, 0.65),
        2: (6, 250.0, -3.5, 0.42),
        3: (6, 480.0, -3.5, 0.42),
        4: (6, 620.0, -6.0, 0.44),
        5: (6, 1500.0, 1.0, 0.42),
        6: (7, 7200.0, 2.0, 0.60),
        7: (3, 13000.0, 0.0, 0.65),
    })
    # EQ Grand Piano (Mixer 8) : nettoie 200 Hz, 500/630 Hz et 2 kHz (XM6) pour son studio droit et clair
    set_eq_params(gl, 8, 1, {
        1: (1, 175.0, 0.0, 0.65),
        2: (6, 210.0, -4.2, 0.42),
        3: (6, 500.0, -3.2, 0.42),
        4: (6, 630.0, -6.5, 0.45),
        5: (6, 2050.0, -3.2, 0.42),
        6: (6, 4800.0, 1.5, 0.45),
        7: (7, 9000.0, 1.8, 0.60),
    })
    # EQ BUS Music (Mixer 11) : passe-haut 155 Hz + coupe-cloche doux 550/630 Hz + léger air
    set_eq_params(gl, 11, 1, {
        1: (1, 155.0, 0.0, 0.65),
        3: (6, 500.0, -2.0, 0.40),
        4: (6, 630.0, -3.0, 0.42),
        7: (7, 8500.0, 1.8, 0.60),
    })
    # Reverb FX Intimate Room (Mixer 12) : ultra sèche/courte (0.30s, Dry=0, HPF 550 Hz)
    gl.call("edit_mixer_track", {"mixer": 12, "name": "FX Intimate Room", "volume": 0.22})
    set_params(gl, "mixer/12/slot/1", [
        (1, 16.0), (2, 0.14), (5, 550.0), (6, 4200.0), (8, 0.08), (9, 0.0),
        (10, 1.0), (11, 0.30),
        (23, 550), (24, 42), (25, 16), (26, 14), (28, 3), (29, 42),
        (32, 8), (33, 0), (34, 10), (35, 128), (38, 1),
    ])

    # Master EQ & Limiter (Mixer 0) : TP garanti <= -1.1 dBTP (Ceiling -3.0 dB), LUFS ~ -11.4
    set_eq_params(gl, 0, 1, {
        1: (1, 30.0, 0.0, 0.65),
        2: (6, 215.0, -1.5, 0.40),
        3: (6, 500.0, -2.0, 0.40),
        4: (6, 630.0, -4.0, 0.42),
        5: (6, 2000.0, -1.5, 0.38),
        6: (7, 8800.0, 2.5, 0.60),
        7: (3, 18000.0, 0.0, 0.65),
    })
    set_params(gl, "mixer/0/slot/3", [(1, 10.2), (2, -2.75), (4, 0.001), (19, 1), (24, 0.085)])

    # 3. Ajoute les nouvelles pistes & canaux enrichis (Ch 9..14) avec balance harmonieuse
    mx = gl.call("get_mixer")["tracks"]
    if len(mx) <= 17:
        gl.call("add_mixer_track", {"name": "Trap Chant"})

    new_channels = [
        # (ch_idx, name, plugin, preset, sample_key, mixer, pan, vol, color, bus_to, reverb_send, mix_vol)
        (9, "Nylon Pluck", "gl.twinstring", "Nylon pluck", None, 9, -0.22, 0.96, "#46A89D", 11, 0.04, 1.02),
        (10, "Classic Tine", "gl.keys", "Classic tine", None, 13, 0.22, 0.92, "#9C75D9", 11, 0.04, 1.00),
        (11, "Rimshot Perk", "gl.sampler", "One shot", "rimshot", 14, 0.18, 0.88, "#E68A3E", 10, 0.03, 0.96),
        (12, "Dream Riser", "gl.sampler", "One shot", "riser", 15, -0.08, 1.10, "#4FB0E8", 11, 0.04, 1.14),
        (13, "Drop Crash", "gl.sampler", "One shot", "crash", 16, 0.12, 0.86, "#D96690", 0, 0.04, 0.92),
        (14, "Trap Chant", "gl.sampler", "One shot", "chant", 17, -0.16, 0.92, "#B86BD9", 0, 0.04, 1.06),
    ]
    for idx_ch, name, plug, preset, skey, m_tr, pan, vol, col, bus_to, rev_s, m_vol in new_channels:
        if len(gl.call("list_channels")["channels"]) < idx_ch:
            gl.call("add_channel", {"plugin": plug, "preset": preset, "name": name, "mixer": m_tr, "pan": pan, "volume": vol, "color": col})
        else:
            gl.call("edit_channel", {"channel": idx_ch, "name": name, "plugin": plug, "preset": preset, "mixer": m_tr, "pan": pan, "volume": vol, "color": col})
        if plug == "gl.twinstring":
            set_params(gl, f"channel/{idx_ch}", [(1, 0.85)])
        if skey:
            gl.call("patch", {"path": f"channel/{idx_ch}", "set": {"plugin": {"data": SAMPLES[skey]}}})
            set_params(gl, f"channel/{idx_ch}", [(9, 0), (10, 1), (18, 1)])
        gl.call("edit_mixer_track", {"mixer": m_tr, "name": name, "volume": m_vol, "pan": pan, "routeTo": bus_to})
        if rev_s > 0:
            gl.call("set_send", {"mixer": m_tr, "to": 12, "level": rev_s})
        gl.call("add_plugin", {"mixer": m_tr, "plugin": "gl.eq", "slot": 1})

    # EQ dédiés pour chaque timbre regroupés dans un seul set_parameters
    eq_changes: list[dict[str, Any]] = []
    def queue_eq(mixer: int, slot: int, bands: dict[int, tuple[int, float, float, float]]) -> None:
        width_params = {1: 53, 2: 55, 3: 57, 4: 59, 5: 61, 6: 63, 7: 65}
        en_params = {1: 38, 2: 40, 3: 42, 4: 44, 5: 46, 6: 48, 7: 50}
        eq_changes.append({"path": f"mixer/{mixer}/slot/{slot}/67", "value": 1})
        for b_idx, (b_type, freq, gain, width) in bands.items():
            base = (b_idx - 1) * 5
            eq_changes.append({"path": f"mixer/{mixer}/slot/{slot}/{base + 1}", "value": b_type})
            eq_changes.append({"path": f"mixer/{mixer}/slot/{slot}/{base + 2}", "value": freq})
            eq_changes.append({"path": f"mixer/{mixer}/slot/{slot}/{base + 3}", "value": gain})
            eq_changes.append({"path": f"mixer/{mixer}/slot/{slot}/{width_params[b_idx]}", "value": width})
            eq_changes.append({"path": f"mixer/{mixer}/slot/{slot}/{en_params[b_idx]}", "value": 0 if b_type == 0 else 1})

    queue_eq(9, 1, {1: (1, 230.0, 0.0, 0.65), 3: (6, 550.0, -4.0, 0.42), 4: (6, 760.0, -3.5, 0.42), 6: (7, 4800.0, 1.5, 0.55), 7: (3, 10000.0, 0.0, 0.65)})
    queue_eq(13, 1, {1: (1, 360.0, 0.0, 0.65), 3: (6, 480.0, -4.5, 0.40), 6: (6, 4200.0, 1.5, 0.42), 7: (3, 8500.0, 0.0, 0.65)})
    queue_eq(14, 1, {1: (1, 320.0, 0.0, 0.65), 7: (3, 7000.0, 0.0, 0.65)})
    queue_eq(15, 1, {1: (1, 260.0, 0.0, 0.65), 7: (3, 10000.0, 0.0, 0.65)})
    queue_eq(16, 1, {1: (1, 650.0, 0.0, 0.65), 6: (7, 9000.0, -4.0, 0.60)})
    queue_eq(17, 1, {1: (1, 350.0, 0.0, 0.65), 7: (3, 4500.0, 0.0, 0.65)})
    gl.call("set_parameters", {"changes": eq_changes})

    # 4. Met à jour les patterns existants et crée les nouveaux patterns enrichis en batch
    cur_pats = len(gl.call("list_patterns")["patterns"])
    pat_specs = [
        (11, "Piano Hero Hook B", 4),
        (12, "Piano Sparse Couplet", 4),
        (13, "Piano Breakdown Pont", 4),
        (14, "Piano Outro", 4),
        (15, "Nylon Pluck Main", 4),
        (16, "Nylon Pluck Verse", 4),
        (17, "Nylon Pluck Climax", 4),
        (18, "Tine Rhodes Accents", 4),
        (19, "Tine Rhodes Pont", 4),
        (20, "Strings Climax", 4),
        (21, "Pont Drums & Build", 4),
        (22, "FX Riser Pre-Drop", 4),
        (23, "FX Drop Crash", 1),
        (24, "Trap Chant Turnaround", 4),
    ]
    pat_setup_calls = [
        {"tool": "edit_pattern", "arguments": {"pattern": 1, "name": "Polar Strings Main", "bars": 4}},
        {"tool": "edit_pattern", "arguments": {"pattern": 2, "name": "Piano Hero Hook A", "bars": 4}},
        {"tool": "edit_pattern", "arguments": {"pattern": 9, "name": "Piano Intro", "bars": 4}},
        {"tool": "edit_pattern", "arguments": {"pattern": 10, "name": "Strings Outro", "bars": 4}},
    ]
    for p_num, p_name, p_bars in pat_specs:
        if p_num > cur_pats:
            pat_setup_calls.append({"tool": "add_pattern", "arguments": {"name": p_name, "bars": p_bars}})
        else:
            pat_setup_calls.append({"tool": "edit_pattern", "arguments": {"pattern": p_num, "name": p_name, "bars": p_bars}})
    gl.batch_ops("Setup Patterns", pat_setup_calls)

    rim_drop = human_notes([(1.5, 60, 0.4, 78), (5.25, 60, 0.4, 72), (9.5, 60, 0.4, 82), (13.75, 60, 0.4, 76), (15.5, 60, 0.4, 84)], rng, 0.004)
    rim_verse = human_notes([(1.5, 60, 0.4, 74), (3.5, 60, 0.4, 64), (5.5, 60, 0.4, 76), (9.5, 60, 0.4, 78), (11.5, 60, 0.4, 68), (13.5, 60, 0.4, 74)], rng, 0.004)
    snare_build = [{"start": 12.0 + i * 0.5, "pitch": 60, "length": 0.22, "velocity": int(56 + i * 7)} for i in range(4)] + [
        {"start": 14.0 + i * 0.25, "pitch": 60, "length": 0.2, "velocity": int(84 + i * 5)} for i in range(8)
    ]

    note_calls = [
        {"tool": "set_notes", "arguments": {"pattern": 1, "channel": 7, "notes": build_harmony_strings("main", rng)}},
        {"tool": "set_notes", "arguments": {"pattern": 2, "channel": 8, "notes": build_piano_notes("hook_a", rng)}},
        {"tool": "set_notes", "arguments": {"pattern": 9, "channel": 8, "notes": build_piano_notes("intro", rng)}},
        {"tool": "set_notes", "arguments": {"pattern": 10, "channel": 7, "notes": build_harmony_strings("outro", rng)}},
        {"tool": "set_notes", "arguments": {"pattern": 4, "channel": 11, "notes": rim_drop}},
        {"tool": "set_notes", "arguments": {"pattern": 5, "channel": 11, "notes": rim_drop}},
        {"tool": "set_notes", "arguments": {"pattern": 6, "channel": 11, "notes": rim_verse}},
        {"tool": "set_notes", "arguments": {"pattern": 7, "channel": 11, "notes": rim_verse}},
        {"tool": "set_notes", "arguments": {"pattern": 11, "channel": 8, "notes": build_piano_notes("hook_b", rng)}},
        {"tool": "set_notes", "arguments": {"pattern": 12, "channel": 8, "notes": build_piano_notes("sparse", rng)}},
        {"tool": "set_notes", "arguments": {"pattern": 13, "channel": 8, "notes": build_piano_notes("pont", rng)}},
        {"tool": "set_notes", "arguments": {"pattern": 14, "channel": 8, "notes": build_piano_notes("outro", rng)}},
        {"tool": "set_notes", "arguments": {"pattern": 15, "channel": 9, "notes": build_nylon_pluck("main", rng)}},
        {"tool": "set_notes", "arguments": {"pattern": 16, "channel": 9, "notes": build_nylon_pluck("verse", rng)}},
        {"tool": "set_notes", "arguments": {"pattern": 17, "channel": 9, "notes": build_nylon_pluck("climax", rng)}},
        {"tool": "set_notes", "arguments": {"pattern": 18, "channel": 10, "notes": build_tine_rhodes("main", rng)}},
        {"tool": "set_notes", "arguments": {"pattern": 19, "channel": 10, "notes": build_tine_rhodes("pont", rng)}},
        {"tool": "set_notes", "arguments": {"pattern": 20, "channel": 7, "notes": build_harmony_strings("climax", rng)}},
        {"tool": "set_notes", "arguments": {"pattern": 21, "channel": 6, "notes": [{"start": 0.0, "pitch": 60, "length": 1.0, "velocity": 66}, {"start": 8.0, "pitch": 60, "length": 1.0, "velocity": 66}]}},
        {"tool": "set_notes", "arguments": {"pattern": 21, "channel": 11, "notes": [{"start": 2.0, "pitch": 60, "length": 0.4, "velocity": 62}, {"start": 6.0, "pitch": 60, "length": 0.4, "velocity": 66}, {"start": 10.0, "pitch": 60, "length": 0.4, "velocity": 70}]}},
        {"tool": "set_notes", "arguments": {"pattern": 21, "channel": 4, "notes": snare_build}},
        {"tool": "set_notes", "arguments": {"pattern": 22, "channel": 12, "notes": [{"start": 12.56, "pitch": 60, "length": 3.44, "velocity": 95}]}},
        {"tool": "set_notes", "arguments": {"pattern": 23, "channel": 13, "notes": [{"start": 0.0, "pitch": 60, "length": 1.0, "velocity": 95}]}},
        {"tool": "set_notes", "arguments": {"pattern": 24, "channel": 14, "notes": [{"start": 14.75, "pitch": 60, "length": 1.2, "velocity": 92}]}},
    ]
    gl.batch_ops("Write All Notes", note_calls)

    # 5. Reconstruit l'arrangement complet 36 mesures (144 beats = 61.71s) en batch
    pl = gl.call("get_playlist")
    m_cnt = len(gl.call("get_markers")["markers"])
    clear_calls = [{"tool": "remove_clip", "arguments": {"clip": c["clip"]}} for c in pl["clips"]] + [
        {"tool": "remove_marker", "arguments": {"marker": 1}} for _ in range(m_cnt)
    ]
    if clear_calls:
        gl.batch_ops("Clear Playlist & Markers", clear_calls)

    markers = [
        ("Intro", 0),
        ("Drop 1", 16),
        ("Couplet", 48),
        ("Pont", 80),
        ("Drop 2 Climax", 96),
        ("Outro", 128),
    ]
    arrangement_clips = [
        # Track 1: Drums & 808
        (3, 1, 0), (4, 1, 16), (5, 1, 32), (6, 1, 48), (7, 1, 64), (21, 1, 80), (4, 1, 96), (5, 1, 112), (8, 1, 128),
        # Track 2: Polar Strings
        (1, 2, 0), (1, 2, 16), (1, 2, 32), (1, 2, 48), (1, 2, 64), (1, 2, 80), (20, 2, 96), (20, 2, 112), (10, 2, 128),
        # Track 3: Grand Piano (Headroom SFZ)
        (9, 3, 0), (2, 3, 16), (11, 3, 32), (12, 3, 48), (12, 3, 64), (13, 3, 80), (2, 3, 96), (11, 3, 112), (14, 3, 128),
        # Track 4: Nylon Pluck Counter-Arp
        (15, 4, 32), (16, 4, 48), (16, 4, 64), (15, 4, 96), (17, 4, 112),
        # Track 5: Classic Tine Rhodes
        (18, 5, 64), (19, 5, 80), (18, 5, 96), (18, 5, 112),
        # Track 6: Transitions Riser & Chants
        (22, 6, 0), (24, 6, 16), (22, 6, 64), (22, 6, 80), (24, 6, 96),
        # Track 7: Impact & Crash
        (23, 7, 16), (23, 7, 96), (23, 7, 128),
    ]
    arrange_calls = (
        [{"tool": "add_marker", "arguments": {"name": name, "position": pos}} for name, pos in markers]
        + [{"tool": "add_clip", "arguments": {"pattern": pat, "track": tr, "start": st}} for pat, tr, st in arrangement_clips]
        + [{"tool": "patch", "arguments": {"path": f"track/{tr_idx}", "set": {"name": tr_name}}} for tr_idx, tr_name in [
            (1, "Drums & 808"), (2, "Polar Strings"), (3, "Grand Piano SFZ"),
            (4, "Nylon Pluck"), (5, "Classic Tine"), (6, "Risers & Chants"), (7, "Drop Crash FX"),
        ]]
    )
    gl.batch_ops("Arrange 36 Bars", arrange_calls)

    gl.call("save_project", {})
    info = gl.call("get_song")
    print(f"Projet sauvegardé : {info['lengthBars']} bars, {info['lengthSeconds']:.2f}s, {info['counts']['channels']} canaux.")
    r = gl.call("render_audio", {"path": "master_final.wav", "scope": "song", "bits": 24, "sampleRate": 44100, "wait": True})
    print("Rendu master_final.wav :", r)
    gl.close()


if __name__ == "__main__":
    main()
