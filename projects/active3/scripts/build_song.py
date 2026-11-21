"""Générateur ultra-léger CPU de 'Souls in Motion' (40 mesures @ 164 BPM, D# minor) utilisant EXCLUSIVEMENT les sons fournis dans projects/active3/Untitled.gridline."""

from __future__ import annotations

import json
import random
import subprocess
import time
from typing import Any

PROJ_PATH = "/home/gamo/Bureau/musicos/projects/active3/projet.gridline"
EXPORT_DIR = "/home/gamo/Bureau/musicos/projects/active3/renders"

MZ = "/home/gamo/Bureau/musicos/pack/JMizzy The Best Drum Kits/!Mike Zara's Stash"
DY = "/home/gamo/Bureau/musicos/pack/JMizzy The Best Drum Kits/DY 808 MAFIA KIT"
NM = "/home/gamo/Bureau/musicos/pack/JMizzy The Best Drum Kits/Nick Mira (Shows the Screen) Drumkit Re-Up to respect DMCA"
STEMS2 = "/home/gamo/Bureau/musicos/pack/Rossi - SOULS (Loop kit + MIDI) Pack/STEMS/2"

# Uniquement les sons préparés par l'utilisateur dans Untitled.gridline
SAMPLES = {
    "kick": f"{MZ}/Kicks/MZ Kick [50k].wav",
    "808_main": f"{MZ}/808s/MZ 808 [Blxckout].wav",
    "808_round": f"{MZ}/808s/MZ 808 [20 Min].wav",
    "808_analog": f"{MZ}/808s/MZ 808 [Analog].wav",
    "808_barely": f"{MZ}/808s/MZ 808 [Barely There].wav",
    "808_piano": f"{NM}/808s/NM - Piano 808.wav",
    "clap_main": f"{MZ}/Claps/Snare_Clap Layers/MZ Clap [Boston].wav",
    "clap_layer": f"{MZ}/Claps/Common Claps/MZ Clap [Faded].wav",
    "hat_main": f"{MZ}/Hi Hats/MZ Hi Hat [Ballin].wav",
    "hat_fast": f"{MZ}/Hi Hats/MZ Hi Hat [100].wav",
    "open_hat": f"{DY}/DY 4500 Hat.wav",
    "triangle": f"{DY}/DY Trap Triangle (1).wav",
    "perc": f"{MZ}/Percs/MZ Perc [Airy Hit].wav",
    "guitar_stem": f"{STEMS2}/2 - Guitar.wav",
    "piano_stem": f"{STEMS2}/2 - Piano.wav",
    "vox_blueface": f"{MZ}/Vox/MZ Vox [Blueface].wav",
    "vox_grrr": f"{MZ}/Vox/MZ Vox [Grrrrraahhh!].wav",
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
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "active3-builder", "version": "2.0"},
                    },
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
        for attempt in range(5):
            self._id += 1
            tid = self._id
            self.proc.stdin.write(
                json.dumps(
                    {"jsonrpc": "2.0", "id": tid, "method": "tools/call", "params": {"name": name, "arguments": arguments or {}}}
                )
                + "\n"
            )
            self.proc.stdin.flush()
            try:
                resp = self._wait_id(tid)
                if resp["result"].get("isError"):
                    raise RuntimeError(f"Tool {name} failed: {resp['result']}")
                content = resp["result"].get("content", [])
                if content and "text" in content[0]:
                    try:
                        parsed = json.loads(content[0]["text"])
                    except Exception:
                        return {"text": content[0]["text"]}
                    if isinstance(parsed, dict) and (parsed.get("ok") is False or parsed.get("error")):
                        raise RuntimeError(f"Tool {name} error: {parsed}")
                    return parsed
                return resp["result"]
            except RuntimeError as exc:
                if "Rate limit" in str(exc) and attempt < 4:
                    time.sleep(0.8)
                    continue
                raise
        raise RuntimeError("Unreachable")

    def batch_ops(self, calls: list[dict[str, Any]]) -> dict[str, Any]:
        res = {}
        for i in range(0, len(calls), 180):
            res = self.call("batch", {"calls": calls[i : i + 180]})
        return res

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=20)


def eq_changes_for(
    mixer: int,
    slot: int,
    bands: dict[int, tuple[int, float, float, float]],
    main_level: float = 0.0,
    quality: int = 0,
) -> list[dict[str, Any]]:
    width_params = {1: 53, 2: 55, 3: 57, 4: 59, 5: 61, 6: 63, 7: 65}
    en_params = {1: 38, 2: 40, 3: 42, 4: 44, 5: 46, 6: 48, 7: 50}
    out: list[dict[str, Any]] = [
        {"path": f"mixer/{mixer}/slot/{slot}/67", "value": quality},
        {"path": f"mixer/{mixer}/slot/{slot}/36", "value": main_level},
    ]
    for b_idx, (b_type, freq, gain, width) in bands.items():
        base = (b_idx - 1) * 5
        out.append({"path": f"mixer/{mixer}/slot/{slot}/{base + 1}", "value": b_type})
        out.append({"path": f"mixer/{mixer}/slot/{slot}/{base + 2}", "value": freq})
        out.append({"path": f"mixer/{mixer}/slot/{slot}/{base + 3}", "value": gain})
        out.append({"path": f"mixer/{mixer}/slot/{slot}/{width_params[b_idx]}", "value": width})
        out.append({"path": f"mixer/{mixer}/slot/{slot}/{en_params[b_idx]}", "value": 0 if b_type == 0 else 1})
    return out


def human_notes(
    raw: list[tuple[float, int, float, int] | tuple[float, int, float, int, float]],
    rng: random.Random,
    jitter_time: float = 0.006,
    vel_jit: int = 5,
) -> list[dict[str, Any]]:
    out = []
    for item in raw:
        st, pt, ln, vl = item[0], item[1], item[2], item[3]
        pan = item[4] if len(item) > 4 else None
        st_h = max(0.0, round(st + (rng.uniform(-jitter_time, jitter_time) if st > 0 else 0.0), 4))
        vl_h = max(22, min(124, vl + rng.randint(-vel_jit, vel_jit)))
        note: dict[str, Any] = {"start": st_h, "pitch": pt, "length": round(ln, 4), "velocity": vl_h}
        if pan is not None:
            note["pan"] = max(-64, min(64, int(round(pan * 64))))
        out.append(note)
    return out


def build_808_pattern(style: str, bars: int, rng: random.Random) -> list[dict[str, Any]]:
    """Lignes 808 en D#m (Root=24 C1 -> 27=D#1, 30=F#1, 32=G#1, 34=A#1, 35=B1, 39=D#2)."""
    raw: list[tuple[float, int, float, int]] = []
    cycles = bars // 4
    for cyc in range(cycles):
        off = cyc * 16.0
        if style == "drop":
            hits = [
                (0.0, 27, 1.35, 114), (1.5, 39, 0.70, 102), (2.75, 27, 0.95, 108),
                (4.0, 32, 1.35, 112), (5.5, 32, 0.45, 96), (6.0, 35, 1.40, 110), (7.5, 34, 0.45, 98),
                (8.0, 34, 1.35, 112), (9.5, 39, 0.85, 104), (10.75, 34, 0.95, 106),
                (12.0, 32, 1.40, 112), (14.0, 30, 0.90, 108),
                (15.0, 34 if cyc == 0 else 42, 0.45, 104),
                (15.5, 39 if cyc == 0 else 34, 0.45, 108),
            ]
        elif style == "verse_round":
            hits = [
                (0.0, 27, 1.60, 108), (2.5, 39, 0.65, 96),
                (4.0, 32, 1.40, 106), (6.0, 35, 1.40, 104),
                (8.0, 34, 1.60, 106), (10.75, 39, 0.65, 98),
                (12.0, 32, 1.40, 106), (14.0, 30, 0.90, 102), (15.25, 34, 0.65, 100),
            ]
        else:  # switch_analog
            hits = [
                (0.0, 27, 0.85, 114), (1.0, 39, 0.45, 100), (1.75, 27, 0.85, 108), (3.0, 34, 0.70, 102),
                (4.0, 32, 1.20, 112), (5.5, 35, 0.95, 106), (7.0, 32, 0.75, 100),
                (8.0, 34, 1.20, 112), (9.5, 39, 0.70, 104), (10.5, 34, 0.90, 106),
                (12.0, 32, 1.20, 110), (13.5, 30, 0.90, 106), (14.75, 34, 0.55, 104), (15.5, 39, 0.45, 108),
            ]
        for st, pt, ln, vl in hits:
            raw.append((off + st, pt, ln, vl))
    return human_notes(raw, rng, jitter_time=0.002, vel_jit=3)


def build_kick_pattern(style: str, bars: int, rng: random.Random) -> list[dict[str, Any]]:
    raw: list[tuple[float, int, float, int]] = []
    cycles = bars // 4
    for cyc in range(cycles):
        off = cyc * 16.0
        if style == "drop":
            beats = [0.0, 1.5, 2.75, 4.0, 5.5, 6.0, 8.0, 9.5, 10.75, 12.0, 14.0, 15.5]
        elif style == "verse_half":
            beats = [0.0, 8.0] if cyc == 0 else [0.0, 1.5, 2.75, 4.0, 6.0, 8.0, 9.5, 10.75, 12.0, 14.0, 15.25]
        else:  # switch
            beats = [0.0, 1.75, 4.0, 5.5, 8.0, 9.5, 10.5, 12.0, 13.5, 15.5]
        for b in beats:
            raw.append((off + b, 60, 0.35, 114 if b % 4 == 0 else 104))
    return human_notes(raw, rng, jitter_time=0.002, vel_jit=4)


def build_clap_snare(style: str, bars: int, rng: random.Random) -> tuple[list, list, list]:
    claps_m, claps_l, snares = [], [], []
    for bar in range(bars):
        off = bar * 4.0
        claps_m.append((off + 2.0, 60, 0.5, 110))
        if style in ("drop", "climax"):
            claps_l.append((off + 2.0, 60, 0.7, 94))
        if style in ("drop", "climax"):
            snares.extend([
                (off + 0.75, 60, 0.2, 62),
                (off + 1.5, 60, 0.22, 82),
                (off + 2.75, 60, 0.2, 68),
                (off + 3.5, 60, 0.22, 94),
            ])
            if bar % 4 == 3:
                snares.extend([(off + 3.0, 60, 0.18, 72), (off + 3.25, 60, 0.18, 86), (off + 3.75, 60, 0.18, 102)])
        elif style == "verse":
            snares.extend([(off + 0.75, 60, 0.2, 64), (off + 1.5, 60, 0.2, 74), (off + 2.75, 60, 0.2, 70), (off + 3.5, 60, 0.22, 88)])
            if bar == bars - 1:
                snares.extend([(off + 3.0, 60, 0.16, 78), (off + 3.25, 60, 0.16, 88), (off + 3.5, 60, 0.16, 98), (off + 3.75, 60, 0.16, 110)])
        elif style == "switch":
            snares.extend([(off + 0.75, 60, 0.2, 76), (off + 1.5, 60, 0.2, 82), (off + 2.75, 60, 0.2, 80), (off + 3.25, 60, 0.2, 88), (off + 3.5, 60, 0.2, 96)])
    return (
        human_notes(claps_m, rng, 0.002, 4),
        human_notes(claps_l, rng, 0.003, 4),
        human_notes(snares, rng, 0.004, 5),
    )


def build_hihats(style: str, bars: int, rng: random.Random) -> tuple[list, list, list, list, list]:
    hh_main, hh_fast, hh_nat, open_h, tri = [], [], [], [], []
    for bar in range(bars):
        off = bar * 4.0
        if style == "intro":
            for s in [0.0, 1.0, 2.0, 3.0]:
                hh_fast.append((off + s, 60, 0.15, 66 + int(s * 5), 0.0))
            if bar == 3:
                for k in range(6):
                    hh_fast.append((off + 3.0 + k / 6.0, 60, 0.1, 68 + k * 6, -0.2 + k * 0.08))
            continue

        # MZ Hi Hat [Ballin] : croches dynamiques
        for step in range(8):
            t = off + step * 0.5
            if style in ("drop", "climax") and (bar % 2 == 1) and step in (6, 7):
                continue
            vel = 98 if step % 2 == 0 else 70
            if step == 4:
                vel = 106
            hh_main.append((t, 60, 0.22, vel, -0.08 if step % 2 == 0 else 0.08))

        # Hat natif (gl.hat) : accents légers syncopés
        if style in ("drop", "climax", "verse"):
            hh_nat.extend([(off + 0.75, 60, 0.12, 62, -0.14), (off + 2.25, 60, 0.12, 68, 0.14)])

        # Rolls rapides sur MZ Hi Hat [100]
        if style == "drop":
            for k in range(3):
                hh_fast.append((off + 1.5 + k * (0.5 / 3.0), 60, 0.12, 78 + k * 8, -0.18 + k * 0.18))
            if bar % 2 == 1:
                for k in range(4):
                    hh_fast.append((off + 3.0 + k * 0.25, 60, 0.12, 76 + k * 9, 0.20 - k * 0.13))
            else:
                hh_fast.extend([(off + 2.75, 60, 0.12, 76, 0.15), (off + 3.75, 60, 0.12, 84, -0.15)])
        elif style == "climax":
            for k in range(3):
                hh_fast.append((off + 0.5 + k * (0.5 / 3.0), 60, 0.10, 76 + k * 10, -0.22 + k * 0.22))
            for k in range(4):
                hh_fast.append((off + 2.5 + k * 0.125, 60, 0.08, 74 + k * 8, 0.24 - k * 0.16))
            if bar % 2 == 1:
                for k in range(6):
                    hh_fast.append((off + 3.0 + k * (1.0 / 6.0), 60, 0.08, 72 + k * 7, -0.25 + k * 0.10))
        elif style in ("verse", "switch"):
            if bar % 2 == 1:
                for k in range(3):
                    hh_fast.append((off + 3.25 + k * (0.5 / 3.0), 60, 0.10, 74 + k * 9, -0.15 + k * 0.15))
            else:
                hh_fast.extend([(off + 1.75, 60, 0.10, 78, 0.12), (off + 3.75, 60, 0.10, 74, -0.12)])

        if style in ("drop", "climax") or (style == "verse" and bar >= 4) or style == "switch":
            open_h.append((off + 1.0, 60, 0.45, 86))
            if bar % 2 == 1:
                open_h.append((off + 3.5, 60, 0.35, 82))

        if bar % 2 == 1:
            tri.append((off + 3.5, 63, 0.65, 84))
            if style == "climax" and bar % 4 == 3:
                tri.append((off + 2.5, 63, 0.50, 78))

    return (
        human_notes(hh_main, rng, 0.003, 5),
        human_notes(hh_fast, rng, 0.002, 4),
        human_notes(hh_nat, rng, 0.003, 4),
        human_notes(open_h, rng, 0.003, 4),
        human_notes(tri, rng, 0.004, 4),
    )


def main() -> None:
    rng = random.Random(2026)
    gl = GridlineClient(PROJ_PATH, EXPORT_DIR)

    # 1. Config chanson : 164 BPM (4/4)
    gl.call("set_song", {"title": "Souls in Motion", "tempo": 164, "numerator": 4, "denominator": 4})

    # 2. Ajouter pistes mixer manquantes (17..23 -> 0..23)
    for m_idx in range(17, 24):
        gl.call("add_mixer_track", {"name": f"Track {m_idx}"})

    # 3. Uniquement les 19 sons fournis par l'utilisateur dans Untitled.gridline (0 ajout externe / 0 synthé électrique)
    channel_specs = [
        # (name, plugin, sample_path, mixer, pan, vol, color)
        ("MZ Kick [50k]", "gl.sampler", SAMPLES["kick"], 1, 0.0, 0.88, "#E05A47"),
        ("MZ 808 [Blxckout]", "gl.sampler", SAMPLES["808_main"], 2, 0.0, 0.80, "#C93B2B"),
        ("MZ 808 [20 Min]", "gl.sampler", SAMPLES["808_round"], 3, 0.0, 0.80, "#D65A31"),
        ("MZ 808 [Analog]", "gl.sampler", SAMPLES["808_analog"], 4, 0.0, 0.78, "#E07A38"),
        ("MZ 808 [Barely There]", "gl.sampler", SAMPLES["808_barely"], 5, 0.0, 0.78, "#B83B5E"),
        ("NM - Piano 808", "gl.sampler", SAMPLES["808_piano"], 6, 0.0, 0.78, "#A23E48"),
        ("MZ Clap [Boston]", "gl.sampler", SAMPLES["clap_main"], 7, 0.0, 0.64, "#D49B35"),
        ("MZ Clap [Faded]", "gl.sampler", SAMPLES["clap_layer"], 8, 0.05, 0.74, "#E2B14C"),
        ("Snare", "gl.snare", None, 9, -0.08, 0.76, "#D4B835"),
        ("MZ Hi Hat [Ballin]", "gl.sampler", SAMPLES["hat_main"], 10, -0.10, 0.82, "#6BBF59"),
        ("MZ Hi Hat [100]", "gl.sampler", SAMPLES["hat_fast"], 11, 0.12, 0.66, "#52B788"),
        ("Hat", "gl.hat", None, 12, -0.14, 0.68, "#74C69D"),
        ("DY 4500 Hat", "gl.sampler", SAMPLES["open_hat"], 13, -0.18, 0.76, "#40916C"),
        ("DY Trap Triangle (1)", "gl.sampler", SAMPLES["triangle"], 14, 0.22, 0.68, "#3AA6B9"),
        ("MZ Perc [Airy Hit]", "gl.sampler", SAMPLES["perc"], 15, -0.15, 0.78, "#4EA8DE"),
        ("2 - Guitar", "gl.sampler", SAMPLES["guitar_stem"], 16, -0.06, 1.05, "#B56576"),
        ("2 - Piano", "gl.sampler", SAMPLES["piano_stem"], 17, 0.06, 1.02, "#E56B6F"),
        ("MZ Vox [Blueface]", "gl.sampler", SAMPLES["vox_blueface"], 18, -0.10, 0.78, "#D96690"),
        ("MZ Vox [Grrrrraahhh!]", "gl.sampler", SAMPLES["vox_grrr"], 19, 0.0, 0.74, "#B86BD9"),
    ]

    gl.batch_ops([
        {"tool": "add_channel", "arguments": {"plugin": plug, "name": nm, "mixer": mx, "pan": pan, "volume": vol, "color": col}}
        for nm, plug, _, mx, pan, vol, col in channel_specs
    ])

    gl.batch_ops([
        {"tool": "patch", "arguments": {"path": f"channel/{idx}", "set": {"plugin": {"data": path}}}}
        for idx, (_, _, path, _, _, _, _) in enumerate(channel_specs, start=1)
        if path
    ])

    # Paramètres des canaux (accordage 808 + gl.snare & gl.hat propres)
    ch_params: list[dict[str, Any]] = []
    def qch(ch: int, pairs: list[tuple[int, Any]]) -> None:
        for p, v in pairs:
            ch_params.append({"path": f"channel/{ch}/{p}", "value": v})

    # Kick (1)
    qch(1, [(10, 1), (18, 1), (23, 1)])
    # 5 x 808 (ch 2..6): Root=24 (C1), Fine tune (-4, -4, -23, -4, +4 ct), mono Voices=1, One shot=0, Release=0.035, Glide=0.045
    for ch_i, ft in [(2, -4), (3, -4), (4, -23), (5, -4), (6, 4)]:
        qch(ch_i, [(1, 24), (2, ft), (6, 0.035), (10, 0), (18, 1), (23, 1), (46, 1), (47, 0.045)])
    # Claps, Hats, Percs one-shots
    for ch_i in [7, 8, 10, 11, 15]:
        qch(ch_i, [(10, 1), (18, 1), (23, 1)])
    for ch_i in [14, 19]:  # Triangle, Vox Grrrr -> Root=63
        qch(ch_i, [(1, 63), (10, 1), (18, 1), (23, 1)])
    # Snare natif (9): sec et punchy
    qch(9, [(1, 0.42), (2, 0.72), (3, 0.09), (4, 0.14), (5, 200.0), (6, 0.80), (9, 2600.0)])
    # Hat natif (12): tick court et doux
    qch(12, [(1, 0.045), (2, 7500.0), (4, 0.48)])
    # Open Hat (13) & Vox Blueface (18): gated
    qch(13, [(6, 0.06), (10, 0), (18, 1), (23, 1)])
    qch(18, [(1, 63), (6, 0.12), (10, 0), (18, 1), (23, 1)])
    # Stems 2 - Guitar (16) & 2 - Piano (17) : Root=63 (D#4), One shot=0, Release=0.06, Declick=2
    qch(16, [(1, 63), (6, 0.06), (10, 0), (18, 1), (23, 2)])
    qch(17, [(1, 63), (6, 0.06), (10, 0), (18, 1), (23, 2)])
    gl.call("set_parameters", {"changes": ch_params})

    # 4. Architecture Mixer ultra-sobre en CPU :
    #   BUS 808 (20)   <- ch 2..6
    #   BUS Drums (21) <- ch 1, 7..15
    #   BUS Music (22) <- ch 16..19
    #   FX Room (23)   <- send léger
    mixer_names = {
        1: ("MZ Kick", 0.96, 0.0, 21),
        2: ("808 Blxckout", 0.72, 0.0, 20),
        3: ("808 20Min", 0.72, 0.0, 20),
        4: ("808 Analog", 0.69, 0.0, 20),
        5: ("808 Barely", 0.70, 0.0, 20),
        6: ("NM Piano 808", 0.70, 0.0, 20),
        7: ("Clap Boston", 0.70, 0.0, 21),
        8: ("Clap Faded", 0.74, 0.05, 21),
        9: ("Snare Native", 0.78, -0.08, 21),
        10: ("Hat Ballin", 0.84, -0.10, 21),
        11: ("Hat 100", 0.70, 0.12, 21),
        12: ("Hat Native", 0.66, -0.14, 21),
        13: ("DY Open Hat", 0.72, -0.18, 21),
        14: ("Trap Triangle", 0.64, 0.22, 21),
        15: ("Airy Hit", 0.76, -0.15, 21),
        16: ("2 - Guitar", 1.16, -0.06, 22),
        17: ("2 - Piano", 1.06, 0.06, 22),
        18: ("Vox Blueface", 0.80, -0.10, 22),
        19: ("Vox Grrrr", 0.74, 0.0, 22),
        20: ("BUS 808", 0.96, 0.0, 0),
        21: ("BUS Drums", 0.98, 0.0, 0),
        22: ("BUS Music", 1.10, 0.0, 0),
        23: ("FX Room", 0.13, 0.0, 0),
    }
    gl.batch_ops([
        {"tool": "edit_mixer_track", "arguments": {"mixer": m_id, "name": nm, "volume": v, "pan": p, "routeTo": rt}}
        for m_id, (nm, v, p, rt) in mixer_names.items()
    ])

    # Sends : sidechain Kick -> BUS 808 (20) & BUS Music (22), et 1 FX Room partagé
    gl.batch_ops([
        {"tool": "set_send", "arguments": {"mixer": 1, "to": 20, "level": 0.95, "preFader": True, "sidechain": True}},
        {"tool": "set_send", "arguments": {"mixer": 1, "to": 22, "level": 0.28, "preFader": True, "sidechain": True}},
        {"tool": "set_send", "arguments": {"mixer": 8, "to": 23, "level": 0.05}},
        {"tool": "set_send", "arguments": {"mixer": 9, "to": 23, "level": 0.04}},
        {"tool": "set_send", "arguments": {"mixer": 14, "to": 23, "level": 0.05}},
        {"tool": "set_send", "arguments": {"mixer": 15, "to": 23, "level": 0.08}},
        {"tool": "set_send", "arguments": {"mixer": 16, "to": 23, "level": 0.06}},
        {"tool": "set_send", "arguments": {"mixer": 17, "to": 23, "level": 0.05}},
        {"tool": "set_send", "arguments": {"mixer": 18, "to": 23, "level": 0.06}},
        {"tool": "set_send", "arguments": {"mixer": 19, "to": 23, "level": 0.05}},
    ])

    # Plugins DSP minimaux (seulement 11 effets dans tout le projet!)
    gl.batch_ops([
        {"tool": "add_plugin", "arguments": {"mixer": 1, "plugin": "gl.eq", "slot": 1}},     # Kick
        {"tool": "add_plugin", "arguments": {"mixer": 16, "plugin": "gl.eq", "slot": 1}},    # 2 - Guitar
        {"tool": "add_plugin", "arguments": {"mixer": 17, "plugin": "gl.eq", "slot": 1}},    # 2 - Piano
        {"tool": "add_plugin", "arguments": {"mixer": 20, "plugin": "gl.comp", "slot": 1}},  # BUS 808 sidechain
        {"tool": "add_plugin", "arguments": {"mixer": 20, "plugin": "gl.eq", "slot": 2}},    # BUS 808 EQ
        {"tool": "add_plugin", "arguments": {"mixer": 21, "plugin": "gl.eq", "slot": 1}},    # BUS Drums EQ
        {"tool": "add_plugin", "arguments": {"mixer": 22, "plugin": "gl.eq", "slot": 1}},    # BUS Music EQ
        {"tool": "add_plugin", "arguments": {"mixer": 22, "plugin": "gl.comp", "slot": 2}},  # BUS Music sidechain
        {"tool": "add_plugin", "arguments": {"mixer": 23, "plugin": "gl.reverb", "slot": 1}},# FX Room
        {"tool": "add_plugin", "arguments": {"mixer": 0, "plugin": "gl.eq", "slot": 1}},     # Master EQ
        {"tool": "add_plugin", "arguments": {"mixer": 0, "plugin": "gl.comp", "slot": 2}},   # Master Glue
        {"tool": "add_plugin", "arguments": {"mixer": 0, "plugin": "gl.limiter", "slot": 3}},# Master Limiter
    ])

    fx_changes: list[dict[str, Any]] = [
        {"path": "mixer/17/stereo", "value": -0.22},  # 2 - Piano
        {"path": "mixer/19/stereo", "value": -0.45},  # Vox Grrrr
        {"path": "mixer/22/stereo", "value": -0.20},  # BUS Music
    ]
    # Kick EQ (Quality=0)
    fx_changes.extend(eq_changes_for(1, 1, {1: (1, 35.0, 0.0, 0.65), 2: (6, 85.0, 1.8, 0.45), 5: (6, 2800.0, 1.5, 0.45)}))
    # 2 - Guitar & 2 - Piano EQs (Quality=0, clarté naturelle mise en avant)
    fx_changes.extend(eq_changes_for(16, 1, {
        1: (1, 165.0, 0.0, 0.65),
        3: (6, 360.0, -3.0, 0.45),
        5: (6, 2600.0, 1.8, 0.45),
        6: (7, 6200.0, 2.2, 0.60),
    }, main_level=5.2))
    fx_changes.extend(eq_changes_for(17, 1, {
        1: (1, 185.0, 0.0, 0.65),
        3: (6, 320.0, -4.5, 0.45),
        5: (6, 2100.0, 1.6, 0.45),
        6: (7, 5500.0, 1.5, 0.60),
    }, main_level=4.2))
    # BUS 808 sidechain + EQ
    fx_changes.extend([
        {"path": "mixer/20/slot/1/1", "value": -20.0},
        {"path": "mixer/20/slot/1/2", "value": 4.5},
        {"path": "mixer/20/slot/1/3", "value": 0.002},
        {"path": "mixer/20/slot/1/4", "value": 0.085},
        {"path": "mixer/20/slot/1/5", "value": 4.0},
        {"path": "mixer/20/slot/1/9", "value": 2.0},
        {"path": "mixer/20/slot/1/10", "value": 1.0},
        {"path": "mixer/20/slot/1/15", "value": 1},
    ])
    fx_changes.extend(eq_changes_for(20, 2, {
        1: (1, 32.0, 0.0, 0.65),
        2: (6, 60.0, -2.0, 0.50),
        3: (6, 88.0, -2.8, 0.45),
        7: (3, 1800.0, 0.0, 0.65),
    }))
    # BUS Drums EQ
    fx_changes.extend(eq_changes_for(21, 1, {
        1: (1, 34.0, 0.0, 0.65),
        6: (7, 8800.0, 1.8, 0.60),
        7: (3, 17500.0, 0.0, 0.65),
    }))
    # BUS Music EQ + sidechain
    fx_changes.extend(eq_changes_for(22, 1, {
        1: (1, 160.0, 0.0, 0.65),
        3: (6, 380.0, -2.5, 0.42),
        4: (6, 510.0, -1.8, 0.42),
        7: (7, 8200.0, 1.8, 0.60),
    }))
    fx_changes.extend([
        {"path": "mixer/22/slot/2/1", "value": -16.0},
        {"path": "mixer/22/slot/2/2", "value": 2.5},
        {"path": "mixer/22/slot/2/3", "value": 0.004},
        {"path": "mixer/22/slot/2/4", "value": 0.095},
        {"path": "mixer/22/slot/2/10", "value": 1.0},
        {"path": "mixer/22/slot/2/15", "value": 1},
    ])
    # FX Room gl.reverb
    fx_changes.extend([
        {"path": "mixer/23/slot/1/1", "value": 16.0},
        {"path": "mixer/23/slot/1/2", "value": 0.16},
        {"path": "mixer/23/slot/1/5", "value": 550.0},
        {"path": "mixer/23/slot/1/6", "value": 4500.0},
        {"path": "mixer/23/slot/1/8", "value": 0.08},
        {"path": "mixer/23/slot/1/9", "value": 0.0},
        {"path": "mixer/23/slot/1/10", "value": 1.0},
        {"path": "mixer/23/slot/1/11", "value": 0.28},
        {"path": "mixer/23/slot/1/23", "value": 550},
        {"path": "mixer/23/slot/1/24", "value": 45},
        {"path": "mixer/23/slot/1/25", "value": 16},
        {"path": "mixer/23/slot/1/26", "value": 16},
        {"path": "mixer/23/slot/1/28", "value": 3},
        {"path": "mixer/23/slot/1/29", "value": 45},
        {"path": "mixer/23/slot/1/32", "value": 8},
        {"path": "mixer/23/slot/1/33", "value": 0},
        {"path": "mixer/23/slot/1/34", "value": 10},
        {"path": "mixer/23/slot/1/35", "value": 128},
        {"path": "mixer/23/slot/1/38", "value": 1},
    ])
    # Master EQ (Quality=1) + Comp + Limiter
    fx_changes.extend(eq_changes_for(0, 1, {
        1: (1, 32.0, 0.0, 0.65),
        2: (6, 62.0, -0.8, 0.55),
        3: (6, 340.0, -1.2, 0.42),
        4: (6, 500.0, -1.2, 0.45),
        5: (6, 2200.0, -0.6, 0.45),
        6: (7, 9200.0, 1.4, 0.60),
        7: (3, 17500.0, 0.0, 0.65),
    }, quality=1))
    fx_changes.extend([
        {"path": "mixer/0/slot/2/1", "value": -13.0},
        {"path": "mixer/0/slot/2/2", "value": 2.5},
        {"path": "mixer/0/slot/2/3", "value": 0.015},
        {"path": "mixer/0/slot/2/4", "value": 0.12},
        {"path": "mixer/0/slot/2/6", "value": 3.2},
        {"path": "mixer/0/slot/2/10", "value": 0.85},
        {"path": "mixer/0/slot/3/1", "value": 13.8},
        {"path": "mixer/0/slot/3/2", "value": -2.25},
        {"path": "mixer/0/slot/3/4", "value": 0.001},
        {"path": "mixer/0/slot/3/19", "value": 1},
        {"path": "mixer/0/slot/3/24", "value": 0.085},
    ])
    gl.call("set_parameters", {"changes": fx_changes})

    # 5. Patterns (1..10)
    pat_defs = [
        (1, "Intro Full", 4),
        (2, "Drop 1 Drums & 808", 8),
        (3, "Drop 1 Stems", 8),
        (4, "Couplet 1 Drums & 808", 8),
        (5, "Couplet 1 Stems", 8),
        (6, "Pont Melodique", 4),
        (7, "Drop 2 Climax Drums", 8),
        (8, "Drop 2 Climax Stems", 8),
        (9, "Couplet 2 Switch-Up", 4),
        (10, "Outro Soft", 4),
    ]
    pat_calls = [{"tool": "edit_pattern", "arguments": {"pattern": 1, "name": pat_defs[0][1], "bars": pat_defs[0][2]}}]
    for _, pname, pbars in pat_defs[1:]:
        pat_calls.append({"tool": "add_pattern", "arguments": {"name": pname, "bars": pbars}})
    gl.batch_ops(pat_calls)

    note_calls: list[dict[str, Any]] = []
    def wn(pat: int, ch: int, notes: list[dict[str, Any]]) -> None:
        if notes:
            note_calls.append({"tool": "set_notes", "arguments": {"pattern": pat, "channel": ch, "notes": notes}})

    # Pat 1: Intro (4 bars)
    wn(1, 17, [{"start": 0.0, "pitch": 63, "length": 15.8, "velocity": 98}])   # 2 - Piano
    wn(1, 16, [{"start": 8.0, "pitch": 63, "length": 7.8, "velocity": 76}])    # 2 - Guitar entre doux sur mes 3-4
    _, hh_intro, _, _, _ = build_hihats("intro", 4, rng)
    wn(1, 11, hh_intro)
    wn(1, 15, [{"start": 0.0, "pitch": 60, "length": 2.0, "velocity": 86}])   # Airy Hit
    wn(1, 19, [{"start": 15.0, "pitch": 63, "length": 0.9, "velocity": 90}])  # Vox Grrrr pickup

    # Pat 2 & 3: Drop 1 (8 bars)
    wn(2, 1, build_kick_pattern("drop", 8, rng))
    wn(2, 2, build_808_pattern("drop", 8, rng))
    cm, cl, sn = build_clap_snare("drop", 8, rng)
    wn(2, 7, cm); wn(2, 8, cl); wn(2, 9, sn)
    hm, hf, hn, oh, tr = build_hihats("drop", 8, rng)
    wn(2, 10, hm); wn(2, 11, hf); wn(2, 12, hn); wn(2, 13, oh); wn(2, 14, tr)

    wn(3, 16, [{"start": 0.0, "pitch": 63, "length": 31.85, "velocity": 106}])
    wn(3, 17, [{"start": 0.0, "pitch": 63, "length": 31.85, "velocity": 102}])

    # Pat 4 & 5: Couplet 1 (8 bars)
    wn(4, 1, build_kick_pattern("verse_half", 8, rng))
    wn(4, 3, build_808_pattern("verse_round", 8, rng))  # 808 20 Min
    cm, _, sn = build_clap_snare("verse", 8, rng)
    wn(4, 7, cm); wn(4, 9, sn)
    hm, hf, hn, oh, tr = build_hihats("verse", 8, rng)
    wn(4, 10, hm); wn(4, 11, hf); wn(4, 12, hn); wn(4, 13, oh); wn(4, 14, tr)

    wn(5, 16, [{"start": 0.0, "pitch": 63, "length": 31.85, "velocity": 106}])
    wn(5, 17, [{"start": 16.0, "pitch": 63, "length": 15.85, "velocity": 100}])
    wn(5, 18, [{"start": 28.0, "pitch": 63, "length": 3.8, "velocity": 88}])

    # Pat 6: Pont Mélodique (4 bars)
    wn(6, 17, [{"start": 0.0, "pitch": 63, "length": 15.8, "velocity": 100}])
    wn(6, 16, [{"start": 8.0, "pitch": 63, "length": 7.8, "velocity": 98}])
    wn(6, 5, [{"start": 0.0, "pitch": 27, "length": 3.8, "velocity": 104}])    # 808 Barely There grave soutenue
    wn(6, 15, [{"start": 0.0, "pitch": 60, "length": 2.0, "velocity": 88}])
    wn(6, 14, [{"start": 3.5, "pitch": 63, "length": 0.65, "velocity": 80}, {"start": 11.5, "pitch": 63, "length": 0.65, "velocity": 84}])
    snare_pont = [{"start": 12.0 + i * 0.5, "pitch": 60, "length": 0.16, "velocity": 56 + i * 6} for i in range(4)] + [
        {"start": 14.0 + i * 0.25, "pitch": 60, "length": 0.15, "velocity": 82 + i * 5} for i in range(8)
    ]
    wn(6, 9, snare_pont)
    wn(6, 19, [{"start": 15.0, "pitch": 63, "length": 0.9, "velocity": 92}])

    # Pat 7 & 8: Drop 2 Climax (8 bars)
    wn(7, 1, build_kick_pattern("drop", 8, rng))
    wn(7, 2, build_808_pattern("drop", 8, rng))
    cm, cl, sn = build_clap_snare("climax", 8, rng)
    wn(7, 7, cm); wn(7, 8, cl); wn(7, 9, sn)
    hm, hf, hn, oh, tr = build_hihats("climax", 8, rng)
    wn(7, 10, hm); wn(7, 11, hf); wn(7, 12, hn); wn(7, 13, oh); wn(7, 14, tr)
    wn(7, 15, [{"start": 0.0, "pitch": 60, "length": 2.0, "velocity": 84}])

    wn(8, 16, [{"start": 0.0, "pitch": 63, "length": 31.85, "velocity": 108}])
    wn(8, 17, [{"start": 0.0, "pitch": 63, "length": 31.85, "velocity": 104}])
    wn(8, 18, [{"start": 12.0, "pitch": 63, "length": 3.8, "velocity": 88}])

    # Pat 9: Couplet 2 Switch-Up (4 bars)
    wn(9, 1, build_kick_pattern("switch", 4, rng))
    sw_808 = build_808_pattern("switch_analog", 4, rng)
    # Alternance MZ 808 [Analog] (bars 1-2) et NM - Piano 808 (bars 3-4)
    wn(9, 4, [n for n in sw_808 if n["start"] < 8.0])
    wn(9, 6, [n for n in sw_808 if n["start"] >= 8.0])
    cm, _, sn = build_clap_snare("switch", 4, rng)
    wn(9, 7, cm); wn(9, 9, sn)
    hm, hf, hn, oh, tr = build_hihats("switch", 4, rng)
    wn(9, 10, hm); wn(9, 11, hf); wn(9, 13, oh); wn(9, 14, tr)
    wn(9, 16, [{"start": 0.0, "pitch": 63, "length": 15.8, "velocity": 104}])
    wn(9, 17, [{"start": 0.0, "pitch": 63, "length": 7.8, "velocity": 98}, {"start": 8.0, "pitch": 63, "length": 7.8, "velocity": 102}])

    # Pat 10: Outro Soft (4 bars)
    wn(10, 5, [{"start": 0.0, "pitch": 27, "length": 3.8, "velocity": 98}])
    wn(10, 15, [{"start": 0.0, "pitch": 60, "length": 2.0, "velocity": 80}])
    wn(10, 16, [{"start": 0.0, "pitch": 63, "length": 7.8, "velocity": 92}])
    wn(10, 17, [{"start": 0.0, "pitch": 63, "length": 15.8, "velocity": 94}])

    gl.batch_ops(note_calls)

    # 6. Arrangement 40 mesures (160 beats = 58.54 s)
    markers = [
        ("Intro", 0),
        ("Refrain / Drop 1", 16),
        ("Couplet 1", 48),
        ("Pont Melodique", 80),
        ("Refrain / Drop 2 Climax", 96),
        ("Couplet 2 (Switch-Up)", 128),
        ("Outro", 144),
    ]
    clips = [
        (2, 1, 16), (4, 1, 48), (7, 1, 96), (9, 1, 128),
        (1, 2, 0), (3, 2, 16), (5, 2, 48), (6, 2, 80), (8, 2, 96), (10, 2, 144),
    ]
    gl.batch_ops(
        [{"tool": "add_marker", "arguments": {"name": nm, "position": pos}} for nm, pos in markers]
        + [{"tool": "add_clip", "arguments": {"pattern": p, "track": t, "start": s}} for p, t, s in clips]
        + [
            {"tool": "patch", "arguments": {"path": "track/1", "set": {"name": "Drums & 808"}}},
            {"tool": "patch", "arguments": {"path": "track/2", "set": {"name": "Melodies & Vox"}}},
        ]
    )

    gl.call("save_project", {})
    info = gl.call("get_song")
    print(f"Sauvegarde OK : {info['lengthBars']} bars, {info['lengthSeconds']:.2f}s, {info['counts']['channels']} canaux.")
    r = gl.call("render_audio", {"path": "master_final.wav", "scope": "song", "bits": 24, "sampleRate": 44100, "wait": True})
    print("Rendu master_final.wav :", r)
    gl.close()


if __name__ == "__main__":
    main()
