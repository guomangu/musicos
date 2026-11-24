#!/usr/bin/env python3
"""
v3.2 #4 `midi_groove_bank.py` — banque de grooves rythmiques extraits des `.mid` locaux
(`soundbanks/`), 0 GPU, jamais de mélodie copiée : seules les pistes **percussives**
(is_drum ou nom hat/snare/perc/kick/808/clap/tom/rim) sont minées.

- `build_groove_bank(soundbanks_dir, out_path)` : indexe chaque `.mid`/`.midi`, quantize les
  notes en pas de 1/16 sur 1-4 mesures, avec courbes de vélocité ; devine le genre depuis le
  chemin et le BPM depuis le méta tempo.
- `query_groove_bank(bank_path, role, bpm=None, max_bars=2)` : meilleur template (proximité
  BPM puis match de rôle).
- `groove_to_notes(template, start_beat, beats_per_bar, pitch)` : conversion en événements
  `{"start", "length", "pitch", "velocity"}` pour `song_builder`.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

DEFAULT_BANK_PATH = os.path.join(ROOT, "golden_references", "midi_groove_bank.json")

PERC_NAME_HINTS = ("hat", "snare", "perc", "kick", "808", "clap", "tom", "rim", "shaker", "drum", "ride", "cymbal")
ROLE_BY_HINT = [
    ("kick", "kick"), ("808", "bass_sub"), ("sub", "bass_sub"), ("snare", "snare_clap"), ("clap", "snare_clap"),
    ("hat", "hihat_perc"), ("perc", "hihat_perc"), ("shaker", "hihat_perc"), ("ride", "hihat_perc"), ("cymbal", "hihat_perc"),
    ("tom", "snare_clap"), ("rim", "hihat_perc"),
]
PROFILE_BY_PATH = [
    ("trap", "trap_rap"), ("drill", "trap_rap"), ("cloud", "trap_rap"), ("phonk", "lofi_chill"),
    ("lofi", "lofi_chill"), ("chillhop", "lofi_chill"), ("house", "edm_club"), ("techno", "edm_club"),
    ("edm", "edm_club"), ("dnb", "edm_club"), ("drum_and_bass", "edm_club"), ("rock", "rock_metal"),
    ("metal", "rock_metal"), ("pop", "pop_rnb"), ("rnb", "pop_rnb"), ("jazz", "jazz_acoustic"),
    ("classical", "classical_ambient"), ("ambient", "classical_ambient"),
]


def _guess_profile(path: str) -> str:
    low = path.lower().replace("\\", "/")
    for hint, prof in PROFILE_BY_PATH:
        if hint in low:
            return prof
    return "trap_rap"


def _guess_role(track_name: str, is_drum: bool) -> str | None:
    if not is_drum:
        low = track_name.lower()
        if not any(h in low for h in PERC_NAME_HINTS):
            return None  # piste mélodique/harmonique : JAMAIS copiée
    low = track_name.lower()
    for hint, role in ROLE_BY_HINT:
        if hint in low:
            return role
    return "hihat_perc" if is_drum else None


def _extract_track_template(mf: Any, track: Any, bpm: float | None) -> dict[str, Any] | None:
    name = getattr(track, "name", "") or ""
    is_drum = bool(getattr(track, "is_drum", False))
    role = _guess_role(name, is_drum)
    if role is None:
        return None
    tpb = mf.ticks_per_beat
    events: list[tuple[float, int, int]] = []  # (ticks, pitch, vel)
    abs_t = 0
    for msg in track:
        abs_t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            events.append((abs_t, msg.note, msg.velocity))
    if not events:
        return None
    # quantize au 1/16 et limite à 4 mesures
    step_ticks = tpb // 4
    max_ticks = tpb * 4 * 4
    quant: dict[int, tuple[int, int]] = {}
    for t, pitch, vel in events:
        if t >= max_ticks:
            break
        q = int(round(t / step_ticks))
        if q not in quant or vel > quant[q][1]:
            quant[q] = (pitch, vel)
    if not quant:
        return None
    bars = max(1, min(4, (max(quant) * step_ticks) // (tpb * 4) + 1))
    steps = [[q, p, v] for q, (p, v) in sorted(quant.items())]
    return {
        "role": role,
        "bars": bars,
        "bpm": round(float(bpm), 1) if bpm else None,
        "steps": steps,
        "n_steps": len(steps),
    }


def build_groove_bank(soundbanks_dir: str, out_path: str = DEFAULT_BANK_PATH) -> dict[str, Any]:
    """Indexe tous les `.mid`/`.midi` de `soundbanks_dir` en templates percussifs."""
    import mido

    bank: list[dict[str, Any]] = []
    root = os.path.abspath(os.path.expanduser(soundbanks_dir))
    n_files = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("__MACOSX", ".git")]
        for fn in sorted(filenames):
            if not fn.lower().endswith((".mid", ".midi")):
                continue
            path = os.path.join(dirpath, fn)
            if os.path.getsize(path) < 128:
                continue
            n_files += 1
            try:
                mf = mido.MidiFile(path)
            except Exception:
                continue
            bpm = None
            for tr in mf.tracks:
                for msg in tr:
                    if msg.type == "set_tempo":
                        bpm = round(mido.tempo2bpm(msg.tempo), 1)
                        break
                if bpm:
                    break
            for track in mf.tracks:
                tpl = _extract_track_template(mf, track, bpm)
                if tpl is None:
                    continue
                tpl["source"] = os.path.relpath(path, root)
                tpl["profile"] = _guess_profile(path)
                bank.append(tpl)
    payload = {"schema_version": "3.2", "source_dir": root, "n_midi_files": n_files, "templates": bank}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return {"out_path": out_path, "n_midi_files": n_files, "n_templates": len(bank)}


def query_groove_bank(bank_path: str = DEFAULT_BANK_PATH, role: str = "hihat_perc", bpm: float | None = None,
                      profile: str | None = None, max_bars: int = 2) -> dict[str, Any] | None:
    """Meilleur template pour un rôle (proximité BPM, puis match de profil/role)."""
    if not os.path.isfile(bank_path):
        return None
    try:
        with open(bank_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    cands = [t for t in data.get("templates", []) if t.get("role") == role and int(t.get("bars", 1)) <= max(1, max_bars)]
    if profile:
        pref = [t for t in cands if t.get("profile") == profile]
        if pref:
            cands = pref
    if not cands:
        return None
    if bpm:
        with_bpm = [t for t in cands if t.get("bpm")]
        if with_bpm:
            cands = sorted(with_bpm, key=lambda t: abs(float(t["bpm"]) - float(bpm)))
    cands.sort(key=lambda t: -int(t.get("n_steps", 0)))
    return cands[0]


def groove_to_notes(template: dict[str, Any], start_beat: float, beats_per_bar: float = 4.0,
                    pitch: int = 42, default_vel: int = 80) -> list[dict[str, Any]]:
    """Convertit un template (steps 1/16) en événements note dict pour song_builder."""
    notes: list[dict[str, Any]] = []
    for step, _p, vel in template.get("steps", []):
        notes.append({
            "start": round(start_beat + step * 0.25, 5),
            "length": 0.22,
            "pitch": int(pitch),
            "velocity": int(max(1, min(127, vel if vel else default_vel))),
        })
    return notes
