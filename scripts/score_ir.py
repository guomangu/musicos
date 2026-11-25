#!/usr/bin/env python3
"""
v3.1 `score_ir.py` — Représentation Intermédiaire symbolique (ScoreIR), chantier IR-02.

Un `ScoreIR` sépare la **structure nominale** d'une note (`grid_beat`, `duration_beats`,
`pitch`, `chord_idx`, `note_role`, `lock_harmony`) de son **expression non destructive**
(`expression: {timing_offset_beats, velocity_delta, gate_scale}`). Les passes de groove
(swing, micro-strum, Haas pocket, anti-machine-gun) accumulent leurs décalages dans
`expression` et ne touchent jamais `grid_beat` ni `pitch`.

La compilation finale `compile_score_ir_to_notes` applique le **clamp de frontière
harmonique** (guard_ms = 20) : aucune note de rôle chords/pad/bass/counter ne peut dépasser
la fin de son segment d'accord (moins le garde) si elle crée un b2/triton ou sort de
l'accord suivant — sauf `note_role == "anticipation"` ou `allow_bleed=True`. Suit
`sanitize_note_overlaps` (garde 10 ms) + vérification post-groove (< 5 ms) : 0 clash b2 de
frontière, 0 chevauchement sur piste monophonique.

Schéma JSON : `schemas/score_ir.schema.json` (contrat §4.1 du doc v3.1).
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from groove_lib import _clamp_vel, ms_to_beats, sanitize_note_overlaps  # noqa: E402

SCHEMA_VERSION = "3.1"

SCALES: dict[str, list[int]] = {
    "ionian": [0, 2, 4, 5, 7, 9, 11],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "phrygian_dominant": [0, 1, 4, 5, 7, 8, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "blues": [0, 3, 5, 6, 7, 10],
}

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]

TONIC_TO_PC: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4,
    "E#": 5, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}

_ROMAN_DEGREE = {"i": 0, "ii": 1, "iii": 2, "iv": 3, "v": 4, "vi": 5, "vii": 6}

# Low Interval Limit (miroir de musicality_metrics.LOW_INTERVAL_LIMIT, stdlib-only ici) :
# intervalle (st) -> hauteur MIDI minimale autorisée pour la note la plus grave.
LOW_INTERVAL_LIMIT: dict[int, int] = {1: 60, 2: 60, 3: 52, 4: 52, 5: 48, 6: 48, 7: 43, 12: 36}

NOTE_ROLES = (
    "bass_root", "bass_passing", "chord_tone", "melody_chord",
    "melody_passing", "appoggiatura", "anticipation", "suspension",
)

CLAMP_ROLE_SUBSTRINGS = ("chord", "pad", "bass", "counter", "808", "sub")
MONO_ROLE_SUBSTRINGS = ("bass", "808", "sub", "lead_mono")

DEFAULT_GUARD_MS = 20.0


def parse_key_tonic_pc(key_str: str | None) -> int | None:
    if not key_str:
        return None
    m = re.match(r"^\s*([A-Ga-g])([#b]?)", str(key_str))
    if not m:
        return None
    return TONIC_TO_PC.get(m.group(1).upper() + m.group(2))


def resolve_mode(hc: dict[str, Any], key_str: str = "") -> str:
    mode = str((hc or {}).get("mode") or "").strip().lower().replace(" ", "_")
    if mode in SCALES:
        return mode
    if re.search(r"minor|aeolian|dorian|phrygian|locrian", str(key_str or "").lower()):
        return "minor"
    return "major"


def parse_roman_degree(symbol: str, tonic_pc: int) -> dict[str, Any] | None:
    """Parse un degré en chiffre romain ('i', 'bII', 'VI', 'V7', 'iim7b5' simplifié).

    La racine se lit sur la gamme majeure parallèle + altération (b/#) ; la qualité
    (majeure/mineure) vient de la casse du chiffre romain.
    """
    s = str(symbol).strip()
    alter = 0
    while s and s[0] in ("b", "#"):
        alter += -1 if s[0] == "b" else 1
        s = s[1:]
    ext = ""
    for e in ("maj7", "maj9", "m7b5", "dim7", "7", "9", "sus4", "dim", "aug", "6"):
        if s.lower().endswith(e):
            ext = e
            s = s[: -len(e)]
            break
    core = s.lower()
    if core not in _ROMAN_DEGREE:
        return None
    deg = _ROMAN_DEGREE[core]
    root = (tonic_pc + MAJOR_SCALE[deg] + alter) % 12
    major = s[:1].isupper() and not ext.startswith("m")
    if ext in ("dim", "dim7"):
        iv = [0, 3, 6] + ([9] if ext == "dim7" else [])
    elif ext == "aug":
        iv = [0, 4, 8]
    elif ext == "sus4":
        iv = [0, 5, 7]
    elif ext == "maj7" or ext == "maj9":
        iv = [0, 4, 7, 11]
    elif ext == "7" or ext == "9":
        iv = [0, 4, 7, 10] if major else [0, 3, 7, 10]
    elif ext == "m7b5":
        iv = [0, 3, 6, 10]
    elif ext == "6":
        iv = [0, 4, 7, 9] if major else [0, 3, 7, 9]
    else:
        iv = [0, 4, 7] if major else [0, 3, 7]
    pcs = sorted({(root + i) % 12 for i in iv})
    quality = "maj" if major else "min"
    if ext:
        quality = f"{quality}{ext}"
    return {"root_pc": root, "chord_pcs": pcs, "quality": quality, "symbol": str(symbol)}


def _section_progression_for(section_spans: list[Any] | None, section_progressions: dict[str, Any], bar: int) -> tuple[list[str], int] | None:
    """v3.2 NarrativeHarmony : (progression déclarée pour la section couvrant `bar`, start_bar)."""
    if not section_spans or not section_progressions:
        return None
    for span in section_spans:
        if isinstance(span, (tuple, list)) and len(span) >= 3:
            name, sb, nb = str(span[0]), int(span[1]), int(span[2])
        elif isinstance(span, dict):
            name, sb, nb = str(span.get("name", "")), int(span.get("start_bar", 0)), int(span.get("bars", 4))
        else:
            continue
        if sb <= bar < sb + nb:
            key = _SECTION_PROG_KEY.get(name.lower().replace("-", "_").replace(" ", "_"))
            if key is None:
                for k in _SECTION_PROG_KEY:
                    if k in name.lower().replace("-", "_").replace(" ", "_"):
                        key = k
                        break
            _entry = section_progressions.get(key)
            if isinstance(_entry, dict):
                _entry = _entry.get("degrees") or []
            if key and isinstance(_entry, list) and _entry:
                return [str(x) for x in _entry], sb
            return None
    return None


_SECTION_PROG_KEY = {
    "intro": "intro", "verse": "verse", "couplet": "verse", "pre_chorus": "pre_chorus",
    "pre_refrain": "pre_chorus", "chorus": "chorus", "refrain": "chorus", "drop": "chorus",
    "climax": "chorus", "hook": "chorus", "bridge": "bridge", "pont": "bridge",
    "outro": "outro", "break": "verse", "breakdown": "verse",
}


def harmonic_grid_from_core(
    harmonic_core: dict[str, Any] | None,
    bpm: float,
    beats_per_bar: int = 4,
    total_bars: int = 8,
    section_spans: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Construit la grille harmonique immuable (contrat §4.1) depuis `harmonic_core`.

    v3.2 : si `harmonic_core.section_progressions` déclare des variantes par section et que
    `section_spans` (name, start_bar, bars, kind) est fourni, chaque segment utilise la
    progression de SA section (NarrativeHarmony — fin de la boucle 4 mesures unique).
    """
    hc = dict(harmonic_core or {})
    key_str = str(hc.get("key") or "C")
    tonic_pc = parse_key_tonic_pc(key_str)
    if tonic_pc is None:
        tonic_pc = 0
    mode = resolve_mode(hc, key_str)
    scale = SCALES[mode]
    scale_pcs = sorted({(tonic_pc + iv) % 12 for iv in scale})
    prog = hc.get("degree_progression") or hc.get("chord_progression") or hc.get("chords") or []
    if not isinstance(prog, list) or not prog:
        prog = ["i", "VI", "III", "VII"] if mode in ("minor", "aeolian", "phrygian", "dorian", "locrian", "harmonic_minor") else ["I", "IV", "V", "I"]
    prog = [str(p) for p in prog]
    section_progressions = hc.get("section_progressions") if isinstance(hc.get("section_progressions"), dict) else {}
    bars_per_chord = max(1, int(hc.get("progression_bars") or 1))
    bpb = max(1, int(beats_per_bar))
    total_beats = max(1, int(total_bars)) * bpb
    seg_beats = bars_per_chord * bpb

    grid: list[dict[str, Any]] = []
    idx = 0
    pos = 0.0
    while pos < total_beats - 1e-6:
        bar = int(pos // bpb)
        sec = _section_progression_for(section_spans, section_progressions, bar)
        if sec is not None:
            sec_prog, sec_start_bar = sec
            local_idx = (bar - sec_start_bar) // bars_per_chord
            sym = sec_prog[local_idx % len(sec_prog)]
        else:
            sym = prog[idx % len(prog)]
        parsed = parse_roman_degree(sym, tonic_pc) or parse_roman_degree("i", tonic_pc)
        root_pc = parsed["root_pc"]
        chord_pcs = parsed["chord_pcs"]
        tension = sorted({
            pc for pc in scale_pcs
            if pc not in chord_pcs and min((pc - root_pc) % 12, (root_pc - pc) % 12) == 1
        })
        grid.append({
            "chord_idx": idx,
            "start_beat": round(pos, 5),
            "end_beat": round(min(total_beats, pos + seg_beats), 5),
            "symbol": str(sym),
            "root_pc": root_pc,
            "bass_pc": root_pc,
            "chord_pcs": chord_pcs,
            "scale_pcs": scale_pcs,
            "modal_tension_pcs": tension,
        })
        pos += seg_beats
        idx += 1
    return grid


def classify_role(track_key: str) -> str:
    rl = str(track_key).lower()
    if any(k in rl for k in ("bass", "808", "sub")):
        return "bass"
    if any(k in rl for k in ("chord", "pad", "keys", "harmony", "string", "rhodes")):
        return "chords"
    if any(k in rl for k in ("lead", "melody", "topline")):
        return "lead"
    if "counter" in rl:
        return "counter"
    if "remix" in rl:
        return "remix_stem"  # v3.3 #5 — scénario C (Remix / Sample Flip)
    if any(k in rl for k in ("vocal", "vox")):
        return "vocals"
    if any(k in rl for k in ("fx", "riser", "ear_candy", "sweep", "impact")):
        return "fx"
    return "drums"


def _is_clampable_role(role: str) -> bool:
    r = str(role).lower()
    return any(k in r for k in CLAMP_ROLE_SUBSTRINGS)


def _is_mono_role(role: str, monophonic: bool) -> bool:
    return bool(monophonic) or any(k in str(role).lower() for k in MONO_ROLE_SUBSTRINGS)


def segment_index_for_beat(grid: list[dict[str, Any]], beat: float) -> int:
    if not grid:
        return 0
    for seg in grid:
        # borne end EXCLUSIVE : une note commençant exactement sur un changement d'accord
        # appartient au NOUVEL accord (sinon clamp de frontière faux + faux clash b2).
        if float(seg["start_beat"]) - 1e-6 <= beat < float(seg["end_beat"]):
            return int(seg["chord_idx"])
    return int(grid[-1]["chord_idx"]) if beat >= float(grid[-1]["end_beat"]) else 0


ARTICULATIONS = ("normal", "legato", "staccato", "glide_to_next", "accent", "swell")
GLIDE_MS_RANGE = (35.0, 90.0)


@dataclass
class ExpressionIR:
    timing_offset_beats: float = 0.0
    velocity_delta: int = 0
    gate_scale: float = 1.0
    # v3.2 ExpressionArticulations : couche d'articulation par note (non destructive)
    articulation: str = "normal"
    vibrato_delay_beats: float = 0.0
    glide_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timing_offset_beats": round(float(self.timing_offset_beats), 5),
            "velocity_delta": int(self.velocity_delta),
            "gate_scale": round(float(self.gate_scale), 5),
            "articulation": str(self.articulation),
            "vibrato_delay_beats": round(float(self.vibrato_delay_beats), 5),
            "glide_ms": round(float(self.glide_ms), 2),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ExpressionIR":
        d = d or {}
        art = str(d.get("articulation", "normal") or "normal")
        return cls(
            timing_offset_beats=float(d.get("timing_offset_beats", 0.0) or 0.0),
            velocity_delta=int(d.get("velocity_delta", 0) or 0),
            gate_scale=float(d.get("gate_scale", 1.0) or 1.0),
            articulation=art if art in ARTICULATIONS else "normal",
            vibrato_delay_beats=float(d.get("vibrato_delay_beats", 0.0) or 0.0),
            glide_ms=float(d.get("glide_ms", 0.0) or 0.0),
        )


@dataclass
class NoteEventIR:
    grid_beat: float
    duration_beats: float
    pitch: int
    velocity: int = 80
    chord_idx: int = 0
    note_role: str = "chord_tone"
    lock_harmony: bool = False
    allow_bleed: bool = False
    expression: ExpressionIR = field(default_factory=ExpressionIR)

    def real_start(self) -> float:
        return max(0.0, float(self.grid_beat) + float(self.expression.timing_offset_beats))

    def real_end(self) -> float:
        return self.real_start() + float(self.duration_beats) * float(self.expression.gate_scale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_beat": round(float(self.grid_beat), 5),
            "duration_beats": round(float(self.duration_beats), 5),
            "pitch": int(self.pitch),
            "velocity": int(self.velocity),
            "chord_idx": int(self.chord_idx),
            "note_role": self.note_role,
            "lock_harmony": bool(self.lock_harmony),
            "allow_bleed": bool(self.allow_bleed),
            "expression": self.expression.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NoteEventIR":
        return cls(
            grid_beat=float(d.get("grid_beat", 0.0)),
            duration_beats=float(d.get("duration_beats", d.get("duration", 0.25))),
            pitch=int(d.get("pitch", 60)),
            velocity=int(d.get("velocity", 80)),
            chord_idx=int(d.get("chord_idx", 0)),
            note_role=str(d.get("note_role", "chord_tone")),
            lock_harmony=bool(d.get("lock_harmony", False)),
            allow_bleed=bool(d.get("allow_bleed", False)),
            expression=ExpressionIR.from_dict(d.get("expression")),
        )


@dataclass
class TrackIR:
    role: str
    monophonic: bool = False
    timbre_class: str = ""
    inharmonicity: float = 0.0
    notes: list[NoteEventIR] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "monophonic": bool(self.monophonic),
            "timbre_class": self.timbre_class,
            "inharmonicity": float(self.inharmonicity),
            "notes": [n.to_dict() for n in self.notes],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrackIR":
        return cls(
            role=str(d.get("role", "chords")),
            monophonic=bool(d.get("monophonic", False)),
            timbre_class=str(d.get("timbre_class", "")),
            inharmonicity=float(d.get("inharmonicity", 0.0) or 0.0),
            notes=[NoteEventIR.from_dict(n) for n in d.get("notes", [])],
        )


@dataclass
class ScoreIR:
    bpm: float
    time_signature: dict[str, int]
    key: str
    mode: str
    valence: float
    harmonic_grid: list[dict[str, Any]]
    tracks: dict[str, TrackIR] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bpm": float(self.bpm),
            "time_signature": dict(self.time_signature),
            "key": self.key,
            "mode": self.mode,
            "valence": float(self.valence),
            "harmonic_grid": self.harmonic_grid,
            "tracks": {k: t.to_dict() for k, t in self.tracks.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScoreIR":
        ts = d.get("time_signature") or {}
        return cls(
            bpm=float(d.get("bpm", 120.0)),
            time_signature={"numerator": int(ts.get("numerator", 4)), "denominator": int(ts.get("denominator", 4))},
            key=str(d.get("key", "C")),
            mode=str(d.get("mode", "minor")),
            valence=float(d.get("valence", 0.0)),
            harmonic_grid=list(d.get("harmonic_grid") or []),
            tracks={k: TrackIR.from_dict(t) for k, t in (d.get("tracks") or {}).items()},
            schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ScoreIR":
        with open(os.path.abspath(os.path.expanduser(path)), encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def _infer_note_role(role: str, pitch: int, seg: dict[str, Any]) -> str:
    pc = int(pitch) % 12
    if role == "bass":
        return "bass_root" if pc == int(seg.get("bass_pc", seg.get("root_pc", 0))) % 12 else "bass_passing"
    if role == "chords":
        return "chord_tone" if pc in set(seg.get("chord_pcs", [])) else "appoggiatura"
    if role == "lead":
        return "melody_chord" if pc in set(seg.get("chord_pcs", [])) else "melody_passing"
    if role == "counter":
        return "chord_tone" if pc in set(seg.get("chord_pcs", [])) else "melody_passing"
    return "chord_tone" if pc in set(seg.get("chord_pcs", [])) else "melody_passing"


def build_score_ir(
    harmonic_core: dict[str, Any] | None,
    meta: dict[str, Any],
    tracks: dict[str, list[dict[str, Any]]],
    channels: list[Any] | None = None,
    voices_by_role: dict[str, int] | None = None,
    section_spans: list[Any] | None = None,
) -> ScoreIR:
    """Construit un ScoreIR depuis les pistes plates + la grille harmonique du plan."""
    bpm = float(meta.get("bpm", 120.0))
    bpb = int(meta.get("beats_per_bar", 4))
    denom = int(meta.get("denominator", 4))
    total_bars = int(meta.get("total_bars", 8) or 8)
    hc = dict(harmonic_core or {})
    key_str = str(hc.get("key") or meta.get("tonic") or meta.get("key") or "C")
    mode = resolve_mode(hc, str(meta.get("key") or key_str))
    grid = harmonic_grid_from_core(hc, bpm=bpm, beats_per_bar=bpb, total_bars=total_bars, section_spans=section_spans)

    ch_by_key = {str(ch.track_key): ch for ch in (channels or [])}
    voices = dict(voices_by_role or {})
    if channels:
        for ch in channels:
            voices.setdefault(str(ch.track_key).lower(), int(getattr(ch, "voices", 4) or 4))

    score = ScoreIR(
        bpm=bpm,
        time_signature={"numerator": bpb, "denominator": denom},
        key=key_str,
        mode=mode,
        valence=float(meta.get("valence", 0.0)),
        harmonic_grid=grid,
    )
    for tkey, notes in tracks.items():
        role = classify_role(tkey)
        ch = ch_by_key.get(str(tkey))
        if ch is not None and getattr(ch, "role", None):
            role = str(ch.role)
        mono = role in ("bass", "lead", "vocals") or int(voices.get(str(tkey).lower(), voices.get(str(tkey), 4)) or 4) == 1
        tir = TrackIR(
            role=role,
            monophonic=_is_mono_role(role, mono),
            timbre_class=str(getattr(ch, "plugin", "") or "") if ch is not None else "",
        )
        for ev in notes or []:
            gb = float(ev.get("start", 0.0))
            dur = float(ev.get("length", ev.get("duration", 0.25)))
            cidx = segment_index_for_beat(grid, gb)
            seg = grid[cidx] if 0 <= cidx < len(grid) else (grid[0] if grid else {})
            tir.notes.append(NoteEventIR(
                grid_beat=gb,
                duration_beats=dur,
                pitch=int(ev.get("pitch", 60)),
                velocity=int(ev.get("velocity", 80)),
                chord_idx=cidx,
                note_role=_infer_note_role(role, int(ev.get("pitch", 60)), seg),
                lock_harmony=role in ("bass", "chords", "counter"),
            ))
        tir.notes.sort(key=lambda n: (n.grid_beat, n.pitch))
        score.tracks[str(tkey)] = tir
    return score


# ---------------------------------------------------------------------------
# Passes de groove en mode IR : n'écrivent QUE dans `expression`.
# ---------------------------------------------------------------------------

def apply_swing_ir(score_ir: ScoreIR, swing: float = 0.0, beats_per_bar: float = 4.0) -> int:
    if swing <= 0.0:
        return 0
    delay = max(0.0, min(0.75, float(swing))) * 0.25
    n = 0
    for tir in score_ir.tracks.values():
        for note in tir.notes:
            t = float(note.grid_beat)
            eighth_idx = t / 0.5
            if abs(eighth_idx - round(eighth_idx)) < 1e-3 and int(round(eighth_idx)) % 2 == 1:
                note.expression.timing_offset_beats = round(note.expression.timing_offset_beats + delay, 5)
                n += 1
    return n


def apply_micro_strum_ir(score_ir: ScoreIR, strum_ms: float = 15.0, bpm: float = 120.0,
                         inner_vel_drop: int = 8, chord_tol_beats: float = 0.04) -> int:
    total_spread = ms_to_beats(max(0.0, float(strum_ms)), bpm)
    n = 0
    for tir in score_ir.tracks.values():
        if tir.role in ("bass", "drums", "fx", "vocals"):
            continue
        notes = sorted(tir.notes, key=lambda x: (x.grid_beat, x.pitch))
        groups: list[list[NoteEventIR]] = []
        for note in notes:
            if groups and abs(note.grid_beat - groups[-1][0].grid_beat) <= chord_tol_beats:
                groups[-1].append(note)
            else:
                groups.append([note])
        for grp in groups:
            if len(grp) < 2 or len({x.pitch for x in grp}) < 2:
                continue
            grp.sort(key=lambda x: x.pitch)
            nv = len(grp)
            step = total_spread / float(nv - 1)
            for idx, note in enumerate(grp):
                offset = idx * step
                if offset <= 0.0:
                    continue
                note.expression.timing_offset_beats = round(note.expression.timing_offset_beats + offset, 5)
                orig_len = max(1e-3, float(note.duration_beats))
                note.expression.gate_scale = round(max(0.05, (orig_len - offset) / orig_len), 5) * float(note.expression.gate_scale)
                if 0 < idx < nv - 1 and inner_vel_drop > 0:
                    note.expression.velocity_delta -= int(inner_vel_drop)
                n += 1
    return n


def apply_haas_pocket_ir(score_ir: ScoreIR, bpm: float, profile: str = "trap_rap") -> int:
    from groove_lib import HAAS_POCKET_MS_PROFILES, _pocket_shift_for_track

    beat_ms = 60000.0 / float(bpm) if bpm else 500.0
    max_shift_ms = 0.05 * beat_ms
    n = 0
    for tkey, tir in score_ir.tracks.items():
        shift_ms = min(_pocket_shift_for_track(tkey, profile), max_shift_ms)
        if shift_ms <= 0.0:
            continue
        shift = ms_to_beats(shift_ms, bpm)
        for note in tir.notes:
            note.expression.timing_offset_beats = round(note.expression.timing_offset_beats + shift, 5)
            n += 1
    return n


def apply_anti_machine_gun_ir(score_ir: ScoreIR, beats_per_bar: float = 4.0,
                              vel_jitter_pct: float = 5.0, seed: int = 42,
                              humanize_sparse: bool = True) -> int:
    import random

    rng = random.Random(seed)
    frac = max(0.0, float(vel_jitter_pct)) / 100.0
    n = 0
    for tir in score_ir.tracks.values():
        notes = sorted(tir.notes, key=lambda x: (x.grid_beat, x.pitch))
        if not notes:
            continue
        in_rapid = [False] * len(notes)
        for i in range(len(notes) - 1):
            ioi = notes[i + 1].grid_beat - notes[i].grid_beat
            if 1e-5 < ioi <= 0.25 + 1e-4:
                in_rapid[i] = True
                in_rapid[i + 1] = True
        for i, note in enumerate(notes):
            t = float(note.grid_beat)
            bar_phase = t % max(1.0, float(beats_per_bar))
            beat_frac = bar_phase % 1.0
            if in_rapid[i]:
                if beat_frac < 0.06 or beat_frac > 0.94:
                    mult = 1.06 if (bar_phase < 0.06 or abs(bar_phase - 2.0) < 0.06) else 1.02
                elif abs(beat_frac - 0.5) < 0.06:
                    mult = 0.92
                else:
                    mult = 0.82
                mult *= 1.0 + rng.uniform(-frac, frac)
            elif humanize_sparse:
                if beat_frac < 0.06 or beat_frac > 0.94:
                    mult = 1.04 if (bar_phase < 0.06 or abs(bar_phase - 2.0) < 0.06) else 1.01
                elif abs(beat_frac - 0.5) < 0.06:
                    mult = 0.96
                else:
                    mult = 0.90
            else:
                continue
            base_v = float(note.velocity) + float(note.expression.velocity_delta)
            new_v = _clamp_vel(base_v * mult)
            note.expression.velocity_delta = int(round(new_v)) - int(note.velocity)
            n += 1
    return n


def apply_groove_to_score_ir(score_ir: ScoreIR, swing: float = 0.0, bpm: float | None = None,
                            profile: str = "trap_rap", beats_per_bar: float = 4.0,
                            seed: int = 42) -> dict[str, int]:
    """Ordre canonique v3.1 : strum -> anti-machine-gun -> haas -> swing (tout en `expression`)."""
    bpm = float(bpm if bpm is not None else score_ir.bpm)
    return {
        "strum": apply_micro_strum_ir(score_ir, strum_ms=15.0, bpm=bpm),
        "amg": apply_anti_machine_gun_ir(score_ir, beats_per_bar=beats_per_bar, humanize_sparse=True, seed=seed),
        "haas": apply_haas_pocket_ir(score_ir, bpm=bpm, profile=profile),
        "swing": apply_swing_ir(score_ir, swing=swing, beats_per_bar=beats_per_bar),
    }


# ---------------------------------------------------------------------------
# Compilation finale : clamp de frontière harmonique + sanitize + vérification.
# ---------------------------------------------------------------------------

def _clash_with_next(pc: int, cur_pcs: set[int], nxt: dict[str, Any]) -> bool:
    nxt_pcs = set(nxt.get("chord_pcs", []))
    if nxt_pcs == cur_pcs:
        return False
    if pc not in nxt_pcs:
        return True
    d = min((pc - int(nxt.get("bass_pc", 0))) % 12, (int(nxt.get("bass_pc", 0)) - pc) % 12)
    return d in (1, 6)


def compile_score_ir_to_notes(
    score_ir: ScoreIR,
    guard_ms: float = DEFAULT_GUARD_MS,
    overlap_guard_ms: float = 10.0,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Compile le ScoreIR en pistes plates avec clamp de frontière harmonique (obligatoire).

    v3.2 : une note mono portant `articulation="glide_to_next"` (808/bass) est compilée avec
    un chevauchement de glide court (`glide_ms` borné à [35, 90] ms) vers la note suivante,
    et cette paire est exemptée du coupe-chevauchement brutal de `sanitize_note_overlaps`.

    Retourne `(tracks, report)` où report contient la vérification post-groove :
    `boundary_bleeds`, `b2_boundary_clashes`, `mono_overlaps` (hors glides exemptés).
    """
    bpm = float(score_ir.bpm)
    guard_beats = ms_to_beats(float(guard_ms), bpm)
    grid = score_ir.harmonic_grid
    tracks: dict[str, list[dict[str, Any]]] = {}
    voices_by_role: dict[str, int] = {}
    boundary_bleeds = 0
    b2_clashes = 0
    glide_pairs: set[tuple[str, float]] = set()
    glide_count = 0

    for tkey, tir in score_ir.tracks.items():
        clampable = _is_clampable_role(tir.role)
        sorted_notes = sorted(tir.notes, key=lambda n: (n.grid_beat, n.pitch))
        out_notes: list[dict[str, Any]] = []
        for ni, note in enumerate(sorted_notes):
            real_start = note.real_start()
            real_end = note.real_end()
            is_glide = bool(tir.monophonic and note.expression.articulation == "glide_to_next")
            cidx = note.chord_idx if 0 <= note.chord_idx < len(grid) else segment_index_for_beat(grid, note.grid_beat)
            if clampable and note.note_role != "anticipation" and not note.allow_bleed and not is_glide and 0 <= cidx < len(grid):
                seg = grid[cidx]
                chord_end = float(seg["end_beat"])
                if cidx + 1 < len(grid):
                    nxt = grid[cidx + 1]
                    pc = int(note.pitch) % 12
                    if _clash_with_next(pc, set(seg.get("chord_pcs", [])), nxt):
                        allowed_end = chord_end - guard_beats
                        if real_end > allowed_end + 1e-6:
                            real_end = max(real_start + 0.02, allowed_end)
            if clampable and not is_glide and 0 <= cidx < len(grid) and cidx + 1 < len(grid):
                seg = grid[cidx]
                nxt = grid[cidx + 1]
                pc = int(note.pitch) % 12
                if real_end > float(seg["end_beat"]) + 1e-6 and _clash_with_next(pc, set(seg.get("chord_pcs", [])), nxt):
                    boundary_bleeds += 1
                    d = min((pc - int(nxt.get("bass_pc", 0))) % 12, (int(nxt.get("bass_pc", 0)) - pc) % 12)
                    if d == 1:
                        b2_clashes += 1
            # v3.2 : glide 808/bass — chevauchement mono court vers la note suivante
            if (tir.monophonic and note.expression.articulation == "glide_to_next"
                    and ni + 1 < len(sorted_notes)):
                nxt = sorted_notes[ni + 1]
                g_ms = min(max(float(note.expression.glide_ms or 60.0), GLIDE_MS_RANGE[0]), GLIDE_MS_RANGE[1])
                g_beats = ms_to_beats(g_ms, bpm)
                nxt_start = nxt.real_start()
                target_end = min(nxt_start + g_beats, real_end + g_beats)
                if target_end > real_end + 1e-6:
                    real_end = target_end
                    glide_pairs.add((tkey, round(real_start, 5)))
                    glide_count += 1
            length = max(0.02, real_end - real_start)
            vel = _clamp_vel(float(note.velocity) + float(note.expression.velocity_delta))
            out_notes.append({"start": round(real_start, 5), "length": round(length, 5), "pitch": int(note.pitch), "velocity": int(vel)})
        out_notes.sort(key=lambda n: (float(n["start"]), int(n["pitch"])))
        tracks[tkey] = out_notes
        voices_by_role[str(tkey).lower()] = 1 if tir.monophonic else 4

    tracks = sanitize_note_overlaps(tracks, bpm=bpm, guard_ms=overlap_guard_ms, voices_by_role=voices_by_role,
                                    exempt_pairs=glide_pairs)

    mono_overlaps = 0
    for tkey, tir in score_ir.tracks.items():
        if not tir.monophonic:
            continue
        ns = sorted(tracks.get(tkey, []), key=lambda n: (float(n["start"]), int(n["pitch"])))
        for i in range(len(ns) - 1):
            if float(ns[i]["start"]) + float(ns[i]["length"]) > float(ns[i + 1]["start"]) + 1e-6:
                if (tkey, round(float(ns[i]["start"]), 5)) in glide_pairs:
                    continue  # glide 808 volontaire (exempté)
                mono_overlaps += 1

    report = {
        "guard_ms": float(guard_ms),
        "boundary_bleeds": boundary_bleeds,
        "b2_boundary_clashes": b2_clashes,
        "mono_overlaps": mono_overlaps,
        "glide_pairs": glide_count,
        "passed": boundary_bleeds == 0 and b2_clashes == 0 and mono_overlaps == 0,
    }
    return tracks, report


def clamp_flat_tracks_at_boundaries(
    tracks: dict[str, list[dict[str, Any]]],
    harmonic_grid: list[dict[str, Any]],
    bpm: float,
    roles_map: dict[str, str] | None = None,
    guard_ms: float = DEFAULT_GUARD_MS,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Variante plate du clamp de frontière (pass post-transitions avant export)."""
    guard_beats = ms_to_beats(float(guard_ms), bpm)
    clamped = 0
    for tkey, notes in tracks.items():
        role = str((roles_map or {}).get(tkey) or classify_role(tkey))
        if not _is_clampable_role(role):
            continue
        for ev in notes:
            st = float(ev.get("start", 0.0))
            ln = float(ev.get("length", ev.get("duration", 0.25)))
            end = st + ln
            cidx = segment_index_for_beat(harmonic_grid, st)
            if not (0 <= cidx < len(harmonic_grid)) or cidx + 1 >= len(harmonic_grid):
                continue
            seg = harmonic_grid[cidx]
            nxt = harmonic_grid[cidx + 1]
            pc = int(ev.get("pitch", 60)) % 12
            if _clash_with_next(pc, set(seg.get("chord_pcs", [])), nxt):
                allowed = float(seg["end_beat"]) - guard_beats
                if end > allowed + 1e-6:
                    new_len = max(0.02, allowed - st)
                    if "length" in ev or "duration" not in ev:
                        ev["length"] = round(new_len, 5)
                    else:
                        ev["duration"] = round(new_len, 5)
                    clamped += 1
    return tracks, clamped


# ---------------------------------------------------------------------------
# Évaluation / autofix directement sur le ScoreIR (court-circuite _infer_roles).
# ---------------------------------------------------------------------------

def _low_interval_violations(tir: TrackIR) -> list[dict[str, Any]]:
    out = []
    if tir.role != "chords":
        return out
    by_beat: dict[float, list[NoteEventIR]] = {}
    for n in tir.notes:
        by_beat.setdefault(round(n.grid_beat, 4), []).append(n)
    for beat, ns in by_beat.items():
        if len(ns) < 2:
            continue
        pitches = sorted(int(n.pitch) for n in ns)
        lowest = pitches[0]
        for a, b in zip(pitches, pitches[1:]):
            interval = b - a
            limit = LOW_INTERVAL_LIMIT.get(interval)
            if limit is not None and lowest < limit:
                out.append({"beat": beat, "interval_st": interval, "lowest_midi": lowest, "limit": limit})
    return out


def _suspension_resolves(tir: TrackIR, note: NoteEventIR, seg: dict[str, Any]) -> bool:
    """v3.2 : une suspension (4->3, 9->8, b6->5) est valide si elle est résolue par mouvement
    conjoint descendant de 1-2 st dans les 0.5 a 1.0 beats suivants sur la même piste."""
    root = int(seg.get("root_pc", 0)) % 12
    pc = int(note.pitch) % 12
    sus_iv = {(pc - root) % 12}
    if not (sus_iv & {5, 2, 8}):  # 4, 9, b6 au-dessus de la racine
        return False
    step = 2 if (pc - root) % 12 == 2 else 1  # 9->8 = 2 st, 4->3 et b6->5 = 1 st
    t0 = float(note.grid_beat)
    for other in tir.notes:
        if other is note:
            continue
        dt = float(other.grid_beat) - t0
        if 0.5 - 1e-6 <= dt <= 1.0 + 1e-6 and int(other.pitch) == int(note.pitch) - step:
            return True
    return False


def evaluate_score_ir(score_ir: ScoreIR, thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    """Audit musicalité directement sur le ScoreIR : rôles explicites, `chord_idx` de la grille.

    Ne passe jamais par `_infer_roles()` — les rôles viennent du compilateur.
    """
    th = dict(thresholds or {})
    grid = score_ir.harmonic_grid
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    n_notes = 0
    for tkey, tir in score_ir.tracks.items():
        for note in tir.notes:
            n_notes += 1
            cidx = note.chord_idx if 0 <= note.chord_idx < len(grid) else segment_index_for_beat(grid, note.grid_beat)
            seg = grid[cidx] if 0 <= cidx < len(grid) else (grid[0] if grid else {})
            pc = int(note.pitch) % 12
            scale_pcs = set(seg.get("scale_pcs", []))
            chord_pcs = set(seg.get("chord_pcs", []))
            if note.note_role == "suspension" and _suspension_resolves(tir, note, seg):
                continue  # v3.2 : suspension 4->3 / 9->8 / b6->5 résolue dans 0.5-1.0 beat = chaleur licite
            if scale_pcs and pc not in scale_pcs and tir.role in ("lead", "counter", "chords", "bass"):
                sev = "error" if note.note_role in ("melody_chord", "chord_tone", "bass_root") and note.lock_harmony else "warning"
                entry = {"track": tkey, "beat": round(note.grid_beat, 4), "pitch": int(note.pitch), "pc": pc, "chord_idx": cidx, "role": note.note_role}
                (errors if sev == "error" else warnings).append({**entry, "kind": "out_of_scale"})
            if cidx + 1 < len(grid) and _is_clampable_role(tir.role) and note.note_role != "anticipation" and not note.allow_bleed:
                nxt = grid[cidx + 1]
                if _clash_with_next(pc, chord_pcs, nxt) and note.real_end() > float(seg["end_beat"]) + 1e-6:
                    d = min((pc - int(nxt.get("bass_pc", 0))) % 12, (int(nxt.get("bass_pc", 0)) - pc) % 12)
                    errors.append({"track": tkey, "beat": round(note.grid_beat, 4), "pitch": int(note.pitch),
                                   "kind": "b2_boundary_clash" if d == 1 else "boundary_clash", "chord_idx": cidx})
        if tir.monophonic:
            ns = sorted(tir.notes, key=lambda n: (n.grid_beat, n.pitch))
            for i in range(len(ns) - 1):
                if ns[i].real_end() > ns[i + 1].real_start() + 1e-6:
                    errors.append({"track": tkey, "beat": round(ns[i].grid_beat, 4), "kind": "mono_overlap", "pitch": int(ns[i].pitch)})
        for v in _low_interval_violations(tir):
            errors.append({"track": tkey, "kind": "low_interval_violation", **v})
    return {
        "passed": not errors,
        "roles_source": "score_ir",
        "n_notes": n_notes,
        "errors": errors,
        "warnings": warnings,
        "thresholds": th,
    }


def _nearest_scale_pitch(pitch: int, scale_pcs: set[int]) -> int:
    pc = pitch % 12
    if pc in scale_pcs:
        return pitch
    best, best_d = pitch, 99
    for q in sorted(scale_pcs):
        d = (q - pc) % 12
        if d > 6:
            d -= 12
        if abs(d) < abs(best_d):
            best, best_d = pitch + d, d
    return best


def autofix_score_ir(score_ir: ScoreIR, thresholds: dict[str, Any] | None = None,
                     guard_ms: float = DEFAULT_GUARD_MS) -> tuple[ScoreIR, dict[str, Any]]:
    """Réparations déterministes sur le ScoreIR (snap gamme, clamp frontière, chevauchements mono)."""
    grid = score_ir.harmonic_grid
    guard_beats = ms_to_beats(float(guard_ms), float(score_ir.bpm))
    changes: list[dict[str, Any]] = []
    for tkey, tir in score_ir.tracks.items():
        for note in tir.notes:
            cidx = note.chord_idx if 0 <= note.chord_idx < len(grid) else segment_index_for_beat(grid, note.grid_beat)
            seg = grid[cidx] if 0 <= cidx < len(grid) else (grid[0] if grid else {})
            pc = int(note.pitch) % 12
            scale_pcs = set(seg.get("scale_pcs", []))
            if scale_pcs and pc not in scale_pcs and tir.role in ("lead", "counter") and note.note_role not in ("appoggiatura", "anticipation"):
                new_pitch = _nearest_scale_pitch(int(note.pitch), scale_pcs)
                if new_pitch != int(note.pitch):
                    changes.append({"track": tkey, "beat": round(note.grid_beat, 4), "kind": "snap_to_scale",
                                    "from": int(note.pitch), "to": int(new_pitch)})
                    note.pitch = int(new_pitch)
                    pc = new_pitch % 12
            if cidx + 1 < len(grid) and _is_clampable_role(tir.role) and note.note_role != "anticipation" and not note.allow_bleed:
                nxt = grid[cidx + 1]
                if _clash_with_next(pc, set(seg.get("chord_pcs", [])), nxt):
                    allowed_end = float(seg["end_beat"]) - guard_beats
                    if note.real_end() > allowed_end + 1e-6:
                        new_len = max(0.02, allowed_end - note.real_start())
                        scale = new_len / max(1e-6, float(note.duration_beats) * float(note.expression.gate_scale))
                        note.expression.gate_scale = round(min(float(note.expression.gate_scale), max(0.02, scale)), 5)
                        changes.append({"track": tkey, "beat": round(note.grid_beat, 4), "kind": "boundary_clamp", "pitch": int(note.pitch)})
        if tir.monophonic:
            ns = sorted(tir.notes, key=lambda n: (n.grid_beat, n.pitch))
            for i in range(len(ns) - 1):
                if ns[i].real_end() > ns[i + 1].real_start() + 1e-6:
                    new_len = max(0.02, ns[i + 1].real_start() - ns[i].real_start())
                    ns[i].duration_beats = round(min(float(ns[i].duration_beats), new_len / max(1e-6, float(ns[i].expression.gate_scale))), 5)
                    changes.append({"track": tkey, "beat": round(ns[i].grid_beat, 4), "kind": "mono_overlap_trim", "pitch": int(ns[i].pitch)})
    return score_ir, {"changes": changes, "n_changes": len(changes)}
