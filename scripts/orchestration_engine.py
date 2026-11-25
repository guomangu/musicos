#!/usr/bin/env python3
"""
v3.3 #2 — Moteur d'Orchestration & Comping (doc audit v3.3, Carence #2).

  - `apply_bass_motion(score_ir, mode, ...)` : ligne de basse conjointe & pédales
    harmoniques. Modes (`bass_mode`) : `root` | `stepwise_inversions` (1ers renversés /
    basses de passage diatoniques quand cela réduit un saut >= 5 st à 1-2 st, avec
    `bass_midi >= 31` et LIL respectés) | `tonic_pedal` (Intro) | `dominant_pedal`
    (Pré-refrain) | `walking`.
  - `apply_comping_to_score_ir(score_ir, section_spans, styles_map)` : 6 styles de
    réalisation harmonique déterministes (`comping_style` par section) : `sustained_pad`,
    `broken_arp_up`, `broken_arp_pendulum`, `rhythmic_comp`, `pulse_ostinato`,
    `waltz_bass_chord` (3/4, 6/8).
  - `default_comping_arc(sections)` : arc de texture automatique
    (Intro pad -> Couplet arpège -> Refrain comping syncopé -> Pont pad sur pédale -> Outro arpège).
  - `revoice_chords_with_lead(score_ir)` : voice-leading conscient du Lead (via
    `musicality_metrics.optimize_voice_leading_with_lead`) : évite le croisement soprano/Lead.
"""
from __future__ import annotations

import os
import sys
from typing import Any

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

BASS_MODES = ("root", "stepwise_inversions", "tonic_pedal", "dominant_pedal", "walking")
COMPING_STYLES = ("sustained_pad", "broken_arp_up", "broken_arp_pendulum", "rhythmic_comp", "pulse_ostinato", "waltz_bass_chord")

MIN_BASS_MIDI = 31


def _tracks(score_ir: Any) -> dict[str, Any]:
    return dict(getattr(score_ir, "tracks", {}) or {})


def _bass_track(score_ir: Any) -> Any | None:
    for key, t in _tracks(score_ir).items():
        role = str(getattr(t, "role", "") or key).lower()
        if any(k in role for k in ("bass", "808", "sub")):
            return t
    return None


def _chords_track(score_ir: Any) -> Any | None:
    for key, t in _tracks(score_ir).items():
        role = str(getattr(t, "role", "") or key).lower()
        if any(k in role for k in ("chord", "pad", "keys", "harmony")):
            return t
    return None


def _lead_track(score_ir: Any) -> Any | None:
    for key, t in _tracks(score_ir).items():
        role = str(getattr(t, "role", "") or key).lower()
        if any(k in role for k in ("lead", "melody", "hook")):
            return t
    return None


def _section_of(section_spans: list[Any] | None, bar: int) -> str:
    if section_spans:
        for sec in section_spans:
            name = str(getattr(sec, "name", "") or (sec.get("name") if isinstance(sec, dict) else "") or "")
            start = int(getattr(sec, "start_bar", 0) if not isinstance(sec, dict) else sec.get("start_bar", 0))
            bars = int(getattr(sec, "bars", 0) if not isinstance(sec, dict) else sec.get("bars", 0))
            if start <= bar < start + max(1, bars):
                return name.lower()
    return ""


def _norm(name: str) -> str:
    return str(name).lower().replace("_", "").replace("-", "").replace(" ", "")


def _mode_for_section(section_modes: dict[str, str], section_name: str, default: str) -> str:
    if not section_name:
        return default
    for key, mode in (section_modes or {}).items():
        if _norm(key) in _norm(section_name) or _norm(section_name) in _norm(key):
            return mode
    return default


def apply_bass_motion(score_ir: Any, mode: str = "root", section_spans: list[Any] | None = None,
                      section_modes: dict[str, str] | None = None, tonic_pc: int = 0,
                      beats_per_bar: int = 4, scale_pcs: list[int] | None = None) -> dict[str, Any]:
    """Applique le `bass_mode` sur la piste basse du ScoreIR (notes modifiées in-place).

    - `tonic_pedal` / `dominant_pedal` : toutes les notes de la section passent sur la pédale
      (pc tonique / dominante) à l'octave la plus proche, `bass_midi >= 31` garanti.
    - `stepwise_inversions` : un saut >= 5 st entre deux racines est réduit par 1er renversé
      (tierce à la basse) ou basse de passage diatonique à 1-2 st.
    - `walking` : une basse de passage diatonique est insérée entre deux racines.
    """
    if mode not in BASS_MODES:
        mode = "root"
    bt = _bass_track(score_ir)
    if bt is None:
        return {"mode": mode, "applied": 0, "reason": "no_bass_track"}
    notes = sorted(bt.notes, key=lambda n: (float(n.grid_beat), int(n.pitch)))
    if not notes:
        return {"mode": mode, "applied": 0}
    grid = getattr(score_ir, "harmonic_grid", []) or []
    scale = set(int(p) for p in (scale_pcs or [])) or None
    rep = {"mode": mode, "n_pedal": 0, "n_inversions": 0, "n_passing": 0, "applied": 0}
    bpb = max(1, int(beats_per_bar))

    def _chord_root_and_pcs(n: Any) -> tuple[int, list[int]]:
        ci = int(getattr(n, "chord_idx", 0) or 0)
        if 0 <= ci < len(grid):
            seg = grid[ci]
            return int(seg.get("root_pc", seg.get("chord_pcs", [int(n.pitch) % 12])[0])), [int(p) for p in seg.get("chord_pcs", [])]
        return int(n.pitch) % 12, [int(n.pitch) % 12]

    # pédales par section
    for n in notes:
        sec = _section_of(section_spans, int(float(n.grid_beat) // bpb))
        m = _mode_for_section(section_modes or {}, sec, mode)
        if m == "tonic_pedal":
            target_pc = tonic_pc
        elif m == "dominant_pedal":
            target_pc = (tonic_pc + 7) % 12
        else:
            continue
        p = int(n.pitch)
        cand = p
        while cand % 12 != target_pc:
            cand += 1 if cand < p + 12 else -1
        # octave la plus proche avec pc cible et >= MIN_BASS_MIDI
        best = None
        for m_c in range(max(MIN_BASS_MIDI, p - 12), p + 13):
            if m_c % 12 == target_pc and (best is None or abs(m_c - p) < abs(best - p)):
                best = m_c
        if best is not None and best != p:
            n.pitch = int(best)
            rep["n_pedal"] += 1

    # inversions / walking sur sauts >= 5 st
    if mode in ("stepwise_inversions", "walking"):
        notes = sorted(bt.notes, key=lambda n: (float(n.grid_beat), int(n.pitch)))
        for i in range(len(notes) - 1):
            a, b = notes[i], notes[i + 1]
            jump = int(b.pitch) - int(a.pitch)
            if abs(jump) < 5:
                continue
            root_pc, pcs = _chord_root_and_pcs(b)
            third_pc = (root_pc + 4) % 12 if len(pcs) >= 3 else (pcs[0] if pcs else int(b.pitch) % 12)
            # 1er renversé : tierce à la basse si à 1-2 st du précédent
            best_inv = None
            for m_c in range(max(MIN_BASS_MIDI, int(a.pitch) - 7), int(a.pitch) + 8):
                if m_c % 12 == third_pc and 1 <= abs(m_c - int(a.pitch)) <= 2:
                    best_inv = m_c
                    break
            if best_inv is not None and mode == "stepwise_inversions":
                b.pitch = int(best_inv)
                rep["n_inversions"] += 1
                rep["applied"] += 1
                continue
            # basse de passage diatonique entre a et b (±1 st d'abord, puis ±2 st)
            if scale:
                step = None
                for d in (1, 2):
                    for cand_step in (int(a.pitch) + (d if jump > 0 else -d),):
                        if cand_step % 12 in scale and MIN_BASS_MIDI <= cand_step and abs(cand_step - int(a.pitch)) <= 2:
                            step = cand_step
                            break
                    if step is not None:
                        break
                if step is not None and abs(step - int(a.pitch)) <= 2:
                    # insérer une note de passage à mi-chemin
                    mid_beat = (float(a.grid_beat) + float(a.duration_beats) + float(b.grid_beat)) / 2.0
                    _insert_passing(bt, mid_beat, step)
                    rep["n_passing"] += 1
                    rep["applied"] += 1
    rep["mode"] = mode
    return rep


def _insert_passing(track: Any, beat: float, pitch: int) -> None:
    from score_ir import NoteEventIR, ExpressionIR
    try:
        n = NoteEventIR(grid_beat=round(float(beat), 4), duration_beats=0.5, pitch=int(pitch),
                        velocity=88, chord_idx=0, expression=ExpressionIR())
        track.notes.append(n)
    except Exception:
        track.notes.append({"grid_beat": round(float(beat), 4), "duration_beats": 0.5,
                            "pitch": int(pitch), "velocity": 88})


def comping_pattern(pitches: list[int], style: str, beats_per_bar: int = 4, bar_start: float = 0.0) -> list[dict[str, Any]]:
    """Génère les événements d'une mesure pour un accord donné (`pitches` = sons de l'accord)."""
    if style not in COMPING_STYLES or not pitches:
        return [{"start": bar_start, "length": beats_per_bar - 0.1, "pitch": p, "velocity": 72} for p in pitches]
    bpb = max(1, int(beats_per_bar))
    ps = sorted(set(int(p) for p in pitches))
    if style == "sustained_pad":
        return [{"start": bar_start, "length": bpb - 0.1, "pitch": p, "velocity": 70} for p in ps]
    if style == "broken_arp_up":
        seq = ps + ps[:-1]
        return [{"start": bar_start + i, "length": 0.9, "pitch": seq[i % len(seq)], "velocity": 74 + (4 if i == 0 else 0)} for i in range(bpb)]
    if style == "broken_arp_pendulum":
        seq = ps + ps[::-1]
        return [{"start": bar_start + i, "length": 0.9, "pitch": seq[i % len(seq)], "velocity": 74 + (4 if i == 0 else 0)} for i in range(bpb)]
    if style == "rhythmic_comp":
        hits = [0.5, 2.5] if bpb == 4 else [bpb / 2.0]
        out = []
        for h in hits:
            for p in ps:
                out.append({"start": bar_start + h, "length": 0.4, "pitch": p, "velocity": 78})
        return out
    if style == "pulse_ostinato":
        out = []
        for i in range(bpb):
            for p in ps:
                out.append({"start": bar_start + i, "length": 0.85, "pitch": p, "velocity": 86 if i == 0 else 72})
        return out
    if style == "waltz_bass_chord":
        # 3/4 ou 6/8 : réponse accords temps 2-3 (la basse gère le temps 1)
        out = []
        for i in (1, 2):
            if i < bpb:
                for p in ps:
                    out.append({"start": bar_start + i, "length": 0.9, "pitch": p, "velocity": 76})
        return out
    return [{"start": bar_start, "length": bpb - 0.1, "pitch": p, "velocity": 70} for p in ps]


def apply_comping_to_score_ir(score_ir: Any, section_spans: list[Any] | None, styles_map: dict[str, str],
                              beats_per_bar: int = 4) -> dict[str, Any]:
    """Remplace la rythmique de la piste chords par section selon `styles_map`
    (section name -> comping_style)."""
    ct = _chords_track(score_ir)
    if ct is None:
        return {"applied": 0, "reason": "no_chords_track"}
    bpb = max(1, int(beats_per_bar))
    notes = sorted(ct.notes, key=lambda n: (float(n.grid_beat), int(n.pitch)))
    if not notes:
        return {"applied": 0}
    total_bars = int(max(float(n.grid_beat) for n in notes) // bpb) + 1
    kept: list[Any] = []   # mesures en sustained_pad : on garde les notes d'origine
    added: list[Any] = []  # mesures retravaillées : nouveaux événements du pattern
    applied = 0
    for bar in range(total_bars):
        bar_notes = [n for n in notes if int(float(n.grid_beat) // bpb) == bar]
        if not bar_notes:
            continue
        sec = _section_of(section_spans, bar)
        style = _section_style(styles_map, sec) or "sustained_pad"
        if style == "sustained_pad":
            kept.extend(bar_notes)
            continue
        applied += 1
        pat = comping_pattern([int(n.pitch) for n in bar_notes], style, bpb, bar * bpb)
        for ev in pat:
            added.append(_make_note(ev))
    if applied:
        ct.notes[:] = sorted(kept + added, key=lambda n: float(n.grid_beat if hasattr(n, "grid_beat") else n.get("grid_beat", 0.0)))
    return {"applied": applied, "styles": dict(styles_map or {})}


def _section_style(styles_map: dict[str, str], sec: str) -> str | None:
    for key, st in (styles_map or {}).items():
        if _norm(key) in _norm(sec) or _norm(sec) in _norm(key):
            return st
    return None


def _make_note(ev: dict[str, Any]) -> Any:
    from score_ir import NoteEventIR, ExpressionIR
    try:
        return NoteEventIR(grid_beat=float(ev["start"]), duration_beats=float(ev["length"]),
                           pitch=int(ev["pitch"]), velocity=int(ev.get("velocity", 72)),
                           chord_idx=0, expression=ExpressionIR())
    except Exception:
        return {"grid_beat": float(ev["start"]), "duration_beats": float(ev["length"]),
                "pitch": int(ev["pitch"]), "velocity": int(ev.get("velocity", 72))}


def default_comping_arc(sections: list[Any] | None) -> dict[str, str]:
    """Arc de texture automatique : Intro/Pont -> sustained_pad, Couplet -> broken_arp_up,
    Refrain/Climax -> rhythmic_comp (+ pulse_ostinato en edm_club), Outro -> broken_arp_pendulum."""
    arc: dict[str, str] = {}
    for sec in sections or []:
        name = str(getattr(sec, "name", "") or (sec.get("name") if isinstance(sec, dict) else "") or "").lower()
        if any(h in name for h in ("intro", "pont", "bridge", "break")):
            arc[name] = "sustained_pad"
        elif any(h in name for h in ("couplet", "verse")):
            arc[name] = "broken_arp_up"
        elif any(h in name for h in ("refrain", "chorus", "drop", "climax", "hook")):
            arc[name] = "rhythmic_comp"
        elif any(h in name for h in ("outro", "fin")):
            arc[name] = "broken_arp_pendulum"
    return arc


def revoice_chords_with_lead(score_ir: Any, lead_min_margin: int = 2) -> dict[str, Any]:
    """Évite le croisement soprano-accords / Lead : si la voix supérieure des accords atteint
    `lead_bar_min - margin`, elle est abaissée d'une octave (registre conservé, LIL OK)."""
    ct = _chords_track(score_ir)
    lt = _lead_track(score_ir)
    if ct is None or lt is None:
        return {"applied": 0, "reason": "no_chords_or_lead"}
    lead_notes = sorted(lt.notes, key=lambda n: float(n.grid_beat))
    if not lead_notes:
        return {"applied": 0}
    bpb = 4.0
    applied = 0
    for n in ct.notes:
        bar = float(n.grid_beat) // bpb
        lead_bar = [int(x.pitch) for x in lead_notes if float(x.grid_beat) // bpb == bar]
        if not lead_bar:
            continue
        lead_min = min(lead_bar)
        if int(n.pitch) >= lead_min - lead_min_margin and int(n.pitch) - 12 >= 36:
            n.pitch = int(n.pitch) - 12
            applied += 1
    return {"applied": applied, "lead_min_margin": lead_min_margin}
