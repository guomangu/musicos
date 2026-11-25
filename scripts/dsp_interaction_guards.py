#!/usr/bin/env python3
"""
v3.3 #3 — Garde-fous d'interaction DSP entre les mécaniques v3.2 (doc audit v3.3, Carence #3).

Contrat (doc v3.3 §6.3) :
  - `compute_probe_gap_bars(hall_decay_s, bpm, beats_per_bar) -> int`
      Tampon de silence dynamique entre le bloc noyau (8 mes.) et le bloc hors-noyau de la
      sonde `--probe` : N = max(2, ceil(T_hall * 1.25 * BPM / (60 * B))) — à 150 BPM avec
      FX Hall Decay 2.4 s, 1 mesure (1.6 s) ne suffit pas à éteindre la réverbération.
  - `remap_automations_for_probe(automations, slice_map) -> list[dict]`
      Remappe les automations (LPF, gain, vibrato_delay...) des plages sources vers les
      offsets de beats du projet sonde, sinon elles seraient absentes ou décalées.
  - `plan_safe_ritardando(score_ir, tracks_meta, ratio) -> dict`
      mode "tempo_ramp" | "symbolic_stretch" + `delay_mute_beat` (delay_send -> 0 une
      mesure... 1 beat avant toute variation de tempo sur FX Delay 21) + `loop_fade_beat`
      (fade-out des boucles audio avant le ritardando, anti-flam).
  - `extract_steady_state_masks(score_ir, bpm) -> dict[str, list[tuple[float, float]]]`
      Intervalles stables (secondes) par piste : les fenêtres de glide 808
      ([glide_start - 15 ms, glide_end + 25 ms]) et d'attaque de vibrato retardé sont
      EXCLUES de l'audit de justesse / chroma-vs-grille (anti faux positifs > 6 ct),
      la note d'arrivée du glide restant vérifiée à ±6 ct par l'évaluateur.
"""
from __future__ import annotations

import math
from typing import Any

GLIDE_PRE_MS = 15.0
GLIDE_POST_MS = 25.0
LOOP_ROLES = ("melodic_loop", "percussive_loop", "loop")


def compute_probe_gap_bars(hall_decay_s: float, bpm: float, beats_per_bar: int = 4) -> int:
    """Nombre de mesures de silence nécessaires pour éteindre FX Hall avant le bloc hors-noyau."""
    b = max(1, int(beats_per_bar))
    decay = max(0.0, float(hall_decay_s))
    bars = decay * 1.25 * max(1.0, float(bpm)) / (60.0 * b)
    return max(2, int(math.ceil(bars - 1e-9)))


def remap_automations_for_probe(automations: list[dict], slice_map: list[tuple[float, float, float]]) -> list[dict]:
    """Remappe chaque automation `{"beat": b, ...}` depuis sa plage source [src_start, src_end)
    vers `dst_start + (b - src_start)`. Les automations hors de toute plage sont conservées
    telles quelles et marquées `_remapped: False`."""
    out: list[dict] = []
    for auto in automations or []:
        if not isinstance(auto, dict):
            out.append(auto)
            continue
        new = dict(auto)
        beat = auto.get("beat", auto.get("bar"))
        if beat is None:
            new["_remapped"] = False
            out.append(new)
            continue
        b = float(beat) * 4.0 if auto.get("bar") is not None and auto.get("beat") is None else float(beat)
        remapped = False
        for src_start, src_end, dst_start in slice_map or []:
            if float(src_start) <= b < float(src_end):
                new["beat"] = round(float(dst_start) + (b - float(src_start)), 5)
                new["_remapped"] = True
                remapped = True
                break
        if not remapped:
            new["_remapped"] = False
        out.append(new)
    return out


def _tracks_of(score_ir: Any) -> dict[str, Any]:
    if hasattr(score_ir, "tracks"):
        return dict(score_ir.tracks or {})
    if isinstance(score_ir, dict):
        return dict(score_ir.get("tracks") or {})
    return {}


def _notes_of(track: Any) -> list[Any]:
    if hasattr(track, "notes"):
        return list(track.notes or [])
    if isinstance(track, dict):
        return list(track.get("notes") or [])
    return []


def _note_span_beats(n: Any) -> tuple[float, float]:
    start = float(getattr(n, "grid_beat", None) if getattr(n, "grid_beat", None) is not None else (n.get("start") if isinstance(n, dict) else 0.0) or 0.0)
    dur = getattr(n, "duration_beats", None)
    if dur is None and isinstance(n, dict):
        dur = n.get("duration_beats", n.get("length", 0.0))
    if dur is None:
        dur = 0.0
    return start, start + float(dur)


def _expr_of(n: Any) -> dict[str, Any]:
    e = getattr(n, "expression", None) if not isinstance(n, dict) else n.get("expression")
    if e is None:
        return {}
    if isinstance(e, dict):
        return e
    return {k: getattr(e, k) for k in ("articulation", "glide_ms", "vibrato_delay_beats", "vibrato_delay_ms") if hasattr(e, k)}


def _total_beats(score_ir: Any) -> float:
    total = 0.0
    for t in _tracks_of(score_ir).values():
        for n in _notes_of(t):
            s, e = _note_span_beats(n)
            total = max(total, e)
    if isinstance(score_ir, dict) and not _tracks_of(score_ir):
        total = max(total, float(score_ir.get("total_beats", 0.0) or 0.0))
    return total


def _is_audio_loop(track_key: str, meta: dict[str, Any] | None) -> bool:
    if meta:
        if meta.get("is_audio_loop") or meta.get("source_type") == "loop":
            return True
        role = str(meta.get("role") or "").lower()
        if any(lr in role for lr in LOOP_ROLES):
            return True
    return any(lr in str(track_key).lower() for lr in LOOP_ROLES)


def plan_safe_ritardando(score_ir: Any, tracks_meta: list[dict], ratio: float, bars: int = 2, beats_per_bar: int = 4) -> dict[str, Any]:
    """Décide d'un ritardando sûr : boucles audio actives dans la fenêtre -> ritardando
    symbolique (étirement des attaques MIDI, BPM global inchangé) + fade des boucles ;
    sinon rampe de tempo autorisée avec `delay_send -> 0` 1 beat avant (anti-artefact FX Delay 21)."""
    total = _total_beats(score_ir)
    bpb = max(1, int(beats_per_bar))
    win_start = max(0.0, total - max(1, int(bars)) * bpb)
    present = set(_tracks_of(score_ir).keys())
    meta_by_track = {str(m.get("track") or m.get("name") or ""): m for m in (tracks_meta or [])}
    loops = []
    for key in present:
        if _is_audio_loop(key, meta_by_track.get(key)):
            loops.append(key)
    # un méta peut aussi déclarer une boucle même si la piste n'est pas dans ce score_ir
    for key, m in meta_by_track.items():
        if key and key not in present and _is_audio_loop(key, m):
            loops.append(key)
    loops = sorted(set(loops))
    if loops:
        mode = "symbolic_stretch"
    else:
        mode = "tempo_ramp"
    return {
        "mode": mode,
        "ratio": round(float(ratio), 4),
        "window_beats": [round(win_start, 3), round(total, 3)],
        "loops_detected": loops,
        "loop_fade_beat": round(win_start, 3) if loops else None,
        "delay_mute_beat": round(max(0.0, win_start - 1.0), 3) if mode == "tempo_ramp" else None,
    }


def extract_steady_state_masks(score_ir: Any, bpm: float) -> dict[str, list[tuple[float, float]]]:
    """Intervalles stables par piste, en secondes : chaque note contribue son span moins les
    fenêtres expressives (glide [end - glide_s - 15 ms, end + 25 ms], attaque vibrato
    [start, start + vibrato_delay_ms])."""
    beat_s = 60.0 / max(1.0, float(bpm))
    masks: dict[str, list[tuple[float, float]]] = {}
    for key, track in _tracks_of(score_ir).items():
        intervals: list[tuple[float, float]] = []
        for n in _notes_of(track):
            s, e = _note_span_beats(n)
            if e <= s:
                continue
            spans = [(s, e)]
            expr = _expr_of(n)
            art = str(expr.get("articulation") or "").lower()
            glide_ms = expr.get("glide_ms")
            if art == "glide_to_next" and glide_ms:
                g = float(glide_ms) / 1000.0 / beat_s  # beats
                gs = max(s, e - g)
                spans = _subtract(spans, (gs - GLIDE_PRE_MS / 1000.0, e + GLIDE_POST_MS / 1000.0))
            vib_ms = expr.get("vibrato_delay_ms") or (float(expr.get("vibrato_delay_beats") or 0.0) * beat_s * 1000.0)
            if vib_ms:
                spans = _subtract(spans, (s, s + float(vib_ms) / 1000.0 / beat_s))  # beats
            for a, b in spans:
                if b - a > 0.02:
                    intervals.append((a * beat_s, b * beat_s))
        intervals.sort()
        merged: list[tuple[float, float]] = []
        for a, b in intervals:
            if merged and a <= merged[-1][1] + 1e-9:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        masks[str(key)] = [(round(a, 4), round(b, 4)) for a, b in merged]
    return masks


def _subtract(spans: list[tuple[float, float]], win: tuple[float, float]) -> list[tuple[float, float]]:
    wa, wb = win
    out: list[tuple[float, float]] = []
    for a, b in spans:
        if wb <= a or wa >= b:
            out.append((a, b))
            continue
        if a < wa:
            out.append((a, wa))
        if wb < b:
            out.append((wb, b))
    return out
