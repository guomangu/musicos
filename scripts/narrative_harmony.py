#!/usr/bin/env python3
"""
v3.2 #1 `narrative_harmony.py` — NarrativeHarmony (moteur de narration harmonique).

Fin du « syndrome de la boucle 4 mesures » :
  - progressions par section via `harmonic_core.section_progressions` (voir
    `score_ir.harmonic_grid_from_core(..., section_spans=...)`) ;
  - phrasé en période de 8 mesures Antécédent / Conséquent déclaré dans
    `arrangement_plan.json["phrasing_8bar"]` ;
  - opérateurs de transformation motivique déterministes sur le ScoreIR :
    `motif_answer` (réponse cadentielle mesure 8), `rhythmic_lift` (syncope au refrain),
    `call_and_response` (le Counter répond dans les silences exacts du Lead),
    `melodic_climax_arch` (un seul sommet émotionnel par période de 8 mesures) ;
  - `mark_suspensions` : marque `note_role="suspension"` les 4->3 / 9->8 / b6->5 résolues
    en 0.5-1.0 beat (chaleur mélodique licite, `score_ir._suspension_resolves`).
"""
from __future__ import annotations

import os
import sys
from typing import Any

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from score_ir import NoteEventIR, ScoreIR, segment_index_for_beat  # noqa: E402

_CHORUS_KINDS = ("chorus", "refrain", "drop", "climax", "hook")


def _lead_key(score_ir: ScoreIR) -> str | None:
    for k, t in score_ir.tracks.items():
        if t.role == "lead" and t.notes:
            return k
    return None


def _counter_key(score_ir: ScoreIR, lead_key: str | None) -> str | None:
    for k, t in score_ir.tracks.items():
        if t.role == "counter" and t.notes and k != lead_key:
            return k
    return None


def motif_answer(score_ir: ScoreIR, period_bars: int = 8) -> int:
    """Réponse cadentielle : la dernière note lead de chaque période de 8 mesures est
    projetée sur une note d'accord du segment final (résolution sur degré 1 ou 3)."""
    lead = _lead_key(score_ir)
    if lead is None:
        return 0
    bpb = float(score_ir.time_signature.get("numerator", 4))
    tir = score_ir.tracks[lead]
    total = max((n.grid_beat + n.duration_beats for n in tir.notes), default=0.0)
    n = 0
    period_beats = period_bars * bpb
    p = 0.0
    while p < total:
        pend = p + period_beats
        in_period = [x for x in tir.notes if p <= x.grid_beat < pend]
        if in_period:
            last = max(in_period, key=lambda x: (x.grid_beat, x.pitch))
            cidx = segment_index_for_beat(score_ir.harmonic_grid, last.grid_beat)
            seg = score_ir.harmonic_grid[cidx] if 0 <= cidx < len(score_ir.harmonic_grid) else {}
            chord_pcs = set(seg.get("chord_pcs", []))
            if chord_pcs and (last.pitch % 12) not in chord_pcs:
                near = min(sorted(chord_pcs), key=lambda q: min((last.pitch % 12 - q) % 12, (q - last.pitch % 12) % 12))
                d = (near - last.pitch % 12) % 12
                if d > 6:
                    d -= 12
                if abs(d) <= 2:
                    last.pitch = int(last.pitch + d)
                    last.note_role = "melody_chord"
                    n += 1
        p += period_beats
    return n


def rhythmic_lift(score_ir: ScoreIR, section_spans: list[Any] | None = None) -> int:
    """Syncope au refrain : dans les sections chorus/drop/climax, chaque 2e croche off-beat
    du lead est avancée d'un 1/16 (lift), sans toucher aux temps forts."""
    lead = _lead_key(score_ir)
    if lead is None:
        return 0
    bpb = float(score_ir.time_signature.get("numerator", 4))
    chorus_bars: set[int] = set()
    if section_spans:
        for span in section_spans:
            if isinstance(span, (tuple, list)) and len(span) >= 3:
                name, sb, nb, kind = str(span[0]), int(span[1]), int(span[2]), str(span[3] if len(span) > 3 else span[0])
            elif isinstance(span, dict):
                name, sb, nb, kind = str(span.get("name", "")), int(span.get("start_bar", 0)), int(span.get("bars", 4)), str(span.get("kind", ""))
            else:
                continue
            hay = (name + " " + kind).lower()
            if any(c in hay for c in _CHORUS_KINDS):
                chorus_bars.update(range(sb, sb + nb))
    if not chorus_bars:
        chorus_bars = set(range(int(max((n.grid_beat for t in score_ir.tracks.values() for n in t.notes), default=0.0) // (8 * bpb)) + 1))
    tir = score_ir.tracks[lead]
    n = 0
    off = 0
    for note in sorted(tir.notes, key=lambda x: x.grid_beat):
        bar = int(note.grid_beat // bpb)
        beat_in_bar = note.grid_beat % bpb
        eighth_idx = round(beat_in_bar * 2)
        if bar in chorus_bars and abs(beat_in_bar * 2 - eighth_idx) < 1e-3 and eighth_idx % 2 == 1:
            off += 1
            if off % 2 == 0:
                note.grid_beat = round(max(0.0, note.grid_beat - 0.25), 5)
                n += 1
    return n


def call_and_response(score_ir: ScoreIR, max_gap_beats: float = 2.0, answer_len: float = 0.5) -> int:
    """Le Counter répond dans les silences >= `max_gap_beats` du Lead : 2 notes courtes
    (accord courant) au début du silence. Jamais de chevauchement avec le lead."""
    lead = _lead_key(score_ir)
    counter = _counter_key(score_ir, lead)
    if lead is None or counter is None:
        return 0
    ltir, ctir = score_ir.tracks[lead], score_ir.tracks[counter]
    lead_notes = sorted(ltir.notes, key=lambda x: x.grid_beat)
    if len(lead_notes) < 2:
        return 0
    total = max(n.grid_beat + n.duration_beats for n in lead_notes)
    added = 0
    for a, b in zip(lead_notes, lead_notes[1:]):
        gap_start = a.grid_beat + a.duration_beats
        gap = b.grid_beat - gap_start
        if gap >= max_gap_beats and gap_start + answer_len <= total:
            cidx = segment_index_for_beat(score_ir.harmonic_grid, gap_start)
            seg = score_ir.harmonic_grid[cidx] if 0 <= cidx < len(score_ir.harmonic_grid) else {}
            chord_pcs = sorted(seg.get("chord_pcs", [])) or [0]
            base = 60 + chord_pcs[0]
            for k, pc in enumerate(chord_pcs[:2]):
                ctir.notes.append(NoteEventIR(
                    grid_beat=round(gap_start + k * 0.5, 5),
                    duration_beats=answer_len,
                    pitch=base + ((pc - chord_pcs[0]) % 12),
                    velocity=64,
                    chord_idx=cidx,
                    note_role="melody_passing",
                    lock_harmony=False,
                ))
                added += 1
    ctir.notes.sort(key=lambda x: (x.grid_beat, x.pitch))
    return added


def melodic_climax_arch(score_ir: ScoreIR, period_bars: int = 8) -> int:
    """Un seul sommet émotionnel par période de 8 mesures : l'apex (note la plus haute)
    est placée dans le conséquent (mesures 5-8) et accentuée ; les ex-aequo sont abaissés
    d'un demi-ton si la gamme le permet."""
    lead = _lead_key(score_ir)
    if lead is None:
        return 0
    bpb = float(score_ir.time_signature.get("numerator", 4))
    tir = score_ir.tracks[lead]
    total = max((n.grid_beat + n.duration_beats for n in tir.notes), default=0.0)
    period_beats = period_bars * bpb
    n = 0
    p = 0.0
    while p < total:
        pend = p + period_beats
        in_period = [x for x in tir.notes if p <= x.grid_beat < pend]
        if len(in_period) >= 3:
            apex = max(in_period, key=lambda x: x.pitch)
            apex_bar = apex.grid_beat // bpb
            # déplacer l'apex dans le conséquent si besoin
            if apex_bar < p / bpb + period_bars / 2:
                target = round(p + (period_bars - 1) * bpb, 5)
                if all(abs(x.grid_beat - target) > 0.24 for x in tir.notes if x is not apex):
                    apex.grid_beat = target
                    n += 1
            apex.expression.velocity_delta += 12
            # casser les ex-aequo de hauteur
            for x in in_period:
                if x is not apex and x.pitch == apex.pitch:
                    cidx = segment_index_for_beat(score_ir.harmonic_grid, x.grid_beat)
                    seg = score_ir.harmonic_grid[cidx] if 0 <= cidx < len(score_ir.harmonic_grid) else {}
                    if (x.pitch - 1) % 12 in set(seg.get("scale_pcs", [])):
                        x.pitch -= 1
                        n += 1
        p += period_beats
    return n


OPERATORS = {
    "motif_answer": motif_answer,
    "rhythmic_lift": rhythmic_lift,
    "call_and_response": call_and_response,
    "melodic_climax_arch": melodic_climax_arch,
}


def apply_narrative_harmony(
    score_ir: ScoreIR,
    plan: dict[str, Any] | None = None,
    section_spans: list[Any] | None = None,
) -> dict[str, Any]:
    """Applique les opérateurs déclarés dans `plan["phrasing_8bar"]["operators"]`
    (période 8 mesures Antécédent/Conséquent). Déterministe, 0 rendu audio."""
    plan = plan or {}
    phr = plan.get("phrasing_8bar") or {}
    period_bars = int(phr.get("period_bars") or phr.get("consequent_bars", 0) + phr.get("antecedent_bars", 0) or 8)
    if period_bars <= 0:
        period_bars = 8
    ops = [str(o) for o in (phr.get("operators") or ["motif_answer", "melodic_climax_arch"])]
    report: dict[str, Any] = {"period_bars": period_bars, "operators": {}}
    for op in ops:
        fn = OPERATORS.get(op)
        if fn is None:
            report["operators"][op] = "unknown"
            continue
        if op == "rhythmic_lift":
            report["operators"][op] = fn(score_ir, section_spans=section_spans)
        elif op in ("motif_answer", "melodic_climax_arch"):
            report["operators"][op] = fn(score_ir, period_bars=period_bars)
        else:
            report["operators"][op] = fn(score_ir)
    report["suspensions_marked"] = mark_suspensions(score_ir)
    return report


def mark_suspensions(score_ir: ScoreIR) -> int:
    """Marque `note_role="suspension"` les notes hors accord de type 4/9/b6 (au-dessus de la
    racine d'accord) résolues par mouvement conjoint descendant en 0.5-1.0 beat."""
    n = 0
    for tir in score_ir.tracks.values():
        for note in tir.notes:
            cidx = note.chord_idx if 0 <= note.chord_idx < len(score_ir.harmonic_grid) else segment_index_for_beat(score_ir.harmonic_grid, note.grid_beat)
            seg = score_ir.harmonic_grid[cidx] if 0 <= cidx < len(score_ir.harmonic_grid) else {}
            chord_pcs = set(seg.get("chord_pcs", []))
            pc = int(note.pitch) % 12
            root = int(seg.get("root_pc", 0)) % 12
            if pc in chord_pcs:
                continue
            if (pc - root) % 12 not in (5, 2, 8):
                continue
            from score_ir import _suspension_resolves
            if _suspension_resolves(tir, note, seg):
                note.note_role = "suspension"
                n += 1
    return n
