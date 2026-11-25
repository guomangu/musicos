#!/usr/bin/env python3
"""
v3.2 #3 `full_coverage_validator.py` — FullCoverageCoreValidator (pré-vol symbolique global).

Dès le Tour 2, AVANT tout rendu Gridline, on évalue la partition complète compilée (ScoreIR)
en < 100 ms, 0 rendu audio :
  - **Clones 8 mesures** : similarité symbolique (start/pitch/length quantisés au 1/16)
    entre blocs de 8 mesures consécutifs — flag si > 0.94 (S_bar8).
  - **Charge cognitive par mesure** : 1..8 pistes actives simultanément (toutes sections).
  - **Arc d'énergie prédit** : densité d'onsets × vélocité moyenne × registre, par section
    (ou bloc de 4 mesures) — contraste réel Intro/Couplet/Pont/Refrain/Outro requis.
  - **Intégrité des transitions & résolution de l'Outro** : coutures majeures avec événement
    déclaré, dernière mesure résolue sur la tonique (basse/accords).

`build_probe_plan` assemble un **plan de sondage stems composite de 12 mesures**
(8 mesures du noyau + 1 mesure de silence + 4 mesures d'une tranche hors-noyau contenant
les pistes absentes du noyau) rendu en UN seul projet Gridline, audité ensuite par
`stem_matrix_evaluator` pour couvrir 100 % des pistes instrumentales.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from typing import Any

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from score_ir import ScoreIR, segment_index_for_beat  # noqa: E402

CLONE_SIMILARITY_MAX = 0.94
MAX_ACTIVE_TRACKS_PER_BAR = 8
BLOCK_BARS = 8


def _track_signature(notes: list[Any], block_start: float, beats_per_block: float) -> set[tuple[int, int, int]]:
    sig: set[tuple[int, int, int]] = set()
    for n in notes:
        rel = float(n.grid_beat) - block_start
        if -0.01 <= rel < beats_per_block:
            sig.add((int(round(rel * 4)), int(n.pitch), int(round(float(n.duration_beats) * 4))))
    return sig


def _block_similarity(score_ir: ScoreIR, b0: float, b1: float, bpb: float) -> float:
    beats = BLOCK_BARS * bpb
    inter = 0
    union = 0
    for tir in score_ir.tracks.values():
        s0 = _track_signature(tir.notes, b0, beats)
        s1 = _track_signature(tir.notes, b1, beats)
        inter += len(s0 & s1)
        union += len(s0 | s1)
    return (inter / union) if union else 1.0


def validate_full_score_ir(
    score_ir: ScoreIR,
    sections: list[tuple[str, int, int, str]] | None = None,
    declared_transition_beats: list[float] | None = None,
) -> dict[str, Any]:
    """Pré-vol symbolique 100 % du morceau sur le ScoreIR compilé. Déterministe, < 100 ms."""
    issues: list[dict[str, Any]] = []
    bpb = float(score_ir.time_signature.get("numerator", 4))
    total_beats = 0.0
    for tir in score_ir.tracks.values():
        for n in tir.notes:
            total_beats = max(total_beats, float(n.grid_beat) + float(n.duration_beats))
    total_beats = max(total_beats, float(score_ir.harmonic_grid[-1]["end_beat"]) if score_ir.harmonic_grid else 8.0 * bpb)

    # 1) clones de blocs de 8 mesures consécutifs
    block_beats = BLOCK_BARS * bpb
    n_blocks = max(1, int(total_beats // block_beats))
    clone_blocks: list[int] = []
    for i in range(n_blocks - 1):
        b0, b1 = i * block_beats, (i + 1) * block_beats
        sim = _block_similarity(score_ir, b0, b1, bpb)
        if sim > CLONE_SIMILARITY_MAX:
            clone_blocks.append(i + 1)
            issues.append({"severity": "error", "code": "clone_8bar", "block": i + 1, "similarity": round(sim, 3)})

    # 2) charge cognitive par mesure (1..8 pistes actives)
    overload_bars: list[int] = []
    n_bars = max(1, int(total_beats // bpb))
    for bar in range(n_bars):
        bs, be = bar * bpb, (bar + 1) * bpb
        active = 0
        for tir in score_ir.tracks.values():
            if any(float(n.grid_beat) < be and float(n.grid_beat) + float(n.duration_beats) > bs for n in tir.notes):
                active += 1
        if active > MAX_ACTIVE_TRACKS_PER_BAR:
            overload_bars.append(bar + 1)
    if overload_bars:
        issues.append({"severity": "error", "code": "cognitive_load", "bars": overload_bars[:12],
                       "max_allowed": MAX_ACTIVE_TRACKS_PER_BAR})

    # 3) arc d'énergie prédit par section (ou bloc de 4 mesures)
    spans: list[tuple[str, float, float]] = []
    if sections:
        for name, sb, nb, _kind in sections:
            spans.append((str(name), float(sb) * bpb, float(sb + nb) * bpb))
    else:
        for i in range(max(1, int(total_beats // (4 * bpb)))):
            spans.append((f"block{i}", i * 4 * bpb, (i + 1) * 4 * bpb))
    arc: list[float] = []
    arc_labels: list[str] = []
    for label, s0, s1 in spans:
        onsets = 0
        vels: list[float] = []
        regs: list[float] = []
        for tir in score_ir.tracks.values():
            for n in tir.notes:
                if s0 - 1e-6 <= float(n.grid_beat) < s1:
                    onsets += 1
                    vels.append(float(n.velocity) / 127.0)
                    regs.append(float(n.pitch) / 127.0)
        if onsets:
            e = (onsets / max(1.0, (s1 - s0))) * (sum(vels) / len(vels)) * (0.5 + 0.5 * (sum(regs) / len(regs)))
        else:
            e = 0.0
        arc.append(e)
        arc_labels.append(label)
    flat = False
    if len(arc) >= 2 and max(arc) > 0:
        import statistics as _st
        if float(_st.pstdev(arc)) < 0.02 * max(arc):
            flat = True
    else:
        flat = len(arc) >= 2
    if flat:
        issues.append({"severity": "warning", "code": "flat_energy_arc", "arc": [round(a, 3) for a in arc], "sections": arc_labels})

    # 4) transitions déclarées / coutures nues + résolution de l'Outro
    if sections and len(sections) > 1:
        declared = {round(float(b), 3) for b in (declared_transition_beats or [])}
        for name, sb, _nb, _kind in sections[1:]:
            boundary = float(sb) * bpb
            has_event = any(
                any(abs(float(n.grid_beat) - boundary) <= 1.0 for n in tir.notes)
                for tir in score_ir.tracks.values()
            )
            if not has_event and round(boundary, 3) not in declared:
                issues.append({"severity": "warning", "code": "bare_seam", "section": str(name), "beat": round(boundary, 3)})
    outro_ok = True
    if score_ir.harmonic_grid:
        tonic_pc = int(score_ir.harmonic_grid[-1].get("root_pc", 0)) % 12
        end_beat = float(score_ir.harmonic_grid[-1]["end_beat"])
        resolved = False
        for tkey, tir in score_ir.tracks.items():
            rl = tir.role.lower()
            if rl not in ("bass", "chords"):
                continue
            for n in tir.notes:
                if float(n.grid_beat) >= end_beat - 2.0 * bpb and int(n.pitch) % 12 == tonic_pc:
                    resolved = True
        if not resolved and any(score_ir.tracks.values()):
            outro_ok = False
            issues.append({"severity": "warning", "code": "outro_resolution", "expected_pc": tonic_pc})

    errors = [i for i in issues if i["severity"] == "error"]
    return {
        "passed": not errors,
        "n_issues": len(issues),
        "issues": issues,
        "metrics": {
            "clone_blocks": clone_blocks,
            "overload_bars": overload_bars[:12],
            "energy_arc": [round(a, 3) for a in arc],
            "energy_sections": arc_labels,
            "outro_resolved": outro_ok,
            "total_beats": round(total_beats, 2),
        },
        "auto_patch_hints": [
            {"code": "MUTATE_BLOCK_DENSITY_OR_LPF", "params": {"bars": [b * BLOCK_BARS + 1 for b in clone_blocks]}} if clone_blocks else
            {"code": "PRUNE_TRACK_IN_SECTION", "params": {"bars": overload_bars}} if overload_bars else
            {"code": "REINFORCE_SECTION_TRANSITION", "params": {}} if any(i["code"] == "bare_seam" for i in issues) else
            {"code": "APPLY_OUTRO_FADE", "params": {}} if not outro_ok else
            {"code": "KEEP", "params": {}}
        ],
    }


_CORE_KIND_HINTS = ("chorus", "refrain", "drop", "climax", "hook")


def _section_roles(sec: dict[str, Any]) -> list[str]:
    return [str(r) for r in (sec.get("active_roles") or [])]


def build_probe_plan(plan: dict[str, Any], core_bars: int = 8, probe_bars: int = 4,
                     hall_decay_s: float | None = None, bpm: float | None = None) -> dict[str, Any]:
    """Plan de sondage composite : 8 mesures du noyau + tampon de silence + 4 mesures
    d'une section hors-noyau (celle qui a le plus de pistes absentes du noyau).

    v3.3 #3 : le tampon vaut `compute_probe_gap_bars(hall_decay_s, bpm, beats_per_bar)`
    (>= 2 mesures) pour éteindre FX Hall avant le bloc hors-noyau, et les automations du
    plan sont remappées vers les offsets de beats de la sonde (`remap_automations_for_probe`)."""
    probe = copy.deepcopy(plan)
    sections = list(probe.get("sections") or [])
    if not sections:
        return probe
    core_idx = 0
    for i, sec in enumerate(sections):
        kind = str(sec.get("kind") or sec.get("name") or "").lower()
        if any(h in kind for h in _CORE_KIND_HINTS):
            core_idx = i
            break
    else:
        core_idx = max(range(len(sections)), key=lambda i: int(sections[i].get("bars", 0)))
    core_sec = dict(sections[core_idx])
    core_roles = set(_section_roles(core_sec))
    core_start = int(core_sec.get("start_bar", 0))

    best_idx, best_score = None, -1
    for i, sec in enumerate(sections):
        if i == core_idx:
            continue
        extra = len(set(_section_roles(sec)) - core_roles)
        if extra > best_score:
            best_idx, best_score = i, extra
    if best_idx is None:
        best_idx = (core_idx + 1) % len(sections)
    probe_sec = dict(sections[best_idx])

    new_core = dict(core_sec)
    new_core["start_bar"] = 0
    new_core["bars"] = min(int(core_sec.get("bars", core_bars)), core_bars)
    bpb = max(1, int(plan.get("beats_per_bar", 4) or 4))
    gap_bars = 1
    if hall_decay_s and bpm:
        from dsp_interaction_guards import compute_probe_gap_bars
        gap_bars = compute_probe_gap_bars(float(hall_decay_s), float(bpm), bpb)
    gap = {"name": "probe_gap", "start_bar": new_core["bars"], "bars": gap_bars, "kind": "break", "active_roles": []}
    new_probe = dict(probe_sec)
    new_probe["start_bar"] = new_core["bars"] + gap_bars
    new_probe["bars"] = min(int(probe_sec.get("bars", probe_bars)), probe_bars)
    probe["sections"] = [new_core, gap, new_probe]
    probe_meta: dict[str, Any] = {
        "core_section": str(core_sec.get("name") or core_sec.get("kind") or ""),
        "probe_section": str(probe_sec.get("name") or probe_sec.get("kind") or ""),
        "core_bars": new_core["bars"],
        "probe_bars": new_probe["bars"],
        "total_bars": new_core["bars"] + gap_bars + new_probe["bars"],
        "gap_bars": gap_bars,
    }
    autos = plan.get("automations") or []
    if autos:
        from dsp_interaction_guards import remap_automations_for_probe
        core_src0 = float(core_start) * bpb
        probe_src0 = float(sections[best_idx].get("start_bar", 0)) * bpb
        dst_probe = float(new_core["bars"] + gap_bars) * bpb
        slice_map = [
            (core_src0, core_src0 + float(new_core["bars"]) * bpb, 0.0),
            (probe_src0, probe_src0 + float(new_probe["bars"]) * bpb, dst_probe),
        ]
        remapped = remap_automations_for_probe(autos, slice_map)
        probe["automations"] = remapped
        probe_meta["automations_remapped"] = sum(1 for a in remapped if isinstance(a, dict) and a.get("_remapped"))
    probe["probe"] = probe_meta
    return probe


def run_full_coverage_check(project_dir: str, validate: bool = True) -> dict[str, Any]:
    """Pré-vol complet depuis `<projet>/score_ir.json` + `arrangement_plan.json`."""
    proj = os.path.abspath(os.path.expanduser(project_dir))
    sir_path = os.path.join(proj, "score_ir.json")
    plan_path = os.path.join(proj, "arrangement_plan.json")
    if not os.path.isfile(sir_path):
        return {"ok": False, "reason": "score_ir.json absent (compiler d'abord avec compile_song)"}
    score = ScoreIR.load(sir_path)
    sections = None
    transitions = []
    probe = None
    if os.path.isfile(plan_path):
        try:
            with open(plan_path, encoding="utf-8") as fh:
                plan = json.load(fh)
            sections = [(str(s.get("name", "")), int(s.get("start_bar", 0)), int(s.get("bars", 4)), str(s.get("kind", "")))
                        for s in plan.get("sections") or []]
            for tr in plan.get("transitions") or []:
                if isinstance(tr, dict) and tr.get("bar"):
                    transitions.append(float(tr["bar"] - 1) * 4.0)
            probe = build_probe_plan(plan)
        except Exception:
            pass
    rep = validate_full_score_ir(score, sections=sections, declared_transition_beats=transitions) if validate else validate_full_score_ir(score, sections=sections)
    rep["score_ir"] = sir_path
    if probe:
        rep["probe_plan"] = probe.get("probe")
    return rep
