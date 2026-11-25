#!/usr/bin/env python3
"""
v3.0 `compile_song.py` — compilateur universel 100 % déclaratif.

Usage: `python3 scripts/compile_song.py <project_dir> [--core-only] [--out projet.gridline]`

Lit uniquement `<project_dir>/song_blueprint.json` (+ `arrangement_plan.json` si présent)
et produit `projet.gridline` via `song_builder` — aucun `build_song.py` à écrire à la main.

Ordre canonique immuable du pipeline MIDI (verrouillé ici, §2.4 du plan v3.0) :
  1. génération symbolique (degrés -> MIDI via SCALES)
  2. optimize_voice_leading() avec Low Interval Limit
  3. phrasing_engine (arcs de vélocité, notes de passage, appoggiatures)
  4. transitions_lib (anticipations couplées Accords+Basse+coupe Lead, fills, dropouts)
  5. verify_midi_gate(auto_fix=True) contour-preserving + règles modales
  6. groove_lib (apply_swing -> micro_strum -> anti_machine_gun -> haas_pocket -> enforce_frequency_inertia)
  7. sanitize_note_overlaps() (garde anti note-stealing sur pistes mono, anti chevauchement same-pitch)

Avant toute compilation, le gate `blueprint_safety` bloque le build (exit 2) si une piste
sample a un `root_note` manquant/faux, un sample non jouable en rôle mélodique, une boucle
mélodique hors gamme/tempo ou un S_max dépassé.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
ANALYZER_DIR = os.path.join(ROOT, "mcp-servers", "audio-analyzer")
for _d in (SCRIPTS_DIR, ANALYZER_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from song_builder import load_blueprint  # noqa: E402

CANONICAL_MIDI_PIPELINE = [
    "1.generate_symbolic(SCALES)",
    "2.optimize_voice_leading(LowIntervalLimit)",
    "3.phrasing_engine",
    "4.transitions_lib(anticipation_couplée)",
    "5.verify_midi_gate(contour_preserving+modal)",
    "6.groove_lib->ScoreIR(expression-only)+clamp_frontiere_harmonique(guard 20ms)",
    "7.sanitize_note_overlaps(guard 10ms)",
]


def compile_song(project_dir: str, core_only: bool = False, out_gridline: str = "",
                 probe: bool = False, validate_full: bool = True) -> dict:
    """Compile `<project_dir>` déclarativement. Retourne un rapport JSON compact."""
    proj = os.path.abspath(os.path.expanduser(project_dir))
    report: dict = {"project_dir": proj, "pipeline_order": CANONICAL_MIDI_PIPELINE}
    bp_path = os.path.join(proj, "song_blueprint.json")
    # v3.3 #1 — enricher d'intention créative (Tour 2) : complète UNIQUEMENT les champs omis
    # (jamais d'écrasement) + journal renders/diagnostics/enrichment_report.json
    try:
        import json as _json
        from blueprint_enricher import enrich_blueprint, write_enrichment_report
        from determinism import resolve_seed
        with open(bp_path, encoding="utf-8") as _fh:
            _bp_raw = _json.load(_fh)
        _bp_enriched, _enr_report = enrich_blueprint(_bp_raw, project_dir=proj)
        report["seed"] = resolve_seed(_bp_enriched)
        if _enr_report["n_decisions"]:
            with open(bp_path, "w", encoding="utf-8") as _fh:
                _json.dump(_bp_enriched, _fh, ensure_ascii=False, indent=2)
                _fh.write("\n")
            write_enrichment_report(proj, _enr_report)
            report["enrichment"] = {"n_decisions": _enr_report["n_decisions"], "summary": _enr_report["summary"][:300]}
    except Exception as _exc:
        report["enrichment"] = {"error": str(_exc)[:120]}
    if not os.path.isfile(bp_path):
        report.update({"ok": False, "stage": "blueprint_missing", "error": bp_path})
        return report
    _bp, meta = load_blueprint(bp_path, validate=True)
    safety = meta.get("musical_safety") or {}
    report["musical_safety"] = {k: safety.get(k) for k in ("checked", "key", "errors", "warnings", "ok", "issues")}
    if safety.get("errors"):
        report.update({"ok": False, "stage": "blueprint_safety",
                       "error": f"{safety['errors']} erreur(s) bloquante(s) — voir musical_safety.issues"})
        return report

    plan_path = os.path.join(proj, "arrangement_plan.json")
    if os.path.isfile(plan_path):
        from song_builder import compile_arrangement_plan, load_arrangement_plan
        if probe:
            # v3.2 #3 — sonde composite 12 mesures (8 noyau + 1 silence + 4 hors-noyau)
            # en UN seul .gridline, pour auditer 100 % des pistes en stems dès le Tour 2.
            from full_coverage_validator import build_probe_plan
            plan_obj, _meta = load_arrangement_plan(plan_path, validate=False)
            # v3.3 #3 : tampon Hall dynamique (>= 2 mes.) selon decay réel + BPM du morceau
            _bpm = float(plan_obj.get("bpm") or meta.get("bpm") or 120.0)
            _hall = float((plan_obj.get("fx") or {}).get("hall_decay_s") or 2.4)
            probe_plan = build_probe_plan(plan_obj, hall_decay_s=_hall, bpm=_bpm)
            out = out_gridline or os.path.join(proj, "core_probe.gridline")
            info = compile_arrangement_plan(probe_plan, proj, out, core_only=False, verify_gate=False, auto_fix_midi=True)
            report.update({"ok": True, "stage": "compile_probe", "out": out,
                           "probe": probe_plan.get("probe"),
                           "info_keys": sorted(info.keys()) if isinstance(info, dict) else str(info)})
            return report
        out = out_gridline or os.path.join(proj, "core_loop.gridline" if core_only else "projet.gridline")
        info = compile_arrangement_plan(plan_path, proj, out, core_only=core_only, verify_gate=True, auto_fix_midi=True)
        report.update({"ok": True, "stage": "compile_arrangement_plan", "out": out,
                       "core_only": core_only, "info_keys": sorted(info.keys()) if isinstance(info, dict) else str(info)})
        # v3.2 #3 — pré-vol symbolique 100 % du morceau sur le ScoreIR compilé (0 rendu audio)
        if validate_full:
            try:
                from full_coverage_validator import run_full_coverage_check
                fc = run_full_coverage_check(proj)
                report["full_coverage"] = {k: fc.get(k) for k in ("passed", "n_issues", "issues", "metrics", "probe_plan")
                                           if k in fc} if fc.get("ok", True) else fc
            except Exception as exc:
                report["full_coverage"] = {"error": str(exc)[:160]}
        return report

    build_song = os.path.join(proj, "build_song.py")
    if os.path.isfile(build_song):
        args = [sys.executable, build_song] + (["--core-only"] if core_only else [])
        proc = subprocess.run(args, cwd=proj, capture_output=True, text=True, timeout=600)
        report.update({"ok": proc.returncode == 0, "stage": "legacy_build_song",
                       "returncode": proc.returncode,
                       "stderr_tail": (proc.stderr or "")[-400:]})
        return report

    report.update({"ok": False, "stage": "no_input",
                   "error": "ni arrangement_plan.json ni build_song.py dans le projet"})
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="v3.0 compilateur déclaratif (blueprint + arrangement_plan -> .gridline)")
    ap.add_argument("project_dir")
    ap.add_argument("--core-only", action="store_true", help="Compile seulement la boucle noyau 8 mesures (core_loop.gridline)")
    ap.add_argument("--probe", action="store_true", help="v3.2 : compile la sonde composite 12 mesures (core_probe.gridline : 8 noyau + 1 silence + 4 hors-noyau)")
    ap.add_argument("--no-validate-full", action="store_true", help="Désactive le pré-vol symbolique 100 % (full_coverage) après compilation")
    ap.add_argument("--out", default="", help="Chemin .gridline de sortie (défaut: <proj>/projet.gridline)")
    args = ap.parse_args()
    report = compile_song(args.project_dir, core_only=args.core_only, out_gridline=args.out,
                          probe=args.probe, validate_full=not args.no_validate_full)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    sys.exit(0 if report.get("ok") else 2)


if __name__ == "__main__":
    main()
