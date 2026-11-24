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
    "6.groove_lib(swing->strum->AMG->haas->inertia)",
    "7.sanitize_note_overlaps(guard 10ms)",
]


def compile_song(project_dir: str, core_only: bool = False, out_gridline: str = "") -> dict:
    """Compile `<project_dir>` déclarativement. Retourne un rapport JSON compact."""
    proj = os.path.abspath(os.path.expanduser(project_dir))
    bp_path = os.path.join(proj, "song_blueprint.json")
    report: dict = {"project_dir": proj, "pipeline_order": CANONICAL_MIDI_PIPELINE}
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
        from song_builder import compile_arrangement_plan
        out = out_gridline or os.path.join(proj, "core_loop.gridline" if core_only else "projet.gridline")
        info = compile_arrangement_plan(plan_path, proj, out, core_only=core_only, verify_gate=True, auto_fix_midi=True)
        report.update({"ok": True, "stage": "compile_arrangement_plan", "out": out,
                       "core_only": core_only, "info_keys": sorted(info.keys()) if isinstance(info, dict) else str(info)})
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
    ap.add_argument("--out", default="", help="Chemin .gridline de sortie (défaut: <proj>/projet.gridline)")
    args = ap.parse_args()
    report = compile_song(args.project_dir, core_only=args.core_only, out_gridline=args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    sys.exit(0 if report.get("ok") else 2)


if __name__ == "__main__":
    main()
