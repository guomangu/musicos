#!/usr/bin/env python3
"""
v3.3 #1 — Enrichisseur d'Intention Créative (doc audit v3.3, Carence #1).

Exécuté automatiquement au début de `studio_build_and_validate_core` (Tour 2), avant
`blueprint_safety` et `compile_song` : 0 GPU, < 30 ms, 100 % déterministe, **non intrusif** —
ne JAMAIS écraser un champ fourni explicitement par le LLM ; on ne complète que les champs
optionnels omis, dérivés de `(mix_profile, emotion_vector.valence, harmonic_core, key, bpm)` :

  1. `seed` absent -> dérivé (SHA-256 title|key|bpm, `determinism.derive_seed`) ;
  2. `harmonic_core.section_progressions` absent -> variantes par section (pré-refrain à
     tension dominante, pont en emprunt modal compatible valence, outro = résolution tonique) ;
  3. `phrase_length_bars` absent -> 8 (phrasé Antécédent/Conséquent) si une section >= 8 mesures ;
  4. piste 808/bass sans articulation en trap_rap/edm_club -> `expression_defaults` avec
     1-2 `glide_to_next` subtils (60 ms) en mesures 4 et 8 du Climax ;
  5. piste hihat/snare + banque `midi_groove_bank` compatible (genre, BPM) -> `groove_preference` ;
  6. piste synthé `gl.*` sans archétype explicite -> meilleur archétype
     `config/timbre_archetypes.json` selon (mix_profile, role, valence).

Toutes les décisions par défaut sont journalisées dans
`<projet>/renders/diagnostics/enrichment_report.json` (transparence totale).
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any

from determinism import derive_seed

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)
if SCRIPTS_DIR not in __import__("sys").path:
    __import__("sys").path.insert(0, SCRIPTS_DIR)

DEFAULT_BANK_PATH = os.path.join(ROOT, "golden_references", "midi_groove_bank.json")
GLIDE_PROFILES = ("trap_rap", "edm_club")
BASS_ROLES = ("bass", "808", "sub", "bass_sub")


def _base_progression(hc: dict[str, Any]) -> list[str]:
    prog = hc.get("degree_progression") or []
    if not prog:
        prog = [c.get("symbol") for c in (hc.get("chord_progression") or []) if isinstance(c, dict) and c.get("symbol")]
    return [str(x) for x in prog] or ["i", "VI", "III", "VII"]


def _canonical_role(role: str) -> str:
    r = str(role or "").lower()
    if any(k in r for k in ("lead", "melody", "hook")):
        return "lead"
    if any(k in r for k in ("chord", "pad", "keys", "harmony")):
        return "chords"
    if any(k in r for k in ("bass", "808", "sub")):
        return "bass"
    return r


def _derive_section_progressions(hc: dict[str, Any], valence: float) -> dict[str, list[str]]:
    """Variantes déterministes : pré-refrain à tension dominante, pont emprunt modal
    (compatible valence), outro = cadence de résolution sur la tonique."""
    base = _base_progression(hc)
    pre = list(base)
    if pre:
        pre[-1] = "V"  # tension dominante avant le refrain
    if valence < -0.2:
        bridge = ["bVI", "bVII", "i", "V7"]     # emprunt modal sombre
    else:
        bridge = ["IV", "V", "iii", "vi"]        # pont clair diatonique
    outro = ["V7", "i"] if len(base) > 1 else ["i"]
    return {
        "verse": list(base),
        "pre_chorus": pre,
        "chorus": list(base),
        "bridge": bridge,
        "outro": outro,
    }


def enrich_blueprint(bp: dict[str, Any], project_dir: str | None = None,
                     bank_path: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retourne (blueprint enrichi, rapport). Ne modifie jamais un champ déjà présent."""
    bp = copy.deepcopy(bp)
    decisions: list[str] = []
    added: dict[str, Any] = {}

    mix_profile = str(bp.get("mix_profile") or "trap_rap")
    valence = float((bp.get("emotion_vector") or {}).get("valence", 0.0))
    key = str(bp.get("key") or "C")
    bpm = float(bp.get("bpm") or 120.0)
    title = str(bp.get("title") or "untitled")

    # 1) seed déterministe
    if bp.get("seed") is None:
        bp["seed"] = derive_seed(title, key, bpm)
        added["seed"] = bp["seed"]
        decisions.append(f"seed={bp['seed']} dérivé de SHA-256(title|key|bpm)")

    # 2) progressions par section
    hc = bp.get("harmonic_core")
    if isinstance(hc, dict) and not hc.get("section_progressions"):
        hc["section_progressions"] = _derive_section_progressions(hc, valence)
        bp["harmonic_core"] = hc
        added["harmonic_core.section_progressions"] = sorted(hc["section_progressions"].keys())
        decisions.append("section_progressions dérivées (pré-refrain V, pont emprunt modal, outro V7-i)")

    # 3) phrasé 8 mesures si une section >= 8 mesures
    if bp.get("phrase_length_bars") is None:
        structure = bp.get("structure") or []
        if any(int(s.get("bars", 0) or 0) >= 8 for s in structure if isinstance(s, dict)):
            bp["phrase_length_bars"] = 8
            added["phrase_length_bars"] = 8
            decisions.append("phrase_length_bars=8 (phrasé Antécédent/Conséquent sur sections >= 8 mes.)")

    # 4) glides 808 subtils au Climax (mesures 4 et 8)
    tracks = bp.get("tracks") or []
    if isinstance(tracks, list):
        has_bass_no_art = any(
            isinstance(t, dict) and str(t.get("role") or "").lower() in BASS_ROLES and "articulation" not in t
            for t in tracks
        )
        if has_bass_no_art and mix_profile in GLIDE_PROFILES and "expression_defaults" not in bp:
            bp["expression_defaults"] = {"bass_808": {"articulation": "glide_to_next", "glide_ms": 60, "climax_bars": [4, 8]}}
            added["expression_defaults"] = bp["expression_defaults"]
            decisions.append("expression_defaults: glide_to_next 60 ms en mesures 4/8 du Climax (808/bass)")

        # 5) préférence groove banque MIDI (hihat/snare) si template compatible
        if "groove_preference" not in bp:
            has_perc = any(isinstance(t, dict) and str(t.get("role") or "").lower() in ("hihat", "hat", "snare", "perc", "hihat_perc", "snare_clap") for t in tracks)
            if has_perc:
                bp_bank = bank_path or (os.path.join(project_dir, "midi_groove_bank.json") if project_dir else None) or DEFAULT_BANK_PATH
                tpl = None
                try:
                    from midi_groove_bank import query_groove_bank
                    tpl = query_groove_bank(bp_bank, role="hihat_perc", bpm=bpm, profile=mix_profile)
                except Exception:
                    tpl = None
                if tpl:
                    bp["groove_preference"] = {"role": "hihat_perc", "template_source": "midi_groove_bank", "template_bpm": tpl.get("bpm"), "source": tpl.get("source")}
                    added["groove_preference"] = bp["groove_preference"]
                    decisions.append(f"groove_preference: template midi_groove_bank bpm={tpl.get('bpm')} source={tpl.get('source')}")

        # 6) archétypes gl.* par (profil, rôle, valence)
        from gl_dsl import resolve_archetype
        for t in tracks:
            if not isinstance(t, dict):
                continue
            if str(t.get("plugin") or "").startswith("gl.") and not t.get("archetype"):
                role = _canonical_role(str(t.get("role") or "lead"))
                arch = resolve_archetype(mix_profile, role, valence)
                if arch:
                    t["archetype"] = f"{mix_profile}:{role}"
                    decisions.append(f"track {t.get('name') or role}: archétype {t['archetype']}")

    report = {
        "mix_profile": mix_profile,
        "valence": valence,
        "n_decisions": len(decisions),
        "added": added,
        "decisions": decisions,
        "summary": "; ".join(decisions) if decisions else "aucun champ omis — blueprint déjà complet",
    }
    return bp, report


def write_enrichment_report(project_dir: str, report: dict[str, Any]) -> str:
    """Journalise les décisions par défaut dans `<projet>/renders/diagnostics/enrichment_report.json`."""
    proj = os.path.abspath(os.path.expanduser(project_dir))
    out_dir = os.path.join(proj, "renders", "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "enrichment_report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return path


def inject_climax_glides(score_ir: Any, bp: dict[str, Any], section_spans: list[Any] | None = None) -> int:
    """Applique `expression_defaults.bass_808` sur le ScoreIR : 1 `glide_to_next` par mesure
    du Climax listée (défaut mes. 4 et 8), sur la 1re note basse de la mesure, registre/gamme
    respectés (glide borné 35-90 ms). Retourne le nombre de notes marquées."""
    defaults = (bp.get("expression_defaults") or {}).get("bass_808") or {}
    art = str(defaults.get("articulation") or "")
    glide_ms = float(defaults.get("glide_ms") or 60.0)
    climax_bars = [int(b) for b in (defaults.get("climax_bars") or [4, 8])]
    if not art or not climax_bars:
        return 0
    glide_ms = min(90.0, max(35.0, glide_ms))
    applied = 0
    bpb = 4.0
    # bars 1-indexed du Climax -> beats absolus si section_spans fourni, sinon mesures du morceau
    target_beats: set[float] = set()
    if section_spans:
        for sec in section_spans:
            name = str(getattr(sec, "name", "") or (sec.get("name") if isinstance(sec, dict) else "") or "").lower()
            if any(h in name for h in ("chorus", "refrain", "climax", "drop", "hook")):
                start = float(getattr(sec, "start_bar", 0) if not isinstance(sec, dict) else sec.get("start_bar", 0))
                for cb in climax_bars:
                    target_beats.add((start + cb - 1) * bpb)
    else:
        target_beats = {(cb - 1) * bpb for cb in climax_bars}
    for tir in getattr(score_ir, "tracks", {}).values():
        role = str(getattr(tir, "role", "") or "").lower()
        if role not in BASS_ROLES:
            continue
        notes = sorted(tir.notes, key=lambda n: (float(n.grid_beat), int(n.pitch)))
        seen_bars: set[float] = set()
        for n in notes:
            bar_beat = (float(n.grid_beat) // bpb) * bpb
            if bar_beat in target_beats and bar_beat not in seen_bars:
                expr = getattr(n, "expression", None)
                if expr is None and isinstance(n, dict):
                    expr = n.setdefault("expression", {})
                if isinstance(expr, dict):
                    expr["articulation"] = art
                    expr["glide_ms"] = glide_ms
                else:
                    try:
                        expr.articulation = art
                        expr.glide_ms = glide_ms
                    except Exception:
                        pass
                seen_bars.add(bar_beat)
                applied += 1
    return applied
