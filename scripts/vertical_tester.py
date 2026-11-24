#!/usr/bin/env python3
"""
Incremental Vertical Core-Loop Tester (`vertical_tester.py`, RFC §4).

Validates the densest 4- or 8-bar Refrain/Drop (`core_loop.gridline`) in
cumulative fault-isolated stages:
  - `V1_LOW_END`: Kick + 808/Bass (or Bass alone if `has_drums=false`)
  - `V2_GROOVE`:  V1 + Snare/Clap + HiHats/Percs (auto-skipped if `has_drums=false`)
  - `V3_HARMONY`: V2 + Chords/Pad (adds audio bass-vs-harmony chroma conflict check)
  - `V4_LEAD`:    V3 + Lead / Counter / Vocals -> locks `core_loop_locked = True` on PASS

Fault isolation rule (RFC §4.1):
  If stage V_{k-1} passed and stage V_k fails, ONLY `target_layer_tracks` (layer k)
  may be edited; `forbidden_tracks` (layers 1..k-1) are strictly locked.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
ANALYZER_DIR = os.path.join(ROOT_DIR, "mcp-servers", "audio-analyzer")
for _d in (SCRIPTS_DIR, ANALYZER_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from artifact_cache import compute_e2_vertical_hash, get_cached_verdict, set_cached_verdict
from deep_diagnostics import VERTICAL_STAGES, analyze_vertical_stage
from studio_lib import GridlineMCP

_STAGE_ORDER = {s: idx for idx, s in enumerate(VERTICAL_STAGES)}


def _extract_expected_onsets_from_gridline(
    project_path: str,
    range_bars: str = "",
) -> tuple[float, dict[str, list[dict[str, Any]]]]:
    """Extract per-channel expected onset beats directly from `.gridline` JSON (< 1 ms)."""
    try:
        with open(project_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return 132.0, {}
    bpm = float(data.get("tempo") or 132.0)
    ppq = float(data.get("ppq") or 96.0)
    bpb = float(data.get("timeSigNum") or 4)
    offset_beats = 0.0
    max_beats = 1e9
    if range_bars.strip() and "-" in range_bars:
        try:
            b0, b1 = [int(x) for x in range_bars.strip().replace("range:", "").split("-", 1)]
            offset_beats = float(max(0, b0 - 1)) * bpb
            max_beats = float(b1 - b0 + 1) * bpb
        except ValueError:
            pass

    chans = {str(ch.get("id")): str(ch.get("name") or f"Channel_{i}") for i, ch in enumerate(data.get("channels") or [], start=1) if "id" in ch}
    pats = {p.get("id"): p for p in (data.get("patterns") or []) if "id" in p}
    onsets_by_ch: dict[str, list[dict[str, Any]]] = {}
    for clip in data.get("playlist") or []:
        pat = pats.get(clip.get("pattern"))
        if not pat:
            continue
        clip_start_q = float(clip.get("start", 0)) / ppq
        notes_map = pat.get("notes") or {}
        if not isinstance(notes_map, dict):
            continue
        for cid, evs in notes_map.items():
            ch_name = chans.get(str(cid))
            if not ch_name:
                continue
            for ev in evs:
                st_q = clip_start_q + float(ev.get("start", 0)) / ppq - offset_beats
                if 0.0 - 1e-4 <= st_q < max_beats - 1e-4:
                    onsets_by_ch.setdefault(ch_name, []).append({"start": round(max(0.0, st_q), 5)})
    return bpm, onsets_by_ch


def classify_channel_stage(channel_name: str, role_hint: str = "") -> str:
    """Classify a Gridline channel into one of `V1_LOW_END`, `V2_GROOVE`, `V3_HARMONY`, `V4_LEAD`."""
    hay = f"{channel_name} {role_hint}".lower()
    if any(k in hay for k in ("kick", "bd", "808", "sub", "bass")):
        return "V1_LOW_END"
    if any(k in hay for k in ("snare", "clap", "hat", "hh", "perc", "rim", "shaker", "crash", "cymbal", "tom", "drum")):
        return "V2_GROOVE"
    if any(k in hay for k in ("lead", "melody", "topline", "vocal", "vox", "chant", "counter", "solo", "bell", "pluck")):
        # Note: if explicitly named 'chord' or 'pad', keep in V3_HARMONY
        if not any(k in hay for k in ("chord", "pad", "harmony", "bed")):
            return "V4_LEAD"
    return "V3_HARMONY"


def _load_blueprint_for_project(project_path: str, blueprint: dict[str, Any] | str | None) -> dict[str, Any]:
    if isinstance(blueprint, dict):
        return blueprint
    if isinstance(blueprint, str) and blueprint.strip() and os.path.isfile(blueprint):
        with open(blueprint, encoding="utf-8") as fh:
            return json.load(fh)
    cand = os.path.join(os.path.dirname(os.path.abspath(project_path)), "song_blueprint.json")
    if os.path.isfile(cand):
        with open(cand, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _resolve_role_hints_from_blueprint(bp: dict[str, Any]) -> dict[str, str]:
    hints: dict[str, str] = {}
    for tr in bp.get("tracks") or []:
        nm = str(tr.get("name") or "").strip().lower()
        rl = str(tr.get("role") or "").strip().lower()
        if nm and rl:
            hints[nm] = rl
    return hints


def analyze_vertical_stage_session(
    mcp: Any,
    stage: str,
    *,
    active_channels: list[int],
    profile: str = "trap_rap",
    bpm: float = 132.0,
    loudness_fn: Any = None,
) -> dict[str, Any]:
    """In-session vertical check WITHOUT rendering any .wav: probe each active channel's
    pattern loudness live (`analyze_loudness` scope=pattern while soloing), and verify the
    layer hierarchy (low end present, lead above harmony, groove present) from levels.

    `loudness_fn(channel, pattern) -> {"lufs": ..., "silent": ...}` is injectable for tests;
    defaults to solo + `mcp.analyze_loudness`.
    """
    st_norm = str(stage).strip().upper()
    if loudness_fn is None:
        all_ch = [int(c.get("channel", i)) for i, c in enumerate(mcp.call("list_channels", {}).get("channels", []), start=1)]

        def loudness_fn(ch: int, pattern: int = 1) -> dict[str, Any]:  # type: ignore[no-redef]
            muted = mcp.solo_channel(ch, all_ch)
            try:
                return mcp.analyze_loudness(scope="pattern", pattern=int(pattern))
            finally:
                mcp.unsolo(muted, keep_solo=ch)

    per_track: dict[str, dict[str, Any]] = {}
    for ch in active_channels:
        info = mcp.call("get_channel", {"channel": ch}) if hasattr(mcp, "call") else {}
        name = str(info.get("name") or f"ch{ch}")
        ld = loudness_fn(ch, 1)
        per_track[name] = {"channel": ch, "lufs": ld.get("lufs"), "silent": bool(ld.get("silent"))}

    def lvl(names: list[str]) -> float | None:
        vals = [float(v["lufs"]) for k, v in per_track.items() if k in names and v["lufs"] is not None and not v["silent"]]
        return max(vals) if vals else None

    low_names = [k for k in per_track if any(w in k.lower() for w in ("kick", "808", "bass", "sub"))]
    lead_names = [k for k in per_track if any(w in k.lower() for w in ("lead", "melody", "bell", "pluck", "vocal"))]
    harm_names = [k for k in per_track if any(w in k.lower() for w in ("chord", "pad", "keys", "string"))]
    groove_names = [k for k in per_track if any(w in k.lower() for w in ("snare", "clap", "hat", "perc"))]

    checks: dict[str, bool] = {}
    low = lvl(low_names)
    lead = lvl(lead_names)
    harm = lvl(harm_names)
    groove = lvl(groove_names)
    if st_norm in ("V1_LOW_END", "V2_GROOVE", "V3_HARMONY", "V4_LEAD"):
        checks["low_end_present"] = low is not None and low > -30.0
    if st_norm in ("V2_GROOVE", "V3_HARMONY", "V4_LEAD"):
        checks["groove_present"] = groove is None or groove > -30.0
    if st_norm in ("V3_HARMONY", "V4_LEAD"):
        checks["harmony_present"] = harm is None or harm > -32.0
    if st_norm == "V4_LEAD":
        checks["lead_present"] = lead is not None and lead > -32.0
        if lead is not None and harm is not None:
            try:
                from mix_config import stem_hierarchy_bounds
                _lvh_min = stem_hierarchy_bounds()["lead_vs_harmony"][0]
            except Exception:
                _lvh_min = 3.0
            checks["lead_hierarchy_ok"] = (lead - harm) >= _lvh_min
    failed = [k for k, v in checks.items() if not v]
    return {
        "verdict": "PASS" if not failed else "FIX",
        "stage": st_norm,
        "audition": "session",
        "checks": checks,
        "per_track_lufs": per_track,
        "failed_checks": failed,
        "fast": True,
    }


def run_vertical_stage(
    project_path: str,
    stage: str,
    *,
    profile: str = "",
    blueprint: dict[str, Any] | str | None = None,
    contract: dict[str, Any] | None = None,
    range_bars: str = "",
    stage_map: dict[str, str] | None = None,
    render_stems_for_stage: bool = True,
    use_cache: bool = True,
    diagnostics_dir: str = "",
    keep_tmp: bool = False,
    audition: str = "render",
) -> dict[str, Any]:
    """
    Create a temporary staged copy of `project_path` with only layers `<= stage` unmuted,
    render its audio (and stems when helpful for V1..V4), run `analyze_vertical_stage`,
    and return the concise RFC §6.2 verdict + isolation guardrails.
    """
    st_norm = str(stage).strip().upper()
    if st_norm not in _STAGE_ORDER:
        raise ValueError(f"Invalid vertical stage {stage!r}; expected one of {VERTICAL_STAGES}")

    abs_proj = os.path.abspath(os.path.expanduser(project_path))
    if not os.path.isfile(abs_proj):
        raise FileNotFoundError(f"Project file not found: {abs_proj}")

    bp = _load_blueprint_for_project(abs_proj, blueprint)
    eff_profile = profile.strip() or str(bp.get("mix_profile") or "trap_rap")
    has_drums = bool(bp["has_drums"]) if "has_drums" in bp else (eff_profile != "classical_ambient")

    proj_dir = os.path.dirname(abs_proj)
    eff_contract = contract
    if eff_contract is None:
        cand_c = os.path.join(proj_dir, "multi_scale_contract.json")
        if os.path.isfile(cand_c):
            try:
                with open(cand_c, encoding="utf-8") as fh:
                    eff_contract = json.load(fh)
            except Exception:
                eff_contract = None

    out_render_dir = os.path.join(proj_dir, "renders", "vertical")
    eff_diag_dir = diagnostics_dir or os.path.join(proj_dir, "renders", "diagnostics")
    os.makedirs(out_render_dir, exist_ok=True)
    os.makedirs(eff_diag_dir, exist_ok=True)

    # Fast skip for V2_GROOVE when has_drums=False
    if st_norm == "V2_GROOVE" and not has_drums:
        res = analyze_vertical_stage(
            wav_path="",
            stage="V2_GROOVE",
            profile=eff_profile,
            blueprint=bp,
            contract=eff_contract,
        )
        res["target_layer_tracks"] = []
        res["forbidden_tracks"] = []
        res["active_tracks"] = []
        return res

    e2_key = f"e2:{st_norm}"
    e2_hash = ""
    if use_cache:
        e2_hash = compute_e2_vertical_hash(
            st_norm,
            project_path=abs_proj,
            profile=eff_profile,
            range_bars=range_bars,
            stage_map=stage_map,
            blueprint=bp,
            contract=eff_contract,
            render_stems=render_stems_for_stage,
        )
        cached = get_cached_verdict(proj_dir, e2_key, e2_hash)
        if cached is not None:
            return cached

    tmp_dir = tempfile.mkdtemp(prefix=f"gl_vert_{st_norm}_", dir=proj_dir)
    staged_proj = os.path.join(tmp_dir, f"{st_norm}.gridline")
    shutil.copyfile(abs_proj, staged_proj)

    stage_rank = _STAGE_ORDER[st_norm]
    role_hints = _resolve_role_hints_from_blueprint(bp)
    custom_map = {k.lower(): v.upper() for k, v in (stage_map or {}).items()}

    active_tracks: list[str] = []
    target_layer_tracks: list[str] = []
    forbidden_tracks: list[str] = []
    muted_tracks: list[str] = []

    try:
        mcp = GridlineMCP(staged_proj, tmp_dir, client_name=f"vertical_{st_norm}")
        try:
            ch_resp = mcp.call("list_channels", {})
            channels = ch_resp.get("channels") or []
            edits: list[dict[str, Any]] = []
            for idx, ch in enumerate(channels, start=1):
                ch_idx = int(ch.get("channel", idx))
                ch_name = str(ch.get("name") or f"Channel_{ch_idx}")
                ch_stage = custom_map.get(ch_name.lower()) or classify_channel_stage(
                    ch_name, role_hint=role_hints.get(ch_name.lower(), "")
                )
                ch_rank = _STAGE_ORDER.get(ch_stage, 2)
                is_active = ch_rank <= stage_rank
                edits.append({
                    "tool": "edit_channel",
                    "arguments": {"channel": ch_idx, "muted": not is_active},
                })
                if is_active:
                    active_tracks.append(ch_name)
                    if ch_rank == stage_rank:
                        target_layer_tracks.append(ch_name)
                    else:
                        forbidden_tracks.append(ch_name)
                else:
                    muted_tracks.append(ch_name)
            if edits:
                mcp.batch(edits)
            mcp.call("save_project", {})
        finally:
            mcp.close()

        if audition == "session":
            # In-session audition: no `gridline render`, no .wav -- probe pattern loudness live.
            mcp2 = GridlineMCP(staged_proj, tmp_dir, client_name=f"vertical_session_{st_norm}")
            try:
                res = analyze_vertical_stage_session(
                    mcp2, st_norm,
                    active_channels=[int(c.get("channel", i)) for i, c in enumerate(mcp2.call("list_channels", {}).get("channels", []), start=1) if not c.get("muted", False)],
                    profile=eff_profile, bpm=140.0,
                )
            finally:
                mcp2.close()
            res["active_tracks"] = active_tracks
            res["target_layer_tracks"] = target_layer_tracks
            res["forbidden_tracks"] = forbidden_tracks
            res["muted_tracks"] = muted_tracks
            diag_path = os.path.join(eff_diag_dir, f"vertical_{st_norm}.json")
            with open(diag_path, "w", encoding="utf-8") as fh:
                json.dump(res, fh, indent=2, ensure_ascii=False)
            res["diagnostics_path"] = diag_path
            if use_cache and e2_hash:
                set_cached_verdict(proj_dir, e2_key, e2_hash, res)
            return res

        stage_wav = os.path.join(out_render_dir, f"{st_norm}.wav")
        cmd = ["gridline", "render", staged_proj, "-o", stage_wav, "--bits", "24"]
        if range_bars.strip():
            rng = range_bars.strip()
            if not rng.startswith("range:"):
                rng = f"range:{rng}"
            cmd.extend(["--scope", rng])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"Vertical render failed for {st_norm} ({proc.returncode}): {proc.stderr or proc.stdout}")

        stems_dir_used: str | None = None
        if render_stems_for_stage and st_norm in ("V1_LOW_END", "V2_GROOVE", "V3_HARMONY", "V4_LEAD"):
            stems_out = os.path.join(tmp_dir, f"stems_{st_norm}")
            os.makedirs(stems_out, exist_ok=True)
            cmd_st = ["gridline", "render", staged_proj, "-o", stems_out.rstrip("/") + "/", "--stems"]
            if range_bars.strip():
                rng = range_bars.strip()
                if not rng.startswith("range:"):
                    rng = f"range:{rng}"
                cmd_st.extend(["--scope", rng])
            p_st = subprocess.run(cmd_st, capture_output=True, text=True, timeout=120)
            if p_st.returncode == 0 and any(f.endswith(".wav") for f in os.listdir(stems_out)):
                stems_dir_used = stems_out

        gl_bpm, expected_onsets = _extract_expected_onsets_from_gridline(staged_proj, range_bars=range_bars)
        res = analyze_vertical_stage(
            wav_path=stage_wav,
            stage=st_norm,
            profile=eff_profile,
            blueprint=bp,
            stems=stems_dir_used,
            expected_onsets_by_role=expected_onsets or None,
            bpm=gl_bpm,
            contract=eff_contract,
        )
        res["wav_path"] = stage_wav
        res["active_tracks"] = active_tracks
        res["target_layer_tracks"] = target_layer_tracks
        res["forbidden_tracks"] = forbidden_tracks
        res["muted_tracks"] = muted_tracks

        diag_path = os.path.join(eff_diag_dir, f"vertical_{st_norm}.json")
        with open(diag_path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
        res["diagnostics_path"] = diag_path

        if use_cache and e2_hash:
            set_cached_verdict(proj_dir, e2_key, e2_hash, res)
        return res
    finally:
        if not keep_tmp and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def run_all_vertical_stages(
    project_path: str,
    *,
    profile: str = "",
    blueprint: dict[str, Any] | str | None = None,
    contract: dict[str, Any] | None = None,
    range_bars: str = "",
    stage_map: dict[str, str] | None = None,
    stop_on_fail: bool = True,
    render_stems_for_stage: bool = True,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Run `V1_LOW_END -> V2_GROOVE -> V3_HARMONY -> V4_LEAD` sequentially with fault isolation."""
    results: dict[str, Any] = {}
    failed_stage: str | None = None
    for st in VERTICAL_STAGES:
        r = run_vertical_stage(
            project_path,
            st,
            profile=profile,
            blueprint=blueprint,
            contract=contract,
            range_bars=range_bars,
            stage_map=stage_map,
            render_stems_for_stage=render_stems_for_stage,
            use_cache=use_cache,
        )
        results[st] = r
        if r.get("verdict") != "PASS":
            failed_stage = st
            if stop_on_fail:
                break

    core_locked = failed_stage is None and results.get("V4_LEAD", {}).get("verdict") == "PASS"
    return {
        "project_path": os.path.abspath(project_path),
        "verdict": "PASS" if core_locked else results[failed_stage]["verdict"],
        "core_loop_locked": core_locked,
        "failed_stage": failed_stage,
        "stages": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Incremental vertical tester (V1..V4) for core_loop.gridline.")
    ap.add_argument("project", help="Path to core_loop.gridline (or full .gridline)")
    ap.add_argument("--stage", default="ALL", choices=["ALL", *VERTICAL_STAGES])
    ap.add_argument("--profile", default="")
    ap.add_argument("--blueprint", default="")
    ap.add_argument("--range", dest="range_bars", default="")
    ap.add_argument("--no-stems", action="store_true")
    ap.add_argument("--continue-on-fail", action="store_true")
    args = ap.parse_args()

    if args.stage == "ALL":
        res = run_all_vertical_stages(
            args.project,
            profile=args.profile,
            blueprint=args.blueprint or None,
            range_bars=args.range_bars,
            stop_on_fail=not args.continue_on_fail,
            render_stems_for_stage=not args.no_stems,
        )
    else:
        res = run_vertical_stage(
            args.project,
            args.stage,
            profile=args.profile,
            blueprint=args.blueprint or None,
            range_bars=args.range_bars,
            render_stems_for_stage=not args.no_stems,
        )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
