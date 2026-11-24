#!/usr/bin/env python3
"""
Closed Deterministic Multi-Scale Auto-Fix Dispatcher (`multiscale_autofix.py`, Phase 6 §4.3).

Executes mechanical S1..S4 repairs purely in Python (0 LLM tokens), respecting
`LoopFirstStateMachine` guardrails (`user_samples_locked`, `forbidden_tracks`).

Supported action codes:
  - `INVERT_808_POLARITY`        (S1/V)
  - `SHIFT_BASS_MICRO_MS`        (S1/V, clamped ±10 ms)
  - `TRIM_SAMPLE_START_MS`       (S2/H)
  - `STAGGER_ATTACK_OR_STRUM`    (S2/V)
  - `FIX_MOTIF_CLONE`            (S3/H)
  - `FIX_CHORDS_LEAD_OVERLAP`    (S3/V)
  - `MUTATE_BLOCK_DENSITY_OR_LPF`(S4/H)
  - `REINFORCE_SECTION_TRANSITION`(S4/H)
  - `APPLY_OUTRO_FADE`           (S4/H)
  - `EQ_RESONANCE_NOTCH` / `ADJUST_MASTER_GAIN` (S1/S5 -> `render_and_audit._apply_autofix_mutations`)
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
from typing import Any

import pretty_midi

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
ANALYZER_DIR = os.path.join(ROOT_DIR, "mcp-servers", "audio-analyzer")
for _d in (SCRIPTS_DIR, ANALYZER_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from artifact_cache import invalidate_cache
from gain_staging_lib import normalize_role_for_gain_staging
from gl_dsl import SYNTH_REGISTRY, clamp
from groove_lib import apply_micro_strum, ms_to_beats
from loop_first_runner import LoopFirstStateMachine, detect_user_samples_locked
from midi_autofix import autofix_midi
from render_and_audit import (
    _apply_autofix_mutations,
    adjust_track_fader,
    apply_dynamic_mirror_eq,
    boost_bass_harmonics,
    relax_bus_compression,
)
from sample_prep_lib import audit_samples_manifest, build_sample_from_spec
from song_builder import compile_arrangement_plan, load_arrangement_plan
from studio_lib import GridlineMCP
from transitions_lib import filter_sweep, outro_fade
from vocal_arrangement_sync import (
    build_formant_ducking_automation,
    extract_vocal_phrases,
    formant_ducking_eq_params,
    mute_counter_during_vocals,
)


def _parse_action_spec(action: dict[str, Any] | str) -> tuple[str, dict[str, Any]]:
    if isinstance(action, dict):
        if "action_payload" in action and isinstance(action["action_payload"], dict):
            ap = action["action_payload"]
            return str(ap.get("code") or "KEEP"), dict(ap.get("params") or {})
        if "code" in action:
            return str(action.get("code") or "KEEP"), dict(action.get("params") or {})
        raw_str = str(action.get("next_action") or "KEEP")
    else:
        raw_str = str(action or "KEEP")

    raw_str = raw_str.strip()
    m = re.match(r"^([A-Z0-9_]+)\((.*)\)$", raw_str)
    if not m:
        return raw_str, {}
    code = m.group(1)
    arg_s = m.group(2).strip()
    params: dict[str, Any] = {}
    if arg_s:
        if "=" in arg_s:
            for part in arg_s.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    try:
                        params[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        params[k] = v
        elif "," in arg_s:
            pos_args = [p.strip().strip("'\"") for p in arg_s.split(",")]
            if code == "PRUNE_TRACK_IN_SECTION":
                if len(pos_args) >= 1 and pos_args[0]:
                    params["section"] = pos_args[0]
                if len(pos_args) >= 2 and pos_args[1]:
                    params["role"] = pos_args[1]
            elif code == "ADJUST_TRACK_FADER":
                if len(pos_args) >= 1 and pos_args[0]:
                    params["track_name"] = pos_args[0]
                if len(pos_args) >= 2 and pos_args[1]:
                    try:
                        params["delta_db"] = float(pos_args[1])
                    except ValueError:
                        params["role"] = pos_args[1]
            else:
                for i, pv in enumerate(pos_args):
                    params[f"arg{i}"] = pv
                if pos_args:
                    params["track"] = pos_args[0]
        elif re.match(r"^['\"]?\d+-\d+['\"]?$", arg_s):
            params["bars"] = arg_s.strip("'\"")
        else:
            clean_v = arg_s.strip("'\"")
            try:
                fval = float(clean_v)
                params["value"] = fval
                if code == "APPLY_DYNAMIC_MIRROR_EQ":
                    params["freq_hz"] = fval
                elif code == "BOOST_BASS_HARMONICS":
                    params["f0_hz"] = fval
                elif code == "REINFORCE_SECTION_TRANSITION":
                    params["bar"] = int(round(fval))
            except ValueError:
                if code in ("REINFORCE_SECTION_TRANSITION", "PRUNE_TRACK_IN_SECTION"):
                    params["section"] = clean_v
                else:
                    params["track"] = clean_v
    return code, params


def _find_gridline_project(project_dir: str, explicit_path: str = "") -> str:
    if explicit_path and os.path.isfile(os.path.expanduser(explicit_path)):
        return os.path.abspath(os.path.expanduser(explicit_path))
    p_dir = os.path.abspath(os.path.expanduser(project_dir))
    for cand in ("core_loop.gridline", "projet.gridline", "project.gridline"):
        p = os.path.join(p_dir, cand)
        if os.path.isfile(p):
            return p
    if os.path.isdir(p_dir):
        for fn in sorted(os.listdir(p_dir)):
            if fn.endswith(".gridline"):
                return os.path.join(p_dir, fn)
    return ""


def _resolve_channel_matches(
    channels: list[dict[str, Any]],
    track_hint: str = "",
    keywords: tuple[str, ...] = (),
) -> list[tuple[int, dict[str, Any]]]:
    hint = (track_hint or "").strip().lower()
    exact = []
    kw_hit = []
    for idx1, ch in enumerate(channels, start=1):
        nm = str(ch.get("name") or "").lower()
        if hint and (nm == hint or hint in nm or nm in hint):
            exact.append((idx1, ch))
        elif keywords and any(k in nm for k in keywords):
            kw_hit.append((idx1, ch))
    return exact or kw_hit


def _update_manifest_sample(samples_dir: str, out_filename: str, patch_fields: dict[str, Any]) -> None:
    man_path = os.path.join(samples_dir, "manifest.json")
    if not os.path.isfile(man_path):
        return
    try:
        with open(man_path, encoding="utf-8") as fh:
            man = json.load(fh)
        entries = man if isinstance(man, list) else man.get("samples", [])
        changed = False
        for ent in entries:
            if isinstance(ent, dict) and os.path.basename(str(ent.get("out") or "")) == out_filename:
                ent.update(patch_fields)
                changed = True
        if changed:
            with open(man_path, "w", encoding="utf-8") as fh:
                json.dump(man, fh, indent=2, ensure_ascii=False)
            audit_samples_manifest(samples_dir, verbose=False)
    except Exception:
        pass


def _fix_invert_808_polarity(
    project_dir: str,
    params: dict[str, Any],
    *,
    gridline_path: str = "",
    user_locked: bool = False,
) -> dict[str, Any]:
    track_hint = str(params.get("track") or "808")
    p_dir = os.path.abspath(os.path.expanduser(project_dir))
    gl_path = _find_gridline_project(p_dir, gridline_path)

    # Try sample-level polarity flip when sample resides inside `<proj>/samples/` and not user-locked
    samples_dir = os.path.join(p_dir, "samples")
    sample_file = str(params.get("sample_path") or "")
    if not sample_file and gl_path and os.path.isfile(gl_path):
        with open(gl_path, encoding="utf-8") as fh:
            gdata = json.load(fh)
        matches = _resolve_channel_matches(
            gdata.get("channels") or [], track_hint, ("808", "sub", "bass")
        )
        if matches:
            ch = matches[0][1]
            plug = ch.get("plugin") or {}
            if plug.get("type") == "gl.sampler" and isinstance(plug.get("data"), str):
                cand = plug["data"]
                if os.path.isfile(cand) and os.path.commonpath([os.path.abspath(cand), samples_dir]) == os.path.abspath(samples_dir):
                    sample_file = os.path.abspath(cand)

    if sample_file and os.path.isfile(sample_file) and not user_locked:
        s_dir = os.path.dirname(sample_file)
        fn = os.path.basename(sample_file)
        build_sample_from_spec(s_dir, s_dir, {"src": fn, "out": fn, "invert_phase": True})
        _update_manifest_sample(s_dir, fn, {"invert_phase": True})
        return {"applied": True, "code": "INVERT_808_POLARITY", "method": "sample_prep_lib", "target": sample_file}

    # Mixer track polarity inversion in Gridline (works for both synths and locked user Samples/)
    if gl_path and os.path.isfile(gl_path):
        with open(gl_path, encoding="utf-8") as fh:
            gdata = json.load(fh)
        matches = _resolve_channel_matches(
            gdata.get("channels") or [], track_hint, ("808", "sub", "bass")
        )
        if not matches:
            return {"applied": False, "code": "INVERT_808_POLARITY", "reason": "No 808/Bass channel matched"}
        ch = matches[0][1]
        mixer_idx = int(ch.get("mixerTrack", 2))
        cur_inv = False
        for m in gdata.get("mixer") or []:
            if int(m.get("index", -1)) == mixer_idx:
                cur_inv = bool(m.get("invertPolarity", False))
                break
        mcp = GridlineMCP(gl_path, os.path.dirname(gl_path))
        try:
            mcp.call("patch", {"target": f"mixer/{mixer_idx}", "patch": {"invertPolarity": not cur_inv}})
            mcp.call("save_project", {})
        finally:
            mcp.close()
        return {
            "applied": True,
            "code": "INVERT_808_POLARITY",
            "method": "gridline_mixer_polarity",
            "mixer": mixer_idx,
            "invertPolarity": not cur_inv,
        }

    return {"applied": False, "code": "INVERT_808_POLARITY", "reason": "No target sample or .gridline found"}


def _fix_shift_bass_micro_ms(
    project_dir: str,
    params: dict[str, Any],
    *,
    gridline_path: str = "",
) -> dict[str, Any]:
    raw_delta = float(params.get("delta_ms", params.get("value", 4.0)))
    delta_ms = max(-10.0, min(10.0, raw_delta))
    if abs(delta_ms) < 0.25:
        return {"applied": False, "code": "SHIFT_BASS_MICRO_MS", "reason": "delta_ms negligible"}

    p_dir = os.path.abspath(os.path.expanduser(project_dir))
    gl_path = _find_gridline_project(p_dir, gridline_path)
    if not gl_path or not os.path.isfile(gl_path):
        return {"applied": False, "code": "SHIFT_BASS_MICRO_MS", "reason": ".gridline not found"}

    with open(gl_path, encoding="utf-8") as fh:
        gdata = json.load(fh)
    bpm = float(gdata.get("tempo") or 132.0)
    ppq = float(gdata.get("ppq") or 96.0)
    delta_beats = ms_to_beats(delta_ms, bpm)
    delta_ticks = delta_beats * ppq

    track_hint = str(params.get("track") or "bass")
    matches = _resolve_channel_matches(
        gdata.get("channels") or [], track_hint, ("808", "sub", "bass")
    )
    if not matches:
        return {"applied": False, "code": "SHIFT_BASS_MICRO_MS", "reason": "No 808/Bass channel matched"}

    # Preserve exact fractional sample shift using floating start or rounded sub-tick + groove_lib
    shifted_notes = 0
    target_cids = {str(ch.get("id")) for _i, ch in matches if "id" in ch}
    for pat in gdata.get("patterns") or []:
        nmap = pat.get("notes") or {}
        if not isinstance(nmap, dict):
            continue
        for cid, evs in nmap.items():
            if str(cid) not in target_cids:
                continue
            for ev in evs:
                new_st = float(ev.get("start", 0.0)) + delta_ticks
                if new_st < 0.0:
                    new_st = 0.0
                ev["start"] = round(new_st, 4)
                shifted_notes += 1

    # Also update master_sketch.mid if present via apply_micro_strum / direct note shift
    mid_path = os.path.join(p_dir, "master_sketch.mid")
    if os.path.isfile(mid_path):
        shift_s = delta_ms / 1000.0
        pm = pretty_midi.PrettyMIDI(mid_path)
        for inst in pm.instruments:
            nm = (inst.name or "").lower()
            if any(k in nm for k in ("808", "sub", "bass")):
                for n in inst.notes:
                    n.start = max(0.0, n.start + shift_s)
                    n.end = max(n.start + 0.03, n.end + shift_s)
        pm.write(mid_path)

    with open(gl_path, "w", encoding="utf-8") as fh:
        json.dump(gdata, fh, indent=2, ensure_ascii=False)

    return {
        "applied": shifted_notes > 0,
        "code": "SHIFT_BASS_MICRO_MS",
        "delta_ms": round(delta_ms, 2),
        "delta_ticks": round(delta_ticks, 4),
        "shifted_notes": shifted_notes,
    }


def _fix_trim_sample_start_ms(
    project_dir: str,
    params: dict[str, Any],
    *,
    gridline_path: str = "",
    user_locked: bool = False,
) -> dict[str, Any]:
    if user_locked:
        return {"applied": False, "code": "TRIM_SAMPLE_START_MS", "reason": "user_samples_locked=True"}
    trim_ms = max(0.0, float(params.get("ms", params.get("value", 0.0))))
    if trim_ms < 1.0:
        return {"applied": False, "code": "TRIM_SAMPLE_START_MS", "reason": "trim ms < 1.0"}

    track_hint = str(params.get("track") or "")
    p_dir = os.path.abspath(os.path.expanduser(project_dir))
    samples_dir = os.path.join(p_dir, "samples")
    sample_path = str(params.get("sample_path") or "")

    gl_path = _find_gridline_project(p_dir, gridline_path)
    if not sample_path and gl_path and os.path.isfile(gl_path):
        with open(gl_path, encoding="utf-8") as fh:
            gdata = json.load(fh)
        matches = _resolve_channel_matches(gdata.get("channels") or [], track_hint)
        for _i, ch in matches:
            plug = ch.get("plugin") or {}
            if plug.get("type") == "gl.sampler" and isinstance(plug.get("data"), str):
                if os.path.isfile(plug["data"]):
                    sample_path = os.path.abspath(plug["data"])
                    break

    if not sample_path and os.path.isdir(samples_dir):
        hint_l = track_hint.lower()
        for fn in sorted(os.listdir(samples_dir)):
            if fn.lower().endswith((".wav", ".flac")) and (not hint_l or hint_l in fn.lower()):
                sample_path = os.path.join(samples_dir, fn)
                break

    if not sample_path or not os.path.isfile(sample_path):
        return {"applied": False, "code": "TRIM_SAMPLE_START_MS", "reason": f"Sample not found for track '{track_hint}'"}

    s_dir = os.path.dirname(sample_path)
    fn = os.path.basename(sample_path)
    build_sample_from_spec(s_dir, s_dir, {"src": fn, "out": fn, "trim_start_ms": trim_ms, "fade_in_ms": 0.5})
    _update_manifest_sample(s_dir, fn, {"trim_start_ms": round(trim_ms, 2)})
    return {
        "applied": True,
        "code": "TRIM_SAMPLE_START_MS",
        "track": track_hint or fn,
        "sample_path": sample_path,
        "trim_start_ms": round(trim_ms, 2),
    }


def _fix_stagger_attack_or_strum(
    project_dir: str,
    params: dict[str, Any],
    *,
    gridline_path: str = "",
) -> dict[str, Any]:
    p_dir = os.path.abspath(os.path.expanduser(project_dir))
    gl_path = _find_gridline_project(p_dir, gridline_path)
    track_hint = str(params.get("track") or "")
    changed_items: list[str] = []

    if gl_path and os.path.isfile(gl_path):
        with open(gl_path, encoding="utf-8") as fh:
            gdata = json.load(fh)
        bpm = float(gdata.get("tempo") or 132.0)
        ppq = float(gdata.get("ppq") or 96.0)
        channels = gdata.get("channels") or []
        matches = _resolve_channel_matches(channels, track_hint, ("pad", "chord", "arp", "pluck", "keys", "string"))

        # 1) Lengthen synth attack on pads/chords via gl_dsl SYNTH_REGISTRY via GridlineMCP
        mcp: GridlineMCP | None = None
        for idx1, ch in matches:
            plug = ch.get("plugin") or {}
            ptype = str(plug.get("type") or "")
            spec = SYNTH_REGISTRY.get(ptype)
            if spec and spec.adsr_params.get("attack"):
                atk_name = spec.adsr_params["attack"]
                lo, hi = spec.adsr_bounds.get("attack", (0.001, 2.0))
                new_atk = min(hi, max(lo, 0.035 if spec.adsr_unit == "sec" else 35.0))
                if mcp is None:
                    mcp = GridlineMCP(gl_path, os.path.dirname(gl_path))
                try:
                    mcp.set_params(channel=idx1, **{atk_name: new_atk})
                    changed_items.append(f"attack:{ch.get('name')}={new_atk}")
                except Exception:
                    pass
        if mcp is not None:
            try:
                mcp.call("save_project", {})
            finally:
                mcp.close()

        # Reload after potential MCP save and apply micro-strum on chords pattern notes
        with open(gl_path, encoding="utf-8") as fh:
            gdata = json.load(fh)
        matches = _resolve_channel_matches(gdata.get("channels") or [], track_hint, ("chord", "pad", "keys"))
        chord_cids = {str(ch.get("id")) for _i, ch in matches if "id" in ch}
        if chord_cids:
            strum_applied = False
            for pat in gdata.get("patterns") or []:
                nmap = pat.get("notes") or {}
                if not isinstance(nmap, dict):
                    continue
                for cid in list(nmap.keys()):
                    if cid in chord_cids and isinstance(nmap[cid], list) and len(nmap[cid]) >= 2:
                        beat_evs = [
                            {
                                **ev,
                                "start": float(ev.get("start", 0.0)) / ppq,
                                "duration": float(ev.get("length", ev.get("duration", ppq))) / ppq,
                                "pitch": int(ev.get("note", ev.get("pitch", 60))),
                            }
                            for ev in nmap[cid]
                        ]
                        apply_micro_strum(beat_evs, strum_ms=16.0, bpm=bpm)
                        for raw_ev, mod_ev in zip(nmap[cid], beat_evs):
                            raw_ev["start"] = round(float(mod_ev["start"]) * ppq, 4)
                            if "length" in raw_ev:
                                raw_ev["length"] = round(float(mod_ev["duration"]) * ppq, 4)
                        strum_applied = True
            if strum_applied:
                with open(gl_path, "w", encoding="utf-8") as fh:
                    json.dump(gdata, fh, indent=2, ensure_ascii=False)
                changed_items.append("micro_strum:16ms")

    # Also micro-strum Chords in master_sketch.mid if present
    mid_path = os.path.join(p_dir, "master_sketch.mid")
    if os.path.isfile(mid_path):
        pm = pretty_midi.PrettyMIDI(mid_path)
        stagger_s = 0.015
        for inst in pm.instruments:
            nm = (inst.name or "").lower()
            if any(k in nm for k in ("chord", "pad", "keys", "string")) and len(inst.notes) >= 2:
                by_t: dict[float, list[pretty_midi.Note]] = {}
                for n in sorted(inst.notes, key=lambda x: (x.start, x.pitch)):
                    t_key = round(n.start, 3)
                    by_t.setdefault(t_key, []).append(n)
                for _tk, grp in by_t.items():
                    if len(grp) >= 2:
                        for v_i, n in enumerate(grp):
                            shift = v_i * (stagger_s / max(1, len(grp)))
                            n.start += shift
                            n.end = max(n.start + 0.04, n.end)
                        changed_items.append(f"midi_strum:{inst.name}")
        pm.write(mid_path)

    return {
        "applied": bool(changed_items),
        "code": "STAGGER_ATTACK_OR_STRUM",
        "changes": changed_items,
    }


def _fix_s3_midi_action(
    project_dir: str,
    code: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    p_dir = os.path.abspath(os.path.expanduser(project_dir))
    mid_path = str(params.get("midi_path") or os.path.join(p_dir, "master_sketch.mid"))
    if not os.path.isfile(mid_path):
        return {"applied": False, "code": code, "reason": f"MIDI not found: {mid_path}"}

    bp_path = os.path.join(p_dir, "song_blueprint.json")
    expected_key = ""
    ts_arg: tuple[int, int] | None = None
    if os.path.isfile(bp_path):
        with open(bp_path, encoding="utf-8") as fh:
            bp = json.load(fh)
        expected_key = str((bp.get("harmonic_core") or {}).get("key") or "")
        ts_obj = bp.get("time_signature") or {}
        if ts_obj.get("numerator") and ts_obj.get("denominator"):
            ts_arg = (int(ts_obj["numerator"]), int(ts_obj["denominator"]))

    rep = autofix_midi(
        midi_path=mid_path,
        out_path=mid_path,
        in_place=True,
        expected_key=expected_key,
        time_signature=ts_arg,
        fix_s3=True,
        strict_s3=True,
    )
    return {
        "applied": bool(rep.get("total_changes", 0) > 0),
        "code": code,
        "fixed_midi_path": rep.get("fixed_midi_path"),
        "counts_by_rule": rep.get("counts_by_rule"),
        "gate_passed": rep.get("gate_passed"),
    }


_PRUNE_PRIORITY = ("counter", "fx", "chords", "pad", "arp", "perc", "clap", "hihat", "snare")


def _arrangement_plan_path(p_dir: str) -> str:
    return os.path.join(p_dir, "arrangement_plan.json")


def _target_gridline_for_plan(p_dir: str, explicit_path: str = "") -> str:
    if explicit_path and os.path.isfile(os.path.expanduser(explicit_path)):
        return os.path.abspath(os.path.expanduser(explicit_path))
    for cand in ("projet.gridline", "project.gridline"):
        p = os.path.join(p_dir, cand)
        if os.path.isfile(p):
            return p
    if os.path.isdir(p_dir):
        for fn in sorted(os.listdir(p_dir)):
            if fn.endswith(".gridline") and fn != "core_loop.gridline":
                return os.path.join(p_dir, fn)
    return os.path.join(p_dir, "projet.gridline")


def _save_and_recompile_plan(p_dir: str, plan: dict[str, Any], gridline_path: str = "") -> str:
    plan_p = _arrangement_plan_path(p_dir)
    with open(plan_p, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2, ensure_ascii=False)
    out_gl = _target_gridline_for_plan(p_dir, gridline_path)
    compile_arrangement_plan(plan_p, p_dir, out_gl, core_only=False, verify_gate=False)
    return out_gl


def _parse_bars_range(params: dict[str, Any], default: tuple[int, int] = (9, 16)) -> tuple[int, int]:
    bars_str = str(params.get("bars") or "").strip("'\"")
    if "-" in bars_str:
        try:
            a, b = bars_str.split("-", 1)
            return int(a), int(b)
        except ValueError:
            pass
    if params.get("start_bar") and params.get("end_bar"):
        return int(params["start_bar"]), int(params["end_bar"])
    return default


def _fix_prune_track_in_section(
    project_dir: str,
    params: dict[str, Any],
    *,
    gridline_path: str = "",
) -> dict[str, Any]:
    p_dir = os.path.abspath(os.path.expanduser(project_dir))
    plan_p = _arrangement_plan_path(p_dir)
    if not os.path.isfile(plan_p):
        return {"applied": False, "code": "PRUNE_TRACK_IN_SECTION", "reason": "arrangement_plan.json not found"}

    plan, _meta = load_arrangement_plan(plan_p, validate=True)
    sections = plan.get("sections") or []
    if not sections:
        return {"applied": False, "code": "PRUNE_TRACK_IN_SECTION", "reason": "No sections in arrangement_plan.json"}

    sec_hint = str(params.get("section") or "").strip().lower()
    target_sec: dict[str, Any] | None = None
    if sec_hint:
        target_sec = next((s for s in sections if str(s.get("name") or "").lower() == sec_hint), None)
        if target_sec is None:
            target_sec = next((s for s in sections if sec_hint in str(s.get("name") or "").lower()), None)
    if target_sec is None and params.get("bars"):
        sb, eb = _parse_bars_range(params, (1, 8))
        target_sec = next(
            (
                s for s in sections
                if int(s["start_bar"]) <= (eb - 1) and (int(s["start_bar"]) + int(s["bars"]) - 1) >= (sb - 1)
            ),
            None,
        )
    if target_sec is None:
        target_sec = max(sections, key=lambda s: len(s.get("active_roles") or []))

    active = list(target_sec.get("active_roles") or [])
    if len(active) <= 1:
        return {"applied": False, "code": "PRUNE_TRACK_IN_SECTION", "reason": "Section has <= 1 active role"}

    req_role = str(params.get("role") or "").strip()
    pruned = ""
    if req_role:
        match_r = next((r for r in active if r.lower() == req_role.lower()), "")
        if match_r:
            pruned = match_r
    if not pruned:
        for cand in _PRUNE_PRIORITY:
            match_r = next((r for r in active if cand in r.lower()), "")
            if match_r:
                pruned = match_r
                break
    if not pruned:
        pruned = active[-1]

    target_sec["active_roles"] = [r for r in active if r != pruned]
    out_gl = _save_and_recompile_plan(p_dir, plan, gridline_path=gridline_path)
    return {
        "applied": True,
        "code": "PRUNE_TRACK_IN_SECTION",
        "section": target_sec["name"],
        "pruned_role": pruned,
        "recompiled": out_gl,
    }


def _mutate_plan_block(
    project_dir: str,
    params: dict[str, Any],
    *,
    gridline_path: str = "",
) -> dict[str, Any]:
    p_dir = os.path.abspath(os.path.expanduser(project_dir))
    plan_p = _arrangement_plan_path(p_dir)
    plan, meta = load_arrangement_plan(plan_p, validate=True)
    sections = plan.get("sections") or []
    bpb = float(meta["beats_per_bar"])

    sb, eb = _parse_bars_range(params, (9, 16))
    sb0 = max(0, sb - 1)
    eb0 = max(sb0, eb - 1)

    matching_indices = [
        i for i, s in enumerate(sections)
        if int(s["start_bar"]) <= eb0 and (int(s["start_bar"]) + int(s["bars"]) - 1) >= sb0
    ]
    if not matching_indices:
        matching_indices = [min(len(sections) - 1, max(0, len(sections) // 2))]

    target_idx = matching_indices[-1]
    cand_sec = sections[target_idx]
    split_performed = False

    # If block is >= 16 bars, split into two 8-bar (or half) sub-blocks first
    if int(cand_sec["bars"]) >= 16:
        orig_bars = int(cand_sec["bars"])
        half = orig_bars // 2
        rest = orig_bars - half
        sec_a = copy.deepcopy(cand_sec)
        sec_b = copy.deepcopy(cand_sec)
        sec_a["bars"] = half
        sec_b["name"] = f"{cand_sec['name']}_B"
        sec_b["start_bar"] = int(cand_sec["start_bar"]) + half
        sec_b["bars"] = rest

        # Slice non-looped section-length motifs into sec_b-specific motifs if needed
        motifs = plan.setdefault("motifs", {})
        split_beat = float(half) * bpb
        for role in list(sec_b.get("active_roles") or []):
            mods_b = sec_b.setdefault("role_modifiers", {})
            rm = mods_b.get(role) or {}
            m_ref = rm.get("motif")
            raw_m = motifs.get(m_ref) if m_ref else motifs.get(role)
            notes_src: list[dict[str, Any]] | None = None
            loop_flag: bool | None = None
            if isinstance(raw_m, list):
                notes_src = raw_m
            elif isinstance(raw_m, dict) and isinstance(raw_m.get("notes"), list):
                notes_src = raw_m["notes"]
                loop_flag = raw_m.get("loop")
            if notes_src and loop_flag is not True:
                max_t = max(float(n["start"]) + float(n.get("length", n.get("duration", 0.25))) for n in notes_src)
                if max_t > split_beat + 0.1:
                    sliced = []
                    for n in notes_src:
                        st = float(n["start"])
                        if split_beat - 1e-4 <= st < float(orig_bars) * bpb - 1e-4:
                            sliced.append({**n, "start": round(st - split_beat, 5)})
                    if sliced:
                        new_m_key = f"{role}_{sec_b['name']}"
                        motifs[new_m_key] = sliced
                        mods_b.setdefault(role, {})["motif"] = new_m_key

        sections[target_idx : target_idx + 1] = [sec_a, sec_b]
        target_sec = sec_b
        split_performed = True
    else:
        target_sec = cand_sec

    active = list(target_sec.get("active_roles") or [])
    mods = target_sec.setdefault("role_modifiers", {})
    mutated_roles: list[str] = []

    for r_key in ("counter", "lead"):
        match_r = next((r for r in active if r_key in r.lower()), "")
        if match_r:
            rm = mods.setdefault(match_r, {})
            cur_oct = int(rm.get("octave_shift", 0) or 0)
            rm["octave_shift"] = 12 if cur_oct == 0 else (0 if cur_oct == 12 else 12)
            cur_cut = float(rm.get("cutoff_hz") or 3600.0)
            rm["cutoff_hz"] = round(min(12000.0, cur_cut + 1200.0), 1)
            mutated_roles.append(f"{match_r}:oct={rm['octave_shift']},lpf={rm['cutoff_hz']}")
            break

    hihat_r = next((r for r in active if "hat" in r.lower()), "")
    if hihat_r:
        rm = mods.setdefault(hihat_r, {})
        cur_d = float(rm.get("density_scale", 1.0) or 1.0)
        rm["density_scale"] = 1.25 if cur_d < 1.15 else 0.75
        mutated_roles.append(f"{hihat_r}:density={rm['density_scale']}")

    chords_r = next((r for r in active if any(k in r.lower() for k in ("chord", "pad"))), "")
    if chords_r:
        rm = mods.setdefault(chords_r, {})
        cur_cut = float(rm.get("cutoff_hz") or 3200.0)
        rm["cutoff_hz"] = round(min(12000.0, cur_cut + 1200.0), 1)
        mutated_roles.append(f"{chords_r}:lpf={rm['cutoff_hz']}")

    if not mutated_roles and active:
        first_r = active[0]
        rm = mods.setdefault(first_r, {})
        cur_cut = float(rm.get("cutoff_hz") or 3200.0)
        rm["cutoff_hz"] = round(min(12000.0, cur_cut + 1200.0), 1)
        mutated_roles.append(f"{first_r}:lpf={rm['cutoff_hz']}")

    out_gl = _save_and_recompile_plan(p_dir, plan, gridline_path=gridline_path)
    return {
        "applied": True,
        "code": "MUTATE_BLOCK_DENSITY_OR_LPF",
        "section": target_sec["name"],
        "split_16bar": split_performed,
        "bars": f"{sb}-{eb}",
        "mutations": mutated_roles,
        "recompiled": out_gl,
    }


def _fix_mutate_block_density_or_lpf(
    project_dir: str,
    params: dict[str, Any],
    *,
    gridline_path: str = "",
) -> dict[str, Any]:
    p_dir = os.path.abspath(os.path.expanduser(project_dir))
    if os.path.isfile(_arrangement_plan_path(p_dir)):
        return _mutate_plan_block(p_dir, params, gridline_path=gridline_path)

    gl_path = _find_gridline_project(p_dir, gridline_path)
    if not gl_path or not os.path.isfile(gl_path):
        return {"applied": False, "code": "MUTATE_BLOCK_DENSITY_OR_LPF", "reason": ".gridline not found"}

    with open(gl_path, encoding="utf-8") as fh:
        gdata = json.load(fh)
    bpb = float(gdata.get("timeSigNum") or 4)
    sb, eb = _parse_bars_range(params, (9, 16))

    start_beat = max(0.0, (sb - 1) * bpb)
    end_beat = max(start_beat + bpb, eb * bpb)

    # Locate BUS Music mixer track & exact gl.eq slot dynamically via GridlineMCP get_mixer
    autos: list[dict[str, Any]] = []
    mcp = GridlineMCP(gl_path, os.path.dirname(gl_path))
    mixer_idx = 0
    eq_slot = 1
    try:
        mx = mcp.call("get_mixer", {})
        tracks = mx.get("tracks") or []
        target_tr = next(
            (t for t in tracks if "bus music" in str(t.get("name") or "").lower()),
            tracks[0] if tracks else {"mixer": 0, "effects": []},
        )
        mixer_idx = int(target_tr.get("mixer", target_tr.get("index", 0)))
        effects = target_tr.get("effects") or []
        eq_eff = next(
            (e for e in effects if isinstance(e, dict) and str(e.get("plugin") or "") == "gl.eq"),
            None,
        )
        if eq_eff is not None:
            eq_slot = int(eq_eff.get("slot", 1))
        else:
            used_slots = {int(e.get("slot", 0)) for e in effects if isinstance(e, dict)}
            eq_slot = next((s for s in range(1, 11) if s not in used_slots), 1)
            mcp.call("add_plugin", {"mixer": mixer_idx, "slot": eq_slot, "plugin": "gl.eq"})

        filter_sweep(
            autos,
            mixer_idx=mixer_idx,
            slot_idx=eq_slot,
            start_beat=start_beat,
            end_beat=end_beat,
            start_hz=1100.0,
            end_hz=11500.0,
            curve="exp",
        )
        for auto in autos:
            mcp.call("add_automation", auto)
        mcp.call("save_project", {})
    finally:
        mcp.close()

    return {
        "applied": True,
        "code": "MUTATE_BLOCK_DENSITY_OR_LPF",
        "bars": f"{sb}-{eb}",
        "mixer": mixer_idx,
        "slot": eq_slot,
        "sweep_hz": "1100->11500",
    }


def _fix_outro_fade(
    project_dir: str,
    params: dict[str, Any],
    *,
    gridline_path: str = "",
) -> dict[str, Any]:
    p_dir = os.path.abspath(os.path.expanduser(project_dir))
    gl_path = _find_gridline_project(p_dir, gridline_path)
    if not gl_path or not os.path.isfile(gl_path):
        return {"applied": False, "code": "APPLY_OUTRO_FADE", "reason": ".gridline not found"}

    with open(gl_path, encoding="utf-8") as fh:
        gdata = json.load(fh)
    bpb = float(gdata.get("timeSigNum") or 4)
    ppq = float(gdata.get("ppq") or 96)
    pats = {p.get("id"): p for p in (gdata.get("patterns") or []) if "id" in p}
    max_ticks = 0
    for clip in gdata.get("playlist") or []:
        pat = pats.get(clip.get("pattern")) or {}
        p_ticks = int(pat.get("bars", 1)) * bpb * ppq
        max_ticks = max(max_ticks, int(clip.get("start", 0)) + int(p_ticks))
    total_bars = max(4, int(round(max_ticks / max(1.0, bpb * ppq))))

    start_beat = max(0.0, (total_bars - 2) * bpb)
    end_beat = float(total_bars) * bpb
    autos: list[dict[str, Any]] = []
    outro_fade(autos, master_or_bus_idx=0, start_beat=start_beat, end_beat=end_beat, base_db=0.0)

    mcp = GridlineMCP(gl_path, os.path.dirname(gl_path))
    try:
        for auto in autos:
            mcp.call("add_automation", auto)
        mcp.call("save_project", {})
    finally:
        mcp.close()

    return {
        "applied": True,
        "code": "APPLY_OUTRO_FADE",
        "bars": f"{max(1, total_bars - 1)}-{total_bars}",
    }


def _fix_reinforce_section_transition(
    project_dir: str,
    params: dict[str, Any],
    *,
    gridline_path: str = "",
) -> dict[str, Any]:
    p_dir = os.path.abspath(os.path.expanduser(project_dir))
    plan_p = _arrangement_plan_path(p_dir)
    if os.path.isfile(plan_p):
        plan, _meta = load_arrangement_plan(plan_p, validate=True)
        sections = plan.get("sections") or []
        if sections:
            sec_hint = str(params.get("section") or "").strip().lower()
            if not sec_hint and params.get("transition") and "->" in str(params["transition"]):
                sec_hint = str(params["transition"]).split("->", 1)[1].strip().lower()
            target_idx = -1
            if sec_hint:
                for i, s in enumerate(sections):
                    nm = str(s.get("name") or "").lower()
                    if nm == sec_hint or sec_hint in nm:
                        target_idx = i
                        break
            if target_idx < 0 and params.get("bar"):
                bar0 = max(0, int(params["bar"]) - 1)
                for i, s in enumerate(sections):
                    if int(s["start_bar"]) <= bar0 < int(s["start_bar"]) + int(s["bars"]):
                        target_idx = i
                        break
            if target_idx < 0:
                target_idx = 1 if len(sections) > 1 else 0

            target_sec = sections[target_idx]
            dropout_injected = False
            if target_idx > 0:
                prev_sec = sections[target_idx - 1]
                last_rel_bar = max(0, int(prev_sec["bars"]) - 1)
                tr_prev = prev_sec.setdefault("transitions", {})
                drops = list(tr_prev.get("beat4_dropout_bars") or [])
                if last_rel_bar not in drops:
                    drops.append(last_rel_bar)
                tr_prev["beat4_dropout_bars"] = drops
                dropout_injected = True

            active = list(target_sec.get("active_roles") or [])
            sweep_role = next(
                (r for r in active if any(k in r.lower() for k in ("chord", "pad", "lead"))),
                active[0] if active else "chords",
            )
            tr_cur = target_sec.setdefault("transitions", {})
            tr_cur["filter_sweep"] = {
                "role": sweep_role,
                "start_hz": 900.0,
                "end_hz": 4500.0,
                "bars": min(2, max(1, int(target_sec["bars"]))),
                "curve": "exp",
            }
            out_gl = _save_and_recompile_plan(p_dir, plan, gridline_path=gridline_path)
            return {
                "applied": True,
                "code": "REINFORCE_SECTION_TRANSITION",
                "section": target_sec["name"],
                "beat4_dropout_injected": dropout_injected,
                "filter_sweep": tr_cur["filter_sweep"],
                "recompiled": out_gl,
            }

    b_bar = max(3, int(params.get("bar", 9)))
    return _fix_mutate_block_density_or_lpf(
        project_dir,
        {"start_bar": b_bar - 2, "end_bar": b_bar},
        gridline_path=gridline_path,
    ) | {"code": "REINFORCE_SECTION_TRANSITION"}


def _load_fader_cum_state(p_dir: str) -> dict[str, float]:
    st_path = os.path.join(p_dir, "renders", "autofix_fader_state.json")
    if os.path.isfile(st_path):
        try:
            with open(st_path, encoding="utf-8") as fh:
                raw = json.load(fh) or {}
            if isinstance(raw, dict):
                return {str(k): float(v) for k, v in raw.items()}
        except Exception:
            pass
    return {}


def _save_fader_cum_state(p_dir: str, cum_map: dict[str, float]) -> None:
    renders_dir = os.path.join(p_dir, "renders")
    os.makedirs(renders_dir, exist_ok=True)
    st_path = os.path.join(renders_dir, "autofix_fader_state.json")
    with open(st_path, "w", encoding="utf-8") as fh:
        json.dump(cum_map, fh, indent=2, ensure_ascii=False)


def _fix_adjust_track_fader(
    project_dir: str,
    params: dict[str, Any],
    *,
    gridline_path: str = "",
) -> dict[str, Any]:
    """
    Phase 8 Q2: Deterministic `ADJUST_TRACK_FADER` handler.
      - Caps per-pass delta to `[-4.0, +4.0] dB`
      - Caps cumulative delta across all passes to `[-8.0, +8.0] dB`
      - Persists `volume_trim_db` in `arrangement_plan.json` (if present) + `gain_staging_overrides.json`
      - Patches `.gridline` mixer track volume via `GridlineMCP` + `gl_dsl.db_to_auto`
    """
    p_dir = os.path.abspath(os.path.expanduser(project_dir))
    track_name = str(params.get("track_name") or params.get("track") or params.get("target_track") or "").strip()
    role_hint = str(params.get("target_role") or params.get("role") or "").strip()
    insert_idx = params.get("insert_idx") if params.get("insert_idx") is not None else params.get("mixer")
    raw_delta = float(params.get("delta_db", 0.0))
    try:
        from mix_config import autofix_step_bounds
        _step_cap = autofix_step_bounds()["max_step_db"]
    except Exception:
        _step_cap = 4.0
    step_req = clamp(raw_delta, -_step_cap, _step_cap)
    if abs(step_req) < 0.05:
        return {"applied": False, "code": "ADJUST_TRACK_FADER", "reason": "delta_db too small"}

    cum_key = (
        f"insert:{int(insert_idx)}"
        if insert_idx is not None
        else f"track:{(track_name or role_hint).lower()}"
    )
    cum_map = _load_fader_cum_state(p_dir)
    prev_cum = float(cum_map.get(cum_key, 0.0))
    new_cum = clamp(prev_cum + step_req, -8.0, 8.0)
    eff_delta = round(new_cum - prev_cum, 3)
    if abs(eff_delta) < 0.05:
        return {
            "applied": False,
            "code": "ADJUST_TRACK_FADER",
            "track": track_name or role_hint,
            "cumulative_delta_db": prev_cum,
            "escalate": "SWAP_SAMPLE",
            "reason": "Cumulative +-8.0 dB limit reached; escalate to SWAP_SAMPLE / CHECK_SOURCE_GAIN.",
        }

    # Update gain_staging_overrides.json so future song_builder recompilations keep the trim
    ov_path = os.path.join(p_dir, "gain_staging_overrides.json")
    overrides: dict[str, Any] = {}
    if os.path.isfile(ov_path):
        try:
            with open(ov_path, encoding="utf-8") as fh:
                overrides = json.load(fh) or {}
        except Exception:
            overrides = {}
    ov_key = track_name or role_hint or (str(int(insert_idx)) if insert_idx is not None else "track")
    prev_ov = float((overrides.get(ov_key) or {}).get("volume_trim_db", 0.0))
    overrides[ov_key] = {
        "volume_trim_db": round(clamp(prev_ov + eff_delta, -8.0, 8.0), 3),
        "role": role_hint or normalize_role_for_gain_staging(track_name),
    }
    if os.path.isdir(p_dir):
        with open(ov_path, "w", encoding="utf-8") as fh:
            json.dump(overrides, fh, indent=2, ensure_ascii=False)

    # Declarative persistence in arrangement_plan.json when present
    plan_p = _arrangement_plan_path(p_dir)
    if os.path.isfile(plan_p):
        try:
            plan, _ = load_arrangement_plan(plan_p, validate=False)
            ch_cfg = plan.get("channels")
            canon = normalize_role_for_gain_staging(role_hint or track_name)
            if isinstance(ch_cfg, list):
                matched = False
                for item in ch_cfg:
                    if not isinstance(item, dict):
                        continue
                    nm = str(item.get("name") or "").lower()
                    tk = str(item.get("track_key") or "").lower()
                    if (
                        (track_name and (track_name.lower() in nm or track_name.lower() == tk))
                        or (role_hint and (role_hint.lower() == tk or normalize_role_for_gain_staging(nm or tk) == canon))
                    ):
                        item["volume_trim_db"] = round(clamp(float(item.get("volume_trim_db", 0.0)) + eff_delta, -8.0, 8.0), 3)
                        matched = True
                        break
                if not matched and isinstance(ch_cfg, list) and len(ch_cfg) > 0:
                    pass
            else:
                if not isinstance(ch_cfg, dict):
                    ch_cfg = {}
                target_role_key = role_hint or next(
                    (r for sec in (plan.get("sections") or []) for r in (sec.get("active_roles") or [])
                     if normalize_role_for_gain_staging(r) == canon or (track_name and track_name.lower() in r.lower())),
                    track_name or canon,
                )
                ent = dict(ch_cfg.get(target_role_key) or {})
                ent["volume_trim_db"] = round(clamp(float(ent.get("volume_trim_db", 0.0)) + eff_delta, -8.0, 8.0), 3)
                ch_cfg[target_role_key] = ent
                plan["channels"] = ch_cfg

            cum_map[cum_key] = round(new_cum, 3)
            _save_fader_cum_state(p_dir, cum_map)
            out_gl = _save_and_recompile_plan(p_dir, plan, gridline_path=gridline_path)
            return {
                "applied": True,
                "code": "ADJUST_TRACK_FADER",
                "mode": "declarative_plan",
                "track": track_name or role_hint,
                "delta_db": eff_delta,
                "cumulative_delta_db": cum_map[cum_key],
                "recompiled": out_gl,
            }
        except Exception:
            pass

    # Direct .gridline patch via GridlineMCP + gl_dsl.db_to_auto
    gl_path = _find_gridline_project(p_dir, gridline_path)
    if not gl_path:
        return {"applied": False, "code": "ADJUST_TRACK_FADER", "reason": ".gridline and arrangement_plan.json not found"}

    fader_state = {"fader_cumulative_db": {f"mixer:{int(insert_idx)}" if insert_idx is not None else "pending": prev_cum}}
    res = adjust_track_fader(
        gl_path,
        track_name=track_name,
        insert_idx=int(insert_idx) if insert_idx is not None else None,
        role_hint=role_hint,
        delta_db=eff_delta,
        state=None,
    )
    if res.get("applied"):
        cum_map[cum_key] = round(new_cum, 3)
        _save_fader_cum_state(p_dir, cum_map)
        res["cumulative_delta_db"] = cum_map[cum_key]
    return res


def _fix_sync_vocal_arrangement(
    project_dir: str,
    params: dict[str, Any],
    *,
    gridline_path: str = "",
) -> dict[str, Any]:
    p_dir = os.path.abspath(os.path.expanduser(project_dir))
    ustx_cands = [os.path.join(p_dir, fn) for fn in sorted(os.listdir(p_dir)) if fn.endswith(".ustx")] if os.path.isdir(p_dir) else []
    vox_wav = ""
    for sub in ("samples", "Samples"):
        cand = os.path.join(p_dir, sub, "vocals.wav")
        if os.path.isfile(cand):
            vox_wav = cand
            break
    if not ustx_cands and not vox_wav:
        return {"applied": False, "code": "SYNC_VOCAL_ARRANGEMENT", "reason": "No .ustx or vocals.wav found"}

    bpm = float(params.get("bpm") or 0.0)
    if bpm <= 0.0:
        for jfn in ("song_blueprint.json", "arrangement_plan.json"):
            jp = os.path.join(p_dir, jfn)
            if os.path.isfile(jp):
                try:
                    with open(jp, encoding="utf-8") as fh:
                        bpm = float(json.load(fh).get("bpm") or 0.0)
                    if bpm > 0.0:
                        break
                except Exception:
                    pass
    if bpm <= 0.0:
        bpm = 120.0

    phrases = extract_vocal_phrases(
        ustx_path=ustx_cands[0] if ustx_cands else "",
        audio_path=vox_wav,
        bpm=bpm,
    )
    if not phrases:
        return {"applied": False, "code": "SYNC_VOCAL_ARRANGEMENT", "reason": "No active vocal phrases detected"}

    gl_path = _find_gridline_project(p_dir, gridline_path)
    if not gl_path:
        plan_p = _arrangement_plan_path(p_dir)
        if os.path.isfile(plan_p):
            out_gl = os.path.join(p_dir, "projet.gridline")
            compile_arrangement_plan(plan_p, p_dir, out_gl, apply_vocal_sync=True, verify_gate=False)
            return {
                "applied": True,
                "code": "SYNC_VOCAL_ARRANGEMENT",
                "phrases_count": len(phrases),
                "recompiled": True,
            }
        return {"applied": False, "code": "SYNC_VOCAL_ARRANGEMENT", "reason": ".gridline not found"}

    with open(gl_path, encoding="utf-8") as fh:
        gdata = json.load(fh)
    bpm_gl = float(gdata.get("tempo") or bpm)
    ppq = float(gdata.get("ppq") or 96.0)

    mcp = GridlineMCP(gl_path, os.path.dirname(gl_path))
    muted_notes_delta = 0
    bm_idx = 19
    eq_slot = 2
    try:
        mx = mcp.call("get_mixer", {}).get("tracks", [])
        bm_track = next((t for t in mx if "bus music" in str(t.get("name") or "").lower()), None)
        if bm_track is None:
            bm_track = next((t for t in mx if int(t["mixer"]) == 19), None)
        if bm_track is not None:
            bm_idx = int(bm_track["mixer"])
            slots = [
                int(e.get("slot", e.get("index", idx + 1)))
                for idx, e in enumerate(bm_track.get("effects") or bm_track.get("slots") or bm_track.get("plugins") or [])
                if e.get("plugin", e.get("name")) == "gl.eq"
            ]
            if not slots:
                mcp.call("add_plugin", {"mixer": bm_idx, "plugin": "gl.eq"})
                mx2 = mcp.call("get_mixer", {}).get("tracks", [])
                bm_track = next((t for t in mx2 if int(t["mixer"]) == bm_idx), bm_track)
                slots = [
                    int(e.get("slot", e.get("index", idx + 1)))
                    for idx, e in enumerate(bm_track.get("effects") or bm_track.get("slots") or bm_track.get("plugins") or [])
                    if e.get("plugin", e.get("name")) == "gl.eq"
                ]
            eq_slot = slots[1] if len(slots) >= 2 else (slots[-1] if slots else 1)

        # Configure Band 5 = 1850 Hz peak Q=1.4 and write formant ducking automation
        mcp.set_params(formant_ducking_eq_params(bm_idx, eq_slot, 5, 1850.0, 1.4))
        auto_payload = build_formant_ducking_automation(
            phrases,
            bpm=bpm_gl,
            mixer_idx=bm_idx,
            slot_idx=eq_slot,
            band_idx=5,
            track_idx=min(31, max(12, len(gdata.get("channels") or []) + 4)),
        )
        mcp.call("add_automation", auto_payload)

        # Also filter overlapping counter notes per pattern clip if counter channels exist
        ch_list = gdata.get("channels") or []
        counter_chs = [
            (idx1, ch.get("name", f"ch{idx1}"))
            for idx1, ch in enumerate(ch_list, start=1)
            if any(k in str(ch.get("name") or "").lower() for k in ("counter", "contre", "reply", "response"))
        ]
        if counter_chs:
            pats_by_id = {int(p["id"]): p for p in (gdata.get("patterns") or []) if "id" in p}
            # Map first clip start beat per pattern for absolute phrase alignment
            pat_start_beat: dict[int, float] = {}
            for cl in gdata.get("playlist") or []:
                pid = int(cl.get("pattern", 0))
                if pid and pid not in pat_start_beat:
                    pat_start_beat[pid] = float(cl.get("start", 0)) / ppq
            for ch_idx1, ch_name in counter_chs:
                for pid, pat in pats_by_id.items():
                    base_b = pat_start_beat.get(pid, 0.0)
                    raw_notes = (pat.get("notes") or {}).get(str(ch_idx1)) or []
                    if not raw_notes:
                        continue
                    abs_tuples = [
                        (base_b + float(n.get("start", 0)) / ppq, float(n.get("length", ppq)) / ppq, int(n.get("pitch", 60)), int(n.get("velocity", 90)))
                        for n in raw_notes
                    ]
                    synced = mute_counter_during_vocals({"counter": abs_tuples}, phrases)["counter"]
                    if len(synced) < len(abs_tuples):
                        muted_notes_delta += len(abs_tuples) - len(synced)
                        new_notes = [
                            {"start": round(s - base_b, 4), "length": round(d, 4), "pitch": p, "velocity": v}
                            for s, d, p, v in synced
                            if s - base_b >= -1e-4
                        ]
                        mcp.call("set_notes", {"pattern": pid, "channel": ch_idx1, "notes": new_notes})
        mcp.call("save_project", {})
    finally:
        mcp.close()

    return {
        "applied": True,
        "code": "SYNC_VOCAL_ARRANGEMENT",
        "phrases_count": len(phrases),
        "mixer": bm_idx,
        "eq_slot": eq_slot,
        "band": 5,
        "freq_hz": 1850.0,
        "duck_db": -3.5,
        "muted_counter_notes": muted_notes_delta,
    }


def _fix_track_root_note(p_dir: str, params: dict[str, Any], gridline_path: str = "") -> dict[str, Any]:
    """v3.1 HARM-03 : patch le `root_note` d'une piste du blueprint (delta en demi-tons),
    ex. après `FIX_TRACK_ROOT_NOTE(track, delta_semitones=-k)` émis par
    `evaluate_barwise_harmonic_alignment` (biais de hauteur constant détecté sur l'audio)."""
    track = str(params.get("track") or params.get("track_name") or "")
    delta = int(params.get("delta_semitones") or params.get("delta") or 0)
    if not track or delta == 0:
        return {"applied": False, "code": "FIX_TRACK_ROOT_NOTE", "reason": "track ou delta manquant"}
    changed: list[dict[str, Any]] = []
    bp_path = os.path.join(p_dir, "song_blueprint.json")
    if os.path.isfile(bp_path):
        try:
            with open(bp_path, encoding="utf-8") as fh:
                bp = json.load(fh)
            tl = track.lower()
            for tr in bp.get("tracks") or []:
                name = str(tr.get("name") or tr.get("role") or "")
                if name and (tl in name.lower() or name.lower() in tl):
                    if tr.get("root_note") is not None:
                        old = int(tr["root_note"])
                        tr["root_note"] = old + delta
                        changed.append({"track": name, "root_note": tr["root_note"], "delta": delta})
                    if tr.get("midi_root") is not None:
                        tr["midi_root"] = int(tr["midi_root"]) + delta
            if changed:
                with open(bp_path, "w", encoding="utf-8") as fh:
                    json.dump(bp, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            return {"applied": False, "code": "FIX_TRACK_ROOT_NOTE", "reason": str(exc)}
    plan_path = _arrangement_plan_path(p_dir)
    if os.path.isfile(plan_path) and changed:
        try:
            with open(plan_path, encoding="utf-8") as fh:
                plan = json.load(fh)
            tl = track.lower()
            for ch in plan.get("channels") or []:
                if not isinstance(ch, dict):
                    continue
                name = str(ch.get("track_key") or ch.get("name") or "")
                if name and (tl in name.lower() or name.lower() in tl) and ch.get("root_note") is not None:
                    ch["root_note"] = int(ch["root_note"]) + delta
            with open(plan_path, "w", encoding="utf-8") as fh:
                json.dump(plan, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return {"applied": bool(changed), "code": "FIX_TRACK_ROOT_NOTE", "track": track,
            "delta_semitones": delta, "changed": changed}


def dispatch_multiscale_autofix(
    project_dir: str,
    deviations: list[dict[str, Any]] | dict[str, Any],
    *,
    gridline_path: str = "",
    render_summary: dict[str, Any] | None = None,
    respect_guardrails: bool = True,
) -> dict[str, Any]:
    """
    Deterministically execute all applicable S1..S4 mechanical autofixes from
    a pyramid/deviation list without LLM intervention.
    """
    p_dir = os.path.abspath(os.path.expanduser(project_dir))
    dev_list = [deviations] if isinstance(deviations, dict) else list(deviations or [])
    user_locked = detect_user_samples_locked(p_dir)
    sm: LoopFirstStateMachine | None = None
    if os.path.isdir(p_dir) and respect_guardrails:
        try:
            sm = LoopFirstStateMachine(p_dir)
            user_locked = bool(sm.state.get("user_samples_locked", user_locked))
        except Exception:
            sm = None

    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    escalate_to_llm: list[dict[str, Any]] = []
    delegated_e4 = False

    for dev in dev_list:
        code, params = _parse_action_spec(dev)
        if code in ("KEEP", "NONE", "PASS", "PROCEED_TO_NEXT_STAGE", "PROCEED_TO_FINAL_RENDER", "REVIEW"):
            continue

        # Check Loop-First fault-isolation guardrails if track targeted
        target_track = str(params.get("track") or params.get("track_name") or params.get("target_track") or "")
        target_role = str(params.get("target_role") or params.get("role") or "")
        if sm is not None and (target_track or target_role):
            cur_stage = sm.state.get("stage")
            v4_passed = ((sm.state.get("vertical_results") or {}).get("V4_LEAD") or {}).get("status") == "PASS"
            enforce_forb = True
            if code == "ADJUST_TRACK_FADER" and (cur_stage in ("E4_FINAL_GATE", "DONE", "E3_HORIZONTAL", "AWAITING_E2_CHECKPOINT") or (cur_stage == "E2_VERTICAL" and v4_passed)):
                enforce_forb = False
            if enforce_forb:
                forb_list = list(((sm.state.get("next_mandatory_step") or {}).get("forbidden_tracks")) or [])
                blocked = False
                if target_track:
                    chk = sm.validate_edit_target(target_track, raise_on_forbidden=False)
                    if not chk.get("allowed", True):
                        blocked = True
                if not blocked and forb_list:
                    t_low = target_track.lower()
                    t_canon = normalize_role_for_gain_staging(target_role or target_track) if (target_role or target_track) else ""
                    for f_name in forb_list:
                        fl = str(f_name).lower()
                        if (t_low and (t_low in fl or fl in t_low)) or (t_canon and normalize_role_for_gain_staging(fl) == t_canon):
                            blocked = True
                            break
                if blocked:
                    skipped.append({"code": code, "reason": "forbidden_tracks guardrail", "track": target_track or target_role})
                    escalate_to_llm.append({"code": code, "reason": "forbidden_tracks", "track": target_track or target_role})
                    continue

        if code == "INVERT_808_POLARITY":
            res = _fix_invert_808_polarity(p_dir, params, gridline_path=gridline_path, user_locked=user_locked)
        elif code == "SHIFT_BASS_MICRO_MS":
            res = _fix_shift_bass_micro_ms(p_dir, params, gridline_path=gridline_path)
        elif code == "TRIM_SAMPLE_START_MS":
            res = _fix_trim_sample_start_ms(p_dir, params, gridline_path=gridline_path, user_locked=user_locked)
        elif code == "STAGGER_ATTACK_OR_STRUM":
            res = _fix_stagger_attack_or_strum(p_dir, params, gridline_path=gridline_path)
        elif code in ("FIX_MOTIF_CLONE", "FIX_CHORDS_LEAD_OVERLAP"):
            res = _fix_s3_midi_action(p_dir, code, params)
        elif code == "PRUNE_TRACK_IN_SECTION":
            res = _fix_prune_track_in_section(p_dir, params, gridline_path=gridline_path)
        elif code == "MUTATE_BLOCK_DENSITY_OR_LPF":
            res = _fix_mutate_block_density_or_lpf(p_dir, params, gridline_path=gridline_path)
        elif code == "REINFORCE_SECTION_TRANSITION":
            res = _fix_reinforce_section_transition(p_dir, params, gridline_path=gridline_path)
        elif code == "APPLY_OUTRO_FADE":
            res = _fix_outro_fade(p_dir, params, gridline_path=gridline_path)
        elif code == "APPLY_DYNAMIC_MIRROR_EQ":
            gl_path = _find_gridline_project(p_dir, gridline_path)
            if gl_path:
                freq_hz = float(params.get("freq_hz") or params.get("value") or 1800.0)
                gain_db = float(params.get("gain_db", -3.5))
                q_val = float(params.get("q", 1.8))
                res = apply_dynamic_mirror_eq(gl_path, freq_hz, gain_db=gain_db, q=q_val)
            else:
                res = {"applied": False, "code": code, "reason": ".gridline not found"}
        elif code == "RELAX_BUS_COMPRESSION":
            gl_path = _find_gridline_project(p_dir, gridline_path)
            if gl_path:
                res = relax_bus_compression(gl_path)
            else:
                res = {"applied": False, "code": code, "reason": ".gridline not found"}
        elif code == "BOOST_BASS_HARMONICS":
            gl_path = _find_gridline_project(p_dir, gridline_path)
            if gl_path:
                f0_hz = float(params.get("f0_hz") or params.get("value") or 45.0)
                res = boost_bass_harmonics(gl_path, f0_hz)
            else:
                res = {"applied": False, "code": code, "reason": ".gridline not found"}
        elif code == "SYNC_VOCAL_ARRANGEMENT":
            res = _fix_sync_vocal_arrangement(p_dir, params, gridline_path=gridline_path)
        elif code == "FIX_TRACK_ROOT_NOTE":
            res = _fix_track_root_note(p_dir, params, gridline_path=gridline_path)
        elif code == "TIGHTEN_HARMONIC_CLAMP_OR_VOICING":
            res = _fix_mutate_block_density_or_lpf(p_dir, params, gridline_path=gridline_path)
        elif code == "ADJUST_TRACK_FADER":
            res = _fix_adjust_track_fader(p_dir, params, gridline_path=gridline_path)
        elif code in ("EQ_RESONANCE_NOTCH", "ADJUST_MASTER_GAIN"):
            if render_summary and not delegated_e4:
                gl_path = _find_gridline_project(p_dir, gridline_path)
                if gl_path:
                    muts = _apply_autofix_mutations(gl_path, render_summary, {"ceiling_drops": 0, "low_mids_applied": False})
                    delegated_e4 = True
                    res = {"applied": bool(muts), "code": code, "mutations": muts}
                else:
                    res = {"applied": False, "code": code, "reason": ".gridline not found"}
            else:
                res = {"applied": False, "code": code, "reason": "No E4 render_summary or already delegated"}
        else:
            # Creative / non-mechanical actions (SWAP_SAMPLE, etc.) escalate cleanly
            escalate_to_llm.append({"code": code, "params": params, "reason": "Requires LLM creative decision"})
            continue

        if res.get("applied"):
            applied.append(res)
        else:
            skipped.append(res)
            escalate_to_llm.append(res)

    if applied and os.path.isdir(p_dir):
        invalidate_cache(p_dir)

    return {
        "project_dir": p_dir,
        "applied_count": len(applied),
        "applied": applied,
        "skipped": skipped,
        "escalate_to_llm": escalate_to_llm,
    }
