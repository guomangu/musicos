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
from gl_dsl import SYNTH_REGISTRY
from groove_lib import apply_micro_strum, ms_to_beats
from loop_first_runner import LoopFirstStateMachine, detect_user_samples_locked
from midi_autofix import autofix_midi
from render_and_audit import _apply_autofix_mutations
from sample_prep_lib import audit_samples_manifest, build_sample_from_spec
from studio_lib import GridlineMCP
from transitions_lib import filter_sweep, outro_fade


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
                    k, v = k.strip(), v.strip()
                    try:
                        params[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        params[k] = v
        elif re.match(r"^\d+-\d+$", arg_s):
            params["bars"] = arg_s
        else:
            try:
                params["value"] = float(arg_s)
            except ValueError:
                params["track"] = arg_s
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


def _fix_mutate_block_density_or_lpf(
    project_dir: str,
    params: dict[str, Any],
    *,
    gridline_path: str = "",
) -> dict[str, Any]:
    p_dir = os.path.abspath(os.path.expanduser(project_dir))
    gl_path = _find_gridline_project(p_dir, gridline_path)
    if not gl_path or not os.path.isfile(gl_path):
        return {"applied": False, "code": "MUTATE_BLOCK_DENSITY_OR_LPF", "reason": ".gridline not found"}

    with open(gl_path, encoding="utf-8") as fh:
        gdata = json.load(fh)
    bpb = float(gdata.get("timeSigNum") or 4)
    bars_str = str(params.get("bars") or "9-16")
    sb, eb = 9, 16
    if "-" in bars_str:
        try:
            a, b = bars_str.split("-", 1)
            sb, eb = int(a), int(b)
        except ValueError:
            pass
    elif params.get("start_bar") and params.get("end_bar"):
        sb, eb = int(params["start_bar"]), int(params["end_bar"])

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
    b_bar = max(3, int(params.get("bar", 9)))
    return _fix_mutate_block_density_or_lpf(
        project_dir,
        {"start_bar": b_bar - 2, "end_bar": b_bar},
        gridline_path=gridline_path,
    ) | {"code": "REINFORCE_SECTION_TRANSITION"}


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
        target_track = str(params.get("track") or "")
        if sm is not None and target_track:
            chk = sm.validate_edit_target(target_track, raise_on_forbidden=False)
            if not chk.get("allowed", True):
                skipped.append({"code": code, "reason": "forbidden_tracks guardrail", "track": target_track})
                escalate_to_llm.append({"code": code, "reason": "forbidden_tracks", "track": target_track})
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
        elif code == "MUTATE_BLOCK_DENSITY_OR_LPF":
            res = _fix_mutate_block_density_or_lpf(p_dir, params, gridline_path=gridline_path)
        elif code == "REINFORCE_SECTION_TRANSITION":
            res = _fix_reinforce_section_transition(p_dir, params, gridline_path=gridline_path)
        elif code == "APPLY_OUTRO_FADE":
            res = _fix_outro_fade(p_dir, params, gridline_path=gridline_path)
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
            # Creative / non-mechanical actions (SWAP_SAMPLE, PRUNE_TRACK_IN_SECTION, etc.) escalate cleanly
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
