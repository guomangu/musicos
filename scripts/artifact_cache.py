#!/usr/bin/env python3
"""
Incremental SHA-256 Artifact Cache (`artifact_cache.py`, Phase 6 §4.2).

Stores and queries `artifact_hashes` inside `<project>/pipeline_state.json` so that:
  - E1 Soundcheck skips unchanged tracks (`0 ms`, `0` Gridline renders).
  - E2 Vertical tests skip unchanged `V1..V4` layers (e.g., editing Lead only re-runs `V4_LEAD`).
  - E3 Horizontal seams skip seams whose adjacent sections and automations are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from typing import Any


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_json(obj: Any) -> str:
    encoded = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_fingerprint(path: str | None) -> dict[str, Any]:
    """Return lightweight deterministic fingerprint (`mtime_ns`, `size`, and quick SHA-256 for files <= 10 MB)."""
    if not path or not str(path).strip():
        return {"path": "", "exists": False}
    abs_p = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(abs_p):
        return {"path": abs_p, "exists": False}
    st = os.stat(abs_p)
    fp: dict[str, Any] = {
        "path": abs_p,
        "exists": True,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }
    if st.st_size <= 10 * 1024 * 1024:
        h = hashlib.sha256()
        with open(abs_p, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        fp["sha256"] = h.hexdigest()
    return fp


def _resolve_sample_path(sample_path: str, project_dir: str = "") -> str:
    if not sample_path:
        return ""
    if os.path.isabs(sample_path):
        return sample_path
    if project_dir:
        cand = os.path.join(os.path.abspath(project_dir), "samples", sample_path)
        if os.path.isfile(cand):
            return cand
        cand2 = os.path.join(os.path.abspath(project_dir), sample_path)
        if os.path.isfile(cand2):
            return cand2
    return sample_path


def compute_e1_track_hash(
    *,
    sample_path: str = "",
    plugin_params: dict[str, Any] | None = None,
    notes_slice: list[dict[str, Any]] | None = None,
    bpm: float = 132.0,
    track_spec: dict[str, Any] | None = None,
    wav_path: str = "",
    role: str = "",
    key: str = "",
    valence: float = 0.0,
    project_dir: str = "",
    gain_staging_overrides: dict[str, Any] | None = None,
    delay_send: float | None = None,
    hall_send: float | None = None,
    expressive: bool | None = None,
) -> str:
    """E1 cache key: `sha256(sample_path_mtime + plugin_params + notes_slice_2_4_bars + bpm)`."""
    eff_sample = _resolve_sample_path(sample_path, project_dir)
    spec_norm = dict(track_spec or {})
    if not eff_sample and spec_norm.get("sample_file"):
        eff_sample = _resolve_sample_path(str(spec_norm["sample_file"]), project_dir)

    payload = {
        "kind": "E1_TRACK",
        "sample": file_fingerprint(eff_sample),
        "wav": file_fingerprint(wav_path) if wav_path else None,
        "plugin_params": plugin_params or spec_norm.get("plugin_params") or {},
        "track_spec": {
            k: v
            for k, v in spec_norm.items()
            if k not in ("plugin_params",)
        },
        "notes_slice": notes_slice or [],
        "bpm": round(float(bpm), 4),
        "role": str(role or spec_norm.get("role") or ""),
        "key": str(key),
        "valence": round(float(valence), 4),
        "gain_staging_overrides": gain_staging_overrides or {},
        "delay_send": spec_norm.get("delay_send", delay_send),
        "hall_send": spec_norm.get("hall_send", hall_send),
        "expressive": spec_norm.get("expressive", expressive),
    }
    return sha256_json(payload)


def extract_gridline_hashes(project_path: str) -> dict[str, Any]:
    """
    Parse a `.gridline` JSON project directly (< 1 ms) and compute fine-grained
    SHA-256 hashes for channels, mixer, tempo/meter, patterns, and playlist.
    """
    abs_proj = os.path.abspath(os.path.expanduser(project_path))
    if not os.path.isfile(abs_proj):
        return {"exists": False, "file_hash": ""}
    try:
        with open(abs_proj, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        fp = file_fingerprint(abs_proj)
        return {"exists": True, "file_hash": fp.get("sha256", ""), "raw": True}

    channels = data.get("channels") or []
    mixer = data.get("mixer") or []
    patterns = data.get("patterns") or []
    playlist = data.get("playlist") or []

    # Include referenced sample file fingerprints so replacing a WAV invalidates channel hash
    sample_fps: list[dict[str, Any]] = []
    for ch in channels:
        pl = ch.get("plugin")
        if isinstance(pl, dict) and isinstance(pl.get("data"), str) and pl["data"].strip():
            sample_fps.append(file_fingerprint(pl["data"]))

    channels_hash = sha256_json({"channels": channels, "samples": sample_fps})
    mixer_hash = sha256_json(mixer)
    song_hash = sha256_json({
        "tempo": data.get("tempo"),
        "timeSigNum": data.get("timeSigNum"),
        "timeSigDen": data.get("timeSigDen"),
        "swing": data.get("swing"),
        "ppq": data.get("ppq"),
    })
    patterns_by_id = {p.get("id"): p for p in patterns if "id" in p}
    patterns_hash = sha256_json(patterns)
    playlist_hash = sha256_json(playlist)

    return {
        "exists": True,
        "channels_hash": channels_hash,
        "mixer_hash": mixer_hash,
        "song_hash": song_hash,
        "patterns_hash": patterns_hash,
        "playlist_hash": playlist_hash,
        "patterns_by_id": patterns_by_id,
        "data": data,
    }


def compute_e2_vertical_hash(
    stage: str,
    *,
    core_loop_channels_hash: str = "",
    mixer_params_hash: str = "",
    project_path: str = "",
    profile: str = "",
    range_bars: str = "",
    stage_map: dict[str, str] | None = None,
    blueprint: dict[str, Any] | None = None,
    contract: dict[str, Any] | None = None,
    wav_path: str = "",
    render_stems: bool = True,
    gain_staging_overrides: dict[str, Any] | None = None,
    expressive: bool | None = None,
) -> str:
    """
    E2 cache key: `sha256(core_loop_channels_hash + mixer_params_hash)` plus active layer metadata.
    Note: when only `V4_LEAD` changes inside `channels`, or better: when `project_path` is passed,
    we hash channels belonging to layers `<= stage` plus mixer/patterns so changes in `V4_LEAD`
    do NOT invalidate `V1_LOW_END..V3_HARMONY`!
    """
    st_norm = str(stage).strip().upper()
    ch_hash = core_loop_channels_hash
    mx_hash = mixer_params_hash
    pats_hash = ""
    song_hash = ""

    if project_path and os.path.isfile(os.path.expanduser(project_path)) and (not ch_hash or not mx_hash):
        ginfo = extract_gridline_hashes(project_path)
        data = ginfo.get("data")
        if isinstance(data, dict):
            # Scope channel & pattern hashes to layers <= st_norm so V4 edits keep V1..V3 cached!
            from vertical_tester import _STAGE_ORDER, classify_channel_stage

            rank = _STAGE_ORDER.get(st_norm, 3)
            custom_map = {k.lower(): v.upper() for k, v in (stage_map or {}).items()}
            role_hints = {
                str(tr.get("name") or "").strip().lower(): str(tr.get("role") or "").strip().lower()
                for tr in ((blueprint or {}).get("tracks") or [])
            }
            active_chans = []
            active_chan_ids: set[str] = set()
            active_sample_fps = []
            active_mixers = {0, 17, 18, 19, 20}
            for idx, ch in enumerate(data.get("channels") or [], start=1):
                nm = str(ch.get("name") or f"Channel_{idx}")
                ch_st = custom_map.get(nm.lower()) or classify_channel_stage(nm, role_hint=role_hints.get(nm.lower(), ""))
                if _STAGE_ORDER.get(ch_st, 2) <= rank:
                    active_chans.append(ch)
                    if "id" in ch:
                        active_chan_ids.add(str(ch["id"]))
                    if "mixerTrack" in ch:
                        active_mixers.add(int(ch["mixerTrack"]))
                    pl = ch.get("plugin")
                    if isinstance(pl, dict) and isinstance(pl.get("data"), str) and pl["data"].strip():
                        active_sample_fps.append(file_fingerprint(pl["data"]))

            scoped_pats = []
            for p in data.get("patterns") or []:
                notes_dict = p.get("notes") or {}
                if isinstance(notes_dict, dict):
                    scoped_notes = {cid: evs for cid, evs in notes_dict.items() if str(cid) in active_chan_ids}
                else:
                    scoped_notes = notes_dict
                scoped_pats.append({"id": p.get("id"), "bars": p.get("bars"), "notes": scoped_notes, "steps": p.get("steps")})

            mixer_list = data.get("mixer") or []
            scoped_mixers = [
                mixer_list[i] for i in sorted(active_mixers) if 0 <= i < len(mixer_list)
            ]
            ch_hash = ch_hash or sha256_json({"channels": active_chans, "samples": active_sample_fps})
            mx_hash = mx_hash or sha256_json(scoped_mixers)
            pats_hash = sha256_json({"patterns": scoped_pats, "playlist": data.get("playlist") or []})
            song_hash = ginfo.get("song_hash", "")
        else:
            ch_hash = ch_hash or ginfo.get("file_hash", "")
            mx_hash = mx_hash or ginfo.get("file_hash", "")

    payload = {
        "kind": "E2_VERTICAL",
        "stage": st_norm,
        "channels_hash": ch_hash,
        "mixer_hash": mx_hash,
        "patterns_hash": pats_hash,
        "song_hash": song_hash,
        "profile": str(profile),
        "range_bars": str(range_bars),
        "render_stems": bool(render_stems),
        "wav": file_fingerprint(wav_path) if wav_path else None,
        "gain_staging_overrides": gain_staging_overrides or {},
        "expressive": expressive,
        "contract_s1_s2": {
            "s1": (contract or {}).get("s1_physical"),
            "s2": (contract or {}).get("s2_acoustic"),
        } if contract else None,
    }
    return sha256_json(payload)


def compute_e3_seam_hash(
    seam_name: str,
    *,
    section_a_notes: Any = None,
    section_b_notes: Any = None,
    transitions_autos: Any = None,
    project_path: str = "",
    bar_range: str = "",
    bpm: float = 132.0,
    beats_per_bar: int = 4,
    wav_path: str = "",
) -> str:
    """E3 cache key: `sha256(section_A_notes + section_B_notes + transitions_autos)`."""
    sec_a = section_a_notes
    sec_b = section_b_notes
    autos = transitions_autos
    extra_hash = ""

    if project_path and os.path.isfile(os.path.expanduser(project_path)) and (sec_a is None and sec_b is None):
        ginfo = extract_gridline_hashes(project_path)
        data = ginfo.get("data")
        if isinstance(data, dict):
            ppq = int(data.get("ppq") or 96)
            bpb = int(data.get("timeSigNum") or beats_per_bar or 4)
            bar_ticks = ppq * bpb
            b0, b1 = 1, 9999
            if "-" in bar_range:
                try:
                    p0, p1 = bar_range.split("-", 1)
                    b0, b1 = int(p0), int(p1)
                except ValueError:
                    pass
            t_lo = (b0 - 1) * bar_ticks
            t_hi = b1 * bar_ticks
            pats = ginfo.get("patterns_by_id") or {}
            win_clips = []
            for clip in data.get("playlist") or []:
                st = int(clip.get("start", 0))
                pat_id = clip.get("pattern")
                pat = pats.get(pat_id) or {}
                p_len = int(pat.get("bars", 1)) * bar_ticks
                if st + p_len >= t_lo and st <= t_hi:
                    win_clips.append({"clip": clip, "pattern": pat})
            sec_a = win_clips
            autos = data.get("links") or []
            extra_hash = sha256_json({"channels": ginfo.get("channels_hash"), "mixer": ginfo.get("mixer_hash"), "song": ginfo.get("song_hash")})
        else:
            extra_hash = ginfo.get("file_hash", "")

    payload = {
        "kind": "E3_SEAM",
        "seam": str(seam_name),
        "bar_range": str(bar_range),
        "bpm": round(float(bpm), 4),
        "beats_per_bar": int(beats_per_bar),
        "section_a_notes": sec_a,
        "section_b_notes": sec_b,
        "transitions_autos": autos,
        "extra_hash": extra_hash,
        "wav": file_fingerprint(wav_path) if wav_path else None,
    }
    return sha256_json(payload)


def _state_path(project_dir: str) -> str:
    return os.path.join(os.path.abspath(os.path.expanduser(project_dir)), "pipeline_state.json")


def _load_state_raw(project_dir: str) -> dict[str, Any]:
    sp = _state_path(project_dir)
    if os.path.isfile(sp):
        try:
            with open(sp, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                data.setdefault("artifact_hashes", {})
                return data
        except Exception:
            pass
    return {
        "project_dir": os.path.abspath(os.path.expanduser(project_dir)),
        "stage": "E0_PALETTE",
        "artifact_hashes": {},
    }


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp_p = tempfile.mkstemp(prefix=".cache_", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_p, path)
    except Exception:
        if os.path.exists(tmp_p):
            os.remove(tmp_p)
        raise


def get_cached_verdict(
    project_dir: str,
    cache_key: str,
    expected_hash: str,
    *,
    verify_paths: list[str] | None = None,
) -> dict[str, Any] | None:
    """
    Lookup `cache_key` in `<project_dir>/pipeline_state.json` `artifact_hashes`.
    Returns stored result dict (with `cache_hit=True`) on exact SHA-256 match (< 2 ms).
    """
    if not project_dir or not str(project_dir).strip() or not expected_hash:
        return None
    sp = _state_path(project_dir)
    if not os.path.isfile(sp):
        return None
    try:
        with open(sp, encoding="utf-8") as fh:
            state = json.load(fh)
    except Exception:
        return None

    entry = (state.get("artifact_hashes") or {}).get(cache_key)
    if not isinstance(entry, dict) or entry.get("hash") != expected_hash:
        return None

    result = entry.get("result")
    if not isinstance(result, dict):
        return None

    for p in (verify_paths or []):
        if p and not os.path.isfile(p):
            return None
    # Also verify wav_path / diagnostics_path if recorded inside cached result
    for path_key in ("wav_path", "diagnostics_path"):
        pv = result.get(path_key)
        if isinstance(pv, str) and pv.strip() and not os.path.isfile(pv):
            return None

    out = dict(result)
    out["cache_hit"] = True
    out["cache_key"] = cache_key
    out["cache_hash"] = expected_hash
    return out


def set_cached_verdict(
    project_dir: str,
    cache_key: str,
    hash_val: str,
    result: dict[str, Any],
) -> None:
    """Persist `cache_key -> {hash, verdict, result}` inside `<project_dir>/pipeline_state.json`."""
    if not project_dir or not str(project_dir).strip() or not hash_val:
        return
    state = _load_state_raw(project_dir)
    hashes = state.setdefault("artifact_hashes", {})
    stored = dict(result)
    stored.pop("cache_hit", None)
    hashes[cache_key] = {
        "hash": hash_val,
        "verdict": stored.get("verdict"),
        "updated_at": round(time.time(), 3),
        "result": stored,
    }
    _atomic_write_json(_state_path(project_dir), state)


def invalidate_cache(project_dir: str, prefix: str = "") -> int:
    """Invalidate all or prefix-matching keys in `artifact_hashes`."""
    if not project_dir or not os.path.isfile(_state_path(project_dir)):
        return 0
    state = _load_state_raw(project_dir)
    hashes = state.get("artifact_hashes") or {}
    if not prefix:
        n = len(hashes)
        state["artifact_hashes"] = {}
    else:
        keys = [k for k in hashes if k.startswith(prefix)]
        for k in keys:
            del hashes[k]
        n = len(keys)
    _atomic_write_json(_state_path(project_dir), state)
    return n
