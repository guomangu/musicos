#!/usr/bin/env python3
"""
Centralized AI-Studio song builder for Gridline 0.7.7.
Eliminates ~80% of boilerplate in per-song `build_song.py`:
  - Blueprint loading & JSON-Schema validation (`schemas/song_blueprint.schema.json`)
  - Deterministic pitch/scale helpers (`SCALES`, `degree_to_midi`)
  - `master_sketch.mid` export + blocking `verify_midi_gate()`
  - Declarative `ChannelSpec` (both `gl.sampler` and native `gl.*` synths)
  - Concise `gl.eq` builders (enforcing `Peak=6`, `Band N width`)
  - Atomic Gridline build with strict `1 sonorité = 1 canal + 1 insert + patterns dédiés + 1 piste Playlist`
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

import jsonschema
import pretty_midi

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYZER_DIR = os.path.join(ROOT, "mcp-servers", "audio-analyzer")
for _d in (SCRIPTS_DIR, ANALYZER_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from gl_dsl import (  # noqa: F401 (re-exported for song scripts)
    EQ_KIND_TO_INT,
    INT_TO_EQ_KIND,
    SYNTH_REGISTRY,
    SynthVoiceConfig,
    clamp,
    db_to_auto,
    eq_band as _dsl_eq_band,
    eq_params,
    limiter_master,
    mixer_gain_to_auto,
    q_to_width,
    reverb_room,
    sidechain_send,
    synth_voice,
    to_mixer_params,
)
from gain_staging_lib import compute_gain_staging_plan, normalize_role_for_gain_staging  # noqa: E402
from groove_lib import (  # noqa: F401 (re-exported for song scripts)
    apply_anti_machine_gun,
    apply_swing,
    apply_haas_pocket,
    apply_micro_strum,
    enforce_frequency_inertia,
    ms_to_beats,
    sanitize_note_overlaps,
)
from score_ir import (  # v3.1 IR-02 — ScoreIR + clamp de frontière harmonique
    apply_groove_to_score_ir,
    build_score_ir,
    clamp_flat_tracks_at_boundaries,
    compile_score_ir_to_notes,
    classify_role,
    harmonic_grid_from_core,
    segment_index_for_beat,
)
from musicality_metrics import adapt_voicing_to_timbre, optimize_voice_leading  # noqa: F401 (re-exported for song scripts)
from blueprint_safety import summarize_musical_safety  # noqa: E402
from phrasing_engine import apply_instrumental_prosody  # noqa: F401 (re-exported for song scripts)
from studio_lib import (
    TEMPLATE_EMPTY_MIXER,
    GridlineMCP,
    check_pattern_audibility,
    setup_standard_buses,
    verify_midi_gate,
)
from transitions_lib import (  # noqa: F401 (re-exported for song scripts)
    add_transition_fx,
    apply_harmonic_anticipation,
    beat4_dropout,
    filter_sweep,
    generate_drum_fill,
    outro_fade,
    section_gain_arc,
)

SCHEMA_PATH = os.path.join(ROOT, "schemas", "song_blueprint.schema.json")
ARRANGEMENT_PLAN_SCHEMA_PATH = os.path.join(ROOT, "schemas", "arrangement_plan.schema.json")

SCALES: dict[str, list[int]] = {
    "ionian": [0, 2, 4, 5, 7, 9, 11],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "phrygian_dominant": [0, 1, 4, 5, 7, 8, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "blues": [0, 3, 5, 6, 7, 10],
}

BUS_MAP = {"drums": 17, "bass": 18, "music": 19, "fx": 20}


def degree_to_midi(tonic_midi: int, mode: str, degree_0idx: int, oct_shift: int = 0) -> int:
    """Deterministically convert 0-indexed scale degree to MIDI note number."""
    sc = SCALES[mode]
    oct_add, idx = divmod(degree_0idx, len(sc))
    return int(tonic_midi + 12 * (oct_add + oct_shift) + sc[idx])


def load_blueprint(blueprint_path: str, validate: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load `song_blueprint.json`, optionally validate schema, and extract normalized metadata."""
    with open(blueprint_path, encoding="utf-8") as fh:
        bp = json.load(fh)
    if validate and os.path.isfile(SCHEMA_PATH):
        with open(SCHEMA_PATH, encoding="utf-8") as sfh:
            schema = json.load(sfh)
        jsonschema.validate(bp, schema)

    ev = bp.get("emotion_vector") or {}
    hc = bp.get("harmonic_core") or {}
    profile = bp.get("mix_profile", "trap_rap")
    meta = {
        "bpm": float(bp["bpm"]),
        "beats_per_bar": int(bp["time_signature"]["numerator"]),
        "denominator": int(bp["time_signature"]["denominator"]),
        "key": bp["key"],
        "mix_profile": profile,
        "has_drums": bool(bp["has_drums"]) if "has_drums" in bp else (profile != "classical_ambient"),
        "phrase_plan": str(hc.get("phrase_plan", "")),
        "valence": float(ev.get("valence", 0.0)),
        "tension": float(ev.get("tension", 0.3)),
        "energy": float(ev.get("energy", 0.7)),
        "depth": float(ev.get("depth", 0.5)),
        "swing": float(bp.get("swing", 0.0) or 0.0),
        "total_bars": sum(int(s["bars"]) for s in bp["structure"]),
    }
    _cat = os.path.join(os.path.dirname(os.path.abspath(blueprint_path)), "samples_catalog.json")
    meta["musical_safety"] = summarize_musical_safety(bp, catalog_path=_cat if os.path.isfile(_cat) else None)
    return bp, meta


@dataclass
class ChannelSpec:
    """Declarative channel specification for `gl.sampler` or native `gl.*` synths."""

    name: str
    track_key: str
    mixer: int
    bus: str | int = "music"  # "drums" (17), "bass" (18), "music" (19), or explicit bus index
    plugin: str = "gl.sampler"
    volume: float = 0.78
    pan: float = 0.0
    # Sampler fields
    sample_file: str = ""
    root_note: int = 60
    fine_tune: int | None = 0  # None = auto-read recommended_gl_fine_tune_ct from manifest.json
    one_shot: int = 0
    voices: int = 4
    attack: float = 0.002
    release: float = 0.080
    declick: int = 2
    remove_dc: int = 1
    lpf_hz: float | None = None
    resonance: float = 0.08
    # Native gl.* extra params & semantic synth_voice support
    plugin_params: dict[str, Any] = field(default_factory=dict)
    role: str | None = None
    valence: float = 0.0
    cutoff_hz: float | None = None
    voice: SynthVoiceConfig | None = None
    # Routing & arrangement behavior
    fx_send: float = 0.0
    delay_send: float = 0.0
    hall_send: float = 0.0
    sidechain_buses: list[tuple[int, float]] = field(default_factory=list)
    single_shot_clips: bool = False  # True for isolated FX triggers (creates 1-bar clip per event)
    glide_ms: float | None = None  # v3.2 : glide/portamento 808 (notes `glide_to_next`), 35-90 ms


# --- Semantic + Legacy gl.eq builders (backed by gl_dsl.py) ---

def eq_band(
    first: int,
    second: int | str,
    third: int | float,
    fourth: float = 0.0,
    fifth: float | None = None,
    width: float | None = None,
    slot: int = 1,
) -> dict[str, Any] | list[tuple[str, Any]]:
    """
    Dual-mode `gl.eq` band builder backed by `gl_dsl`:
      - Semantic (new): `eq_band(band_idx, kind_str, freq_hz, gain_db=0.0, q=1.4)` -> dict
      - Legacy (compat): `eq_band(mixer, band, btype_int, freq, gain=0.0, width=None, slot=1)` -> list[(path, val)]
    """
    if isinstance(second, str):
        q_val = 1.4 if fifth is None else float(fifth)
        return _dsl_eq_band(int(first), second, float(third), float(fourth), q=q_val)  # type: ignore[arg-type]

    mixer = int(first)
    band = int(second)
    btype = int(third)
    freq = float(fourth)
    gain = 0.0 if fifth is None else float(fifth)
    kind = INT_TO_EQ_KIND.get(btype)
    if kind is None:
        raise ValueError(f"Invalid gl.eq Band type int: {btype}")
    band_dict = _dsl_eq_band(band, kind, freq, gain_db=gain)
    if btype in (2, 6):
        if width is not None:
            band_dict[f"Band {band} width"] = float(width)
        else:
            band_dict.pop(f"Band {band} width", None)
    return to_mixer_params(mixer, slot, band_dict)


def eq_insert(mixer: int, bands: list[tuple[Any, ...] | dict[str, Any]], slot: int = 1) -> list[tuple[str, Any]]:
    """
    Expand a list of compact band tuples OR `gl_dsl.eq_band(...)` dicts into `gl.eq` parameters:
      - dict from `gl_dsl.eq_band(...)`
      - (band_idx, type_int_or_kind, freq_hz)
      - (band_idx, type_int_or_kind, freq_hz, gain_db)
      - (band_idx, type_int_or_kind, freq_hz, gain_db, width)
    """
    out: list[tuple[str, Any]] = []
    for item in bands:
        if isinstance(item, dict):
            out.extend(to_mixer_params(mixer, slot, item))
            continue
        b = int(item[0])
        bt_raw = item[1]
        fr = float(item[2])
        gn = float(item[3]) if len(item) > 3 else 0.0
        wd = float(item[4]) if len(item) > 4 else None
        if isinstance(bt_raw, str):
            d = _dsl_eq_band(b, bt_raw, fr, gain_db=gn)  # type: ignore[arg-type]
            if wd is not None and f"Band {b} width" in d:
                d[f"Band {b} width"] = wd
            out.extend(to_mixer_params(mixer, slot, d))
        else:
            out.extend(eq_band(mixer, b, int(bt_raw), fr, gn, width=wd, slot=slot))  # type: ignore[arg-type]
    return out


def eq_music_pocket(
    mixer: int,
    hpf_hz: float = 175.0,
    mud_hz: float = 300.0,
    mud_db: float = -3.0,
    box_hz: float = 480.0,
    box_db: float = -2.8,
    pocket_hz: float = 1600.0,
    pocket_db: float = -3.2,
) -> list[tuple[str, Any]]:
    """Standard anti-mud + Lead/Vocal mirror-pocket EQ for harmonic/pad/loop inserts."""
    return eq_insert(mixer, [
        (1, "hpf", hpf_hz),
        (2, "peak", mud_hz, mud_db, 0.45),
        (3, "peak", box_hz, box_db, 0.40),
        (4, "peak", pocket_hz, pocket_db, 0.42),
    ])


def eq_kick_clean(mixer: int, hpf_hz: float = 42.0, sub_cut_db: float = -9.5, punch_db: float = 3.0) -> list[tuple[str, Any]]:
    """Standard Trap/Rap kick EQ (removes sub rumble, trims 50 Hz boom, keeps 95 Hz punch)."""
    return eq_insert(mixer, [
        (1, "hpf", hpf_hz),
        (2, "peak", 50.0, sub_cut_db, 0.40),
        (3, "peak", 95.0, punch_db, 0.45),
        (4, "peak", 210.0, -4.0, 0.38),
    ])


def export_master_sketch(
    tracks: dict[str, list[dict[str, Any]]],
    sketch_path: str,
    bpm: float,
    bass_keys: list[str],
    chords_key: str = "sketch_chords",
    lead_key: str = "lead_bell",
    numerator: int = 4,
    denominator: int = 4,
) -> str:
    """Export 3-track `master_sketch.mid` (`Bass`, `Chords`, `Lead`) from 0-indexed beat events."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    if int(numerator) != 4 or int(denominator) != 4:
        pm.time_signature_changes.append(pretty_midi.TimeSignature(int(numerator), int(denominator), 0.0))
    b2s = 60.0 / float(bpm)

    bass_all: list[dict[str, Any]] = []
    for bk in bass_keys:
        bass_all.extend(tracks.get(bk, []))
    chords_all = tracks.get(chords_key, [])
    lead_all = tracks.get(lead_key, [])

    for name, evs in [("Bass", bass_all), ("Chords", chords_all), ("Lead", lead_all)]:
        if not evs:
            continue
        inst = pretty_midi.Instrument(program=0, name=name)
        for ev in sorted(evs, key=lambda x: x["start"]):
            st = float(ev["start"]) * b2s
            en = (float(ev["start"]) + float(ev.get("length", ev.get("duration", 0.25)))) * b2s
            inst.notes.append(pretty_midi.Note(int(ev["velocity"]), int(ev["pitch"]), st, en))
        pm.instruments.append(inst)

    os.makedirs(os.path.dirname(os.path.abspath(sketch_path)), exist_ok=True)
    pm.write(sketch_path)
    return sketch_path


def _resolve_bus(bus: str | int) -> int:
    if isinstance(bus, int):
        return bus
    return BUS_MAP[str(bus).lower()]


def _load_soundcheck_metrics(proj_dir: str) -> dict[str, dict[str, Any]]:
    """Load É1 soundcheck measurements from `renders/diagnostics/soundcheck_metrics.json` or `soundcheck_*.json`."""
    diag_dir = os.path.join(proj_dir, "renders", "diagnostics")
    out: dict[str, dict[str, Any]] = {}
    agg_path = os.path.join(diag_dir, "soundcheck_metrics.json")
    if os.path.isfile(agg_path):
        try:
            with open(agg_path, encoding="utf-8") as fh:
                raw = json.load(fh) or {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, dict):
                        out[str(k).lower()] = v
                        if v.get("slug"):
                            out[str(v["slug"]).lower()] = v
        except Exception:
            pass
    if os.path.isdir(diag_dir):
        for fn in sorted(os.listdir(diag_dir)):
            if fn.startswith("soundcheck_") and fn.endswith(".json") and fn != "soundcheck_metrics.json":
                try:
                    with open(os.path.join(diag_dir, fn), encoding="utf-8") as fh:
                        obj = json.load(fh) or {}
                    res = obj.get("result") or {}
                    raw_m = ((obj.get("raw_dsp") or {}).get("metrics") or {})
                    tname = str(res.get("track") or "").strip()
                    if tname and ("active_lufs" in res or "active_lufs" in raw_m):
                        entry = {
                            "track": tname,
                            "role": res.get("role") or (obj.get("raw_dsp") or {}).get("role") or "",
                            "active_rms_db": res.get("active_rms_db", raw_m.get("active_rms_db")),
                            "active_lufs": res.get("active_lufs", raw_m.get("active_lufs")),
                            "peak_dbfs": raw_m.get("peak_dbfs", (res.get("metrics_summary") or {}).get("peak_dbfs")),
                            "rendered_volume": raw_m.get("rendered_volume", 0.80),
                            "recommended_volume_auto": res.get("recommended_volume_auto"),
                            "recommended_gain_db": res.get("recommended_gain_db"),
                        }
                        out.setdefault(tname.lower(), entry)
                except Exception:
                    continue
    return out


def _apply_auto_gain_staging(
    proj_dir: str,
    channels: list[ChannelSpec],
    meta: dict[str, Any],
    *,
    auto_gain_stage: bool = True,
) -> tuple[list[ChannelSpec], dict[int, float], dict[str, Any]]:
    """
    Apply Q1 active-loudness gain staging from É1 soundcheck metrics + Q2 persisted
    `gain_staging_overrides.json` trims, and write `renders/gain_staging_report.json`.
    """
    eff_channels = [copy.copy(ch) for ch in channels]
    insert_vol_overrides: dict[int, float] = {}
    sc_metrics = _load_soundcheck_metrics(proj_dir) if auto_gain_stage else {}

    ov_path = os.path.join(proj_dir, "gain_staging_overrides.json")
    overrides: dict[str, Any] = {}
    if os.path.isfile(ov_path):
        try:
            with open(ov_path, encoding="utf-8") as fh:
                overrides = json.load(fh) or {}
        except Exception:
            overrides = {}

    meas_for_plan: dict[str, dict[str, Any]] = {}
    ch_by_key: dict[str, ChannelSpec] = {}
    for ch in eff_channels:
        cands = [
            ch.track_key.lower(),
            ch.name.lower(),
            ch.name.lower().replace(" ", "_"),
        ]
        hit = next((sc_metrics[c] for c in cands if c in sc_metrics), None)
        if hit and hit.get("active_lufs") is not None:
            role_hint = normalize_role_for_gain_staging(f"{ch.name} {ch.role or ''} {ch.track_key} {hit.get('role', '')}")
            meas_for_plan[ch.track_key] = {
                "role": role_hint,
                "active_lufs": float(hit["active_lufs"]),
                "peak_dbfs": float(hit.get("peak_dbfs") if hit.get("peak_dbfs") is not None else -6.0),
                "current_volume": float(hit.get("rendered_volume") or ch.volume or 0.80),
                "is_effectively_silent": bool(hit.get("is_effectively_silent", False)),
            }
            ch_by_key[ch.track_key] = ch

    plan: dict[str, Any] = {}
    if auto_gain_stage and not meas_for_plan:
        # No measured soundcheck metrics (cache hit / fast session audition): fall back to
        # NOMINAL per-role balance so tracks are still calibrated relative to the anchor role,
        # instead of keeping arbitrary default faders (uneven track volumes).
        from gain_staging_lib import apply_nominal_balance

        eff_channels = apply_nominal_balance(eff_channels, meta)
        plan = {"nominal_balance": True}
    if auto_gain_stage and meas_for_plan:
        plan = compute_gain_staging_plan(
            meas_for_plan,
            mix_profile=str(meta.get("mix_profile", "trap_rap")),
            has_drums=bool(meta.get("has_drums", True)),
        )

    report_tracks: list[dict[str, Any]] = []
    for ch in eff_channels:
        vol_before = round(float(ch.volume), 4)
        plan_entry = (plan.get("tracks") or {}).get(ch.track_key)
        status = "UNCHANGED"
        delta_db = 0.0
        insert_vol = None

        if plan_entry is not None:
            ch.volume = float(plan_entry["recommended_volume"])
            delta_db = float(plan_entry["recommended_gain_db"])
            res_db = float(plan_entry.get("residual_gain_db") or 0.0)
            # Soundcheck renders at insert volume 1.0, so match 1.0 reference on calibrated channels
            insert_vol = round(clamp(1.0 * (10.0 ** (res_db / 40.0)), 0.15, 1.25), 4)
            insert_vol_overrides[ch.mixer] = insert_vol
            status = str(plan_entry["status"])

        # Apply persisted Q2 ADJUST_TRACK_FADER overrides (`volume_trim_db`) if present
        ov_entry = None
        for cand in (ch.track_key, ch.name, ch.name.lower(), ch.track_key.lower(), str(ch.mixer)):
            if cand in overrides and isinstance(overrides[cand], dict):
                ov_entry = overrides[cand]
                break
        if ov_entry and ov_entry.get("volume_trim_db") is not None:
            trim_db = float(clamp(float(ov_entry["volume_trim_db"]), -8.0, 8.0))
            if abs(trim_db) > 0.01:
                raw_fader = db_to_auto(trim_db, ref_fader=float(ch.volume)) * 1.25
                new_ch_vol = round(clamp(raw_fader, 0.10, 1.15), 4)
                applied_ch_db = 40.0 * math.log10(max(new_ch_vol, 1e-6) / max(float(ch.volume), 1e-6))
                ch.volume = new_ch_vol
                rem_db = trim_db - applied_ch_db
                if abs(rem_db) > 0.1:
                    base_ins = insert_vol_overrides.get(ch.mixer, 0.80)
                    insert_vol = round(clamp(base_ins * (10.0 ** (rem_db / 40.0)), 0.15, 1.25), 4)
                    insert_vol_overrides[ch.mixer] = insert_vol
                delta_db = round(delta_db + trim_db, 2)
                status = "OVERRIDE_APPLIED"

        report_tracks.append({
            "name": ch.name,
            "track_key": ch.track_key,
            "mixer": ch.mixer,
            "role": normalize_role_for_gain_staging(f"{ch.name} {ch.role or ''} {ch.track_key}"),
            "volume_before": vol_before,
            "volume_after": round(float(ch.volume), 4),
            "insert_volume": insert_vol,
            "delta_db": round(delta_db, 2),
            "status": status,
        })

    report = {
        "auto_gain_stage": bool(auto_gain_stage),
        "mix_profile": str(meta.get("mix_profile", "trap_rap")),
        "has_drums": bool(meta.get("has_drums", True)),
        "soundcheck_matched_count": len(meas_for_plan),
        "tracks": report_tracks,
        "plan": plan,
    }
    renders_dir = os.path.join(proj_dir, "renders")
    os.makedirs(renders_dir, exist_ok=True)
    report_path = os.path.join(renders_dir, "gain_staging_report.json")
    tmp_rp = report_path + ".tmp"
    with open(tmp_rp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    os.replace(tmp_rp, report_path)
    return eff_channels, insert_vol_overrides, report


def build_gridline_project(
    proj_path: str,
    export_dir: str,
    samples_dir: str,
    meta: dict[str, Any],
    sections: list[tuple[str, int, int, str]],
    channels: list[ChannelSpec],
    tracks: dict[str, list[dict[str, Any]]],
    markers: list[dict[str, Any]],
    insert_eq_params: list[tuple[str, Any]] | None = None,
    insert_eq_mixers: list[int] | None = None,
    bus_eq_params: list[tuple[str, Any]] | None = None,
    extra_plugins: list[tuple[int, str]] | None = None,
    automations: list[dict[str, Any]] | None = None,
    limiter_ceiling_db: float | None = None,
    limiter_gain_db: float | None = None,
    fx_room_vol: float = 0.32,
    min_pattern_lufs: float = -32.0,
    auto_gain_stage: bool = True,
) -> None:
    """
    Build a complete `.gridline` project in one atomic MCP session enforcing strict isolation
    (`1 sonorité = 1 canal + 1 insert mixeur + patterns dédiés + 1 piste Playlist nommée`).
    """
    abs_proj_path = os.path.abspath(proj_path)
    proj_dir = os.path.dirname(abs_proj_path)
    os.makedirs(proj_dir, exist_ok=True)
    shutil.copyfile(TEMPLATE_EMPTY_MIXER, proj_path)

    channels, insert_vol_overrides, _gs_report = _apply_auto_gain_staging(
        proj_dir, channels, meta, auto_gain_stage=auto_gain_stage
    )
    # Q3.3 makeup compensation is applied by the Q1 auto-gain stage (soundcheck metrics present),
    # so hand-tuned reference builds without É1 metrics keep their calibrated faders.
    apply_makeup = bool(((_gs_report or {}).get("soundcheck_matched_count") or 0) > 0)

    bpm = float(meta["bpm"])
    beats_per_bar = int(meta["beats_per_bar"])
    denom = int(meta.get("denominator", 4))
    profile = str(meta.get("mix_profile", "trap_rap"))
    valence = float(meta.get("valence", 0.0))

    mcp = GridlineMCP(proj_path, export_dir)
    try:
        mcp.call("set_song", {"tempo": bpm, "numerator": beats_per_bar, "denominator": denom})
        setup_standard_buses(
            mcp,
            bpm=bpm,
            profile=profile,
            valence=valence,
            limiter_ceiling_db=limiter_ceiling_db,
            limiter_gain_db=limiter_gain_db,
            fx_room_vol=fx_room_vol,
            apply_dark_mid_dip=not all(
                (ch.plugin != "gl.sampler" and (bool(ch.role) or ch.lpf_hz is not None or ch.cutoff_hz is not None))
                for ch in channels
                if any(k in ch.track_key.lower() for k in ("chord", "pad", "lead", "melody", "counter"))
            ),
        )

        # Ensure any mixer tracks > 22 (FX Delay/Hall buses) exist before adding channels
        max_mixer_needed = max(ch.mixer for ch in channels)
        if max_mixer_needed > 22:
            mixer_by_idx = {ch.mixer: ch.name for ch in channels}
            add_extras = [
                {"tool": "add_mixer_track", "arguments": {"name": mixer_by_idx.get(idx, f"Insert {idx}")}}
                for idx in range(23, max_mixer_needed + 1)
            ]
            mcp.batch(add_extras)

        # Pre-compute synth voices so LPF makeup gain can trim the channel fader deterministically
        voice_by_idx: dict[int, SynthVoiceConfig] = {}
        for idx, ch in enumerate(channels, start=1):
            if ch.voice is not None:
                voice_by_idx[idx] = ch.voice
            elif ch.plugin != "gl.sampler" and ch.role and ch.plugin in SYNTH_REGISTRY:
                voice_by_idx[idx] = synth_voice(
                    ch.plugin,
                    ch.role,  # type: ignore[arg-type]
                    valence=ch.valence if ch.valence != 0.0 else valence,
                    cutoff_hz=ch.lpf_hz if ch.lpf_hz is not None else ch.cutoff_hz,
                )
        eff_volumes: dict[int, float] = {}
        for idx, ch in enumerate(channels, start=1):
            v = float(ch.volume)
            mk = float(voice_by_idx[idx].makeup_gain_db) if (idx in voice_by_idx and apply_makeup) else 0.0
            if mk > 0.0:
                v = clamp(db_to_auto(mk, ref_fader=v) * 1.25, 0.10, 1.15)
            eff_volumes[idx] = round(v, 4)

        # Add channels (1 sonorité = 1 canal + 1 insert dédié)
        mcp.batch([
            {
                "tool": "add_channel",
                "arguments": {
                    "name": ch.name,
                    "plugin": ch.plugin,
                    "mixer": ch.mixer,
                    "volume": eff_volumes[idx],
                    "pan": ch.pan,
                },
            }
            for idx, ch in enumerate(channels, start=1)
        ])

        # Load optional manifest.json for automatic fine-tune lookup when fine_tune is None
        manifest_data: dict[str, Any] = {}
        manifest_path = os.path.join(samples_dir, "manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as mfh:
                manifest_data = json.load(mfh)

        patches: list[dict[str, Any]] = []
        param_changes: list[tuple[str, Any]] = []
        auto_insert_lpf_params: list[tuple[str, Any]] = []
        auto_insert_eq_mixers: list[int] = []
        for idx, ch in enumerate(channels, start=1):
            voice_cfg: SynthVoiceConfig | None = voice_by_idx.get(idx)

            if ch.plugin == "gl.sampler":
                spath = ch.sample_file if os.path.isabs(ch.sample_file) else os.path.join(samples_dir, ch.sample_file)
                ft_val = ch.fine_tune
                if ft_val is None:
                    bname = os.path.basename(spath)
                    ft_val = int((manifest_data.get(bname) or {}).get("recommended_gl_fine_tune_ct", 0))
                patches.append({"path": f"channel/{idx}", "set": {"plugin": {"data": spath}}})
                param_changes.extend([
                    (f"channel/{idx}/Root note", ch.root_note),
                    (f"channel/{idx}/Fine tune", int(ft_val)),
                    (f"channel/{idx}/One shot", ch.one_shot),
                    (f"channel/{idx}/Voices", ch.voices),
                    (f"channel/{idx}/Attack", ch.attack),
                    (f"channel/{idx}/Release", ch.release),
                    (f"channel/{idx}/Declick", ch.declick),
                    (f"channel/{idx}/Remove DC", ch.remove_dc),
                ])
                if ch.lpf_hz is not None:
                    param_changes.extend([
                        (f"channel/{idx}/Filter", 1),
                        (f"channel/{idx}/Cutoff", ch.lpf_hz),
                        (f"channel/{idx}/Resonance", ch.resonance),
                    ])
                if getattr(ch, "glide_ms", None):
                    # v3.2 : portamento 808 vérifié (Glide en secondes, clamp [0.035, 0.090])
                    param_changes.append((f"channel/{idx}/Glide", round(min(max(float(ch.glide_ms) / 1000.0, 0.035), 0.090), 4)))
            elif voice_cfg is not None:
                param_changes.extend(voice_cfg.to_channel_tuples(idx))
                if voice_cfg.needs_insert_lpf:
                    auto_insert_eq_mixers.append(ch.mixer)
                    auto_insert_lpf_params.extend(voice_cfg.to_insert_eq_tuples(ch.mixer, slot=1))

            for pname, pval in ch.plugin_params.items():
                param_changes.append((f"channel/{idx}/{pname}", pval))

        if patches:
            mcp.call("patch", {"patches": patches})
        if param_changes:
            mcp.set_params(param_changes)

        # Route each insert to its bus (17=Drums, 18=Bass, 19=Music)
        mcp.batch([
            {
                "tool": "edit_mixer_track",
                "arguments": {
                    "mixer": ch.mixer,
                    "name": ch.name,
                    "routeTo": _resolve_bus(ch.bus),
                    **({"volume": insert_vol_overrides[ch.mixer]} if ch.mixer in insert_vol_overrides else {}),
                },
            }
            for ch in channels
        ])

        # Add insert gl.eq plugins & apply all insert + bus EQ parameters
        eq_mixers = list(insert_eq_mixers or [])
        if not eq_mixers and insert_eq_params:
            seen_m: set[int] = set()
            for p_str, _ in insert_eq_params:
                parts = p_str.split("/")
                if len(parts) >= 2 and parts[0] == "mixer":
                    m_id = int(parts[1])
                    if m_id not in (0, 17, 18, 19, 20, 21, 22) and m_id not in seen_m:
                        seen_m.add(m_id)
                        eq_mixers.append(m_id)

        existing_eq_set = set(eq_mixers)
        existing_band7_mixers = {
            int(p_str.split("/")[1])
            for p_str, _ in (insert_eq_params or [])
            if p_str.startswith("mixer/") and "Band 7 type" in p_str
        }
        filtered_auto_lpf: list[tuple[str, Any]] = []
        for m_auto in auto_insert_eq_mixers:
            if m_auto not in existing_eq_set:
                eq_mixers.append(m_auto)
                existing_eq_set.add(m_auto)
        for p_str, val in auto_insert_lpf_params:
            m_id = int(p_str.split("/")[1])
            if m_id in existing_band7_mixers and "Band 7" in p_str:
                continue
            filtered_auto_lpf.append((p_str, val))

        if eq_mixers:
            mcp.batch([{"tool": "add_plugin", "arguments": {"mixer": m, "plugin": "gl.eq"}} for m in eq_mixers])
        if extra_plugins:
            mcp.batch([{"tool": "add_plugin", "arguments": {"mixer": int(m), "plugin": str(pl)}} for m, pl in extra_plugins])

        all_eq = list(insert_eq_params or []) + filtered_auto_lpf + list(bus_eq_params or [])
        if all_eq:
            mcp.set_params(all_eq)

        # Sends (Sidechain + FX Room)
        sends: list[dict[str, Any]] = []
        for ch in channels:
            for sc_bus, sc_lvl in ch.sidechain_buses:
                sends.append({
                    "tool": "set_send",
                    "arguments": sidechain_send(ch.mixer, sc_bus, sc_lvl),
                })
            if ch.fx_send > 0.0:
                sends.append({"tool": "set_send", "arguments": {"mixer": ch.mixer, "to": 20, "level": ch.fx_send}})
            if ch.delay_send > 0.0:
                sends.append({"tool": "set_send", "arguments": {"mixer": ch.mixer, "to": 21, "level": min(0.18, float(ch.delay_send))}})
            if ch.hall_send > 0.0:
                sends.append({"tool": "set_send", "arguments": {"mixer": ch.mixer, "to": 22, "level": min(0.30, float(ch.hall_send))}})
        if sends:
            mcp.batch(sends)

        # Name Playlist tracks cleanly
        track_patches = [
            {"path": f"track/{idx}", "set": {"name": ch.name}}
            for idx, ch in enumerate(channels, start=1)
        ]
        auto_list = list(automations or [])
        for a in auto_list:
            track_patches.append({"path": f"track/{int(a['track'])}", "set": {"name": f"Auto {a['name']}"}})
        mcp.call("patch", {"patches": track_patches})

        # Create strictly 1 dedicated Pattern per (channel, section) and add Playlist clips
        pat_idx = 0
        clip_calls: list[dict[str, Any]] = []
        for sec_name, start_bar, bars, _kind in sections:
            sec_start = float(start_bar * beats_per_bar)
            sec_end = float((start_bar + bars) * beats_per_bar)

            for ch_idx, ch in enumerate(channels, start=1):
                evs = [ev for ev in tracks.get(ch.track_key, []) if sec_start - 1e-6 <= ev["start"] < sec_end - 1e-6]
                if not evs:
                    continue

                if ch.single_shot_clips:
                    for fx_i, ev in enumerate(evs):
                        pat_idx += 1
                        pat_name = f"{ch.name} - {sec_name} #{fx_i + 1}"
                        if pat_idx == 1:
                            mcp.call("edit_pattern", {"pattern": 1, "name": pat_name, "bars": 1})
                        else:
                            mcp.call("add_pattern", {"name": pat_name, "bars": 1})
                        mcp.call("set_notes", {
                            "pattern": pat_idx,
                            "channel": ch_idx,
                            "notes": [{"start": 0.0, "length": ev["length"], "pitch": int(ev["pitch"]), "velocity": int(ev["velocity"])}],
                        })
                        clip_calls.append({
                            "tool": "add_clip",
                            "arguments": {"pattern": pat_idx, "track": ch_idx, "start": ev["start"], "length": float(beats_per_bar)},
                        })
                    continue

                pat_idx += 1
                pat_name = f"{ch.name} - {sec_name}"
                if pat_idx == 1:
                    mcp.call("edit_pattern", {"pattern": 1, "name": pat_name, "bars": bars})
                else:
                    mcp.call("add_pattern", {"name": pat_name, "bars": bars})

                notes_payload = [
                    {
                        "start": round(float(ev["start"]) - sec_start, 4),
                        "length": float(ev["length"]),
                        "pitch": int(ev["pitch"]),
                        "velocity": int(ev["velocity"]),
                    }
                    for ev in evs
                ]
                mcp.call("set_notes", {"pattern": pat_idx, "channel": ch_idx, "notes": notes_payload})
                clip_calls.append({
                    "tool": "add_clip",
                    "arguments": {"pattern": pat_idx, "track": ch_idx, "start": sec_start, "length": float(bars * beats_per_bar)},
                })

        if clip_calls:
            mcp.batch(clip_calls)
        if markers:
            mcp.batch([{"tool": "add_marker", "arguments": m} for m in markers])

        for a in auto_list:
            mcp.call("add_automation", a)

        issues = check_pattern_audibility(mcp, min_lufs=min_pattern_lufs)
        if issues:
            raise RuntimeError(f"Silent/weak patterns detected: {issues}")

        mcp.call("save_project", {})
    finally:
        mcp.close()


def select_core_section(
    sections: list[tuple[str, int, int, str]],
    tracks: dict[str, list[dict[str, Any]]] | None = None,
    beats_per_bar: int = 4,
    max_bars: int = 8,
) -> tuple[str, int, int, str]:
    """
    Pick the densest Refrain / Drop / Chorus / Climax section (clamped to 4 or 8 bars)
    for Stage 2 incremental vertical testing (`--core-only`).
    Note: `start_bar` in `sections` is 0-indexed (matching `build_gridline_project`).
    """
    if not sections:
        raise ValueError("sections cannot be empty")

    priority_kw = ("drop", "refrain", "chorus", "climax", "hook", "peak")
    best_sec = sections[0]
    best_rank = -1.0

    for sec in sections:
        sec_name, start_bar, bars, kind = sec
        hay = f"{sec_name} {kind}".lower()
        score = 2.0 if any(k in hay for k in priority_kw) else 0.0
        if tracks:
            s0 = float(start_bar * beats_per_bar)
            s1 = float((start_bar + bars) * beats_per_bar)
            active_tr = sum(
                1 for evs in tracks.values()
                if any(s0 - 1e-6 <= float(e["start"]) < s1 - 1e-6 for e in evs)
            )
            score += float(active_tr)
        if score > best_rank:
            best_rank = score
            best_sec = sec

    name, start_bar, bars, kind = best_sec
    eff_bars = min(int(bars), int(max_bars))
    return (name, int(start_bar), max(1, eff_bars), kind)


def build_core_loop_project(
    proj_path: str,
    export_dir: str,
    samples_dir: str,
    meta: dict[str, Any],
    sections: list[tuple[str, int, int, str]],
    channels: list[ChannelSpec],
    tracks: dict[str, list[dict[str, Any]]],
    *,
    max_bars: int = 8,
    insert_eq_params: list[tuple[str, Any]] | None = None,
    insert_eq_mixers: list[int] | None = None,
    bus_eq_params: list[tuple[str, Any]] | None = None,
    extra_plugins: list[tuple[int, str]] | None = None,
    limiter_ceiling_db: float | None = None,
    limiter_gain_db: float | None = None,
    fx_room_vol: float = 0.32,
    min_pattern_lufs: float = -36.0,
    auto_gain_stage: bool = True,
) -> dict[str, Any]:
    """
    Build a compact 4- or 8-bar `core_loop.gridline` containing only the densest
    Refrain/Drop section re-based to bar 1 (`start=0.0`), ready for `vertical_tester.py`.
    """
    bpb = int(meta["beats_per_bar"])
    sec_name, orig_start_bar, eff_bars, kind = select_core_section(
        sections, tracks=tracks, beats_per_bar=bpb, max_bars=max_bars
    )
    orig_start_beat = float(orig_start_bar * bpb)
    orig_end_beat = float((orig_start_bar + eff_bars) * bpb)

    rebased_tracks: dict[str, list[dict[str, Any]]] = {}
    for tkey, evs in tracks.items():
        sliced: list[dict[str, Any]] = []
        for ev in evs:
            st = float(ev["start"])
            if orig_start_beat - 1e-6 <= st < orig_end_beat - 1e-6:
                new_ev = dict(ev)
                new_ev["start"] = round(st - orig_start_beat, 4)
                ln = float(new_ev.get("length", new_ev.get("duration", 0.25)))
                max_ln = (orig_end_beat - st)
                new_ev["length"] = round(max(0.05, min(ln, max_ln)), 4)
                sliced.append(new_ev)
        if sliced:
            rebased_tracks[tkey] = sliced

    core_sections = [(f"Core {sec_name}", 0, eff_bars, kind)]
    core_markers = [{"name": f"Core {sec_name}", "position": 0.0}]
    core_meta = dict(meta)
    core_meta["total_bars"] = eff_bars

    build_gridline_project(
        proj_path=proj_path,
        export_dir=export_dir,
        samples_dir=samples_dir,
        meta=core_meta,
        sections=core_sections,
        channels=channels,
        tracks=rebased_tracks,
        markers=core_markers,
        insert_eq_params=insert_eq_params,
        insert_eq_mixers=insert_eq_mixers,
        bus_eq_params=bus_eq_params,
        extra_plugins=extra_plugins,
        automations=[],
        limiter_ceiling_db=limiter_ceiling_db,
        limiter_gain_db=limiter_gain_db,
        fx_room_vol=fx_room_vol,
        min_pattern_lufs=min_pattern_lufs,
        auto_gain_stage=auto_gain_stage,
    )
    return {
        "core_gridline": os.path.abspath(proj_path),
        "source_section": sec_name,
        "orig_start_bar": orig_start_bar + 1,
        "bars": eff_bars,
        "active_track_keys": sorted(rebased_tracks.keys()),
    }


def run_song_cli(
    proj_dir: str,
    out_gridline: str,
    sections: list[tuple[str, int, int, str]],
    channels: list[ChannelSpec],
    generate_fn: Callable[[dict[str, Any]], tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]],
    bass_keys: list[str],
    chords_key: str = "sketch_chords",
    lead_key: str = "lead_bell",
    insert_eq_params: list[tuple[str, Any]] | None = None,
    insert_eq_mixers: list[int] | None = None,
    bus_eq_params: list[tuple[str, Any]] | None = None,
    extra_plugins: list[tuple[int, str]] | None = None,
    automations_fn: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
    limiter_ceiling_db: float | None = None,
    limiter_gain_db: float | None = None,
    fx_room_vol: float = 0.32,
    samples_subdir: str = "samples",
    auto_gain_stage: bool = True,
) -> None:
    """Turnkey CLI entrypoint (`--midi-only`, `--core-only`, or full build) for per-song scripts."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi-only", action="store_true", help="Only export master_sketch.mid and run verify_midi_gate")
    ap.add_argument("--core-only", action="store_true", help="Build only the 4-8 bar Refrain/Drop into core_loop.gridline for vertical testing")
    ap.add_argument("--core-bars", type=int, default=8, help="Max bars for --core-only (4 or 8)")
    ap.add_argument("--no-auto-gain-stage", action="store_true", help="Disable Q1 active-loudness auto-gain staging")
    args = ap.parse_args()
    eff_auto_gs = bool(auto_gain_stage) and (not args.no_auto_gain_stage)

    bp_path = os.path.join(proj_dir, "song_blueprint.json")
    _bp, meta = load_blueprint(bp_path, validate=True)
    for _iss in (meta.get("musical_safety") or {}).get("issues") or []:
        if _iss.get("severity") == "error":
            print(f"[musical_safety] ERROR {_iss.get('code')} track={_iss.get('track')}: {_iss.get('fix')}", file=sys.stderr)

    tracks, markers = generate_fn(meta)
    sketch_path = os.path.join(proj_dir, "master_sketch.mid")
    export_master_sketch(
        tracks,
        sketch_path,
        meta["bpm"],
        bass_keys=bass_keys,
        chords_key=chords_key,
        lead_key=lead_key,
        numerator=int(meta.get("beats_per_bar", 4)),
        denominator=int(meta.get("denominator", 4)),
    )

    max_pitch = 72 if meta["valence"] < -0.3 else 76
    res = verify_midi_gate(
        sketch_path,
        expected_key=meta["key"],
        max_pitch=max_pitch,
        valence=meta["valence"],
        tension=meta["tension"],
        time_signature=(int(meta.get("beats_per_bar", 4)), int(meta.get("denominator", 4))),
        phrase_plan=str(meta.get("phrase_plan", "")),
    )
    print(json.dumps({
        "midi_gate": res.get("gate_passed"),
        "musicality": res.get("musicality_metrics"),
        "out_of_scale": res.get("out_of_scale_notes"),
    }, indent=2))

    if args.midi_only:
        return

    if args.core_only:
        core_path = os.path.join(proj_dir, "core_loop.gridline")
        info = build_core_loop_project(
            proj_path=core_path,
            export_dir=os.path.join(proj_dir, "exports"),
            samples_dir=os.path.join(proj_dir, samples_subdir),
            meta=meta,
            sections=sections,
            channels=channels,
            tracks=tracks,
            max_bars=max(4, min(8, int(args.core_bars))),
            insert_eq_params=insert_eq_params,
            insert_eq_mixers=insert_eq_mixers,
            bus_eq_params=bus_eq_params,
            extra_plugins=extra_plugins,
            limiter_ceiling_db=limiter_ceiling_db,
            limiter_gain_db=limiter_gain_db,
            fx_room_vol=fx_room_vol,
            auto_gain_stage=eff_auto_gs,
        )
        print(f"Built core loop {info['core_gridline']} ({info['bars']} bars from '{info['source_section']}')")
        return

    autos = automations_fn(meta) if automations_fn else []
    build_gridline_project(
        proj_path=out_gridline,
        export_dir=os.path.join(proj_dir, "exports"),
        samples_dir=os.path.join(proj_dir, samples_subdir),
        meta=meta,
        sections=sections,
        channels=channels,
        tracks=tracks,
        markers=markers,
        insert_eq_params=insert_eq_params,
        insert_eq_mixers=insert_eq_mixers,
        bus_eq_params=bus_eq_params,
        extra_plugins=extra_plugins,
        automations=autos,
        limiter_ceiling_db=limiter_ceiling_db,
        limiter_gain_db=limiter_gain_db,
        fx_room_vol=fx_room_vol,
        auto_gain_stage=eff_auto_gs,
    )
    print(f"Built {out_gridline} ({meta['total_bars']} bars @ {meta['bpm']} BPM, {meta['key']})")


# ---------------------------------------------------------------------------
# Phase 7 P1 — Declarative Arrangement Plan Compiler (`arrangement_plan.json`)
# ---------------------------------------------------------------------------

def load_arrangement_plan(
    plan_path: str | dict[str, Any],
    validate: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate `<project>/arrangement_plan.json` against `schemas/arrangement_plan.schema.json`."""
    if isinstance(plan_path, dict):
        plan = copy.deepcopy(plan_path)
    else:
        with open(os.path.abspath(os.path.expanduser(plan_path)), encoding="utf-8") as fh:
            plan = json.load(fh)
    if validate and os.path.isfile(ARRANGEMENT_PLAN_SCHEMA_PATH):
        with open(ARRANGEMENT_PLAN_SCHEMA_PATH, encoding="utf-8") as sfh:
            schema = json.load(sfh)
        jsonschema.validate(plan, schema)

    pmeta = plan["meta"]
    ts = pmeta["time_signature"]
    profile = str(pmeta.get("mix_profile", "trap_rap"))
    total_bars = sum(int(s["bars"]) for s in plan["sections"])
    meta = {
        "bpm": float(pmeta["bpm"]),
        "beats_per_bar": int(ts["numerator"]),
        "denominator": int(ts["denominator"]),
        "key": f"{pmeta['key']} {pmeta.get('mode', '')}".strip(),
        "tonic": str(pmeta["key"]),
        "mode": str(pmeta.get("mode", "minor")),
        "mix_profile": profile,
        "has_drums": bool(pmeta.get("has_drums", profile != "classical_ambient")),
        "phrase_plan": str(pmeta.get("phrase_plan", "A-A'-A''-B")),
        "valence": float(pmeta.get("valence", 0.0)),
        "tension": float(pmeta.get("tension", 0.4)),
        "energy": float(pmeta.get("energy", 0.7)),
        "depth": float(pmeta.get("depth", 0.5)),
        "total_bars": total_bars,
    }
    return plan, meta


def _extract_motif_entry(
    motifs: dict[str, Any],
    role: str,
    motif_name: str | None = None,
) -> tuple[list[dict[str, Any]], int | None, bool | None]:
    raw: Any = None
    if motif_name:
        if motif_name in motifs:
            raw = motifs[motif_name]
        elif isinstance(motifs.get(role), dict) and motif_name in motifs[role]:
            raw = motifs[role][motif_name]
    if raw is None:
        raw = motifs.get(role)
    if raw is None:
        # Search for motif objects declaring `"role": role`
        for v in motifs.values():
            if isinstance(v, dict) and str(v.get("role") or "").lower() == role.lower():
                raw = v
                break
    if raw is None:
        return [], None, None

    if isinstance(raw, list):
        return [copy.copy(n) for n in raw], None, None
    if isinstance(raw, dict):
        if "notes" in raw and isinstance(raw["notes"], list):
            bars = int(raw["bars"]) if raw.get("bars") is not None else None
            loop_flag = bool(raw["loop"]) if "loop" in raw else None
            return [copy.copy(n) for n in raw["notes"]], bars, loop_flag
        # Sub-motif dictionary: pick "default" or first entry
        sub = raw.get("default") if "default" in raw else next(iter(raw.values()), None)
        if isinstance(sub, list):
            return [copy.copy(n) for n in sub], None, None
        if isinstance(sub, dict) and "notes" in sub:
            bars = int(sub["bars"]) if sub.get("bars") is not None else None
            loop_flag = bool(sub["loop"]) if "loop" in sub else None
            return [copy.copy(n) for n in sub["notes"]], bars, loop_flag
    return [], None, None


def _tile_and_modify_section_notes(
    base_notes: list[dict[str, Any]],
    *,
    role: str,
    mod: dict[str, Any],
    sec_bars: int,
    bpb: float,
    motif_bars: int | None,
    loop_flag: bool | None,
) -> list[dict[str, Any]]:
    if not base_notes:
        return []
    sec_beats = float(sec_bars) * bpb
    norm_notes: list[dict[str, Any]] = []
    for n in base_notes:
        ev = {
            "start": float(n["start"]),
            "length": float(n.get("length", n.get("duration", 0.25))),
            "pitch": int(n["pitch"]),
            "velocity": int(n["velocity"]),
        }
        norm_notes.append(ev)

    max_end = max(ev["start"] + ev["length"] for ev in norm_notes)
    eff_cycle = motif_bars
    if eff_cycle is None and loop_flag is not False:
        raw_bars = max(1, int(math.ceil((max_end - 1e-4) / max(1.0, bpb))))
        for cand in (1, 2, 4, 8, 16, 32):
            if raw_bars <= cand:
                eff_cycle = cand
                break
        if eff_cycle is None:
            eff_cycle = raw_bars

    tiled: list[dict[str, Any]] = []
    should_tile = (
        eff_cycle is not None
        and eff_cycle < sec_bars
        and role.lower() != "fx"
        and (
            loop_flag is True
            or sec_bars >= 2 * eff_cycle
            or role.lower() in ("kick", "snare", "clap", "hihat", "perc", "bass", "808", "chords", "pad")
        )
    )
    if should_tile and eff_cycle:
        cycle_beats = float(eff_cycle) * bpb
        n_reps = int(math.ceil(sec_beats / cycle_beats))
        for rep in range(n_reps):
            off = rep * cycle_beats
            for ev in norm_notes:
                st = ev["start"] + off
                if st >= sec_beats - 1e-4:
                    continue
                ln = min(ev["length"], max(0.05, sec_beats - st))
                tiled.append({**ev, "start": round(st, 5), "length": round(ln, 5)})
    else:
        for ev in norm_notes:
            if ev["start"] >= sec_beats - 1e-4:
                continue
            ln = min(ev["length"], max(0.05, sec_beats - ev["start"]))
            tiled.append({**ev, "start": round(ev["start"], 5), "length": round(ln, 5)})

    if not tiled:
        return []

    oct_shift = int(mod.get("octave_shift", 0) or 0)
    vel_scale = float(mod.get("velocity_scale", 1.0) or 1.0)
    dens_scale = float(mod.get("density_scale", 1.0) or 1.0)

    for ev in tiled:
        if oct_shift != 0:
            ev["pitch"] = int(max(24, min(108, ev["pitch"] + oct_shift)))
        if abs(vel_scale - 1.0) > 1e-4:
            ev["velocity"] = int(max(20, min(127, round(ev["velocity"] * vel_scale))))

    if dens_scale < 0.999 and len(tiled) > 1:
        keep_n = max(1, int(round(len(tiled) * max(0.25, dens_scale))))
        scored = sorted(
            enumerate(tiled),
            key=lambda it: (
                1 if abs(it[1]["start"] - round(it[1]["start"])) < 0.08 else 0,
                it[1]["velocity"],
                -it[1]["start"],
            ),
            reverse=True,
        )
        keep_idx = {idx for idx, _ in scored[:keep_n]}
        tiled = [ev for idx, ev in enumerate(tiled) if idx in keep_idx]
    elif dens_scale > 1.001 and len(tiled) >= 1:
        add_n = max(1, int(round(len(tiled) * min(0.5, dens_scale - 1.0))))
        ranked = sorted(range(len(tiled)), key=lambda i: (tiled[i]["velocity"], -tiled[i]["start"]), reverse=True)
        extra_notes: list[dict[str, Any]] = []
        for idx in ranked[:add_n]:
            base_ev = tiled[idx]
            step = min(0.25, max(0.125, base_ev["length"] * 0.5))
            new_st = base_ev["start"] + step
            if new_st < sec_beats - 0.08:
                extra_notes.append({
                    "start": round(new_st, 5),
                    "length": round(min(step, sec_beats - new_st), 5),
                    "pitch": base_ev["pitch"],
                    "velocity": int(max(24, min(127, round(base_ev["velocity"] * 0.82)))),
                })
        tiled.extend(extra_notes)

    tiled.sort(key=lambda ev: (ev["start"], ev["pitch"]))
    return tiled


def _fold_group_into_register(group: list[dict[str, Any]], lo: int, hi: int) -> None:
    """Shift a simultaneous chord/lead group by whole octaves to fit `[lo, hi]` (keeps intervals)."""
    if not group:
        return
    center = (lo + hi) / 2.0
    best_shift = 0
    best_cost = float("inf")
    for shift in (-24, -12, 0, 12, 24):
        pitches = [int(ev["pitch"]) + shift for ev in group]
        if min(pitches) < lo or max(pitches) > hi:
            continue
        cost = abs(sum(pitches) / len(pitches) - center)
        if cost < best_cost:
            best_cost = cost
            best_shift = shift
    for ev in group:
        ev["pitch"] = int(max(lo, min(hi, int(ev["pitch"]) + best_shift)))


def _apply_dark_register_split(
    tracks: dict[str, list[dict[str, Any]]],
    valence: float,
    chords_keys: list[str],
    lead_keys: list[str],
) -> dict[str, Any]:
    """
    Phase 8 Q5.2 — When `V < -0.2`, fold Chords/Pad into `[48, 60]` and Lead into `[60, 72]`
    by whole-octave groups (no interval distortion), guaranteeing a >= 5-semitone centroid gap.
    """
    if float(valence) >= -0.2:
        return {"applied": False, "valence": float(valence)}

    def _process(keys: list[str], lo: int, hi: int) -> list[dict[str, Any]]:
        all_notes: list[dict[str, Any]] = []
        for k in keys:
            notes = tracks.get(k) or []
            if not notes:
                continue
            groups: dict[float, list[dict[str, Any]]] = {}
            for ev in notes:
                groups.setdefault(round(float(ev["start"]), 3), []).append(ev)
            for grp in groups.values():
                _fold_group_into_register(grp, lo, hi)
            notes.sort(key=lambda ev: (float(ev["start"]), int(ev["pitch"])))
            all_notes.extend(notes)
        return all_notes

    ch_notes = _process(chords_keys, 48, 60)
    ld_notes = _process(lead_keys, 60, 72)
    if not ch_notes or not ld_notes:
        return {"applied": True, "chords_notes": len(ch_notes), "lead_notes": len(ld_notes), "gap_st": None}

    ch_centroid = sum(int(n["pitch"]) for n in ch_notes) / len(ch_notes)
    ld_centroid = sum(int(n["pitch"]) for n in ld_notes) / len(ld_notes)
    gap = ld_centroid - ch_centroid
    if gap < 5.0 and min(int(n["pitch"]) for n in ch_notes) >= 48:
        for n in ch_notes:
            if int(n["pitch"]) - 12 >= 48:
                n["pitch"] = int(n["pitch"]) - 12
        for k in chords_keys:
            if tracks.get(k):
                tracks[k].sort(key=lambda ev: (float(ev["start"]), int(ev["pitch"])))
        ch_centroid = sum(int(n["pitch"]) for n in ch_notes) / len(ch_notes)
        gap = ld_centroid - ch_centroid
    return {
        "applied": True,
        "chords_notes": len(ch_notes),
        "lead_notes": len(ld_notes),
        "chords_centroid_midi": round(ch_centroid, 2),
        "lead_centroid_midi": round(ld_centroid, 2),
        "gap_st": round(gap, 2),
    }


def _apply_trim_db_to_volume(base_vol: float, trim_db: float) -> float:
    clamped_trim = clamp(float(trim_db), -8.0, 8.0)
    raw_fader = db_to_auto(clamped_trim, ref_fader=float(base_vol)) * 1.25
    return round(clamp(raw_fader, 0.10, 1.15), 4)


DEFAULT_PITCH_ZONES: tuple[tuple[int, int], ...] = ((48, 55), (56, 63), (64, 72))


def split_notes_by_pitch_zones(
    notes: list[dict[str, Any]],
    zones: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Phase 8 Q6.2 — Route notes to 3 pitch-split sampler zones while guaranteeing each
    note stays within +/-4 semitones of its zone centre (whole-octave folding only).
    The caller routes every zone sub-channel to the same mixer insert.
    """
    eff_zones = [tuple(int(v) for v in z) for z in (zones or DEFAULT_PITCH_ZONES)]
    out: dict[str, list[dict[str, Any]]] = {f"zone_{i + 1}": [] for i in range(len(eff_zones))}

    def _dist(p: int, lo: int, hi: int) -> int:
        return 0 if lo <= p <= hi else (lo - p if p < lo else p - hi)

    for ev in notes:
        p = int(ev.get("pitch", 60))
        idx = min(range(len(eff_zones)), key=lambda i: _dist(p, *eff_zones[i]))
        lo, hi = eff_zones[idx]
        while p < lo:
            p += 12
        while p > hi:
            p -= 12
        p = int(max(lo, min(hi, p)))
        out[f"zone_{idx + 1}"].append({**ev, "pitch": p})
    for k in out:
        out[k].sort(key=lambda e: (float(e["start"]), int(e["pitch"])))
    return out


def _default_channels_for_plan(
    plan: dict[str, Any],
    roles_ordered: list[str],
    proj_dir: str,
    valence: float,
) -> list[ChannelSpec]:
    explicit = plan.get("channels")
    if isinstance(explicit, list) and explicit:
        out: list[ChannelSpec] = []
        for i, item in enumerate(explicit, start=1):
            if isinstance(item, ChannelSpec):
                out.append(item)
            elif isinstance(item, dict):
                kw = dict(item)
                kw.setdefault("mixer", i)
                trim_db = kw.pop("volume_trim_db", None)
                spec = ChannelSpec(**kw)
                if trim_db is not None:
                    spec.volume = _apply_trim_db_to_volume(spec.volume, float(trim_db))
                out.append(spec)
        return out

    bp_tracks: dict[str, dict[str, Any]] = {}
    bp_path = os.path.join(proj_dir, "song_blueprint.json")
    if os.path.isfile(bp_path):
        try:
            with open(bp_path, encoding="utf-8") as fh:
                bp_obj = json.load(fh)
            for tr in bp_obj.get("tracks") or []:
                rk = str(tr.get("role") or tr.get("name") or "").lower()
                if rk:
                    bp_tracks[rk] = tr
        except Exception:
            pass

    ch_dict_cfg = explicit if isinstance(explicit, dict) else {}
    channels: list[ChannelSpec] = []
    for idx, role in enumerate(roles_ordered, start=1):
        rl = role.lower()
        ov = dict(ch_dict_cfg.get(role) or ch_dict_cfg.get(rl) or {})
        bp_tr = bp_tracks.get(rl) or {}

        if "kick" in rl:
            spec = ChannelSpec("Kick", role, idx, "drums", plugin="gl.kick", volume=0.76, sidechain_buses=[(18, 0.8), (19, 0.45)])
        elif "snare" in rl:
            spec = ChannelSpec("Snare", role, idx, "drums", plugin="gl.snare", volume=0.68, fx_send=0.08)
        elif "clap" in rl:
            spec = ChannelSpec("Clap", role, idx, "drums", plugin="gl.clap", volume=0.64, fx_send=0.10)
        elif "hat" in rl or "perc" in rl:
            spec = ChannelSpec("HiHat", role, idx, "drums", plugin="gl.hat", volume=0.52, pan=0.22)
        elif any(k in rl for k in ("bass", "808", "sub")):
            spec = ChannelSpec("Bass", role, idx, "bass", plugin="gl.warmsynth", role="bass", valence=valence, volume=0.62)
        elif any(k in rl for k in ("pad", "string")):
            spec = ChannelSpec("Pad", role, idx, "music", plugin="gl.warmsynth", role="pad", valence=valence, volume=0.46, fx_send=0.18, hall_send=0.22)
        elif any(k in rl for k in ("chord", "keys")):
            spec = ChannelSpec("Chords", role, idx, "music", plugin="gl.keys", role="chords", valence=valence, volume=0.50, fx_send=0.14, hall_send=0.14)
        elif "counter" in rl:
            spec = ChannelSpec("Counter Harmony", role, idx, "music", plugin="gl.fm", role="bell", valence=valence, volume=0.52, pan=-0.25, fx_send=0.12, delay_send=0.10, hall_send=0.16)
        elif "fx" in rl:
            spec = ChannelSpec("FX", role, idx, "music", plugin="gl.fm", role="bell", valence=valence, volume=0.44, pan=0.35, fx_send=0.20, delay_send=0.12, hall_send=0.25, single_shot_clips=True)
        elif any(k in rl for k in ("vocal", "vox")):
            spec = ChannelSpec("Vocals", role, idx, "music", plugin="gl.warmsynth", role="lead", valence=valence, volume=0.74, delay_send=0.12, hall_send=0.18)
        else:
            spec = ChannelSpec(role.capitalize(), role, idx, "music", plugin="gl.pluck", role="lead", valence=valence, volume=0.74, fx_send=0.16, delay_send=0.12, hall_send=0.10)

        if bp_tr:
            if bp_tr.get("source_type") == "sample" and bp_tr.get("sample_path"):
                spec.plugin = "gl.sampler"
                spec.sample_file = str(bp_tr["sample_path"])
            elif bp_tr.get("plugin") and str(bp_tr["plugin"]).startswith("gl."):
                spec.plugin = str(bp_tr["plugin"])
            if bp_tr.get("pan") is not None:
                spec.pan = float(bp_tr["pan"])
            # Root note / fine tune from palette detection: a sample not at C4 must not be
            # played from the default C4 root (systematic dissonance otherwise).
            if bp_tr.get("root_note") is not None:
                spec.root_note = int(bp_tr["root_note"])
            if bp_tr.get("fine_tune_ct") is not None:
                spec.fine_tune = int(bp_tr["fine_tune_ct"])
        for k, v in ov.items():
            if hasattr(spec, k):
                setattr(spec, k, v)
        if ov.get("volume_trim_db") is not None:
            spec.volume = _apply_trim_db_to_volume(spec.volume, float(ov["volume_trim_db"]))
        channels.append(spec)
    return channels


def apply_wetness_sends(
    channels: list[ChannelSpec],
    proj_dir: str,
    dry_gain: float = 0.8,
) -> dict[str, dict[str, float]]:
    """v3.2 TimbralCohesion — pondère `fx_send`/`hall_send` par `(1 - dry_gain * wetness)` :
    un sample déjà réverbéré (wetness haut, lu dans `samples_catalog.json`) reçoit quasi 0
    send FX Hall, un synthé sec reçoit l'ambiance nécessaire pour s'unifier dans le même espace."""
    cat_path = os.path.join(proj_dir, "samples_catalog.json")
    wet_by_base: dict[str, float] = {}
    if os.path.isfile(cat_path):
        try:
            with open(cat_path, encoding="utf-8") as fh:
                cat = json.load(fh)
            entries = cat.get("samples") if isinstance(cat, dict) else cat
            if isinstance(entries, dict):
                entries = list(entries.values())
            for e in entries or []:
                if isinstance(e, dict):
                    p = str(e.get("path") or e.get("file") or e.get("sample_path") or "")
                    if p:
                        wet_by_base[os.path.basename(p)] = float(e.get("wetness_score", 0.5) or 0.5)
        except Exception:
            pass
    report: dict[str, dict[str, float]] = {}
    for ch in channels:
        sf = str(getattr(ch, "sample_file", "") or "")
        if not sf:
            continue
        w = wet_by_base.get(os.path.basename(sf))
        if w is None:
            continue
        factor = max(0.0, 1.0 - dry_gain * w)
        old_fx, old_hall = float(ch.fx_send), float(ch.hall_send)
        if old_fx > 0.0:
            ch.fx_send = round(old_fx * factor, 4)
        if old_hall > 0.0:
            ch.hall_send = round(old_hall * factor, 4)
        if old_fx > 0.0 or old_hall > 0.0:
            report[ch.track_key] = {"wetness": w, "fx_send": [old_fx, ch.fx_send], "hall_send": [old_hall, ch.hall_send]}
    return report


def _timbre_features_for_channel(ch: Any, proj_dir: str) -> dict[str, Any]:
    """v3.1 HARM-03 : features timbre d'un canal (synthé gl.* ou sample du catalogue)."""
    plugin = str(getattr(ch, "plugin", "") or "")
    fam = plugin.replace("gl.", "").lower()
    feats: dict[str, Any] = {"plugin": plugin, "family": fam}
    if fam in ("fm", "matrixfm"):
        feats.update(inharmonicity=0.35, spectral_centroid_hz=2600.0, family="fm")
    elif fam in ("sawswarm", "wavetable", "analog", "dcosynth"):
        feats.update(spectral_centroid_hz=2400.0, family="saw")
    elif fam in ("bell", "pluck"):
        feats.update(inharmonicity=0.2, spectral_centroid_hz=2800.0, family="bell")
    sf = str(getattr(ch, "sample_file", "") or "")
    if sf:
        cat_path = os.path.join(proj_dir, "samples_catalog.json")
        if os.path.isfile(cat_path):
            try:
                with open(cat_path, encoding="utf-8") as fh:
                    cat = json.load(fh)
                base = os.path.basename(sf)
                entries = cat.get("samples") if isinstance(cat, dict) else cat
                if isinstance(entries, dict):
                    entries = list(entries.values())
                for e in entries or []:
                    if not isinstance(e, dict):
                        continue
                    p = str(e.get("path") or e.get("file") or e.get("sample_path") or "")
                    if p and (p.endswith(sf) or os.path.basename(p) == base):
                        feats.update({
                            "inharmonicity": float(e.get("inharmonicity", 0.0) or 0.0),
                            "spectral_centroid_hz": float(e.get("spectral_centroid_hz", 0.0) or 0.0),
                            "family": str(e.get("timbre_card") or feats.get("family", "")),
                        })
                        break
            except Exception:
                pass
    return feats


def _adapt_voicings_to_timbre(
    tracks: dict[str, list[dict[str, Any]]],
    channels: list[ChannelSpec],
    grid: list[dict[str, Any]],
    valence: float,
    proj_dir: str,
) -> dict[str, Any]:
    """v3.1 HARM-03 : allège les voicings d'accords selon le timbre (drop-2/shell,
    confinement {1,5,8} sur timbre inharmonique, LIL) avant construction du ScoreIR."""
    ch_by_key = {ch.track_key: ch for ch in channels}
    report: dict[str, Any] = {"tracks": {}}
    for tkey, notes in tracks.items():
        if classify_role(tkey) != "chords" or not notes:
            continue
        ch = ch_by_key.get(tkey)
        feats = _timbre_features_for_channel(ch, proj_dir) if ch is not None else {}
        groups: dict[float, list[dict[str, Any]]] = {}
        for ev in notes:
            groups.setdefault(round(float(ev.get("start", 0.0)), 4), []).append(ev)
        n_changed = 0
        for beat, evs in groups.items():
            if len(evs) < 2:
                continue
            cidx = segment_index_for_beat(grid, beat)
            seg = grid[cidx] if 0 <= cidx < len(grid) else (grid[0] if grid else {"root_pc": 0})
            voicing = sorted({int(e["pitch"]) for e in evs})
            new_v, _rep = adapt_voicing_to_timbre(voicing, feats, valence=valence, chord_root_pc=int(seg.get("root_pc", 0)))
            if new_v != voicing:
                evs_sorted = sorted(evs, key=lambda e: int(e["pitch"]))
                for e, p in zip(evs_sorted, new_v):
                    e["pitch"] = int(p)
                if len(new_v) < len(evs_sorted):
                    for e in evs_sorted[len(new_v):]:
                        notes.remove(e)
                n_changed += 1
        if n_changed:
            report["tracks"][tkey] = {
                "voicings_changed": n_changed,
                "timbre": {k: feats.get(k) for k in ("family", "inharmonicity", "spectral_centroid_hz")},
            }
    return report


def compile_arrangement_plan(
    plan_path: str | dict[str, Any],
    proj_dir: str,
    out_gridline: str,
    core_only: bool = False,
    *,
    channels: list[ChannelSpec] | None = None,
    apply_prosody: bool = True,
    apply_vocal_sync: bool = True,
    verify_gate: bool = True,
    auto_fix_midi: bool = True,
    core_bars: int = 8,
    samples_subdir: str = "samples",
    limiter_ceiling_db: float | None = None,
    limiter_gain_db: float | None = None,
    fx_room_vol: float = 0.30,
    auto_gain_stage: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Phase 7 P1 compiler: validate `arrangement_plan.json`, expand motifs & `role_modifiers`,
    apply `groove_lib`, `phrasing_engine` (P2), `transitions_lib` and `vocal_arrangement_sync` (P6),
    export `master_sketch.mid`, and compile `.gridline` (or `core_loop.gridline` if `core_only=True`).
    """
    abs_proj = os.path.abspath(os.path.expanduser(proj_dir))
    os.makedirs(abs_proj, exist_ok=True)
    plan, meta = load_arrangement_plan(plan_path, validate=True)

    # v3.3 #5 — déterminisme strict : seed du blueprint (ou dérivé SHA-256), propagé au groove
    _bp_raw: dict[str, Any] = {}
    try:
        with open(os.path.join(abs_proj, "song_blueprint.json"), encoding="utf-8") as _fh:
            _bp_raw = json.load(_fh)
    except Exception:
        _bp_raw = {}
    if seed is None:
        from determinism import resolve_seed
        seed = resolve_seed(_bp_raw if isinstance(_bp_raw, dict) and _bp_raw else meta)
    seed = int(seed)

    bpm = float(meta["bpm"])
    bpb = int(meta["beats_per_bar"])
    denom = int(meta["denominator"])
    valence = float(meta["valence"])
    motifs = plan["motifs"]

    roles_ordered: list[str] = []
    for sec in plan["sections"]:
        for r in sec.get("active_roles") or []:
            if r not in roles_ordered:
                roles_ordered.append(r)
    for r in motifs.keys():
        if r not in roles_ordered and isinstance(motifs[r], list):
            roles_ordered.append(r)

    eff_channels = list(channels) if channels is not None else _default_channels_for_plan(plan, roles_ordered, abs_proj, valence)
    role_to_ch = {ch.track_key: ch for ch in eff_channels}
    # v3.2 TimbralCohesion : sends FX pondérés par le DRR natif des samples (wetness)
    wetness_report = apply_wetness_sends(eff_channels, abs_proj)

    tracks: dict[str, list[dict[str, Any]]] = {ch.track_key: [] for ch in eff_channels}
    sections_tuples: list[tuple[str, int, int, str]] = []
    markers: list[dict[str, Any]] = []
    autos: list[dict[str, Any]] = []
    auto_track_cursor = max(len(eff_channels) + 1, 9)

    role_cutoffs: dict[str, list[tuple[float, float, float]]] = {}
    role_gains: dict[str, list[tuple[float, float, float]]] = {}

    for sec in plan["sections"]:
        sec_name = str(sec["name"])
        sec_kind = str(sec["kind"])
        s_bar = int(sec["start_bar"])  # 0-indexed
        n_bars = int(sec["bars"])
        sections_tuples.append((sec_name, s_bar, n_bars, sec_kind))
        markers.append({"name": sec_name, "position": float(s_bar * bpb)})

        sec_start_beat = float(s_bar * bpb)
        sec_end_beat = float((s_bar + n_bars) * bpb)
        mods = sec.get("role_modifiers") or {}

        for role in sec.get("active_roles") or []:
            mod = dict(mods.get(role) or {})
            base_notes, m_bars, loop_flag = _extract_motif_entry(motifs, role, mod.get("motif"))
            sec_notes = _tile_and_modify_section_notes(
                base_notes,
                role=role,
                mod=mod,
                sec_bars=n_bars,
                bpb=float(bpb),
                motif_bars=m_bars,
                loop_flag=loop_flag,
            )
            for ev in sec_notes:
                ev["start"] = round(float(ev["start"]) + sec_start_beat, 5)
            tracks.setdefault(role, []).extend(sec_notes)

            if mod.get("cutoff_hz") is not None:
                role_cutoffs.setdefault(role, []).append((sec_start_beat, sec_end_beat, float(mod["cutoff_hz"])))
            if mod.get("gain_db") is not None:
                role_gains.setdefault(role, []).append((sec_start_beat, sec_end_beat, float(mod["gain_db"])))

    # v3.1 HARM-03 — couplage timbre <-> voicing avant le ScoreIR (allège les accords denses
    # sur timbre riche/inharmonique : drop-2/shell, confinement {1,5,8}, LIL).
    hc_plan = plan.get("harmonic_core") or {"key": meta["tonic"], "mode": meta["mode"]}
    # v3.2 NarrativeHarmony : progressions par section (harmonic_core.section_progressions)
    _grid0 = harmonic_grid_from_core(hc_plan, bpm=bpm, beats_per_bar=bpb, total_bars=int(meta["total_bars"]), section_spans=sections_tuples)
    voicing_report = _adapt_voicings_to_timbre(tracks, eff_channels, _grid0, valence, abs_proj)

    # v3.1 IR-02 — ScoreIR : l'inertie de fréquence reste structurelle (fusion de sub),
    # puis tout le groove (strum/AMG/haas/swing) s'accumule dans `expression` sans toucher
    # grid_beat/pitch, et compile_score_ir_to_notes applique le clamp de frontière harmonique
    # (guard 20 ms) + sanitize_note_overlaps (garde 10 ms) en étape finale.
    for r_key, r_notes in list(tracks.items()):
        rl = r_key.lower()
        if any(k in rl for k in ("bass", "808", "sub")):
            tracks[r_key] = enforce_frequency_inertia(r_notes, bpm=bpm)
    _voices_by_role = {str(ch.track_key).lower(): int(getattr(ch, "voices", 4) or 4) for ch in eff_channels}
    # v3.2 #4 — banque de grooves extraits des .mid locaux (pistes percussives uniquement)
    groove_bank_report: dict[str, Any] = {}
    gb_cfg = plan.get("groove_bank") or {}
    if isinstance(gb_cfg, dict) and gb_cfg.get("enabled"):
        try:
            from midi_groove_bank import DEFAULT_BANK_PATH, groove_to_notes, query_groove_bank
            tpl = query_groove_bank(gb_cfg.get("bank_path") or DEFAULT_BANK_PATH,
                                    role=gb_cfg.get("role", "hihat_perc"), bpm=bpm, profile=meta["mix_profile"])
            if tpl:
                target = gb_cfg.get("track") or next((k for k in tracks if "hat" in k.lower()), None)
                if target:
                    pitch = int(gb_cfg.get("pitch", 42))
                    added = 0
                    for _nm, s_bar, n_bars, _k in sections_tuples:
                        sec_start = float(s_bar * bpb)
                        for rep in range(max(1, n_bars // max(1, int(tpl.get("bars", 1))))):
                            tracks.setdefault(target, []).extend(
                                groove_to_notes(tpl, sec_start + rep * int(tpl.get("bars", 1)) * bpb, float(bpb), pitch)
                            )
                            added += len(tpl.get("steps", []))
                    groove_bank_report = {"template": tpl.get("source"), "role": tpl.get("role"), "added": added, "track": target}
        except Exception as exc:
            groove_bank_report = {"error": str(exc)[:120]}
    score_ir = build_score_ir(hc_plan, meta, tracks, channels=eff_channels, voices_by_role=_voices_by_role, section_spans=sections_tuples)
    # v3.2 NarrativeHarmony : opérateurs motiviques déterministes (phrasé 8 mesures Q/R,
    # syncope au refrain, call-and-response, apex unique) + suspensions 4->3/9->8/b6->5.
    narrative_report: dict[str, Any] = {}
    if plan.get("phrasing_8bar") or (isinstance(hc_plan, dict) and hc_plan.get("section_progressions")):
        try:
            from narrative_harmony import apply_narrative_harmony
            narrative_report = apply_narrative_harmony(score_ir, plan, section_spans=sections_tuples)
        except ImportError:
            narrative_report = {"error": "narrative_harmony indisponible"}
    # v3.2 #5a — tournoi de 2-3 variantes d'ornementation en Tour 2 (0 rendu audio) :
    # on retient celle qui maximise le MelodicAppealIndex à invariants durs égaux.
    motif_selection_report: dict[str, Any] = {}
    if plan.get("motif_selection"):
        try:
            import copy as _copy
            from musicality_metrics import select_best_motif_variant
            from narrative_harmony import melodic_climax_arch, rhythmic_lift
            variants = [(score_ir, "base")]
            v1 = _copy.deepcopy(score_ir)
            melodic_climax_arch(v1, period_bars=8)
            variants.append((v1, "climax_arch"))
            v2 = _copy.deepcopy(score_ir)
            rhythmic_lift(v2, section_spans=sections_tuples)
            variants.append((v2, "rhythmic_lift"))
            sel = select_best_motif_variant(score_ir, variants)
            score_ir = sel["score_ir"]
            motif_selection_report = {"selected": sel["selected"], "mai": sel["mai"],
                                      "evaluate_passed": sel["evaluate_passed"]}
        except Exception as exc:
            motif_selection_report = {"error": str(exc)[:120]}
    swing = float(meta.get("swing", 0.0) or (plan.get("meta") or {}).get("swing", 0.0) or 0.0)
    # v3.3 #2 — OrchestrationEngine : bass_mode (inversions/pédales), comping par section,
    # revoicing des accords conscient du Lead (anti-croisement soprano/Lead).
    orch_report: dict[str, Any] = {}
    try:
        from orchestration_engine import apply_bass_motion, apply_comping_to_score_ir, default_comping_arc, revoice_chords_with_lead
        from score_ir import parse_key_tonic_pc
        _hc_plan = plan.get("harmonic_core") or {}
        _bass_modes: dict[str, str] = {}
        for _k, _v in (_hc_plan.get("section_progressions") or {}).items():
            if isinstance(_v, dict) and _v.get("bass_mode"):
                _bass_modes[_k] = str(_v["bass_mode"])
        if not _bass_modes:
            _bass_modes = {"intro": "tonic_pedal", "pre_chorus": "dominant_pedal", "pre_refrain": "dominant_pedal"}
        _tonic_pc = parse_key_tonic_pc(str(meta.get("key") or meta.get("tonic") or "C")) or 0
        orch_report["bass"] = apply_bass_motion(score_ir, mode="stepwise_inversions", section_spans=sections_tuples,
                                                section_modes=_bass_modes, tonic_pc=int(_tonic_pc), beats_per_bar=int(bpb))
        _styles: dict[str, str] = {}
        for _sec in plan.get("sections") or []:
            if isinstance(_sec, dict) and _sec.get("comping_style"):
                _styles[str(_sec.get("name") or _sec.get("kind") or "")] = str(_sec["comping_style"])
        if not _styles:
            _styles = default_comping_arc(plan.get("sections") or [])
        orch_report["comping"] = apply_comping_to_score_ir(score_ir, sections_tuples, _styles, beats_per_bar=int(bpb))
        orch_report["voice_leading"] = revoice_chords_with_lead(score_ir)
    except Exception as exc:
        orch_report = {"error": str(exc)[:120]}

    # v3.3 #1 — enricher : glides 808 subtils au Climax si expression_defaults (jamais d'écrasement)
    try:
        from blueprint_enricher import inject_climax_glides
        if isinstance(_bp_raw, dict) and _bp_raw.get("expression_defaults"):
            inject_climax_glides(score_ir, _bp_raw, sections_tuples)
    except Exception:
        pass
    groove_counts = apply_groove_to_score_ir(score_ir, swing=swing, bpm=bpm, profile=meta["mix_profile"], beats_per_bar=float(bpb), seed=seed)
    tracks, score_ir_report = compile_score_ir_to_notes(score_ir)
    try:
        score_ir.save(os.path.join(abs_proj, "score_ir.json"))
    except Exception:
        pass

    # Phase 7 P2 — Instrumental prosody (phrase velocity arc, LPF breathing, cadential ornaments, section pocket)
    if apply_prosody:
        try:
            from phrasing_engine import apply_instrumental_prosody

            hc = plan.get("harmonic_core") or {"key": meta["tonic"], "mode": meta["mode"]}
            ev_vec = {
                "valence": meta["valence"],
                "tension": meta["tension"],
                "energy": meta["energy"],
                "depth": meta["depth"],
            }
            ch_mixer_map = {ch.track_key: ch.mixer for ch in eff_channels}
            tracks, prosody_autos = apply_instrumental_prosody(
                tracks,
                sections_tuples,
                hc,
                ev_vec,
                bpm=bpm,
                time_signature=(bpb, denom),
                track_idx_start=auto_track_cursor,
                channel_mixer_map=ch_mixer_map,
            )
            for pa in prosody_autos:
                autos.append(pa)
                auto_track_cursor = max(auto_track_cursor, int(pa.get("track", auto_track_cursor)) + 1)
        except ImportError:
            pass

    # Phase 8 Q5 — Rich transitions (fills/risers/anticipation) & dark register split
    bass_keys_all = [r for r in roles_ordered if any(k in r.lower() for k in ("bass", "808", "sub"))]
    chords_keys_all = [r for r in roles_ordered if any(k in r.lower() for k in ("chord", "pad", "keys", "harmony"))]
    lead_keys_all = [r for r in roles_ordered if any(k in r.lower() for k in ("lead", "melody", "topline"))]
    fx_ch = next((ch for ch in eff_channels if any(k in ch.track_key.lower() for k in ("fx", "riser", "ear_candy"))), None)
    fx_ch_idx = (eff_channels.index(fx_ch) + 1) if fx_ch is not None else None
    fx_mixer_idx = fx_ch.mixer if fx_ch is not None else 21

    declared_transitions: list[dict[str, Any]] = []
    for tr in plan.get("transitions") or []:
        if isinstance(tr, dict):
            declared_transitions.append(tr)
    for sec in plan["sections"]:
        raw_tr = sec.get("transitions")
        if isinstance(raw_tr, list):
            declared_transitions.extend([t for t in raw_tr if isinstance(t, dict)])

    transition_reports: list[dict[str, Any]] = []
    for tr in declared_transitions:
        bar_1 = int(tr.get("bar", 0) or 0)
        if bar_1 <= 0:
            continue
        t_beat = float(bar_1 - 1) * float(bpb)
        typ = str(tr.get("type") or "").strip().lower()
        rep: dict[str, Any] = {"bar": bar_1, "type": typ, "beat": round(t_beat, 3)}
        if "anticipation" in typ:
            pivots: list[float] = []
            for nm, s_bar, n_bars, _k in sections_tuples:
                for local_b in (2, 4):
                    if local_b < n_bars:
                        pivots.append(float((s_bar + local_b) * bpb))
            anticipation = apply_harmonic_anticipation(
                tracks.get(chords_keys_all[0], []) if chords_keys_all else [],
                tracks.get(bass_keys_all[0], []) if bass_keys_all else [],
                pivots,
                offset_beats=float(tr.get("offset_beats", 0.5)),
                ratio=float(tr.get("ratio", 0.35)),
            )
            rep["anticipation"] = anticipation
        else:
            if "fill" in typ:
                rep["drum_fill"] = generate_drum_fill(
                    tracks,
                    t_beat,
                    bpb=float(bpb),
                    profile=str(meta["mix_profile"]),
                    style=str(tr.get("style") or "auto"),
                    bpm=bpm,
                )
            if any(k in typ for k in ("riser", "impact", "fx")):
                rep["fx"] = add_transition_fx(
                    tracks,
                    autos,
                    t_beat,
                    bpb=float(bpb),
                    kind="riser_impact" if ("riser" in typ or "fx" in typ) else "impact",
                    fx_track_idx=min(31, auto_track_cursor),
                    fx_channel_idx=fx_ch_idx,
                    fx_mixer_idx=fx_mixer_idx,
                    impact=("impact" in typ or "riser" in typ or "fx" in typ),
                )
                auto_track_cursor = min(31, auto_track_cursor + 1)
        transition_reports.append(rep)

    # Merge helper-generated keys that have no dedicated channel back into the nearest real track
    for synth_key, kw_tuple in (("Snare", ("snare", "clap", "kick", "hat")), ("FX", ("fx", "riser", "ear_candy"))):
        if synth_key in tracks and synth_key not in role_to_ch:
            fallback = next((k for k in eff_channels if any(kw in k.track_key.lower() for kw in kw_tuple)), None)
            if fallback is not None and fallback.track_key != synth_key:
                tracks.setdefault(fallback.track_key, []).extend(tracks.pop(synth_key))
                tracks[fallback.track_key].sort(key=lambda ev: (float(ev["start"]), int(ev["pitch"])))
            else:
                tracks.pop(synth_key, None)

    register_split = _apply_dark_register_split(
        tracks,
        valence,
        chords_keys=chords_keys_all,
        lead_keys=lead_keys_all,
    )

    # Insert EQ params for kick / chords / section cutoffs (+ vocal formant ducking Band 5 if vocal present)
    insert_eq_params: list[tuple[str, Any]] = []
    insert_eq_mixers: list[int] = []

    # Phase 7 P6 — Vocal-arrangement sync if vocals asset exists in project
    if apply_vocal_sync:
        ustx_cands = [os.path.join(abs_proj, fn) for fn in sorted(os.listdir(abs_proj)) if fn.endswith(".ustx")] if os.path.isdir(abs_proj) else []
        vox_wav = os.path.join(abs_proj, samples_subdir, "vocals.wav")
        if ustx_cands or os.path.isfile(vox_wav):
            try:
                from vocal_arrangement_sync import (
                    build_formant_ducking_automation,
                    extract_vocal_phrases,
                    formant_ducking_eq_params,
                    mute_counter_during_vocals,
                )

                phrases = extract_vocal_phrases(
                    ustx_path=ustx_cands[0] if ustx_cands else "",
                    audio_path=vox_wav if os.path.isfile(vox_wav) else "",
                    bpm=bpm,
                )
                if phrases:
                    tracks = mute_counter_during_vocals(tracks, phrases)
                    insert_eq_params.extend(formant_ducking_eq_params(mixer_idx=19, slot_idx=2, band_idx=5, freq_hz=1850.0, q=1.4))
                    f_auto = build_formant_ducking_automation(
                        phrases,
                        bpm=bpm,
                        track_idx=min(31, auto_track_cursor),
                    )
                    autos.append(f_auto)
                    auto_track_cursor = min(31, auto_track_cursor + 1)
            except ImportError:
                pass

    for ch in eff_channels:
        rl = ch.track_key.lower()
        if "kick" in rl:
            insert_eq_params.extend(eq_kick_clean(ch.mixer))
            insert_eq_mixers.append(ch.mixer)
        elif any(k in rl for k in ("chord", "pad")):
            insert_eq_params.extend(eq_music_pocket(ch.mixer))
            insert_eq_mixers.append(ch.mixer)

    # Role modifiers: cutoff_hz & gain_db automations
    for role, spans in role_cutoffs.items():
        ch = role_to_ch.get(role)
        if ch is None:
            continue
        if ch.mixer not in insert_eq_mixers:
            insert_eq_mixers.append(ch.mixer)
            insert_eq_params.extend(eq_params(ch.mixer, [_dsl_eq_band(7, "lpf", spans[0][2])], slot=1, quality=0))
        t_lane = min(31, auto_track_cursor)
        auto_track_cursor = min(31, auto_track_cursor + 1)
        for s_b, e_b, cut_hz in spans:
            filter_sweep(
                autos,
                mixer_idx=ch.mixer,
                slot_idx=1,
                start_beat=s_b,
                end_beat=e_b,
                start_hz=cut_hz,
                end_hz=cut_hz,
                curve="linear",
                track_idx=t_lane,
                steps=2,
                name=f"{ch.name} LPF",
            )

    for role, spans in role_gains.items():
        ch = role_to_ch.get(role)
        if ch is None:
            continue
        t_lane = min(31, auto_track_cursor)
        auto_track_cursor = min(31, auto_track_cursor + 1)
        pts: list[dict[str, Any]] = []
        tot_beats = float(meta["total_bars"] * bpb)
        for s_b, e_b, gdb in spans:
            val = db_to_auto(gdb)
            pts.append({"position": round(s_b, 4), "value": val, "shape": "hold"})
            pts.append({"position": round(max(s_b + 0.1, e_b - 0.05), 4), "value": val, "shape": "hold"})
        if pts:
            autos.append({
                "name": f"{ch.name} Gain",
                "mixer": int(ch.mixer),
                "param": "volume",
                "track": t_lane,
                "start": 0.0,
                "length": round(tot_beats, 4),
                "points": sorted(pts, key=lambda p: p["position"]),
            })

    # Macro section_gain_arc + section transitions (beat4_dropout, filter_sweep, outro_fade)
    if len(sections_tuples) >= 2:
        arc_plan = [(nm, sb + 1, sb + nb, knd) for nm, sb, nb, knd in sections_tuples]
        section_gain_arc(autos, bus_mixer_idx=19, sections_plan=arc_plan, beats_per_bar=float(bpb), track_idx=min(31, auto_track_cursor))
        auto_track_cursor = min(31, auto_track_cursor + 1)

    for sec in plan["sections"]:
        tr_cfg = sec.get("transitions") or {}
        s_bar = int(sec["start_bar"])
        n_bars = int(sec["bars"])
        for rel_b in tr_cfg.get("beat4_dropout_bars") or []:
            rb = int(rel_b)
            if rb < 0:
                rb = n_bars + rb
            if 0 <= rb < n_bars:
                beat4_dropout(
                    tracks,
                    autos,
                    bar_idx=s_bar + rb,
                    beats_per_bar=bpb,
                    fx_room_mixer=20,
                    track_idx=min(31, auto_track_cursor),
                )
                auto_track_cursor = min(31, auto_track_cursor + 1)

        fsw = tr_cfg.get("filter_sweep")
        if isinstance(fsw, dict):
            sw_role = str(fsw.get("role") or "chords")
            ch = role_to_ch.get(sw_role)
            target_mx = ch.mixer if ch is not None else 19
            target_slot = 1 if ch is not None else 2
            if ch is not None and ch.mixer not in insert_eq_mixers:
                insert_eq_mixers.append(ch.mixer)
                insert_eq_params.extend(eq_params(ch.mixer, [_dsl_eq_band(7, "lpf", float(fsw["start_hz"]))], slot=1, quality=0))
            sw_s = float(s_bar * bpb)
            sw_e = min(float((s_bar + n_bars) * bpb), sw_s + float(fsw["bars"]) * float(bpb))
            filter_sweep(
                autos,
                mixer_idx=target_mx,
                slot_idx=target_slot,
                start_beat=sw_s,
                end_beat=sw_e,
                start_hz=float(fsw["start_hz"]),
                end_hz=float(fsw["end_hz"]),
                curve=str(fsw.get("curve") or "exp"),  # type: ignore[arg-type]
                track_idx=min(31, auto_track_cursor),
            )
            auto_track_cursor = min(31, auto_track_cursor + 1)

        fade_bars = int(tr_cfg.get("outro_fade_bars") or 0)
        if fade_bars > 0:
            f_end = float((s_bar + n_bars) * bpb)
            f_start = max(float(s_bar * bpb), f_end - float(fade_bars * bpb))
            outro_fade(autos, master_or_bus_idx=0, start_beat=f_start, end_beat=f_end, track_idx=min(31, auto_track_cursor))
            auto_track_cursor = min(31, auto_track_cursor + 1)

    # v3.2 ExpressionArticulations — respiration finale organique (ritardando 4-8 % sur les
    # 2 dernières mesures) pour les styles organiques, puis couplage vélocité -> Cutoff sur
    # l'harmonie (automation douce par phrase de 4 mesures).
    ritardando_report: dict[str, Any] = {}
    organic_profiles = ("lofi_chill", "jazz_acoustic", "classical_ambient")
    rit_cfg = plan.get("outro_ritardando") or {}
    if isinstance(rit_cfg, dict) and rit_cfg.get("enabled", str(meta["mix_profile"]) in organic_profiles):
        from transitions_lib import apply_outro_ritardando
        _loop_meta = [
            {"track": k, "is_audio_loop": ("loop" in k.lower()) or (
                role_to_ch.get(k) is not None
                and getattr(role_to_ch[k], "plugin", "") == "gl.sampler"
                and "loop" in str(getattr(role_to_ch[k], "sample_file", "") or "").lower())}
            for k in tracks
        ]
        ritardando_report = apply_outro_ritardando(
            tracks, bpm,
            bars=int(rit_cfg.get("bars", 2)),
            amount=float(rit_cfg.get("amount", 0.06)),
            tracks_meta=_loop_meta,
        )
    expr_layers = plan.get("expression_layers") or {}
    vel_cutoff_report: dict[str, Any] = {}
    if expr_layers.get("velocity_cutoff", True):
        try:
            from phrasing_engine import velocity_cutoff_automation
            for ck in chords_keys_all:
                ch = role_to_ch.get(ck)
                if ch is None:
                    continue
                t_lane = min(31, auto_track_cursor)
                auto_track_cursor = min(31, auto_track_cursor + 1)
                vel_cutoff_report[ck] = velocity_cutoff_automation(
                    tracks, sections_tuples, autos, bpm,
                    track_key=ck, mixer_idx=ch.mixer, track_idx=t_lane,
                )
        except ImportError:
            pass

    # v3.1 IR-02 — clamp de frontière harmonique final (post-groove / post-transitions) :
    # aucune note chords/pad/bass/counter ne bave sur un accord suivant incompatible,
    # puis sanitize final (vérification post-groove < 5 ms : 0 clash b2, 0 chevauchement 808).
    _roles_map = {
        ch.track_key: (str(ch.role) if getattr(ch, "role", None) else classify_role(ch.track_key))
        for ch in eff_channels
    }
    tracks, _n_boundary_clamped = clamp_flat_tracks_at_boundaries(
        tracks, score_ir.harmonic_grid, bpm, roles_map=_roles_map
    )
    tracks = sanitize_note_overlaps(tracks, bpm=bpm, voices_by_role=_voices_by_role)
    post_groove_check = {
        "boundary_clamps_applied": _n_boundary_clamped,
        "groove_counts": groove_counts,
        "score_ir_report": score_ir_report,
        "voicing_adaptation": voicing_report,
        "narrative_harmony": narrative_report,
        "ritardando": ritardando_report,
        "velocity_cutoff": vel_cutoff_report,
        "wetness_sends": wetness_report,
        "motif_selection": motif_selection_report,
        "groove_bank": groove_bank_report,
        "orchestration": orch_report,
    }

    # Export master_sketch.mid and verify MIDI gate
    bass_keys = [r for r in roles_ordered if any(k in r.lower() for k in ("bass", "808", "sub"))]
    chords_key = next((r for r in roles_ordered if any(k in r.lower() for k in ("chord", "pad"))), "chords")
    lead_key = next((r for r in roles_ordered if any(k in r.lower() for k in ("lead", "melody"))), "lead")
    sketch_path = os.path.join(abs_proj, "master_sketch.mid")
    export_master_sketch(
        tracks,
        sketch_path,
        bpm,
        bass_keys=bass_keys,
        chords_key=chords_key,
        lead_key=lead_key,
        numerator=bpb,
        denominator=denom,
    )

    gate_res: dict[str, Any] | None = None
    if verify_gate and any(tracks.get(k) for k in [*bass_keys, chords_key, lead_key]):
        max_pitch = 72 if valence < -0.3 else 76
        gate_res = verify_midi_gate(
            sketch_path,
            expected_key=meta["key"],
            max_pitch=max_pitch,
            valence=valence,
            tension=float(meta["tension"]),
            time_signature=(bpb, denom),
            phrase_plan=meta["phrase_plan"],
            auto_fix=auto_fix_midi,
        )

    abs_out = os.path.abspath(os.path.expanduser(out_gridline))
    if core_only:
        core_info = build_core_loop_project(
            proj_path=abs_out,
            export_dir=os.path.join(abs_proj, "exports"),
            samples_dir=os.path.join(abs_proj, samples_subdir),
            meta=meta,
            sections=sections_tuples,
            channels=eff_channels,
            tracks=tracks,
            max_bars=max(4, min(8, int(core_bars))),
            insert_eq_params=insert_eq_params or None,
            insert_eq_mixers=insert_eq_mixers or None,
            limiter_ceiling_db=limiter_ceiling_db,
            limiter_gain_db=limiter_gain_db,
            fx_room_vol=fx_room_vol,
            min_pattern_lufs=-38.0,
            auto_gain_stage=auto_gain_stage,
        )
        return {
            "gridline_path": abs_out,
            "core_only": True,
            "core_info": core_info,
            "meta": meta,
            "sections": sections_tuples,
            "midi_gate": gate_res,
            "transitions": transition_reports,
            "register_split": register_split,
            "post_groove_check": post_groove_check,
        }

    build_gridline_project(
        proj_path=abs_out,
        export_dir=os.path.join(abs_proj, "exports"),
        samples_dir=os.path.join(abs_proj, samples_subdir),
        meta=meta,
        sections=sections_tuples,
        channels=eff_channels,
        tracks=tracks,
        markers=markers,
        insert_eq_params=insert_eq_params or None,
        insert_eq_mixers=insert_eq_mixers or None,
        automations=autos,
        limiter_ceiling_db=limiter_ceiling_db,
        limiter_gain_db=limiter_gain_db,
        fx_room_vol=fx_room_vol,
        min_pattern_lufs=-38.0,
        auto_gain_stage=auto_gain_stage,
    )
    return {
        "gridline_path": abs_out,
        "core_only": False,
        "meta": meta,
        "sections": sections_tuples,
        "midi_gate": gate_res,
        "transitions": transition_reports,
        "register_split": register_split,
        "post_groove_check": post_groove_check,
    }
