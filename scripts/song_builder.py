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
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

import jsonschema
import pretty_midi

ROOT = "/home/gamo/Bureau/musicos"
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYZER_DIR = os.path.join(ROOT, "mcp-servers", "audio-analyzer")
for _d in (SCRIPTS_DIR, ANALYZER_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from musicality_metrics import optimize_voice_leading  # noqa: F401 (re-exported for song scripts)
from studio_lib import (
    TEMPLATE_EMPTY_MIXER,
    GridlineMCP,
    check_pattern_audibility,
    setup_standard_buses,
    verify_midi_gate,
)

SCHEMA_PATH = os.path.join(ROOT, "schemas", "song_blueprint.schema.json")

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
    meta = {
        "bpm": float(bp["bpm"]),
        "beats_per_bar": int(bp["time_signature"]["numerator"]),
        "denominator": int(bp["time_signature"]["denominator"]),
        "key": bp["key"],
        "mix_profile": bp.get("mix_profile", "trap_rap"),
        "valence": float(ev.get("valence", 0.0)),
        "tension": float(ev.get("tension", 0.3)),
        "energy": float(ev.get("energy", 0.7)),
        "depth": float(ev.get("depth", 0.5)),
        "total_bars": sum(int(s["bars"]) for s in bp["structure"]),
    }
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
    # Native gl.* extra params
    plugin_params: dict[str, Any] = field(default_factory=dict)
    # Routing & arrangement behavior
    fx_send: float = 0.0
    sidechain_buses: list[tuple[int, float]] = field(default_factory=list)
    single_shot_clips: bool = False  # True for isolated FX triggers (creates 1-bar clip per event)


# --- Concise gl.eq builders (Band type: 1=Low cut, 3=High cut, 5=Low shelf, 6=Peak, 7=High shelf) ---

def eq_band(
    mixer: int,
    band: int,
    btype: int,
    freq: float,
    gain: float = 0.0,
    width: float | None = None,
    slot: int = 1,
) -> list[tuple[str, Any]]:
    """Build `(path, value)` pairs for one `gl.eq` band (`Peak=6`, width via `Band N width`)."""
    prefix = f"mixer/{mixer}/slot/{slot}/Band {band}"
    out: list[tuple[str, Any]] = [(f"{prefix} type", int(btype)), (f"{prefix} freq", float(freq))]
    if btype in (5, 6, 7):
        out.append((f"{prefix} gain", float(gain)))
    if width is not None and btype in (2, 6):
        out.append((f"{prefix} width", float(width)))
    return out


def eq_insert(mixer: int, bands: list[tuple[Any, ...]], slot: int = 1) -> list[tuple[str, Any]]:
    """
    Expand a list of compact band tuples into `gl.eq` parameters:
      (band_idx, type_int, freq_hz)
      (band_idx, type_int, freq_hz, gain_db)
      (band_idx, type_int, freq_hz, gain_db, width)
    """
    out: list[tuple[str, Any]] = []
    for item in bands:
        b = int(item[0])
        bt = int(item[1])
        fr = float(item[2])
        gn = float(item[3]) if len(item) > 3 else 0.0
        wd = float(item[4]) if len(item) > 4 else None
        out.extend(eq_band(mixer, b, bt, fr, gn, wd, slot=slot))
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
        (1, 1, hpf_hz),
        (2, 6, mud_hz, mud_db, 0.45),
        (3, 6, box_hz, box_db, 0.40),
        (4, 6, pocket_hz, pocket_db, 0.42),
    ])


def eq_kick_clean(mixer: int, hpf_hz: float = 42.0, sub_cut_db: float = -9.5, punch_db: float = 3.0) -> list[tuple[str, Any]]:
    """Standard Trap/Rap kick EQ (removes sub rumble, trims 50 Hz boom, keeps 95 Hz punch)."""
    return eq_insert(mixer, [
        (1, 1, hpf_hz),
        (2, 6, 50.0, sub_cut_db, 0.40),
        (3, 6, 95.0, punch_db, 0.45),
        (4, 6, 210.0, -4.0, 0.38),
    ])


def export_master_sketch(
    tracks: dict[str, list[dict[str, Any]]],
    sketch_path: str,
    bpm: float,
    bass_keys: list[str],
    chords_key: str = "sketch_chords",
    lead_key: str = "lead_bell",
) -> str:
    """Export 3-track `master_sketch.mid` (`Bass`, `Chords`, `Lead`) from 0-indexed beat events."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    b2s = 60.0 / float(bpm)

    bass_all: list[dict[str, Any]] = []
    for bk in bass_keys:
        bass_all.extend(tracks.get(bk, []))
    chords_all = tracks.get(chords_key, [])
    lead_all = tracks.get(lead_key, [])

    for name, evs in [("Bass", bass_all), ("Chords", chords_all), ("Lead", lead_all)]:
        inst = pretty_midi.Instrument(program=0, name=name)
        for ev in sorted(evs, key=lambda x: x["start"]):
            st = float(ev["start"]) * b2s
            en = (float(ev["start"]) + float(ev["length"])) * b2s
            inst.notes.append(pretty_midi.Note(int(ev["velocity"]), int(ev["pitch"]), st, en))
        pm.instruments.append(inst)

    os.makedirs(os.path.dirname(os.path.abspath(sketch_path)), exist_ok=True)
    pm.write(sketch_path)
    return sketch_path


def _resolve_bus(bus: str | int) -> int:
    if isinstance(bus, int):
        return bus
    return BUS_MAP[str(bus).lower()]


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
    limiter_ceiling_db: float = -2.55,
    limiter_gain_db: float = 12.0,
    fx_room_vol: float = 0.32,
    min_pattern_lufs: float = -32.0,
) -> None:
    """
    Build a complete `.gridline` project in one atomic MCP session enforcing strict isolation
    (`1 sonorité = 1 canal + 1 insert mixeur + patterns dédiés + 1 piste Playlist nommée`).
    """
    os.makedirs(os.path.dirname(os.path.abspath(proj_path)), exist_ok=True)
    shutil.copyfile(TEMPLATE_EMPTY_MIXER, proj_path)

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
        )

        # Ensure any mixer tracks > 20 exist before adding channels
        max_mixer_needed = max(ch.mixer for ch in channels)
        if max_mixer_needed > 20:
            mixer_by_idx = {ch.mixer: ch.name for ch in channels}
            add_extras = [
                {"tool": "add_mixer_track", "arguments": {"name": mixer_by_idx.get(idx, f"Insert {idx}")}}
                for idx in range(21, max_mixer_needed + 1)
            ]
            mcp.batch(add_extras)

        # Add channels (1 sonorité = 1 canal + 1 insert dédié)
        mcp.batch([
            {
                "tool": "add_channel",
                "arguments": {
                    "name": ch.name,
                    "plugin": ch.plugin,
                    "mixer": ch.mixer,
                    "volume": ch.volume,
                    "pan": ch.pan,
                },
            }
            for ch in channels
        ])

        # Load optional manifest.json for automatic fine-tune lookup when fine_tune is None
        manifest_data: dict[str, Any] = {}
        manifest_path = os.path.join(samples_dir, "manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as mfh:
                manifest_data = json.load(mfh)

        patches: list[dict[str, Any]] = []
        param_changes: list[tuple[str, Any]] = []
        for idx, ch in enumerate(channels, start=1):
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
                "arguments": {"mixer": ch.mixer, "name": ch.name, "routeTo": _resolve_bus(ch.bus)},
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
                    if m_id not in (0, 17, 18, 19, 20) and m_id not in seen_m:
                        seen_m.add(m_id)
                        eq_mixers.append(m_id)
        if eq_mixers:
            mcp.batch([{"tool": "add_plugin", "arguments": {"mixer": m, "plugin": "gl.eq"}} for m in eq_mixers])
        if extra_plugins:
            mcp.batch([{"tool": "add_plugin", "arguments": {"mixer": int(m), "plugin": str(pl)}} for m, pl in extra_plugins])

        all_eq = list(insert_eq_params or []) + list(bus_eq_params or [])
        if all_eq:
            mcp.set_params(all_eq)

        # Sends (Sidechain + FX Room)
        sends: list[dict[str, Any]] = []
        for ch in channels:
            for sc_bus, sc_lvl in ch.sidechain_buses:
                sends.append({
                    "tool": "set_send",
                    "arguments": {"mixer": ch.mixer, "to": sc_bus, "level": sc_lvl, "sidechain": True, "preFader": True},
                })
            if ch.fx_send > 0.0:
                sends.append({"tool": "set_send", "arguments": {"mixer": ch.mixer, "to": 20, "level": ch.fx_send}})
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
    limiter_ceiling_db: float = -2.55,
    limiter_gain_db: float = 12.0,
    fx_room_vol: float = 0.32,
    samples_subdir: str = "samples",
) -> None:
    """Turnkey CLI entrypoint (`--midi-only` + full build) for per-song scripts."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi-only", action="store_true", help="Only export master_sketch.mid and run verify_midi_gate")
    args = ap.parse_args()

    bp_path = os.path.join(proj_dir, "song_blueprint.json")
    _bp, meta = load_blueprint(bp_path, validate=True)

    tracks, markers = generate_fn(meta)
    sketch_path = os.path.join(proj_dir, "master_sketch.mid")
    export_master_sketch(tracks, sketch_path, meta["bpm"], bass_keys=bass_keys, chords_key=chords_key, lead_key=lead_key)

    max_pitch = 72 if meta["valence"] < -0.3 else 76
    res = verify_midi_gate(
        sketch_path,
        expected_key=meta["key"],
        max_pitch=max_pitch,
        valence=meta["valence"],
        tension=meta["tension"],
    )
    print(json.dumps({
        "midi_gate": res.get("gate_passed"),
        "musicality": res.get("musicality_metrics"),
        "out_of_scale": res.get("out_of_scale_notes"),
    }, indent=2))

    if not args.midi_only:
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
        )
        print(f"Built {out_gridline} ({meta['total_bars']} bars @ {meta['bpm']} BPM, {meta['key']})")
