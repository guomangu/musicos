#!/usr/bin/env python3
"""
Shared AI-Studio toolkit for Gridline 0.7.7 orchestration, sample hygiene,
standard bus setup (BUS Drums / BUS 808 / BUS Music / FX Room / Master),
and Top-Down MIDI gate verification.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from gl_dsl import (
    eq_band as dsl_eq_band,
    eq_params as dsl_eq_params,
    limiter_master,
    reverb_room,
    to_mixer_params,
)

ROOT = "/home/gamo/Bureau/musicos"
TEMPLATE_EMPTY_MIXER = "/usr/share/gridline/assets/templates/Basics/Empty with mixer tracks.gridline"


class GridlineMCP:
    """Persistent stdio JSON-RPC client for `gridline mcp <proj> --export-dir <dir>`."""

    def __init__(self, proj_path: str, export_dir: str, client_name: str = "ai_studio"):
        os.makedirs(export_dir, exist_ok=True)
        self.proc = subprocess.Popen(
            ["gridline", "mcp", proj_path, "--export-dir", export_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._id = 0
        self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": client_name, "version": "1.0"},
            },
        )
        self._notify("notifications/initialized", {})

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n")
        self.proc.stdin.flush()

    def _rpc(self, method: str, params: dict[str, Any]) -> Any:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self._id += 1
        rid = self._id
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                err = self.proc.stderr.read() if self.proc.stderr else ""
                raise RuntimeError(f"MCP EOF on {method}: {err}")
            resp = json.loads(line)
            if resp.get("id") == rid:
                if "error" in resp:
                    raise RuntimeError(f"RPC error {method}: {resp['error']}")
                return resp["result"]

    def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        for attempt in range(8):
            try:
                res = self._rpc("tools/call", {"name": tool, "arguments": arguments})
                text = res["content"][0]["text"]
                data = json.loads(text) if text.startswith("{") or text.startswith("[") else text
                if res.get("isError"):
                    raise RuntimeError(f"Tool {tool} error: {text}")
                return data
            except RuntimeError as exc:
                if "Rate limit" in str(exc) and attempt < 7:
                    time.sleep(0.8)
                    continue
                raise

    def batch(self, calls: list[dict[str, Any]], chunk_size: int = 100) -> list[Any]:
        out = []
        for i in range(0, len(calls), chunk_size):
            r = self.call("batch", {"calls": calls[i : i + chunk_size]})
            if isinstance(r, dict) and r.get("errors"):
                raise RuntimeError(f"Batch errors: {r['errors']}")
            out.append(r)
        return out

    def set_params(self, changes: list[tuple[str, Any]], chunk_size: int = 250) -> None:
        for i in range(0, len(changes), chunk_size):
            chunk = [{"path": p, "value": v} for p, v in changes[i : i + chunk_size]]
            self.call("set_parameters", {"changes": chunk})

    def close(self) -> None:
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream:
                try:
                    stream.close()
                except Exception:
                    pass
        self.proc.wait(timeout=15)


def clean_audio_sample(
    in_path: str,
    out_path: str,
    lpf_hz: float | None = None,
    hpf_hz: float | None = None,
    fade_in_ms: float = 2.0,
    fade_out_ms: float = 15.0,
    remove_dc: bool = True,
    max_peak: float | None = 0.85,
    trim_end_below_ratio: float | None = None,
) -> dict[str, Any]:
    """
    Create a clean, de-clicked, DC-removed WAV derivative in `out_path`.
    Returns audio telemetry (sr, dur_s, max_diff, peak).
    """
    y, sr = sf.read(in_path, dtype="float32", always_2d=True)
    if remove_dc:
        y = y - np.mean(y, axis=0, keepdims=True)
    if trim_end_below_ratio is not None and len(y) > int(0.05 * sr):
        pk = float(np.max(np.abs(y))) + 1e-12
        nz = np.where(np.max(np.abs(y), axis=1) > trim_end_below_ratio * pk)[0]
        if len(nz):
            end_idx = min(len(y), int(nz[-1] + 0.03 * sr))
            y = y[:end_idx]
    if hpf_hz:
        sos_hp = signal.butter(2, float(hpf_hz), btype="highpass", fs=sr, output="sos")
        y = np.stack([signal.sosfilt(sos_hp, y[:, c]) for c in range(y.shape[1])], axis=1).astype(np.float32)
    if lpf_hz and lpf_hz < sr * 0.48:
        sos_lp = signal.butter(2, float(lpf_hz), btype="lowpass", fs=sr, output="sos")
        y = np.stack([signal.sosfilt(sos_lp, y[:, c]) for c in range(y.shape[1])], axis=1).astype(np.float32)

    fi = min(len(y) // 4, int(fade_in_ms * sr / 1000.0))
    fo = min(len(y) // 4, int(fade_out_ms * sr / 1000.0))
    if fi > 0:
        y[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)[:, None]
    if fo > 0:
        y[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)[:, None]

    if max_peak is not None:
        pk = float(np.max(np.abs(y)))
        if pk > max_peak:
            y *= float(max_peak / pk)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, y, sr, subtype="PCM_24")
    return {
        "out_path": out_path,
        "sr": int(sr),
        "dur_s": round(len(y) / sr, 3),
        "max_diff": round(float(np.max(np.abs(np.diff(y[:, 0])))) if len(y) > 1 else 0.0, 4),
        "peak": round(float(np.max(np.abs(y))), 4),
    }


def stabilize_stereo_low_mid(
    in_path: str,
    out_path: str,
    cutoff_hz: float = 650.0,
    side_keep_low: float = 0.18,
    side_keep_high: float = 0.85,
) -> str:
    """Apply gentle M/S <650 Hz narrowing while preserving high-frequency stereo width."""
    y, sr = sf.read(in_path, dtype="float32", always_2d=True)
    if y.shape[1] < 2:
        y = np.repeat(y, 2, axis=1)
    M = 0.5 * (y[:, 0] + y[:, 1])
    S = 0.5 * (y[:, 0] - y[:, 1])
    sos_lp = signal.butter(2, cutoff_hz, btype="lowpass", fs=sr, output="sos")
    sos_hp = signal.butter(2, cutoff_hz, btype="highpass", fs=sr, output="sos")
    S_new = side_keep_low * signal.sosfilt(sos_lp, S) + side_keep_high * signal.sosfilt(sos_hp, S)
    out = np.column_stack([M + S_new, M - S_new]).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, out, sr, subtype="PCM_24")
    return out_path


def extract_clean_oneshot(
    in_path: str,
    out_path: str,
    start_s: float = 0.0,
    max_dur_s: float = 0.65,
    fade_in_ms: float = 2.0,
    fade_out_ms: float = 25.0,
    lpf_hz: float | None = None,
    hpf_hz: float | None = None,
) -> dict[str, Any]:
    """
    Extract a single stable mono-note one-shot from a sample or sequence:
    automatically cuts before the next detected onset if `start_s == 0.0`,
    removes DC, applies clean fades, and verifies `playable_chromatically`.
    """
    import librosa
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from audio_metrics import analyze_single_file

    y, sr = sf.read(in_path, dtype="float32", always_2d=True)
    y = y - np.mean(y, axis=0, keepdims=True)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)

    i0 = max(0, int(start_s * sr))
    y_sub = y[i0:]
    # Trim leading silence
    mono = np.max(np.abs(y_sub), axis=1)
    pk = float(np.max(mono)) + 1e-12
    nz = np.where(mono > 0.025 * pk)[0]
    if len(nz):
        y_sub = y_sub[max(0, int(nz[0] - 0.0015 * sr)) :]

    # Find 2nd distinct onset (> 100ms) so we cut cleanly before the next note in a sequence
    m_sub = y_sub.mean(axis=1)
    o_env = librosa.onset.onset_strength(y=m_sub, sr=sr, hop_length=256)
    o_times = librosa.frames_to_time(
        librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=256, delta=0.18, wait=max(4, int(0.08 * sr / 256))),
        sr=sr, hop_length=256,
    )
    end_s = max_dur_s
    for t in o_times:
        if t > 0.11:
            end_s = min(end_s, float(t) - 0.012)
            break
    n_end = max(int(0.08 * sr), min(len(y_sub), int(end_s * sr)))
    y_cut = y_sub[:n_end].copy()

    tmp = out_path + ".tmp.wav"
    sf.write(tmp, y_cut, sr, subtype="PCM_24")
    tel = clean_audio_sample(
        tmp, out_path, lpf_hz=lpf_hz, hpf_hz=hpf_hz,
        fade_in_ms=fade_in_ms, fade_out_ms=fade_out_ms,
        remove_dc=True, max_peak=0.85,
    )
    os.remove(tmp)
    entry, _ = analyze_single_file(out_path, with_clap=False)
    dsp = entry["dsp"]
    tel.update({
        "sample_nature": dsp.get("sample_nature"),
        "playable_chromatically": dsp.get("playable_chromatically"),
        "detected_note": dsp.get("detected_note"),
        "f0_hz": dsp.get("detected_f0_hz"),
        "midi_root": dsp.get("midi_root"),
        "recommended_fine_tune_ct": dsp.get("recommended_fine_tune_ct"),
        "f0_std_semitones": dsp.get("f0_std_semitones"),
    })
    return tel


def check_pattern_audibility(mcp: GridlineMCP, min_lufs: float = -24.0) -> list[dict[str, Any]]:
    """Verify every non-empty pattern is audible (`silent == False` and `lufs >= min_lufs`)."""
    pats = mcp.call("list_patterns", {}).get("patterns", [])
    issues: list[dict[str, Any]] = []
    for p in pats:
        idx = p["pattern"]
        pat_info = mcp.call("get_pattern", {"pattern": idx})
        if not pat_info.get("notes") and not pat_info.get("steps"):
            continue
        ld = mcp.call("analyze_loudness", {"scope": "pattern", "pattern": idx})
        lufs = ld.get("lufs")
        if ld.get("silent") or (lufs is not None and float(lufs) < min_lufs):
            issues.append({
                "pattern": idx,
                "name": p.get("name"),
                "lufs": lufs,
                "silent": ld.get("silent"),
            })
    return issues


def verify_midi_gate(
    midi_path: str,
    expected_key: str,
    max_pitch: int = 74,
    valence: float = 0.0,
    tension: float = 0.3,
    min_pitch: int = 31,
    auto_fix: bool = False,
    time_signature: tuple[int, int] | str | None = None,
    phrase_plan: str = "",
    fixed_out_path: str | None = None,
) -> dict[str, Any]:
    """
    Run strict musicality gate via audio-analyzer and raise on FAIL.
    When `auto_fix=True`, delegates deterministic repairs to `scripts/midi_autofix.py`
    before raising on remaining structural errors.
    `min_pitch` defaults to 31 (G1 = 49 Hz) to prevent < 42 Hz infra-sub 808s;
    pass `min_pitch=28` explicitly if an acoustic/electric bass legitimately uses E1..F#1.
    """
    if f"{ROOT}/mcp-servers/audio-analyzer" not in sys.path:
        sys.path.insert(0, f"{ROOT}/mcp-servers/audio-analyzer")
    if f"{ROOT}/scripts" not in sys.path:
        sys.path.insert(0, f"{ROOT}/scripts")
    from deep_diagnostics import audit_midi_file

    res = audit_midi_file(
        midi_path=midi_path,
        expected_key=expected_key,
        min_pitch=min_pitch,
        max_pitch=max_pitch,
        strict_musicality=True,
        valence=valence,
        tension=tension,
        time_signature=time_signature,
        phrase_plan=phrase_plan,
    )
    if not res.get("gate_passed") and auto_fix:
        from midi_autofix import autofix_midi

        fix_report = autofix_midi(
            midi_path=midi_path,
            expected_key=expected_key,
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            strict_musicality=True,
            valence=valence,
            tension=tension,
            time_signature=time_signature,
            phrase_plan=phrase_plan,
            out_path=fixed_out_path or midi_path,
            in_place=(fixed_out_path is None),
        )
        post_res = dict(fix_report.get("post_audit") or res)
        post_res["fix_report"] = {
            "fixed_midi_path": fix_report.get("fixed_midi_path"),
            "counts_by_rule": fix_report.get("counts_by_rule"),
            "total_changes": fix_report.get("total_changes"),
            "changes": fix_report.get("changes"),
            "remaining_structural_failures": fix_report.get("remaining_structural_failures"),
        }
        res = post_res

    if not res.get("gate_passed"):
        raise RuntimeError(
            f"MIDI GATE FAILED ({midi_path}): failures={res.get('musicality_failures')} "
            f"too_high={res.get('too_high_count')} "
            f"too_low={res.get('too_low_count')} "
            f"out_of_scale={res.get('out_of_scale_notes')} "
            f"blocking_clashes={res.get('blocking_minor_2nd_clashes')} "
            f"sub_collisions={res.get('low_end_simultaneous_collisions')}"
        )
    return res


def setup_standard_buses(
    mcp: GridlineMCP,
    bpm: float = 140.0,
    bus_drums: int = 17,
    bus_808: int = 18,
    bus_music: int = 19,
    fx_room: int = 20,
    limiter_ceiling_db: float = -2.35,
    limiter_gain_db: float = 7.5,
    fx_room_vol: float = 0.30,
    profile: str = "trap_rap",
    valence: float = 0.0,
) -> None:
    """
    Instantiate calibrated Master + BUS Drums + BUS 808/Bass + BUS Music (Timbral Glue)
    + FX Room (Dry=0, Dry full=0, Decay <= T_noire), adapted to `profile` and `valence`:
    - BUS Music places `gl.dist` (Slot 1, Tone <= 7500 Hz) BEFORE `gl.eq` (Slot 2, HPF + LPF)
      so any saturation harmonics above LPF are cleanly filtered out.
    - trap_rap: smooth Kick->808 sidechain (Ratio 2.4, Threshold -18 dB, Release 65 ms) to preserve 808 body,
      gentle soft saturation, drums air (unless valence < -0.2)
    - edm_club: punchy sidechain (5:1, -20 dB, 85 ms), gentle soft saturation, drums air (unless valence < -0.2)
    - pop_rnb / rock_metal: balanced glue, moderate sidechain (3:1)
    - lofi_chill: warmer LPF on BUS Music, soft glue, light drums air
    - jazz_acoustic / classical_ambient: clean transparent bus (no distortion, gentle/minimal comp, larger room decay)
    """
    t_noire_s = 60.0 / float(bpm)
    acoustic_mode = profile in ("jazz_acoustic", "classical_ambient")
    lofi_mode = profile == "lofi_chill"
    dark_mode = valence < -0.2
    rt60 = round(min(1.10 if acoustic_mode else 0.42, max(0.18, t_noire_s * (1.2 if acoustic_mode else 0.65))), 3)

    # Ensure mixer tracks up to max bus index exist (Empty template only has 0..16)
    mixer_info = mcp.call("get_mixer", {})
    existing_count = len(mixer_info.get("tracks", []))
    max_needed = max(bus_drums, bus_808, bus_music, fx_room)
    if existing_count <= max_needed:
        add_tracks = [
            {"tool": "add_mixer_track", "arguments": {"name": f"Insert {idx}"}}
            for idx in range(existing_count, max_needed + 1)
        ]
        mcp.batch(add_tracks)

    fx_adds = [
        {"tool": "add_plugin", "arguments": {"mixer": 0, "plugin": "gl.eq"}},
        {"tool": "add_plugin", "arguments": {"mixer": 0, "plugin": "gl.comp"}},
        {"tool": "add_plugin", "arguments": {"mixer": 0, "plugin": "gl.limiter"}},
        {"tool": "add_plugin", "arguments": {"mixer": bus_drums, "plugin": "gl.eq"}},
        {"tool": "add_plugin", "arguments": {"mixer": bus_drums, "plugin": "gl.comp"}},
        {"tool": "add_plugin", "arguments": {"mixer": bus_808, "plugin": "gl.stereo"}},
        {"tool": "add_plugin", "arguments": {"mixer": bus_808, "plugin": "gl.comp"}},
        {"tool": "add_plugin", "arguments": {"mixer": bus_808, "plugin": "gl.eq"}},
        # BUS Music: gl.dist (Slot 1) BEFORE gl.eq (Slot 2) so saturation harmonics are filtered by LPF
        {"tool": "add_plugin", "arguments": {"mixer": bus_music, "plugin": "gl.dist"}},
        {"tool": "add_plugin", "arguments": {"mixer": bus_music, "plugin": "gl.eq"}},
        {"tool": "add_plugin", "arguments": {"mixer": bus_music, "plugin": "gl.comp"}},
        {"tool": "add_plugin", "arguments": {"mixer": bus_music, "plugin": "gl.comp"}},
        {"tool": "add_plugin", "arguments": {"mixer": fx_room, "plugin": "gl.reverb"}},
    ]
    mcp.batch(fx_adds)

    music_hpf = 90 if acoustic_mode else (110 if lofi_mode else 135)
    if lofi_mode:
        music_lpf = 8500 if dark_mode else 9500
    elif acoustic_mode:
        music_lpf = 12000 if dark_mode else 13500
    else:
        music_lpf = 10500 if dark_mode else 12500
    music_dist_mix = 0.0 if acoustic_mode else (0.10 if lofi_mode else 0.08)
    music_dist_tone = 6500 if (lofi_mode or dark_mode) else 7500
    drums_air_gain = 0.0 if (acoustic_mode or lofi_mode or dark_mode) else 1.8
    if acoustic_mode:
        sc_ratio_bass, sc_thresh_bass, sc_release_bass = 1.5, -20.0, 0.085
    elif profile == "trap_rap":
        sc_ratio_bass, sc_thresh_bass, sc_release_bass = 2.4, -18.0, 0.065
    elif profile == "edm_club":
        sc_ratio_bass, sc_thresh_bass, sc_release_bass = 5.0, -20.0, 0.085
    else:
        sc_ratio_bass, sc_thresh_bass, sc_release_bass = 3.0, -20.0, 0.085
    sc_ratio_music = 1.2 if acoustic_mode else 2.2
    master_comp_thresh = -8.5 if acoustic_mode else -10.5
    master_comp_makeup = 0.5 if acoustic_mode else 1.5

    changes: list[tuple[str, Any]] = [
        # Master (look-ahead 0.002s avoids True Peak overshoots & hi-hat micro-glitches)
        *dsl_eq_params(0, [dsl_eq_band(1, "hpf", 28), dsl_eq_band(7, "lpf", 17000)], slot=1, quality=1),
        ("mixer/0/slot/2/Threshold", master_comp_thresh), ("mixer/0/slot/2/Ratio", 1.6),
        ("mixer/0/slot/2/Attack", 0.025), ("mixer/0/slot/2/Release", 0.110),
        ("mixer/0/slot/2/Makeup", master_comp_makeup), ("mixer/0/slot/2/Detector high-pass", 95),
        *to_mixer_params(0, 3, limiter_master(limiter_gain_db, limiter_ceiling_db)),
        # BUS Drums
        *dsl_eq_params(
            bus_drums,
            [dsl_eq_band(1, "hpf", 34), dsl_eq_band(7, "high_shelf", 10000, drums_air_gain)],
            slot=1,
        ),
        (f"mixer/{bus_drums}/slot/2/Threshold", -15.0), (f"mixer/{bus_drums}/slot/2/Ratio", 2.2),
        (f"mixer/{bus_drums}/slot/2/Attack", 0.015), (f"mixer/{bus_drums}/slot/2/Release", 0.085), (f"mixer/{bus_drums}/slot/2/Makeup", 2.0),
        # BUS Bass / 808
        (f"mixer/{bus_808}/slot/1/Width", 0.0),
        (f"mixer/{bus_808}/slot/2/Detector input", 1),
        (f"mixer/{bus_808}/slot/2/Threshold", sc_thresh_bass), (f"mixer/{bus_808}/slot/2/Ratio", sc_ratio_bass),
        (f"mixer/{bus_808}/slot/2/Attack", 0.002), (f"mixer/{bus_808}/slot/2/Release", sc_release_bass), (f"mixer/{bus_808}/slot/2/Lookahead", 2),
        *dsl_eq_params(bus_808, [dsl_eq_band(1, "hpf", 30)], slot=3),
        # BUS Music: Slot 1 gl.dist (Tone <= 7500 Hz) -> Slot 2 gl.eq (HPF + LPF) -> Slot 3 Glue -> Slot 4 Duck
        (f"mixer/{bus_music}/slot/1/Mode", 0), (f"mixer/{bus_music}/slot/1/Drive", 3.5),
        (f"mixer/{bus_music}/slot/1/Tone", music_dist_tone), (f"mixer/{bus_music}/slot/1/Mix", music_dist_mix),
        (f"mixer/{bus_music}/slot/1/Output", 0.0), (f"mixer/{bus_music}/slot/1/Oversampling", 2), (f"mixer/{bus_music}/slot/1/Remove DC", 1),
        *dsl_eq_params(bus_music, [dsl_eq_band(1, "hpf", music_hpf), dsl_eq_band(7, "lpf", music_lpf)], slot=2),
    ]
    if dark_mode:
        changes.extend(to_mixer_params(bus_music, 2, dsl_eq_band(6, "peak", 3200, -2.0, q=1.4)))
    fx_room_params = reverb_room(
        decay_sec=rt60,
        hpf_hz=350.0 if acoustic_mode else 480.0,
        lpf_hz=6800.0,
        wet_pct=45.0,
        er_pct=11.0 if acoustic_mode else 8.0,
        pre_delay_ms=12.0,
        room_size=0.45 if acoustic_mode else 0.24,
    )
    fx_room_params.update({"High damping": 4200, "Width": 0.80})
    changes.extend([
        (f"mixer/{bus_music}/slot/3/Threshold", -13.5), (f"mixer/{bus_music}/slot/3/Ratio", 1.8),
        (f"mixer/{bus_music}/slot/3/Attack", 0.022), (f"mixer/{bus_music}/slot/3/Release", 0.110), (f"mixer/{bus_music}/slot/3/Makeup", 1.5),
        (f"mixer/{bus_music}/slot/4/Detector input", 1),
        (f"mixer/{bus_music}/slot/4/Threshold", -21.0), (f"mixer/{bus_music}/slot/4/Ratio", sc_ratio_music),
        (f"mixer/{bus_music}/slot/4/Attack", 0.005), (f"mixer/{bus_music}/slot/4/Release", 0.085), (f"mixer/{bus_music}/slot/4/Lookahead", 1),
        # FX Room (Dry=0, Dry full=0, Decay <= T_noire for rhythmic genres)
        *to_mixer_params(fx_room, 1, fx_room_params),
    ])
    mcp.set_params(changes)
    mcp.batch([
        {"tool": "edit_mixer_track", "arguments": {"mixer": bus_drums, "name": "BUS Drums", "routeTo": 0, "volume": 1.0, "pan": 0.0}},
        {"tool": "edit_mixer_track", "arguments": {"mixer": bus_808, "name": "BUS Bass", "routeTo": 0, "volume": 1.0, "pan": 0.0}},
        {"tool": "edit_mixer_track", "arguments": {"mixer": bus_music, "name": "BUS Music", "routeTo": 0, "volume": 1.0, "pan": 0.0}},
        {"tool": "edit_mixer_track", "arguments": {"mixer": fx_room, "name": "FX Room", "routeTo": 0, "volume": fx_room_vol, "pan": 0.0}},
    ])
