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
        if self.proc.stdin:
            self.proc.stdin.close()
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


def verify_midi_gate(midi_path: str, expected_key: str, max_pitch: int = 84) -> dict[str, Any]:
    """Run strict 7-metric musicality gate via audio-analyzer and raise on FAIL."""
    sys.path.insert(0, f"{ROOT}/mcp-servers/audio-analyzer")
    from deep_diagnostics import audit_midi_file

    res = audit_midi_file(
        midi_path=midi_path,
        expected_key=expected_key,
        min_pitch=24,
        max_pitch=max_pitch,
        strict_musicality=True,
    )
    if not res.get("gate_passed"):
        raise RuntimeError(
            f"MIDI GATE FAILED ({midi_path}): failures={res.get('musicality_failures')} "
            f"clashes={res.get('minor_2nd_clashes')} sub_collisions={res.get('low_end_simultaneous_collisions')}"
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
) -> None:
    """
    Instantiate calibrated Master + BUS Drums + BUS 808 + BUS Music (4-slot Timbral Glue)
    + FX Room (Dry=0, Dry full=0, Decay <= T_noire).
    """
    t_noire_s = 60.0 / float(bpm)
    rt60 = round(min(0.35, max(0.16, t_noire_s * 0.50)), 3)

    fx_adds = [
        {"tool": "add_plugin", "arguments": {"mixer": 0, "plugin": "gl.eq"}},
        {"tool": "add_plugin", "arguments": {"mixer": 0, "plugin": "gl.comp"}},
        {"tool": "add_plugin", "arguments": {"mixer": 0, "plugin": "gl.limiter"}},
        {"tool": "add_plugin", "arguments": {"mixer": bus_drums, "plugin": "gl.eq"}},
        {"tool": "add_plugin", "arguments": {"mixer": bus_drums, "plugin": "gl.comp"}},
        {"tool": "add_plugin", "arguments": {"mixer": bus_808, "plugin": "gl.stereo"}},
        {"tool": "add_plugin", "arguments": {"mixer": bus_808, "plugin": "gl.comp"}},
        {"tool": "add_plugin", "arguments": {"mixer": bus_808, "plugin": "gl.eq"}},
        {"tool": "add_plugin", "arguments": {"mixer": bus_music, "plugin": "gl.eq"}},
        {"tool": "add_plugin", "arguments": {"mixer": bus_music, "plugin": "gl.dist"}},
        {"tool": "add_plugin", "arguments": {"mixer": bus_music, "plugin": "gl.comp"}},
        {"tool": "add_plugin", "arguments": {"mixer": bus_music, "plugin": "gl.comp"}},
        {"tool": "add_plugin", "arguments": {"mixer": fx_room, "plugin": "gl.reverb"}},
    ]
    mcp.batch(fx_adds)

    changes = [
        # Master
        ("mixer/0/slot/1/Quality", 1),
        ("mixer/0/slot/1/Band 1 type", 1), ("mixer/0/slot/1/Band 1 freq", 28),
        ("mixer/0/slot/1/Band 7 type", 3), ("mixer/0/slot/1/Band 7 freq", 16000),
        ("mixer/0/slot/2/Threshold", -11.5), ("mixer/0/slot/2/Ratio", 1.7),
        ("mixer/0/slot/2/Attack", 0.020), ("mixer/0/slot/2/Release", 0.110),
        ("mixer/0/slot/2/Makeup", 2.0), ("mixer/0/slot/2/Detector high-pass", 95),
        ("mixer/0/slot/3/Gain", limiter_gain_db), ("mixer/0/slot/3/Ceiling", limiter_ceiling_db),
        ("mixer/0/slot/3/Attack (look-ahead)", 0.005), ("mixer/0/slot/3/0.2 dB safety", 1),
        # BUS Drums
        (f"mixer/{bus_drums}/slot/1/Band 1 type", 1), (f"mixer/{bus_drums}/slot/1/Band 1 freq", 34),
        (f"mixer/{bus_drums}/slot/1/Band 7 type", 7), (f"mixer/{bus_drums}/slot/1/Band 7 freq", 10000), (f"mixer/{bus_drums}/slot/1/Band 7 gain", 1.5),
        (f"mixer/{bus_drums}/slot/2/Threshold", -17.0), (f"mixer/{bus_drums}/slot/2/Ratio", 2.6),
        (f"mixer/{bus_drums}/slot/2/Attack", 0.012), (f"mixer/{bus_drums}/slot/2/Release", 0.085), (f"mixer/{bus_drums}/slot/2/Makeup", 3.0),
        # BUS 808
        (f"mixer/{bus_808}/slot/1/Width", 0.0),
        (f"mixer/{bus_808}/slot/2/Detector input", 1),
        (f"mixer/{bus_808}/slot/2/Threshold", -22.0), (f"mixer/{bus_808}/slot/2/Ratio", 5.0),
        (f"mixer/{bus_808}/slot/2/Attack", 0.002), (f"mixer/{bus_808}/slot/2/Release", 0.070), (f"mixer/{bus_808}/slot/2/Lookahead", 2),
        (f"mixer/{bus_808}/slot/3/Band 1 type", 1), (f"mixer/{bus_808}/slot/3/Band 1 freq", 28),
        # BUS Music (4-slot Timbral Glue)
        (f"mixer/{bus_music}/slot/1/Band 1 type", 1), (f"mixer/{bus_music}/slot/1/Band 1 freq", 145),
        (f"mixer/{bus_music}/slot/1/Band 7 type", 3), (f"mixer/{bus_music}/slot/1/Band 7 freq", 12500),
        (f"mixer/{bus_music}/slot/2/Mode", 0), (f"mixer/{bus_music}/slot/2/Drive", 4.0),
        (f"mixer/{bus_music}/slot/2/Tone", 12000), (f"mixer/{bus_music}/slot/2/Mix", 0.08),
        (f"mixer/{bus_music}/slot/2/Output", 0.0), (f"mixer/{bus_music}/slot/2/Oversampling", 2), (f"mixer/{bus_music}/slot/2/Remove DC", 1),
        (f"mixer/{bus_music}/slot/3/Threshold", -19.5), (f"mixer/{bus_music}/slot/3/Ratio", 2.0),
        (f"mixer/{bus_music}/slot/3/Attack", 0.020), (f"mixer/{bus_music}/slot/3/Release", 0.110), (f"mixer/{bus_music}/slot/3/Makeup", 3.5),
        (f"mixer/{bus_music}/slot/4/Detector input", 1),
        (f"mixer/{bus_music}/slot/4/Threshold", -23.0), (f"mixer/{bus_music}/slot/4/Ratio", 2.6),
        (f"mixer/{bus_music}/slot/4/Attack", 0.005), (f"mixer/{bus_music}/slot/4/Release", 0.080), (f"mixer/{bus_music}/slot/4/Lookahead", 1),
        # FX Room (Dry=0, Dry full=0, RT60 <= T_noire)
        (f"mixer/{fx_room}/slot/1/Model", 1),
        (f"mixer/{fx_room}/slot/1/Pre-delay", 10), (f"mixer/{fx_room}/slot/1/Room size", 0.22),
        (f"mixer/{fx_room}/slot/1/Decay time", rt60),
        (f"mixer/{fx_room}/slot/1/Low cut", 500), (f"mixer/{fx_room}/slot/1/High cut", 6400), (f"mixer/{fx_room}/slot/1/High damping", 4000),
        (f"mixer/{fx_room}/slot/1/Dry", 0), (f"mixer/{fx_room}/slot/1/Wet", 0.45), (f"mixer/{fx_room}/slot/1/Width", 0.78),
        (f"mixer/{fx_room}/slot/1/Low cut (full)", 500), (f"mixer/{fx_room}/slot/1/Decay (full)", 2),
        (f"mixer/{fx_room}/slot/1/Dry (full)", 0), (f"mixer/{fx_room}/slot/1/Early reflections (full)", 8),
        (f"mixer/{fx_room}/slot/1/Wet (full)", 45),
    ]
    mcp.set_params(changes)
    mcp.call("edit_mixer_track", {"mixer": fx_room, "name": "FX Room", "routeTo": 0, "volume": fx_room_vol, "pan": 0.0})
