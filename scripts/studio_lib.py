#!/usr/bin/env python3
"""
Shared AI-Studio toolkit for Gridline 0.7.7 orchestration, sample hygiene,
standard bus setup (BUS Drums / BUS 808 / BUS Music / FX Room / Master),
and Top-Down MIDI gate verification.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from gl_dsl import (
    delay_send_fx,
    eq_band as dsl_eq_band,
    eq_params as dsl_eq_params,
    limiter_master,
    reverb_hall,
    reverb_room,
    to_mixer_params,
)

def studio_root() -> str:
    """Universal repo root: `AI_STUDIO_DIR` env var wins, else the directory that contains `scripts/`."""
    env = os.environ.get("AI_STUDIO_DIR", "").strip()
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return str(Path(__file__).resolve().parents[1])


ROOT = studio_root()

SYSTEM_EMPTY_TEMPLATE = "/usr/share/gridline/assets/templates/Basics/Empty with mixer tracks.gridline"
BUNDLED_EMPTY_TEMPLATE = os.path.join(ROOT, "assets", "empty_with_mixer_tracks.gridline")


def resolve_empty_template() -> str:
    """Pick the first available empty-project template: `GRIDLINE_EMPTY_TEMPLATE` env > system Gridline > bundled fallback."""
    candidates = [
        os.environ.get("GRIDLINE_EMPTY_TEMPLATE", "").strip(),
        SYSTEM_EMPTY_TEMPLATE,
        BUNDLED_EMPTY_TEMPLATE,
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    return BUNDLED_EMPTY_TEMPLATE


TEMPLATE_EMPTY_MIXER = resolve_empty_template()


class GridlineMCP:
    """Persistent stdio JSON-RPC client for `gridline mcp <proj> --export-dir <dir>`."""

    def __init__(self, proj_path: str, export_dir: str, client_name: str = "ai_studio", rpc_timeout_s: float = 180.0):
        os.makedirs(export_dir, exist_ok=True)
        self.rpc_timeout_s = float(rpc_timeout_s)
        self.proc = subprocess.Popen(
            ["gridline", "mcp", proj_path, "--export-dir", export_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._id = 0
        self._rbuf = b""
        self._stderr_lines: list[str] = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": client_name, "version": "1.0"},
            },
        )
        self._notify("notifications/initialized", {})

    def _drain_stderr(self) -> None:
        """Continuously drain the child stderr pipe so it can never deadlock on a full buffer."""
        proc_stderr = self.proc.stderr
        if proc_stderr is None:
            return
        try:
            for line in proc_stderr:
                self._stderr_lines.append(line.rstrip("\n"))
                del self._stderr_lines[:-200]
        except Exception:
            pass

    def _readline_timeout(self, timeout_s: float) -> str | None:
        """Read one line from child stdout with a hard timeout (select on the raw fd). Returns None on EOF."""
        assert self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        deadline = time.monotonic() + float(timeout_s)
        while True:
            if b"\n" in self._rbuf:
                line, self._rbuf = self._rbuf.split(b"\n", 1)
                return line.decode("utf-8", "replace")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"MCP RPC timeout after {timeout_s}s (waiting for response)")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                return None
            self._rbuf += chunk

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
            line = self._readline_timeout(self.rpc_timeout_s)
            if line is None or not line.strip():
                err = "\n".join(self._stderr_lines[-20:])
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

    # --- In-session audition (no .wav render needed) ---

    def analyze_loudness(self, scope: str = "pattern", **kw: Any) -> dict[str, Any]:
        """In-session loudness probe (pattern/playlist/song) straight from the live Gridline session."""
        args: dict[str, Any] = {"scope": scope}
        args.update(kw)
        return self.call("analyze_loudness", args)

    def play_transport(self, action: str = "play") -> bool:
        """Best-effort transport control (play/stop/pause) on the live session. Returns False if unsupported."""
        for tool in (action, "transport", "set_playing"):
            try:
                self.call(tool, {"action": action} if tool != action else {})
                return True
            except Exception:
                continue
        return False

    def set_channel_muted(self, channel: int, muted: bool) -> None:
        self.call("edit_channel", {"channel": int(channel), "muted": bool(muted)})

    def solo_channel(self, channel: int, channels: list[int] | None = None) -> list[int]:
        """Mute every channel except `channel` (in-session solo). Returns the previously-muted list."""
        if channels is None:
            channels = [int(c.get("channel", i)) for i, c in enumerate(self.call("list_channels", {}).get("channels", []), start=1)]
        muted: list[int] = []
        for ch in channels:
            if int(ch) != int(channel):
                self.set_channel_muted(ch, True)
                muted.append(int(ch))
        self.set_channel_muted(channel, False)
        return muted

    def unsolo(self, muted: list[int], keep_solo: int | None = None) -> None:
        for ch in muted:
            self.set_channel_muted(ch, False)
        if keep_solo is not None:
            self.set_channel_muted(keep_solo, False)

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        for stream in (self.proc.stdout, self.proc.stderr):
            if stream:
                try:
                    stream.close()
                except Exception:
                    pass
        self._stderr_thread.join(timeout=2)


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
    fx_delay: int = 21,
    fx_hall: int = 22,
    limiter_ceiling_db: float | None = None,
    limiter_gain_db: float | None = None,
    fx_room_vol: float = 0.30,
    fx_delay_vol: float = 0.32,
    fx_hall_vol: float = 0.30,
    profile: str = "trap_rap",
    valence: float = 0.0,
    apply_dark_mid_dip: bool = True,
) -> None:
    """
    Instantiate calibrated Master + BUS Drums + BUS 808/Bass + BUS Music (Timbral Glue)
    + FX Room (Dry=0, Dry full=0, Decay <= T_noire) + FX Delay (21) + FX Hall (22),
    adapted to `profile` and `valence`:
    - BUS Music places `gl.dist` (Slot 1, Tone <= 7500 Hz) BEFORE `gl.eq` (Slot 2, HPF + LPF)
      so any saturation harmonics above LPF are cleanly filtered out.
    - trap_rap: smooth Kick->808 sidechain (Ratio 2.4, Threshold -18 dB, Release 65 ms) to preserve 808 body,
      gentle soft saturation, drums air (unless valence < -0.2)
    - edm_club: punchy sidechain (5:1, -20 dB, 85 ms), gentle soft saturation, drums air (unless valence < -0.2)
    - pop_rnb / rock_metal: balanced glue, moderate sidechain (3:1)
    - lofi_chill: warmer LPF on BUS Music, soft glue, light drums air
    - jazz_acoustic / classical_ambient: clean transparent bus (no distortion, gentle/minimal comp, larger room decay)
    """
    from mix_config import resolve_limiter_params
    limiter_ceiling_db, limiter_gain_db = resolve_limiter_params(limiter_ceiling_db, limiter_gain_db)
    t_noire_s = 60.0 / float(bpm)
    acoustic_mode = profile in ("jazz_acoustic", "classical_ambient")
    lofi_mode = profile == "lofi_chill"
    dark_mode = valence < -0.2
    rt60 = round(min(1.10 if acoustic_mode else 0.42, max(0.18, t_noire_s * (1.2 if acoustic_mode else 0.65))), 3)
    hall_decay = 2.3 if acoustic_mode else 1.8

    # Ensure mixer tracks up to max bus index exist (Empty template only has 0..16)
    mixer_info = mcp.call("get_mixer", {})
    existing_count = len(mixer_info.get("tracks", []))
    max_needed = max(bus_drums, bus_808, bus_music, fx_room, fx_delay, fx_hall)
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
        # BUS Music mid/high stereo widener (keeps < 250 Hz mono, symmetric => no L/R drift)
        {"tool": "add_plugin", "arguments": {"mixer": bus_music, "plugin": "gl.stereo"}},
        {"tool": "add_plugin", "arguments": {"mixer": bus_music, "plugin": "gl.stereo"}},
        {"tool": "add_plugin", "arguments": {"mixer": fx_room, "plugin": "gl.reverb"}},
        # FX Delay (21): gl.delay -> gl.eq (wet-only, band-limited)
        {"tool": "add_plugin", "arguments": {"mixer": fx_delay, "plugin": "gl.delay"}},
        {"tool": "add_plugin", "arguments": {"mixer": fx_delay, "plugin": "gl.eq"}},
        # FX Hall (22): gl.reverb -> gl.eq (wet-only, band-limited)
        {"tool": "add_plugin", "arguments": {"mixer": fx_hall, "plugin": "gl.reverb"}},
        {"tool": "add_plugin", "arguments": {"mixer": fx_hall, "plugin": "gl.eq"}},
    ]
    mcp.batch(fx_adds)

    music_hpf = 90 if acoustic_mode else (110 if lofi_mode else 135)
    if lofi_mode:
        music_lpf = 9500 if dark_mode else 10500
    elif acoustic_mode:
        music_lpf = 12500 if dark_mode else 14000
    else:
        # Q3.3 anti-over-filtering: dark trap/pop must keep BUS Music LPF >= 12 kHz (natural air)
        music_lpf = 12000 if (dark_mode and profile in ("trap_rap", "pop_rnb")) else (11500 if dark_mode else 13500)
    # Keep saturation subtle: a loud gl.dist reads as "electronic"; samples already carry character.
    music_dist_mix = 0.0 if acoustic_mode else (0.06 if lofi_mode else 0.05)
    music_dist_tone = 6500 if (lofi_mode or dark_mode) else 7500
    drums_air_gain = 0.0 if (acoustic_mode or lofi_mode or dark_mode) else 1.2
    if acoustic_mode:
        sc_ratio_bass, sc_thresh_bass, sc_release_bass = 1.5, -20.0, 0.085
    elif profile == "trap_rap":
        sc_ratio_bass, sc_thresh_bass, sc_release_bass = 2.4, -18.0, 0.065
    elif profile == "edm_club":
        sc_ratio_bass, sc_thresh_bass, sc_release_bass = 5.0, -20.0, 0.085
    else:
        sc_ratio_bass, sc_thresh_bass, sc_release_bass = 3.0, -20.0, 0.085
    sc_ratio_music = 1.2 if acoustic_mode else 2.2
    # Gentler master bus: less compression/limiting = more natural dynamics (less "electronic" squash).
    master_comp_thresh = -9.5 if acoustic_mode else -12.0
    master_comp_makeup = 0.3 if acoustic_mode else 0.8

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
        (f"mixer/{bus_music}/slot/1/Mode", 0), (f"mixer/{bus_music}/slot/1/Drive", 2.5),
        (f"mixer/{bus_music}/slot/1/Tone", music_dist_tone), (f"mixer/{bus_music}/slot/1/Mix", music_dist_mix),
        (f"mixer/{bus_music}/slot/1/Output", 0.0), (f"mixer/{bus_music}/slot/1/Oversampling", 2), (f"mixer/{bus_music}/slot/1/Remove DC", 1),
        *dsl_eq_params(
            bus_music,
            [
                dsl_eq_band(1, "hpf", music_hpf),
                dsl_eq_band(5, "high_shelf", 9500.0, 0.8 if (acoustic_mode or lofi_mode) else 1.2),
                dsl_eq_band(7, "lpf", music_lpf),
            ],
            slot=2,
        ),
    ]
    if dark_mode and apply_dark_mid_dip:
        changes.extend(to_mixer_params(bus_music, 2, dsl_eq_band(6, "peak", 3200, -1.5, q=1.4)))
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
        # Q3.2 default stereo width (legacy engine = real side synthesis, symmetric)
        (f"mixer/{bus_music}/slot/5/Reference engine", 0),
        (f"mixer/{bus_music}/slot/5/Width", 1.60 if acoustic_mode else 2.00),
        (f"mixer/{bus_music}/slot/5/Side level", 10.0 if acoustic_mode else 12.0),
        (f"mixer/{bus_music}/slot/5/Mono below", 250.0),
        (f"mixer/{bus_music}/slot/5/Dephase", 0.0),
        # Second gentle stage: lifts very mono/synth-only arrangements into the Q3.2 width window
        (f"mixer/{bus_music}/slot/6/Reference engine", 0),
        (f"mixer/{bus_music}/slot/6/Width", 1.10),
        (f"mixer/{bus_music}/slot/6/Side level", 4.0),
        (f"mixer/{bus_music}/slot/6/Mono below", 250.0),
        (f"mixer/{bus_music}/slot/6/Dephase", 0.0),
        # FX Room (Dry=0, Dry full=0, Decay <= T_noire for rhythmic genres)
        *to_mixer_params(fx_room, 1, fx_room_params),
        # FX Delay (21) — BPM-synced, Dry=0, band-limited feedback
        *to_mixer_params(fx_delay, 1, delay_send_fx(bpm=bpm, valence=valence)),
        *dsl_eq_params(
            fx_delay,
            [dsl_eq_band(1, "hpf", 400.0), dsl_eq_band(7, "lpf", 2800.0 if dark_mode else 3500.0)],
            slot=2,
            quality=0,
        ),
        # FX Hall (22) — long decay, Dry=0, band-limited
        *to_mixer_params(fx_hall, 1, reverb_hall(decay_sec=hall_decay, valence=valence)),
        *dsl_eq_params(
            fx_hall,
            [dsl_eq_band(1, "hpf", 500.0), dsl_eq_band(7, "lpf", 4500.0 if dark_mode else 5500.0)],
            slot=2,
            quality=0,
        ),
    ])
    mcp.set_params(changes)
    mcp.batch([
        {"tool": "edit_mixer_track", "arguments": {"mixer": bus_drums, "name": "BUS Drums", "routeTo": 0, "volume": 1.0, "pan": 0.0}},
        {"tool": "edit_mixer_track", "arguments": {"mixer": bus_808, "name": "BUS Bass", "routeTo": 0, "volume": 1.0, "pan": 0.0}},
        {"tool": "edit_mixer_track", "arguments": {"mixer": bus_music, "name": "BUS Music", "routeTo": 0, "volume": 1.0, "pan": 0.0}},
        {"tool": "edit_mixer_track", "arguments": {"mixer": fx_room, "name": "FX Room", "routeTo": 0, "volume": fx_room_vol, "pan": 0.0}},
        {"tool": "edit_mixer_track", "arguments": {"mixer": fx_delay, "name": "FX Delay", "routeTo": 0, "volume": fx_delay_vol, "pan": 0.0}},
        {"tool": "edit_mixer_track", "arguments": {"mixer": fx_hall, "name": "FX Hall", "routeTo": 0, "volume": fx_hall_vol, "pan": 0.0}},
    ])
