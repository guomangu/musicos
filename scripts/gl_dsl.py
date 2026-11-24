#!/usr/bin/env python3
"""
Semantic Gridline 0.7.7 DSL (`gl_dsl.py`).

Encapsulates 100% of human-unit -> Gridline 0.7.7 parameter conversions so that
neither LLM prompts nor song scripts ever have to manually recall low-level
enums (`Peak = 6` vs `Notch = 4`), `Band N width` vs legacy `Q`, `gl.reverb`
deciseconds/hHz/k128 units, normalized automation scaling, or per-synth ADSR
and cutoff units.
"""
from __future__ import annotations

import json
import os

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

EqBandKind = Literal[
    "off",
    "hpf",
    "lpf",
    "peak",
    "low_shelf",
    "high_shelf",
    "notch",
    "bandpass",
]

SynthRole = Literal["pad", "chords", "lead", "pluck", "bell", "bass"]

EQ_KIND_TO_INT: dict[str, int] = {
    "off": 0,
    "hpf": 1,
    "low_cut": 1,
    "bandpass": 2,
    "band_pass": 2,
    "lpf": 3,
    "high_cut": 3,
    "notch": 4,
    "low_shelf": 5,
    "peak": 6,
    "high_shelf": 7,
}

INT_TO_EQ_KIND: dict[int, str] = {
    0: "off",
    1: "hpf",
    2: "bandpass",
    3: "lpf",
    4: "notch",
    5: "low_shelf",
    6: "peak",
    7: "high_shelf",
}

# Verified monotone calibration anchors for `gl.eq` in Reference mode (`Band model = 1`):
# Q=0.707 -> 0.6122 (Gridline default), Q=1.2 -> 0.45, Q=1.4 -> 0.39, Q=1.6 -> 0.35, Q=2.8 -> 0.22.
_Q_WIDTH_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.35, 0.88),
    (0.707, 0.6122),
    (1.2, 0.45),
    (1.4, 0.39),
    (1.6, 0.35),
    (2.8, 0.22),
    (6.0, 0.11),
    (12.0, 0.05),
)


def clamp(val: float, lo: float, hi: float) -> float:
    """Clamp numeric value to `[lo, hi]`."""
    return float(max(lo, min(hi, val)))


def q_to_width(q: float) -> float:
    """
    Convert an acoustic Q factor to Gridline `gl.eq` `Band N width` (`Band model = 1`).
    Strictly monotone decreasing and clamped to `[0.05, 0.95]`.
    """
    q_val = float(q)
    if q_val <= 0.0:
        raise ValueError(f"Q factor must be > 0, got {q}")
    if q_val <= _Q_WIDTH_ANCHORS[0][0]:
        return clamp(_Q_WIDTH_ANCHORS[0][1] + 0.08 * math.log(_Q_WIDTH_ANCHORS[0][0] / q_val), 0.05, 0.95)
    if q_val >= _Q_WIDTH_ANCHORS[-1][0]:
        return 0.05
    log_q = math.log(q_val)
    for (q0, w0), (q1, w1) in zip(_Q_WIDTH_ANCHORS[:-1], _Q_WIDTH_ANCHORS[1:]):
        if q0 <= q_val <= q1:
            t = (log_q - math.log(q0)) / (math.log(q1) - math.log(q0))
            return round(clamp(w0 + t * (w1 - w0), 0.05, 0.95), 4)
    return 0.39


def eq_band(
    band_idx: int,
    kind: EqBandKind,
    freq_hz: float,
    gain_db: float = 0.0,
    q: float = 1.4,
) -> dict[str, Any]:
    """
    Build a Gridline `gl.eq` band parameter dict using human units.
    Never emits `Band N Q (legacy)`; always uses `Band N width` (`Band model = 1`).
    """
    idx = int(band_idx)
    if not (1 <= idx <= 7):
        raise ValueError(f"gl.eq band_idx must be in 1..7, got {band_idx}")
    k_norm = str(kind).strip().lower()
    if k_norm not in EQ_KIND_TO_INT:
        raise ValueError(f"Unknown eq_band kind {kind!r}; expected one of {sorted(EQ_KIND_TO_INT)}")

    btype = EQ_KIND_TO_INT[k_norm]
    freq = round(clamp(float(freq_hz), 20.0, 20000.0), 2)
    out: dict[str, Any] = {
        f"Band {idx} type": btype,
        f"Band {idx} freq": freq,
    }
    if btype in (5, 6, 7):  # low_shelf, peak, high_shelf
        out[f"Band {idx} gain"] = round(clamp(float(gain_db), -18.0, 18.0), 2)
    if btype in (2, 6):  # bandpass, peak
        out[f"Band {idx} width"] = q_to_width(q)
    return out


def to_mixer_params(mixer: int, slot: int, params: dict[str, Any]) -> list[tuple[str, Any]]:
    """Convert a plugin parameter dict into `(mixer/M/slot/S/Name, value)` tuples."""
    mx = int(mixer)
    sl = int(slot)
    out: list[tuple[str, Any]] = []
    for k, v in params.items():
        if "Q (legacy)" in k:
            raise ValueError(f"Forbidden legacy EQ parameter emitted: {k}")
        out.append((f"mixer/{mx}/slot/{sl}/{k}", v))
    return out


def eq_params(
    mixer: int,
    bands: list[dict[str, Any]],
    slot: int = 1,
    quality: int | None = None,
) -> list[tuple[str, Any]]:
    """Combine multiple `eq_band(...)` dicts into mixer slot parameter tuples."""
    merged: dict[str, Any] = {}
    if quality is not None:
        merged["Quality"] = int(clamp(float(quality), 0, 1))
    for b in bands:
        merged.update(b)
    return to_mixer_params(mixer, slot, merged)


def reverb_room(
    decay_sec: float,
    hpf_hz: float = 400.0,
    lpf_hz: float = 6800.0,
    wet_pct: float = 40.0,
    er_pct: float = 12.0,
    pre_delay_ms: float = 12.0,
    room_size: float = 0.25,
) -> dict[str, Any]:
    """
    Build calibrated `gl.reverb` parameters (`Model = 1`, `Dry = 0`, `Dry (full) = 0`),
    converting decay to deciseconds `[1..200]`, LPF to hHz `[5..221]`, and wet/ER to k128.
    """
    decay_ds = int(round(clamp(float(decay_sec) * 10.0, 1.0, 200.0)))
    decay_s_clamped = round(decay_ds / 10.0, 2)
    hpf = round(clamp(float(hpf_hz), 20.0, 3000.0), 1)
    lpf_hhz = int(round(clamp(float(lpf_hz) / 100.0, 5.0, 221.0)))
    lpf_actual_hz = float(lpf_hhz * 100)
    wet_k128 = int(round(clamp(float(wet_pct) * 128.0 / 100.0, 0.0, 160.0)))
    er_k128 = int(round(clamp(float(er_pct) * 128.0 / 100.0, 0.0, 160.0)))
    wet_norm = round(clamp(float(wet_pct) / 100.0, 0.0, 1.0), 4)
    er_norm = round(clamp(float(er_pct) / 100.0, 0.0, 1.0), 4)
    pre_ms = round(clamp(float(pre_delay_ms), 0.0, 384.0), 1)
    rsize_norm = round(clamp(float(room_size), 0.0, 1.0), 3)
    rsize_pct = int(round(clamp(float(room_size) * 100.0, 1.0, 100.0)))

    return {
        "Model": 1,
        "Pre-delay": pre_ms,
        "Pre-delay (full)": int(round(pre_ms)),
        "Room size": rsize_norm,
        "Room size (full)": rsize_pct,
        "Decay time": decay_s_clamped,
        "Decay (full)": decay_ds,
        "Low cut": hpf,
        "Low cut (full)": hpf,
        "High cut": lpf_actual_hz,
        "High cut (full)": lpf_hhz,
        "Dry": 0,
        "Dry (full)": 0,
        "Wet": wet_norm,
        "Wet (full)": wet_k128,
        "Early reflections": er_norm,
        "Early reflections (full)": er_k128,
    }


# Gridline `gl.delay` (Tracer) tempo-synced divisions (verified via `gridline plugin-params`).
_DELAY_TIME_INDEX: dict[str, int] = {
    "1/32": 0,
    "1/16T": 1,
    "1/16.": 2,
    "1/16": 3,
    "1/8T": 4,
    "1/16..": 5,
    "1/8": 6,
    "1/4T": 7,
    "1/8.": 8,
    "1/4": 9,
    "1/4.": 10,
    "1/2": 11,
    "1/64": 12,
    "1/2T": 13,
    "1/2.": 14,
    "1 bar": 15,
}


def delay_send_fx(
    bpm: float = 140.0,
    valence: float = 0.0,
    *,
    division: str = "1/8.",
    feedback_pct: float = 26.0,
    wet_pct: float = 55.0,
    ping_pong: bool = True,
    hpf_hz: float = 400.0,
    lpf_hz: float | None = None,
) -> dict[str, Any]:
    """
    Build calibrated `gl.delay` (Tracer) parameters for the dedicated `FX Delay` bus:
    BPM-synced division (dotted eighth by default), `Feedback 20-32%`, `Dry = 0`,
    band-limited feedback (`HPF 400 Hz`, `LPF 3500 Hz` / `2800 Hz` when `V < -0.2`).
    """
    div = str(division).strip()
    if div not in _DELAY_TIME_INDEX:
        raise ValueError(f"Unknown delay division {division!r}; expected one of {sorted(_DELAY_TIME_INDEX)}")
    dark = float(valence) < -0.2
    eff_lpf = float(lpf_hz) if lpf_hz is not None else (2800.0 if dark else 3500.0)
    fb = clamp(float(feedback_pct) / 100.0, 0.0, 0.32)
    wet = clamp(float(wet_pct) / 100.0, 0.0, 1.0)
    return {
        "Tempo sync": 1,
        "Time": _DELAY_TIME_INDEX[div],
        "Feedback": round(fb, 4),
        "Feedback mode": 1 if ping_pong else 0,
        "Feedback cutoff": round(clamp(eff_lpf, 300.0, 20000.0), 1),
        "Feedback low cut": round(clamp(float(hpf_hz), 20.0, 2000.0), 1),
        "Feedback filter": 0,
        "Mod rate": 0.45,
        "Mod depth": 0.0,
        "Saturation": 0.0,
        "Distortion": 0,
        "Dry": 0.0,
        "Wet": round(wet, 4),
        "Delay model": 1,
        "Delay type": 1,
        "Stereo spread": 1.0,
        "Output tone": 0.0,
    }


def reverb_hall(
    decay_sec: float = 1.8,
    *,
    hpf_hz: float = 500.0,
    lpf_hz: float = 5500.0,
    wet_pct: float = 48.0,
    er_pct: float = 14.0,
    pre_delay_ms: float = 18.0,
    room_size: float = 0.62,
    valence: float = 0.0,
) -> dict[str, Any]:
    """
    Long `gl.reverb` Hall (`Model = 0`) for the dedicated `FX Hall` bus:
    `Decay 1.6-2.4 s`, `Dry = 0`, `Dry (full) = 0`, `HPF 500 Hz`,
    `LPF 5500 Hz` (`4500 Hz` when `V < -0.2`).
    """
    dark = float(valence) < -0.2
    eff_lpf = 4500.0 if dark else float(lpf_hz)
    params = reverb_room(
        decay_sec=clamp(float(decay_sec), 1.6, 2.4),
        hpf_hz=clamp(float(hpf_hz), 20.0, 3000.0),
        lpf_hz=clamp(eff_lpf, 500.0, 22100.0),
        wet_pct=wet_pct,
        er_pct=er_pct,
        pre_delay_ms=pre_delay_ms,
        room_size=room_size,
    )
    params["Model"] = 0  # Hall
    params["High damping"] = 5200 if dark else 6800
    params["Width"] = 0.92
    return params


def lpf_makeup_gain_db(cutoff_hz: float) -> float:
    """Compensate the loudness loss of `synth_voice` low-pass filtering (Phase 8 Q3.3)."""
    c = float(cutoff_hz)
    if c < 3500.0:
        return 2.5
    if c < 5500.0:
        return 1.5
    return 0.0


def limiter_master(gain_db: float = 0.0, ceiling_db: float = -2.2) -> dict[str, Any]:
    """
    Build `gl.limiter` parameters for Master (`Attack (look-ahead) = 0.002`, `0.2 dB safety = 1`).
    Emits a `UserWarning` if `ceiling_db > -2.0`.
    """
    ceil = float(ceiling_db)
    if ceil > -2.0:
        warnings.warn(
            f"limiter_master ceiling_db={ceil:+.2f} dB exceeds recommended <= -2.0 dB threshold; "
            "risk of True Peak > -1.0 dBTP.",
            UserWarning,
            stacklevel=2,
        )
    return {
        "Gain": round(clamp(float(gain_db), -36.0, 24.0), 2),
        "Ceiling": round(clamp(ceil, -36.0, 12.0), 2),
        "Attack (look-ahead)": 0.002,
        "0.2 dB safety": 1,
    }


def sidechain_send(src_mixer: int, dst_mixer: int, amount: float = 1.0) -> dict[str, Any]:
    """
    Build safe `set_send` arguments forcing `sidechain=True` and `preFader=True`
    to prevent raw Kick audio from bleeding into sidechained buses.
    """
    return {
        "mixer": int(src_mixer),
        "to": int(dst_mixer),
        "level": round(clamp(float(amount), 0.0, 1.25), 4),
        "sidechain": True,
        "preFader": True,
    }


def mixer_gain_to_auto(gain_linear: float) -> float:
    """
    Convert Gridline linear mixer fader value (`0.0..1.25`, default unity `1.0`,
    Gridline default `0.8`) into normalized `[0.0, 1.0]` for `add_automation`.
    """
    return round(clamp(float(gain_linear) / 1.25, 0.0, 1.0), 5)


def db_to_auto(db: float, ref_fader: float = 1.0) -> float:
    """
    Convert a gain offset in dB (relative to `ref_fader=1.0`, matching
    `setup_standard_buses` bus tracks) into normalized `[0.0, 1.0]` for
    Gridline `add_automation`.
    Gridline's mixer volume follows a quadratic taper (`gain_db = 40 * log10(fader / ref_fader)`),
    so `fader = ref_fader * 10**(db / 40)` yields exact acoustic dB changes on render.
    """
    db_f = float(db)
    if db_f <= -90.0:
        return 0.0
    fader = float(ref_fader) * (10.0 ** (db_f / 40.0))
    return mixer_gain_to_auto(fader)


def eq_gain_to_auto(gain_db: float) -> float:
    """
    Convert a `gl.eq` Band gain in dB (`[-18.0, +18.0]`) into normalized `[0.0, 1.0]`
    for Gridline `add_automation` (`0.0 dB` -> `0.5`).
    """
    return round(clamp((float(gain_db) + 18.0) / 36.0, 0.0, 1.0), 5)


# ---------------------------------------------------------------------------
# Typed Registry of Safe Native Gridline Synths (`synth_voice`)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SynthSpec:
    plugin: str
    adsr_unit: Literal["seconds", "normalized", "fixed"]
    adsr_params: dict[str, str]  # keys: attack, decay, sustain, release (optional)
    adsr_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    cutoff_param: str | None = None
    cutoff_unit: Literal["hz", "norm", "brightness", "none"] = "none"
    cutoff_bounds: tuple[float, float] = (20.0, 20000.0)
    polyphony_param: str | None = None
    polyphony_mode: Literal["int", "zero_based", "sawswarm_enum", "none"] = "none"
    unison_param: str | tuple[str, ...] | None = None
    unison_mode: Literal["int", "zero_based", "none"] = "none"
    glide_param: str | None = None
    glide_unit: Literal["seconds", "normalized", "none"] = "none"
    extra_defaults: dict[str, Any] = field(default_factory=dict)
    has_internal_lpf: bool = True


SYNTH_REGISTRY: dict[str, SynthSpec] = {
    "gl.analog": SynthSpec(
        plugin="gl.analog",
        adsr_unit="seconds",
        adsr_params={"attack": "Amp attack", "decay": "Amp decay", "sustain": "Amp sustain", "release": "Amp release"},
        adsr_bounds={"attack": (0.001, 10.0), "decay": (0.005, 10.0), "sustain": (0.0, 1.0), "release": (0.005, 15.0)},
        cutoff_param="Cutoff",
        cutoff_unit="hz",
        cutoff_bounds=(20.0, 20000.0),
        polyphony_param="Polyphony",
        polyphony_mode="int",
        unison_param="Unison voices",
        unison_mode="int",
        glide_param="Glide",
        glide_unit="seconds",
        extra_defaults={"Filter mode": 1, "Resonance": 0.12, "Chorus": 0, "Delay": 0, "Reverb": 0},
        has_internal_lpf=True,
    ),
    "gl.wavetable": SynthSpec(
        plugin="gl.wavetable",
        adsr_unit="seconds",
        adsr_params={"attack": "Amp env attack", "decay": "Amp env decay", "sustain": "Amp env sustain", "release": "Amp env release"},
        adsr_bounds={"attack": (0.001, 10.0), "decay": (0.005, 10.0), "sustain": (0.0, 1.0), "release": (0.005, 15.0)},
        cutoff_param="Filter 1 cutoff",
        cutoff_unit="hz",
        cutoff_bounds=(20.0, 20000.0),
        polyphony_param="Polyphony",
        polyphony_mode="int",
        unison_param=("Osc A unison", "Osc B unison"),
        unison_mode="int",
        glide_param="Glide",
        glide_unit="seconds",
        extra_defaults={"Filter 1 on": 1, "Filter 1 type": 1, "Filter 1 resonance": 0.12, "Chorus": 0, "Delay": 0, "Reverb": 0},
        has_internal_lpf=True,
    ),
    "gl.warmsynth": SynthSpec(
        plugin="gl.warmsynth",
        adsr_unit="normalized",
        adsr_params={"attack": "Attack", "decay": "Decay", "sustain": "Sustain", "release": "Release"},
        cutoff_param="Cutoff",
        cutoff_unit="norm",
        cutoff_bounds=(0.0, 1.0),
        extra_defaults={"Filter": 1, "Filter high": 0, "Filter band": 0, "Emphasis": 0.08, "Chorus": 0},
        has_internal_lpf=True,
    ),
    "gl.dcosynth": SynthSpec(
        plugin="gl.dcosynth",
        adsr_unit="normalized",
        adsr_params={"attack": "Env 1 attack", "decay": "Env 1 decay", "sustain": "Env 1 sustain", "release": "Env 1 release"},
        cutoff_param="Cutoff",
        cutoff_unit="norm",
        cutoff_bounds=(0.0, 1.0),
        glide_param="Portamento",
        glide_unit="normalized",
        extra_defaults={"VCA": 0, "Slope": 3, "Resonance": 0.10, "Chorus": 0, "Delay": 0, "Reverb": 0},
        has_internal_lpf=True,
    ),
    "gl.twinva": SynthSpec(
        plugin="gl.twinva",
        adsr_unit="normalized",
        adsr_params={"attack": "Amp attack", "decay": "Amp decay", "sustain": "Amp sustain", "release": "Amp release"},
        cutoff_param="Cutoff",
        cutoff_unit="norm",
        cutoff_bounds=(0.0, 1.0),
        polyphony_param="Polyphony",
        polyphony_mode="zero_based",
        unison_param="Unison",
        unison_mode="zero_based",
        glide_param="Glide",
        glide_unit="normalized",
        extra_defaults={"Filter mode": 0, "Resonance": 0.10, "Chorus": 0, "Delay": 0},
        has_internal_lpf=True,
    ),
    "gl.sawswarm": SynthSpec(
        plugin="gl.sawswarm",
        adsr_unit="normalized",
        adsr_params={"attack": "Attack", "decay": "Decay", "sustain": "Sustain", "release": "Release"},
        cutoff_param="Cutoff",
        cutoff_unit="norm",
        cutoff_bounds=(0.0, 1.0),
        polyphony_param="Polyphony",
        polyphony_mode="sawswarm_enum",
        unison_param="Unison",
        unison_mode="zero_based",
        glide_param="Glide",
        glide_unit="normalized",
        extra_defaults={"Filter mode": 0, "Resonance": 0.10, "Chorus": 0, "Phaser": 0, "Delay": 0, "Reverb": 0},
        has_internal_lpf=True,
    ),
    "gl.string": SynthSpec(
        plugin="gl.string",
        adsr_unit="seconds",
        adsr_params={"attack": "Attack", "decay": "Decay", "sustain": "Sustain", "release": "Release"},
        adsr_bounds={"attack": (0.001, 5.0), "decay": (0.1, 30.0), "sustain": (0.0, 1.0), "release": (0.01, 10.0)},
        cutoff_param="Hi cut",
        cutoff_unit="hz",
        cutoff_bounds=(200.0, 20000.0),
        polyphony_param="Voices",
        polyphony_mode="int",
        extra_defaults={"Chorus": 0, "Delay": 0, "Reverb": 0},
        has_internal_lpf=True,
    ),
    "gl.pluck": SynthSpec(
        plugin="gl.pluck",
        adsr_unit="seconds",
        adsr_params={"decay": "Decay", "release": "Release"},
        adsr_bounds={"decay": (0.1, 20.0), "release": (0.005, 2.0)},
        cutoff_param="Brightness",
        cutoff_unit="brightness",
        cutoff_bounds=(0.0, 1.0),
        extra_defaults={"Damping": 0.42, "Gate (damp on release)": 1},
        has_internal_lpf=True,
    ),
    "gl.fm": SynthSpec(
        plugin="gl.fm",
        adsr_unit="seconds",
        adsr_params={"attack": "Op 1 attack", "decay": "Op 1 decay", "sustain": "Op 1 sustain", "release": "Op 1 release"},
        adsr_bounds={"attack": (0.001, 5.0), "decay": (0.005, 10.0), "sustain": (0.0, 1.0), "release": (0.005, 10.0)},
        polyphony_param="Voices",
        polyphony_mode="int",
        glide_param="Glide",
        glide_unit="seconds",
        has_internal_lpf=False,
    ),
    "gl.matrixfm": SynthSpec(
        plugin="gl.matrixfm",
        adsr_unit="fixed",
        adsr_params={},
        polyphony_param="Polyphony",
        polyphony_mode="int",
        unison_param="Unison voices",
        unison_mode="int",
        has_internal_lpf=False,
    ),
    "gl.keys": SynthSpec(
        plugin="gl.keys",
        adsr_unit="seconds",
        adsr_params={"decay": "Decay", "release": "Release"},
        adsr_bounds={"decay": (0.3, 12.0), "release": (0.02, 3.0)},
        cutoff_param="Muffle",
        cutoff_unit="hz",
        cutoff_bounds=(500.0, 20000.0),
        has_internal_lpf=False,  # Also attach clean gl.eq LPF on insert per spec
    ),
    "gl.fingerbass": SynthSpec(
        plugin="gl.fingerbass",
        adsr_unit="fixed",
        adsr_params={},
        extra_defaults={"Treble": 0.35},
        has_internal_lpf=False,
    ),
    "gl.sampler": SynthSpec(
        plugin="gl.sampler",
        adsr_unit="seconds",
        adsr_params={"attack": "Attack", "decay": "Decay", "sustain": "Sustain", "release": "Release"},
        adsr_bounds={"attack": (0.0005, 5.0), "decay": (0.005, 10.0), "sustain": (0.0, 1.0), "release": (0.002, 8.0)},
        cutoff_param="Cutoff",
        cutoff_unit="hz",
        cutoff_bounds=(20.0, 20000.0),
        polyphony_param="Voices",
        polyphony_mode="int",
        glide_param="Glide",
        glide_unit="seconds",
        extra_defaults={"Filter": 1, "Resonance": 0.08, "Declick": 2, "Remove DC": 1},
        has_internal_lpf=True,
    ),
}

_ROLE_ADSR_DEFAULTS: dict[str, dict[str, float]] = {
    "pad":    {"attack_ms": 240.0, "decay_ms": 800.0, "sustain_pct": 80.0, "release_ms": 580.0},
    "chords": {"attack_ms": 22.0,  "decay_ms": 600.0, "sustain_pct": 65.0, "release_ms": 340.0},
    "lead":   {"attack_ms": 18.0,  "decay_ms": 450.0, "sustain_pct": 70.0, "release_ms": 140.0},
    "pluck":  {"attack_ms": 4.0,   "decay_ms": 380.0, "sustain_pct": 15.0, "release_ms": 160.0},
    "bell":   {"attack_ms": 3.0,   "decay_ms": 500.0, "sustain_pct": 10.0, "release_ms": 280.0},
    "bass":   {"attack_ms": 4.0,   "decay_ms": 350.0, "sustain_pct": 85.0, "release_ms": 50.0},
}


def _default_cutoff_hz(role: str, valence: float) -> float:
    """v3.0: source cutoffs with anti brick-wall floors from `config/mix_profiles.json`
    (chords/pad >= 6.5 kHz, lead/pluck >= 9 kHz). Dark mode (V < -0.2) is expressed by a
    gentle high-shelf on BUS Music (`dark_high_shelf_db` @ 5.5 kHz), NOT by a hard 3 kHz
    source LPF that would make the mix hollow ("mix en U")."""
    try:
        from mix_config import source_lpf_floor_hz
        floor = source_lpf_floor_hz(role)
    except Exception:
        floor = None
    t = clamp((float(valence) + 1.0) / 2.0, 0.0, 1.0)
    if role in ("pad", "chords"):
        base = 6500.0 + t * 2500.0  # 6.5 - 9.0 kHz (floor 6.5 kHz)
    elif role == "lead":
        base = 9000.0 + t * 3500.0  # 9.0 - 12.5 kHz (floor 9 kHz)
    elif role == "pluck":
        base = 9000.0 + t * 3000.0  # floor 9 kHz
    elif role == "bell":
        base = 6500.0 + t * 3000.0  # floor 6.5 kHz
    elif role == "bass":
        base = 900.0 + t * 1100.0   # 0.9 - 2.0 kHz (bass keeps its own low cap)
    else:
        base = 6500.0 + t * 3500.0
    if floor is not None:
        base = max(base, floor)
    return round(min(base, 16000.0), 1)


def _ms_to_norm_env(ms: float) -> float:
    """Convert milliseconds `[0..5000]` to normalized `[0..1]` envelope slider."""
    m = clamp(float(ms), 1.0, 8000.0)
    return round(clamp((m / 5000.0) ** 0.35, 0.0, 1.0), 4)


def _hz_to_norm_cutoff(hz: float) -> float:
    """Convert Hz `[20..20000]` to normalized `[0..1]` synth cutoff (log-calibrated)."""
    f = clamp(float(hz), 20.0, 20000.0)
    return round(clamp(math.log(f / 20.0) / math.log(1000.0), 0.0, 1.0), 4)


def _encode_polyphony(spec: SynthSpec, target_poly: int) -> int:
    if spec.polyphony_mode == "int":
        return int(target_poly)
    if spec.polyphony_mode == "zero_based":
        return max(0, int(target_poly) - 1)
    if spec.polyphony_mode == "sawswarm_enum":
        # Choices: ['4', '8', '12', '16', '20', '24']
        choices = [4, 8, 12, 16, 20, 24]
        valid = [i for i, v in enumerate(choices) if v <= target_poly]
        return valid[-1] if valid else 0
    return int(target_poly)


def _encode_unison(spec: SynthSpec, target_unison: int) -> int:
    if spec.unison_mode == "int":
        return max(1, min(2, int(target_unison)))
    if spec.unison_mode == "zero_based":
        return 0 if target_unison <= 1 else 1
    return 1


def _apply_expressive_params(
    plugin: str,
    role: str,
    params: dict[str, Any],
    vibrato_delay_ms: float = 220.0,
) -> dict[str, Any]:
    """
    Add verified organic motion to native `gl.*` synths (`Phase 8 Q6.1`).
    Every emitted parameter name/range was checked with `gridline plugin-params` (0.7.7).

    v3.2 : `vibrato_delay_ms` (défaut 220 ms) retarde la profondeur du LFO de hauteur :
    l'attaque reste juste (<= 6 ct), la tenue "chante" ensuite — jamais de vibrato
    constant dès l'attaque (lu comme un désaccord).
    """
    if role in ("pad", "chords"):
        if plugin in ("gl.analog", "gl.wavetable", "gl.dcosynth", "gl.twinva", "gl.sawswarm"):
            if plugin == "gl.analog":
                params["LFO 1 rate"] = round(0.18 + (0.08 if role == "pad" else 0.0), 3)
                params["LFO 1 target"] = 7
                params["LFO 1 amount"] = 0.06
                params["LFO 1 shape"] = 0
            elif plugin == "gl.wavetable":
                params["LFO 1 rate"] = round(0.18 + (0.08 if role == "pad" else 0.0), 3)
                params["Mod 1 source"] = 4  # LFO 1
                params["Mod 1 target"] = 12  # F1 cutoff
                params["Mod 1 amount"] = 0.08
            elif plugin == "gl.twinva":
                params["LFO target"] = 0
                params["LFO amount"] = 0.06
                params["LFO rate"] = 0.18
                params["LFO shape"] = 3
            elif plugin == "gl.sawswarm":
                params["LFO target"] = 0
                params["LFO amount"] = 0.06
                params["LFO speed"] = 0.18
                params["LFO shape"] = 0
            elif plugin == "gl.dcosynth":
                params["LFO 1 > cutoff"] = 0.06
                params["LFO 1 rate"] = 0.18
        elif plugin == "gl.warmsynth":
            params["Filter LFO"] = 0.10
            params["LFO rate"] = 0.18
            params["LFO shape"] = 2
        elif plugin == "gl.string":
            params["Spread"] = 0.45
            params["Detune"] = 6
        elif plugin == "gl.fm":
            params["LFO to index"] = 0.03
            params["LFO rate"] = 0.25
        elif plugin == "gl.keys":
            params["Modulation"] = 0
            params["Tremolo rate"] = 0.35
            params["Tremolo depth"] = 0.06
            params["Velocity sens."] = 0.75
        elif plugin == "gl.sampler":
            params["LFO target"] = 3  # Cutoff
            params["LFO rate"] = 0.25
            params["LFO amount"] = 0.04
            params["LFO delay"] = _ms_to_norm_env(float(vibrato_delay_ms))  # v3.2 vibrato retardé
    elif role == "lead":
        # Timbral motion via filter LFOs: no measurable pitch bias (|tuning| <= 6.5 ct).
        if plugin == "gl.analog":
            params.update({"LFO 1 target": 7, "LFO 1 amount": 0.06, "LFO 1 rate": 4.8, "LFO 1 shape": 0})
        elif plugin == "gl.wavetable":
            params.update({"Mod 1 source": 4, "Mod 1 target": 12, "Mod 1 amount": 0.10, "LFO 1 rate": 4.8})
        elif plugin == "gl.twinva":
            params.update({"LFO target": 0, "LFO amount": 0.06, "LFO rate": 0.40, "LFO shape": 3})
        elif plugin == "gl.sawswarm":
            params.update({"LFO target": 0, "LFO amount": 0.05, "LFO speed": 0.35})
        elif plugin == "gl.dcosynth":
            params.update({"LFO 1 > cutoff": 0.04, "LFO 1 rate": 0.40})
        elif plugin == "gl.string":
            params.update({"Vibrato": 0.02, "Detune": 4, "Spread": 0.35})
        elif plugin == "gl.fm":
            params.update({"Vibrato": 1.0, "LFO rate": 4.8, "LFO to index": 0.04})
        elif plugin == "gl.keys":
            params.update({"Velocity sens.": 0.80, "Velocity to hardness": 0.20})
        elif plugin == "gl.sampler":
            params.update({"LFO target": 3, "LFO rate": 4.8, "LFO amount": 0.05, "LFO delay": 0.20})
    elif role in ("pluck", "bell"):
        if plugin in ("gl.analog", "gl.wavetable"):
            if plugin == "gl.analog":
                params["Velocity to cutoff"] = 0.35
            else:
                params.update({"Mod 1 source": 8, "Mod 1 target": 12, "Mod 1 amount": 0.18})
        elif plugin == "gl.twinva":
            params.update({"Velocity tracking": 0.35, "LFO amount": 0.02})
        elif plugin == "gl.sawswarm":
            params["Velocity tracking"] = 0.35
        elif plugin == "gl.fm":
            params.update({"Op 1 velocity": 0.65, "LFO to index": 0.02})
        elif plugin == "gl.keys":
            params.update({"Velocity to hardness": 0.25, "Velocity to muffle": 0.10})
        elif plugin == "gl.sampler":
            params.update({"Velocity to cutoff": 0.18, "LFO target": 3, "LFO amount": 0.06})
    elif role == "bass":
        if plugin in ("gl.analog", "gl.wavetable"):
            if plugin == "gl.analog":
                params["Filter drive"] = 0.12
            params["Drive"] = 0.05
        elif plugin == "gl.dcosynth":
            params["Saturation"] = 1
            params["Saturation amount"] = 0.12
        elif plugin == "gl.string":
            params["Drive"] = 0.08
    return params


@dataclass
class SynthVoiceConfig:
    plugin: str
    role: str
    valence: float
    cutoff_hz: float
    channel_params: dict[str, Any]
    needs_insert_lpf: bool
    insert_eq_bands: list[dict[str, Any]]
    polyphony: int
    unison: int
    makeup_gain_db: float = 0.0

    def to_channel_tuples(self, channel_idx: int) -> list[tuple[str, Any]]:
        return [(f"channel/{int(channel_idx)}/{k}", v) for k, v in self.channel_params.items()]

    def to_insert_eq_tuples(self, mixer: int, slot: int = 1) -> list[tuple[str, Any]]:
        if not self.insert_eq_bands:
            return []
        return eq_params(mixer, self.insert_eq_bands, slot=slot, quality=0)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.channel_params:
            return self.channel_params[key]
        raise KeyError(key)


def synth_voice(
    plugin: str,
    role: SynthRole,
    valence: float = 0.0,
    cutoff_hz: float | None = None,
    *,
    attack_ms: float | None = None,
    decay_ms: float | None = None,
    sustain_pct: float | None = None,
    release_ms: float | None = None,
    polyphony: int | None = None,
    unison: int | None = None,
    glide_ms: float | None = None,
    expressive: bool = True,
    vibrato_delay_ms: float = 220.0,
) -> SynthVoiceConfig:
    """
    Build a safe, anti-stridence, CPU-bounded parameter profile for a native `gl.*` synth.
    Automatically translates ADSR into the plugin's native units (seconds vs normalized 0..1),
    applies role/valence LPF cutoffs (attaching a `gl.eq` LPF spec if the synth lacks an
    internal LPF such as `gl.fm`, `gl.matrixfm`, `gl.keys`), and caps Polyphony/Unison.
    """
    if plugin not in SYNTH_REGISTRY:
        raise ValueError(f"Unsupported synth plugin {plugin!r}; registered: {sorted(SYNTH_REGISTRY)}")
    if role not in _ROLE_ADSR_DEFAULTS:
        raise ValueError(f"Unsupported role {role!r}; expected one of {sorted(_ROLE_ADSR_DEFAULTS)}")

    spec = SYNTH_REGISTRY[plugin]
    defaults = _ROLE_ADSR_DEFAULTS[role]
    atk_ms = defaults["attack_ms"] if attack_ms is None else max(0.5, float(attack_ms))
    dec_ms = defaults["decay_ms"] if decay_ms is None else max(5.0, float(decay_ms))
    sus_pct = defaults["sustain_pct"] if sustain_pct is None else clamp(float(sustain_pct), 0.0, 100.0)
    rel_ms = defaults["release_ms"] if release_ms is None else max(2.0, float(release_ms))

    eff_cutoff = _default_cutoff_hz(role, valence) if cutoff_hz is None else clamp(float(cutoff_hz), 120.0, 16000.0)

    # Enforce CPU bounds: Polyphony <= 12 (max 16 for sustained keys/piano), Unison <= 2
    max_poly = 16 if (plugin in ("gl.keys", "gl.samplekeys") and role in ("chords", "pad")) else 12
    default_poly = 1 if role == "bass" else (12 if role in ("chords", "pad") else 6)
    eff_poly = int(clamp(default_poly if polyphony is None else polyphony, 1, max_poly))
    eff_unison = int(clamp(1 if unison is None else unison, 1, 2))

    params: dict[str, Any] = dict(spec.extra_defaults)

    # ADSR encoding
    if spec.adsr_unit == "seconds":
        raw_vals = {
            "attack": atk_ms / 1000.0,
            "decay": dec_ms / 1000.0,
            "sustain": sus_pct / 100.0,
            "release": rel_ms / 1000.0,
        }
        for key, pname in spec.adsr_params.items():
            lo, hi = spec.adsr_bounds[key]
            params[pname] = round(clamp(raw_vals[key], lo, hi), 4)
    elif spec.adsr_unit == "normalized":
        raw_vals = {
            "attack": _ms_to_norm_env(atk_ms),
            "decay": _ms_to_norm_env(dec_ms),
            "sustain": round(sus_pct / 100.0, 4),
            "release": _ms_to_norm_env(rel_ms),
        }
        for key, pname in spec.adsr_params.items():
            params[pname] = raw_vals[key]

    # Cutoff encoding
    if spec.cutoff_param:
        if spec.cutoff_unit == "hz":
            lo, hi = spec.cutoff_bounds
            params[spec.cutoff_param] = round(clamp(eff_cutoff, lo, hi), 1)
        elif spec.cutoff_unit == "norm":
            params[spec.cutoff_param] = _hz_to_norm_cutoff(eff_cutoff)
        elif spec.cutoff_unit == "brightness":
            params[spec.cutoff_param] = round(clamp((eff_cutoff - 1000.0) / 10000.0, 0.22, 0.72), 3)

    # Polyphony & Unison
    if spec.polyphony_param:
        params[spec.polyphony_param] = _encode_polyphony(spec, eff_poly)
    if spec.unison_param:
        u_enc = _encode_unison(spec, eff_unison)
        if isinstance(spec.unison_param, tuple):
            for up in spec.unison_param:
                params[up] = u_enc
        else:
            params[spec.unison_param] = u_enc

    # Optional Glide
    if glide_ms is not None and spec.glide_param:
        if spec.glide_unit == "seconds":
            params[spec.glide_param] = round(clamp(float(glide_ms) / 1000.0, 0.001, 2.0), 4)
        elif spec.glide_unit == "normalized":
            params[spec.glide_param] = _ms_to_norm_env(float(glide_ms))

    # Role-specific voicing hygiene for gl.analog / gl.wavetable mono bass
    if role == "bass" and plugin in ("gl.analog", "gl.wavetable"):
        params["Voice mode"] = 1  # Mono

    # Phase 8 Q6.1 — Organic timbral motion (verified native params only)
    if expressive:
        params = _apply_expressive_params(plugin, role, params, vibrato_delay_ms=vibrato_delay_ms)

    needs_insert_lpf = not spec.has_internal_lpf
    insert_bands: list[dict[str, Any]] = []
    if needs_insert_lpf:
        insert_bands.append(eq_band(7, "lpf", eff_cutoff))

    return SynthVoiceConfig(
        plugin=plugin,
        role=role,
        valence=float(valence),
        cutoff_hz=round(eff_cutoff, 1),
        channel_params=params,
        needs_insert_lpf=needs_insert_lpf,
        insert_eq_bands=insert_bands,
        polyphony=eff_poly,
        unison=eff_unison,
        makeup_gain_db=lpf_makeup_gain_db(eff_cutoff),
    )


# ---------------------------------------------------------------------------
# v3.2 TimbralCohesion — archétypes timbraux gl.* calibrés par (profil, rôle, valence)
# ---------------------------------------------------------------------------

def load_timbre_archetypes(path: str | None = None) -> dict[str, Any]:
    """Charge `config/timbre_archetypes.json` (21+ recettes vérifiées)."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "timbre_archetypes.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("archetypes", {})
    except Exception:
        return {}


def resolve_archetype(profile: str, role: str, valence: float = 0.0) -> dict[str, Any] | None:
    """v3.2 : recette `gl.*` calibrée pour (mix_profile, role), avec `dark_overrides`
    appliqués si valence < -0.2. Retourne {"plugin", "params"} ou None."""
    arch = load_timbre_archetypes()
    key = f"{str(profile).strip()}:{str(role).strip()}"
    entry = arch.get(key) or arch.get(f"trap_rap:{role}")
    if not entry:
        return None
    params = dict(entry.get("params") or {})
    if float(valence) < -0.2:
        params.update(entry.get("dark_overrides") or {})
    return {"plugin": entry.get("plugin", "gl.warmsynth"), "params": params}
