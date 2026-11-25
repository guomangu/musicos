#!/usr/bin/env python3
"""
v3.3 #5 — `build_preview_highlights` : bande d'écoute rapide `renders/preview_highlights.wav`
(24 s : 8 s Couplet + 8 s Transition/Refrain + 8 s Pont/Outro, fondus enchaînés 150 ms) pour
auditer l'arc global d'un morceau en ~20 s après un raffinement chirurgical (`studio_refine_track`).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np


def _pick_segment_centers(duration_s: float, seg_s: float = 8.0) -> list[float]:
    """3 centres de segments : ~25 % (couplet), ~50 % (refrain), ~75 % (pont/outro)."""
    if duration_s <= seg_s * 3:
        starts = [max(0.0, duration_s * f - seg_s / 2) for f in (0.25, 0.5, 0.75)]
        return [min(s, max(0.0, duration_s - seg_s)) for s in starts]
    return [duration_s * f - seg_s / 2 for f in (0.25, 0.5, 0.75)]


def build_preview_highlights(master_wav: str, out_wav: str | None = None,
                             seg_s: float = 8.0, fade_ms: float = 150.0) -> dict[str, Any]:
    """Construit la bande 24 s (3 x 8 s, crossfade 150 ms) depuis un master WAV.
    Retourne {"out", "duration_s", "segments_s", "source_duration_s"} — 0 GPU, < 100 ms."""
    import soundfile as sf

    src = os.path.abspath(os.path.expanduser(master_wav))
    if not os.path.isfile(src):
        return {"ok": False, "error": f"master introuvable: {src}"}
    data, sr = sf.read(src, always_2d=True, dtype="float32")
    mono = data.mean(axis=1)
    duration_s = len(mono) / float(sr)
    seg_n = int(seg_s * sr)
    centers = _pick_segment_centers(duration_s, seg_s)
    segs: list[np.ndarray] = []
    labels = ["verse", "chorus", "bridge_outro"]
    for c in centers:
        a = int(max(0.0, c) * sr)
        seg = mono[a:a + seg_n]
        if len(seg) < seg_n:
            seg = np.pad(seg, (0, seg_n - len(seg)))
        segs.append(seg.astype(np.float64))
    fade_n = max(1, int(fade_ms / 1000.0 * sr))
    out = np.zeros(seg_n * len(segs), dtype=np.float64)
    for i, seg in enumerate(segs):
        base = i * seg_n
        out[base:base + seg_n] += seg
        if i > 0 and fade_n > 1:  # crossfade 150 ms avec le segment précédent
            t = np.linspace(0.0, 1.0, fade_n)
            out[base:base + fade_n] = out[base:base + fade_n] * (1 - t) + seg[:fade_n] * t
    # normalisation douce (évite tout clip)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.95:
        out = out * (0.95 / peak)
    if out_wav is None:
        out_wav = os.path.join(os.path.dirname(src), "preview_highlights.wav")
    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
    sf.write(out_wav, out.astype(np.float32), sr)
    return {
        "ok": True,
        "out": os.path.abspath(out_wav),
        "duration_s": round(len(out) / float(sr), 2),
        "segments_s": [round(seg_s, 2)] * 3,
        "labels": labels,
        "source_duration_s": round(duration_s, 2),
        "fade_ms": fade_ms,
    }
