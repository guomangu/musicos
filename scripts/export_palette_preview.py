#!/usr/bin/env python3
"""
Micro-Checkpoint Humain Optionnel (`export_palette_preview.py`, RFC §3.5).

Assembles 2.5 s of each validated Sound Check track separated by 0.4 s of silence
into `<proj>/renders/palette_preview.wav`, alongside a timestamped text index
(`0:00 Kick | 0:03 808 | 0:06 Chords | ...`).

Optional `--human-gate` interactive prompt lets the user accept or reject any
specific track in 15 seconds before starting vertical tests.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import librosa
import numpy as np
import soundfile as sf


def _fmt_mmss(sec: float) -> str:
    s = int(round(max(0.0, sec)))
    return f"{s // 60}:{s % 60:02d}"


def export_palette_preview(
    tracks: list[tuple[str, str]],
    out_wav: str,
    *,
    slice_sec: float = 2.5,
    gap_sec: float = 0.4,
    target_sr: int = 44100,
    fade_ms: float = 18.0,
) -> dict[str, Any]:
    """
    Assemble `(label, wav_path)` pairs into one clean preview WAV + text index.
    """
    if not tracks:
        raise ValueError("No soundcheck tracks provided to export_palette_preview.")

    n_slice = int(round(slice_sec * target_sr))
    n_gap = int(round(gap_sec * target_sr))
    n_fade = min(n_slice // 4, int(round(fade_ms * target_sr / 1000.0)))
    gap = np.zeros((n_gap, 2), dtype=np.float32)

    chunks: list[np.ndarray] = []
    index_entries: list[dict[str, Any]] = []
    cursor_s = 0.0

    for idx, (label, path) in enumerate(tracks):
        abs_p = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(abs_p):
            raise FileNotFoundError(f"Soundcheck track WAV not found: {abs_p}")
        y, sr = sf.read(abs_p, dtype="float32", always_2d=True)
        if y.shape[1] == 1:
            y = np.repeat(y, 2, axis=1)
        elif y.shape[1] > 2:
            y = y[:, :2]
        if sr != target_sr and len(y) > 16:
            ch0 = librosa.resample(y[:, 0], orig_sr=sr, target_sr=target_sr)
            ch1 = librosa.resample(y[:, 1], orig_sr=sr, target_sr=target_sr)
            n_min = min(len(ch0), len(ch1))
            y = np.column_stack([ch0[:n_min], ch1[:n_min]]).astype(np.float32)

        seg = y[:n_slice].copy()
        if len(seg) < n_slice:
            seg = np.vstack([seg, np.zeros((n_slice - len(seg), 2), dtype=np.float32)])

        # Gentle peak normalization (-3 dBFS) so every track is clearly audible
        pk = float(np.max(np.abs(seg)))
        if pk > 1e-6:
            seg = seg * (0.7079 / pk)

        if n_fade > 0:
            ramp = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)[:, None]
            seg[:n_fade] *= ramp
            seg[-n_fade:] *= ramp[::-1]

        index_entries.append({
            "label": label,
            "start_s": round(cursor_s, 2),
            "end_s": round(cursor_s + slice_sec, 2),
            "timestamp": _fmt_mmss(cursor_s),
            "source_wav": abs_p,
        })
        chunks.append(seg)
        cursor_s += slice_sec
        if idx < len(tracks) - 1:
            chunks.append(gap)
            cursor_s += gap_sec

    out_abs = os.path.abspath(os.path.expanduser(out_wav))
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    full = np.vstack(chunks).astype(np.float32)
    sf.write(out_abs, full, target_sr, subtype="PCM_24")

    index_line = " | ".join(f"{e['timestamp']} {e['label']}" for e in index_entries)
    index_txt = os.path.splitext(out_abs)[0] + "_index.txt"
    with open(index_txt, "w", encoding="utf-8") as fh:
        fh.write(index_line + "\n")

    return {
        "preview_wav": out_abs,
        "index_txt": index_txt,
        "index_line": index_line,
        "total_duration_s": round(len(full) / float(target_sr), 2),
        "tracks": index_entries,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble validated soundcheck tracks into palette_preview.wav.")
    ap.add_argument("--soundcheck-dir", default="", help="Folder containing soundcheck *.wav files.")
    ap.add_argument("--tracks", nargs="*", default=[], help="Explicit 'Label=/path/to.wav' entries.")
    ap.add_argument("-o", "--out", required=True, help="Output palette_preview.wav path.")
    ap.add_argument("--slice-sec", type=float, default=2.5)
    ap.add_argument("--gap-sec", type=float, default=0.4)
    ap.add_argument("--human-gate", action="store_true", help="Prompt interactively to approve or reject tracks.")
    args = ap.parse_args()

    pairs: list[tuple[str, str]] = []
    for item in args.tracks:
        if "=" in item:
            lbl, path = item.split("=", 1)
            pairs.append((lbl.strip(), path.strip()))
        else:
            pairs.append((os.path.splitext(os.path.basename(item))[0], item.strip()))
    if not pairs and args.soundcheck_dir and os.path.isdir(args.soundcheck_dir):
        for fn in sorted(os.listdir(args.soundcheck_dir)):
            if fn.endswith(".wav"):
                pairs.append((os.path.splitext(fn)[0], os.path.join(args.soundcheck_dir, fn)))

    res = export_palette_preview(pairs, args.out, slice_sec=args.slice_sec, gap_sec=args.gap_sec)
    if args.human_gate:
        print(f"Preview ready: {res['preview_wav']}")
        print(f"Index: {res['index_line']}")
        ans = input("Approve all tracks? [Y/n or comma-separated track names to reject]: ").strip()
        if ans.lower() in ("", "y", "yes", "o", "oui"):
            res["human_gate"] = {"approved": True, "rejected_tracks": []}
        else:
            rej = [x.strip() for x in ans.split(",") if x.strip() and x.strip().lower() not in ("n", "no", "non")]
            res["human_gate"] = {"approved": False, "rejected_tracks": rej}
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
