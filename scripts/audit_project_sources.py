#!/usr/bin/env python3
"""
One-shot inventory & acoustic pre-audit of all channels/samples in an initial .gridline project.
Replaces 5-8 exploratory Python/CLI commands with a single token-efficient report:
  - Channel list (native synths + sample paths)
  - Sample duration, sr, channels, DC offset, start silence, trailing silence, max_diff (Nyquist spike flag)
  - Harmonic purity check on 808/bass (f0, 2*f0, 3*f0 in cents + Root note / Fine tune advice)
  - Spectral band energy + 150-300 Hz mud flag
  - Automatic cycle / implied BPM / chroma progression analysis for long loops (> 5 s)
  - Optional --prepare-samples <dir> to emit clean DC-removed & de-clicked drum/808 samples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(__file__))
from studio_lib import clean_audio_sample
from audio_metrics import analyze_single_file, analyze_vocal_signature


def _analyze_808_harmonics(mono: np.ndarray, sr: int) -> dict[str, Any]:
    dur = len(mono) / sr
    t0, t1 = (0.35, min(dur, 1.50)) if dur > 0.6 else (0.04, dur)
    s0, s1 = int(t0 * sr), max(int(t0 * sr) + 512, min(len(mono), int(t1 * sr)))
    seg = mono[s0:s1]
    if len(seg) < 512:
        return {}
    sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=131072))
    fr = np.fft.rfftfreq(131072, 1.0 / sr)
    m1 = (fr >= 26) & (fr <= 110)
    f0 = float(fr[m1][np.argmax(sp[m1])])
    midi_f = 69.0 + 12.0 * np.log2(f0 / 440.0)
    root_midi = int(round(midi_f))
    ct_f0 = round((midi_f - root_midi) * 100.0, 1)

    # Check 2nd & 3rd harmonics vs exact 2*f0 / 3*f0
    harm_info: dict[str, Any] = {
        "f0_hz": round(f0, 2),
        "recommended_root_midi": root_midi,
        "recommended_note": librosa.midi_to_note(root_midi),
        "recommended_fine_tune_ct": int(round(-ct_f0)),
    }
    inharmonic_warnings = []
    for mult in (2, 3):
        lo, hi = f0 * mult * 0.80, f0 * mult * 1.25
        mh = (fr >= lo) & (fr <= hi)
        if np.any(mh):
            fh = float(fr[mh][np.argmax(sp[mh])])
            err_ct = round(1200.0 * np.log2(fh / (f0 * mult)), 1)
            harm_info[f"h{mult}_err_ct"] = err_ct
            if abs(err_ct) > 40.0:
                inharmonic_warnings.append(f"h{mult} deviates {err_ct:+.1f}ct from {mult}*f0 ({fh:.1f}Hz)")
    if inharmonic_warnings:
        harm_info["inharmonic_warning"] = inharmonic_warnings
    if f0 < 42.0:
        harm_info["infra_sub_warning"] = (
            f"f0={f0:.1f}Hz < 42Hz (infra-sub saturating limiter): keep 808 notes >= MIDI 31 (G1=49Hz), "
            "or write Octave 2 (MIDI 36-42) for keys C..F#."
        )
    return harm_info


def _analyze_long_loop(mono: np.ndarray, sr: int) -> dict[str, Any]:
    dur = len(mono) / sr
    chroma = librosa.feature.chroma_cqt(y=mono.astype(np.float32), sr=sr, hop_length=512)
    hop_s = 512.0 / sr
    tuning_ct = round(float(librosa.estimate_tuning(y=mono.astype(np.float32), sr=sr)) * 100.0, 1)

    # Self-similarity across lags 3.5..18s
    best_sim, best_lag = -1.0, 0.0
    for lag_s in np.arange(3.5, min(18.0, dur * 0.6), 0.05):
        lag = int(round(lag_s / hop_s))
        if lag >= chroma.shape[1] - 4:
            break
        a, b = chroma[:, :-lag], chroma[:, lag:]
        num = np.sum(a * b, axis=0)
        den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-9
        sim = float(np.mean(num / den))
        if sim > best_sim:
            best_sim, best_lag = sim, float(lag_s)

    # RMS envelope minima to detect phrase swell boundaries
    rms_t = np.arange(0, dur - 0.05, 0.05)
    rms_v = np.array([np.sqrt(np.mean(mono[int(t * sr) : int((t + 0.05) * sr)] ** 2)) for t in rms_t])
    minima, _ = find_peaks(-rms_v, distance=40, prominence=float(np.max(rms_v) * 0.12))

    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    mean_c = chroma.mean(axis=1)
    top_notes = [notes[i] for i in np.argsort(mean_c)[::-1][:6]]
    implied_4bar_bpm = round(16.0 * 60.0 / best_lag, 2) if best_lag > 0 else None
    implied_2bar_bpm = round(8.0 * 60.0 / best_lag, 2) if best_lag > 0 else None

    return {
        "tuning_cents": tuning_ct,
        "best_chroma_cycle_s": round(best_lag, 2),
        "cycle_similarity": round(best_sim, 3),
        "implied_bpm_if_4bars": implied_4bar_bpm,
        "implied_bpm_if_2bars": implied_2bar_bpm,
        "rms_minima_s": [round(float(rms_t[i]), 2) for i in minima[:16]],
        "top_pitch_classes": top_notes,
    }


def audit_gridline_sources(proj_path: str, prepare_dir: str = "") -> dict[str, Any]:
    # v3.1 ASSET-05: rapport d'hygiène des assets (fichiers pièges exclus proprement)
    asset_hygiene: dict[str, Any] = {"scanned": 0, "safe_audio": 0, "skipped_count": 0, "skipped": []}
    if prepare_dir and os.path.isdir(os.path.expanduser(prepare_dir)):
        try:
            from asset_hygiene import scan_audio_tree
            found, skipped = scan_audio_tree(os.path.expanduser(prepare_dir))
            asset_hygiene = {
                "scanned": len(found) + len(skipped),
                "safe_audio": len(found),
                "skipped_count": len(skipped),
                "skipped": skipped[:20],
            }
        except Exception as exc:
            asset_hygiene = {"error": str(exc)}
    d = json.load(open(proj_path, encoding="utf-8"))
    mixer_by_id = {m.get("id"): m for m in d.get("mixer", [])}
    mixer_warnings: list[str] = []

    # Check mixer hygiene: sidechain buses must receive sidechain=True sends (not direct audio!), flag muted tracks
    for m in d.get("mixer", []):
        mname = (m.get("name") or "").lower()
        if m.get("muted"):
            mixer_warnings.append(f"MUTED_MIXER_TRACK({m.get('name') or m.get('id')})")
        # Detect if this track has a sidechain compressor (gl.comp with Detector input=1)
        has_sc_comp = any(
            (sl.get("plugin") or {}).get("type") == "gl.comp"
            and len((sl.get("plugin") or {}).get("params", [])) > 8
            and (sl.get("plugin") or {}).get("params", [0]*10)[8] == 1
            for sl in m.get("slots", [])
        )
        if has_sc_comp:
            mid = m.get("id")
            sc_senders = [
                src.get("name") or src.get("id")
                for src in d.get("mixer", [])
                for s in src.get("sends", [])
                if s.get("to") == mid and s.get("sidechain")
            ]
            audio_senders = [
                src.get("name") or src.get("id")
                for src in d.get("mixer", [])
                for s in src.get("sends", [])
                if s.get("to") == mid and not s.get("sidechain") and "808" not in (src.get("name") or "").lower() and "bus" not in mname
            ]
            if not sc_senders:
                mixer_warnings.append(f"MISSING_SIDECHAIN_SENDER_TO({m.get('name') or mid})")

    track_isolation_warnings: list[str] = []

    # Check "1 sonorité = 1 piste dédiée" (dedicated mixer track, single-channel patterns, dedicated named playlist track)
    mixer_to_channels: dict[Any, list[str]] = {}
    ch_id_to_name: dict[str, str] = {}
    for idx, ch in enumerate(d.get("channels", [])):
        cid = str(ch.get("id", idx))
        cname = ch.get("name") or f"ch_{idx}"
        ch_id_to_name[cid] = cname
        mt = ch.get("mixerTrack")
        if mt is not None:
            mixer_to_channels.setdefault(mt, []).append(cname)
    for mt, cnames in mixer_to_channels.items():
        if len(cnames) > 1:
            mname = mixer_by_id.get(mt, {}).get("name") or mt
            track_isolation_warnings.append(f"SHARED_MIXER_TRACK({mname}: {cnames})")

    pat_by_id = {p.get("id"): p for p in d.get("patterns", [])}
    pat_channels: dict[Any, set[str]] = {}
    for pid, pat in pat_by_id.items():
        ch_ids = {
            str(k) for k, v in (pat.get("notes") or {}).items() if v
        } | {
            str(k) for k, v in (pat.get("steps") or {}).items() if v
        }
        pat_channels[pid] = ch_ids

    used_pats = {item.get("pattern") for item in d.get("playlist", []) if item.get("pattern")}
    for pid in sorted(used_pats):
        ch_ids = pat_channels.get(pid, set())
        if len(ch_ids) > 1:
            pname = pat_by_id.get(pid, {}).get("name") or f"pat_{pid}"
            names = [ch_id_to_name.get(c, c) for c in sorted(ch_ids)]
            track_isolation_warnings.append(f"MULTI_CHANNEL_PATTERN({pname}: {names})")

    playlist_track_sources: dict[int, set[str]] = {}
    for item in d.get("playlist", []):
        tr_idx = item.get("track")
        if tr_idx is None:
            continue
        if item.get("kind") == "automation":
            playlist_track_sources.setdefault(int(tr_idx), set()).add(f"auto:{item.get('source')}")
        elif item.get("channel"):
            cid = str(item.get("channel"))
            playlist_track_sources.setdefault(int(tr_idx), set()).add(f"ch:{ch_id_to_name.get(cid, cid)}")
        elif item.get("pattern"):
            for cid in pat_channels.get(item.get("pattern"), set()):
                playlist_track_sources.setdefault(int(tr_idx), set()).add(f"ch:{ch_id_to_name.get(cid, cid)}")

    tracks_meta = d.get("tracks", [])
    for tr_idx, srcs in sorted(playlist_track_sources.items()):
        if len(srcs) > 1:
            track_isolation_warnings.append(f"SHARED_PLAYLIST_TRACK(track_{tr_idx + 1}: {sorted(srcs)})")
        tr_name = tracks_meta[tr_idx].get("name", "").strip() if tr_idx < len(tracks_meta) else ""
        if not tr_name:
            track_isolation_warnings.append(f"UNNAMED_PLAYLIST_TRACK(track_{tr_idx + 1})")

    report: dict[str, Any] = {
        "project": os.path.abspath(proj_path),
        "tempo": d.get("tempo"),
        "ppq": d.get("ppq"),
        "mixer_warnings": mixer_warnings,
        "track_isolation_warnings": track_isolation_warnings,
        "channels": [],
        "prepared_samples": [],
    }
    seen_paths: set[str] = set()

    for idx, ch in enumerate(d.get("channels", [])):
        plug = ch.get("plugin", {})
        spath = plug.get("data")
        name = ch.get("name", f"ch_{idx}")
        m_obj = mixer_by_id.get(ch.get("mixerTrack"), {})
        entry: dict[str, Any] = {
            "index": idx,
            "name": name,
            "plugin": plug.get("id") or plug.get("type") or ("gl.sampler" if spath else "native"),
            "mixer_id": ch.get("mixerTrack"),
            "mixer_name": m_obj.get("name"),
        }
        if spath and os.path.isfile(spath):
            y, sr = sf.read(spath, dtype="float64", always_2d=True)
            mono = y.mean(axis=1)
            dur = len(mono) / sr
            pk = float(np.max(np.abs(y)))
            dc = float(np.mean(mono))
            max_diff = float(np.max(np.abs(np.diff(y[:, 0])))) if len(y) > 1 else 0.0
            nz = np.where(np.abs(mono) > max(1e-4, pk * 0.02))[0]
            start_ms = round(float(nz[0] / sr * 1000.0), 1) if len(nz) else 0.0
            eff_dur = round(float(nz[-1] / sr), 3) if len(nz) else 0.0

            sp = np.abs(np.fft.rfft(mono[: min(len(mono), sr * 8)]))
            fr = np.fft.rfftfreq(min(len(mono), sr * 8), 1.0 / sr)
            tot = float(np.sum(sp**2)) + 1e-12
            bands = {
                "sub_20_60": round(100.0 * float(np.sum(sp[(fr >= 20) & (fr < 60)] ** 2)) / tot, 1),
                "bass_60_150": round(100.0 * float(np.sum(sp[(fr >= 60) & (fr < 150)] ** 2)) / tot, 1),
                "mud_150_300": round(100.0 * float(np.sum(sp[(fr >= 150) & (fr < 300)] ** 2)) / tot, 1),
                "mid_300_4k": round(100.0 * float(np.sum(sp[(fr >= 300) & (fr < 4000)] ** 2)) / tot, 1),
                "hi_4k_20k": round(100.0 * float(np.sum(sp[(fr >= 4000) & (fr < 20000)] ** 2)) / tot, 1),
            }
            flags = []
            if abs(dc) > 0.005:
                flags.append(f"DC_OFFSET({dc:+.4f})")
            if max_diff > 0.45:
                flags.append(f"NYQUIST_JUMP({max_diff:.2f})")
            if dur - eff_dur > 0.35:
                flags.append(f"TRAILING_SILENCE({dur - eff_dur:.2f}s)")
            if start_ms > 8.0:
                flags.append(f"PRE_TRANSIENT_SILENCE({start_ms:.1f}ms)")
            if bands["mud_150_300"] >= 35.0:
                flags.append(f"LOW_MID_HEAVY({bands['mud_150_300']}% -> HPF 175Hz + -4.5dB @ 240Hz)")

            # Deterministic DSP pitch/purity & phase audit on every sample
            dsp_entry, _ = analyze_single_file(spath, with_clap=False)
            dsp = dsp_entry["dsp"]
            nature = dsp.get("sample_nature")
            playable = dsp.get("playable_chromatically")
            low_mid_corr = dsp.get("low_mid_corr", 1.0)
            lname = name.lower()
            is_perc_ch = any(k in lname for k in ("kick", "snare", "clap", "hat", "hh", "perc", "rim", "crash", "riser", "fx"))
            is_tonal_playable = any(k in lname for k in ("lead", "chord", "piano", "pluck", "bell", "pad", "keys", "808", "bass"))
            if is_tonal_playable and not is_perc_ch and not playable:
                flags.append(f"DISSONANCE_RISK({nature}, onsets={dsp.get('onset_count')}, std_st={dsp.get('f0_std_semitones')}, poly={dsp.get('polyphony_ratio')})")
            if not is_perc_ch and low_mid_corr < 0.50:
                flags.append(f"LOW_MID_WIDE_PHASE(corr={low_mid_corr} -> M/S <650Hz)")

            entry.update({
                "file": os.path.basename(spath),
                "path": spath,
                "dur_s": round(dur, 3),
                "eff_dur_s": eff_dur,
                "sr": int(sr),
                "ch": int(y.shape[1]),
                "peak": round(pk, 3),
                "max_sample_jump": round(max_diff, 3),
                "low_mid_corr": low_mid_corr,
                "pitch_purity": {
                    "sample_nature": nature,
                    "playable_chromatically": playable,
                    "note": dsp.get("detected_note"),
                    "f0_hz": dsp.get("detected_f0_hz"),
                    "midi_root": dsp.get("midi_root"),
                    "fine_tune_ct": dsp.get("recommended_fine_tune_ct"),
                    "hnr_db": dsp.get("hnr_db"),
                    "f0_std_st": dsp.get("f0_std_semitones"),
                    "polyphony_ratio": dsp.get("polyphony_ratio"),
                },
                "bands_pct": bands,
                "flags": flags,
            })

            if "808" in lname or "bass" in lname or "sub" in lname:
                entry["bass_harmonics"] = _analyze_808_harmonics(mono, sr)
            elif any(k in lname for k in ("vox", "vocal", "chant", "voice", "air")):
                entry["vocal_signature"] = analyze_vocal_signature(spath)
            elif dur >= 5.0:
                entry["loop_analysis"] = _analyze_long_loop(mono, sr)

            if prepare_dir and spath not in seen_paths and dur < 5.0:
                seen_paths.add(spath)
                safe_name = os.path.splitext(os.path.basename(spath))[0].lower().replace(" ", "_") + "_clean.wav"
                out_p = os.path.join(prepare_dir, safe_name)
                lpf = 12000.0 if max_diff > 0.45 and bands["hi_4k_20k"] > 30.0 else None
                trim_r = 0.005 if (dur - eff_dur > 0.35) else None
                prep = clean_audio_sample(
                    spath, out_p, lpf_hz=lpf, fade_in_ms=2.0, fade_out_ms=15.0,
                    remove_dc=True, max_peak=0.85, trim_end_below_ratio=trim_r,
                )
                report["prepared_samples"].append(prep)
        report["channels"].append(entry)

    report["asset_hygiene"] = asset_hygiene
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-audit all channels and samples in a .gridline project.")
    ap.add_argument("project", help="Path to .gridline file")
    ap.add_argument("--prepare-samples", default="", help="Optional output folder for auto-cleaned <5s samples")
    args = ap.parse_args()
    rep = audit_gridline_sources(args.project, prepare_dir=args.prepare_samples)
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
