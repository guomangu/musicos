"""Serveur MCP de Reverse Engineering Musical (Stem Split + Basic Pitch ONNX Audio->MIDI + Plan Rap Flip)."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

try:
    import essentia

    essentia.log.infoActive = False
    essentia.log.warningActive = False
    import essentia.standard as es

    _HAS_ESSENTIA = True
except Exception:  # pragma: no cover
    es = None
    _HAS_ESSENTIA = False
import librosa
import mido
import numpy as np

try:
    import onnxruntime as ort

    _HAS_ORT = True
except Exception:  # pragma: no cover - heavy native dep, optional until transcription is used
    ort = None
    _HAS_ORT = False
import soundfile as sf
from mcp.server.mcpserver import MCPServer
from scipy import signal

STUDIO_DIR = os.environ.get("AI_STUDIO_DIR") or str(Path(__file__).resolve().parents[2])
BASIC_PITCH_ONNX = os.path.join(STUDIO_DIR, "tools/models/nmp.onnx")

# Constantes du modèle Basic Pitch (Apache-2.0, Spotify ICASSP 2022)
BP_SR = 22050
BP_FFT_HOP = 256
BP_WINDOW_SEC = 2
BP_FPS = BP_SR // BP_FFT_HOP  # 86
BP_N_SAMPLES = BP_SR * BP_WINDOW_SEC - BP_FFT_HOP  # 43844
BP_ANNOT_FRAMES = BP_FPS * BP_WINDOW_SEC  # 172
BP_OVERLAP_FRAMES = 30
BP_MIDI_OFFSET = 21
BP_MAX_FREQ_IDX = 87

mcp = MCPServer(
    name="reverse-engineer",
    instructions=(
        "Outils de reverse engineering musical (classiques du jeu vidéo -> instrus Rap) : "
        "séparation de stems via Gridline (htdemucs/htdemucs_6s), transcription polyphonique "
        "Audio->MIDI (Spotify Basic Pitch ONNX) prête pour Gridline set_notes, détection "
        "d'accords/BPM/tonalité et plan de flip Rap/Trap/Boom-Bap."
    ),
)

_BP_SESSION: "ort.InferenceSession | None" = None

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def _get_bp_session() -> "ort.InferenceSession":
    global _BP_SESSION
    if not _HAS_ORT:
        raise RuntimeError(
            "onnxruntime n'est pas installé : la transcription Audio->MIDI (Basic Pitch) est indisponible. "
            "Installez-le via `pip install onnxruntime` (voir requirements.txt)."
        )
    if _BP_SESSION is None:
        if not os.path.isfile(BASIC_PITCH_ONNX):
            raise FileNotFoundError(f"Modèle Basic Pitch ONNX introuvable : {BASIC_PITCH_ONNX}")
        _BP_SESSION = ort.InferenceSession(BASIC_PITCH_ONNX, providers=["CPUExecutionProvider"])
    return _BP_SESSION


def _unwrap_bp_output(output: np.ndarray, orig_len: int, n_overlap: int, hop_size: int) -> np.ndarray:
    half = n_overlap // 2
    if half > 0:
        output = output[:, half:-half, :]
    unwrapped = output.reshape(output.shape[0] * output.shape[1], output.shape[2])
    expected = int((orig_len / hop_size) * ((BP_WINDOW_SEC * BP_FPS) - n_overlap))
    return unwrapped[:expected, :]


def _infer_onsets(onsets: np.ndarray, frames: np.ndarray, n_diff: int = 2) -> np.ndarray:
    diffs = []
    for n in range(1, n_diff + 1):
        appended = np.concatenate([np.zeros((n, frames.shape[1]), dtype=frames.dtype), frames])
        diffs.append(appended[n:, :] - appended[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    mx = float(np.max(frame_diff))
    if mx > 1e-8:
        frame_diff = float(np.max(onsets)) * frame_diff / mx
    return np.maximum(onsets, frame_diff)


def _decode_bp_notes(
    frames: np.ndarray,
    onsets: np.ndarray,
    onset_thresh: float,
    frame_thresh: float,
    min_note_frames: int,
    min_freq_hz: float | None,
    max_freq_hz: float | None,
    energy_tol: int = 11,
) -> list[tuple[float, float, int, float]]:
    onsets = onsets.copy()
    frames = frames.copy()
    min_idx = int(np.round(librosa.hz_to_midi(min_freq_hz) - BP_MIDI_OFFSET)) if min_freq_hz else 0
    max_idx = int(np.round(librosa.hz_to_midi(max_freq_hz) - BP_MIDI_OFFSET)) if max_freq_hz else onsets.shape[1]
    min_idx = max(0, min_idx)
    max_idx = min(onsets.shape[1], max_idx)
    onsets[:, :min_idx] = 0
    frames[:, :min_idx] = 0
    onsets[:, max_idx:] = 0
    frames[:, max_idx:] = 0

    onsets = _infer_onsets(onsets, frames)
    n_frames = frames.shape[0]
    peak_mat = np.zeros_like(onsets)
    peaks = signal.argrelmax(onsets, axis=0)
    peak_mat[peaks] = onsets[peaks]

    onset_t, onset_f = np.where(peak_mat >= onset_thresh)
    remaining = frames.copy()
    events: list[tuple[int, int, int, float]] = []

    for start_i, freq_i in zip(onset_t[::-1], onset_f[::-1]):
        if start_i >= n_frames - 1:
            continue
        i = start_i + 1
        k = 0
        while i < n_frames - 1 and k < energy_tol:
            k = k + 1 if remaining[i, freq_i] < frame_thresh else 0
            i += 1
        end_i = i - k
        if end_i - start_i <= min_note_frames:
            continue
        remaining[start_i:end_i, freq_i] = 0
        if freq_i < BP_MAX_FREQ_IDX:
            remaining[start_i:end_i, freq_i + 1] = 0
        if freq_i > 0:
            remaining[start_i:end_i, freq_i - 1] = 0
        amp = float(np.mean(frames[start_i:end_i, freq_i]))
        events.append((int(start_i), int(end_i), int(freq_i + BP_MIDI_OFFSET), amp))

    # Passe Melodia sur l'énergie restante
    while float(np.max(remaining)) > frame_thresh:
        i_mid, freq_i = np.unravel_index(int(np.argmax(remaining)), remaining.shape)
        remaining[i_mid, freq_i] = 0
        i, k = i_mid + 1, 0
        while i < n_frames - 1 and k < energy_tol:
            k = k + 1 if remaining[i, freq_i] < frame_thresh else 0
            remaining[i, freq_i] = 0
            if freq_i < BP_MAX_FREQ_IDX:
                remaining[i, freq_i + 1] = 0
            if freq_i > 0:
                remaining[i, freq_i - 1] = 0
            i += 1
        i_end = i - 1 - k

        i, k = i_mid - 1, 0
        while i > 0 and k < energy_tol:
            k = k + 1 if remaining[i, freq_i] < frame_thresh else 0
            remaining[i, freq_i] = 0
            if freq_i < BP_MAX_FREQ_IDX:
                remaining[i, freq_i + 1] = 0
            if freq_i > 0:
                remaining[i, freq_i - 1] = 0
            i -= 1
        i_start = i + 1 + k
        if i_end - i_start <= min_note_frames:
            continue
        amp = float(np.mean(frames[i_start:i_end, freq_i]))
        events.append((int(i_start), int(i_end), int(freq_i + BP_MIDI_OFFSET), amp))

    # Conversion frames -> secondes
    frame_ids = np.arange(n_frames)
    orig_times = librosa.frames_to_time(frame_ids, sr=BP_SR, hop_length=BP_FFT_HOP)
    win_nums = np.floor(frame_ids / BP_ANNOT_FRAMES)
    win_offset = (BP_FFT_HOP / BP_SR) * (BP_ANNOT_FRAMES - (BP_N_SAMPLES / BP_FFT_HOP)) + 0.0018
    times = orig_times - (win_offset * win_nums)

    out: list[tuple[float, float, int, float]] = []
    for s_i, e_i, pitch, amp in events:
        t0 = max(0.0, float(times[min(s_i, len(times) - 1)]))
        t1 = max(t0 + 0.03, float(times[min(e_i, len(times) - 1)]))
        out.append((t0, t1, pitch, amp))
    out.sort(key=lambda x: (x[0], x[2]))
    return out


def _write_midi_file(notes_sec: list[tuple[float, float, int, float]], bpm: float, out_mid: str) -> None:
    ppq = 480
    mid = mido.MidiFile(ticks_per_beat=ppq)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    raw_events: list[tuple[int, int, int, int]] = []
    for t0, t1, pitch, amp in notes_sec:
        s_tick = max(0, int(round(mido.second2tick(t0, ppq, tempo))))
        e_tick = max(s_tick + 15, int(round(mido.second2tick(t1, ppq, tempo))))
        vel = max(20, min(127, int(round(amp * 127))))
        raw_events.append((s_tick, 1, pitch, vel))
        raw_events.append((e_tick, 0, pitch, 0))
    raw_events.sort(key=lambda e: (e[0], e[1]))

    prev_tick = 0
    for tick, is_on, pitch, vel in raw_events:
        delta = max(0, tick - prev_tick)
        prev_tick = tick
        if is_on:
            track.append(mido.Message("note_on", note=pitch, velocity=vel, time=delta))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))

    os.makedirs(os.path.dirname(os.path.abspath(out_mid)), exist_ok=True)
    mid.save(out_mid)


def _match_chord(chroma_vec: np.ndarray) -> str:
    best_name = "N"
    best_score = -1.0
    v = chroma_vec / (float(np.linalg.norm(chroma_vec)) + 1e-8)
    for root in range(12):
        for suffix, intervals in (("maj", (0, 4, 7)), ("min", (0, 3, 7)), ("m7", (0, 3, 7, 10)), ("7", (0, 4, 7, 10))):
            tmpl = np.zeros(12, dtype=np.float32)
            for iv in intervals:
                tmpl[(root + iv) % 12] = 1.0
            tmpl /= float(np.linalg.norm(tmpl))
            score = float(np.dot(v, tmpl))
            if score > best_score:
                best_score = score
                best_name = f"{NOTE_NAMES[root]}{'' if suffix == 'maj' else suffix}"
    return best_name


@mcp.tool()
def split_stems(
    audio_path: str,
    output_dir: str,
    model: str = "htdemucs_6s",
    stems: str = "",
) -> str:
    """
    Sépare un morceau (MP3/WAV/FLAC/OGG) en stems isolés via Gridline (`gridline stems`).
    Modèles disponibles : 'htdemucs' (4 stems : drums,bass,other,vocals) ou
    'htdemucs_6s' (6 stems : drums,bass,other,vocals,guitar,piano).
    """
    abs_in = os.path.abspath(os.path.expanduser(audio_path))
    if not os.path.isfile(abs_in):
        raise FileNotFoundError(f"Fichier introuvable : {abs_in}")
    abs_out = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(abs_out, exist_ok=True)

    cmd = ["gridline", "stems", abs_in, "--model", model, "--out", abs_out]
    if stems.strip():
        cmd.extend(["--stems", stems.strip()])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

    generated: list[dict[str, Any]] = []
    for fname in sorted(os.listdir(abs_out)):
        if fname.lower().endswith((".wav", ".flac", ".ogg")):
            fpath = os.path.join(abs_out, fname)
            info = sf.info(fpath)
            generated.append(
                {
                    "stem": os.path.splitext(fname)[0],
                    "path": fpath,
                    "duration_sec": round(info.frames / info.samplerate, 3) if info.samplerate else 0.0,
                    "sample_rate": info.samplerate,
                }
            )
    return json.dumps(
        {
            "input": abs_in,
            "model": model,
            "output_dir": abs_out,
            "stems": generated,
            "cli_log": proc.stdout.strip(),
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def transcribe_audio_to_midi(
    audio_path: str,
    output_mid: str = "",
    bpm: float | None = None,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_len_ms: float = 110.0,
    min_freq_hz: float | None = None,
    max_freq_hz: float | None = None,
    quantize_beats: float = 0.25,
    max_notes: int = 400,
) -> str:
    """
    Transcrit un stem ou une boucle audio en notes MIDI polyphoniques (Spotify Basic Pitch ONNX),
    quantifiées en beats pour Gridline `set_notes` et exportables en fichier .mid.
    """
    abs_in = os.path.abspath(os.path.expanduser(audio_path))
    if not os.path.isfile(abs_in):
        raise FileNotFoundError(f"Fichier audio introuvable : {abs_in}")

    y, _ = librosa.load(abs_in, sr=BP_SR, mono=True)
    if len(y) < 512:
        raise ValueError("Audio trop court pour transcription MIDI.")

    detected_bpm = float(bpm) if bpm else float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=BP_SR)[0])[0])
    if detected_bpm < 40.0 or detected_bpm > 240.0:
        detected_bpm = 120.0

    sess = _get_bp_session()
    overlap_len = BP_OVERLAP_FRAMES * BP_FFT_HOP
    hop_size = BP_N_SAMPLES - overlap_len
    orig_len = len(y)
    padded = np.concatenate([np.zeros(overlap_len // 2, dtype=np.float32), y.astype(np.float32)])

    windows = []
    for i in range(0, len(padded), hop_size):
        w = padded[i : i + BP_N_SAMPLES]
        if len(w) < BP_N_SAMPLES:
            w = np.pad(w, (0, BP_N_SAMPLES - len(w)))
        windows.append(w[:, np.newaxis])
    batch = np.stack(windows, axis=0).astype(np.float32)

    note_batches, onset_batches = [], []
    for start in range(0, len(batch), 16):
        n_out, o_out = sess.run(
            ["StatefulPartitionedCall:1", "StatefulPartitionedCall:2"],
            {"serving_default_input_2:0": batch[start : start + 16]},
        )
        note_batches.append(n_out)
        onset_batches.append(o_out)

    frames = _unwrap_bp_output(np.concatenate(note_batches, axis=0), orig_len, BP_OVERLAP_FRAMES, hop_size)
    onsets = _unwrap_bp_output(np.concatenate(onset_batches, axis=0), orig_len, BP_OVERLAP_FRAMES, hop_size)
    min_frames = max(2, int(round((min_note_len_ms / 1000.0) * BP_FPS)))

    events_sec = _decode_bp_notes(
        frames,
        onsets,
        onset_thresh=onset_threshold,
        frame_thresh=frame_threshold,
        min_note_frames=min_frames,
        min_freq_hz=min_freq_hz,
        max_freq_hz=max_freq_hz,
    )

    if output_mid.strip():
        _write_midi_file(events_sec, detected_bpm, os.path.abspath(os.path.expanduser(output_mid)))

    beats_per_sec = detected_bpm / 60.0
    gridline_notes: list[dict[str, Any]] = []
    for t0, t1, pitch, amp in events_sec[:max_notes]:
        start_b = t0 * beats_per_sec
        len_b = max(0.125, (t1 - t0) * beats_per_sec)
        if quantize_beats and quantize_beats > 0:
            start_b = round(start_b / quantize_beats) * quantize_beats
            len_b = max(quantize_beats, round(len_b / quantize_beats) * quantize_beats)
        gridline_notes.append(
            {
                "start": round(start_b, 4),
                "length": round(len_b, 4),
                "pitch": int(pitch),
                "velocity": max(25, min(127, int(round(amp * 127)))),
            }
        )

    pitches = [n["pitch"] for n in gridline_notes]
    return json.dumps(
        {
            "audio_path": abs_in,
            "bpm_used": round(detected_bpm, 2),
            "total_notes_detected": len(events_sec),
            "returned_notes": len(gridline_notes),
            "midi_file": os.path.abspath(os.path.expanduser(output_mid)) if output_mid.strip() else None,
            "pitch_range_midi": [min(pitches), max(pitches)] if pitches else None,
            "gridline_notes": gridline_notes,
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def analyze_song_for_rap_flip(audio_path: str, target_rap_bpm: float = 140.0) -> str:
    """
    Analyse un classique VGM/morceau source (BPM, tonalité, grille d'accords par mesure,
    points de découpe) et propose un plan complet de flip Rap/Trap/Boom-Bap dans Gridline.
    """
    abs_in = os.path.abspath(os.path.expanduser(audio_path))
    if not os.path.isfile(abs_in):
        raise FileNotFoundError(f"Fichier introuvable : {abs_in}")

    y, sr = librosa.load(abs_in, sr=44100, mono=True)
    duration_sec = round(len(y) / float(sr), 3)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    src_bpm = round(float(np.atleast_1d(tempo)[0]), 2)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if _HAS_ESSENTIA:
        key, scale, key_conf = es.KeyExtractor(sampleRate=float(sr))(y.astype(np.float32))
    else:
        key, scale, key_conf = "C", "minor", 0.0
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    # Estimation d'accords toutes les 4 pulsations (1 mesure 4/4)
    chords_by_bar: list[dict[str, Any]] = []
    if len(beat_frames) >= 4:
        for bar_idx, b_start in enumerate(range(0, len(beat_frames) - 3, 4), 1):
            f0 = int(beat_frames[b_start])
            f1 = int(beat_frames[min(len(beat_frames) - 1, b_start + 4)])
            if f1 > f0:
                mean_c = np.mean(chroma[:, f0:f1], axis=1)
                chords_by_bar.append(
                    {
                        "bar": bar_idx,
                        "start_sec": round(float(beat_times[b_start]), 3),
                        "chord": _match_chord(mean_c),
                    }
                )
            if bar_idx >= 32:
                break

    # Calcul des options de tempo pour Rap (Trap ~130-150 ou Boom-Bap ~85-95)
    candidates = [src_bpm, src_bpm * 2.0, src_bpm / 2.0]
    best_src = min(candidates, key=lambda c: abs(c - target_rap_bpm))
    stretch_ratio = round(target_rap_bpm / max(best_src, 1.0), 4)
    semitones_if_repitch = round(12.0 * math.log2(max(stretch_ratio, 0.25)), 2)

    return json.dumps(
        {
            "source_file": abs_in,
            "duration_seconds": duration_sec,
            "detected_bpm": src_bpm,
            "key": f"{key} {scale}",
            "key_confidence": round(float(key_conf), 3),
            "chords_first_bars": chords_by_bar[:16],
            "rap_flip_plan": {
                "target_bpm": target_rap_bpm,
                "matched_source_bpm": round(best_src, 2),
                "time_stretch_ratio": stretch_ratio,
                "semitones_if_vinyl_repitch": semitones_if_repitch,
                "recommended_workflow": [
                    "1. Appeler split_stems(audio_path, output_dir, model='htdemucs_6s') pour isoler piano/guitar/other/bass.",
                    "2. Appeler transcribe_audio_to_midi sur le stem mélodique (other/piano/guitar) pour récupérer la grille MIDI exacte.",
                    "3. Charger le stem découpé dans Gridline via gl.seltmann (Kerf/Seltmann) ou ré-orchestrer le MIDI avec Surge XT / Lattice / Plasma.",
                    "4. Filtrer le sample sous 140 Hz (ZL Equalizer 2 / TDR Nova / gl.eq) et ajouter CHOW Tape Model pour la patine.",
                    f"5. Poser une 808 accordée en {key} {scale} sur les fondamentales des accords détectés + drums de mcp-audio-index.",
                ],
            },
        },
        ensure_ascii=False,
        indent=2,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
