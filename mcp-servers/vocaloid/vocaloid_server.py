"""Serveur MCP de synthèse vocale pour Gridline (OpenUTAU .ustx + aperçu vocal synthétisé)."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import yaml
from mcp.server.mcpserver import MCPServer

STUDIO_DIR = os.environ.get("AI_STUDIO_DIR") or str(Path(__file__).resolve().parents[2])
VOICEBANKS_DIRS = [
    os.path.join(STUDIO_DIR, "voicebanks"),
    os.path.expanduser("~/.local/share/OpenUtau/Singers"),
]
OPENUTAU_BIN = os.path.join(STUDIO_DIR, "tools/OpenUtau/OpenUtau")

mcp = MCPServer(
    name="vocaloid",
    instructions=(
        "Automatisation vocale pour Gridline : syllabation des paroles, génération de projets "
        "OpenUTAU/DiffSinger (.ustx), gestion des voicebanks et synthèse d'aperçus vocaux "
        "calés en BPM et hauteur MIDI vers .wav."
    ),
)

FRENCH_VOWELS = "aeiouyàâäéèêëîïôöùûüœæ"
EN_VOWELS = "aeiouy"


def _midi_to_hz(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))


def _syllabify_word(word: str, language: str = "fr") -> list[str]:
    clean = re.sub(r"[^\w'’-]", "", word, flags=re.UNICODE)
    if not clean:
        return []
    vowels = FRENCH_VOWELS if language.startswith("fr") else EN_VOWELS
    # Découpage simple par noyaux vocaliques + consonnes d'attaque
    pattern = re.compile(rf"[^ {vowels}]*[{vowels}]+(?:[^ {vowels}](?![{vowels}]))?", re.IGNORECASE)
    parts = [m.group(0) for m in pattern.finditer(clean) if m.group(0)]
    if not parts:
        return [clean]
    joined = "".join(parts)
    if len(joined) < len(clean):
        parts[-1] += clean[len(joined):]
    return parts


@mcp.tool()
def list_voicebanks() -> str:
    """Liste les banques de voix OpenUTAU / DiffSinger trouvées localement."""
    found: list[dict[str, Any]] = []
    for base in VOICEBANKS_DIRS:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            markers = [f for f in ("character.yaml", "character.txt", "dsconfig.yaml") if f in files]
            if markers:
                found.append(
                    {
                        "name": os.path.basename(root),
                        "path": root,
                        "is_diffsinger": "dsconfig.yaml" in files,
                        "metadata_files": markers,
                    }
                )
    return json.dumps(
        {
            "openutau_installed": os.path.isfile(OPENUTAU_BIN),
            "openutau_binary": OPENUTAU_BIN,
            "search_dirs": VOICEBANKS_DIRS,
            "voicebanks": found,
            "note": (
                "Aucune voicebank DiffSinger détectée : dépose un dossier voicebank dans "
                f"{VOICEBANKS_DIRS[0]} ou utilise synthesize_vocal_preview pour générer un guide vocal immédiat."
                if not found
                else f"{len(found)} banque(s) trouvée(s)."
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def syllabify_lyrics(text: str, language: str = "fr") -> str:
    """Découpe des paroles en syllabes chantables pour les notes d'un fichier .ustx."""
    lines_out: list[dict[str, Any]] = []
    all_syllables: list[str] = []
    for line in text.splitlines():
        words = line.strip().split()
        if not words:
            continue
        syls: list[str] = []
        for w in words:
            syls.extend(_syllabify_word(w, language=language))
        lines_out.append({"line": line.strip(), "syllables": syls})
        all_syllables.extend(syls)
    return json.dumps(
        {
            "language": language,
            "total_syllables": len(all_syllables),
            "syllables": all_syllables,
            "lines": lines_out,
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def create_ustx(
    output_path: str,
    notes: list[dict[str, Any]],
    bpm: float = 120.0,
    time_sig_num: int = 4,
    time_sig_den: int = 4,
    singer: str = "",
    phonemizer: str = "OpenUtau.Core.DiffSinger.DiffSingerFrenchPhonemizer",
    track_name: str = "Lead Vocal",
) -> str:
    """
    Crée un projet OpenUTAU/DiffSinger (.ustx) au format YAML.
    Chaque note : {"lyric": str, "start_beat": float, "duration_beats": float, "pitch_midi": int}.
    """
    ppq = 480
    abs_out = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(os.path.dirname(abs_out), exist_ok=True)

    ustx_notes: list[dict[str, Any]] = []
    for item in sorted(notes, key=lambda n: float(n["start_beat"])):
        pos_ticks = max(0, int(round(float(item["start_beat"]) * ppq)))
        dur_ticks = max(60, int(round(float(item["duration_beats"]) * ppq)))
        tone = max(24, min(108, int(item.get("pitch_midi", 60))))
        lyric = str(item.get("lyric", "la")).strip() or "la"
        ustx_notes.append(
            {
                "position": pos_ticks,
                "duration": dur_ticks,
                "tone": tone,
                "lyric": lyric,
                "pitch": {
                    "data": [
                        {"x": -40.0, "y": 0.0, "shape": "io"},
                        {"x": 40.0, "y": 0.0, "shape": "io"},
                    ],
                    "snap_first": True,
                },
                "vibrato": {
                    "length": 60.0 if dur_ticks >= 480 else 0.0,
                    "period": 175.0,
                    "depth": 22.0,
                    "in": 20.0,
                    "out": 20.0,
                    "shift": 0.0,
                    "drift": 0.0,
                    "vol_link": 0.0,
                },
            }
        )

    doc = {
        "name": os.path.splitext(os.path.basename(abs_out))[0],
        "comment": "Généré par AI-Studio mcp-vocaloid",
        "output_dir": "Vocal",
        "cache_dir": "UCache",
        "ustx_version": "0.6",
        "resolution": ppq,
        "bpm": float(bpm),
        "beat_per_bar": int(time_sig_num),
        "beat_unit": int(time_sig_den),
        "expressions": {},
        "exp_selectors": ["dyn", "pitd", "clr", "eng", "vel"],
        "exp_primary": 0,
        "exp_secondary": 1,
        "key": 0,
        "time_signatures": [{"bar_position": 0, "beat_per_bar": int(time_sig_num), "beat_unit": int(time_sig_den)}],
        "tempos": [{"position": 0, "bpm": float(bpm)}],
        "tracks": [
            {
                "singer": singer,
                "phonemizer": phonemizer,
                "renderer_settings": {"renderer": "DIFFSINGER"} if "diffsinger" in phonemizer.lower() else {},
                "track_name": track_name,
                "track_color": "Blue",
                "mute": False,
                "solo": False,
                "volume": 0.0,
                "pan": 0.0,
                "voice_color_names": [""],
            }
        ],
        "voice_parts": [
            {
                "name": track_name,
                "comment": "",
                "track_no": 0,
                "position": 0,
                "notes": ustx_notes,
                "curves": [],
            }
        ],
        "wave_parts": [],
    }

    with open(abs_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)

    return json.dumps(
        {
            "ustx_path": abs_out,
            "note_count": len(ustx_notes),
            "bpm": bpm,
            "openutau_installed": os.path.isfile(OPENUTAU_BIN),
        },
        ensure_ascii=False,
        indent=2,
    )


def _synth_syllable_espeak(lyric: str, language: str, base_pitch_hz: float, sr: int = 44100) -> np.ndarray:
    voice = "fr+f3" if language.startswith("fr") else "en+f3"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp_wav = tf.name
    try:
        subprocess.run(
            ["espeak-ng", "-v", voice, "-s", "130", "-p", "50", "-w", tmp_wav, lyric],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        y, file_sr = sf.read(tmp_wav, dtype="float32", always_2d=False)
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    y, _ = librosa.effects.trim(y, top_db=28)
    if len(y) < 256:
        return np.zeros(int(0.1 * sr), dtype=np.float32)

    # Estime la F0 moyenne de la syllabe parlée pour la transposer sur la note MIDI cible
    f0 = librosa.yin(y, fmin=70.0, fmax=420.0, sr=sr)
    voiced = f0[(f0 > 75.0) & (f0 < 400.0)]
    src_f0 = float(np.median(voiced)) if len(voiced) > 0 else 185.0
    semis = 12.0 * math.log2(max(base_pitch_hz, 40.0) / max(src_f0, 40.0))
    semis = max(-18.0, min(18.0, semis))
    if abs(semis) > 0.25:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semis)
    return y.astype(np.float32)


def _load_notes_from_ustx(ustx_path: str) -> tuple[list[dict[str, Any]], float]:
    with open(os.path.abspath(os.path.expanduser(ustx_path)), "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    ppq = float(doc.get("resolution", 480))
    bpm = float(doc.get("bpm", 120.0))
    notes: list[dict[str, Any]] = []
    for vp in doc.get("voice_parts", []):
        offset_beats = float(vp.get("position", 0)) / ppq
        for n in vp.get("notes", []):
            notes.append(
                {
                    "lyric": n.get("lyric", "la"),
                    "start_beat": offset_beats + float(n["position"]) / ppq,
                    "duration_beats": float(n["duration"]) / ppq,
                    "pitch_midi": int(n["tone"]),
                }
            )
    return notes, bpm


@mcp.tool()
def synthesize_vocal_preview(
    output_wav: str,
    notes: list[dict[str, Any]] | None = None,
    ustx_path: str | None = None,
    bpm: float = 120.0,
    language: str = "fr",
    pitch_offset_semitones: float = 0.0,
) -> str:
    """
    Synthétise un guide vocal chanté (.wav) calé sur le BPM et les hauteurs MIDI
    (à partir de notes ou d'un fichier .ustx) pour import direct dans Gridline.
    """
    if ustx_path:
        loaded_notes, loaded_bpm = _load_notes_from_ustx(ustx_path)
        notes = loaded_notes
        bpm = loaded_bpm
    if not notes:
        raise ValueError("Fournir 'notes' ou un 'ustx_path' contenant au moins une note.")

    sr = 44100
    sec_per_beat = 60.0 / float(bpm)
    max_end_sec = max((float(n["start_beat"]) + float(n["duration_beats"])) * sec_per_beat for n in notes)
    mix = np.zeros(int((max_end_sec + 0.5) * sr), dtype=np.float32)

    for item in notes:
        start_sec = float(item["start_beat"]) * sec_per_beat
        dur_sec = max(0.08, float(item["duration_beats"]) * sec_per_beat)
        target_hz = _midi_to_hz(float(item.get("pitch_midi", 60)) + float(pitch_offset_semitones))
        lyric = str(item.get("lyric", "la")).strip() or "la"

        syl = _synth_syllable_espeak(lyric, language=language, base_pitch_hz=target_hz, sr=sr)
        target_samples = max(int(0.06 * sr), int(dur_sec * sr))
        if len(syl) > 64:
            rate = max(0.25, min(3.5, len(syl) / float(target_samples)))
            if abs(rate - 1.0) > 0.08:
                syl = librosa.effects.time_stretch(syl, rate=rate)

        if len(syl) > target_samples:
            syl = syl[:target_samples]
        elif len(syl) < target_samples:
            # Prolonge doucement la queue en fondu
            pad = np.zeros(target_samples - len(syl), dtype=np.float32)
            syl = np.concatenate([syl, pad])

        fade = min(int(0.015 * sr), len(syl) // 4)
        if fade > 1:
            syl[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            syl[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

        start_idx = int(round(start_sec * sr))
        end_idx = min(len(mix), start_idx + len(syl))
        mix[start_idx:end_idx] += syl[: end_idx - start_idx]

    peak = float(np.max(np.abs(mix)))
    if peak > 1e-6:
        mix = mix * (0.7079 / peak)  # -3 dBFS

    abs_wav = os.path.abspath(os.path.expanduser(output_wav))
    os.makedirs(os.path.dirname(abs_wav), exist_ok=True)
    sf.write(abs_wav, mix, sr, subtype="PCM_24")

    return json.dumps(
        {
            "wav_path": abs_wav,
            "sample_rate": sr,
            "duration_seconds": round(len(mix) / sr, 3),
            "bpm": bpm,
            "note_count": len(notes),
            "gridline_tip": (
                "Charger dans Gridline via add_channel(plugin='gl.sampler') puis "
                f"patch(path='channel/N', set={{'plugin': {{'data': '{abs_wav}'}}}}) "
                "et ajouter gl.pitcher (Plumb) / gl.vocoder (Cipher) / gl.reverb si souhaité."
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def open_in_openutau(ustx_path: str) -> str:
    """Ouvre un fichier .ustx dans OpenUTAU GUI (pour rendu DiffSinger haute fidélité)."""
    abs_ustx = os.path.abspath(os.path.expanduser(ustx_path))
    if not os.path.isfile(OPENUTAU_BIN):
        return json.dumps({"error": f"OpenUTAU introuvable : {OPENUTAU_BIN}"}, ensure_ascii=False)
    if not os.path.isfile(abs_ustx):
        return json.dumps({"error": f"Fichier .ustx introuvable : {abs_ustx}"}, ensure_ascii=False)
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return json.dumps(
            {
                "started": False,
                "reason": "Pas de session graphique DISPLAY/WAYLAND_DISPLAY dans l'environnement MCP.",
                "manual_command": f'"{OPENUTAU_BIN}" "{abs_ustx}"',
            },
            ensure_ascii=False,
        )
    subprocess.Popen([OPENUTAU_BIN, abs_ustx], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.dumps({"started": True, "ustx_path": abs_ustx}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
