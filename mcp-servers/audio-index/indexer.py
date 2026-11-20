"""Indexeur audio hybride (Essentia + métadonnées musicales + embeddings CLAP optionnels)."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import essentia

essentia.log.infoActive = False
essentia.log.warningActive = False
import essentia.standard as es
import librosa
import numpy as np
import soundfile as sf

STUDIO_DIR = os.environ.get("AI_STUDIO_DIR") or str(Path(__file__).resolve().parents[2])
DEFAULT_SAMPLES_DIR = os.path.join(STUDIO_DIR, "samples")
DEFAULT_DB_PATH = os.path.join(STUDIO_DIR, "mcp-servers/audio-index/samples.db")
AUDIO_EXTS = {".wav", ".ogg", ".flac", ".mp3", ".aiff", ".aif"}
MIDI_EXTS = {".mid", ".midi"}

CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("808", ("808", "sub", "reese")),
    ("kick", ("kick", "bd", "bassdrum")),
    ("snare", ("snare", "sd", "rimshot", "rim")),
    ("clap", ("clap", "cp")),
    ("open_hat", ("open hat", "open_hat", "openhat", "oh", "open-hat")),
    ("hat", ("hihat", "hi-hat", "hi_hat", "hat", "hh", "ch")),
    ("perc", ("perc", "shaker", "tamb", "conga", "bongo", "tom", "cowbell", "crash", "ride", "cymbal")),
    ("vocal", ("vocal", "acapella", "voice", "vox", "chant", "choir")),
    ("guitar", ("guitar", "guit", "gtr")),
    ("piano", ("piano", "keys", "rhodes", "epiano", "organ", "chord")),
    ("synth", ("synth", "pad", "lead", "pluck", "arp", "bell", "mallet", "brass", "strings")),
    ("fx", ("fx", "riser", "sweep", "impact", "downlifter", "uplifter", "glitch", "noise", "transition")),
    ("bass", ("bass",)),
    ("loop", ("loop", "stem", "drumless", "instrumental", "full")),
]

KEY_REGEX = re.compile(
    r"(?<![a-zA-Z0-9])([A-G](?:#|b)?)\s*(maj|major|min|minor|m)(?![a-zA-Z0-9])",
    re.IGNORECASE,
)
BPM_REGEX = re.compile(r"(?:^|[\s_\-\[(])(\d{2,3})\s*bpm(?:$|[\s_\-\])\.])", re.IGNORECASE)
BPM_STANDALONE_REGEX = re.compile(r"(?:^|[\s_\-])(6\d|7\d|8\d|9\d|1\d\d|200)(?:[\s_\-\.]|$)")

# Équivalence enharmonique pour correspondance harmonique
NOTE_TO_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11,
}


def init_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            pack_name TEXT NOT NULL,
            rel_path TEXT NOT NULL,
            ext TEXT NOT NULL,
            file_mtime REAL NOT NULL,
            file_size INTEGER NOT NULL,
            category TEXT NOT NULL,
            is_loop INTEGER NOT NULL,
            duration_sec REAL NOT NULL,
            sample_rate INTEGER NOT NULL,
            channels INTEGER NOT NULL,
            peak_db REAL,
            rms_db REAL,
            bpm REAL,
            musical_key TEXT,
            key_confidence REAL,
            spectral_centroid_hz REAL,
            brightness_band TEXT,
            attack_ms REAL,
            decay_profile TEXT,
            tags TEXT NOT NULL,
            clap_embedding BLOB
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON samples(category);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bpm ON samples(bpm);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_key ON samples(musical_key);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_loop ON samples(is_loop);")
    conn.commit()
    return conn


def _infer_category_and_pack(rel_path: str, filename: str, duration_sec: float) -> tuple[str, str, bool]:
    parts = Path(rel_path).parts
    pack_name = parts[0] if len(parts) > 1 else "root"
    haystack = f"{rel_path} {filename}".lower()

    is_loop = duration_sec >= 2.2 or any(
        k in haystack for k in ("loop", "stem", "full", "drumless", "instrumental", "acapella")
    )
    if any(k in haystack for k in ("one shot", "oneshot", "one-shot", "hit", "drum kit", "808s", "kicks", "snares", "claps", "hi hats", "hi-hats")):
        if duration_sec < 2.5:
            is_loop = False

    category = "other"
    # Cherche d'abord dans le nom du fichier puis le chemin complet
    for target in (filename.lower(), haystack):
        for cat, kws in CATEGORY_RULES:
            if any(kw in target for kw in kws):
                category = cat
                break
        if category != "other":
            break

    if category == "other" and is_loop:
        category = "loop"
    return category, pack_name, is_loop


def _parse_metadata_from_name(text: str) -> tuple[float | None, str | None]:
    bpm: float | None = None
    m_bpm = BPM_REGEX.search(text)
    if m_bpm:
        val = int(m_bpm.group(1))
        if 50 <= val <= 220:
            bpm = float(val)
    else:
        m2 = BPM_STANDALONE_REGEX.search(text)
        if m2:
            val = int(m2.group(1))
            if 60 <= val <= 200:
                bpm = float(val)

    key_str: str | None = None
    m_key = KEY_REGEX.search(text)
    if m_key:
        root = m_key.group(1).capitalize()
        if len(root) == 2 and root[1] == "b":
            root = root[0] + "b"
        qual = m_key.group(2).lower()
        mode = "minor" if qual in ("m", "min", "minor") else "major"
        key_str = f"{root} {mode}"
    return bpm, key_str


def _load_mono_audio(path: str, target_sr: int = 44100, max_seconds: float = 15.0) -> tuple[np.ndarray, int, int, float]:
    """Charge l'audio en mono float32; fallback ffmpeg si format non géré par soundfile."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mp3":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_wav = tf.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-t", str(max_seconds), "-i", path, "-ac", "1", "-ar", str(target_sr), tmp_wav],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            y, sr = sf.read(tmp_wav, dtype="float32", always_2d=False)
            info = sf.info(tmp_wav)
            # Durée exacte via ffprobe si besoin, sinon approximation
            return y.astype(np.float32), int(sr), 2, float(len(y) / sr)
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

    info = sf.info(path)
    total_dur = float(info.frames / info.samplerate) if info.samplerate > 0 else 0.0
    max_frames = int(max_seconds * info.samplerate) if info.samplerate > 0 else -1
    y, sr = sf.read(path, frames=max_frames, dtype="float32", always_2d=True)
    channels = int(y.shape[1])
    mono = np.mean(y, axis=1).astype(np.float32)
    if sr != target_sr and len(mono) > 16:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr).astype(np.float32)
        sr = target_sr
    return mono, int(sr), channels, total_dur


def _brightness_band(centroid_hz: float) -> str:
    if centroid_hz < 120:
        return "sub"
    if centroid_hz < 350:
        return "bass"
    if centroid_hz < 900:
        return "low_mid"
    if centroid_hz < 2800:
        return "mid"
    if centroid_hz < 6500:
        return "high_mid"
    return "air"


def analyze_sample(path: str, rel_path: str) -> dict[str, Any]:
    """Extrait métadonnées, tonalité, BPM, centroïde spectral et profil transitoire."""
    filename = os.path.basename(path)
    ext = os.path.splitext(filename)[1].lower()
    st = os.stat(path)

    if ext in MIDI_EXTS:
        cat, pack_name, _ = _infer_category_and_pack(rel_path, filename, 0.0)
        bpm_tag, key_tag = _parse_metadata_from_name(f"{rel_path} {filename}")
        return {
            "path": path,
            "filename": filename,
            "pack_name": pack_name,
            "rel_path": rel_path,
            "ext": ext,
            "file_mtime": st.st_mtime,
            "file_size": st.st_size,
            "category": "midi",
            "is_loop": 1,
            "duration_sec": 0.0,
            "sample_rate": 0,
            "channels": 0,
            "peak_db": None,
            "rms_db": None,
            "bpm": bpm_tag,
            "musical_key": key_tag,
            "key_confidence": 1.0 if key_tag else 0.0,
            "spectral_centroid_hz": None,
            "brightness_band": None,
            "attack_ms": None,
            "decay_profile": "midi",
            "tags": f"midi {cat} {pack_name} {filename}".lower(),
        }

    mono, sr, channels, duration_sec = _load_mono_audio(path)
    if len(mono) < 64:
        raise ValueError("Audio trop court")

    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono**2)))
    peak_db = round(20.0 * np.log10(max(peak, 1e-7)), 2)
    rms_db = round(20.0 * np.log10(max(rms, 1e-7)), 2)

    # Centroïde spectral via Essentia
    centroid = float(es.SpectralCentroidTime(sampleRate=float(sr))(mono))
    brightness = _brightness_band(centroid)

    # Profil d'attaque et decay
    abs_env = np.abs(mono)
    peak_idx = int(np.argmax(abs_env))
    attack_ms = round((peak_idx / float(sr)) * 1000.0, 1)
    if duration_sec < 0.65 and attack_ms < 35.0:
        decay_profile = "punchy"
    elif duration_sec < 1.8:
        decay_profile = "medium"
    else:
        decay_profile = "sustained"

    category, pack_name, is_loop = _infer_category_and_pack(rel_path, filename, duration_sec)
    bpm_tag, key_tag = _parse_metadata_from_name(f"{rel_path} {filename}")

    # Tonalité via Essentia KeyExtractor (si pas déjà fiable ou pour compléter)
    key_conf = 0.0
    musical_key = key_tag
    try:
        k, scale, strength = es.KeyExtractor(sampleRate=float(sr))(mono)
        key_conf = round(float(strength), 3)
        if not musical_key and strength >= 0.35:
            musical_key = f"{k} {scale}"
    except Exception:
        pass

    # BPM pour les boucles (> 1.8s) si absent du nom
    bpm = bpm_tag
    if bpm is None and (is_loop or duration_sec >= 2.0) and len(mono) >= sr * 1.5:
        try:
            tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
            t_val = float(np.atleast_1d(tempo)[0])
            if 55.0 <= t_val <= 210.0:
                bpm = round(t_val, 1)
        except Exception:
            pass

    tags = " ".join(
        filter(
            None,
            [
                category,
                pack_name,
                brightness,
                decay_profile,
                "loop" if is_loop else "oneshot",
                musical_key or "",
                f"{int(round(bpm))}bpm" if bpm else "",
                rel_path.replace("/", " ").replace("_", " ").replace("-", " "),
            ],
        )
    ).lower()

    return {
        "path": path,
        "filename": filename,
        "pack_name": pack_name,
        "rel_path": rel_path,
        "ext": ext,
        "file_mtime": st.st_mtime,
        "file_size": st.st_size,
        "category": category,
        "is_loop": 1 if is_loop else 0,
        "duration_sec": round(duration_sec, 3),
        "sample_rate": sr,
        "channels": channels,
        "peak_db": peak_db,
        "rms_db": rms_db,
        "bpm": bpm,
        "musical_key": musical_key,
        "key_confidence": key_conf,
        "spectral_centroid_hz": round(centroid, 1),
        "brightness_band": brightness,
        "attack_ms": attack_ms,
        "decay_profile": decay_profile,
        "tags": tags,
    }


def _iter_media_files(samples_dir: str) -> list[tuple[str, str]]:
    root_abs = os.path.abspath(samples_dir)
    results: list[tuple[str, str]] = []
    seen_real: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root_abs, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if d != "__MACOSX" and not d.startswith("."))
        for fname in sorted(filenames):
            if fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in AUDIO_EXTS and ext not in MIDI_EXTS:
                continue
            full = os.path.join(dirpath, fname)
            real = os.path.realpath(full)
            if real in seen_real:
                continue
            seen_real.add(real)
            rel = os.path.relpath(full, root_abs)
            results.append((full, rel))
    return results


def _compute_clap_embeddings(conn: sqlite3.Connection, batch_limit: int = 0, batch_size: int = 32) -> int:
    """Calcule les embeddings audio CLAP par lots (laion/clap-htsat-unfused) pour les samples manquants."""
    import torch
    from transformers import ClapModel, ClapProcessor

    rows = conn.execute(
        "SELECT id, path FROM samples WHERE clap_embedding IS NULL AND ext != '.mid' AND ext != '.midi' AND duration_sec > 0.05"
    ).fetchall()
    if batch_limit > 0:
        rows = rows[:batch_limit]
    if not rows:
        return 0

    model_id = "laion/clap-htsat-unfused"
    try:
        processor = ClapProcessor.from_pretrained(model_id, local_files_only=True)
        model = ClapModel.from_pretrained(model_id, local_files_only=True)
    except Exception:
        processor = ClapProcessor.from_pretrained(model_id)
        model = ClapModel.from_pretrained(model_id)
    model.eval()

    updated = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        valid_ids: list[int] = []
        audios: list[np.ndarray] = []
        for row in chunk:
            try:
                y, _, _, _ = _load_mono_audio(row["path"], target_sr=48000, max_seconds=10.0)
                if len(y) > 32:
                    audios.append(y)
                    valid_ids.append(row["id"])
            except Exception:
                continue
        if not audios:
            continue

        inputs = processor(audio=audios, sampling_rate=48000, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = model.get_audio_features(**inputs)
            if not isinstance(feats, torch.Tensor):
                feats = getattr(feats, "pooler_output", getattr(feats, "last_hidden_state", feats[0]))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        vecs = feats.cpu().numpy().astype(np.float32)
        for sid, vec in zip(valid_ids, vecs):
            conn.execute("UPDATE samples SET clap_embedding = ? WHERE id = ?", (vec.tobytes(), sid))
            updated += 1
        conn.commit()
    return updated


def index_directory(
    samples_dir: str = DEFAULT_SAMPLES_DIR,
    db_path: str = DEFAULT_DB_PATH,
    force: bool = False,
    with_clap: bool = False,
) -> dict[str, int]:
    """Indexe tous les samples du dossier dans SQLite (incrémental par mtime/size)."""
    conn = init_db(db_path)
    existing = {
        r["path"]: (r["file_mtime"], r["file_size"])
        for r in conn.execute("SELECT path, file_mtime, file_size FROM samples")
    }
    files = _iter_media_files(samples_dir)
    indexed = 0
    skipped = 0
    errors = 0

    for full, rel in files:
        try:
            st = os.stat(full)
            prev = existing.get(full)
            if not force and prev and abs(prev[0] - st.st_mtime) < 1e-3 and prev[1] == st.st_size:
                skipped += 1
                continue
            meta = analyze_sample(full, rel)
            cols = ", ".join(meta.keys())
            placeholders = ", ".join(f":{k}" for k in meta.keys())
            updates = ", ".join(f"{k}=excluded.{k}" for k in meta.keys() if k != "path")
            conn.execute(
                f"INSERT INTO samples ({cols}) VALUES ({placeholders}) "
                f"ON CONFLICT(path) DO UPDATE SET {updates}",
                meta,
            )
            indexed += 1
            if indexed % 100 == 0:
                conn.commit()
        except Exception:
            errors += 1
    conn.commit()

    clap_count = _compute_clap_embeddings(conn) if with_clap else 0
    total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    conn.close()
    return {"total": total, "indexed": indexed, "skipped": skipped, "errors": errors, "clap_embedded": clap_count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexeur de samples audio AI-Studio")
    parser.add_argument("--samples-dir", default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--with-clap", action="store_true")
    args = parser.parse_args()
    stats = index_directory(args.samples_dir, args.db_path, force=args.force, with_clap=args.with_clap)
    print(stats)
