"""Indexeur audio hybride (Essentia + métadonnées musicales + embeddings CLAP optionnels)."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import essentia

    essentia.log.infoActive = False
    essentia.log.warningActive = False
    import essentia.standard as es

    _HAS_ESSENTIA = True
except Exception:  # pragma: no cover - depends on native build availability
    es = None
    _HAS_ESSENTIA = False
import librosa
import numpy as np
import soundfile as sf

STUDIO_DIR = os.environ.get("AI_STUDIO_DIR") or str(Path(__file__).resolve().parents[2])
SCRIPTS_DIR = os.path.join(STUDIO_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

DEFAULT_SAMPLES_DIR = os.path.join(STUDIO_DIR, "samples")
DEFAULT_DB_PATH = os.path.join(STUDIO_DIR, "mcp-servers/audio-index/samples.db")
DEFAULT_ANCHORS_NPZ = os.path.join(STUDIO_DIR, "mcp-servers/audio-index/semantic_anchors.npz")
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


DB_SCHEMA_VERSION = 3  # v3.3 #5 : version de schéma samples.db (PRAGMA user_version)


def init_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        cur = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if cur < DB_SCHEMA_VERSION:
            conn.execute(f"PRAGMA user_version={DB_SCHEMA_VERSION}")
            conn.commit()
    except Exception:
        pass
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
            clap_embedding BLOB,
            clap_updated_at REAL
        )
        """
    )
    # Migrate existing samples table if clap_updated_at column is absent
    existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(samples)").fetchall()}
    if "clap_updated_at" not in existing_cols:
        conn.execute("ALTER TABLE samples ADD COLUMN clap_updated_at REAL;")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sample_dsp_metrics (
            sample_id INTEGER PRIMARY KEY REFERENCES samples(id) ON DELETE CASCADE,
            sample_nature TEXT,
            playable_chromatically INTEGER NOT NULL DEFAULT 0,
            f0_hz REAL,
            recommended_root_midi INTEGER,
            recommended_fine_tune_ct INTEGER NOT NULL DEFAULT 0,
            detected_note TEXT,
            centroid_hz REAL,
            r_harsh REAL,
            peak_harsh_db REAL,
            roughness REAL,
            sfm REAL,
            r_harm REAL,
            spectral_flux REAL,
            hnr_db REAL,
            low_mid_corr REAL,
            phase_corr REAL,
            crest_factor_db REAL,
            attack_ms REAL,
            peak_dbfs REAL,
            duration_s REAL,
            f0_std_semitones REAL,
            onset_count INTEGER,
            polyphony_ratio REAL,
            needs_fade INTEGER NOT NULL DEFAULT 0,
            hard_reject INTEGER NOT NULL DEFAULT 0,
            reject_reasons TEXT,
            band_sub REAL,
            band_bass REAL,
            band_lowmid REAL,
            band_mid REAL,
            band_harsh REAL,
            band_air REAL,
            bark_24_json TEXT,
            inharmonic_warnings_json TEXT,
            updated_at REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sample_semantic_cards (
            sample_id INTEGER PRIMARY KEY REFERENCES samples(id) ON DELETE CASCADE,
            clap_vec BLOB,
            top_tags_json TEXT,
            defect_tags_json TEXT,
            timbre_card TEXT,
            indexed_at TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON samples(category);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bpm ON samples(bpm);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_key ON samples(musical_key);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_loop ON samples(is_loop);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dsp_playable ON sample_dsp_metrics(playable_chromatically);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dsp_nature ON sample_dsp_metrics(sample_nature);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dsp_centroid ON sample_dsp_metrics(centroid_hz);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dsp_r_harsh ON sample_dsp_metrics(r_harsh);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dsp_low_mid_corr ON sample_dsp_metrics(low_mid_corr);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dsp_hard_reject ON sample_dsp_metrics(hard_reject);")
    conn.commit()
    return conn


def _fused_tokens(text: str) -> set[str]:
    """Tokenize a path/filename into raw tokens + separator-fused compounds.

    "open_hat 909.wav" -> {"open", "hat", "909", "wav", "openhat", ...} so multi-word
    keywords ("open hat") and compounds ("hi-hat") match without ever matching inside
    unrelated words ("chat" must NOT match "hat", "scalp" must NOT match "cp").
    """
    raw = re.findall(r"[a-z0-9]+", text.lower())
    toks: set[str] = set(raw)
    for w in raw:
        if len(w) > 3 and w.endswith("s"):
            toks.add(w[:-1])  # plural singularization: "kicks" -> "kick", "808s" -> "808"
    toks.add(re.sub(r"[^a-z0-9]+", "", text.lower()))
    for i in range(len(raw)):
        for j in range(i + 1, min(i + 3, len(raw)) + 1):
            toks.add("".join(raw[i : j + 1]))
    return toks


def _kw_fused(kw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", kw.lower())


def _infer_category_and_pack(rel_path: str, filename: str, duration_sec: float) -> tuple[str, str, bool]:
    parts = Path(rel_path).parts
    pack_name = parts[0] if len(parts) > 1 else "root"
    haystack = f"{rel_path} {filename}".lower()

    loop_kws = {"loop", "stem", "full", "drumless", "instrumental", "acapella"}
    oneshot_kws = {"oneshot", "hit", "drumkit", "kicks", "snares", "claps", "hihats"}
    is_loop = duration_sec >= 2.2 or bool(_fused_tokens(haystack) & loop_kws)
    if _fused_tokens(haystack) & oneshot_kws:
        if duration_sec < 2.5:
            is_loop = False

    category = "other"
    # Cherche d'abord dans le nom du fichier puis le chemin complet (correspondance par tokens, jamais par sous-chaîne)
    for target in (filename.lower(), haystack):
        toks = _fused_tokens(target)
        for cat, kws in CATEGORY_RULES:
            if any(_kw_fused(kw) in toks for kw in kws):
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

    # Centroïde spectral via Essentia (repli librosa si Essentia absent)
    if _HAS_ESSENTIA:
        try:
            centroid = float(es.SpectralCentroidTime(sampleRate=float(sr))(mono))
        except Exception:
            centroid = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr)))
    else:
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr)))
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
    if _HAS_ESSENTIA:
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


def _iter_media_files(samples_dir: str) -> tuple[list[tuple[str, str]], list[dict[str, str]]]:
    """Scanne `samples_dir`. Retourne (acceptés [(abs, rel)], rejetés [{path, reason}]).

    v3.3 hotfix : les rejets (dotfiles, extensions composites type `.wav.zpa`, hygiène
    ASSET-05 : magic bytes / taille < 1 Ko, doublons de symlinks) sont désormais
    **comptés et remontés avec leur raison** au lieu d'être silencieux — le rapport
    d'indexation distingue `scanned` / `rejected` / `skipped` (déjà en base).
    """
    root_abs = os.path.abspath(samples_dir)
    results: list[tuple[str, str]] = []
    rejected: list[dict[str, str]] = []
    seen_real: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root_abs, followlinks=True):
        dirnames[:] = sorted(
            d for d in dirnames if d not in ("__MACOSX", "sfz") and not d.startswith(".")
        )
        for fname in sorted(filenames):
            full = os.path.join(dirpath, fname)
            if fname.startswith("."):
                rejected.append({"path": full, "reason": "dotfile"})
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in MIDI_EXTS:
                pass  # MIDI accepté tel quel (pas de métriques DSP audio)
            elif ext in AUDIO_EXTS:
                # v3.1 ASSET-05: extension whitelist + taille + magic bytes (jamais ".wav" in name)
                try:
                    from asset_hygiene import is_audio_file_safe
                    ok, reason = is_audio_file_safe(full)
                except Exception:
                    ok, reason = True, "ok"
                if not ok:
                    rejected.append({"path": full, "reason": reason})
                    continue
            else:
                # Suffixe non média : ne remonter que si le nom contient une extension audio
                # trompeuse (ex. `pack.wav.zpa`, `loop.flac.fst`) — jamais `".wav" in name` seul.
                low = fname.lower()
                if any(f".{e}" in low for e in ("wav", "ogg", "flac", "mp3", "aiff", "aif")):
                    try:
                        from asset_hygiene import is_decodable_audio_file
                        _ok, reason = is_decodable_audio_file(full)
                    except Exception:
                        reason = "unknown"
                    rejected.append({"path": full, "reason": reason or "unknown"})
                continue
            real = os.path.realpath(full)
            if real in seen_real:
                rejected.append({"path": full, "reason": "symlink_duplicate"})
                continue
            seen_real.add(real)
            rel = os.path.relpath(full, root_abs)
            results.append((full, rel))
    return results, rejected


def _compute_dsp_worker(task: tuple[int, str, float | None]) -> tuple[int, dict[str, Any] | None]:
    """Worker computing deterministic DSP metrics (`sample_dsp_metrics` row) without CLAP."""
    sample_id, path, peak_db = task
    try:
        from audio_metrics import analyze_single_file

        entry, _ = analyze_single_file(path, with_clap=False, max_seconds=8.0)
        dsp = entry["dsp"]
        bands = dsp.get("bands") or {}
        return sample_id, {
            "sample_id": sample_id,
            "sample_nature": dsp.get("sample_nature"),
            "playable_chromatically": 1 if dsp.get("playable_chromatically") else 0,
            "f0_hz": dsp.get("detected_f0_hz"),
            "recommended_root_midi": dsp.get("midi_root"),
            "recommended_fine_tune_ct": int(dsp.get("recommended_fine_tune_ct") or 0),
            "detected_note": dsp.get("detected_note"),
            "centroid_hz": dsp.get("spectral_centroid_hz"),
            "r_harsh": dsp.get("r_harsh"),
            "peak_harsh_db": dsp.get("peak_harsh_db"),
            "roughness": dsp.get("roughness"),
            "sfm": dsp.get("sfm"),
            "r_harm": dsp.get("r_harm"),
            "spectral_flux": dsp.get("spectral_flux"),
            "hnr_db": dsp.get("hnr_db"),
            "low_mid_corr": dsp.get("low_mid_corr"),
            "phase_corr": dsp.get("phase_corr"),
            "crest_factor_db": dsp.get("crest_factor_db"),
            "attack_ms": dsp.get("attack_ms"),
            "peak_dbfs": peak_db,
            "duration_s": entry.get("duration_sec"),
            "f0_std_semitones": dsp.get("f0_std_semitones"),
            "onset_count": dsp.get("onset_count"),
            "polyphony_ratio": dsp.get("polyphony_ratio"),
            "needs_fade": 1 if entry.get("needs_fade") else 0,
            "hard_reject": 1 if entry.get("hard_reject") else 0,
            "reject_reasons": json.dumps(entry.get("reject_reasons") or []),
            "band_sub": float(bands.get("sub", 0.0)),
            "band_bass": float(bands.get("bass", 0.0)),
            "band_lowmid": float(bands.get("lowmid", 0.0)),
            "band_mid": float(bands.get("mid", 0.0)),
            "band_harsh": float(bands.get("harsh", 0.0)),
            "band_air": float(bands.get("air", 0.0)),
            "bark_24_json": json.dumps(dsp.get("bark_24") or []),
            "inharmonic_warnings_json": json.dumps(dsp.get("inharmonic_warnings") or []),
            "updated_at": time.time(),
        }
    except Exception:
        return sample_id, None


def upsert_dsp_metrics(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    cols = ", ".join(row.keys())
    placeholders = ", ".join(f":{k}" for k in row.keys())
    updates = ", ".join(f"{k}=excluded.{k}" for k in row.keys() if k != "sample_id")
    conn.execute(
        f"INSERT INTO sample_dsp_metrics ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT(sample_id) DO UPDATE SET {updates}",
        row,
    )


def populate_dsp_metrics(
    conn: sqlite3.Connection,
    force: bool = False,
    limit: int = 0,
    workers: int = 4,
) -> dict[str, int]:
    """Populate `sample_dsp_metrics` incrementally for audio rows missing DSP analysis."""
    if force:
        sql = "SELECT id, path, peak_db FROM samples WHERE ext != '.mid' AND ext != '.midi'"
    else:
        sql = (
            "SELECT s.id, s.path, s.peak_db FROM samples s "
            "LEFT JOIN sample_dsp_metrics d ON d.sample_id = s.id "
            "WHERE s.ext != '.mid' AND s.ext != '.midi' "
            "AND (d.sample_id IS NULL OR d.updated_at < s.file_mtime)"
        )
    if limit > 0:
        sql += f" LIMIT {int(limit)}"
    tasks = [(int(r["id"]), r["path"], r["peak_db"]) for r in conn.execute(sql).fetchall() if os.path.isfile(r["path"])]
    if not tasks:
        return {"dsp_indexed": 0, "dsp_errors": 0}

    dsp_ok = 0
    dsp_err = 0
    if workers > 1 and len(tasks) > 4:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for _, row in pool.map(_compute_dsp_worker, tasks, chunksize=8):
                if row is not None:
                    upsert_dsp_metrics(conn, row)
                    dsp_ok += 1
                    if dsp_ok % 50 == 0:
                        conn.commit()
                else:
                    dsp_err += 1
    else:
        for t in tasks:
            _, row = _compute_dsp_worker(t)
            if row is not None:
                upsert_dsp_metrics(conn, row)
                dsp_ok += 1
                if dsp_ok % 50 == 0:
                    conn.commit()
            else:
                dsp_err += 1
    conn.commit()
    return {"dsp_indexed": dsp_ok, "dsp_errors": dsp_err}


def _compute_clap_embeddings(conn: sqlite3.Connection, batch_limit: int = 0, batch_size: int = 64) -> int:
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
        now_ts = time.time()
        for sid, vec in zip(valid_ids, vecs):
            conn.execute(
                "UPDATE samples SET clap_embedding = ?, clap_updated_at = ? WHERE id = ?",
                (vec.tobytes(), now_ts, sid),
            )
            updated += 1
        conn.commit()
    return updated


_ANCHORS_CACHE: dict[str, dict[str, Any] | None] = {}

_CATEGORY_FR: dict[str, str] = {
    "808": "808 sub",
    "kick": "Kick",
    "snare": "Snare",
    "clap": "Clap",
    "open_hat": "Hi-hat ouvert",
    "hat": "Hi-hat",
    "perc": "Percussion",
    "vocal": "Voix",
    "guitar": "Guitare",
    "piano": "Piano/Keys",
    "synth": "Synthé",
    "fx": "FX",
    "bass": "Basse",
    "loop": "Boucle",
    "other": "Son",
}


def load_semantic_anchors(anchors_path: str = DEFAULT_ANCHORS_NPZ) -> dict[str, Any] | None:
    """Charge en pur NumPy (0 PyTorch) les ancres sémantiques pré-encodées."""
    abs_p = os.path.abspath(os.path.expanduser(anchors_path))
    if abs_p in _ANCHORS_CACHE:
        return _ANCHORS_CACHE[abs_p]
    if not os.path.isfile(abs_p):
        _ANCHORS_CACHE[abs_p] = None
        return None
    try:
        z = np.load(abs_p, allow_pickle=True)
        emb = np.asarray(z["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-12)
        bundle = {
            "embeddings": emb,
            "embeddings_f16": emb.astype(np.float16),
            "ids": [str(x) for x in z["ids"].tolist()],
            "axes": [str(x) for x in z["axes"].tolist()],
            "fr_labels": [str(x) for x in z["fr_labels"].tolist()],
            "prompts": [str(x) for x in z["prompts"].tolist()],
            "thresholds": [float(x) for x in z["thresholds"].tolist()] if "thresholds" in z else [0.24] * len(emb),
        }
        _ANCHORS_CACHE[abs_p] = bundle
        return bundle
    except Exception:
        _ANCHORS_CACHE[abs_p] = None
        return None


def _truncate_card(text: str, max_len: int = 140) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1].rstrip(" ,;(") + "…"


def build_timbre_card_from_dsp(
    *,
    category: str = "other",
    sample_nature: str | None = None,
    detected_note: str | None = None,
    centroid_hz: float | None = None,
    attack_ms: float | None = None,
    duration_sec: float | None = None,
    r_harsh: float | None = None,
    r_harm: float | None = None,
    roughness: float | None = None,
    low_mid_corr: float | None = None,
    playable_chromatically: bool | None = None,
    comfort_score: float | None = None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Construit une `timbre_card` déterministe (<= 140 chars) + tags à partir des métriques DSP."""
    fam = _CATEGORY_FR.get(str(category or "other").lower(), str(category or "Son").capitalize())
    cent = float(centroid_hz or 1000.0)
    att = float(attack_ms if attack_ms is not None else 15.0)
    rh = float(r_harsh or 0.0)
    harm = float(r_harm if r_harm is not None else 0.5)
    rg = float(roughness or 0.0)
    dur = float(duration_sec or 0.5)

    if cent < 180:
        bri_lbl = "très grave/sub"
        bri_tag = "bri_deep_sub"
    elif cent < 650:
        bri_lbl = "sombre/chaud"
        bri_tag = "bri_dark_warm"
    elif cent < 1800:
        bri_lbl = "chaud/équilibré"
        bri_tag = "bri_warm_mellow"
    elif cent < 3600:
        bri_lbl = "clair/présent"
        bri_tag = "bri_clear_present"
    elif rh > 0.38:
        bri_lbl = "brillant/métallique"
        bri_tag = "bri_harsh_metallic"
    else:
        bri_lbl = "brillant/aéré"
        bri_tag = "bri_bright_shimmer"

    if att <= 12.0 and dur < 0.7:
        env_lbl = "attaque sèche"
        env_tag = "env_sharp_perc"
    elif att <= 30.0:
        env_lbl = "attaque franche"
        env_tag = "env_plucked"
    elif att >= 110.0:
        env_lbl = "attaque lente"
        env_tag = "env_slow_pad"
    else:
        env_lbl = "attaque douce"
        env_tag = "env_soft_attack"

    defects: list[dict[str, Any]] = []
    if rh > 0.36:
        defects.append({"id": "def_piercing_whistle", "label": "aigu dur", "score": round(min(1.0, rh), 3)})
    if rg > 0.28 and category not in ("hat", "open_hat", "perc"):
        defects.append({"id": "def_clipping_dist", "label": "rugueux", "score": round(min(1.0, rg), 3)})
    if low_mid_corr is not None and float(low_mid_corr) < 0.45 and category not in ("hat", "open_hat", "fx"):
        defects.append({"id": "def_hollow_reverb", "label": "phase bas-médiums large", "score": round(1.0 - float(low_mid_corr), 3)})

    tex_lbl = "propre" if not defects else defects[0]["label"]
    if comfort_score is None:
        comfort_score = max(0.0, min(1.0, 1.0 - max(0.0, rh - 0.25) * 1.1 - max(0.0, rg - 0.20) * 0.8))

    nature_str = sample_nature or ("mono_oneshot" if playable_chromatically else ("perc" if category in ("kick", "snare", "clap", "hat", "open_hat", "perc") else "sample"))
    note_part = f" {detected_note}" if detected_note else ""
    card = _truncate_card(f"{fam} {bri_lbl}, {env_lbl}, {tex_lbl} ({nature_str}{note_part}, confort {comfort_score:.2f})")

    top_tags = [
        {"id": bri_tag, "axis": "brightness", "label": bri_lbl, "score": 0.75},
        {"id": env_tag, "axis": "envelope", "label": env_lbl, "score": 0.72},
        {
            "id": "tex_clean_studio" if not defects else "tex_gritty_dist",
            "axis": "texture",
            "label": "harmonique" if harm >= 0.65 else "percussif/bruité",
            "score": round(max(0.5, harm), 3),
        },
    ]
    return card, top_tags, defects


def build_semantic_card_from_clap(
    clap_vec_f32: np.ndarray,
    anchors_bundle: dict[str, Any],
    *,
    category: str = "other",
    sample_nature: str | None = None,
    detected_note: str | None = None,
    playable_chromatically: bool | None = None,
    comfort_score: float | None = None,
) -> tuple[bytes, str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Projette un vecteur audio CLAP sur les ancres pré-encodées en pur NumPy (< 0.1 ms)."""
    v = np.asarray(clap_vec_f32, dtype=np.float32).reshape(-1)
    nrm = float(np.linalg.norm(v))
    if nrm > 1e-12:
        v = v / nrm
    v16_bytes = v.astype(np.float16).tobytes()

    emb = anchors_bundle["embeddings"]
    sims = emb @ v
    axes = anchors_bundle["axes"]
    ids = anchors_bundle["ids"]
    labels = anchors_bundle["fr_labels"]
    thresholds = anchors_bundle["thresholds"]

    by_axis: dict[str, list[tuple[float, int]]] = {}
    defects: list[dict[str, Any]] = []
    pos_candidates: list[tuple[float, int]] = []

    for i, ax in enumerate(axes):
        sc = float(sims[i])
        if ax == "defect_flags":
            if sc >= float(thresholds[i]):
                defects.append({"id": ids[i], "label": labels[i], "score": round(sc, 4)})
        else:
            by_axis.setdefault(ax, []).append((sc, i))
            pos_candidates.append((sc, i))

    defects.sort(key=lambda x: x["score"], reverse=True)
    pos_candidates.sort(key=lambda x: x[0], reverse=True)
    top_tags = [
        {"id": ids[i], "axis": axes[i], "label": labels[i], "score": round(sc, 4)}
        for sc, i in pos_candidates[:5]
    ]

    def _best_label(ax: str, fallback: str = "") -> str:
        items = by_axis.get(ax)
        if not items:
            return fallback
        items.sort(key=lambda x: x[0], reverse=True)
        return labels[items[0][1]]

    inst_lbl = _best_label("instrument_family", _CATEGORY_FR.get(category, "Son"))
    env_lbl = _best_label("envelope", "")
    mood_lbl = _best_label("mood", _best_label("brightness", ""))
    tex_lbl = "propre" if not defects else f"alerte: {defects[0]['label']}"

    parts = [p for p in (inst_lbl, env_lbl, mood_lbl, tex_lbl) if p]
    nature_str = sample_nature or ("mono_oneshot" if playable_chromatically else "sample")
    note_part = f" {detected_note}" if detected_note else ""
    conf_part = f", confort {comfort_score:.2f}" if comfort_score is not None else ""
    card = _truncate_card(f"{', '.join(parts)} ({nature_str}{note_part}{conf_part})")
    return v16_bytes, card, top_tags, defects[:4]


def populate_semantic_cards(
    conn: sqlite3.Connection,
    anchors_path: str = DEFAULT_ANCHORS_NPZ,
    force: bool = False,
    limit: int = 0,
) -> dict[str, int]:
    """Remplit `sample_semantic_cards` en pur NumPy (0 PyTorch) : ancres CLAP si vecteur dispo, sinon DSP -> texte."""
    anchors = load_semantic_anchors(anchors_path)
    if force:
        sql = (
            "SELECT s.id, s.category, s.duration_sec, s.spectral_centroid_hz, s.attack_ms, s.clap_embedding, "
            "d.sample_nature, d.playable_chromatically, d.detected_note, d.centroid_hz, d.attack_ms AS d_attack_ms, "
            "d.r_harsh, d.r_harm, d.roughness, d.low_mid_corr "
            "FROM samples s LEFT JOIN sample_dsp_metrics d ON d.sample_id = s.id "
            "WHERE s.ext NOT IN ('.mid', '.midi')"
        )
    else:
        sql = (
            "SELECT s.id, s.category, s.duration_sec, s.spectral_centroid_hz, s.attack_ms, s.clap_embedding, "
            "d.sample_nature, d.playable_chromatically, d.detected_note, d.centroid_hz, d.attack_ms AS d_attack_ms, "
            "d.r_harsh, d.r_harm, d.roughness, d.low_mid_corr "
            "FROM samples s "
            "LEFT JOIN sample_dsp_metrics d ON d.sample_id = s.id "
            "LEFT JOIN sample_semantic_cards c ON c.sample_id = s.id "
            "WHERE s.ext NOT IN ('.mid', '.midi') "
            "AND (c.sample_id IS NULL OR (s.clap_embedding IS NOT NULL AND c.clap_vec IS NULL))"
        )
    if limit > 0:
        sql += f" LIMIT {int(limit)}"

    rows = conn.execute(sql).fetchall()
    if not rows:
        return {"cards_updated": 0, "cards_with_clap": 0, "cards_dsp_only": 0}

    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    updated = 0
    with_clap_n = 0
    dsp_only_n = 0
    batch_rows: list[tuple[Any, ...]] = []

    for r in rows:
        sid = int(r["id"])
        rh = float(r["r_harsh"] or 0.0)
        rg = float(r["roughness"] or 0.0)
        comfort = max(0.0, min(1.0, 1.0 - max(0.0, rh - 0.25) * 1.1 - max(0.0, rg - 0.20) * 0.8))
        playable = bool(r["playable_chromatically"]) if r["playable_chromatically"] is not None else None

        blob = r["clap_embedding"]
        clap_v16: bytes | None = None
        if blob is not None and anchors is not None:
            vec = np.frombuffer(blob, dtype=np.float32)
            if len(vec) == anchors["embeddings"].shape[1]:
                clap_v16, card, top_tags, defects = build_semantic_card_from_clap(
                    vec,
                    anchors,
                    category=r["category"],
                    sample_nature=r["sample_nature"],
                    detected_note=r["detected_note"],
                    playable_chromatically=playable,
                    comfort_score=comfort,
                )
                with_clap_n += 1
            else:
                card, top_tags, defects = build_timbre_card_from_dsp(
                    category=r["category"],
                    sample_nature=r["sample_nature"],
                    detected_note=r["detected_note"],
                    centroid_hz=r["centroid_hz"] if r["centroid_hz"] is not None else r["spectral_centroid_hz"],
                    attack_ms=r["d_attack_ms"] if r["d_attack_ms"] is not None else r["attack_ms"],
                    duration_sec=r["duration_sec"],
                    r_harsh=r["r_harsh"],
                    r_harm=r["r_harm"],
                    roughness=r["roughness"],
                    low_mid_corr=r["low_mid_corr"],
                    playable_chromatically=playable,
                    comfort_score=comfort,
                )
                dsp_only_n += 1
        else:
            if blob is not None:
                vec = np.frombuffer(blob, dtype=np.float32)
                nrm = float(np.linalg.norm(vec))
                if nrm > 1e-12:
                    clap_v16 = (vec / nrm).astype(np.float16).tobytes()
            card, top_tags, defects = build_timbre_card_from_dsp(
                category=r["category"],
                sample_nature=r["sample_nature"],
                detected_note=r["detected_note"],
                centroid_hz=r["centroid_hz"] if r["centroid_hz"] is not None else r["spectral_centroid_hz"],
                attack_ms=r["d_attack_ms"] if r["d_attack_ms"] is not None else r["attack_ms"],
                duration_sec=r["duration_sec"],
                r_harsh=r["r_harsh"],
                r_harm=r["r_harm"],
                roughness=r["roughness"],
                low_mid_corr=r["low_mid_corr"],
                playable_chromatically=playable,
                comfort_score=comfort,
            )
            dsp_only_n += 1

        batch_rows.append((
            sid,
            clap_v16,
            json.dumps(top_tags, ensure_ascii=False),
            json.dumps(defects, ensure_ascii=False),
            card,
            now_iso,
        ))
        updated += 1

    conn.executemany(
        """
        INSERT INTO sample_semantic_cards (sample_id, clap_vec, top_tags_json, defect_tags_json, timbre_card, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(sample_id) DO UPDATE SET
            clap_vec=excluded.clap_vec,
            top_tags_json=excluded.top_tags_json,
            defect_tags_json=excluded.defect_tags_json,
            timbre_card=excluded.timbre_card,
            indexed_at=excluded.indexed_at
        """,
        batch_rows,
    )
    conn.commit()
    return {"cards_updated": updated, "cards_with_clap": with_clap_n, "cards_dsp_only": dsp_only_n}


def enrich_clap_embeddings(
    db_path: str = DEFAULT_DB_PATH,
    batch_size: int = 64,
    batch_limit: int = 0,
) -> dict[str, int]:
    """Standalone incremental CLAP enrichment for CLI usage (`scripts/index_samples.py --enrich-clap`)."""
    conn = init_db(db_path)
    embedded = _compute_clap_embeddings(conn, batch_limit=batch_limit, batch_size=batch_size)
    cards = populate_semantic_cards(conn, force=False)
    total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    clap_total = conn.execute("SELECT COUNT(*) FROM samples WHERE clap_embedding IS NOT NULL").fetchone()[0]
    conn.close()
    return {
        "total": total,
        "clap_embedded_now": embedded,
        "clap_total": clap_total,
        "cards_updated": cards["cards_updated"],
        "cards_with_clap": cards["cards_with_clap"],
    }


def index_directory(
    samples_dir: str = DEFAULT_SAMPLES_DIR,
    db_path: str = DEFAULT_DB_PATH,
    force: bool = False,
    with_clap: bool = False,
    with_dsp: bool = True,
    limit: int = 0,
    workers: int = 4,
) -> dict[str, int]:
    """Indexe tous les samples du dossier dans SQLite unifié (incrémental par mtime/size, hors `sfz/`)."""
    conn = init_db(db_path)
    existing = {
        r["path"]: (r["file_mtime"], r["file_size"])
        for r in conn.execute("SELECT path, file_mtime, file_size FROM samples")
    }
    files, rejected = _iter_media_files(samples_dir)
    if limit > 0:
        files = files[:limit]
    indexed = 0
    skipped = 0
    errors = 0
    rejected_reasons: dict[str, int] = {}
    for _r in rejected:
        rejected_reasons[_r.get("reason", "unknown")] = rejected_reasons.get(_r.get("reason", "unknown"), 0) + 1

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

    dsp_stats = populate_dsp_metrics(conn, force=force, limit=limit, workers=workers) if with_dsp else {"dsp_indexed": 0, "dsp_errors": 0}
    clap_count = _compute_clap_embeddings(conn) if with_clap else 0
    card_stats = populate_semantic_cards(conn, force=force, limit=limit)
    total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    dsp_total = conn.execute("SELECT COUNT(*) FROM sample_dsp_metrics").fetchone()[0]
    cards_total = conn.execute("SELECT COUNT(*) FROM sample_semantic_cards").fetchone()[0]
    conn.close()
    return {
        "total": total,
        "indexed": indexed,
        "skipped": skipped,
        "errors": errors,
        "scanned": len(files) + len(rejected),
        "rejected": len(rejected),
        "rejected_reasons": rejected_reasons,
        "dsp_indexed": dsp_stats["dsp_indexed"],
        "dsp_errors": dsp_stats["dsp_errors"],
        "dsp_total": dsp_total,
        "clap_embedded": clap_count,
        "cards_updated": card_stats["cards_updated"],
        "cards_total": cards_total,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexeur de samples audio AI-Studio")
    parser.add_argument("--samples-dir", default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--with-clap", action="store_true")
    parser.add_argument("--no-dsp", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    stats = index_directory(
        args.samples_dir,
        args.db_path,
        force=args.force,
        with_clap=args.with_clap,
        with_dsp=not args.no_dsp,
        workers=args.workers,
    )
    print(stats)
