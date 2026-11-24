#!/usr/bin/env python3
"""
v3.1 ASSET-05 — Hygiène des banques de sons, formats non-PCM et garde-fou de licence.

Module partagé importé par `mcp-servers/audio-index/indexer.py`,
`scripts/index_project_samples.py`, `scripts/audit_project_sources.py` et
`scripts/palette_builder.py` :

  - liste blanche stricte d'extensions audio décodables (jamais de `".wav" in filename`,
    toujours `Path(p).suffix.lower()` + rejet des extensions composites `.wav.zpa`, `.fst`, …) ;
  - exclusion des parasites OS (`__MACOSX/`, `._*`, `.DS_Store`, `Thumbs.db`, `desktop.ini`,
    `Icon\\r`, fichiers < 1024 octets) ;
  - vérification rapide d'en-tête audio (magic bytes) avant soundfile/librosa : un fichier
    corrompu est rejeté proprement, sans stacktrace, avec un compteur `skipped_reasons` ;
  - détecteur de restriction de licence (`check_folder_license_safety`) : un dossier (ou un
    parent jusqu'à `soundbanks/`) contenant un texte `NOT ROYALTY FREE` / `clearance required` /
    `non-commercial only` tague ses samples `restricted_non_royalty_free`, exclus par défaut de
    la sélection (sauf `--allow-restricted` ou scénario B `user_samples_locked` où l'utilisateur
    a fourni lui-même les fichiers dans `<projet>/Samples/`).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

AUDIO_EXTS_WHITELIST = {".wav", ".flac", ".ogg", ".aif", ".aiff", ".mp3"}

REJECTED_SUFFIXES = {
    ".zpa", ".fst", ".flp", ".als", ".sf2", ".sfz", ".mid", ".midi",
    ".asd", ".pkf", ".reapeaks", ".exe", ".dll", ".zip", ".rar", ".7z",
}

BAD_DIR_SEGMENTS = {"__macosx", ".git", ".venv", "sfz", "node_modules"}
BAD_BASENAMES = {".ds_store", "thumbs.db", "desktop.ini", "icon\r", "icon", "autorun.inf"}

MIN_FILE_BYTES = 1024

LICENSE_RESTRICTED_RE = re.compile(
    r"not\s+royalty[\s\-]?free|clearance\s+required|non[\s\-]?commercial\s+only"
    r"|do\s+not\s+(?:use|share|distribute)\s+(?:commercially|in\s+commercial)",
    re.IGNORECASE,
)
LICENSE_TEXT_SUFFIX = ".txt"
LICENSE_TEXT_NAME_RE = re.compile(r"^(readme|license|licence|eula|notice|copying)", re.IGNORECASE)
LICENSE_MAX_BYTES = 50 * 1024


def is_decodable_audio_file(path: str | os.PathLike) -> tuple[bool, str]:
    """Filtre de chemin/extension/taille. Retourne (ok, reason) — jamais d'exception."""
    try:
        p = Path(path)
        name = p.name
        low = name.lower()
        if low.startswith("._"):
            return False, "apple_double"
        if low in BAD_BASENAMES:
            return False, "os_metadata"
        if any(seg.lower() in BAD_DIR_SEGMENTS for seg in p.parts):
            return False, "bad_dir_segment"
        suf = p.suffix.lower()
        if suf in REJECTED_SUFFIXES:
            return False, f"non_pcm_suffix:{suf}"
        if suf not in AUDIO_EXTS_WHITELIST:
            return False, f"ext_not_whitelisted:{suf or '(none)'}"
        size = p.stat().st_size
        if size < MIN_FILE_BYTES:
            return False, f"too_small:{size}B"
        return True, "ok"
    except OSError as exc:
        return False, f"stat_error:{exc.errno}"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"error:{type(exc).__name__}"


def has_audio_magic_bytes(path: str | os.PathLike, read: int = 16) -> bool:
    """Vérifie l'en-tête audio réel (RIFF/WAVE, fLaC, OggS, FORM/AIFF, ID3/MPEG)."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(read)
    except OSError:
        return False
    if len(head) < 4:
        return False
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return True
    if head[:4] == b"fLaC":
        return True
    if head[:4] == b"OggS":
        return True
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return True
    if head[:3] == b"ID3":
        return True
    if head[0] == 0xFF and len(head) >= 2 and (head[1] & 0xE0) == 0xE0:
        return True
    return False


def is_audio_file_safe(path: str | os.PathLike) -> tuple[bool, str]:
    """Filtre complet : chemin + extension + taille + magic bytes."""
    ok, reason = is_decodable_audio_file(path)
    if not ok:
        return ok, reason
    if not has_audio_magic_bytes(path):
        return False, "bad_magic_bytes"
    return True, "ok"


def _iter_license_dirs(folder: str, stop_at: str | None) -> list[str]:
    """Dossier + parents jusqu'à (et inclus) le premier segment nommé `stop_at` (ex. soundbanks)."""
    folder = os.path.abspath(os.path.expanduser(folder))
    out: list[str] = []
    d = folder
    while True:
        out.append(d)
        if stop_at and os.path.basename(d) == stop_at:
            break
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return out


def check_folder_license_safety(folder: str | os.PathLike, stop_at: str | None = "soundbanks") -> dict[str, Any]:
    """Détecte une restriction de licence dans les fichiers texte du dossier et de ses parents.

    Retourne {"license_status": "restricted_non_royalty_free" | "ok" | "unknown", "evidence": [...]}.
    """
    evidence: list[str] = []
    try:
        for d in _iter_license_dirs(str(folder), stop_at):
            if not os.path.isdir(d):
                continue
            try:
                names = sorted(os.listdir(d))
            except OSError:
                continue
            for fn in names:
                low = fn.lower()
                if not (low.endswith(LICENSE_TEXT_SUFFIX) or LICENSE_TEXT_NAME_RE.match(fn)):
                    continue
                fp = os.path.join(d, fn)
                try:
                    if os.path.getsize(fp) > LICENSE_MAX_BYTES:
                        continue
                    with open(fp, encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                except OSError:
                    continue
                if LICENSE_RESTRICTED_RE.search(text):
                    evidence.append(f"{fp}: {LICENSE_RESTRICTED_RE.search(text).group(0)!r}")
    except Exception:
        pass
    status = "restricted_non_royalty_free" if evidence else "ok"
    return {"license_status": status, "evidence": evidence}


def scan_audio_tree(root: str, stop_at_license: str | None = "soundbanks") -> tuple[list[tuple[str, str]], list[dict[str, str]]]:
    """Walk `root`, retourne (fichiers audio sûrs [(abs, rel)], skipped [{path, reason}])."""
    root_abs = os.path.abspath(os.path.expanduser(root))
    found: list[tuple[str, str]] = []
    skipped: list[dict[str, str]] = []
    seen_real: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root_abs, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if d not in ("__MACOSX", "sfz") and not d.startswith("."))
        for fname in sorted(filenames):
            if fname.startswith("."):
                skipped.append({"path": os.path.join(dirpath, fname), "reason": "dotfile"})
                continue
            full = os.path.join(dirpath, fname)
            ok, reason = is_audio_file_safe(full)
            if not ok:
                skipped.append({"path": full, "reason": reason})
                continue
            real = os.path.realpath(full)
            if real in seen_real:
                continue
            seen_real.add(real)
            found.append((full, os.path.relpath(full, root_abs)))
    return found, skipped
