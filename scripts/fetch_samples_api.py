#!/usr/bin/env python3
"""
Online Sample Fetcher (Freesound API v2 & Lots of Sounds API) with Automatic
24-bit WAV Normalization and Psychoacoustic Quality Gate.

Documentation:
  docs/04_ANALYSE_SONORE_SAMPLES_APIS_ET_VGM.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_metrics import (
    ClapEmbedder,
    VALID_ROLES,
    analyze_single_file,
    load_and_normalize_audio,
    extract_dsp_metrics,
    score_candidate,
)

STUDIO_DIR = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(STUDIO_DIR, "pack", "fetched")
DEFAULT_ENV_PATH = os.path.join(STUDIO_DIR, ".env")

MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024  # 25 Mo max par fichier téléchargé


def _safe_download_bytes(url: str, timeout: float = 30.0) -> tuple[bytes, str]:
    """Téléchargement HTTPS uniquement, plafonné à MAX_DOWNLOAD_BYTES (streaming).

    Retourne `(contenu, content-type)`. Refuse toute URL non-https (SSRF/schème file:).
    """
    if not str(url).lower().startswith("https://"):
        raise ValueError(f"Refus de télécharger une URL non-HTTPS : {url}")
    with requests.get(url, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        buf = bytearray()
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                buf.extend(chunk)
            if len(buf) > MAX_DOWNLOAD_BYTES:
                raise ValueError(f"Téléchargement trop volumineux (> {MAX_DOWNLOAD_BYTES} o) : {url}")
    return bytes(buf), ct


def load_studio_env(env_path: str = DEFAULT_ENV_PATH) -> dict[str, str]:
    """Load key=value secrets from .env without overriding already-set env vars."""
    loaded: dict[str, str] = {}
    if os.path.isfile(env_path):
        with open(env_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ and v:
                    os.environ[k] = v
                loaded[k] = v
    return loaded


def _sanitize_slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text.strip().lower()).strip("_")
    return s[:64] or "sample"


def normalize_to_wav24(
    src_path: str,
    dst_wav: str,
    apply_fade_ms: float = 3.0,
    is_short_oneshot: bool = False,
) -> None:
    """
    Convert downloaded MP3/OGG/WAV to 24-bit 44.1 kHz WAV.
    For short one-shots (<2.5s or perc/drums), use clean peak normalization (-1 dBFS)
    to preserve exact transient phase; for longer beds/loops, use EBU R128 loudnorm.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dst_wav)), exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp_wav = tf.name
    try:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", src_path]
        if not is_short_oneshot:
            cmd.extend(["-af", "loudnorm=I=-14:TP=-1.0:LRA=11"])
        cmd.extend(["-ar", "44100", "-c:a", "pcm_s24le", tmp_wav])
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import soundfile as sf
        y, sr = sf.read(tmp_wav, dtype="float32", always_2d=True)
        y = y - np.mean(y, axis=0, keepdims=True)
        fi = min(len(y) // 4, int(min(apply_fade_ms, 2.0 if is_short_oneshot else apply_fade_ms) * sr / 1000.0))
        fo = min(len(y) // 4, int((10.0 if is_short_oneshot else apply_fade_ms) * sr / 1000.0))
        if fi > 0:
            y[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)[:, None]
        if fo > 0:
            y[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)[:, None]
        pk = float(np.max(np.abs(y)))
        if is_short_oneshot and pk > 1e-6:
            y *= (0.891 / pk)  # exact -1.0 dBFS peak norm
        elif pk > 0.891:
            y *= (0.891 / pk)
        sf.write(dst_wav, y, sr, subtype="PCM_24")
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def search_freesound(
    query: str,
    api_key: str,
    duration_min: float = 0.1,
    duration_max: float = 5.0,
    cc0_only: bool = True,
    sort: str = "rating_desc",
    limit: int = 8,
    similar_to_id: int | None = None,
) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Token {api_key}"}
    if similar_to_id is not None:
        url = f"https://freesound.org/apiv2/sounds/{similar_to_id}/similar/"
        params = {"fields": "id,name,duration,tags,previews,license,username,avg_rating", "page_size": min(50, limit)}
    else:
        url = "https://freesound.org/apiv2/search/text/"
        filt_parts = [f"duration:[{duration_min:.2f} TO {duration_max:.2f}]"]
        if cc0_only:
            filt_parts.append('license:"Creative Commons 0"')
        params = {
            "query": query,
            "filter": " ".join(filt_parts),
            "sort": sort,
            "fields": "id,name,duration,tags,previews,license,username,avg_rating",
            "page_size": min(50, limit),
        }

    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    out: list[dict[str, Any]] = []
    for item in data.get("results", []):
        previews = item.get("previews", {}) or {}
        dl_url = previews.get("preview-hq-mp3") or previews.get("preview-hq-ogg") or previews.get("preview-lq-mp3")
        if not dl_url:
            continue
        out.append({
            "provider": "freesound",
            "id": item.get("id"),
            "name": item.get("name", f"fs_{item.get('id')}"),
            "duration": item.get("duration"),
            "license": item.get("license"),
            "tags": item.get("tags", []),
            "avg_rating": item.get("avg_rating"),
            "download_url": dl_url,
        })
    return out


def search_lotsofsounds(
    query: str,
    api_key: str = "",
    category: str = "",
    max_duration: float = 5.0,
    limit: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = "https://api.lotsofsounds.com"
    headers = {"x-api-key": api_key} if api_key else {}
    rate_info: dict[str, Any] = {}

    if api_key:
        params: dict[str, Any] = {"q": query, "max_duration": max_duration, "limit": min(50, limit)}
        if category:
            params["category"] = category
        resp = requests.get(f"{base}/api/v1/sounds", headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        if "X-RateLimit-Remaining" in resp.headers:
            rate_info["ratelimit_remaining"] = resp.headers["X-RateLimit-Remaining"]
        payload = resp.json()
        raw_items = payload.get("data", payload.get("sounds", payload.get("results", [])))
    else:
        # Public /sample demo endpoint (6 CC0 sample sounds without key)
        resp = requests.get(f"{base}/api/v1/sounds/sample", timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        raw_items = payload.get("data", payload.get("sounds", payload.get("results", [])))

    out: list[dict[str, Any]] = []
    q_low = (query or "").strip().lower()
    for item in raw_items:
        sid = item.get("id") or item.get("sound_id")
        if not sid:
            continue
        if not api_key and q_low:
            hay = " ".join([
                str(item.get("name", "")),
                str(item.get("description", "")),
                " ".join(item.get("tags", []) or []),
            ]).lower()
            if q_low not in hay and len(raw_items) > 1:
                pass  # still keep if needed below, or prioritize matches
        dl_url = item.get("download_url") or item.get("stream_url") or item.get("preview_url") or item.get("url")
        if dl_url and str(dl_url).startswith("/"):
            dl_url = base + str(dl_url)
        if not dl_url and api_key:
            dl_resp = requests.get(f"{base}/api/v1/sounds/{sid}/download", headers=headers, timeout=15)
            if dl_resp.status_code == 200:
                d_json = dl_resp.json()
                dl_url = (d_json.get("data") or {}).get("download_url") or d_json.get("download_url")
                if dl_url and str(dl_url).startswith("/"):
                    dl_url = base + str(dl_url)
        if not dl_url:
            continue
        hay_score = 1 if (q_low and q_low in f"{item.get('name', '')} {' '.join(item.get('tags', []) or [])}".lower()) else 0
        out.append({
            "provider": "lotsofsounds",
            "id": sid,
            "name": item.get("title") or item.get("name") or f"los_{sid}",
            "duration": item.get("duration"),
            "license": "Creative Commons 0 (CC0)",
            "tags": item.get("tags", []),
            "download_url": dl_url,
            "_match": hay_score,
        })
    out.sort(key=lambda x: x.pop("_match", 0), reverse=True)
    return out[:limit], rate_info


def fetch_and_filter_samples(
    query: str,
    role: str = "",
    provider: str = "auto",
    duration_min: float = 0.1,
    duration_max: float = 5.0,
    cc0_only: bool = True,
    limit: int = 5,
    out_dir: str = DEFAULT_OUT_DIR,
    mix_path: str = "",
    target_key: str = "",
    min_score: float = 0.55,
    with_clap: bool = True,
    similar_to_fs_id: int | None = None,
    monophonic_only: bool = False,
) -> dict[str, Any]:
    load_studio_env()
    fs_key = os.environ.get("FREESOUND_API_KEY", "").strip()
    los_key = os.environ.get("LOTSOFSOUNDS_API_KEY", "").strip()

    candidates: list[dict[str, Any]] = []
    api_errors: list[str] = []
    rate_info: dict[str, Any] = {}

    if provider in ("auto", "freesound"):
        if fs_key:
            try:
                candidates.extend(search_freesound(
                    query=query, api_key=fs_key, duration_min=duration_min,
                    duration_max=duration_max, cc0_only=cc0_only, limit=max(limit * 2, 8),
                    similar_to_id=similar_to_fs_id,
                ))
            except Exception as exc:
                api_errors.append(f"freesound: {exc}")
        elif provider == "freesound":
            api_errors.append("FREESOUND_API_KEY not configured in environment or .env")

    if provider in ("auto", "lotsofsounds") and (los_key or provider == "lotsofsounds" or not candidates):
        try:
            los_res, rate_info = search_lotsofsounds(
                query=query.split()[0] if query.strip() else "808",
                api_key=los_key, max_duration=duration_max, limit=max(limit * 2, 6),
            )
            candidates.extend(los_res)
        except Exception as exc:
            api_errors.append(f"lotsofsounds: {exc}")

    if not candidates:
        return {
            "query": query,
            "role": role,
            "fetched_count": 0,
            "accepted": [],
            "rejected": [],
            "api_errors": api_errors,
            "hint": "Set FREESOUND_API_KEY and/or LOTSOFSOUNDS_API_KEY in .env (see .env.example).",
        }

    mix_bark: list[float] | None = None
    if mix_path and os.path.isfile(mix_path):
        m_load = load_and_normalize_audio(mix_path)
        m_dsp = extract_dsp_metrics(m_load["y_stereo"], m_load["mono_norm"], sr=m_load["sr"])
        mix_bark = m_dsp["bark_24"]

    embedder = ClapEmbedder(device="cpu") if with_clap else None
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    role_eval = role if role in VALID_ROLES else "lead_melody"

    for item in candidates:
        prov = item["provider"]
        sid = str(item["id"])
        slug = _sanitize_slug(str(item["name"]))
        prov_dir = os.path.join(os.path.abspath(out_dir), prov)
        os.makedirs(prov_dir, exist_ok=True)
        out_wav = os.path.join(prov_dir, f"{slug}_{sid}.wav")

        try:
            if not os.path.isfile(out_wav):
                content, ct = _safe_download_bytes(item["download_url"], timeout=30)
                if "application/json" in ct.lower():
                    j = json.loads(content.decode("utf-8", "replace"))
                    real_url = (j.get("data") or {}).get("download_url") or (j.get("data") or {}).get("stream_url") or j.get("download_url") or j.get("stream_url")
                    if not real_url:
                        raise RuntimeError(f"JSON without audio URL: {j}")
                    content, ct = _safe_download_bytes(real_url, timeout=30)
                    ext = ".ogg" if ".ogg" in real_url.lower() else ".mp3"
                else:
                    ext = ".ogg" if ".ogg" in item["download_url"].lower() else ".mp3"
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
                    tf.write(content)
                    tmp_raw = tf.name
                try:
                    item_dur = float(item.get("duration") or 3.0)
                    short_flag = item_dur < 2.5 or role_eval in ("kick", "snare_clap", "hihat_perc")
                    normalize_to_wav24(tmp_raw, out_wav, is_short_oneshot=short_flag)
                finally:
                    if os.path.exists(tmp_raw):
                        os.remove(tmp_raw)

            entry, vec = analyze_single_file(out_wav, embedder=embedder, with_clap=with_clap)
            if entry["hard_reject"]:
                rejected.append({"filepath": out_wav, "reasons": entry["reject_reasons"], "provider": prov})
                continue

            scored = score_candidate(
                entry,
                role_id=role_eval,
                clap_vec=vec,
                embedder=embedder,
                mix_bark=mix_bark,
                target_key=target_key or None,
                monophonic_only=monophonic_only,
            )
            if scored.get("rejected") or scored["final_score"] < min_score:
                rejected.append({
                    "filepath": out_wav,
                    "final_score": scored.get("final_score", 0.0),
                    "reason": scored.get("reject_reason") or f"below_min_score({scored.get('final_score')}<{min_score})",
                    "provider": prov,
                })
                continue

            scored["provider"] = prov
            scored["license"] = item.get("license")
            scored["source_id"] = sid
            accepted.append(scored)
            if len(accepted) >= limit:
                break
        except Exception as exc:
            rejected.append({"name": item.get("name"), "reason": str(exc), "provider": prov})

    accepted.sort(key=lambda x: x["final_score"], reverse=True)
    return {
        "query": query,
        "role": role_eval,
        "provider_mode": provider,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted": accepted,
        "rejected": rejected[:8],
        "rate_info": rate_info,
        "api_errors": api_errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch online samples via Freesound v2 / Lots of Sounds and filter psychoacoustically.")
    ap.add_argument("--query", required=True, help="Search keywords (short for Lots of Sounds, boolean OK for Freesound).")
    ap.add_argument("--role", default="lead_melody", choices=list(VALID_ROLES))
    ap.add_argument("--provider", default="auto", choices=["auto", "freesound", "lotsofsounds"])
    ap.add_argument("--duration-min", type=float, default=0.1)
    ap.add_argument("--duration-max", type=float, default=5.0)
    ap.add_argument("--allow-cc-by", action="store_true", help="Include non-CC0 licenses on Freesound.")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--mix-path", default="")
    ap.add_argument("--target-key", default="")
    ap.add_argument("--min-score", type=float, default=0.55)
    ap.add_argument("--similar-to-id", type=int, default=None, help="Freesound sound ID for acoustic similarity search.")
    ap.add_argument("--mono-only", action="store_true", help="Reject melodic sequences / polyphonic chords (require single playable note).")
    ap.add_argument("--no-clap", action="store_true")
    args = ap.parse_args()

    res = fetch_and_filter_samples(
        query=args.query,
        role=args.role,
        provider=args.provider,
        duration_min=args.duration_min,
        duration_max=args.duration_max,
        cc0_only=not args.allow_cc_by,
        limit=args.limit,
        out_dir=args.out_dir,
        mix_path=args.mix_path,
        target_key=args.target_key,
        min_score=args.min_score,
        with_clap=not args.no_clap,
        similar_to_fs_id=args.similar_to_id,
        monophonic_only=args.mono_only,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
