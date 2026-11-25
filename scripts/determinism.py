#!/usr/bin/env python3
"""
v3.3 #5 — Déterminisme strict par `seed` (doc audit v3.3, Carence #5.3).

  - `derive_seed(title, key, bpm)` : seed par défaut = SHA-256(title|key|bpm) — un morceau
    identique (titre/clé/BPM) est reproductible bit-à-bit même sans `seed` explicite.
  - `resolve_seed(blueprint)` : `blueprint["seed"]` si présent, sinon dérivé.
  - `seed_all(seed)` : graine les RNG globaux (`random`, `np.random`) ET retourne des
    instances dédiées (`random.Random`, `np.random.Generator`) à passer explicitement à
    `groove_lib`, `phrasing_engine`, `narrative_harmony`, `orchestration_engine`,
    `midi_groove_bank` et `compile_song` — 0 appel au RNG global le long du pipeline.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np


def derive_seed(title: str, key: str, bpm: float) -> int:
    """Seed déterministe dérivé du hash SHA-256 de `title|key|bpm` (mod 2^31)."""
    h = hashlib.sha256(f"{title}|{key}|{float(bpm):.3f}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % (2**31)


def resolve_seed(blueprint: dict[str, Any] | None) -> int:
    """`seed` explicite du blueprint, sinon dérivé de (title, key, bpm)."""
    if isinstance(blueprint, dict):
        s = blueprint.get("seed")
        if isinstance(s, (int, float)) and not isinstance(s, bool):
            return int(s) % (2**31)
        return derive_seed(str(blueprint.get("title") or "untitled"),
                           str(blueprint.get("key") or "C"),
                           float(blueprint.get("bpm") or 120.0))
    return derive_seed("untitled", "C", 120.0)


def seed_all(seed: int) -> tuple[random.Random, "np.random.Generator"]:
    """Graine les RNG globaux (filet de sécurité) et retourne des instances dédiées."""
    seed = int(seed) % (2**31)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    return random.Random(seed), np.random.default_rng(seed)
