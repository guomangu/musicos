#!/usr/bin/env python3
"""
Offline CLAP Semantic Anchor Encoder (0 PyTorch at runtime!).

Encodes the curated descriptors in `mcp-servers/audio-index/semantic_anchors.json`
once with LAION-CLAP (`laion/clap-htsat-unfused`) and saves L2-normalized float16
vectors into `mcp-servers/audio-index/semantic_anchors.npz`.

Usage:
  .venv/bin/python scripts/build_semantic_anchors.py
  .venv/bin/python scripts/build_semantic_anchors.py --populate-db
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
INDEX_DIR = os.path.join(ROOT_DIR, "mcp-servers", "audio-index")
for _d in (SCRIPTS_DIR, INDEX_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

DEFAULT_ANCHORS_JSON = os.path.join(INDEX_DIR, "semantic_anchors.json")
DEFAULT_ANCHORS_NPZ = os.path.join(INDEX_DIR, "semantic_anchors.npz")


def load_anchors_definition(json_path: str = DEFAULT_ANCHORS_JSON) -> dict[str, Any]:
    with open(json_path, encoding="utf-8") as fh:
        return json.load(fh)


def encode_anchors_with_clap(
    json_path: str = DEFAULT_ANCHORS_JSON,
    out_npz: str = DEFAULT_ANCHORS_NPZ,
    device: str = "cpu",
) -> dict[str, Any]:
    """Encode all semantic anchor prompts once via LAION-CLAP and persist L2-normalized float16 vectors."""
    import torch
    from audio_metrics import ClapEmbedder

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    data = load_anchors_definition(json_path)
    anchors = data.get("anchors") or []
    if not anchors:
        raise ValueError(f"No anchors found in {json_path}")

    model_id = str(data.get("model_id") or "laion/clap-htsat-unfused")
    embedder = ClapEmbedder(device=device, model_id=model_id)
    embedder._ensure_loaded()
    assert embedder._processor is not None and embedder._model is not None

    prompts = [str(a["prompt"]) for a in anchors]
    ids = np.array([str(a["id"]) for a in anchors], dtype=object)
    axes = np.array([str(a["axis"]) for a in anchors], dtype=object)
    fr_labels = np.array([str(a.get("fr_label", a["id"])) for a in anchors], dtype=object)
    thresholds = np.array([float(a.get("threshold", 0.24)) for a in anchors], dtype=np.float32)

    # Encode in chunks to keep RAM modest on CPU
    vecs_list: list[np.ndarray] = []
    batch_size = 32
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        inputs = embedder._processor(text=chunk, return_tensors="pt", padding=True).to(embedder._device)
        with torch.no_grad():
            out = embedder._model.get_text_features(**inputs)
            if not isinstance(out, torch.Tensor):
                out = getattr(out, "text_embeds", getattr(out, "pooler_output", out[0]))
            out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        vecs_list.append(out.cpu().numpy().astype(np.float32))

    embeddings_f32 = np.vstack(vecs_list)
    norms = np.linalg.norm(embeddings_f32, axis=1, keepdims=True)
    embeddings_f32 = embeddings_f32 / np.maximum(norms, 1e-12)
    embeddings_f16 = embeddings_f32.astype(np.float16)

    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    np.savez_compressed(
        out_npz,
        embeddings=embeddings_f16,
        ids=ids,
        axes=axes,
        fr_labels=fr_labels,
        prompts=np.array(prompts, dtype=object),
        thresholds=thresholds,
    )
    return {
        "json_path": os.path.abspath(json_path),
        "out_npz": os.path.abspath(out_npz),
        "anchor_count": int(len(anchors)),
        "embedding_dim": int(embeddings_f16.shape[1]),
        "dtype": str(embeddings_f16.dtype),
        "axes_counts": {
            ax: int(np.sum(axes == ax))
            for ax in sorted(set(axes.tolist()))
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-encode semantic audio anchors with LAION-CLAP (offline).")
    ap.add_argument("--anchors-json", default=DEFAULT_ANCHORS_JSON)
    ap.add_argument("--out-npz", default=DEFAULT_ANCHORS_NPZ)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--populate-db", action="store_true", help="Also build/refresh sample_semantic_cards in samples.db.")
    ap.add_argument("--db-path", default="", help="Optional custom SQLite DB path for --populate-db.")
    args = ap.parse_args()

    res = encode_anchors_with_clap(
        json_path=args.anchors_json,
        out_npz=args.out_npz,
        device=args.device,
    )
    if args.populate_db:
        from indexer import DEFAULT_DB_PATH, init_db, populate_semantic_cards

        target_db = args.db_path.strip() or DEFAULT_DB_PATH
        conn = init_db(target_db)
        res["db_cards"] = populate_semantic_cards(conn, anchors_path=args.out_npz, force=True)
        conn.close()

    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
