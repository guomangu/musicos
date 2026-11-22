# Golden References (`golden_references/`)

Dépose ici **5 à 15 samples de référence** validés à l'oreille pour chaque rôle :
- `kick/`
- `bass_sub/`
- `snare_clap/`
- `hihat_perc/`
- `lead_melody/`
- `pad_atmosphere/`
- `chord_keys/`

Puis reconstruis la matrice d'embeddings CLAP avec :
```bash
/home/gamo/Bureau/musicos/.venv/bin/python /home/gamo/Bureau/musicos/scripts/build_references.py \
  --ref-dir /home/gamo/Bureau/musicos/golden_references \
  --output-refs /home/gamo/Bureau/musicos/mcp-servers/audio-index/golden_embeddings.npz
```
Lorsqu'un rôle possède des références dans `golden_embeddings.npz`, `select_sample.py` et `audio-index:select_best_samples` combinent automatiquement la proximité top-3 (`0.7 * S_ref`) et le scoring zero-shot CLAP (`0.3 * S_zs`).
