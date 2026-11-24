# External Pro Masters (`golden_references/external_masters/`)

Dépose ici **2 à 3 masters commerciaux par profil** (WAV / FLAC / MP3 / OGG) pour casser
la boucle d'auto-référence spectrale (Phase 8 `Q4.1`) et ancrer `spectral_profiles.json`
sur des références de production externes plutôt que sur les rendus internes du studio.

```text
golden_references/external_masters/
  trap_rap/
  pop_rnb/
  edm_club/
  rock_metal/
  lofi_chill/
  jazz_acoustic/
  classical_ambient/
```

## Règles

- **Les fichiers audio ne sont jamais versionnés** (`.gitignore` couvre
  `external_masters/*/*.{wav,mp3,flac,ogg}`) : ne pas committer de masters commerciaux protégés.
- Seules les **empreintes statistiques anonymes** (`bark24_mean/std`, pente rose, LUFS, LRA,
  facteur de crête, largeur médiums/aigus, ratios de bandes) sont écrites dans
  `golden_references/spectral_profiles.json` avec `"provenance": "external_pro_reference"`.
- Les profils sans master externe conservent le repli interne
  (`"provenance": "internal_bootstrap"`) et émettent un avertissement non bloquant.

## Reconstruire les profils

```bash
.venv/bin/python scripts/bootstrap_golden_refs.py \
  --external-dir golden_references/external_masters \
  --out golden_references/spectral_profiles.json
```

`target_fingerprint.py` et `render_and_audit.py` chargent ensuite ces ancres en priorité ;
la mémoire de goût (`memory/user_taste_profile.json`) ne peut dériver que de **30 %**
(ou **60 %** avec `--approve-taste`) lorsqu'une ancre externe existe.
