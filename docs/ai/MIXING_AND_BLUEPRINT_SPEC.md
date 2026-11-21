# Spécifications Blueprint, Bandes Fréquentielles & Cibles de Mixage

## 1. Cibles de Loudness & Dynamique par style
| Style / Usage | Cible `target_lufs` | True Peak max (`true_peak_dbtp`) | Corrélation basse `< 150 Hz` |
| :--- | :--- | :--- | :--- |
| Streaming standard / Lo-Fi / Boom-Bap | `-14.0` LUFS (+/- 1.2 dB) | `<= -1.0` dBTP | `>= 0.6` |
| Trap / Drill / Rap moderne | `-11.0` à `-9.5` LUFS | `<= -1.0` dBTP | `>= 0.7` |
| Club / Sound design agressif | `-9.0` LUFS | `<= -1.0` dBTP | `>= 0.75` |

## 2. Plan fréquentiel de référence (Anti-Masquage)
| Bande | Plage Hz | Rôle autorisé | Règle d'isolation |
| :--- | :--- | :--- | :--- |
| **Sub** | `20 – 60 Hz` | 808 / Sub Bass uniquement | Mono strict (`gl.stereo` width = 0 ou `gl.eq` M/S), HPF à `28 Hz`. |
| **Bass / Kick** | `60 – 250 Hz` | Kick (`50–120 Hz`) + harmoniques basse | Sidechain Kick -> Basse + HPF `>= 130 Hz` sur tout le reste. |
| **Low-Mids** | `250 – 500 Hz` | Corps piano, guitares, synthés, caisse claire | Garder sous `24%` d'énergie totale (coupe douce à `~300 Hz` si boueux). |
| **High-Mids** | `500 Hz – 6 kHz` | Leads, mélodie VGM, voix / chant, attaque snare | Creuser légèrement `1.5–3 kHz` sur les instruments si voix lead active. |
| **Air** | `6 – 20 kHz` | Hi-hats, claps, brillance reverb (`Dragonfly Plate/Hall`) | Viser `>= 3%` d'énergie dans l'air pour éviter un mix sourd. |

## 3. Contrat `song_blueprint.json`
Doit toujours valider `schemas/song_blueprint.schema.json` :
```bash
/home/gamo/Bureau/musicos/.venv/bin/python -c \
  "import json, jsonschema; jsonschema.validate(json.load(open('projects/active/song_blueprint.json')), json.load(open('schemas/song_blueprint.schema.json')))"
```
Champs obligatoires : `title`, `genre`, `bpm`, `time_signature`, `key`, `structure`, `tracks`.
Champs fortement recommandés : `target_lufs`, `frequency_plan`, `vocals` (si chant).

## 4. Organisation de la Playlist (Lisibilité humaine)
- Placer **chaque instrument et chaque variation sur sa propre piste Playlist dédiée** (ne jamais empiler tout le morceau sur `Track 1`), avec des patterns nommés par instrument et par section (ex. `808 Infinite - M1`, `808 Infinite - Climax`, `Pad High +12`).
