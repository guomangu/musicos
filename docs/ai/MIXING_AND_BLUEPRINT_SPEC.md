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
Champs obligatoires : `title`, `genre`, `bpm`, `time_signature`, `key`, `harmonic_core`, `structure`, `tracks`.
Champs fortement recommandés : `target_lufs`, `frequency_plan`, `vocals` (si chant).

## 4. Organisation de la Playlist (Lisibilité humaine)
- Placer **chaque instrument et chaque variation sur sa propre piste Playlist dédiée** (ne jamais empiler tout le morceau sur `Track 1`), avec des patterns nommés par instrument et par section (ex. `808 Infinite - M1`, `808 Infinite - Climax`, `Pad High +12`).
- Pour les règles d'harmonie modale (Phrygien, Éolien, Dorien), de voicing ouvert, de rolls de hi-hats (pitch-down) et de sous-genres Trap (Cloud, Dark/Hard, Rage, Drill), consulter [docs/sp_cifications_th_oriques_et_techniques_harmonie_rythme_et_sound_design_de_la_trap.md](../sp_cifications_th_oriques_et_techniques_harmonie_rythme_et_sound_design_de_la_trap.md).

## 5. Architecture Multi-Échelles, Composition Top-Down & Ciment Timbral
Consulter [docs/regles_melodie_et_cohesion_inter_pistes.md](../regles_melodie_et_cohesion_inter_pistes.md), [docs/protocole_universel_d_architecture_musicale_coh_rence_multi_chelles_mod_le_pr_dictif_orientation_trap.md](../protocole_universel_d_architecture_musicale_coh_rence_multi_chelles_mod_le_pr_dictif_orientation_trap.md) et [docs/harmonisation_verticale_et_horizontale.md](../harmonisation_verticale_et_horizontale.md) :
- **Réduction Unique & Gate Bloquant (`master_sketch.mid`)** : composer d'abord la réduction 3 voix (`Bass`, `Chords` en *voice leading* `<= 2.5 st` via `optimize_voice_leading`, `Lead` `A-A'-A''-B`) et valider via `audio-analyzer:audit_midi_harmony(..., strict_musicality=True)` (`musicality_verdict == "PASS"`) avant toute orchestration ou batterie.
- **Cohésion verticale & horizontale** : accordage strict `A4 = 440 Hz ± 5 ct`, horloge rythmique unique (`one-shots` à `t = 0`), interdiction de transposition parallèle d'accords complets samplés, mouvement contraire/oblique `>= 55%`, relais d'attaques (`<= 15%` de chevauchement).
- **Chaîne de Ciment Timbral `BUS Music`** : `gl.eq` (`HPF 120–150 Hz` + `LPF 11–14 kHz`) -> `gl.dist` (`Soft 4x`, `Mix 8–12%`) -> `gl.comp` Bus Glue (`Ratio 2.0`, `Attack 20 ms`, `Release 110 ms`) -> `gl.comp` Micro-Ducking Sidechain depuis le Kick (`Attack 5 ms`, `Release 80 ms`, `GR 1.5–2.5 dB`), complétés par `FX Room` unique (`Dry = 0`, `Dry (full) = 0`, `RT60 <= T_noire`).
- **Double axe** : cohérence verticale (spectre/harmonie, max **3 couches fonctionnelles** hors batterie : Fondation, Harmonie/Texture, Motif/Voix) et horizontale (modèle prédictif en 4 étapes : Encodage -> Validation -> Surprise contrôlée 70/30 -> Résolution, avec renouvellement toutes les 4 et 8 mesures et ancrage *half-time* du Clap/Snare sur le 3e temps).
- **Physique des durées & dimensionnement ADSR** ($T_{\text{noire}} = \frac{60\,000}{\text{BPM}}\text{ ms}$, ex. `333.3 ms` à `180 BPM`) :
  - *Ticks / Hi-Hats / Rimshots* : **`20–90 ms`** (`Attack 0 ms`, `Decay 30–70 ms`, `Sustain 0%`), extinction avant la subdivision suivante pour créer un silence intercalaire.
  - *Snare / Clap* : **`120–250 ms`**, queue de réverbe éteinte avant le temps suivant.
  - *Kick* : **`80–160 ms`** (`Sustain 0%`) pour laisser la place à la 808 qui le prolonge.
  - *Basse 808* : **Mode Monophonique (*Cut Itself*) OBLIGATOIRE** (`Voices = 1`, `Release = 30–60 ms`), longue (`1–4` temps) ou bounce (`1/8–1/4` temps en couplet rapide).
  - *Contraste Staccato / Legato* : opposer une 808 longue (`Legato`) à des percussions/mélodies courtes (`Staccato`), ou inversement des pads tenus à une 808 rebondissante (`Bounce`).
  - *Queues d'effets (Reverb / Delay)* : `RT60` et échos calés sur un multiple/sous-multiple exact de $T_{\text{noire}}$ et **coupés automatiquement** lors des silences/mutes de transition (fins de 4e/8e mesures).
- **Checklist de validation en 5 points avant rendu** : Test de la Soustraction, Test de la Densité (`<= 3` couches), Test des Durées et Silences (*cut itself* 808 + decay court des ticks), Test des 8 Mesures (marqueur distinctif par bloc), Test d'Espace Vocal (`800 Hz – 2.5 kHz` dégagé en couplets).
