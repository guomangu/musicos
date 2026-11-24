# AI-Studio Musical — Gridline + Zed + Antigravity

Studio complet de production musicale assistée par IA sur **Fedora / Ultramarine 44 (PipeWire + Wayland)** dans `$AI_STUDIO_DIR` (accessible aussi via `~/Music/AI-Studio`).

## Documentation (5 Facettes Universelles & Optimisées pour Agents IA)
- **Règles normatives agents IA (Routeur + Machine à états 5 tours)** : [`.rules`](.rules) (liées par `GEMINI.md` et `AGENTS.md`)
- **Facette 1 — Système, Stack, Gridline `0.7.7`, Unités & Guide Utilisateur** : [`docs/01_SYSTEME_STACK_ET_GUIDE.md`](docs/01_SYSTEME_STACK_ET_GUIDE.md)
- **Facette 2 — Composition Universelle, Émotion `(V, T, E, D)`, Harmonie, Mélodie & Rythmes Multi-Genres** : [`docs/02_COMPOSITION_EMOTIONNELLE_ET_THEORIE.md`](docs/02_COMPOSITION_EMOTIONNELLE_ET_THEORIE.md)
- **Facette 3 — Blueprint, Arrangement, Top-Down, MIDI Gate & Patron `build_song.py`** : [`docs/03_ARCHITECTURE_BLUEPRINT_ET_COHESION.md`](docs/03_ARCHITECTURE_BLUEPRINT_ET_COHESION.md)
- **Facette 4 — DSP Déterministe, Samples, APIs Audio, Remix & Chant (Vocaloid)** : [`docs/04_ANALYSE_SONORE_SAMPLES_APIS_ET_VGM.md`](docs/04_ANALYSE_SONORE_SAMPLES_APIS_ET_VGM.md)
- **Facette 5 — Mixage Multi-Genres, Buses/FX, Quality Gate, Symptôme→Fix & Playbook** : [`docs/05_MIXAGE_QUALITY_GATE_ET_APPRENTISSAGE.md`](docs/05_MIXAGE_QUALITY_GATE_ET_APPRENTISSAGE.md)

## Briques installées et actives
1. **Gridline `0.7.7`** : MCP natif activé (`mcp.enabled: true`, `mcp.allowEdits: true`, port `47390`), modèles de séparation de stems téléchargés (`htdemucs` et `htdemucs_6s`).
2. **5 serveurs MCP connectés à Antigravity et Zed** :
   - `gridline` (mode live via `--attach` si Gridline GUI est ouvert, sinon édition directe sur `projects/active/projet.gridline`)
   - `audio-index` (`1 898` fichiers indexés dans `samples.db`, vecteurs **LAION CLAP** + analyse **Essentia**, sélection psychoacoustique à 4 couches `select_best_samples` / `evaluate_sample_quality` sur `sample_index.json` + `golden_references/`, et récupération à la volée via **Freesound API v2** & **Lots of Sounds API** `fetch_samples_online`)
   - `audio-analyzer` (`pyloudnorm` LUFS BS.1770, True Peak 4x, spectre 5 bandes, phase, masquage Kick/808)
   - `vocaloid` (syllabation FR/EN, génération `.ustx`, guides vocaux `.wav`, OpenUTAU `0.1.576` dans `tools/OpenUtau/`)
   - `reverse-engineer` (split 6 stems + **Spotify Basic Pitch ONNX** Audio→MIDI + détection d'accords/BPM/tonalité pour flips Rap)
3. **Plugins Linux natifs installés (729 classes scannées, 0 échec)** :
   - **TDR Nova** (`2.2.3` VST3), **ZL Equalizer 2** (`1.4.2` VST3), **Surge XT** (`1.3.4` CLAP/VST3), **CHOW Tape Model** (`2.11.4` CLAP/VST3/LV2), **Dragonfly Reverb** (`3.2.10` Hall/Room/Plate/ER VST3/LV2), **LSP Plugins** (`1.2.37` CLAP/VST3 : Spectrum Analyzer, Oscilloscope, Crossover, Compressors, Limiter).

## Installation automatisée (Fedora / nouveau clone GitHub)
```bash
chmod +x install.sh
./install.sh
```
Le script [`install.sh`](install.sh) détecte automatiquement les paquets système (`dnf`), Gridline, les modèles Demucs, Python `.venv`, PyTorch CPU + Essentia + LAION CLAP, Spotify Basic Pitch ONNX, OpenUTAU, les plugins Linux natifs (`CLAP`/`VST3`/`LV2`), configure Zed/Antigravity/Gridline et indexe vos samples.
Options utiles : `./install.sh --check-only`, `--skip-plugins`, `--skip-clap`, `--no-sudo`.

**Portabilité du clone** : aucun chemin machine n'est codé en dur. La racine se résout via `AI_STUDIO_DIR` (sinon dossier parent de `scripts/`) ; le template projet vide retombe sur `assets/empty_with_mixer_tracks.gridline` si Gridline n'est pas installé (`GRIDLINE_EMPTY_TEMPLATE` pour forcer un autre template). `requirements.txt` est installable depuis PyPI (`essentia>=2.1b6.dev1389`) ; `essentia`/`onnxruntime` sont optionnels au démarrage (dégradation librosa / erreur explicite pour Basic Pitch). **Audition sans rendu .wav** : avec Gridline ouvert (`gridline mcp --attach`), teste chaque piste et le morceau en session via `audio-analyzer:audit_track_soundcheck_fast` (`dsp`/`session`), `balance_tracks_in_session` (volumes équilibrés par solo + `analyze_loudness`, aucun .wav) et `audit_vertical_stage_session`. Le rendu .wav n'est réservé qu'au rendu final (`render_and_audit.py`).

Tests : `.venv/bin/python -m unittest discover -s tests -v` (les tests nécessitant le binaire `gridline` ou des données gitignorées sont automatiquement ignorés sur un clone frais).

## Démarrage rapide
```bash
# Option A (recommandée) : ouvrir Gridline en direct puis lancer Antigravity ou Zed
$AI_STUDIO_DIR/scripts/launch-gridline.sh

# Option B : générer sans ouvrir l'interface, puis ouvrir projects/active/projet.gridline
```
Puis ouvre `$AI_STUDIO_DIR` dans **Antigravity** ou **Zed** et suis les tutoriels dans [`docs/01_SYSTEME_STACK_ET_GUIDE.md`](docs/01_SYSTEME_STACK_ET_GUIDE.md).
