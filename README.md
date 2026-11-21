# AI-Studio Musical — Gridline + Zed + Antigravity

Studio complet de production musicale assistée par IA sur **Fedora / Ultramarine 44 (PipeWire + Wayland)** dans `/home/gamo/Bureau/musicos` (accessible aussi via `~/Music/AI-Studio`).

## Documentation
- **Pour apprendre et produire pas à pas (Humain)** : [`docs/GUIDE_COMPLET_UTILISATEUR.md`](docs/GUIDE_COMPLET_UTILISATEUR.md)
- **Technologies employées, ordre et rôles** : [`docs/TECHNOLOGIES_ET_ARCHITECTURE.md`](docs/TECHNOLOGIES_ET_ARCHITECTURE.md)
- **Pour les agents IA (Antigravity / Zed)** :
  - Règles principales : [`.rules`](.rules) (liées par `GEMINI.md` et `AGENTS.md`)
  - Architecture, outils MCP & IDs plugins : [`docs/ai/ARCHITECTURE_AGENTS.md`](docs/ai/ARCHITECTURE_AGENTS.md)
  - Reverse Engineering VGM → Instru Rap : [`docs/ai/WORKFLOW_VGM_TO_RAP.md`](docs/ai/WORKFLOW_VGM_TO_RAP.md)
  - Spécifications Blueprint & Mixage : [`docs/ai/MIXING_AND_BLUEPRINT_SPEC.md`](docs/ai/MIXING_AND_BLUEPRINT_SPEC.md)
  - Protocole Multi-Échelles, Physique des Durées & Trap : [`docs/protocole_universel_d_architecture_musicale_coh_rence_multi_chelles_mod_le_pr_dictif_orientation_trap.md`](docs/protocole_universel_d_architecture_musicale_coh_rence_multi_chelles_mod_le_pr_dictif_orientation_trap.md) · [`docs/sp_cifications_th_oriques_et_techniques_harmonie_rythme_et_sound_design_de_la_trap.md`](docs/sp_cifications_th_oriques_et_techniques_harmonie_rythme_et_sound_design_de_la_trap.md)
- **Architecture source du projet** : [`docs/stack_technologique_studio_musical_assist_par_ia_gridline_zed_antigravity.md`](docs/stack_technologique_studio_musical_assist_par_ia_gridline_zed_antigravity.md)

## Briques installées et actives
1. **Gridline `0.7.7`** : MCP natif activé (`mcp.enabled: true`, `mcp.allowEdits: true`, port `47390`), modèles de séparation de stems téléchargés (`htdemucs` et `htdemucs_6s`).
2. **5 serveurs MCP connectés à Antigravity et Zed** :
   - `gridline` (mode live via `--attach` si Gridline GUI est ouvert, sinon édition directe sur `projects/active/projet.gridline`)
   - `audio-index` (`1 898` fichiers indexés, `1 682` embeddings sémantiques **LAION CLAP** + analyse **Essentia**)
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

## Démarrage rapide
```bash
# Option A (recommandée) : ouvrir Gridline en direct puis lancer Antigravity ou Zed
/home/gamo/Bureau/musicos/scripts/launch-gridline.sh

# Option B : générer sans ouvrir l'interface, puis ouvrir projects/active/projet.gridline
```
Puis ouvre `/home/gamo/Bureau/musicos` dans **Antigravity** ou **Zed** et suis les tutoriels dans [`docs/GUIDE_COMPLET_UTILISATEUR.md`](docs/GUIDE_COMPLET_UTILISATEUR.md).
