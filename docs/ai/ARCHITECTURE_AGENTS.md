# Architecture Technique & Manuel de Référence pour Agents IA

Ce document décrit l'infrastructure exacte de `/home/gamo/Bureau/musicos` (lien symbolique `~/Music/AI-Studio`) à destination des agents IA (Antigravity, Zed, agents locaux).

## 1. Inventaire des 5 serveurs MCP

| Serveur | Binaire / Script | Rôle principal |
| :--- | :--- | :--- |
| `gridline` | `scripts/gridline-mcp.sh` | Pilote Gridline 0.7.7 (live via `--attach` si port `47390` actif, sinon fichier `projects/active/projet.gridline`). |
| `audio-index` | `.venv/bin/python mcp-servers/audio-index/audio_index_server.py` | Recherche hybride Essentia + LAION CLAP (`laion/clap-htsat-unfused`) sur `mcp-servers/audio-index/samples.db` (1 898 fichiers dont 1 682 avec vecteurs CLAP). |
| `audio-analyzer` | `.venv/bin/python mcp-servers/audio-analyzer/audio_analyzer_server.py` | Analyse ITU-R BS.1770 (`pyloudnorm`), True Peak 4x, spectre 5 bandes, corrélation de phase, masquage Kick/808, audit psychoacoustique profond (`audit_mix_deep`) et Gate de musicalité/voice leading (`audit_midi_harmony` + `musicality_metrics.py`). |
| `vocaloid` | `.venv/bin/python mcp-servers/vocaloid/vocaloid_server.py` | Syllabation FR/EN, génération `.ustx` (OpenUTAU/DiffSinger) et synthèse d'aperçu vocal `.wav` via `espeak-ng` + `librosa`. |
| `reverse-engineer` | `.venv/bin/python mcp-servers/reverse-engineer/reverse_engineer_server.py` | Séparation de stems (`gridline stems` avec `htdemucs` / `htdemucs_6s`), transcription Audio->MIDI (Basic Pitch ONNX `tools/models/nmp.onnx`), analyse harmonique et plan de flip Rap. |

## 2. Pièges critiques vérifiés dans Gridline 0.7.7

1. **Chargement de samples dans `gl.sampler` / `gl.seltmann` / `gl.slicer` / `gl.granular`** :
   - Le champ `plugin.data` attend **une chaîne de chemin absolu brute**, jamais un objet JSON.
   - Bon appel :
     ```json
     {"path": "channel/1", "set": {"plugin": {"data": "/home/gamo/Bureau/musicos/samples/.../kick.wav"}}}
     ```
   - Après chargement, vérifie systématiquement avec `analyze_loudness(scope="pattern")` que `"silent": false`.

2. **Scripts Lua (`run_script`)** :
   - **Toujours encapsuler le code dans `function apply() ... end`** et **ne pas appeler `apply()` au top-level**. Sur Gridline 0.7.7, le code top-level échoue avec `no such channel` sur les canaux créés via MCP.
   - **Unités de temps** :
     - Outil MCP `set_notes` : `start` et `length` sont en **beats** (noires).
     - API Lua `project.patterns.setNotes(pat, ch, list)` : `time` et `length` sont en **ticks** (`project.song.ppq()`, par défaut `96` ticks/beat).
     - Longueur d'un pattern en Lua : `project.patterns.setBars(pat, bars)`.
     - Routage mixer en Lua : `project.channels.setRoute(ch, mixerTrack)` puis `project.mixer.routeTo(mixerTrack, busTrack)`.
     - Clip audio long (voix, stem complet) sur la playlist : charge dans un `gl.sampler`, puis via `run_script` :
       ```lua
       function apply()
         project.playlist.addAudioClip(1, 1, 0, 0)
       end
       ```

3. **Transport MCP stdio et sécurité** :
   - Les serveurs Python utilisent `mcp 2.3.0` (`MCPServer`). Ne jamais écrire de logs sur `stdout` (`print` vers `stdout` casserait JSON-RPC).

## 3. Catalogue vérifié des plugins tiers installés (`gridline scan-plugins`)

> **Important (Headless vs Live GUI)** :
> - Quand le pont `gridline` tourne en mode **headless** (`gridline mcp <fichier.gridline>`), `list_plugin_catalog` n'expose que les **179 plugins natifs `gl.*`** ; les appels `add_channel`/`add_plugin` avec un ID `clap:...` ou `vst3:...` retournent `unknown plugin type`.
> - Quand **Gridline GUI** est lancé (`scripts/launch-gridline.sh`) et connecté via `--attach`, vérifie `list_plugin_catalog` avant d'instancier un plugin tiers, ou utilise les `gl.*` natifs pour garantir une compatibilité 100% headless + live.

### Synthétiseur tiers
- **Surge XT** : `clap:org.surge-synth-team.surge-xt` (instrument) / `clap:org.surge-synth-team.surge-xt-fx` (effets). Synthèse hybride, FM, wavetable, textures chiptune/rétro pour VGM->Rap.

### Effets tiers (Mixage, Reverse Engineering, Grain Vintage)
- **ZL Equalizer 2** : `vst3:ABCDEF019182FAEB5A6C697545717532` (EQ dynamique 16 bandes + analyseur FFT temps réel + sidechain).
- **TDR Nova** : `vst3:56535454643561746472206E6F766100` (EQ dynamique 4 bandes + analyseur intégré, idéal pour creuser un stem VGM sous la voix ou sous le kick).
- **CHOWTapeModel** : `clap:org.chowdsp.CHOWTapeModel` (émulation bande magnétique, saturation harmonique, wow/flutter pour patine boom-bap/lo-fi).
- **Dragonfly Reverb** :
  - Hall : `vst3:2046504473616C633172666400000000`
  - Room : `vst3:2046504473616C633272666400000000`
  - Plate : `vst3:2046504473616C633370666400000000`
  - Early Reflections : `vst3:2046504473616C637265666400000000`
- **LSP Plugins (`1.2.37` CLAP)** — utilitaires d'analyse visuelle pour l'utilisateur :
  - Analyseurs de spectre : `clap:in.lsp-plug.spectrum_analyzer_x1`, `_x2`, `_x4`, `_x8`, `_x16`
  - Oscilloscopes : `clap:in.lsp-plug.oscilloscope_x1`, `_x2`, `_x4`
  - Crossover / Mid-Side : `clap:in.lsp-plug.crossover_stereo`, `clap:in.lsp-plug.crossover_ms`
  - Compresseurs / Limiteurs : `clap:in.lsp-plug.compressor_stereo`, `clap:in.lsp-plug.gott_compressor_stereo`, `clap:in.lsp-plug.limiter_stereo`

## 4. Performance CPU, PipeWire & Sobriété DSP (Obligatoire)
- **Strict respect des sons fournis par l'utilisateur** : Lorsqu'un `.gridline` initial contient déjà des canaux préparés, n'ajouter aucun synthé tiers ou natif lourd (`gl.matrixfm`, `gl.dcosynth`, `gl.twinstring`, `gl.multisampler`) ni aucun sample externe non fourni. Travailler uniquement avec les `gl.sampler` et canaux natifs du projet pour garantir zéro pic CPU (`> 100 %`) et une sonorité naturelle fidèle aux choix de l'utilisateur. En revanche, l'agent a le droit de **générer plusieurs versions d'un même sample fourni**, d'**en sélectionner ou découper des parties** (stems, slices, extraits vocaux, one-shots) et de **modifier le sample** (pitch, time-stretch, reverse, filtrage) selon les besoins du morceau.
- **PipeWire & Gridline Live** :
  - Configuration utilisateur persistante à `quantum = 1024` (`~/.config/pipewire/pipewire.conf.d/99-ai-studio-lowcpu.conf` & `pipewire-pulse.conf.d/99-ai-studio-lowcpu.conf`) + `pw-metadata -n settings 0 clock.force-quantum 1024` dans `scripts/launch-gridline.sh`.
  - Réglages `~/.config/gridline/settings.json` : `audio.bufferFrames = 1024`, `audio.threads = 4` (cœurs P sur Intel hybride), `freeze.autoHibernate = true` (`sleepSeconds = 15`), `live.pluginWindowsSeveral = false`, `ui.animations = false`, `draw.smooth.* = false`.
- **Budget DSP par projet** :
  - N'ajouter `gl.eq` que sur les pistes ayant un besoin fréquentiel réel (regrouper les percussions propres sous `BUS Drums` avec un seul EQ global).
  - Laisser `gl.eq` `Quality = 0` (`Normal`) sur les inserts individuels (`Quality = 1` réservé au `Master` uniquement).
  - Limiter la polyphonie et l'unison (`gl.multisampler <= 24`, synthés `Unison <= 2`, `Polyphony <= 16`) et mutualiser la réverbe sur 1 bus send (`FX Room`).
