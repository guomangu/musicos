# Facette 1 — Système, Stack Technologique, Gridline 0.7.7, Unités & Guide Utilisateur

> **Cheatsheet IA (L'essentiel en 12 lignes)**
> - **Moteur** : Gridline `0.7.7` (Live `--attach` port `47390` ou Headless `.gridline`). Toujours privilégier les plugins natifs `gl.*` pour 100 % portabilité GUI + CLI.
> - **Mémoire sonore** : `audio-index:search_samples` interroge `samples.db` (SQLite, **< 10 ms**). `select_best_samples` lit `sample_index.json` (attention : si absent, ne pas l'appeler en MCP — lancer `scripts/index_samples.py --no-clap` en CLI ou utiliser `search_samples` + `analyze_sample_dsp`).
> - **Unités** : Toujours composer en **beats 0-indexed** dans Python (`0.0, 0.5, 1.0...`) puis convertir automatiquement vers `pretty_midi` (`beat * 60/BPM`), MCP `set_notes` (`beat`), Lua (`int(round(beat * 96))` ticks).
> - **Pièges Gridline mortels** : `gl.eq Peak = 6` (pas `4`=Notch) · `edit_channel(mixer=M)` (jamais `patch mixerTrack`) · `set_send(..., sidechain=True, preFader=True)` · `gl.reverb Decay (full)` est en **décisecondes (`1..200`, ex. `4` = `0.4 s`)** et `Dry (full) = 0` · `gl.sampler` WAV 24-bit (pas `gl.multisampler` `.sfz` externe, muet en GUI) · Lua toujours dans `function apply() ... end`.
> - **Économie Tokens & CPU** : 1 seul script `build_song.py` (`GridlineMCP.batch` + `set_params`) ; `--no-clap` pour DSP pur ; `--range <b1>-<b2> --no-neural` pour les itérations mixage ; 1 seul rendu final complet avec `--mp3 --sections`.

---

## Chapitre 1 — Architecture Globale & Sobriété CPU (Linux / PipeWire)

```text
Prompt Utilisateur (Antigravity / Zed)
   ├──▶ [1. Blueprint & Sons]  audio-index (samples.db) · reverse-engineer · vocaloid
   │       └── Scripts Python DSP déterministes (.venv : parselmouth, librosa, scipy)
   ├──▶ [2. Partition & Gate]  master_sketch.mid (Python en beats) → verify_midi_gate() [0 rendu audio]
   ├──▶ [3. Build Atomique]    build_song.py → GridlineMCP (batch / set_params / Lua) → projet.gridline
   └──▶ [4. Mix & Validation]  scripts/render_and_audit.py (--range --no-neural → rendu final --mp3)
```

### 1.1 Environnement & Arborescence Clé
* **OS / Audio** : Fedora Linux + **PipeWire** (`quantum >= 1024` via `~/.config/pipewire/pipewire.conf.d/99-ai-studio-lowcpu.conf` et `scripts/launch-gridline.sh`).
* **Gridline (`~/.config/gridline/settings.json`)** : `audio.bufferFrames = 1024`, `audio.threads = 4`, `freeze.autoHibernate = true` (`sleepSeconds = 15`), `ui.animations = false`.
* **Stack Python (`.venv`)** :
  * *DSP rapide (0 GPU)* : `praat-parselmouth>=0.4.7`, `librosa`, `scipy`, `numpy`, `soundfile`, `essentia`, `mosqito`, `pyloudnorm`, `pyroomacoustics`.
  * *MIDI & Schémas* : `pretty_midi`, `music21`, `jsonschema`.
  * *Perceptuel CPU (coûteux : 15–40 s par appel, réserver au rendu final)* : `transformers` (LAION-CLAP), `audiobox_aesthetics`.
* **Fichiers & Dossiers** :
  * `.rules` (`GEMINI.md` et `AGENTS.md` sont des liens symboliques dessus).
  * `schemas/song_blueprint.schema.json` : contrat de validation universel.
  * `samples -> soundbanks` : lien symbolique vers la sonothèque locale.
  * `.env` : clés API (`FREESOUND_API_KEY`, `LOTSOFSOUNDS_API_KEY`).
  * `projects/active/` (ou `active2`..`activeN`, ou dossier dédié `morceaux/<nom>/`) : `song_blueprint.json`, `master_sketch.mid`, `build_song.py`, `samples/`, `exports/`, `renders/`, `projet.gridline`.

### 1.2 Budget DSP Strict
1. N'ajouter `gl.eq` sur un insert individuel que si un besoin fréquentiel est mesuré (sinon regrouper sur `BUS Drums` / `BUS Music`).
2. `gl.eq` `Quality = 0` (`Normal`) sur les inserts ; `Quality = 1` sur `Master 0` uniquement.
3. Limiter `Polyphony <= 12` (max `16` pour pianos soutenus), `Unison <= 2`, et mutualiser la réverbération sur un unique bus `FX Room`.
4. **Template vide** : `/usr/share/gridline/assets/templates/Basics/Empty with mixer tracks.gridline` ne contient que les pistes mixeur **`0..16`**. `studio_lib.setup_standard_buses()` crée automatiquement les pistes manquantes jusqu'à `20`.

---

## Chapitre 2 — Inventaire des 5 Serveurs MCP & État des Index

Les serveurs Python utilisent `mcp 2.3.0` (`MCPServer`, transport `stdio`). Tout log va sur `stderr`.

| Serveur | Script | Outils Clés & Coût CPU |
| :--- | :--- | :--- |
| **`gridline`** | `scripts/gridline-mcp.sh` | Pilote Gridline `0.7.7` (Live `--attach` port `47390` ou Headless). En build complet, préférer `studio_lib.GridlineMCP` dans `build_song.py` (`batch` + `set_params`). |
| **`audio-index`** | `mcp-servers/audio-index/audio_index_server.py` | **Rapide (< 10 ms)** : `search_samples`, `find_compatible_samples`, `get_sample_info`, `list_sample_packs`, `search_synth_presets`, `analyze_sample_dsp`. **Lourd (CLAP)** : `evaluate_sample_quality(with_clap=False` pour rapide`)`, `select_best_samples`, `fetch_samples_online`. |
| **`audio-analyzer`** | `mcp-servers/audio-analyzer/audio_analyzer_server.py` | **Rapide (< 1 s)** : `audit_midi_harmony`, `analyze_mix`, `analyze_low_end_masking`, `compare_renders`, `check_pipewire_health`. **Complet** : `audit_mix_deep(use_neural=False` pour rapide`)`, `match_reference_audio`, `render_project_stems`. |
| **`reverse-engineer`** | `mcp-servers/reverse-engineer/reverse_engineer_server.py` | `analyze_song_for_rap_flip` (BPM, tonalité, accords), `transcribe_audio_to_midi` (Basic Pitch ONNX), `split_stems` (`htdemucs_6s`/`htdemucs` — lourd CPU). |
| **`vocaloid`** | `mcp-servers/vocaloid/vocaloid_server.py` | `list_voicebanks`, `syllabify_lyrics` (FR/EN), `create_ustx`, `synthesize_vocal_preview` (.wav), `open_in_openutau`. |

> **Clarification Critique — Deux Index Distincts dans `audio-index`**
> 1. **`samples.db` (SQLite, déjà prêt)** : alimenté par `mcp-servers/audio-index/indexer.py` (`reindex_samples`). Utilisé par `search_samples` et `find_compatible_samples` → **instantané (< 10 ms)**.
> 2. **`sample_index.json` + `golden_embeddings.npz` (Psychoacoustique + CLAP)** : alimenté par `scripts/index_samples.py` et `scripts/build_references.py` (dossiers `golden_references/`). **Attention** : si `sample_index.json` n'existe pas sur disque, appeler `select_best_samples` déclenche une indexation CLAP CPU synchrone sur toute la banque (**timeout assuré**). Vérifier sa présence ou utiliser `search_samples` + `analyze_sample_dsp` / `fetch_samples_online(..., with_clap=False)`.

---

## Chapitre 3 — Catalogue Vérifié des Plugins Gridline 0.7.7 (Multi-Genres)

> **Headless vs Live GUI** : En mode headless (`gridline mcp`), seuls les **plugins natifs `gl.*`** sont disponibles. Les plugins tiers (`clap:`, `vst3:`) nécessitent Gridline GUI (`scripts/launch-gridline.sh`). Utiliser `gl.*` par défaut pour garantir un rendu identique GUI + CLI.

### 3.1 Instruments Natifs Vérifiés (`gl.*`) classés par Rôle & Genre

| Rôle Musical | Plugins Natifs `gl.*` Vérifiés | Usage Recommandé par Genre |
| :--- | :--- | :--- |
| **Samples / One-Shots / Boucles** | `gl.sampler` (one-shots, 808, pianos WAV), `gl.slicer` (*Kerf*), `gl.seltmann`, `gl.granular` | Universel (Trap, Lo-Fi, Pop, Cinématique, Remix) |
| **Claviers & Pianos** | `gl.keys`, `gl.samplekeys`, `gl.sampler` (WAV piano 24-bit `Voices=12`) | Pop, Jazz, Lo-Fi, R&B, Ballade, Classique |
| **Basses (Électriques, Synthé, Sub)** | `gl.fingerbass` (basse électrique/acoustique), `gl.stepbass`, `gl.acid`, `gl.analog`, `gl.sampler` (808 sub) | Rock/Funk/Jazz/Pop (`gl.fingerbass`), House/EDM (`gl.acid`/`gl.analog`), Trap (`gl.sampler` 808) |
| **Synthés Analogiques / Pads / Leads** | `gl.warmsynth` (*Warm Trio*), `gl.dcosynth`, `gl.analog`, `gl.twinva`, `gl.sawswarm`, `gl.wavetable` (*Isoline*), `gl.synth` (*Tercet*), `gl.plasma` | Synthwave, Pop, House, EDM, Ambient, Cloud Trap, Rage |
| **Cordes & Pincés (Physiques)** | `gl.string` (*String Model*), `gl.twinstring`, `gl.plucker` (*Pluck Model*), `gl.pluck` | Cinématique, Orchestral, Folk/Indie, Lo-Fi, Acoustic, Trap mélodique |
| **FM / Métallique / Cloches / Chiptune** | `gl.fm`, `gl.fmduo`, `gl.matrixfm` (*Lattice*), `gl.alloy`, `gl.pixeltone`, `gl.additive` (*Facet*) | Dark Trap (cloches), R&B/EP, Chiptune/VGM, Electro, Ambient |
| **Batterie Synthétisée / Kits** | `gl.drumkit` (*Drum Model*), `gl.membranekit`, `gl.kick`, `gl.kicksculpt`, `gl.sweepkick`, `gl.snare`, `gl.clap`, `gl.hat`, `gl.pads` (*Abacus*) | Electro, Techno, House, Pop, Rock, Trap (complément ou substitut aux samples) |

> **Règle `gl.multisampler` (`.sfz`)** : certains `.sfz` externes (`Headroom Piano Clean.sfz`) peuvent être muets (`-144 LUFS`) en GUI `--attach`. Privilégier `gl.sampler` avec un WAV 24-bit mono-note calibré `A4=440 Hz` (`Voices = 12`) ou `gl.keys` / `gl.warmsynth`.

### 3.2 Effets Natifs Clés (`gl.*`) & Paramètres Exacts Vérifiés
* **`gl.sampler`** : `plugin.data = "/chemin/absolu.wav"` (chaîne brute). Clés : `Root note` (MIDI), `Fine tune` (cents), `One shot` (`0`/`1`), `Voices` (`1` mono cut-itself pour 808/sub, `12` pour accords), `Attack`, `Release`, `Glide`, `Declick`, `Remove DC`, `Loop`, `Filter`/`Cutoff`/`Resonance`.
* **`gl.eq` (7 bandes)** : `Quality` (`0=Normal`, `1=Master`). **Enum exact `Band N type`** : `0=Off`, `1=Low cut` (HPF), `2=Band pass`, `3=High cut` (LPF), **`4=Notch`**, `5=Low shelf`, **`6=Peak`**, `7=High shelf`.
* **`gl.comp`** : `Detector input` (`0=Main`, `1=Sidechain`), `Threshold` (dB), `Ratio`, `Attack` (s), `Release` (s), `Makeup` (dB), `Lookahead`, `Detector high-pass` (Hz).
* **`gl.limiter`** : `Gain` (dB), `Ceiling` (dB), **`Attack (look-ahead)` = `0.002` s** (jamais `0.005`), `0.2 dB safety = 1`.
* **`gl.dist`** : `Mode = 0` (`Soft`), `Drive`, `Tone` (Hz), `Mix` (`0..1`), `Oversampling` (`2` = 4x), `Remove DC = 1`.
* **`gl.stereo`** : `Width` (`0.0` = mono strict pour `BUS Bass/808`, `1.0` = normal).
* **`gl.reverb` (Mode `Model = 1` Full — Unités Exactes !)** :
  * Paramètres standard : `Decay time` (`0.1..20` **secondes**), `Dry` (`0..1`), `Wet` (`0..1`), `Low cut` (`20..3000` Hz), `High cut` (`500..22100` Hz).
  * Paramètres **`(full)` obligatoires** (sinon l'audio sec fuit !) :
    * `Dry (full)` : **`0`** (`0..160` en unité `k128`, où `128 = 100 %`).
    * `Wet (full)` : `45..64` (`k128`).
    * `Early reflections (full)` : `8..24` (`k128`).
    * `Low cut (full)` : `350..500` (Hz).
    * **`Decay (full)`** : **`1..200` en décisecondes (`ds = 0.1 s`)** → ex. pour `0.4 s`, mettre **`4`** (jamais `0.4` !).
    * `High cut (full)` : `5..221` en centaines de Hz (`hHz`, ex. `68` = `6.8 kHz`).
* **`gl.delay`** : délai stéréo (mode `Reference`).

### 3.3 Plugins Tiers Vérifiés (Session GUI `--attach` uniquement)
* **Synthé** : Surge XT (`clap:org.surge-synth-team.surge-xt`, FX `clap:org.surge-synth-team.surge-xt-fx`).
* **EQ / Saturation / Réverbe** : ZL Equalizer 2 (`vst3:ABCDEF019182FAEB5A6C697545717532`), TDR Nova (`vst3:56535454643561746472206E6F766100`), CHOWTapeModel (`clap:org.chowdsp.CHOWTapeModel`), Dragonfly Reverb (Hall `vst3:2046504473616C633172666400000000`, Room `vst3:2046504473616C633272666400000000`, Plate `vst3:2046504473616C633370666400000000`).
* **Analyse LSP (`1.2.37` CLAP)** : `clap:in.lsp-plug.spectrum_analyzer_x1..x16`, `oscilloscope_x1..x4`, `crossover_stereo`, `compressor_stereo`, `limiter_stereo`.

---

## Chapitre 4 — Table Universelle des Unités & Syntaxe Gridline (Zéro Erreur LLM)

> **Règle d'Or Temporelle** : Ne jamais mélanger manuellement les unités. **Toujours stocker toute la composition en `beats` (float 0-indexed)** dans Python, puis convertir par formule unique vers `pretty_midi`, MCP et Lua.

| Contexte | Unité | Indexation | Formule depuis `beat` (`b`, 0-indexed) en `B` beats/mesure (`4` en 4/4, `3` en 3/4) |
| :--- | :--- | :---: | :--- |
| **`song_blueprint.json`** | `bar` (mesure) | **1-indexed** (`1, 2, ...`) | `bar = int(b // B) + 1` ; inverse : `b_start = (bar - 1) * B` |
| **MCP `set_notes` / `add_automation`** | `beats` (`time`, `length`, `position`) | **0-indexed** (`0.0, 0.5, ...`) | `b` directement (en 4/4, les 4 temps d'une mesure sont `0.0, 1.0, 2.0, 3.0`) |
| **Lua `project.patterns.setNotes`** | `ticks` (`ppq = 96`/beat) | **0-indexed** (`0, 48, 96...`) | `tick = int(round(b * 96))` (1 mesure 4/4 = `384` ticks) |
| **`pretty_midi` (`master_sketch.mid`)** | `secondes` (`start`, `end`) | **0-indexed** (`0.0 s...`) | `sec = b * (60.0 / bpm)` |
| **Grille 16 pas (double-croche 4/4)** | `step` (`0..15`) | **0-indexed** | `b = step * 0.25` (temps 1=`step 0`, 2=`step 4`, 3=`step 8`, 4=`step 12`) |

### Référentiel Mathématique des Durées (`4/4` et BPM)
* **Noire en ms** : $T_{\text{noire (ms)}} = \frac{60\,000}{\text{BPM}}$ · **Figure de fraction $F$ d'une mesure** : $T_{\text{note (ms)}} = \left(\frac{60\,000}{\text{BPM}} \times 4\right) \times F$.
* **Modificateurs** : Pointé (`.`) $\times 1{,}5$ · Triolet (`T`) $\times \frac{2}{3}$.

| Figure | Notation | Fraction | Temps | `90 BPM` | `120 BPM` | `140 BPM` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ronde / Blanche / Noire** | `1/1`, `1/2`, `1/4` | `1`, `1/2`, `1/4` | `4`, `2`, `1` | `2667 / 1333 / 667 ms` | `2000 / 1000 / 500 ms` | `1714 / 857 / 429 ms` |
| **Croche / Triolet croche** | `1/8`, `1/8T` | `1/8`, `1/12` | `0.5`, `0.33` | `333 / 222 ms` | `250 / 167 ms` | `214 / 143 ms` |
| **Double / Triolet double / Triple** | `1/16`, `1/16T`, `1/32` | `1/16`, `1/24`, `1/32` | `0.25`, `0.167`, `0.125` | `167 / 111 / 83 ms` | `125 / 83 / 62.5 ms` | `107 / 71 / 53.5 ms` |

### Syntaxe Critique MCP & Lua (`run_script`)
1. **`set_song`** : `numerator` et `denominator` (jamais `timeSigNum`/`timeSigDen`).
2. **Routage mixeur** : `add_channel(..., mixer=M)` ou `edit_channel(channel=N, mixer=M)` — **jamais** `patch("channel/N/mixerTrack", ...)`.
3. **Largeur stéréo mixeur** : `set_parameters` sur `mixer/N/stereo` (ex. `-0.25` rétrécit vers mono).
4. **Sends & Sidechain** : `"sidechain": True, "preFader": True` obligatoire pour tout envoi sidechain.
5. **Automations (`add_automation`)** : `"param": "volume"`, points avec `"position"` (beats, jamais `"time"`) et `"value"` normalisée (`0.0..1.0` ; max mixeur `1.25` → gain `0.80` = `0.64`).
6. **Lua (`run_script`)** : toujours définir `function apply() ... end` sans l'appeler au top-level :
   * `local ppq = project.song.ppq()` (`96`).
   * `project.patterns.setNotes(pat, ch, list)` (**3 arguments**, temps/durée en **ticks**).
   * `project.patterns.setBars(pat, bars)`, `project.playlist.setTrackName(tr, name)`, `project.playlist.addPatternClip(pat, tr, tick)`.
   * Clips audio longs : `project.playlist.addAudioClip(ch, tr, tick, 0)`.

---

## Chapitre 5 — Scripts Partagés (`scripts/`) & Protocole Économie Tokens / CPU

| Script | Rôle Exact & Vrais Flags CLI | Coût CPU / Usage Optimal |
| :--- | :--- | :--- |
| **`scripts/studio_lib.py`** | `GridlineMCP`, `setup_standard_buses(..., profile=...)`, `verify_midi_gate`, `check_pattern_audibility`, `clean_audio_sample`, `extract_clean_oneshot`, `stabilize_stereo_low_mid`. | Importer dans `build_song.py` → 1 seul appel au lieu de 40 appels MCP. |
| **`scripts/audio_metrics.py`** | `analyze_single_file(..., with_clap=False)`, `analyze_pitch_and_purity`, `analyze_vocal_signature`, `score_candidate`. | `< 50 ms`/fichier sans CLAP. |
| **`scripts/audit_project_sources.py`** | Audit pré-mix 1-shot d'un `.gridline` + `--prepare-samples <dir>` (détecte `DISSONANCE_RISK`, `LOW_MID_WIDE_PHASE`, `NYQUIST_JUMP`, `mixer_warnings`). | Rapide (0 GPU, 0 CLAP). |
| **`scripts/render_and_audit.py`** | Rendu silencieux + `analyze_mix` + `audit_mix_deep` + `section_lufs` + Quality Gate multi-genres. Flags : `--profile <genre>`, `--target-lufs`, `--bpm`, `--beats-per-bar`, `--range B1-B2`, `--no-neural`, `--sections`, `--mp3`. | Itérer avec `--range --no-neural` (~5× plus rapide) ; audit neural complet **1 seule fois** à la fin. |
| **`scripts/index_samples.py`** | Indexe `samples/` -> `mcp-servers/audio-index/sample_index.json` (hors `sfz/`). Flags : `--no-clap`, `--force`, `--limit`. | Lancer en CLI hors session IA si besoin de `select_sample.py`. |
| **`scripts/select_sample.py`** | Sélection depuis `sample_index.json`. **Note** : `--query` filtre le **nom de fichier/chemin** (pas du texte CLAP). Flags : `--role`, `--query`, `--target-key`, `--mono-only`, `--no-clap`, `--auto-fetch`. | Utiliser `--no-clap` si urgence CPU. |
| **`scripts/fetch_samples_api.py`** | Téléchargement CC0 Freesound v2 / Lots of Sounds + normalisation 24-bit + filtrage `--mono-only` + `--no-clap`. | Appeler uniquement si son manquant en local. |
| **`scripts/build_references.py`** | Calcule `golden_embeddings.npz` depuis `golden_references/<role>/` (actuellement vide tant qu'aucun sample référence n'y est placé). | Optionnel (CLI humain). |

---

## Chapitre 6 — Guide Utilisateur Rapide

```bash
# 1. Ouvrir Gridline en mode Live GUI connecté aux agents (port 47390)
/home/gamo/Bureau/musicos/scripts/launch-gridline.sh

# 2. Réindexer la base SQLite rapide (samples.db) ou l'index JSON psychoacoustique (sans CLAP rapide)
cd /home/gamo/Bureau/musicos && .venv/bin/python mcp-servers/audio-index/indexer.py
cd /home/gamo/Bureau/musicos && .venv/bin/python scripts/index_samples.py --no-clap

# 3. Vérifier la stack Python DSP
.venv/bin/python -c "import parselmouth, librosa, essentia, pretty_midi; print('DSP stack OK')"
```

* **Exemples de prompts multi-genres** :
  * *« Crée une ballade piano/cordes mélancolique en 3/4 à 84 BPM en Ré mineur, 90 secondes, intime puis épique. »*
  * *« Compose un morceau Lo-Fi Jazz Hop chaleureux à 82 BPM en Fa majeur (accords Maj7/m9), 2 minutes, sans dureté dans les aigus. »*
  * *« Crée une instru Dark Trap à 142 BPM en Fa mineur de 2 minutes, triste et nerveuse, avec contraste Intro / Refrain. »*
  * *« Produis un morceau Synthwave / House lumineux à 122 BPM en La mineur, groove 4-on-the-floor et lead mélodique accrocheur. »*
  * *« Analyse ce thème audio (.mp3) et transforme-le en remix Electro / Rap propre à -11 LUFS. »*
