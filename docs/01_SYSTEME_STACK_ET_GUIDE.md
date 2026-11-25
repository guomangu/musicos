# Facette 1 — Système, Stack, Catalogue `gl.*`, DSL `gl_dsl.py` & Guide Utilisateur

> **Cheatsheet IA (L'essentiel v2.0)**
> - **Orientation** : Consulter [`docs/00_SOMMAIRE.md`](00_SOMMAIRE.md) (Tour 0) pour localiser tout dossier/fichier/MCP dans l'ordre du workflow.
> - **Moteur & DSL** : Gridline `0.7.7` (Live `--attach` port `47390` ou Headless `.gridline`). Privilégier les plugins natifs `gl.*` et **importer `scripts/gl_dsl.py`** (`eq_band`, `eq_params`, `reverb_room`, `limiter_master`, `sidechain_send`, `db_to_auto`, `synth_voice`) : zéro calcul manuel de `Band N width`, `k128`, décisecondes ou `Dry (full)`.
> - **Mémoire sonore unifiée** : `audio-index:search_samples` et `scripts/select_sample.py` interrogent la base SQLite **`samples.db`** (`samples` + `sample_dsp_metrics`, **`< 25 ms`**, zéro CLAP synchrone ; si vecteurs CLAP absents, fallback DSP automatique + `"warning"`).
> - **Unités** : Composer uniquement en **beats 0-indexed** dans Python (`song_builder.py` / `groove_lib.py` / `transitions_lib.py` gèrent `pretty_midi`, MCP et automations).
> - **Économie Tokens & CPU** : 1 script `build_song.py` via `song_builder.run_song_cli` · `--range <b1>-<b2> --no-neural --auto-fix --max-passes 2` en itération · 1 rendu final `--mp3 --sections`.

---

## Chapitre 1 — Architecture Globale & Sobriété CPU (Linux / PipeWire)

### 1.0 Portabilité du dépôt (clone universel)
- **Racine du studio** : tous les scripts et tests résolvent la racine via la variable d'environnement **`AI_STUDIO_DIR`**, sinon le dossier parent de `scripts/` (déduit de `__file__`). Aucun chemin absolu machine n'est codé en dur : un clone GitHub ailleurs que `$AI_STUDIO_DIR` fonctionne tel quel (`export AI_STUDIO_DIR=/chemin/du/clone` si besoin).
- **Template projet vide** : `studio_lib.resolve_empty_template()` prend le premier disponible parmi `$GRIDLINE_EMPTY_TEMPLATE`, `/usr/share/gridline/assets/templates/Basics/Empty with mixer tracks.gridline`, puis le **template embarqué `assets/empty_with_mixer_tracks.gridline`** (même repli dans `scripts/gridline-mcp.sh` / `launch-gridline.sh`).
- **Dépendances** : `requirements.txt` est installable tel quel depuis PyPI (`essentia>=2.1b6.dev1389` = max publié ; les builds dev plus récents se font via conda/source). `essentia` et `onnxruntime` sont des imports **gardés** : sans eux, l'indexeur retombe sur librosa et seul l'outil de transcription Basic Pitch est indisponible (erreur explicite).
- **Client MCP Gridline** (`studio_lib.GridlineMCP`) : lecture stdout avec **timeout** (`rpc_timeout_s`, défaut 180 s), thread de vidange du stderr (anti-deadlock), `close()` avec `kill()` si le processus ne meurt pas en 15 s. **Audition in-session sans .wav** : `analyze_loudness(scope="pattern"|"playlist")`, `solo_channel`/`unsolo`, `set_channel_muted`, `play_transport` (best-effort) — permettent de tester chaque piste et le morceau dans la session Gridline ouverte (`gridline mcp --attach`) **sans jamais compiler de .wav**.
- **Outils MCP rapides (audio-analyzer)** : `audit_track_soundcheck_fast` (`audition="dsp"` ~50 ms sans Gridline, `"session"` = sonde in-session, `"render"` = legacy wav), `balance_tracks_in_session` (équilibrage des volumes par solo + mesure de pattern, trims bornés), `audit_vertical_stage_session` (V1..V4 par niveaux in-session).

```text
Prompt Utilisateur → [0. Sommaire] docs/00_SOMMAIRE.md
   ├──▶ [1. Blueprint & Sons]  audio-index (samples.db) · reverse-engineer · vocaloid · gl_dsl
   ├──▶ [2. Partition & Gate]  master_sketch.mid + groove_lib/transitions_lib → verify_midi_gate(auto_fix=True)
   ├──▶ [3. Build Atomique]    build_song.py (song_builder + gl_dsl) → projet.gridline
   └──▶ [4. Mix & Validation]  render_and_audit.py (--no-neural --auto-fix → rendu final --mp3) + analyze_stem_hierarchy
```

### 1.1 Environnement & Budget DSP Strict
* **OS / Gridline / Python (`.venv`)** : Fedora + PipeWire (`quantum >= 1024`, `launch-gridline.sh`) · Gridline (`bufferFrames = 1024`, `freeze.autoHibernate = true`) · DSP rapide 0 GPU (`praat-parselmouth>=0.4.7`, `librosa`, `scipy`, `soundfile`, `essentia`, `mosqito`, `pyloudnorm`, `pyroomacoustics`, `pretty_midi`, `music21`, `jsonschema`) · Perceptuel CPU réservé au rendu final (`transformers`/CLAP, `audiobox_aesthetics`). Arborescence détaillée → [`docs/00_SOMMAIRE.md`](00_SOMMAIRE.md).
* **Filtrage & Enveloppe à la source (`gl_dsl.synth_voice`)** : tout synthé `gl.*` reçoit dès sa création son LPF et son enveloppe ADSR par rôle (`bass`/`chords`/`pad`/`lead`/`pluck`) + `Polyphony <= 12` (`<= 16` piano), `Unison <= 2`, `gl.eq Quality = 0` sur inserts (`1` sur `Master 0`), 1 seul bus `FX Room`.
> - **Template vide & Isolation (`1 sonorité = 1 piste dédiée`)** : `/usr/share/gridline/assets/templates/Basics/Empty with mixer tracks.gridline` s'arrête à l'insert `16` ; `studio_lib.setup_standard_buses()` ajoute les bus `17..22` (`BUS Drums`, `BUS Bass`, `BUS Music`, `FX Room 20`, `FX Delay 21`, `FX Hall 22`) + 2 étages de largeur mid/high sur `BUS Music`. Chaque sonorité (`Kick`, `Clap`, `HiHat`, `808`, `Chords`, `Lead`…) garde son canal, son insert, ses patterns propres et sa piste Playlist nommée.

---

## Chapitre 2 — Inventaire des 5 Serveurs MCP & Index Unifié (`samples.db`)

Serveurs Python `mcp>=2.3.0` (`stdio`, logs sur `stderr`). Liste complète des outils → [`docs/00_SOMMAIRE.md §5`](00_SOMMAIRE.md#5-index-des-serveurs-mcp-mcp-servers--outils-associés).

| Serveur | Script | Outils Clés & Coût CPU |
| :--- | :--- | :--- |
| **`gridline`** | `scripts/gridline-mcp.sh` | Live (`--attach` `47390`) ou Headless ; en build complet, passer par `studio_lib.GridlineMCP` (`batch` + `set_params`). |
| **`audio-index`** | `mcp-servers/audio-index/audio_index_server.py` | **Instantané (`< 25 ms`)** : `search_samples`, `find_compatible_samples`, `get_sample_info`, `list_sample_packs`, `search_synth_presets`, `analyze_sample_dsp`, `select_best_samples` *(lit désormais **`samples.db`** unifiée : `samples` + `sample_dsp_metrics`, fallback DSP sans CLAP si absent)* · **Optionnels** : `evaluate_sample_quality(with_clap=False)`, `fetch_samples_online`, `reindex_samples`. |
| **`audio-analyzer`** | `mcp-servers/audio-analyzer/audio_analyzer_server.py` | **Rapide (`< 1 s`)** : `audit_midi_harmony` *(multi-signatures + `phrase_plan`)*, `analyze_mix`, `analyze_low_end_masking`, `compare_renders`, `check_pipewire_health` · **Stems / Complet** : `render_project_stems`, `analyze_stem_hierarchy`, `audit_mix_deep(use_neural=False)`, `match_reference_audio`. |
| **`reverse-engineer`** | `mcp-servers/reverse-engineer/reverse_engineer_server.py` | `analyze_song_for_rap_flip`, `transcribe_audio_to_midi` (Basic Pitch ONNX), `split_stems` (`htdemucs_6s`/`htdemucs`). |
| **`vocaloid`** | `mcp-servers/vocaloid/vocaloid_server.py` | `list_voicebanks`, `syllabify_lyrics`, `create_ustx`, `synthesize_vocal_preview`, `open_in_openutau`. |

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

> **Règle Anti-Stridence / Anti-« mix en U » (v3.1+, source unique `config/mix_profiles.json` via `scripts/mix_config.py`)** :
> - **Jamais de LPF brique (`gl.eq` 24 dB/oct) sous `6.5 kHz` sur l'harmonie (Pads/Accords) ni sous `9 kHz` sur le Lead** (planchers `source_lpf_floors_hz` dans `config/mix_profiles.json`) : un tel filtre creuse les haut-médiums et produit un « mix en U ».
> - Pour **assombrir** un timbre brillant sans ce défaut : *high-shelf* doux atténuateur (`-3 dB` ~) ou **filtre de synthé musical** (pente douce 6–12 dB/oct, `Cutoff` du `gl.sampler`), jamais un LPF brique agressif. Ne pas laisser un synthé *saw/FM* complètement ouvert jusqu'à `20 kHz` non plus : adoucir par shelf.
> - **Cloches / FM / Métallique (`gl.fm`, `gl.alloy`, `gl.matrixfm`)** : adoucir par shelf/filtre synthé (jamais de LPF brique < 9 kHz) + creux `gl.eq` (`Band type = 6 Peak`, `-3.0 dB @ 3200 Hz`, `Band N width = 0.35` *≈ Q 1.6*).
> - **`gl.sampler` (mélodique ou hats brillants)** : Exploiter nativement les paramètres `Filter` (`1=Low pass`, `2=Low pass 24`) / `Cutoff` / `Resonance` du sampler si le sample est trop brillant.

> **Règle d'Enveloppes ADSR sur Synthés `gl.*` (Anti-Rendu « Orgue Statique »)** :
> Ne jamais laisser un synthé `gl.*` sur son enveloppe *init* (`Attack = 0`, `Sustain = 100 %`, `Release = 0`) ; sculpter l'enveloppe selon le rôle dès la création :
> - **Pads / Nappes** : `Attack` lent (**`150–400 ms`**), `Release` ample (**`400–800 ms`**).
> - **Plucks / Bells / Cloches** : `Attack` court (**`2–8 ms`**), `Decay` moyen (**`250–600 ms`**), `Sustain` bas (**`0–20 %`**).
> - **Leads expressifs** : `Attack` doux (**`10–30 ms`**), `Release` court (**`80–180 ms`**), léger **`Glide`** si mono.

### 3.2 Effets Natifs Clés & DSL Sémantique Obligatoire (`scripts/gl_dsl.py`)
Ne plus calculer manuellement les unités internes Gridline (`k128`, `ds`, `hHz`, `Band N width` vs `legacy Q`) : utiliser **`gl_dsl.py`** + `ChannelSpec.role`/`cutoff_hz` :
* **`gl.sampler`** : `plugin.data = "/chemin/absolu.wav"` (WAV 24-bit ; éviter `.sfz` externe, muet en GUI), `Root note`, `Fine tune` (cents signés exacts), `One shot`, `Voices` (`1` pour 808/sub, `12` pour accords), `Filter` (`1=LPF`, `2=LPF 24`), `Cutoff` (Hz), `Resonance`.
* **Synthés (`synth_voice(plugin, role, valence, cutoff_hz, expressive=True)`)** : couvre 13 synthés `gl.*` dans `SYNTH_REGISTRY` (`gl.analog`, `gl.wavetable`, `gl.warmsynth`, `gl.dcosynth`, `gl.twinva`, `gl.sawswarm`, `gl.string`, `gl.pluck`, `gl.fm`, `gl.matrixfm`, `gl.keys`, `gl.fingerbass`, `gl.sampler`), en appliquant LPF + enveloppe ADSR adaptée au rôle + ajout auto d'un `gl.eq` insert si le synthé n'a pas de LPF interne (`gl.fm`, `gl.keys`, etc.). **Phase 8 `Q6`** : `expressive=True` ajoute un mouvement LFO filtre/vélo/vibrato subtil uniquement via des paramètres vérifiés (`gridline plugin-params`) et expose `makeup_gain_db` (`+2.5/+1.5/0 dB` selon `cutoff < 3500 / < 5500 / >= 5500 Hz`), appliqué par l'étage É1 quand `soundcheck_metrics.json` existe.
* **Espace 3D (`delay_send_fx`, `reverb_hall`)** : `delay_send_fx(bpm, valence, division="1/8.")` paramètre `gl.delay` synchro tempo (`Dry=0`, feedback `20-32 %`, low/high cut `400/3500 Hz`) ; `reverb_hall(decay_sec=1.8)` paramètre `gl.reverb` Hall (`Dry=0`, `Dry (full)=0`, `HPF 500`, `LPF 5500/4500 Hz`). `ChannelSpec.delay_send` / `hall_send` routent vers les inserts `21` / `22`.
* **EQ (`eq_band`, `eq_params`, `eq_gain_to_auto`)** : traduit `type` (`"hpf"`, `"lpf"`, `"peak"`, `"notch"`, `"low_shelf"`, `"high_shelf"`, `"band_pass"` → `Peak=6`, jamais `4=Notch`), `q` → **`Band N width`** (`0.45`≈Q1.2 · `0.39`≈Q1.4 · `0.35`≈Q1.6 · `0.22`≈Q2.8) et `eq_gain_to_auto(gain_db)` pour les automations `Band N gain` (`[-18, +18] dB`).
* **Reverb / Limiter / Sidechain / Automation** : `reverb_room(decay_sec, hpf_hz, lpf_hz, wet_pct, er_pct)` (force `Dry (full)=0`, gère `ds`/`k128`/`hHz`) · `reverb_hall(...)` (Hall long, `Dry=0`) · `delay_send_fx(bpm, valence)` (`gl.delay` synchro BPM, feedback bande limitée) · `limiter_master(gain_db, ceiling_db)` (force `Attack (look-ahead)=0.002`, `0.2 dB safety=1`) · `sidechain_send(src, dst, amount)` (`sidechain=True, preFader=True`) · `db_to_auto(db)` (fader `0..1.25`).
* **`gl.comp` / `gl.dist` / `gl.stereo`** : `comp` (`Detector input` `0=Main, 1=Sidechain`, `Threshold`, `Ratio`, `Attack`, `Release`, `Makeup`, `Lookahead`) · `dist` (`Mode=0 Soft`, `Oversampling=2`, `Remove DC=1`) · `stereo` (`Width=0.0` mono sur `BUS Bass`).

```python
from gl_dsl import delay_send_fx, eq_band, eq_params, limiter_master, reverb_hall, reverb_room, sidechain_send, synth_voice, to_mixer_params
voice = synth_voice("gl.warmsynth", role="chords", valence=-0.6, cutoff_hz=4500.0, expressive=True)
eq_tuples = eq_params(3, [eq_band(1, "hpf", 220), eq_band(4, "peak", 800, -4.0, q=2.4), eq_band(7, "lpf", 10500)])
fx_tuples = to_mixer_params(20, 1, reverb_room(decay_sec=0.35, hpf_hz=450, lpf_hz=6800, wet_pct=45))
hall_tuples = to_mixer_params(22, 1, reverb_hall(decay_sec=1.8, valence=-0.4))
delay_tuples = to_mixer_params(21, 1, delay_send_fx(bpm=132.0, valence=0.0))
```

### 3.3 Plugins Tiers Vérifiés (Session GUI `--attach` uniquement)
* **Synthé / EQ / Tape / Reverb / Analyse** : Surge XT (`clap:org.surge-synth-team.surge-xt`), ZL Equalizer 2, TDR Nova, CHOWTapeModel, Dragonfly Hall/Room/Plate, LSP (`spectrum_analyzer_x1..16`, `oscilloscope`, `crossover`, `compressor`, `limiter`).

---

## Chapitre 4 — Table Universelle des Unités & Syntaxe MCP / Lua

> **Règle d'Or Temporelle** : Stocker **100 %** de la composition en **beats 0-indexed (`float`)** dans Python ; `song_builder.py` et `transitions_lib.py` effectuent les conversions vers `pretty_midi` (`b * 60/BPM`), MCP `set_notes` (`b`), Lua (`int(round(b * 96))` ticks), et blueprints (`bar = int(b // B) + 1`).

| Contexte | Unité | Indexation | Formule (`B` = numérateur métrique) |
| :--- | :---: | :---: | :--- |
| `song_blueprint.json` / `--sections` | `bar` | **1-indexed** | `bar = int(b // B) + 1` ; `b_start = (bar - 1) * B` |
| MCP `set_notes` / `add_automation` | `beats` | **0-indexed** | `b` (`add_automation` : point `position` + `value` normalisée) |
| `build_gridline_project(sections=...)` | `start_bar` | **0-indexed** | `(name, bar_1idx - 1, bar_count, kind)` |
| Lua (`ppq = 96`/beat) / `pretty_midi` | `ticks` / `s` | **0-indexed** | `tick = int(round(b * 96))` / `sec = b * 60.0 / bpm` |

* **Durée figure $F$ (`ms`)** : $(60\,000 / \text{BPM}) \times 4 \times F$ (pointé $\times 1{,}5$, triolet $\times 2/3$ ; à `120 BPM` : ronde `2000`, blanche `1000`, noire `500`, croche `250`, crocheT `167`, double `125`, triple `62.5` ms).
* **Syntaxe critique** : `set_song` utilise `"tempo"`, `"numerator"`, `"denominator"` · routage via `edit_channel(..., mixer=M)` (jamais `patch mixerTrack`) · `set_send(..., sidechain=True, preFader=True)` · `add_automation` exige `"track": <playlist_idx>`, `"param": "volume"` ou paramètre exact (`"Band 7 freq"` sur `gl.eq`) et `"position"` en beats (`db_to_auto(db)` pour valeur normalisée sur fader max `1.25`) · Lua `run_script` encapsulé dans `function apply() ... end` sans appel top-level (`setNotes(pat, ch, list)` en 3 args/ticks).

---

## Chapitre 5 — Raccourci Workflow des Scripts & Guide Utilisateur

Inventaire exhaustif fonction par fonction → [`docs/00_SOMMAIRE.md §4`](00_SOMMAIRE.md#4-index-complet-de-scripts-classé-par-ordre-dutilité-workflow).

| Couche Workflow | Scripts Clés |
| :--- | :--- |
| **Mémoire & Sources** | `audit_project_sources.py` · `select_sample.py` (`< 25 ms`, `samples.db`) · `fetch_samples_api.py --mono-only --no-clap` · `audio_metrics.py` · `index_samples.py --no-clap --workers 6` |
| **Préparation & Expressivité** | `sample_prep_lib.py` (`process_sample_specs`) · `groove_lib.py` (`apply_micro_strum`, `apply_haas_pocket`, `apply_anti_machine_gun`, `enforce_frequency_inertia`) · `transitions_lib.py` (`beat4_dropout`, `filter_sweep`, `section_gain_arc`, `outro_fade`) |
| **Validation MIDI & Build** | `midi_autofix.py` (`autofix_midi`) · `studio_lib.py` (`verify_midi_gate(..., auto_fix=...)`) · `gl_dsl.py` · `song_builder.py` (`build_gridline_project`, `run_song_cli`) |
| **Rendu, Auto-Fix & CI** | `render_and_audit.py` (`--range --no-neural --auto-fix --max-passes 2`) · `bootstrap_golden_refs.py` (`spectral_corridors.json` versionné) · `tests/regression_multi_genre.py` |

```bash
# 1. Gridline GUI connecté aux agents (port 47390)
$AI_STUDIO_DIR/scripts/launch-gridline.sh

# 2. Indexer / rafraîchir samples.db unifiée (métriques DSP rapides 0 CLAP)
cd $AI_STUDIO_DIR && OMP_NUM_THREADS=1 .venv/bin/python scripts/index_samples.py --no-clap --workers 6

# 3. Suite CI complète multi-genres (7/7 headless)
cd $AI_STUDIO_DIR && .venv/bin/python tests/regression_multi_genre.py -v
```
