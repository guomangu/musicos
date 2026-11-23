# Facette 0 — SOMMAIRE & Index Autonome du Framework AI-Studio

> **Objectif** : Point d'entrée unique pour qu'un agent IA autonome localise immédiatement **chaque dossier et fichier** du framework (`/home/gamo/Bureau/musicos`), connaisse ses **fonctions / outils MCP exacts**, et suive l'**ordre d'utilité exact dans le workflow** sans explorer à l'aveugle.

---

## 1. Ordre d'Utilité dans le Workflow (Pipeline Standard 0 → 6)

| Ordre | Étape Workflow | Scénario | Fichiers / Outils Principaux à Utiliser | Produit / Gate de Sortie |
| :---: | :--- | :---: | :--- | :--- |
| **0** | **Orientation & Règles** | Tous | `.rules` (symlink `AGENTS.md`, `GEMINI.md`), `docs/00_SOMMAIRE.md`, puis la facette `docs/01..05` du tour courant uniquement | Route choisie (`A`, `B`, `C`, `D`) |
| **1** | **Environnement & Démarrage** | Tous | `scripts/launch-gridline.sh` (GUI `--attach`), `scripts/gridline-mcp.sh` (MCP), `audio-analyzer:check_pipewire_health` | Gridline / PipeWire prêts (`quantum >= 1024`) |
| **2** | **Conception & Blueprint** | A / B / C | `schemas/song_blueprint.schema.json`, `docs/02_*`, `docs/03_*` | `<projet>/song_blueprint.json` valide (`has_drums`, `mix_profile`, `emotion_vector`, `harmonic_core`) |
| **3** | **Analyse Sources / Reverse-Eng / Sons** | A / B / C / D | **B/D** : `scripts/audit_project_sources.py`, `scripts/sample_prep_lib.py` · **C** : `reverse-engineer` (`analyze_song_for_rap_flip`, `split_stems`, `transcribe_audio_to_midi`) · **A** : `audio-index:search_samples` / `scripts/select_sample.py`, `audio-index:analyze_sample_dsp`, `audio-index:search_synth_presets`, `scripts/gl_dsl.py` (`gl.*`), `scripts/fetch_samples_api.py` · **Chant** : `vocaloid` | `<projet>/samples/` + `manifest.json` (`recommended_gl_fine_tune_ct`) ou palette `gl.*` |
| **4** | **Composition & Gate MIDI** | A / B / C | `musicality_metrics.py:optimize_voice_leading`, `scripts/song_builder.py` (`degree_to_midi`, `export_master_sketch`), `scripts/groove_lib.py`, `scripts/transitions_lib.py`, `scripts/studio_lib.py:verify_midi_gate(..., auto_fix=...)` / `scripts/midi_autofix.py`, `audio-analyzer:audit_midi_harmony` | `<projet>/master_sketch.mid` validé (`gate_passed == true`, **0** rendu audio avant PASS) |
| **5** | **Construction `.gridline`** | A / B / C | `scripts/song_builder.py` (`build_gridline_project`, `run_song_cli`, `ChannelSpec`), `scripts/gl_dsl.py` (`eq_band`, `eq_params`, `reverb_room`, `limiter_master`, `sidechain_send`, `synth_voice`), `scripts/studio_lib.py` (`GridlineMCP`, `setup_standard_buses`, `check_pattern_audibility`) | `<projet>/projet.gridline` sauvegardé + patterns vérifiés (`1 sonorité = 1 piste`) |
| **6** | **Mixage, Stems & Quality Gate** | A / B / C / D | `scripts/render_and_audit.py` (`--range --no-neural` puis `--auto-fix --max-passes 2`, puis rendu final complet `--mp3 --sections`), `audio-analyzer` (`render_project_stems`, `analyze_stem_hierarchy`, `audit_mix_deep`, `match_reference_audio`), `scripts/bootstrap_golden_refs.py` (`golden_references/spectral_profiles.json`) | `renders/master.wav` + `renders/autofix_history.json` (`quality_gate.passed == true`) |
| **7** | **CI & Non-Régression Framework** | Dev / CI | `tests/test_gl_dsl.py`, `tests/test_phase2_memory_and_midi.py`, `tests/test_phase3_autofix_and_stems.py`, `tests/test_phase4_expressivity.py`, `tests/regression_multi_genre.py`, `mcp-servers/audio-analyzer/test_musicality_metrics.py` | Tous tests unitaires + CI multi-genres `7/7 PASS` |

---

## 2. Index des Fichiers Racine & Dossiers Config / Assets (`/home/gamo/Bureau/musicos/`)

| Chemin | Rôle & Fonctions | Ordre / Usage |
| :--- | :--- | :--- |
| `.rules` *(symlinks `AGENTS.md`, `GEMINI.md`)* | Règles normatives globales : routeur de scénario `A/B/C/D`, routage documentaire just-in-time, pipeline en 5 tours et 14 invariants non-négociables. | **0** (lu automatiquement par l'agent) |
| `README.md` | Présentation générale de l'environnement AI-Studio, installation rapide et prérequis Linux. | **0** (référence humaine) |
| `.env` / `.env.example` | Variables d'environnement API (`FREESOUND_API_KEY`, `LOTSOFSOUNDS_API_KEY`). | **3** (requis par `fetch_samples_api.py` / `fetch_samples_online`) |
| `.zed/settings.json` | Configuration éditeur Zed + déclaration des serveurs MCP locaux (`gridline`, `audio-index`, `audio-analyzer`, `reverse-engineer`, `vocaloid`). | **1** (config IDE/MCP) |
| `install.sh` | Installeur Fedora idempotent (paquets audio PipeWire, `.venv`, plugins CLAP/VST3, modèles, rescan). Flags `--check-only`, `--skip-plugins`, `--skip-clap`, `--skip-zed`, `--no-sudo`, `--force-rescan`. | **1** (setup initial machine) |
| `requirements.txt` | Dépendances Python du `.venv` (`mcp`, `numpy`, `scipy`, `soundfile`, `pyloudnorm`, `librosa`, `praat-parselmouth`, `essentia`, `mosqito`, `pyroomacoustics`, `matchering`, `music21`, `pretty_midi`, `onnxruntime`, `mido`, `transformers`, `audiobox_aesthetics`, `scikit-learn`). | **1** (setup `.venv`) |
| `.gitignore` | Règles d'exclusion Git (`.venv`, `.env`, gros média, bases temporaires). | **—** (Git) |
| `.venv/` | Environnement virtuel Python isolé. **Toujours** invoquer `/home/gamo/Bureau/musicos/.venv/bin/python`. | **1–7** |
| `schemas/song_blueprint.schema.json` | Schéma JSON Schema normatif validant tout `song_blueprint.json` (`mix_profile`, `has_drums`, `emotion_vector`, `bpm`, `time_signature`, `harmonic_core`, `motif_cell`, `structure`, `frequency_plan`, `tracks`, `vocals`). | **2** (`song_builder.load_blueprint(..., validate=True)`) |
| `samples` *(symlink → `soundbanks/`)* | Point d'entrée canonique vers les banques locales utilisées par les indexeurs (`sfz/` exclu automatiquement). | **3** |
| `soundbanks/` | Stockage réel des packs locaux WAV/OGG/MP3, boucles et sous-dossiers `.sfz` (exclus du scan mélodique). | **3** |
| `SAMPLES/` | Arborescence pré-classée par rôle (`808/`, `bass/`, `clap/`, `fx/`, `guitar/`, `hat/`, `kick/`, `loop/`, `open_hat/`, `perc/`, `piano/`, `snare/`, `synth/`, `synth-samples/`, `vocal/`). | **3** |
| `golden_references/` | Références golden par rôle (`bass_sub/`, `chord_keys/`, `hihat_perc/`, `kick/`, `lead_melody/`, `pad_atmosphere/`, `snare_clap/`), `README.md`, et **`spectral_profiles.json`** (empreintes Bark 24 bandes + pente spectrale + ratios `sub`/`low_mid`/`air` des 7 profils). | **3 & 6** (`build_references.py`, `bootstrap_golden_refs.py`) |
| `voicebanks/` | Banques vocales OpenUTAU / DiffSinger (`README.md`, `TIGER_DS_v106/`). | **3** (`vocaloid`) |
| `tools/` | Binaires et modèles externes : `tools/models/nmp.onnx` (Basic Pitch ONNX) et `tools/OpenUtau/OpenUtau` (rendu chant Linux `0.1.576`). | **3** (`reverse-engineer`, `vocaloid`) |
| `projects/` | Projets Gridline actifs (`active/`, `active2/`..`active9/`, `active99/`, `selectionson/`, `validation_fm_dark/`), contenant chacun `song_blueprint.json`, `master_sketch.mid`, `samples/`, `scripts/`, `renders/`, `.gridline`. | **2–6** |
| `renders/` | Dossier de sortie par défaut / temporaire pour rendus audio à la racine. | **6** |

---

## 3. Index des Facettes Documentaires (`docs/`)

Lire **just-in-time** uniquement la facette correspondant à l'étape courante :

| Fichier | Quand la Consulter | Résumé du Contenu |
| :--- | :--- | :--- |
| **`docs/00_SOMMAIRE.md`** *(ce fichier)* | **Tour 0** (Orientation) | Carte complète des dossiers/fichiers, fonctions clés, outils MCP et ordre exact d'utilisation. |
| **`docs/01_SYSTEME_STACK_ET_GUIDE.md`** | **Tour 0 / 3** (Outils & DSL) | Stack Linux/PipeWire, budget DSP strict, inventaire des 5 MCP, catalogue vérifié des synthés/effets `gl.*`, DSL `gl_dsl.py`, conversion d'unités beats/ticks/s et syntaxe MCP/Lua. |
| **`docs/02_COMPOSITION_EMOTIONNELLE_ET_THEORIE.md`** | **Tour 1 / 2** (Musique) | Table MIDI & `SCALES`, modèle `(V, T, E, D)`, modes & verrouillage par valence, formules d'accords & `optimize_voice_leading`, 14 progressions, recette mélodie (`A-A'-A''-B` et sombre), humanisation `groove_lib.py`, 11 matrices rythmiques, palettes multi-genres. |
| **`docs/03_ARCHITECTURE_BLUEPRINT_ET_COHESION.md`** | **Tour 1 / 2 / 3** (Structure) | Contrat `song_blueprint.json`, templates multi-genres, matrice d'arrangement par section, grille 4/8/16 & `transitions_lib.py`, plafond cognitif (5–8 pistes simultanées vs isolation `1 sonorité = 1 piste`), `master_sketch.mid`, seuils `audit_midi_harmony`, table `_infer_roles()`, correction `midi_autofix.py`, patron `build_song.py`. |
| **`docs/04_ANALYSE_SONORE_SAMPLES_APIS_ET_VGM.md`** | **Tour 1 / 3** (Audio & Chant) | Arbre de choix des sons, index unifié `samples.db` (`< 25 ms`), pipeline DSP 0 GPU, pureté `mono_oneshot` & signature vocale, lois d'hygiène & `sample_prep_lib.py`, APIs CC0 (`fetch_samples_api.py`), protocole Reverse-Engineering/Remix (`reverse-engineer`), workflow chant (`vocaloid` / OpenUTAU). |
| **`docs/05_MIXAGE_QUALITY_GATE_ET_APPRENTISSAGE.md`** | **Tour 4 / 5** (Mix & Validation) | 7 `mix_profile`s, plan 5 bandes, pente rose (`-3..-4.5 dB/oct`), spatialisation 3D, topologie bus `17..20` + `Master 0`, hiérarchie de stems (`analyze_stem_hierarchy`), `spectral_match_score`, Quality Gate conditionnel (dont `has_drums=false`), table Symptôme→Fix + `--auto-fix`, playbook des pièges, journal de références. |

---

## 4. Index Complet de `scripts/` (Classé par Ordre d'Utilité Workflow)

| Ordre | Script | Fonctions / Classes Clés | Rôle Exact dans le Pipeline |
| :---: | :--- | :--- | :--- |
| **1** | `scripts/launch-gridline.sh` | Script bash | Lance Gridline GUI avec environnement PipeWire/CPU optimisé (`quantum >= 1024`) pour `--attach` (`127.0.0.1:47390`). |
| **1** | `scripts/gridline-mcp.sh` | Script bash | Wrapper MCP lancé par Zed/Antigravity : attache au Gridline GUI vivant si présent, sinon démarre Gridline headless. |
| **2** | `scripts/audit_project_sources.py` | `_analyze_808_harmonics`, `_analyze_long_loop`, `audit_gridline_sources`, `main` | **Audit pré-mix/pré-compo** d'un `.gridline` (et `--prepare-samples <dir>`) : repère `DISSONANCE_RISK`, `LOW_MID_WIDE_PHASE`, `NYQUIST_JUMP`, dérive d'accordage et `mixer_warnings`. |
| **3a** | `scripts/select_sample.py` | `_fast_comfort_score`, `_fast_role_score`, `_select_best_from_sqlite`, `select_best_from_db`, `main` | Sélection ultra-rapide (`< 25 ms` sur 5k+ lignes) depuis SQLite `samples.db` unifiée (filtres `role`, `target_key`, `mono_only`, métriques DSP, score d'affinité rôle + pitch-shift optimal, repli sans CLAP). |
| **3b** | `scripts/audio_metrics.py` | `file_md5`, `load_and_normalize_audio`, `check_hard_rejects`, `analyze_pitch_and_purity`, `analyze_vocal_signature`, `extract_dsp_metrics`, `compute_comfort_score`, `compute_role_score`, `ClapEmbedder`, `compute_masking_penalty`, `compute_pitch_shift`, `score_candidate`, `analyze_single_file` | Bibliothèque DSP déterministe de référence (`praat-parselmouth`, `librosa`, `essentia`, phase M/S, Bark 24 bandes, HPSS rapide) + encodeur CLAP optionnel. |
| **3c** | `scripts/fetch_samples_api.py` | `load_studio_env`, `normalize_to_wav24`, `search_freesound`, `search_lotsofsounds`, `fetch_and_filter_samples`, `main` | Téléchargement CC0 depuis Freesound v2 / Lots of Sounds (`.env`), normalisation WAV 24-bit `44.1 kHz` et filtrage `mono_oneshot` / DSP (`--no-clap`). |
| **3d** | `scripts/index_samples.py` | `index_directory_psychoacoustic`, `main` | Alimente `mcp-servers/audio-index/samples.db` (table `samples` + `sample_dsp_metrics`, hors `sfz/`) ou l'index JSON hérité ; supporte `--no-clap --workers N --force --limit`. |
| **3e** | `scripts/sample_prep_lib.py` | `read_stereo`, `write_tmp_clean`, `apply_eq_peak`, `apply_soft_saturation`, `varispeed_resample`, `pitch_shift_stereo`, `time_stretch_stereo`, `trim_leading_silence`, `measure_fft_tuning`, `build_sample_from_spec`, `audit_samples_manifest`, `process_sample_specs` | Pipeline déclaratif centralisé de préparation de samples : découpe, EQ, pitch/varispeed, saturation, time-stretch, stabilisation phase grave, export 24-bit et calcul exact `recommended_gl_fine_tune_ct` dans `manifest.json`. |
| **4a** | `scripts/groove_lib.py` | `ms_to_beats`, `apply_micro_strum`, `apply_haas_pocket`, `apply_anti_machine_gun`, `enforce_frequency_inertia` | Humanisation déterministe des listes de notes en beats : micro-strum accords, décalage Haas Kick/Snare/Hat/Basse, modulation vélo/longueur anti-mitraillette, durée mini sub `< 100 Hz`. |
| **4b** | `scripts/transitions_lib.py` | `beat4_dropout`, `filter_sweep`, `section_gain_arc`, `outro_fade` | Transition et macro-dynamique automatisées : drop-out beat 4 (coupe batterie/basse + `FX Room`), balayage LPF (`gl.eq`/`Cutoff`), arche de gain `BUS Music` par section, fondu de sortie propre. |
| **4c** | `scripts/midi_autofix.py` | `autofix_midi`, `main` | Correction déterministe et ré-audit d'un `.mid` : quantifie notes hors gamme, corrige bornes sub/apex, résout collisions `< MIDI 48`, projette les temps forts hors-accord, élimine frottements `b2`, avec rapport JSON (`fix_report`). |
| **5a** | `scripts/gl_dsl.py` | `clamp`, `q_to_width`, `eq_band`, `to_mixer_params`, `eq_params`, `reverb_room`, `limiter_master`, `sidechain_send`, `mixer_gain_to_auto`, `db_to_auto`, `SynthSpec`, `SynthVoiceConfig`, `synth_voice` (`SYNTH_REGISTRY` pour 18 synthés `gl.*`) | **DSL sémantique anti-erreur Gridline `0.7.7`** : traduit automatiquement Q→`Band N width`, types EQ, `gl.reverb` `k128`/`ds`/`hHz`, `gl.limiter`, sends `sidechain`, courbes d'automation dB, LPF & ADSR par rôle/valence sur 18 synthés natifs. |
| **5b** | `scripts/studio_lib.py` | `GridlineMCP`, `clean_audio_sample`, `stabilize_stereo_low_mid`, `extract_clean_oneshot`, `check_pattern_audibility`, `verify_midi_gate`, `setup_standard_buses` | Socle bas niveau : client JSON-RPC `GridlineMCP`, nettoyage/isolation audio, gate MIDI (avec support `time_signature`, `phrase_plan`, `auto_fix`), topologie des bus `17..20` + `Master 0` profilés par genre/valence. |
| **5c** | `scripts/song_builder.py` | `SCALES`, `degree_to_midi`, `load_blueprint`, `ChannelSpec`, `eq_band`, `eq_insert`, `eq_music_pocket`, `eq_kick_clean`, `export_master_sketch`, `build_gridline_project`, `run_song_cli` | **Moteur central de build `.gridline`** : valide le blueprint, exporte `master_sketch.mid`, câble canaux + `gl_dsl.synth_voice` + inserts/bus + sidechain + patterns par section/sonorité + Playlist + markers + automations + audibilité. |
| **6a** | `scripts/bootstrap_golden_refs.py` | `compute_bark24_distribution`, `build_spectral_profiles`, `compute_spectral_match`, `main` | Construit `golden_references/spectral_profiles.json` depuis les rendus valides et calcule `spectral_match_score` (0 PyTorch) d'un WAV contre son `mix_profile`. |
| **6b** | `scripts/render_and_audit.py` | `parse_sections`, `render_and_evaluate`, `main` | Pipeline fermé Rendu + Audit + Quality Gate multi-genres (`--profile`, `--range`, `--no-neural`, `--no-drums`, `--blueprint`, `--sections`, `--mp3`) avec boucle **`--auto-fix --max-passes N`** écrivant `renders/autofix_history.json`. |
| **7** | `scripts/build_references.py` | `build_golden_references`, `main` | Script CLI optionnel encodant les fichiers de `golden_references/<role>/` dans `mcp-servers/audio-index/golden_embeddings.npz` (CLAP CPU). |

---

## 5. Index des Serveurs MCP (`mcp-servers/`) & Outils Associés

### 5.1 Serveur `gridline` *(Binaire Gridline `0.7.7`, lancé via `scripts/gridline-mcp.sh`)*
- **Lecture / Inspection** : `get_song`, `list_channels`, `list_patterns`, `get_pattern`, `get_playlist`, `get_mixer`, `get_parameters`, `get_automation`, `get_markers`, `list_plugin_catalog`, `list_presets`, `get_history`, `list_swing_presets`, `find`, `inspect`, `get_values`, `list_commands`, `windows`, `get_settings`.
- **Édition Atomique / Construction** : `set_song`, `add_channel`, `edit_channel`, `remove_channel`, `add_pattern`, `edit_pattern`, `remove_pattern`, `set_notes`, `set_steps`, `edit_notes`, `add_clip`, `edit_clip`, `remove_clip`, `add_automation`, `edit_automation`, `remove_automation`, `add_plugin`, `remove_plugin`, `set_parameter`, `set_parameters`, `edit_mixer_track`, `add_mixer_track`, `remove_mixer_track`, `set_send`, `stage_room`, `arrangement`, `add_marker`, `remove_marker`, `patch`, `batch`, `duplicate`, `select`, `run_script` (Lua), `run_command`, `transport`, `set_setting`, `undo`, `redo`.
- **Analyse & Export** : `analyze_loudness`, `render_audio`, `export_project`, `get_job`, `cancel_job`, `save_project`.

### 5.2 Serveur `audio-index` (`mcp-servers/audio-index/`)
| Fichier | Fonctions / Outils `@mcp.tool()` | Usage & Latence |
| :--- | :--- | :--- |
| `audio_index_server.py` | **`search_samples`**, **`find_compatible_samples`**, **`get_sample_info`**, **`list_sample_packs`**, **`search_synth_presets`**, **`analyze_sample_dsp`** | **Voie rapide (`< 25 ms`)** sur `samples.db` (fallback automatique sans CLAP + champ `warning` si vecteurs absents). |
| `audio_index_server.py` | **`evaluate_sample_quality`**, **`select_best_samples`**, **`fetch_samples_online`**, **`reindex_samples`**, **`reindex_synth_presets`** | Indexation/évaluation complètes (désactiver CLAP avec `with_clap=False` en session temps réel si besoin). |
| `indexer.py` | `init_db`, `analyze_sample`, `populate_dsp_metrics`, `upsert_dsp_metrics`, `enrich_clap_embeddings`, `index_directory` | Indexeur unifié alimentant tables `samples` et `sample_dsp_metrics` dans `samples.db` (exclut `sfz/`). |
| `preset_indexer.py` | `init_presets_table`, `index_gridline_presets` | Indexe les presets natifs `gl.*` dans la table SQLite `synth_presets`. |
| `samples.db` | Base SQLite unifiée | Stockage métadonnées + métriques DSP rapides + embeddings optionnels. |
| `golden_embeddings.npz` | Archive NumPy | Embeddings CLAP des références par rôle (si générés par `scripts/build_references.py`). |

### 5.3 Serveur `audio-analyzer` (`mcp-servers/audio-analyzer/`)
| Fichier | Fonctions / Outils `@mcp.tool()` | Usage & Rôle |
| :--- | :--- | :--- |
| `audio_analyzer_server.py` | **`analyze_mix`**, **`analyze_low_end_masking`**, **`compare_renders`**, **`audit_mix_deep`**, **`audit_midi_harmony`**, **`match_reference_audio`**, **`render_project_stems`**, **`analyze_stem_hierarchy`**, **`check_pipewire_health`** | Tous les audits DSP mix, masquage kick/basse, MIDI Gate multi-métriques (`time_signature`, `phrase_plan`), stems & hiérarchie Lead/Harmonie/Sub, santé PipeWire. |
| `deep_diagnostics.py` | `_third_octave_profile`, `_analyze_transients_and_roominess`, `_scan_glitches_and_crackle`, `_compute_psychoacoustics`, `_simulate_headphone_translation`, `run_deep_audit`, `resolve_allowed_pcs`, `audit_midi_file`, `match_with_reference`, `render_stems_and_audit`, `compute_pink_slope_db_per_oct`, `analyze_stem_hierarchy`, `inspect_pipewire` | Moteur analytique sous-jacent (tiers d'octave sans faux positifs sur pentes monotones, résonances, glitches, Loudness/Sharpness/Roughness, simulation casques `targets/Sony_WH-1000XM*.csv`, CLAP/Audiobox optionnels, hiérarchie stems). |
| `musicality_metrics.py` | `MusicalityThresholds`, `optimize_voice_leading`, `evaluate_midi_musicality`, `detect_minor_2nd_clashes` (+ helpers multi-métriques `4/4`, `3/4`, `6/8`, `5/4`, `free_ambient`) | Règles d'analyse musicale strictes utilisées par `verify_midi_gate` et `audit_midi_harmony`. |
| `targets/` | `Sony_WH-1000XM4.csv`, `Sony_WH-1000XM5.csv`, `Sony_WH-1000XM6.csv` | Courbes de réponse casques pour simulation de translation dans `deep_diagnostics.py`. |
| `test_musicality_metrics.py` | `TestMusicalityMetrics` | Tests unitaires du module de métriques musicales & `optimize_voice_leading`. |

### 5.4 Serveur `reverse-engineer` (`mcp-servers/reverse-engineer/`)
| Fichier | Outils `@mcp.tool()` | Usage & Rôle |
| :--- | :--- | :--- |
| `reverse_engineer_server.py` | **`analyze_song_for_rap_flip`**, **`split_stems`**, **`transcribe_audio_to_midi`** | Détection BPM/gamme/accords, séparation de stems Demucs (`htdemucs_6s`/`htdemucs`) et transcription audio→MIDI via Basic Pitch ONNX (`tools/models/nmp.onnx`). |

### 5.5 Serveur `vocaloid` (`mcp-servers/vocaloid/`)
| Fichier | Outils `@mcp.tool()` | Usage & Rôle |
| :--- | :--- | :--- |
| `vocaloid_server.py` | **`list_voicebanks`**, **`syllabify_lyrics`**, **`create_ustx`**, **`synthesize_vocal_preview`**, **`open_in_openutau`** | Découverte des voix (`voicebanks/`), syllabation FR/EN, génération `.ustx`, pré-synthèse WAV calée BPM/MIDI et ouverture dans `tools/OpenUtau/OpenUtau`. |

---

## 6. Index de la Suite de Tests & Validation (`tests/`)

| Fichier | Commande de Lancement | Couverture Exacte |
| :--- | :--- | :--- |
| `tests/test_gl_dsl.py` | `.venv/bin/python -m unittest tests/test_gl_dsl.py -v` | Phase 1 : `q_to_width`, `eq_band`, `reverb_room`, `limiter_master`, `sidechain_send`, `db_to_auto`, 18 synthés `SYNTH_REGISTRY`, compatibilité `song_builder.py`. |
| `tests/test_phase2_memory_and_midi.py` | `.venv/bin/python -m unittest tests/test_phase2_memory_and_midi.py -v` | Phase 2 : requête `< 25 ms` sur DB SQLite 5k entrées + fallback DSP, `audit_midi_harmony` en `3/4` / `6/8` / `free_ambient` sans basse, `autofix_midi` et `verify_midi_gate(auto_fix=True)`. |
| `tests/test_phase3_autofix_and_stems.py` | `.venv/bin/python -m unittest tests/test_phase3_autofix_and_stems.py -v` | Phase 3 : `analyze_stem_hierarchy`, `bootstrap_golden_refs` / `spectral_match_score`, boucle fermée `render_and_audit.py --auto-fix --max-passes 2` (`renders/autofix_history.json`). |
| `tests/test_phase4_expressivity.py` | `.venv/bin/python -m unittest tests/test_phase4_expressivity.py -v` | Phase 4 : `groove_lib.py` (`apply_micro_strum`, `apply_haas_pocket`, `apply_anti_machine_gun`, `enforce_frequency_inertia`) et `transitions_lib.py` (`beat4_dropout`, `filter_sweep`, `section_gain_arc`, `outro_fade`). |
| `tests/regression_multi_genre.py` | `.venv/bin/python tests/regression_multi_genre.py -v` | CI bout-en-bout : génère, valide (MIDI Gate), compile 100 % headless (`gl.*` natifs) et audite (`--no-neural`) les 7 `mix_profile`s (`trap_rap`, `pop_rnb`, `edm_club`, `rock_metal`, `lofi_chill`, `jazz_acoustic`, `classical_ambient` sans batterie) en `< 3 min` CPU. |
| `mcp-servers/audio-analyzer/test_musicality_metrics.py` | `PYTHONPATH=mcp-servers/audio-analyzer .venv/bin/python -m unittest test_musicality_metrics -v` | Métriques musicales de base, détection des frottements `b2` et `optimize_voice_leading`. |
