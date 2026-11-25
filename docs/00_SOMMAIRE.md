# Facette 0 — SOMMAIRE & Index Autonome du Framework AI-Studio
> **v3.0** : en session de production, seul [AGENT_RUNBOOK.md](AGENT_RUNBOOK.md) est chargé (3 macro-tours, < 1 500 tokens) ; l'archive détaillée est dans [REFERENCE_ARCHIVE.md](REFERENCE_ARCHIVE.md).
> **v3.2** : roadmap des 5 carences musicales majeures post-v3.1 dans [AUDIT_CARENCES_ET_SOLUTIONS_V3_2.md](AUDIT_CARENCES_ET_SOLUTIONS_V3_2.md) (NarrativeHarmony, ExpressionArticulations, FullCoverageCoreValidator, TimbralCohesion, MelodicAppealIndex + CI documentaire).


> **Objectif** : Point d'entrée unique pour qu'un agent IA autonome localise immédiatement **chaque dossier et fichier** du framework (racine `$AI_STUDIO_DIR`), connaisse ses **fonctions / outils MCP exacts**, et suive l'**ordre d'utilité exact dans le workflow** sans explorer à l'aveugle.

---

## 1. Ordre d'Utilité dans le Workflow (Pipeline « Loop-First » É0 → É4 + CI)

| Ordre | Étape Workflow | Scénario | Fichiers / Outils Principaux à Utiliser | Produit / Gate de Sortie |
| :---: | :--- | :---: | :--- | :--- |
| **0** | **Orientation & État** | Tous | `.rules` (symlink `AGENTS.md`, `GEMINI.md`), `docs/00_SOMMAIRE.md`, `scripts/loop_first_runner.py` (`pipeline_state.json`), puis la facette `docs/01..05` du tour courant | Route (`A`, `B`, `C`, `D`) + `pipeline_state.json` initialisé |
| **1** | **Environnement & Démarrage** | Tous | `scripts/launch-gridline.sh` (GUI `--attach`), `scripts/gridline-mcp.sh` (MCP), `audio-analyzer:check_pipewire_health` | Gridline / PipeWire prêts (`quantum >= 1024`) |
| **2** | **Conception & Blueprint** | A / B / C | `schemas/song_blueprint.schema.json`, `docs/02_*`, `docs/03_*` | `<projet>/song_blueprint.json` valide (`has_drums`, `mix_profile`, `emotion_vector`, `harmonic_core`) |
| **3** | **É0 : Mémoire Sémantique & Palettes** | A / B / C / D | **B/D** : `scripts/index_project_samples.py` (`audio-index:index_project_samples`), `scripts/audit_project_sources.py`, `scripts/sample_prep_lib.py` · **C** : `reverse-engineer` · **A/B** : `audio-index:propose_palettes` (`scripts/palette_builder.py`), `audio-index:search_samples` / `scripts/select_sample.py` (`timbre_card`, `semantic_anchors.npz`, 0 PyTorch), `audio-index:analyze_sample_dsp`, `scripts/gl_dsl.py` (`gl.*`), `scripts/fetch_samples_api.py` · **Chant** : `vocaloid` | `<projet>/samples_catalog.json` (B) + 3 palettes complémentaires + `<projet>/samples/` + `manifest.json` |
| **4** | **É1 : Sound Check Unitaire & Gate MIDI** | A / B / C | `audio-analyzer:audit_track_soundcheck` (`scripts/soundcheck_lib.py`, `scripts/listener_backend.py`), `scripts/export_palette_preview.py` (`palette_preview.wav`) · `musicality_metrics.py:optimize_voice_leading`, `scripts/song_builder.py` (`degree_to_midi`, `export_master_sketch`), `scripts/groove_lib.py`, `scripts/transitions_lib.py`, `scripts/studio_lib.py:verify_midi_gate(..., auto_fix=...)` / `scripts/midi_autofix.py` | Chaque piste à `PASS` en contexte joué (`2–4` mes.) + `<projet>/master_sketch.mid` validé (`gate_passed == true`) |
| **5** | **É2 : Noyau 4/8 mes. & Tests Verticaux** | A / B / C | `scripts/song_builder.py` (`--core-only`, `build_core_loop_project`, `ChannelSpec`), `scripts/gl_dsl.py`, `scripts/vertical_tester.py` (`audio-analyzer:audit_vertical_step` : `V1_LOW_END → V2_GROOVE → V3_HARMONY → V4_LEAD`, isolation `forbidden_tracks`) | `<projet>/core_loop.gridline` verrouillé (`core_loop_locked == true`, `1 sonorité = 1 piste`) |
| **6** | **É3/É4 : Dépliage, Coutures & Quality Gate** | A / B / C / D | `scripts/song_builder.py` (`build_gridline_project`), `scripts/horizontal_tester.py` (`audio-analyzer:audit_horizontal_seam`, fenêtres `4` mes. + `Outro`), `scripts/render_and_audit.py` (`--mp3 --sections --auto-fix --max-passes 2`), `audio-analyzer` (`render_project_stems`, `analyze_stem_hierarchy`, `audit_mix_deep`, `match_reference_audio`), `scripts/bootstrap_golden_refs.py` | Coutures + Outro `PASS` → `renders/master.wav` + `renders/autofix_history.json` (`quality_gate.passed == true`) |
| **7** | **CI & Non-Régression Framework** | Dev / CI | `tests/test_gl_dsl.py`, `tests/test_phase2_memory_and_midi.py`, `tests/test_phase3_autofix_and_stems.py`, `tests/test_phase4_expressivity.py`, `tests/test_phase5_loop_first.py`, `tests/regression_multi_genre.py`, `mcp-servers/audio-analyzer/test_musicality_metrics.py` | Tous tests unitaires (Phases 1–5) + CI multi-genres `7/7 PASS` |

---

## 2. Index des Fichiers Racine & Dossiers Config / Assets (racine du dépôt = `AI_STUDIO_DIR`, ex. `$AI_STUDIO_DIR/`)

| Chemin | Rôle & Fonctions | Ordre / Usage |
| :--- | :--- | :--- |
| `.rules` *(symlinks `AGENTS.md`, `GEMINI.md`)* | Règles normatives globales : routeur de scénario `A/B/C/D`, routage documentaire just-in-time, pipeline en 5 tours et 14 invariants non-négociables. | **0** (lu automatiquement par l'agent) |
| `README.md` | Présentation générale de l'environnement AI-Studio, installation rapide et prérequis Linux. | **0** (référence humaine) |
| `.env` / `.env.example` | Variables d'environnement API (`FREESOUND_API_KEY`, `LOTSOFSOUNDS_API_KEY`). | **3** (requis par `fetch_samples_api.py` / `fetch_samples_online`) |
| `.zed/settings.json` | Configuration éditeur Zed + déclaration des serveurs MCP locaux (`gridline`, `audio-index`, `audio-analyzer`, `reverse-engineer`, `vocaloid`). | **1** (config IDE/MCP) |
| `assets/empty_with_mixer_tracks.gridline` | Template `.gridline` minimal embarqué (Master + Insert 1..16) : repli automatique si le template système Gridline est absent (ordre : `$GRIDLINE_EMPTY_TEMPLATE` > `/usr/share/gridline/...` > `assets/`). | **2** (template nouveau projet) |
| `install.sh` | Installeur Fedora idempotent (paquets audio PipeWire, `.venv`, plugins CLAP/VST3, modèles, rescan). Flags `--check-only`, `--skip-plugins`, `--skip-clap`, `--skip-zed`, `--no-sudo`, `--force-rescan`. | **1** (setup initial machine) |
| `requirements.txt` | Dépendances Python du `.venv` (`mcp`, `numpy`, `scipy`, `soundfile`, `pyloudnorm`, `librosa`, `praat-parselmouth`, `essentia`, `mosqito`, `pyroomacoustics`, `matchering`, `music21`, `pretty_midi`, `onnxruntime`, `mido`, `transformers`, `audiobox_aesthetics`, `scikit-learn`). | **1** (setup `.venv`) |
| `.gitignore` | Règles d'exclusion Git (`.venv`, `.env`, gros média, bases temporaires). | **—** (Git) |
| `.venv/` | Environnement virtuel Python isolé. **Toujours** invoquer `$AI_STUDIO_DIR/.venv/bin/python`. | **1–7** |
| `schemas/song_blueprint.schema.json` | Schéma JSON Schema normatif validant tout `song_blueprint.json` (`mix_profile`, `has_drums`, `emotion_vector`, `bpm`, `time_signature`, `harmonic_core`, `motif_cell`, `structure`, `frequency_plan`, `tracks`, `vocals`). | **2** (`song_builder.load_blueprint(..., validate=True)`) |
| `samples` *(symlink → `soundbanks/`)* | Point d'entrée canonique vers les banques locales utilisées par les indexeurs (`sfz/` exclu automatiquement). | **3** |
| `soundbanks/` | Stockage réel des packs locaux WAV/OGG/MP3, boucles et sous-dossiers `.sfz` (exclus du scan mélodique). | **3** |
| `SAMPLES/` | Arborescence pré-classée par rôle (`808/`, `bass/`, `clap/`, `fx/`, `guitar/`, `hat/`, `kick/`, `loop/`, `open_hat/`, `perc/`, `piano/`, `snare/`, `synth/`, `synth-samples/`, `vocal/`). | **3** |
| `golden_references/` | Références golden par rôle (`bass_sub/`, `chord_keys/`, `hihat_perc/`, `kick/`, `lead_melody/`, `pad_atmosphere/`, `snare_clap/`), `README.md`, et **`spectral_corridors.json`** (versionné Git : corridors `[P10,P90]` Bark24 + 8 métriques des 7 profils, référence de gate v3.1+) ; `spectral_profiles.json` = format legacy du builder (empreintes Bark 24 bandes + pente spectrale). | **3 & 6** (`build_references.py`, `bootstrap_golden_refs.py`) |
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
| **`docs/AGENT_RUNBOOK.md`** | **Production (Tours 1–3)** — seul fichier lu en session | Runbook compact ($\le 1\,500$ tokens) pilotant les 3 macro-tours MCP (v3.3 : `--probe`, `ScoreIR`, `NarrativeHarmony`, articulations, tournoi `MelodicAppealIndex`). |
| **`docs/REFERENCE_ARCHIVE.md`** | **Dev / Débogage** (jamais lu en production) | Archive technique v3.0–v3.3, playbook des pièges et journal PQ. |
| **`docs/AUDIT_CARENCES_ET_SOLUTIONS_V3_2.md`** | **Roadmap / Amélioration continue** | Les 5 carences majeures post-v3.1 et leurs solutions, avec état d'implémentation P1–P5 (commits). |
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
| **1** | `scripts/verify_install.py` | `validate_active_blueprint`, `handshake_servers`, `main` | **Vérification finale post-install** (étape 8/8 d'`install.sh`) : valide `projects/active/song_blueprint.json` contre le schéma **si présent** (saut propre sinon — plus de crash `FileNotFoundError` sur installation fraîche), puis handshake JSON-RPC `initialize` + `tools/list` des 5 serveurs MCP. |
| **2** | `scripts/audit_project_sources.py` | `_analyze_808_harmonics`, `_analyze_long_loop`, `audit_gridline_sources`, `main` | **Audit pré-mix/pré-compo** d'un `.gridline` (et `--prepare-samples <dir>`) : repère `DISSONANCE_RISK`, `LOW_MID_WIDE_PHASE`, `NYQUIST_JUMP`, dérive d'accordage et `mixer_warnings`. |
| **3a** | `scripts/select_sample.py` | `_fast_comfort_score`, `_fast_role_score`, `_select_best_from_sqlite`, `select_best_from_db`, `main` | Sélection ultra-rapide (`< 25 ms`, 0 PyTorch) depuis `samples.db` + `sample_semantic_cards` + `semantic_anchors.npz` (`timbre_card`, `top_tags`, repli DSP). |
| **3b** | `scripts/palette_builder.py` | `propose_palettes`, `main` | **É0** : Génère 3 palettes complémentaires (`4–6` rôles, `< 50 ms`, 0 PyTorch) par scoring Bark24 + relais d'attaques + humeur ; verrou strict si `project_samples_dir` fourni. |
| **3c** | `scripts/index_project_samples.py` | `index_project_samples`, `main` | **É0 (Scénario B)** : Scanne exclusivement `<projet>/Samples/` (et dérivés `samples/`), écrit `samples_catalog.json` (`timbre_card`, `playable_chromatically`, `recommended_gl_fine_tune_ct`, recettes `sample_prep_lib.py` dont `slice_s`). |
| **3d** | `scripts/build_semantic_anchors.py` | `encode_anchors_with_clap`, `main` | Script offline (1 seule exécution) encodant `semantic_anchors.json` (72 ancres sur 6 axes) dans `semantic_anchors.npz` (`float16`) + `--populate-db` pour `sample_semantic_cards`. |
| **3e** | `scripts/audio_metrics.py` | `file_md5`, `load_and_normalize_audio`, `check_hard_rejects`, `analyze_pitch_and_purity`, `analyze_vocal_signature`, `extract_dsp_metrics`, `compute_comfort_score`, `compute_role_score`, `ClapEmbedder`, `compute_masking_penalty`, `compute_pitch_shift`, `score_candidate`, `analyze_single_file` | Bibliothèque DSP déterministe de référence (`praat-parselmouth`, `librosa`, `essentia`, phase M/S, Bark 24 bandes, HPSS rapide) + encodeur CLAP optionnel. |
| **3f** | `scripts/fetch_samples_api.py` | `load_studio_env`, `normalize_to_wav24`, `search_freesound`, `search_lotsofsounds`, `fetch_and_filter_samples`, `main` | Téléchargement CC0 depuis Freesound v2 / Lots of Sounds (`.env`), normalisation WAV 24-bit `44.1 kHz` et filtrage `mono_oneshot` / DSP (`--no-clap`). |
| **3g** | `scripts/index_samples.py` | `index_directory_psychoacoustic`, `main` | Alimente `mcp-servers/audio-index/samples.db` (`samples`, `sample_dsp_metrics`, `sample_semantic_cards`) ; supporte `--no-clap`, `--with-clap`, `--enrich-clap`, `--populate-cards`. |
| **3h** | `scripts/sample_prep_lib.py` | `read_stereo`, `write_tmp_clean`, `apply_eq_peak`, `apply_soft_saturation`, `varispeed_resample`, `pitch_shift_stereo`, `time_stretch_stereo`, `trim_leading_silence`, `measure_fft_tuning`, `build_sample_from_spec`, `audit_samples_manifest`, `process_sample_specs` | Pipeline déclaratif centralisé de préparation de samples : découpe, EQ, pitch/varispeed, saturation, time-stretch, stabilisation phase grave, export 24-bit et calcul exact `recommended_gl_fine_tune_ct` dans `manifest.json`. |
| **4a** | `scripts/listener_backend.py` | `ListenerBackend`, `DeterministicSemanticListener`, `MultimodalAudioListener`, `get_listener_backend` | **É1** : Couche d'écoute pluggable (défaut 0 GPU déterministe ou backend audio multimodal optionnel) produisant un verdict uniforme `PASS/FIX/SWAP` + diagnostic 1–3 phrases + `next_action`. |
| **4b** | `scripts/soundcheck_lib.py` | `audit_soundcheck_wav`, `audit_track_soundcheck`, `main` | **É1** : Rend et audite chaque piste isolée jouée sur `2–4` mes. dans sa vraie tessiture/BPM (audibilité, justesse `<= 8 ct`, intégrité chromatique, propreté `2–5 kHz`, enveloppe, fidélité sémantique) ; nettoie `.tmp_soundcheck/`. |
| **4c** | `scripts/export_palette_preview.py` | `export_palette_preview`, `main` | **É1 (optionnel)** : Assemble `2.5 s` de chaque piste validée (séparées par `0.4 s`) dans `renders/palette_preview.wav` + index texte (`--human-gate`). |
| **4d** | `scripts/groove_lib.py` | `ms_to_beats`, `apply_micro_strum`, `apply_haas_pocket`, `apply_anti_machine_gun`, `enforce_frequency_inertia` | Humanisation déterministe des listes de notes en beats : micro-strum accords, décalage Haas Kick/Snare/Hat/Basse, modulation vélo/longueur anti-mitraillette, durée mini sub `< 100 Hz`. |
| **4e** | `scripts/transitions_lib.py` | `beat4_dropout`, `filter_sweep`, `section_gain_arc`, `outro_fade`, `generate_drum_fill`, `add_transition_fx`, `apply_harmonic_anticipation` | Transition et macro-dynamique automatisées : drop-out beat 4, balayage LPF, arche de gain par section, fondu de sortie, **fills de batterie idiomatiques**, **riser + impact filtré** (`FX Hall 22`) et **anticipation harmonique** d'une croche sans chevauchement. |
| **4f** | `scripts/midi_autofix.py` | `autofix_midi`, `main` | Correction déterministe et ré-audit d'un `.mid` : quantifie notes hors gamme, corrige bornes sub/apex, résout collisions `< MIDI 48`, projette les temps forts hors-accord, élimine frottements `b2`, avec rapport JSON (`fix_report`). |
| **4h** | `scripts/gain_staging_lib.py` | `measure_active_loudness`, `normalize_role_for_gain_staging`, `ROLE_TARGET_OFFSETS_DB`, `compute_gain_staging_plan`, `resolve_mix_profile` | **Phase 8 (`Q1`)** : sonie active K-weighted (BS.1770) sur trames actives, 9 rôles canoniques, cibles relatives par `mix_profile`, trim borné + peak guard, faders recommandés (`db_to_auto`). |
| **5a** | `scripts/gl_dsl.py` | `clamp`, `q_to_width`, `eq_band`, `to_mixer_params`, `eq_params`, `reverb_room`, `limiter_master`, `sidechain_send`, `mixer_gain_to_auto`, `db_to_auto`, `eq_gain_to_auto`, `SynthSpec`, `SynthVoiceConfig`, `synth_voice` (`SYNTH_REGISTRY` pour 13 synthés `gl.*`) | **DSL sémantique anti-erreur Gridline `0.7.7`** : traduit automatiquement Q→`Band N width`, types EQ, `gl.reverb` `k128`/`ds`/`hHz`, `gl.limiter`, sends `sidechain`, courbes d'automation dB & EQ `[-18, +18]` dB, LPF & ADSR par rôle/valence sur 13 synthés natifs. |
| **5b** | `scripts/studio_lib.py` | `GridlineMCP`, `clean_audio_sample`, `stabilize_stereo_low_mid`, `extract_clean_oneshot`, `check_pattern_audibility`, `verify_midi_gate`, `setup_standard_buses` | Socle bas niveau : client JSON-RPC `GridlineMCP`, nettoyage/isolation audio, gate MIDI (avec support `time_signature`, `phrase_plan`, `auto_fix`), topologie des bus `17..22` (`FX Delay 21`, `FX Hall 22`) + `Master 0`, wideners mid/high et garde-fou LPF. |
| **5c** | `scripts/song_builder.py` | `SCALES`, `degree_to_midi`, `load_blueprint`, `load_arrangement_plan`, `compile_arrangement_plan`, `ChannelSpec` (`delay_send`, `hall_send`), `split_notes_by_pitch_zones`, `eq_band`, `eq_insert`, `eq_music_pocket`, `eq_kick_clean`, `export_master_sketch`, `select_core_section`, `build_core_loop_project`, `build_gridline_project`, `run_song_cli` | **Moteur central de build `.gridline`** : valide blueprint et `arrangement_plan.json`, applique `phrasing_engine` (`P2`), les transitions riches (`Q5`), la séparation de registre sombre (`V < -0.2`) et `vocal_arrangement_sync` (`P6`), exporte `master_sketch.mid`, supporte `--core-only` et le build complet atomique. |
| **5e** | `scripts/phrasing_engine.py` | `apply_instrumental_prosody`, `apply_phrase_velocity_arc`, `apply_cadential_ornaments`, `apply_section_pocket` | **Phase 7 (`P2`)** : arc de dynamique et automations LPF sur 4 mesures, approche basse/appoggiatures cadentielles (avec garde `detect_minor_2nd_clashes`), micro-pocket inter-sections (`+3.5 ms` couplets / `-2.0 ms` refrains). |
| **5f** | `scripts/vocal_arrangement_sync.py` | `extract_vocal_phrases`, `mute_counter_during_vocals`, `build_formant_ducking_automation`, `formant_ducking_eq_params` | **Phase 7 (`P6`)** : extrait phrases chantées (`.ustx` ou `vocals.wav`), libère le chant dans les fenêtres de respiration (`>= 0.5 beat`) et place le creux formantique `1.85 kHz` (`0 -> -3.5 dB`, `40/120 ms`) sur `BUS Music`. |
| **5g** | `scripts/taste_memory.py` | `record_validated_taste`, `query_taste_prior` | **Phase 7 (`P4`)** : mémoire épisodique `memory/user_taste_profile.json` (max 50 entrées, dédup `project_id`, poids `2.0` sur `--approve-taste`) + requête pondérée `(V, T, E, D)` en `0` token. |
| **5d** | `scripts/vertical_tester.py` | `classify_channel_stage`, `run_vertical_stage`, `run_all_vertical_stages`, `main` | **É2** : Orchestre les paliers `V1_LOW_END → V2_GROOVE → V3_HARMONY → V4_LEAD` par mute/unmute sur `core_loop.gridline`, retourne `target_layer_tracks` / `forbidden_tracks` et verrouille `core_loop_locked`. |
| **6a** | `scripts/horizontal_tester.py` | `audit_seam_wav`, `audit_outro_wav`, `extract_sections_from_blueprint`, `audit_project_horizontal_seams`, `main` | **É3** : Rend uniquement les fenêtres de couture `4` mes. (`±2` mes.) autour des transitions + `Outro` : audite `ΔLUFS`, `beat4_dropout`/`filter_sweep`, `0` glitch `±150 ms` et extinction `< -45 dBFS`. |
| **6b** | `scripts/loop_first_runner.py` | `LoopFirstStateMachine`, `detect_user_samples_locked`, `main` | **Orchestrateur d'états** (`<projet>/pipeline_state.json`) : pilote `E0 → E4`, verrouille `user_samples_locked`, bloque toute édition sur `forbidden_tracks` et escalade automatiquement vers `SWAP_SAMPLE` après 2 échecs. |
| **6c** | `scripts/bootstrap_golden_refs.py` | `compute_bark24_distribution`, `build_spectral_profiles`, `compute_spectral_match`, `main` | Construit `golden_references/spectral_profiles.json` (legacy) et `golden_references/spectral_corridors.json` (versionné, corridors `[P10,P90]`) depuis `golden_references/external_masters/<profil>/` (provenance `external_pro_reference`, audio jamais versionné) ou les rendus internes (`internal_bootstrap`) et calcule `spectral_match_score` (0 PyTorch). |
| **6d** | `scripts/render_and_audit.py` | `parse_sections`, `render_and_evaluate`, `main` | Pipeline fermé Rendu + Audit + Quality Gate multi-genres (`--profile`, `--range`, `--no-neural`, `--no-drums`, `--blueprint`, `--sections`, `--mp3`) avec boucle **`--auto-fix --max-passes N`** écrivant `renders/autofix_history.json`. |
| **7** | `scripts/build_references.py` | `build_golden_references`, `main` | Script CLI optionnel encodant les fichiers de `golden_references/<role>/` dans `mcp-servers/audio-index/golden_embeddings.npz` (CLAP CPU). |


### 4b. Modules v3.0/v3.1/v3.2 (index complémentaire)

| # | Script | Fonctions clés | Rôle |
| :--- | :--- | :--- | :--- |
| **v3.x** | `scripts/narrative_harmony.py` | `apply_narrative_harmony`, `motif_answer`, `rhythmic_lift`, `call_and_response`, `melodic_climax_arch`, `mark_suspensions` | v3.2 NarrativeHarmony : progressions par section (`section_progressions`), phrasé 8 mesures Antécédent/Conséquent, opérateurs motiviques déterministes, suspensions 4→3/9→8/b6→5 vérifiées. |
| **v3.x** | `scripts/midi_groove_bank.py` | `build_groove_bank`, `query_groove_bank`, `groove_to_notes` | v3.2 banque de grooves : mine les `.mid` percussifs locaux (soundbanks) en templates 1-4 mesures + vélocité, sans jamais copier de mélodie. |
| **v3.x** | `scripts/full_coverage_validator.py` | `validate_full_score_ir`, `build_probe_plan`, `run_full_coverage_check` | v3.2 pré-vol symbolique 100 % du morceau (clones 8 mesures, charge cognitive 1-8/mesure, arc d'énergie, outro) + sonde stems (8 noyau + tampon Hall dynamique ≥ 2 mes. v3.3 + 4 hors-noyau, automations remappées). |
| **v3.x** | `scripts/dsp_interaction_guards.py` | `compute_probe_gap_bars`, `remap_automations_for_probe`, `plan_safe_ritardando`, `extract_steady_state_masks` | v3.3 #3 garde-fous DSP : tampon silence sonde selon decay Hall/BPM, remap automations `--probe`, ritardando sûr (symbolique si boucles audio, sinon tempo_ramp + delay_mute 1 beat avant), masques de fenêtres glide/vibrato pour l'audit de justesse. |
| **v3.x** | `scripts/blueprint_enricher.py` | `enrich_blueprint`, `write_enrichment_report`, `inject_climax_glides` | v3.3 #1 enricher d'intention créative (Tour 2, 0 GPU, déterministe, non intrusif) : complète seed / section_progressions / phrase_length_bars / glides 808 Climax / groove_preference / archétypes gl.* omis + journal `renders/diagnostics/enrichment_report.json`. |
| **v3.x** | `scripts/determinism.py` | `derive_seed`, `resolve_seed`, `seed_all` | v3.3 #5 déterminisme strict : `seed` blueprint (défaut SHA-256 title\|key\|bpm), RNG dédiés propagés à groove_lib/phrasing_engine/narrative_harmony/compile_song. |
| **v3.x** | `scripts/orchestration_engine.py` | `apply_bass_motion`, `apply_comping_to_score_ir`, `comping_pattern`, `default_comping_arc`, `revoice_chords_with_lead` | v3.3 #2 : bass_mode (stepwise_inversions/pédales tonique-dominante, bass_midi >= 31, LIL), 6 `comping_style` par section (schéma plan), voice-leading conscient du Lead (`optimize_voice_leading_with_lead`). Câblé dans `compile_arrangement_plan` (`post_groove_check.orchestration`). |
| **v3.x** | `scripts/preview_lib.py` | `build_preview_highlights` | v3.3 #5 : bande d'écoute rapide `renders/preview_highlights.wav` (24 s = 3×8 s Couplet/Refrain/Pont-Outro, crossfade 150 ms) après `studio_refine_track` ou le Tour 3. |
| **v3.x** | `scripts/asset_hygiene.py` | `asset_hygiene` | Hygiène banques : whitelist extensions audio, magic bytes, parasites OS (`__MACOSX`, `.DS_Store`, `.wav.zpa`, `.fst`), fichiers < 1 Ko. Pas de détection de licence (responsabilité utilisateur). |
| **v3.x** | `scripts/autofix_transaction.py` | `autofix_transaction` | Autofix transactionnel 3 paliers : rollback immédiat si invariant Palier 1 régresse, blacklist anti-oscillation, journal `renders/autofix_history.json`. |
| **v3.x** | `scripts/blueprint_safety.py` | `blueprint_safety` | Audit musical projet-wide du blueprint : verrou octave exact `root_note == midi_root`, plafonds `S_max`, key-fit/BPM boucles, confinement cloches {1,5,8} ; bloque le rendu en `severity=error`. |
| **v3.x** | `scripts/compile_song.py` | `compile_song` | Compilateur 100 % déclaratif (`--core-only` / `--probe` / complet) appliquant l'ordre canonique MIDI -> ScoreIR -> clamp frontière 20 ms ; tournoi de variantes `MelodicAppealIndex` en Tour 2 (v3.2). |
| **v3.x** | `scripts/mix_config.py` | `mix_config` | Chargeur unique de `config/mix_profiles.json` (100 % des seuils mix/gate : `get_profile_config`, `tuning_tolerance`, `notch_budget`, `stem_hierarchy_bounds`, `autofix_step_bounds`). |
| **v3.x** | `scripts/score_ir.py` | `score_ir` | ScoreIR (v3.1) + v3.2 articulations (`articulation`, `glide_ms`, `vibrato_delay_beats`) : groove non destructif (`expression`), clamp de frontière harmonique 20 ms, suspensions `4->3`/`9->8` résolues. |
| **v3.x** | `scripts/stem_matrix_evaluator.py` | `stem_matrix_evaluator` | Évaluation NumPy V1..V4 d'un rendu stems + `evaluate_barwise_harmonic_alignment` (chroma-vs-grille par segment, contrat §4.3) + solveur de gains. |

---

## 5. Index des Serveurs MCP (`mcp-servers/`) & Outils Associés

### 5.1 Serveur `gridline` *(Binaire Gridline `0.7.7`, lancé via `scripts/gridline-mcp.sh`)*
- **Lecture / Inspection** : `get_song`, `list_channels`, `list_patterns`, `get_pattern`, `get_playlist`, `get_mixer`, `get_parameters`, `get_automation`, `get_markers`, `list_plugin_catalog`, `list_presets`, `get_history`, `list_swing_presets`, `find`, `inspect`, `get_values`, `list_commands`, `windows`, `get_settings`.
- **Édition Atomique / Construction** : `set_song`, `add_channel`, `edit_channel`, `remove_channel`, `add_pattern`, `edit_pattern`, `remove_pattern`, `set_notes`, `set_steps`, `edit_notes`, `add_clip`, `edit_clip`, `remove_clip`, `add_automation`, `edit_automation`, `remove_automation`, `add_plugin`, `remove_plugin`, `set_parameter`, `set_parameters`, `edit_mixer_track`, `add_mixer_track`, `remove_mixer_track`, `set_send`, `stage_room`, `arrangement`, `add_marker`, `remove_marker`, `patch`, `batch`, `duplicate`, `select`, `run_script` (Lua), `run_command`, `transport`, `set_setting`, `undo`, `redo`.
- **Analyse & Export** : `analyze_loudness`, `render_audio`, `export_project`, `get_job`, `cancel_job`, `save_project`.

### 5.2 Serveur `audio-index` (`mcp-servers/audio-index/`)
| Fichier | Fonctions / Outils `@mcp.tool()` | Usage & Latence |
| :--- | :--- | :--- |
| `audio_index_server.py` | **`search_samples`**, **`propose_palettes`**, **`index_project_samples`**, **`find_compatible_samples`**, **`get_sample_info`**, **`list_sample_packs`**, **`search_synth_presets`**, **`analyze_sample_dsp`** | **Voie rapide (`< 50 ms`, 0 PyTorch)** sur `samples.db` + `sample_semantic_cards` (`timbre_card`) + `semantic_anchors.npz` + palettes complémentaires. |
| `audio_index_server.py` | **`evaluate_sample_quality`**, **`select_best_samples`**, **`fetch_samples_online`**, **`reindex_samples`**, **`reindex_synth_presets`** | Indexation/évaluation complètes (désactiver CLAP avec `with_clap=False` en session temps réel si besoin). |
| `indexer.py` | `init_db`, `analyze_sample`, `populate_dsp_metrics`, `upsert_dsp_metrics`, `load_semantic_anchors`, `build_timbre_card_from_dsp`, `build_semantic_card_from_clap`, `populate_semantic_cards`, `enrich_clap_embeddings`, `index_directory` | Indexeur unifié alimentant `samples`, `sample_dsp_metrics` et `sample_semantic_cards` dans `samples.db` (exclut `sfz/`). |
| `semantic_anchors.json` / `.npz` | Dictionnaire + vecteurs `float16[72, 512]` | 72 ancres sémantiques pré-encodées sur 6 axes (0 PyTorch au runtime). |
| `preset_indexer.py` | `init_presets_table`, `index_gridline_presets` | Indexe les presets natifs `gl.*` dans la table SQLite `synth_presets`. |
| `samples.db` | Base SQLite unifiée | Stockage métadonnées + métriques DSP rapides + `sample_semantic_cards` (`timbre_card`, `clap_vec` `float16`). |
| `golden_embeddings.npz` | Archive NumPy | Embeddings CLAP des références par rôle (si générés par `scripts/build_references.py`). |

### 5.3 Serveur `audio-analyzer` (`mcp-servers/audio-analyzer/`)
| Fichier | Fonctions / Outils `@mcp.tool()` | Usage & Rôle |
| :--- | :--- | :--- |
| `audio_analyzer_server.py` | **`audit_track_soundcheck`**, **`audit_vertical_step`**, **`audit_horizontal_seam`**, **`analyze_mix`**, **`analyze_low_end_masking`**, **`compare_renders`**, **`audit_mix_deep`**, **`audit_midi_harmony`**, **`match_reference_audio`**, **`render_project_stems`**, **`analyze_stem_hierarchy`**, **`check_pipewire_health`** | Audits Loop-First (`E1` Sound Check unitaire, `E2` Vertical `V1..V4`, `E3` Coutures/Outro), audits DSP mix, masquage kick/basse, MIDI Gate multi-métriques, stems & hiérarchie, santé PipeWire. |
| `deep_diagnostics.py` | `_third_octave_profile`, `_analyze_transients_and_roominess`, `_scan_glitches_and_crackle`, `_compute_psychoacoustics`, `_simulate_headphone_translation`, `run_deep_audit`, `resolve_allowed_pcs`, `audit_midi_file`, `match_with_reference`, `render_stems_and_audit`, `compute_pink_slope_db_per_oct`, `analyze_stem_hierarchy`, `analyze_vertical_stage`, `inspect_pipewire` | Moteur analytique sous-jacent (tiers d'octave, résonances, glitches, Loudness/Sharpness/Roughness, simulation casques `targets/Sony_WH-1000XM*.csv`, CLAP/Audiobox optionnels, hiérarchie stems, paliers verticaux `V1..V4` + conflit chroma CQT). |
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
| `tests/test_gl_dsl.py` | `.venv/bin/python -m unittest tests/test_gl_dsl.py -v` | Phase 1 : `q_to_width`, `eq_band`, `reverb_room`, `limiter_master`, `sidechain_send`, `db_to_auto`, 13 synthés `SYNTH_REGISTRY`, compatibilité `song_builder.py`. |
| `tests/test_phase2_memory_and_midi.py` | `.venv/bin/python -m unittest tests/test_phase2_memory_and_midi.py -v` | Phase 2 : requête `< 25 ms` sur DB SQLite 5k entrées + fallback DSP, `audit_midi_harmony` en `3/4` / `6/8` / `free_ambient` sans basse, `autofix_midi` et `verify_midi_gate(auto_fix=True)`. |
| `tests/test_phase3_autofix_and_stems.py` | `.venv/bin/python -m unittest tests/test_phase3_autofix_and_stems.py -v` | Phase 3 : `analyze_stem_hierarchy`, `bootstrap_golden_refs` / `spectral_match_score`, boucle fermée `render_and_audit.py --auto-fix --max-passes 2` (`renders/autofix_history.json`). |
| `tests/test_phase4_expressivity.py` | `.venv/bin/python -m unittest tests/test_phase4_expressivity.py -v` | Phase 4 : `groove_lib.py` (`apply_micro_strum`, `apply_haas_pocket`, `apply_anti_machine_gun`, `enforce_frequency_inertia`) et `transitions_lib.py` (`beat4_dropout`, `filter_sweep`, `section_gain_arc`, `outro_fade`). |
| `tests/test_phase5_loop_first.py` | `.venv/bin/python -m unittest tests/test_phase5_loop_first.py -v` | Phase 5 : Pipeline Loop-First complet — `semantic_anchors.npz` + `timbre_card` + `propose_palettes` (`< 50 ms`, 0 PyTorch, exclusivité `Samples/`), `audit_track_soundcheck` + `palette_preview.wav` + `ListenerBackend`, `vertical_tester` (`V1..V4` headless `trap_rap` & `classical_ambient`), `audit_horizontal_seam` et garde-fous `loop_first_runner.py`. |
| `tests/test_phase8_balance_space_expressivity.py` | `.venv/bin/python -m unittest tests/test_phase8_balance_space_expressivity.py -v` | Phase 8 `Q1..Q7` : sonie active & gain staging, matrice stems bornée + `ADJUST_TRACK_FADER`, bus `17..22` (`Dry=0`), audits `stereo_balance_db` / `mid_high_side_to_mid_db`, références externes & garde `taste_memory`, fills/risers/anticipation, registres sombres, `synth_voice(expressive=True)` validé contre `gridline plugin-params`, budget CPU Phase 8 `< 1.5 s` et pyramide `<= 1500` chars. |
| `tests/test_quality_improvements.py` | `.venv/bin/python -m unittest tests/test_quality_improvements.py -v` | Régressions qualité/portabilité : classification par tokens, filtrage BPM des boucles, propagation `root_note`, résonances notchables, swing/humanisation/pocket, template embarqué, pin `essentia`, verrou `auto_fetch`. |
| `tests/regression_multi_genre.py` | `.venv/bin/python tests/regression_multi_genre.py -v` | CI bout-en-bout : génère, valide (MIDI Gate), compile 100 % headless (`gl.*` natifs) et audite (`--no-neural`) les 7 `mix_profile`s (`trap_rap`, `pop_rnb`, `edm_club`, `rock_metal`, `lofi_chill`, `jazz_acoustic`, `classical_ambient` sans batterie) en `< 3 min` CPU. |
| `mcp-servers/audio-analyzer/test_musicality_metrics.py` | `PYTHONPATH=mcp-servers/audio-analyzer .venv/bin/python -m unittest test_musicality_metrics -v` | Métriques musicales de base, détection des frottements `b2` et `optimize_voice_leading`. |


---

## Phase 6 — Synchronisation Multi-Échelles ($H \times V$) & Sobriété Tokens

| Élément | Fichier / Outil | Rôle exact |
| ----- | ----- | ----- |
| **Contrat Top-Down** | `schemas/multi_scale_contract.schema.json`, `scripts/target_fingerprint.py`, MCP `extract_target_fingerprint` | Compile `song_target_profile.json` + `multi_scale_contract.json` ($S5 \rightarrow S1$) depuis `0..3` références audio (0 GPU) ou repli `spectral_profiles.json` + `emotion_vector`. |
| **Cache SHA-256** | `scripts/artifact_cache.py`, `scripts/loop_first_runner.py` | Stocke `artifact_hashes` (É1 par piste, É2 par palier `V1..V4`, É3 par couture) + `contract_path` + `multiscale_scores` dans `pipeline_state.json`. |
| **Acoustique S1/S2 & Syntaxe S3** | `deep_diagnostics.py`, `musicality_metrics.py`, `scripts/midi_autofix.py`, `scripts/sample_prep_lib.py` | Phase Kick/808 (`INVERT_808_POLARITY`, `SHIFT_BASS_MICRO_MS`), flux spectral soutenu, alignement pocket audio (`TRIM_SAMPLE_START_MS`), collisions transitoires 1–4 kHz (`STAGGER_ATTACK_OR_STRUM`), cohésion motivique A/A'/B, tension harmonique et interlocking (`fix_motif_clone`, `fix_chords_lead_overlap`). |
| **Macro-Structure S4 & Catalogue S5/H** | `scripts/macro_structure_audit.py`, `scripts/catalog_coherence.py` | SSM `Chroma(0.35)+MFCC(0.40)+Rythme(0.25)` (détection clones 8 mes. `> 0.94`), nouveauté Checkerboard (`±1` mes.), Arc 5D (compatible `has_drums=false`), plafond cognitif par mesure (`-28 dBFS`), cohérence EP/Album/DJ Set via `master_fingerprint.json` + roue de Camelot. |
| **Pyramide & Auto-Fix Local** | `scripts/multiscale_pyramid.py`, `scripts/multiscale_autofix.py` | Agrégation Bottom-Up $S1 \rightarrow S5$ avec blocage en cascade, top-3 déviations strictement borné à `<= 1 500` caractères (`<= 400` tokens), diagnostic complet écrit sur disque (`renders/diagnostics/`), correction mécanique sans LLM (max 2 passes). |
| **Orchestrateur CLI & MCP** | `scripts/run_autonomous_pipeline.py`, MCP `audio-analyzer:audit_multiscale_hv` | Point d'entrée unique (`--stage E1\|E2\|E3\|E4\|ALL --auto-fix --checkpoint-e2 --resume-from-checkpoint --json-compact`) et outil MCP agrégé respectant la limite `<= 400` tokens. |
| **Tests Phase 6** | `.venv/bin/python -m unittest tests/test_phase6_multiscale_hv.py -v` | Suite déterministe couvrant $S1..S5$, cache hit (`< 5 ms`), blocage en cascade, budget `<= 1 500` chars, auto-fix E2E et benchmark CPU `< 4 s` sur 2 minutes. |

---

## Phase 7 — Expressivité, Psychoacoustique & Autonomie Paramétrique (`P1..P6`)

| Chantier | Fichiers / Schémas | Rôle exact |
| :---: | :--- | :--- |
| **`P1` Plan Déclaratif** | `schemas/arrangement_plan.schema.json`, `scripts/song_builder.py` (`compile_arrangement_plan`), `scripts/multiscale_autofix.py` | Sépare composition JSON et compilation déterministe : `PRUNE_TRACK_IN_SECTION`, `REINFORCE_SECTION_TRANSITION`, `MUTATE_BLOCK_DENSITY_OR_LPF` mutent `arrangement_plan.json` et recompilent localement en `0` token. |
| **`P2` Prosodie Instrumentale** | `scripts/phrasing_engine.py` | Arc dynamique 4 mes. (vélocité + LPF automatisé), approche basse & appoggiature cadentielle sûre (`b2`), micro-pocket inter-sections (`+3.5 ms` / `-2.0 ms`). |
| **`P3` Psychoacoustique** | `mcp-servers/audio-analyzer/deep_diagnostics.py`, `scripts/render_and_audit.py` | Masquage dynamique Bark (`APPLY_DYNAMIC_MIRROR_EQ`), `LRA` EBU R128 mono-passe (`compute_ebu_r128_stats`, `RELAX_BUS_COMPRESSION`), traduction harmonique sub `< 65 Hz` (`BOOST_BASS_HARMONICS`). |
| **`P4` Mémoire de Goût** | `scripts/taste_memory.py`, `memory/user_taste_profile.json`, `scripts/target_fingerprint.py` | Capitalise les mixes `PASS` (poids `2.0` sur `--approve-taste`) et injecte `"source": "user_taste_memory"` en priorité si `reference_tracks` est vide. |
| **`P5` Checkpoint É2 15 s** | `scripts/loop_first_runner.py`, `scripts/run_autonomous_pipeline.py` | Pause optionnelle `--checkpoint-e2` (`AWAITING_E2_CHECKPOINT`, `renders/e2_checkpoint_15s.wav` à `-14 LUFS`, fiche 5 lignes `<= 300` tokens) et reprise `--resume-from-checkpoint` sans recalcul É1/É2. |
| **`P6` Synchro Vocale** | `scripts/vocal_arrangement_sync.py` | Détection phrases `.ustx`/audio, silence des contre-chants hors fenêtres `>= 0.5 beat`, automation ducking formantique `1.85 kHz` (`Band 5 gain` `0 -> -3.5 dB`) et code `SYNC_VOCAL_ARRANGEMENT`. |
| **Tests Phase 7** | `.venv/bin/python -m unittest tests/test_phase7_expressivity_autonomy.py -v` | Valide `P1..P6` bout-en-bout, budgets tokens/CPU (`<= 2.5 s` P1..P6 sur 2 min) et auto-fix déterministe sans LLM. |

---

## Phase 8 — Équilibre Inter-Pistes, Espace 3D, Réalisme Timbral & Finitions (`Q1..Q7`)

| Chantier | Fichiers / Outils | Rôle exact |
| :---: | :--- | :--- |
| **`Q1` Gain Staging** | `scripts/gain_staging_lib.py`, `scripts/soundcheck_lib.py`, `scripts/song_builder.py` | Sonie **active** K-weighted par rôle (`kick`…`vocals`), table `ROLE_TARGET_OFFSETS_DB` par `mix_profile`, peak guard `-2 dBFS`, `recommended_volume_auto` appliqué aux faders + `renders/gain_staging_report.json`. |
| **`Q2` Hiérarchie Stems** | `deep_diagnostics.analyze_stem_hierarchy` v2, `macro_structure_audit.audit_section_track_audibility`, `multiscale_autofix._fix_adjust_track_fader`, `render_and_audit.py` | Matrice bilatérale `lead_vs_harmony / bass_vs_kick / snare_vs_kick / hihat_vs_kick / counter_vs_lead / vocals_vs_lead`, plancher d'audibilité par section (`-13/-18/-22 dB`), correction `ADJUST_TRACK_FADER` bornée `±4 dB/passe` (cumul `8 dB`, `forbidden_tracks` respectés). |
| **`Q3` Espace 3D** | `studio_lib.setup_standard_buses`, `gl_dsl.delay_send_fx` / `reverb_hall`, `ChannelSpec.delay_send/hall_send`, `deep_diagnostics.analyze_stereo_image` | Bus `FX Delay (21)` et `FX Hall (22)` (`Dry=0`, bandes limitées), sends par rôle, `stereo_balance_db <= 1.2 dB`, largeur `mid_high_side_to_mid_db` (`[-14,-1.5]` / `[-18,-1.5]` acoustique), `FIX_STEREO_BALANCE` / `WIDEN_HARMONY_OR_PERC` / `NARROW_EXCESSIVE_SIDE`, LPF `BUS Music >= 11.5 kHz` en sombre. |
| **`Q4` Références Externes** | `scripts/bootstrap_golden_refs.py --external-dir`, `golden_references/external_masters/README.md`, `scripts/taste_memory.py`, `scripts/target_fingerprint.py` | Casse l'auto-référence : ancres commerciales `external_pro_reference` (Bark24, pente rose, LUFS, LRA, crest, largeur, ratios de bandes), `record_validated_taste` refuse `PQ < 7.8` sans `--approve-taste`, fusion bornée `30 %/60 %` max de mémoire de goût. |
| **`Q5` Transitions Riches** | `scripts/transitions_lib.py`, `schemas/arrangement_plan.schema.json`, `scripts/song_builder.py`, `musicality_metrics.py` | `generate_drum_fill`, `add_transition_fx` (riser `300 Hz -> 9 kHz` + impact), `apply_harmonic_anticipation`, registres sombres `Chords 48-60` / `Lead 60-72` (`gap >= 5 st`), 2 sauts expressifs/8 mes. tolérés, anticipations évaluées sur l'accord suivant. |
| **`Q6` Timbres Vivants** | `gl_dsl.synth_voice(..., expressive=True)`, `sample_prep_lib.build_multizone_samples`, `song_builder.split_notes_by_pitch_zones` | LFO filtre/vélocité par rôle sur paramètres `gl.*` vérifiés (`gridline plugin-params`), `makeup_gain_db` LPF, sampler 3 zones pré-transposées (`±4 st` max par zone, même insert mixeur). |
| **`Q7` Orchestration** | `scripts/artifact_cache.py`, `scripts/multiscale_pyramid.py`, `tests/test_phase8_balance_space_expressivity.py` | Hashes E1/E2 incluant `gain_staging_overrides`, `delay_send`, `hall_send`, `expressive` ; nouveaux codes dans le Top-3 ; suite `Q1..Q7` et budget CPU Phase 8 `< 1.5 s`. |
