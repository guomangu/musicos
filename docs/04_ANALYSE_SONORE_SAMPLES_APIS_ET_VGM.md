# Facette 4 — Mémoire Sonore `samples.db`, DSP Déterministe, APIs, Remix & Chant

> **Cheatsheet IA (L'essentiel v2.1 — Loop-First É0)**
> - **Samples fournis initialement** : Si `Samples/` ou un `.gridline` initial existe, lancer **`scripts/index_project_samples.py <proj>`** (MCP `audio-index:index_project_samples`) → `<proj>/samples_catalog.json` (`user_samples_locked = true`, **interdiction d'ajouter de nouveaux samples** externes/APIs ; autorisé de créer des dérivés via `sample_prep_lib.py` dans `samples/`).
> - **Mémoire Sémantique Offline & Palettes (`0 PyTorch`)** : `samples.db` intègre `sample_semantic_cards` (`timbre_card` 1 ligne `<= 140 chars` + `clap_vec` `float16[512]`) projetés contre les 72 ancres pré-encodées `semantic_anchors.npz` (`scripts/build_semantic_anchors.py`). Appeler **`audio-index:propose_palettes`** (`scripts/palette_builder.py`, `< 50 ms`) pour obtenir **3 palettes complémentaires** (Bark24 + relais d'attaques + humeur) au lieu de fouiller 5 000+ fichiers.
> - **Ordre économique** : (1) `index_project_samples.py` (si `Samples/`) → (2) `audio-index:propose_palettes` / `search_samples` / `scripts/select_sample.py` (`< 25 ms`, `timbre_card`, 0 PyTorch) ou synthés natifs `gl.*` (`gl_dsl.synth_voice`) → (3) `audio-index:analyze_sample_dsp` (`< 50 ms`) → (4) `fetch_samples_online(..., monophonic_only=True)` si besoin.
> - **Zéro dissonance chromatique** : Exiger `playable_chromatically == True` (`sample_nature == "mono_oneshot"`) avant de charger un sample mélodique dans `gl.sampler` ; sinon isoler 1 note via `studio_lib.extract_clean_oneshot(...)`.
> - **Hygiène** : `A4 = 440.0 Hz ± 5 ct` (`measure_fft_tuning()` → `recommended_gl_fine_tune_ct`, jamais `pitch_shift` sur `< 80 Hz`) · one-shots `< 2.5 s` en peak norm `-1.0 dBFS` · boucles `>= 2.5 s` stabilisées (`stabilize_stereo_low_mid`) · carte fichiers → [`docs/00_SOMMAIRE.md`](00_SOMMAIRE.md).
> - **Remix / Chant** : `reverse-engineer` (`analyze_song_for_rap_flip` → `split_stems` → `transcribe_audio_to_midi` → `HPF >= 130 Hz`) · `vocaloid` (`list_voicebanks` → `syllabify_lyrics` → `create_ustx` → `synthesize_vocal_preview` → `addAudioClip`).

---

## Chapitre 1 — Arbre de Décision : Sélection & Mémoire Sonore Unifiée (`samples.db`)

* **Classification par tokens** (`indexer._infer_category_and_pack`) : les mots-clés de catégorie (`kick`, `snare`, `open hat`, `808`…) sont comparés aux **tokens** du nom de fichier (séparateurs `-_ .` ignorés, pluriels singularisés), jamais par sous-chaîne : `"chat"` ne matche pas `hat`, `"scalplover"` ne matche pas `cp`. `is_loop` = durée >= 2.2 s ou token `loop/stem/full/drumless/instrumental/acapella` (un one-shot nommé reste one-shot si < 2.5 s).
* **BPM détecté** : nom de fichier (`140 BPM`, `140bpm`) puis, pour les boucles, estimation `librosa.beat.beat_track` (stocké dans `samples_catalog.json` et `samples.db`) ; utilisé par `propose_palettes(bpm=...)` pour exclure les boucles hors tempo.
* **Hauteur détectée** : `midi_root` + `recommended_gl_fine_tune_ct` stockés dans le catalogue et remontés dans `propose_palettes` → `blueprint_tracks[].root_note` / `fine_tune_ct` → appliqués à `gl.sampler` par `song_builder`. La fondamentale passe par YIN + **HPS sur candidats d'octaves + verrou anti-partiel inharmonique** (`_refine_f0_harmonic`) : une cloche dont le partiel 2,76× est plus fort que la fondamentale garde sa vraie racine (sinon classe de hauteur fausse = dissonance). Fenêtre de snap FFT resserrée ±6 %.
* **Détection d'accords** : pics graves (50–900 Hz) mutuellement non-harmoniques (>= 2) OU chroma >= 3 classes fortes quand YIN échoue → `poly_chord_or_dyad`, `playable_chromatically=False` (jamais utilisé en rôle mélodique). Cloche inharmonique à hauteur stable : reste jouable. **Key-fit** : pénalité `key_fit_penalty` si la racine détectée est hors gamme (> 2 st) pour un one-shot mélodique ; le soundcheck rapide remonté aussi `key_fit_ok`. Le schéma blueprint **exige `root_note` pour toute piste `source_type="sample"`**.

Suivre cet ordre exact :

1. **Projet / Samples fournis initialement (`scripts/index_project_samples.py`)** :
   - Exécuter `.venv/bin/python scripts/index_project_samples.py <proj>` (ou MCP `audio-index:index_project_samples`) : scanne **uniquement** `<proj>/Samples/` (et dérivés `samples/`), écrit `<proj>/samples_catalog.json` avec rôles compatibles, `playable_chromatically`, `recommended_gl_fine_tune_ct`, `timbre_card` et recettes `sample_prep_lib.py` (dont les 2 meilleurs `slice_s` d'onsets si `melodic_sequence`).
   - Compléter si besoin par `scripts/audit_project_sources.py <projet.gridline> --prepare-samples <proj>/samples`. Zéro ajout externe (`user_samples_locked = true`).
2. **Mémoire Sémantique Offline (`timbre_card`) & Générateur de Palettes (`propose_palettes`, `< 50 ms`, 0 PyTorch)** :
   - **`semantic_anchors.json` + `.npz`** (`scripts/build_semantic_anchors.py --populate-db`) : 72 descripteurs sur 6 axes (`instrument_family`, `envelope`, `brightness`, `texture`, `mood`, `defect_flags`) pré-encodés une seule fois via CLAP en `float16`. La table **`sample_semantic_cards`** de `samples.db` stocke `clap_vec` (`float16`), `top_tags_json`, `defect_tags_json` et une **`timbre_card`** lisible sur 1 ligne (`<= 140 chars`).
   - **`audio-index:propose_palettes` / `scripts/palette_builder.py`** : combine score rôle, pénalité de chevauchement **Bark24** (surpondérée sur `Kick/Bass` `40–120 Hz` et `Chords/Lead` `800–3500 Hz`), relais d'attaques et adéquation humeur pour proposer **3 palettes distinctes** (`palette_A` Organique/Feutrée, `palette_B` Sombre/Synthétique, `palette_C` Hybride/Percussive). Respecte strictement le verrou `project_samples_dir` en Scénario B.
   - `search_samples`, `select_best_samples` et `scripts/select_sample.py` (`--role`, `--target-key`, `--mono-only`, `--top 8`) retournent `timbre_card` en pur NumPy (`< 25 ms`, 0 PyTorch) avec repli DSP + `"warning": "clap_index_missing_fallback_dsp"` si aucun vecteur CLAP n'est présent.
   - *Maintenance CLI* : `OMP_NUM_THREADS=1 .venv/bin/python scripts/index_samples.py --no-clap --workers 6` (ou `--enrich-clap --batch-size 64` / `--populate-cards`).
3. **Synthèse native `gl.*`** : 13 synthés dans `gl_dsl.SYNTH_REGISTRY` (`search_synth_presets`, zéro téléchargement).
4. **Récupération CC0 en ligne (si absent en local)** : `audio-index:fetch_samples_online(..., monophonic_only=True)` ou `scripts/fetch_samples_api.py --mono-only --no-clap` (clés `.env` : `FREESOUND_API_KEY`, `LOTSOFSOUNDS_API_KEY`).

---

## Chapitre 2 — DSP Déterministe, Pureté `mono_oneshot` & Signature Vocale

* **Outils (`0 GPU`, hors `sfz/`)** : `librosa` (YIN, onsets, HPSS rapide), `praat-parselmouth` (`HNR`, $F_0$ dès `26 Hz`, formants), `scipy`/`essentia`/`soundfile` (phase M/S, Nyquist, harmoniques 808).
* **`sample_nature`** : **`mono_oneshot`** (`playable_chromatically = True`, seul autorisé chromatiquement) · `melodic_sequence` (isoler 1 note via `extract_clean_oneshot`) · `poly_chord_or_dyad` (`DISSONANCE_RISK`) · `inharmonic_bell_or_fx` / `ambient_bed` / `percussion`.
* **`analyze_vocal_signature`** : `est_voix_probable` (`80..400 Hz`, `HNR > 7 dB`), `chant_lent_tenu` (`< 2` attaques/s), `ratio_presence > 0.75`.
* **Scoring 4 couches** : (1) Hard Filter (clipping, silence, phase `< 0.2`, `DISSONANCE_RISK`, `LOW_MID_WIDE_PHASE`), (2) Confort (`r_harsh` `2–5 kHz`, Plomp-Levelt ; assombrir par shelf doux / filtre synthé musical si centroïde `> 2.5 kHz` en `valence < 0` / `lofi_chill` (jamais de LPF brique < 6,5 kHz, voir `config/mix_profiles.json`)), (3) Adéquation rôle & Bark 24 bandes, (4) CLAP optionnel.

---

## Chapitre 3 — Nettoyage, Transitoires & Préparation (`sample_prep_lib.py`)

* **Transitoires rapides ("Tics" / Hi-Hats `>= 1/16`)** : spectre lisse sans pic étroit `Q > 8` (`> 6 dB` dans `2–6 kHz`, sinon creux `gl_dsl.eq_band` `-3..-6 dB`), `Attack >= 1–3 ms`, $T_{\text{decay}} < 0{,}8 \times T_{\text{note}}$, énergie `7–14 kHz` (`HPF >= 500 Hz`), humanisation via `groove_lib.apply_anti_machine_gun`.
* **Normalisation & Extraction** : one-shots `< 2.5 s` via `studio_lib.clean_audio_sample(..., max_peak=0.85)` (`-1.0 dBFS`, fades `2/15 ms`, jamais `loudnorm`) ; boucles `>= 2.5 s` en 24-bit `44.1 kHz` (`loudnorm` doux + `stabilize_stereo_low_mid` si corrélation `< 0.65`) ; extraction 1 note pure via `extract_clean_oneshot(...)`.
* **Pipeline déclaratif centralisé (`scripts/sample_prep_lib.py`)** : utiliser `process_sample_specs(src_dir, out_dir, specs)` (`slice_s`, `eq_peaks`, `varispeed_hz`, `pitch_shift_st`, `saturate_drive`, `time_stretch_rate`, `reverse`, `stabilize_low_mid`, export 24-bit et `manifest.json` avec `measure_fft_tuning()` → `recommended_gl_fine_tune_ct`, où `+X` monte de `+X ct`).
* **Hygiène Basse / 808** : vérifier `f0`, `2*f0`, `3*f0` via `audit_project_sources.py` ; régler `Root note` + `Fine tune` dans `gl.sampler` (**jamais** `pitch_shift` sur `< 80 Hz`).

---

## Chapitre 4 — APIs Audio en Ligne (`Freesound v2` & `Lots of Sounds`)

Clés dans `.env` (`FREESOUND_API_KEY`, `LOTSOFSOUNDS_API_KEY`). Client : `scripts/fetch_samples_api.py` ou `audio-index:fetch_samples_online`.
* **Sécurité des téléchargements** : URLs **HTTPS uniquement** (`_safe_download_bytes`), plafond **25 Mo** par fichier (streaming), refus des réponses JSON sans URL audio. **Jamais de fetch automatique quand `user_samples_locked=True`** (samples fournis : `auto_fetch` est ignoré dans `select_best_from_db`, Invariant #1).
* **Listener multimodal externe** (`listener_backend.py`) : endpoint **HTTPS uniquement**, WAV envoyé en base64 plafonné à **10 Mo**, sinon repli déterministe local.
* **Freesound v2 (`https://freesound.org/apiv2`)** : header `Authorization: Token <KEY>`, requête courte en anglais (`"warm rhodes C4"`, `"acoustic snare dry"`), filtre CC0 automatique, `fields=id,name,duration,tags,previews,license`.
* **Lots of Sounds (`https://api.lotsofsounds.com`)** : 100 % CC0, mots-clés courts (`1–2` mots), conversion obligatoire FFmpeg `-c:a pcm_s24le -ar 44100`.
* **Exemple CLI rapide (DB SQLite < 25 ms & API CC0 sans CLAP)** :
  ```bash
  .venv/bin/python scripts/select_sample.py --role lead_melody --target-key "C minor" --mono-only --top 6
  .venv/bin/python scripts/fetch_samples_api.py --role lead_melody --query "glass bell C4" --mono-only --no-clap --limit 6
  ```
*(Rappel `--query` : filtre lexicographique sur chemin/fichier SQLite ; combinable avec `--role`, `--target-key`, `--mono-only`, `--json`.)*

---

## Chapitre 5 — Protocole Universel Reverse-Engineering & Remix (Audio Source → Tout Genre)

Applicable à **toute** source audio (`.mp3`, `.wav`, `.flac`, `.ogg` : thème de jeu vidéo/VGM, chanson, boucle instrumentale) vers **tout** genre cible (Rap/Trap, Lo-Fi, House, Rock, Orchestral) :

1. **Diagnostic musical** : `reverse-engineer:analyze_song_for_rap_flip(audio_path, target_rap_bpm=<bpm_cible>)` → récupère `detected_bpm`, `key`, `chords_first_bars`, `time_stretch_ratio` et `semitones_if_vinyl_repitch`.
2. **Séparation de stems** : `reverse-engineer:split_stems(audio_path, output_dir=..., model="htdemucs_6s")` (`drums`, `bass`, `other`, `vocals`, `guitar`, `piano` ; utiliser `htdemucs` pour chiptune/8-bit). *Note CPU* : si le fichier dépasse `3 min`, lancer `gridline stems` en commande shell asynchrone pour éviter un timeout MCP.
3. **Transcription MIDI** : `reverse-engineer:transcribe_audio_to_midi(audio_path=stem, bpm=..., quantize_beats=0.25)` (Basic Pitch ONNX `tools/models/nmp.onnx`).
4. **Blueprint & Gate Top-Down** : nettoyer la transcription, figer `harmonic_core` dans `song_blueprint.json` et passer `verify_midi_gate()` (`PASS`). Choisir la stratégie :
   * **A (Sample Flip / Chop)** : découper les stems propres en tranches calées sur la grille (`HPF >= 130–160 Hz` obligatoire pour libérer la nouvelle basse).
   * **B (Remake MIDI)** : rejouer les accords/mélodies transcrits sur les instruments du genre cible (`gl.*` / `gl.sampler`).
   * **C (Hybride)** : garder un stem emblématique filtré (`other`/`piano`/`vocals`) + nouvelle rythmique/basse/contre-chant.
5. **Mixage Remix** : `HPF >= 130–160 Hz` sur les stems conservés, `stabilize_stereo_low_mid` si phase `< 0.65`, EQ miroir sous la nouvelle voix/lead, puis boucle Quality Gate (Facette 5).

---

## Chapitre 6 — Workflow Chant & Synthèse Vocale (`vocaloid` / OpenUTAU)

Lorsqu'un utilisateur demande une piste chantée avec paroles :

1. **Vérifier les banques installées** : `vocaloid:list_voicebanks()` (ex. `TIGER_DS_v106` dans `voicebanks/`).
2. **Syllaber les paroles** : `vocaloid:syllabify_lyrics(text=..., language="fr"|"en")` → compter le nombre exact de syllabes par phrase.
3. **Aligner sur la mélodie validée (`master_sketch.mid`)** :
   * Respecter une tessiture humaine confortable : **Ténor/Baryton `C3–G4` (`MIDI 48–67`)**, **Alto `F3–D5` (`MIDI 53–74`)**, **Soprano `C4–G5` (`MIDI 60–79`)**.
   * **1 syllabe = 1 note** (ou 2 notes liées sur syllabe longue), en conservant des **pauses respiratoires de `0.5–1.5 beat`** toutes les 2–4 mesures.
4. **Générer le projet & la pré-synthèse WAV** :
   * `vocaloid:create_ustx(output_path=".../vocals.ustx", bpm=..., notes=[...])`
   * `vocaloid:synthesize_vocal_preview(ustx_path=".../vocals.ustx", output_wav=".../samples/vocals.wav")` (ou ouvrir dans OpenUTAU via `open_in_openutau` pour rendu DiffSinger final).
5. **Intégrer & Mixer dans Gridline** :
   * Charger `samples/vocals.wav` dans un `gl.sampler` dédié (piste mixeur dédiée → `BUS Music` ou direct `Master 0`), placer sur la playlist via Lua `function apply() project.playlist.addAudioClip(ch, track, 0, 0) end`.
   * **Mix voix & Synchro Arrangement (`P6`)** : `HPF 100–120 Hz`, compression `3:1` (`GR 3–5 dB`), creuser **`1.85 kHz`** (`0 -> -3.5 dB`, `40/120 ms`) automatiquement sur `BUS Music` via `scripts/vocal_arrangement_sync.py` (ou code `SYNC_VOCAL_ARRANGEMENT`) et limiter les contre-chants aux fenêtres respiratoires `>= 0.5 beat` (`mute_counter_during_vocals`).
