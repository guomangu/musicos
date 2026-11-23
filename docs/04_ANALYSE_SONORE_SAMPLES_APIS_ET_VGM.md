# Facette 4 — DSP Déterministe, Sélection de Samples, APIs Audio, Remix & Chant (Vocaloid)

> **Cheatsheet IA (L'essentiel en 14 lignes)**
> - **Ordre économique de choix des sons** : (1) Sons déjà présents dans le `.gridline` utilisateur → (2) `audio-index:search_samples` (**< 10 ms** sur `samples.db`) ou synthés natifs `gl.*` vérifiés (Facette 1 §3) → (3) Vérification DSP rapide `audio-index:analyze_sample_dsp` (0 GPU, 0 CLAP) → (4) `audio-index:fetch_samples_online(..., monophonic_only=True)` uniquement si un timbre manque.
> - **Attention timeout CPU** : Ne **pas** appeler `select_best_samples` si `mcp-servers/audio-index/sample_index.json` n'existe pas (sinon indexation CLAP CPU complète synchrone). Préférer `search_samples` + `analyze_sample_dsp`, ou `evaluate_sample_quality(..., with_clap=False)`.
> - **Zéro dissonance chromatique** : Pour tout sample joué en mélodie/accords dans `gl.sampler`, exiger `playable_chromatically == True` (`sample_nature == "mono_oneshot"`). Si le sample est `melodic_sequence` ou `poly_chord_or_dyad` (`DISSONANCE_RISK`), isoler 1 note via `studio_lib.extract_clean_oneshot(...)` ou remplacer.
> - **Hygiène audio** : Accordage `A4 = 440.0 Hz ± 5 ct` (`Root note` + `Fine tune`, jamais `pitch_shift` sur 808 `< 80 Hz`) · one-shots `< 2.5 s` en **peak norm `-1.0 dBFS`** (jamais `loudnorm`) · boucles `>= 2.5 s` avec `stabilize_stereo_low_mid` si `low_mid_corr < 0.65`.
> - **Remix / Flip (tout audio → tout genre)** : `analyze_song_for_rap_flip` → `split_stems` (`htdemucs_6s`) → `transcribe_audio_to_midi` → `master_sketch.mid` → `HPF >= 130 Hz` sur stems.
> - **Chant (Vocaloid / OpenUTAU)** : `list_voicebanks` → `syllabify_lyrics` → `create_ustx` → `synthesize_vocal_preview` → `gl.sampler` + `addAudioClip`.

---

## Chapitre 1 — Arbre de Décision : Choisir des Sons Sans Gaspiller Tokens ni CPU

Suivre cet ordre exact (s'arrêter dès qu'un son propre est trouvé) :

1. **Projet `.gridline` utilisateur contenant déjà des sons** :
   - Exécuter en 1 appel `.venv/bin/python scripts/audit_project_sources.py <projet.gridline> --prepare-samples <proj>/samples`.
   - Ne rien ajouter d'externe ; nettoyer/découper/décliner uniquement les sons fournis.
2. **Recherche locale instantanée (`samples.db`, `< 10 ms`)** :
   - `audio-index:search_samples(query="warm piano", category="...", musical_key="C minor", limit=8)` ou `find_compatible_samples(bpm=..., musical_key=...)`.
   - Vérifier les 2–3 meilleurs candidats avec `audio-index:analyze_sample_dsp(filepath)` (`< 50 ms`, 0 CLAP).
3. **Synthèse native `gl.*` (0 téléchargement, 0 dissonance, 100 % polyphonique)** :
   - Idéal pour basses (`gl.fingerbass`, `gl.analog`, `gl.acid`), claviers (`gl.keys`), cordes (`gl.string`, `gl.twinstring`), plucks (`gl.plucker`), nappes (`gl.warmsynth`, `gl.dcosynth`) et kits (`gl.drumkit`). Chercher des presets avec `audio-index:search_synth_presets(query=...)`.
4. **Scoring psychoacoustique local (`sample_index.json`)** :
   - Utiliser `scripts/select_sample.py --role ... --mono-only --no-clap` **uniquement si** `mcp-servers/audio-index/sample_index.json` existe déjà.
   - **Piège CLI** : dans `scripts/select_sample.py`, `--query` est un **filtre sur le nom de fichier/chemin** (mots simples comme `"bell"`, `"kick"`), **pas** une phrase descriptive CLAP.
5. **Récupération API en ligne CC0 (Freesound v2 / Lots of Sounds)** :
   - Uniquement si aucun candidat local propre n'existe : `audio-index:fetch_samples_online(query="glass bell C4", role="lead_melody", monophonic_only=True, limit=5)` ou CLI `scripts/fetch_samples_api.py --mono-only --no-clap`.

---

## Chapitre 2 — Pipeline DSP Déterministe (Zéro GPU) & Exclusion `sfz/`

* **CLI système** : `sox`, `aubio` (`aubiopitch`, `aubioonset`, `aubiotrack`), `ffmpeg`.
* **Bibliothèques Python (`.venv`)** :
  * **`librosa`** : RMS, `effects.split`, `onset_detect`, Centroïde/Rolloff/ZCR, YIN.
  * **`praat-parselmouth`** (`0.4.7`) : **HNR** (*Harmonics-to-Noise Ratio* dB), $F_0$ dès `26 Hz`, **Formants** ($F_1, F_2, F_3$), **Jitter / Shimmer**.
  * **`scipy` / `soundfile` / `essentia`** : corrélation de phase stéréo par bande, détection sauts Nyquist, harmoniques `1*f0, 2*f0, 3*f0` sur basses/808.
* **Exclusion `sfz/`** : tous les indexeurs (`indexer.py`, `index_samples.py`, `evaluate_sample_quality`) ignorent automatiquement tout répertoire `sfz/`.

---

## Chapitre 3 — Pureté Mélodique (`mono_oneshot`), Signature Vocale & Nettoyage

### 3.1 Classification `sample_nature` (`analyze_pitch_and_purity` / `analyze_sample_dsp`)
* **`mono_oneshot`** (`playable_chromatically = True`) : note unique stable (`onsets <= 2`, `f0_std_st <= 0.65`, faible `polyphony_ratio`) → **seul type autorisé** pour jouer mélodies ou accords dans `gl.sampler`.
* **`melodic_sequence`** : riff/arpège multi-notes → interdit en piano-roll chromatique. Extraire 1 note via `studio_lib.extract_clean_oneshot(...)` ou n'utiliser que comme boucle entière dans sa tonalité exacte.
* **`poly_chord_or_dyad`** (`DISSONANCE_RISK`) : accord pré-enregistré → interdit de transposition parallèle (Règle V2).
* **`inharmonic_bell_or_fx` / `ambient_bed` / `percussion`** : textures, risers, lits ou percussions.
* **Hygiène Basse / 808** : vérifier `f0`, `2*f0`, `3*f0` via `audit_project_sources.py` ; appliquer `recommended_root_midi` + `recommended_fine_tune_ct` dans `gl.sampler` (**jamais** `librosa.effects.pitch_shift` sur `< 80 Hz`).

### 3.2 Signature Vocale (`analyze_vocal_signature`)
Répond en quelques millisecondes à *« Voix ? Chant lent ? Chant continu ? »* :
* **Voix probable** (`est_voix_probable`) : $80 \le F_0 \le 400\text{ Hz}$, $\text{HNR} > 7\text{ dB}$, formants $F_1, F_2, F_3$ vocaux.
* **Chant lent / tenu** (`chant_lent_tenu`) : `< 2.0` attaques/s, `longest_voiced_s` élevé, vibrato régulier.
* **Continuité** : `ratio_presence = duree_active_s / duree_totale_s > 0.75`.

### 3.3 Entonnoir de Scoring 4 Couches & Normalisation (`studio_lib.py`)
1. **Hard Filter** : rejet si clipping consécutif, silence, phase destructrice ($\rho_{LR} < 0.2$), `DISSONANCE_RISK` si `monophonic_only=True`, ou `LOW_MID_WIDE_PHASE`.
2. **Confort Auditif** : dureté `2–5 kHz` (`r_harsh`), pics résonants, rugosité Plomp-Levelt :
   * **Plafond de brillance selon la Valence** : Pour tout projet avec `valence < 0` ou profil `lofi_chill` / `trap_rap` sombre, rejeter ou filtrer (`gl.sampler Filter = 1 [Low pass], Cutoff <= 6 kHz`) tout sample `lead_melody` ou `chord_keys` dont le centroïde spectral dépasse **`2.5 kHz`** ou présentant une énergie excessive dans `2.5–5 kHz` (`r_harsh`).
   * **Hi-hats doux** : Privilégier des hats courts sans pic résonant entre `3 kHz` et `6 kHz` ; si le sample est métallique, appliquer systématiquement un LPF à `10–12 kHz` et un creux `-3 dB @ 4 kHz`.
3. **Adéquation au Rôle & Masquage Bark (24 bandes)** : roles valides = `kick`, `bass_sub`, `snare_clap`, `hihat_perc`, `lead_melody`, `pad_atmosphere`, `chord_keys`.
4. **Esthétique CLAP** (optionnel CPU) + calcul `Root note` / `Fine tune` vers `440.0 Hz`.

* **Nettoyage obligatoire & 4 Lois d'agrément des transitoires rapides ("Tics" / Hi-Hats)** :
  * **1. Ratio Harmonie/Bruit (`2–6 kHz`)** : spectre large et lisse sans pic isolé (**$Q > 8$** dépassant de **`> 6 dB`** le voisinage spectral) ; si présent, appliquer un EQ soustractif chirurgical (**`-3` à `-6 dB`**, $Q$ étroit).
  * **2. Enveloppe transitoire** : temps d'attaque (**`Attack`) de `1 à 3 ms`** (jamais `0 ms` numérique) et relâchement strictement inférieur à l'écart entre frappes (**$T_{\text{decay}} < 0{,}8 \times T_{\text{note}}$**).
  * **3. Loi Anti-Mitraillette (micro-modulation stochastique)** : varier chaque frappe sur la **vélocité** (ex. temps fort `100 %`, contretemps `65 %`, subdivisions `40 %`, **`±5 %`**), la **hauteur** (**`±5` à `±15 ct`**) et le **filtre LPF** (fréquence de coupure proportionnelle à la vélocité).
  * **4. Ancrage spectral (`>= 1/16`)** : concentrer l'énergie principale entre **`7 kHz et 14 kHz`** avec **HPF `500 Hz – 1 kHz`** minimum pour supprimer tout résidu grave.
  * **One-shots courts (`< 2.5 s`, percussions)** : `clean_audio_sample(..., max_peak=0.85, fade_in_ms=2.0, fade_out_ms=15.0)` (**peak norm `-1.0 dBFS`**, jamais `loudnorm`).
  * **Boucles / Pads (`>= 2.5 s`)** : `pcm_s24le 44.1 kHz`, `loudnorm` doux (`I=-14:TP=-1.0:LRA=11`), et `stabilize_stereo_low_mid(...)` (`< 650 Hz`) si `low_mid_corr < 0.65`.
  * **Extraction 1 note pure** : `extract_clean_oneshot(in_path, out_path, start_s=0.0, max_dur_s=0.65, fade_in_ms=2.0, fade_out_ms=25.0)`.

---

## Chapitre 4 — APIs Audio en Ligne (`Freesound v2` & `Lots of Sounds`)

Clés dans `.env` (`FREESOUND_API_KEY`, `LOTSOFSOUNDS_API_KEY`). Client : `scripts/fetch_samples_api.py` ou `audio-index:fetch_samples_online`.
* **Freesound v2 (`https://freesound.org/apiv2`)** : header `Authorization: Token <KEY>`, requête courte en anglais (`"warm rhodes C4"`, `"acoustic snare dry"`), filtre CC0 automatique, `fields=id,name,duration,tags,previews,license`.
* **Lots of Sounds (`https://api.lotsofsounds.com`)** : 100 % CC0, mots-clés courts (`1–2` mots), conversion obligatoire FFmpeg `-c:a pcm_s24le -ar 44100`.
* **Exemple CLI rapide (sans CLAP)** :
  ```bash
  .venv/bin/python scripts/fetch_samples_api.py --role lead_melody --query "glass bell C4" --mono-only --no-clap --limit 6
  ```

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
   * **Mix voix** : `HPF 100–120 Hz`, compression `3:1` (`GR 3–5 dB`), creuser **`-2.5 à -4 dB` à `1.2–2.8 kHz`** sur `BUS Music` (EQ miroir) et couper tout instrument Lead concurrent pendant le chant (Règle des 3 Couches).
