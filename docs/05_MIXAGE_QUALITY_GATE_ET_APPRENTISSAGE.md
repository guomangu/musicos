# Facette 5 — Mixage Multi-Genres, Stems, Auto-Fix, Quality Gate & Références

> **Cheatsheet IA (L'essentiel v2.1 — Loop-First É1 → É4)**
> - **Validation Incrémentale Loop-First (`scripts/loop_first_runner.py` → `pipeline_state.json`)** :
>   - **É1 Sound Check Unitaire (`audio-analyzer:audit_track_soundcheck` / `scripts/soundcheck_lib.py`)** : teste chaque piste isolée jouée sur `2–4` mes. (audibilité, justesse `<= 8 ct` (soundcheck ; fondation 6 ct / peak expressif 12 ct / warning 6,5 ct — `config/mix_profiles.json`), intégrité chromatique, propreté `2–5 kHz`, enveloppe vs BPM, fidélité valence) via `ListenerBackend` (`DeterministicSemanticListener` 0 GPU ou `MultimodalAudioListener`). Optionnel : `scripts/export_palette_preview.py --human-gate` (`renders/palette_preview.wav`).
>   - **É2 Tests Verticaux sur Noyau `4/8` mes. (`--core-only` + `audio-analyzer:audit_vertical_step` / `scripts/vertical_tester.py`)** : `V1_LOW_END` → `V2_GROOVE` *(sauté si `has_drums=false`)* → `V3_HARMONY` *(conflit chroma CQT basse/accords + `low_mids`)* → `V4_LEAD` → gèle `core_loop_locked = true`. **Isolation stricte** : interdit d'éditer `forbidden_tracks` ($1 \dots k-1$) ; max 2 retries avant `SWAP_SAMPLE`.
>   - **É3 Audit des Coutures (`audio-analyzer:audit_horizontal_seam` / `scripts/horizontal_tester.py`)** : rend uniquement les fenêtres `4` mes. (`±2` mes.) sur chaque transition majeure + `Outro` (`ΔLUFS`, `beat4_dropout`/`filter_sweep`, `0` glitch `±150 ms`, Outro `< -45 dBFS`).
> - **É4 Boucle fermée & Auto-Fix** : Lancer `render_and_audit.py <projet.gridline> --profile <genre> --blueprint song_blueprint.json --range <b1>-<b2> --no-neural --auto-fix --max-passes 2` puis **1 seul rendu complet** `--mp3 --sections "..."` (`renders/autofix_history.json`).
> - **Stems & Référence Spectrale** : Auditer les stems via `render_project_stems` + `analyze_stem_hierarchy` (bornes `lead_vs_harmony` `[3.0, 9.5] dB`, `kick_bass_transient_overlap`, `pink_slope_db_per_oct` `-3..-6.5`). `render_and_audit` calcule aussi `spectral_match_score` + `corridor_score` (`0..1`) via `golden_references/spectral_corridors.json` versionné (0 PyTorch).
> - **Genres sans batterie (`has_drums=false` / `classical_ambient`)** : Passer `--no-drums` ou `--blueprint song_blueprint.json` (`"has_drums": false`) ; le gate désactive l'exigence `dynamic_arc` et filtre les résonances inter-harmoniques.
> - **Topologie Buses (`setup_standard_buses`)** : `BUS Drums (17)` · `BUS Bass (18)` · `BUS Music (19)` · `FX Room (20)` · `FX Delay (21)` · `FX Hall (22)` · `Master (0)` configurés via **`scripts/gl_dsl.py`** (carte → [`docs/00_SOMMAIRE.md`](00_SOMMAIRE.md)).

---

## Chapitre 1 — Profils de Mix, Spatialisation & Pente Rose (`--profile`)

> **Source unique des seuils (v3.1+)** : Toute valeur de Ceiling limiteur (`-1.2 dBTP`), `gl.dist` (`<= 0.04`), tolérance d'accordage (`6 ct` fondation / `12 ct` peak expressif / `6.5 ct` warning master, `8 ct` soundcheck), bornes `lead_vs_harmony` (`[3.0, 9.5] dB`), budget notches (`2`/piste, `3` total) et planchers LPF sources (**jamais de LPF brique 24 dB/oct sous `6.5 kHz` en harmonie / `9 kHz` en lead** ; utiliser un *high-shelf* doux ou un filtre de synthé musical pour assombrir un timbre sans créer de « mix en U »**) est gouvernée **exclusivement** par `config/mix_profiles.json` via `scripts/mix_config.py`. Les corridors spectraux de référence résident dans `golden_references/spectral_corridors.json` (versionné, remplaçant `spectral_profiles.json` comme référence de gate).

Les seuils de `scripts/render_and_audit.py` et `studio_lib.setup_standard_buses()` s'adaptent au profil choisi dans `song_blueprint.json` :

| `mix_profile` | Genres Couverts | Cible LUFS | `low_mids` max | `air` min *(adaptatif `V < -0.2`)* | Sharpness max | Arc Dynamique min | Caractère des Buses (`setup_standard_buses`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`trap_rap`** | Trap, Cloud, Drill, Rap moderne | `-11.0` (`-12..-9.8`) | `<= 24 %` | `>= 3.0 %` *(sombre : **`>= 1.2 %`**)* | `< 1.90` *(sombre `< 1.85`)* | `>= 2.5 dB` | Sidechain Kick→808 doux **`2.4:1`** (`-18 dB`, Release `65–80 ms`), `gl.dist` Soft `<= 0.04` (`Tone <= 7.5k`) avant `gl.eq` LPF, `FX Room` court |
| **`pop_rnb`** | Pop, Synthwave, R&B, Afrobeat, Reggaeton | `-10.5` (`-11.5..-9.5`) | `<= 26 %` | `>= 3.0 %` *(mélancolique : **`>= 1.5 %`**)* | `< 1.95` *(sombre `< 1.85`)* | `>= 2.5 dB` | Sidechain modéré `3:1`, `BUS Music` équilibré, poche voix/lead `1.5–2.5 kHz` |
| **`edm_club`** | House, Techno, EDM, DnB, Rage | `-9.0` (`-10.0..-7.8`) | `<= 24 %` | `>= 3.5 %` *(sombre : `>= 1.75 %`)* | `< 2.05` *(sombre `< 1.90`)* | `>= 2.5 dB` | Sidechain punchy `5:1` sur basse & pads, air contrôlé, sub mono verrouillé |
| **`rock_metal`** | Rock, Indie, Punk, Metal | `-10.0` (`-11.0..-8.5`) | `<= 28 %` | `>= 2.5 %` *(sombre : `>= 1.25 %`)* | `< 2.10` *(sombre `< 1.90`)* | `>= 2.0 dB` | Sidechain léger `3:1`, médiums guitares riches, batterie organique punchy |
| **`lofi_chill`** | Lo-Fi, Boom Bap, Chillhop, Downtempo | `-14.0` (`-15.0..-12.5`) | `<= 34 %` | `>= 0.5 %` *(sombre : `>= 0.25 %`)* | `< 1.75` | `>= 1.8 dB` | LPF chaud `8.5–9.5 kHz` sur `BUS Music`, `gl.dist` saturation douce `<= 0.04`, 0 boost air agressif |
| **`jazz_acoustic`** | Jazz, Soul acoustique, Folk, Bossa | `-14.5` (`-16.0..-13.0`) | `<= 36 %` | `>= 1.0 %` *(sombre : `>= 0.5 %`)* | `< 1.85` *(sombre `< 1.80`)* | `>= 2.5 dB` | **0 distorsion**, compression douce/transparente, `FX Room` naturel (`0.6–1.0 s`) |
| **`classical_ambient`** | Classique, Orchestral, Piano solo, Ambient | `-16.0` (`-18.0..-14.0`) | `<= 40 %` | `>= 0.3 %` *(sombre : `>= 0.15 %`)* | `< 1.75` | `>= 3.0 dB` | **0 distorsion**, sidechain minimal, dynamique ample, `FX Room` profond (`0.8–1.1 s`) |

> **Règle d'adaptation de `air min` à la Valence** : Pour tout morceau sombre (`emotion_vector.valence < -0.2` ou flag `--dark` dans `render_and_audit.py`), réduire le seuil `air min` de moitié (ex. `trap_rap` sombre : **`>= 1.2 %`** au lieu de `3.0 %` ; `pop_rnb` mélancolique : **`>= 1.5 %`**) et durcir `Sharpness max` (`< 1.85`), afin de ne jamais forcer l'éclaircissement artificiel d'un mix sombre.

### Plan Fréquentiel Universel à 5 Bandes
| Bande | Rôle Principal | Isolation & Règles |
| :--- | :--- | :--- |
| **Sub (`20–60 Hz`)** | 808 / Sub / Contrebasse | Mono strict (`Width = 0`), `HPF 28–35 Hz` sur `BUS Bass`. |
| **Bass (`60–250 Hz`)** | Kick + corps de Basse | `HPF 90–145 Hz` sur `BUS Music` (selon profil) ; sidechain ou poche EQ `-2 à -4 dB @ 80 Hz` sur basse. |
| **Low-Mids (`250–500 Hz`)** | Chaleur accords / guitares / piano | Creuser `-3 à -5 dB @ 250–350 Hz` si boueux ; stabiliser M/S `< 650 Hz` si `low_mid_corr < 0.65`. |
| **High-Mids (`500 Hz–4 kHz`)** | Lead, Voix, Snare, Pluck | Relais temporel + EQ miroir `-2 à -4 dB` (`1–3 kHz`) sur Harmonie ; **Lead domine l'harmonie de `>= +3 dB`**. |
| **Air (`4–20 kHz`)** | Hats, Cymbales, Shimmer, Souffle | `LPF 8.5–13.5 kHz` sur `BUS Music` (selon profil/valence) pour adoucir les harmoniques et laisser l'air aux percussions. |

### Cartographie Psychoacoustique des Aigus (`2–20 kHz`) & Loi de la Pente Rose
* **Justification physiologique (Fletcher-Munson)** : le conduit auditif résonne naturellement entre **`2,5 kHz` et `4 kHz`** (sensibilité évolutive maximale aux cris/parole) — un excès y cause une fatigue physique immédiate, alors que `> 10 kHz` reste spatial et non douloureux.

| Sous-zone Aigus | Fréquences | Rôle Musical | Tolérance & Risques |
| :--- | :---: | :--- | :--- |
| **1. Présence / Attaque** | `2–5 kHz` | Clic Kick, claquement Snare, diction vocale. | **Critique (Bas)** : excès = son métallique, strident ("perce-tympan"). |
| **2. Brillance / Détail** | `5–10 kHz` | Sifflantes (*s, ch, t*), *crispness* des cymbales/hats. | **Modéré** : excès = sibilance douloureuse, rendu plastique/artificiel. |
| **3. Air / Soie** | `10–20 kHz` | Légèreté, ouverture spatiale "studio", élévation du mix. | **Élevé** : excès = souffle blanc (*hiss*) ou électricité statique. |

* **Loi de la Pente Rose (`-3` à `-4,5 dB/octave`)** : l'énergie globale des graves vers les aigus doit décroître linéairement (environ `-3` à `-4,5 dB/oct`, comme pluie/vent) — un spectre plat (`0 dB/oct`) sonne agressif et maigre. **Zone cible "tics" rapides (`>= 1/16`) : `7–14 kHz`** (+ HPF `500 Hz–1 kHz`).

| Plage d'Octave | Niveau Cible (Réf. `100 Hz = 0 dB`) | Diagnostic en cas d'écart |
| :--- | :---: | :--- |
| `50–100 Hz` | **`+3` à `+4,5 dB`** | Trop bas = léger/sans fondation · Trop haut = boueux/saturé. |
| `100–200 Hz` | **`0 dB` (Réf.)** | Zone de chaleur et d'impact corporel. |
| `400–800 Hz` | **`-6` à `-9 dB`** | Trop haut = son "carton" (*boxy*), nasal, encombré. |
| `1,6–3,2 kHz` | **`-12` à `-18 dB`** | Trop haut = fatigue auditive immédiate, agression. |
| `6,4–12,8 kHz` | **`-18` à `-27 dB`** | Trop bas = étouffé/sombre · Trop haut = strident/acide. |

### Matrice de Spatialisation 3D (`X` Largeur · `Y` Profondeur · `Z` Hauteur)
| Axe Spatial | Paramètre Technique | Règle d'Application par Catégorie |
| :--- | :--- | :--- |
| **Largeur (`X`)** | Panoramique (`Pan`) & M/S | **Centre strict (`0%`, 100 % Mono `< 120 Hz`)** : Kick, Sub/Basse, Snare, Lead/Voix.<br>**Décalé (`±15–35 %`)** : Hi-hats, Percussions, Arpèges.<br>**Large (`±50–100 %`)** : Pads stéréo, Guitares doublées, Cymbales, Ear Candy. |
| **Profondeur (`Y`)** | Volume + Réverb + LPF | **1er plan (Sec, fort, brillant)** : Lead, Kick, Snare.<br>**2e plan (Réverb courte, `-3 à -6 dB`)** : Rythmique secondaire, Contre-chants.<br>**Arrière-plan (Réverb longue, aigus atténués)** : Pads, Nappes, Ambiances. |
| **Hauteur (`Z`)** | Égalisation (EQ) | **Sol** : Sub & Basse · **Milieu** : Corps & Voix · **Plafond** : Air (`> 10 kHz`), Cymbales, brillance. |

---

## Chapitre 2 — Topologie des Buses (`gl_dsl` + `studio_lib.setup_standard_buses`)

> **Source unique des seuils (v3.1+)** : Toute valeur de Ceiling limiteur (`-1.2 dBTP`), `gl.dist` (`<= 0.04`), tolérance d'accordage (`6 ct` fondation / `12 ct` peak expressif / `6.5 ct` warning master, `8 ct` soundcheck), bornes `lead_vs_harmony` (`[3.0, 9.5] dB`), budget notches (`2`/piste, `3` total) et planchers LPF sources (**jamais de LPF brique 24 dB/oct sous `6.5 kHz` en harmonie / `9 kHz` en lead** ; utiliser un *high-shelf* doux ou un filtre de synthé musical pour assombrir un timbre sans créer de « mix en U »**) est gouvernée **exclusivement** par `config/mix_profiles.json` via `scripts/mix_config.py`. Les corridors spectraux de référence résident dans `golden_references/spectral_corridors.json` (versionné, remplaçant `spectral_profiles.json` comme référence de gate).

`setup_standard_buses(mcp, bpm=..., profile=..., valence=...)` crée les pistes manquantes jusqu'à `22` et configure via `gl_dsl` :
1. **`BUS Drums` (Insert `17`)** : `gl.eq` (`HPF 34 Hz`, shelf `10 kHz` selon profil/valence) + `gl.comp` (`2.2:1`).
2. **`BUS Bass` (Insert `18`)** : `gl.stereo` (`Width = 0.0`) + `gl.comp` sidechain (`1`, preFader Kick ; `2.4:1 / -18 dB / 65 ms` en `trap_rap`, `5:1` `edm_club`, `3:1` pop/rock/lofi, `1.5:1` acoustique) + `gl.eq` (`HPF 30 Hz`).
3. **`BUS Music` (Insert `19`)** : `gl.dist` (`Mix=0.0` acoustique/classique, `<= 0.04` moderne, `Tone<=7500`, `Remove DC=1` — seuil `config/mix_profiles.json`) → `gl.eq` (`HPF 90–135`, shelf `9.5 kHz`, `LPF 8.5–13.5 kHz`, Peak `-2 dB @ 3.2k` si `V<-0.2`) → `gl.comp` glue (`Threshold=-13.5 dB`, préserve l'arc) → `gl.comp` ducking (`Detector=1`) → **2 × `gl.stereo`** (largeur mid/high legacy, `Mono below=250 Hz`, `Dephase=0`). En `trap_rap`/`pop_rnb` sombres, LPF maintenu `>= 11.5 kHz` (garde-fou anti-sur-filtrage `Q3.3`).
4. **`FX Room` (Insert `20`)** : `gl.reverb` via `gl_dsl.reverb_room()` (`Dry=0`, `Dry (full)=0`, `Decay <= T_noire` rythmique ou `0.7–1.1 s` acoustique/ambient ; jamais de send sur Kick/Sub/Hats).
5. **`FX Delay` (Insert `21`)** : `gl.delay` via `delay_send_fx(bpm, valence)` (`Tempo sync`, croche pointée par défaut, `Feedback 20–32 %`, `Dry=0`) + `gl.eq` `HPF 400 / LPF 3500` (`2800` si `V<-0.2`). Sends `delay_send` (`0.08–0.18`) réservés à Lead/Pluck/Counter/Voix.
6. **`FX Hall` (Insert `22`)** : `gl.reverb` Hall via `reverb_hall()` (`Decay 1.6–2.4 s`, `Dry=0`, `HPF 500 / LPF 5500`, `4500` si `V<-0.2`) + `gl.eq`. Sends `hall_send` (`0.14–0.30`) sur Pads/Strings/Ear Candy/Impacts (jamais Kick/Sub/Snare sec).
7. **`Master` (`0`)** : `gl.eq` (`Quality=1`, `HPF 28`, `LPF 17k`) → `gl.comp` (`1.6:1`) → `gl.limiter` via `gl_dsl.limiter_master()` (`Attack=0.002`, `0.2 dB safety=1`, `Ceiling=-1.2 dBTP` — `config/mix_profiles.json`).

---

## Chapitre 3 — Pipeline Loop-First (`É1 → É3`), Quality Gate Conditionnel (`É4`), Stems & Auto-Fix

### 3.0 Audits Incrémentaux Loop-First (`soundcheck_lib`, `vertical_tester`, `horizontal_tester`, `loop_first_runner`)
Tous les nouveaux outils retournent un **contrat JSON uniforme concis** (`verdict`: `PASS|FIX|SWAP`, `score`, `human_diagnosis` 1–3 phrases FR, `next_action` unique) et écrivent les métriques détaillées dans `<proj>/renders/diagnostics/`.

| Étage | Outil MCP / Script | Ce qui est vérifié | Action unique / Garde-fou |
| :---: | :--- | :--- | :--- |
| **É1** | `audio-analyzer:audit_track_soundcheck` (`scripts/soundcheck_lib.py`, `scripts/listener_backend.py`) | Rendu headless `2–4` mes. par piste isolée : RMS `> -42 dBFS`, justesse jouée `<= 8 ct` (soundcheck), intégrité chromatique, `0` pic `> +5 dB` (`2–5 kHz`) ni glitch, enveloppe vs BPM, fidélité `valence`. | `CHECK_ROUTING_OR_SWAP`, `ADJUST_FINE_TUNE`, `APPLY_LPF`, `ADD_NOTCH_EQ`, `ADJUST_ADSR` · `export_palette_preview.py` (`palette_preview.wav`). |
| **É2** | `audio-analyzer:audit_vertical_step` (`scripts/vertical_tester.py`, `song_builder.py --core-only`) | Paliers cumulatifs sur `core_loop.gridline` (`4/8` mes.) : `V1_LOW_END` (overlap/phase `<150 Hz`), `V2_GROOVE` (sharpness/mitraillette, sauté si `has_drums=false`), `V3_HARMONY` (conflit chroma CQT basse/accords + `low_mids`), `V4_LEAD` (hiérarchie `>= +3 dB`, pente rose, `spectral_match`). | **Isolation** : `forbidden_tracks` verrouillé (`loop_first_runner.py` bloque toute édition hors couche $k$ ; escalade `SWAP_SAMPLE` après 2 échecs) → `core_loop_locked = true`. |
| **É3** | `audio-analyzer:audit_horizontal_seam` (`scripts/horizontal_tester.py`) | Fenêtres `4` mes. (`±2` mes.) sur transitions majeures + `Outro` : contraste `ΔLUFS`, événement (`beat4_dropout` / `filter_sweep`), `0` glitch `±150 ms`, RMS dernière seconde Outro `< -45 dBFS`. | `INCREASE_SECTION_CONTRAST`, `ADD_TRANSITION_EVENT`, `CLEAN_SEAM_FADE`, `APPLY_OUTRO_FADE` via `transitions_lib.py`. |

### 3.1 Critères `render_and_audit.py` + `analyze_stem_hierarchy` + `spectral_match_score`
* **Checklist Structure & Timbre Pré-Rendu** : `100 % Mono < 120 Hz` · aucune note `< 100 Hz` plus courte que `1/8` (`Release < 50 ms`) · `1–8` pistes simultanées par mesure · Lead dominant de `>= +3 dB` · aucune boucle de 8 mesures identique · isolation strictement `1 sonorité = 1 piste` · aucun pic isolé `> +4 dB` dans `2,5–4,5 kHz` · pente rose `-3..-5 dB/oct` · transitoires décalés `>= 5 ms` (`groove_lib`).
* **12 critères Quality Gate** (valeurs gouvernées exclusivement par `config/mix_profiles.json` via `mix_config.py`) : (1) `clipped == 0`, (2) `true_peak <= ceiling+0.2` (`-1.0 dBTP`), (3) `|lufs - target| <= 1.2`, (4) `low_end_corr >= 0.50`, (5) `low_mid_corr >= 0.55`, (6) `sharpness < max`, (7) `micro_glitches == 0`, (8) `resonances == []`, (9) `low_mids <= max %`, (10) `air >= min %`, (11) `|tuning_offset| <= 6.5 ct` (warning master ; fondation 6 ct), (12) `dynamic_arc >= min dB`.
* **Résonances « notchable » uniquement** (`deep_diagnostics.classify_resonances`) : un pic 1/3 d'octave ne déclenche la gate / l'auto-fix `EQ_RESONANCE_NOTCH` que s'il est **étroit** (`peak_width_bands <= 2`) **et** fort (`local_excess_db >= +6 dB`). Les plateaux larges (formantes, pads) sont du contenu musical : `third_octave_resonance_candidates` les rapporte sans les toucher. Le check 2–5 kHz du soundcheck applique la même règle de largeur.
* **Son naturel (anti-« électronique/écrasé »)** : master limiter `Gain <= 5.5 dB` / `Ceiling -1.2 dBTP`, comp master `-12 dB / makeup 0.8`, `gl.dist` BUS Music `Mix <= 0.04, Drive 2.5` (tous ces seuils : `config/mix_profiles.json`), LPF BUS Music >= 12 kHz, shelf aigus batterie `+1.2 dB @ 10 kHz` max. Palette A organique favorisée (bonus 0.09) quand des samples réels existent. **Gate `low_end_share`** : sub+bass <= `low_end_max` % (42 % trap_rap, voir `GENRE_GATES`) sinon auto-fix : BUS Bass `-1.5 dB` (max 2 passes) + HPF BUS Music 160 Hz. Mesuré sur master_v9 : 56 % d'énergie < 120 Hz et air 1,5 % = mix boueux/sombre.
* **Floors de présence** (`gain_staging_lib.ROLE_PRESENCE_FLOOR_DB`) : lead >= -3 dB, chords_pad >= -10 dB, etc. vs l'ancre — une piste ne peut plus devenir imperceptible ; compensation **+2 dB si centroid > 4 kHz** (cloches/aigus perceptivement faibles). Le check vertical V4 exige toujours lead >= harmonie + 3 dB.
* **Audit musical projet-wide du blueprint (`scripts/blueprint_safety.py`)** : `blueprint_musical_safety_issues(bp, catalog_path)` contrôle **chaque piste `source_type="sample"` avant build/rendu** : `sample_path` + `root_note` obligatoires, cross-check `root_note` vs `midi_root` détecté dans `samples_catalog.json` (même classe de hauteur requise — erreur d'octave inoffensive, erreur de classe = dissonance, avec `recommended_root_note`), sample `playable_chromatically == false` (accord/séquence) interdit en rôle mélodique, key-fit vs la gamme (mode-aware via `harmonic_core.mode`). Tourné dans `song_builder.load_blueprint` (`meta["musical_safety"]`), après `apply_palette_to_blueprint` (`loop_first_runner`, état `pipeline_state.json`) et dans le **quality gate** de `render_and_audit` : rapport `musical_safety`, et toute issue `severity="error"` bloque le rendu (`gate_failures += musical_safety[...]`). Rôles percussifs et boucles exemptés du key-fit.
* **v3.0 — plan de refonte complet** : verrou d'octave exact `root_note == midi_root`, plafond `S_max` 5/7 st, boucles `melodic_loop` (key-fit CQT + verrou BPM 0,05/r_eff [0,88;1,14]), `kick_tuning_guard` + confinement cloches {1,5,8}, Low Interval Limit, autofix contour-preserving, `sanitize_note_overlaps` final, source unique `config/mix_profiles.json` (LPF sources >= 6,5/9 kHz anti « mix en U », notches 2/3 -> SWAP_SAMPLE, justesse 6 ct stems / 12 ct vibrato), `stem_matrix_evaluator` (1 rendu stems 8 mesures -> V1..V4 + solveur de gains en NumPy), `compile_song.py` 100 % déclaratif, 3 macro-outils MCP, `docs/AGENT_RUNBOOK.md` (< 6 000 cars) + `docs/REFERENCE_ARCHIVE.md`. Tests : `tests/test_pitch_and_loop_safety.py`, `test_harmonic_realism.py`, `test_mix_coherence_dag.py`, `test_3turn_pipeline_budget.py`, `test_docs_hygiene.py`.
* **v3.1 — spécifications techniques (6 chantiers, doc `audit_spécifications_techniques_ai_studio_v3_1.md`)** : **source unique renforcée** — 100 % des seuils de mix/gate passent exclusivement par `config/mix_profiles.json` via `scripts/mix_config.py` (`get_profile_config`, `tuning_tolerance`, `notch_budget`, `stem_hierarchy_bounds`, `autofix_step_bounds`) ; autofix **transactionnel** (`scripts/autofix_transaction.py` : paliers 1/2/3, rollback immédiat si un invariant de Palier 1 régresse, blacklist + journal `renders/autofix_history.json`) ; **ScoreIR** (`scripts/score_ir.py` + `schemas/score_ir.schema.json`, artefact debug `<projet>/score_ir.json`) : le groove (swing/strum/Haas/AMG) s'accumule dans `expression` sans toucher `grid_beat`/`pitch`, et le **clamp de frontière harmonique** (guard 20 ms) garantit 0 clash b2 / 0 chevauchement 808 post-groove ; couplage timbre↔voicing (`adapt_voicing_to_timbre` : confinement {1,5,8} cloches, drop-2/shell, LIL) + audit audio chroma-vs-grille par segment (`stem_matrix_evaluator.evaluate_barwise_harmonic_alignment`, action `FIX_TRACK_ROOT_NOTE` câblée dans `multiscale_autofix`) ; **corridors spectraux versionnés** `golden_references/spectral_corridors.json` (7 profils, bark24 p10/p50/p90 + 8 métriques + `dark_valence_overrides`, dead-zone corridor loss) avec repli 3 niveaux (`reference_tier` 1 = audio utilisateur / 2 = corridor versionné + taste ≤ 30 % / 3 = baseline psychoacoustique, `reference_provenance` toujours renseigné) ; hygiène des banques (`scripts/asset_hygiene.py` : whitelist d'extensions audio + magic bytes + parasites OS ; **pas de détection de licence/copyright** — responsabilité de l'utilisateur). Tests : `test_config_single_source.py`, `test_post_groove_harmony.py`, `test_autofix_rollback.py`, `test_offline_corridors.py`, `test_soundbank_hygiene.py`.
* **Équilibrage des volumes** : `balance_tracks_in_session` (in-session, sans .wav, trims ±6 dB vers cibles de rôle, ancrage -16 LUFS actifs via `ROLE_TARGET_OFFSETS_DB`) ; repli `apply_nominal_balance` dans `song_builder` quand aucune métrique soundcheck n'existe (évite les faders par défaut inégaux).
* **Humanisation rythmique** (`groove_lib`) : `apply_anti_machine_gun(..., humanize_sparse=True)` appliqué à **toutes** les pistes dans `song_builder` (accent métrique doux sans jitter sur les notes tenues) ; `apply_swing` décale les croches off-beat de `swing × 0.25` temps ; `apply_haas_pocket` plafonne chaque décalage à **5 % du temps de beat** (anti-flou aux tempos élevés).
* **Garde-fou sans batterie (`has_drums == False` ou `classical_ambient`)** : passer `--no-drums` ou `--blueprint song_blueprint.json` ; désactive l'obligation `dynamic_arc` (adapté aux crescendos continus sans drops batterie) et filtre les faux positifs inter-harmoniques sous `150 Hz` ou `> 6 kHz` (l'analyse tiers d'octave exclut aussi les pentes monotones).
* **Hiérarchie des Stems (`render_project_stems` + `analyze_stem_hierarchy`)** : vérifie `lead_vs_harmony` dans `[3.0, 9.5] dB` (`stem_hierarchy_bounds` config), `kick_bass_transient_overlap <= 0.15` et `pink_slope_db_per_oct` (`-3..-6.5 dB/oct`).
* **Référence Spectrale (`scripts/bootstrap_golden_refs.py`)** : `golden_references/spectral_corridors.json` (versionné) compare la distribution Bark 24 bandes + 8 métriques aux corridors `[P10,P90]` du profil via `corridor_score`/`spectral_match_score` (`0..1`, 0 PyTorch) ; `spectral_profiles.json` reste le format legacy du builder.

### 3.2 Boucle `--auto-fix --max-passes 2` & Table Symptôme → Action Unique
En cas d'échec, **`render_and_audit.py --auto-fix --max-passes 2`** applique automatiquement les fixes déterministes (`gl_dsl.eq_band` Peak `-3.5..-4.5 dB` sur résonances/`280 Hz`, ajustement `Master Gain`, abaissement `Limiter Ceiling`), re-rend et journalise chaque passe dans **`renders/autofix_history.json`**.

| Symptôme (`failures`) | Action Unique (`gl_dsl` / Python) |
| :--- | :--- |
| **`clipped > 0` / `true_peak > -1.0`** | `limiter_master(gain, ceiling - 0.4)` (`Attack=0.002`, `safety=1`). |
| **`lufs` trop bas / haut** | Ajuster `Master slot 3 Gain` de `±Δ` (`\|Δ\| <= 4 dB`/passe, plafond `-1.2 dBTP`). |
| **`low_end_corr < 0.50`** | `BUS Bass Width=0.0`, Kick `pan=0`, `fx_send=0` sur Kick/Basse, `BUS Music HPF >= 120`. |
| **`stereo_width < min dB`** | `WIDEN_HARMONY_OR_PERC` : `gl.stereo` legacy (`Width 2.0`, `Side level 6-12`, `Mono below 250`) + sends `FX Hall` `+0.06` sur Chords/Pads. |
| **`stereo_balance > 1.2 dB`** | `FIX_STEREO_BALANCE` : recentrer/nudge `±0.10-0.16` pan sur 1–2 inserts panoramiques, max 2 passes. |
| **`lead_vs_harmony` hors `[3.0, 9.5]`** | `ADJUST_TRACK_FADER(target_track, delta_db)` borné `±4 dB/passe` (cumul `8 dB`), `forbidden_tracks` respectés en É2. |
| **`low_mid_corr < 0.55`** | `stabilize_stereo_low_mid()` sur le sample ou `mixer/N/stereo = -0.35`. |
| **`sharpness >= max`** | Vérifier Lead `<= MIDI 74`, filtre de synthé musical `5–7 kHz` (pente douce, jamais LPF brique 24 dB/oct < 9 kHz lead), Peak `-3.5 dB @ 3.2 kHz` (`q=1.4`). |
| **`micro_glitches > 0`** | `audit_project_sources.py --prepare-samples`, `fade_in >= 2 ms`, hats `Attack >= 4 ms`, `LPF 13 kHz`. |
| **`resonances = [F]`** | `eq_band(slot, "peak", F, -4.0, q=2.4)` sur l'insert dominant (`Chords`/`Bass`/`HiHat`/`BUS Music`). |
| **`low_mids > max %`** | `BUS Music slot 2` : `HPF 140 Hz` + Peak `-4.0 dB @ 280–330 Hz` (`q=1.2`). |
| **`air < min %`** | Si `V >= 0` : `BUS Drums` `+1.5 dB @ 10 kHz` / ouvrir LPF · Si `V < 0` : passer `valence`/`--dark` et relever hats `+1 dB`. |
| **`tuning_offset > 6.5 ct`** (warning master ; fondation 6 ct / expressif 12 ct — config) | Recalibrer `Root note` + `Fine tune` (`measure_fft_tuning()`) ou annuler detune/chorus excessif. |
| **`dynamic_arc < min dB`** | Alléger Intro/Pont (`Matrice 1.3`), `BUS Music comp Threshold >= -13.5`, `transitions_lib.section_gain_arc`. |

---

## Chapitre 4 — Playbook des Pièges Techniques Vérifiés (Ce qu'il faut retenir)

*(Les erreurs manuelles de Q `Band N width`, `gl.reverb (full)`, `gl.limiter Attack`, `sidechain`, `db_to_auto` et enveloppes `gl.*` sont désormais éliminées automatiquement par **`scripts/gl_dsl.py`**.)*

| Piège Vérifié | Conséquence | Règle Définitive |
| :--- | :--- | :--- |
| **`gl.multisampler` (`.sfz`) en GUI** | Silencieux (`-144 LUFS`). | Utiliser `gl.sampler` WAV 24-bit (`440 Hz`) ou `gl.*` natif. |
| **`patch mixerTrack` / `set_song bpm`** | Mauvais routage / tempo ignoré. | Utiliser `song_builder.py` (`edit_channel`, `set_song tempo`). |
| **`loudnorm` sur one-shots `< 2.5 s`** | Écrase transitoires Kick/Snare. | `clean_audio_sample` (peak `-1.0 dBFS`, fades `2/15 ms`). |
| **`pitch_shift` sur 808 `< 80 Hz`** | Dérive `-35 ct`, attaque molle. | Régler `Root note` + `Fine tune` dans `gl.sampler` (`+X` monte de `+X ct`). |
| **808 mitraillées / infra-sub `< 42 Hz`** | Pics saturation & `-13.5 LUFS`. | `1–2` notes tenues/mes. (`>= MIDI 31`, `36–42` en `C..F#`), `Release 50 ms`, `enforce_frequency_inertia`. |
| **Phase dégradée post-`pitch_shift`** | `low_mid_corr` chute (`0.09`). | `stabilize_stereo_low_mid(..., cutoff_hz=700)` après `pitch_shift`. |
| **Cloche `mono_oneshot` depuis FX** | Un FX rééchantillonné garde des lobes faux (`-15 ct`). | Varispeed sur le noyau tonal stable seul (`0.14–0.39 s`) + decay exponentiel. |
| **Glissando d'attaque / FX faux** | Fausse le diapason global (`-17 ct`). | Couper glissando d'attaque et accorder aussi les FX tonaux (`440 Hz`). |
| **`gl.sampler` signe `Fine tune`** | Offset inversé double l'erreur (`±18 ct`). | `+X` monte (`+X ct`), `-X` baisse (`-X ct`) ; utiliser `measure_fft_tuning()`. |
| **Espace Rap & Collage pré-limiteur** | Pistes vocales / transitoires bloquent `-13 LUFS`. | Zéro vox en instru Rap, creux `1–2.5 kHz` (`-3.5 dB`) sur `Chords/Pad`, `30–45 %` silences Lead. |
| **`add_automation` sans `"track"`** | Automation créée non placée (`0` clips). | Toujours passer `"track": <idx_playlist>` (`transitions_lib.py` le gère). |
| **Collision `_infer_roles()` MIDI** | `"Vocal"` fusionné avec `Lead`, `"Piano"` arpégé pris pour `Lead`. | Respecter la table Facette 3 §5.2 (`Bass`, `Chords Piano`, `Lead`, `Counter Harmony`). |

---

## Chapitre 5 — Journal de Références Compact (Capitalisation)

> **Règle** : Consigner chaque nouveau morceau validé sur **1 seule ligne** pour garder ce document léger en tokens.

| Projet | Genre / BPM / Key | Durée | LUFS / dBTP | Phase (`<150` / `200-800`) | PQ | Points Clés Validés |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `active6` *Airey Horizon* | Trap / — | `60 s` | `-11.83` / `-1.50` | OK | `7.12` | Intro `audit_project_sources`, `studio_lib`, `render_and_audit` |
| `active7` *Midnight Drift* | Trap / `150` / `Bbm` | `45 s` | `-10.71` / `-1.40` | `—` / `0.707` | **`8.08`** | Arc dynamique réel `3.8 dB` |
| `active8` *Simba d'Ea'alah* | VGM→Rap / `144` / `Em` | `107 s` | `-11.43` / `-1.82` | OK | `7.46` | 100 % stems `htdemucs_6s` |
| `active9` *Glass Horizon* | Trap / `172` / `Fm` | `181 s` | `-12.12` / `-1.13` | OK (`+1.0 ct`) | `7.23` | Format long 3 min |
| `roshi_serpent` *Pigalle* | Dark Trap / `136` / `E Phryg` | `121 s` | `-11.48` / `-1.38` | OK | `7.39` | Fix `gl.eq Peak=6`, Freesound API, `gl.sampler` WAV |
| `morceaux/rosh2` *Astral Pigalle* | Cloud/Dark / `136` / `Cm` | `120 s` | `-11.86` / `-1.21` | `1.00` / `0.943` | **`7.96`** | 100 % CC0 `mono_oneshot`, 0 glitch/résonance, arc `>4.5 dB` |
| `morceaux/ryan` *Tokyo Expressway 1993* | Cloud Phonk / `192` / `Am` | `121.7 s` | `-13.27` / `-2.35` | `0.95` / `0.825` | **`7.80`** | `lofi_chill`, `96` half-time, Lead Keys `C4-D5` LPF `2.6k`, Pente Rose OK |
| `morceaux/song1` *Ombre Phrygienne* | Dark Trap / `142` / `C Phryg` | `121.7 s` | `-12.14` / `-1.62` | `1.00` / `0.898` | **`8.05`** | Samples `morceaux/SAMPLES`, `440.25 Hz` (`+1 ct`), 0 glitch/résonance, arc `7.04 dB` |
| `morceaux/song3` *Sillage de Minuit (Rap Inst.)* | Dark Trap / `144` / `Gm` | `120.0 s` | `-12.05` / `-1.32` | `1.00` / `0.922` | **`7.47`** | Instru Rap sans vox (`song3/Samples/` + alts `samples/`), 808 longues `1-2/mes`, `440.0 Hz (0.0 ct)`, 0 glitch/résonance, arc `4.44 dB` |
| `morceaux/song4` *Ondes de Verre (Rap Inst.)* | Dark Trap / `172` / `Fm` | `178.6 s` | `-12.18` / `-1.29` | `1.00` / `0.990` | **`7.28`** | 100 % samples fournis (`air/Samples/` + 22 alts), 22 pistes isolées, `440.51 Hz (+2.0 ct)`, 0 glitch/résonance, arc `10.19 dB`, centralisation `sample_prep_lib` + `song_builder` |
| `morceaux/song5` *Brume d'Automne (Rap Inst.)* | Melancholic Trap / `134` / `Dm` | `122.4 s` | `-11.59` / `-1.08` | `0.999` / `0.730` | **`7.44`** | 100 % samples fournis (`Untitled/Samples/` + 13 alts) + `gl.*`, 15 pistes isolées, `440.0 Hz (0.0 ct)`, 0 glitch/résonance, pente `-4.44 dB/oct`, `spectral_match=0.894` |


---

## Phase 6 — Synchronisation $S1/S2$, Pyramide $H \times V$, Cache & Auto-Fix Déterministe

* **Micro-timing & Phase ($S1 / S2$)** :
  * `S1/V` (`analyze_kick_bass_transient_phase`) : détecte les inversions de polarité Kick/808 (`INVERT_808_POLARITY` via `sample_prep_lib.invert_phase=True` ou `invertPolarity` mixer) et les micro-décalages (`SHIFT_BASS_MICRO_MS` borné à `±10 ms`).
  * `S1/H` (`analyze_sustained_spectral_flux`) : signale les timbres soutenus figés (`STATIC_TIMBRE`).
  * `S2/H` (`analyze_audio_pocket_alignment`) : mesure le retard d'attaque audio vs MIDI à la milliseconde et rogne les silences pré-attaque (`TRIM_SAMPLE_START_MS` via `sample_prep_lib.trim_start_ms`).
  * `S2/V` (`analyze_transient_stacking`) : détecte `> 2` attaques non-percussives simultanées dans `1–4 kHz` / `LowMid` (`STAGGER_ATTACK_OR_STRUM` via `groove_lib.apply_micro_strum` ou enveloppe synthé `gl_dsl`).
* **Surcharge par contrat & empreinte finale (`scripts/render_and_audit.py`)** : `--contract <multi_scale_contract.json>` surcharge localement `target_lufs`, `low_mids_max`, `air_min`, `sharp_max` et compare `bark24`; tout `PASS` complet exécute `macro_structure_audit.py` et exporte `<projet>/renders/master_fingerprint.json`.
* **Workflow économe en tokens** :
  * Préférer `audio-analyzer:audit_multiscale_hv(project_dir, stage, contract_path)` ou le CLI one-shot `.venv/bin/python scripts/run_autonomous_pipeline.py --project <dir> --stage E1|E2|E3|E4|ALL --auto-fix --max-autofix-passes 2 --json-compact`.
  * Sortie strictement bornée par `scripts/multiscale_pyramid.py` (`<= 1 500` caractères, `<= 400` tokens, Top 3 déviations, blocage en cascade $S1/S2 \Rightarrow S4=S5=\text{null}$), détails complets écrits dans `renders/diagnostics/`.
  * Cache SHA-256 incrémental (`scripts/artifact_cache.py` dans `pipeline_state.json`) sautant instantanément pistes É1, paliers É2 (`V1..V4`) et coutures É3 inchangés ; dispatcher local `scripts/multiscale_autofix.py` exécuté avant toute escalade LLM.

---

## Phase 7 — Psychoacoustique (`P3`) & Mémoire Épisodique de Goût (`P4`)

* **3 Audits Psychoacoustiques `P3` (`deep_diagnostics.py`, `render_and_audit.py`)** :
  * **Masquage dynamique Bark (`analyze_dynamic_bark_masking`)** : mesure `SMR(t, b*)` sur les trames actives du `Lead` (> `-35 dBFS`) ; si `SMR < +1.5 dB` sur `> 25 %` des trames actives, déclenche `APPLY_DYNAMIC_MIRROR_EQ(freq_hz)` (`-3.5 dB`, `Q=1.8` sur `Chords/Pad` ou `BUS Music`).

  * **Plage Dynamique `LRA` (`compute_ebu_r128_stats` / `compute_lra_ebu`)** : contrôle EBU R128/Tech 3342 mono-passe (`edm/trap >= 2.5 LU`, `pop/rock/lofi >= 3.5 LU`, `jazz/classical >= 5.0 LU`) ; sous le seuil, `RELAX_BUS_COMPRESSION` relève `Threshold` `BUS Music` de `+2.5 dB` et ajuste `Gain` limiteur (`-1.5 dB`).
  * **Traduction Sub-Basse petits haut-parleurs (`analyze_bass_harmonic_translation`)** : sur basse `f0 < 65 Hz`, vérifie `R_trans = 10 log10(E[1.7 f0..3.4 f0] / E[0.8 f0..1.25 f0]) >= -14.0 dB` ; sinon `BOOST_BASS_HARMONICS(f0_hz)` active `gl.dist` Soft + boost `gl.eq +2.5 dB @ 2.5 f0`.

---

## Phase 8 — Équilibre Inter-Pistes, Espace 3D & Références Externes (`Q1..Q7`)

* **Auto-gain staging É1 (`gain_staging_lib.py`)** : `measure_active_loudness()` (K-weighting BS.1770, trames `50 ms/hop 25 ms`, gates `-45 dBFS` + `-20 dB` relatif), 9 rôles canoniques, `ROLE_TARGET_OFFSETS_DB[mix_profile]` (ancre `kick` ou `lead`), peak guard `-2.0 dBFS`, `recommended_volume_auto` et `renders/gain_staging_report.json`.
* **Matrice multi-stems bornée (`analyze_stem_hierarchy` v2)** : `lead_vs_harmony` `[3.0, 9.5]`, `bass_vs_kick` `[-3.5,+2.0]`, `snare_vs_kick` `[-4.5,+1.5]`, `hihat_vs_kick` `[-14,-5.5]`, `counter_vs_lead` `[-12,-3]`, `vocals_vs_lead` `[+1,+6]` dB ; toute déviation émet `ADJUST_TRACK_FADER` (`±4 dB/passe`, cumul `8 dB`). `audit_section_track_audibility()` ajoute les planchers `section_lufs -13/-18/-22 dB` (`TRACK_UNDER_AUDIBLE_FLOOR`).
* **Audit stéréo (`run_deep_audit`)** : `stereo_balance_db = |LUFS_L - LUFS_R| <= 1.2 dB` (`FIX_STEREO_BALANCE`) et `mid_high_side_to_mid_db` sur `500 Hz-8 kHz` (`[-14,-1.5]` rythmique, `[-18,-1.5]` jazz/classique) → `WIDEN_HARMONY_OR_PERC` / `NARROW_EXCESSIVE_SIDE`.
* **Références externes (`bootstrap_golden_refs.py --external-dir`)** : `golden_references/external_masters/<profil>/` (2–3 masters, audio **jamais versionné**), empreintes `external_pro_reference` (`bark24`, pente rose, LUFS, LRA, crest, largeur, ratios de bandes) ; `taste_memory.record_validated_taste()` refuse `PQ < 7.8` sauf `--approve-taste` ; fusion `target_fingerprint` plafonnée à `30 %` (ou `60 %` approuvé) de mémoire de goût.
* **Mémoire de Goût (`scripts/taste_memory.py` → `memory/user_taste_profile.json`)** :
  * `record_validated_taste()` mémorise automatiquement tout morceau `PASS` (hors dossiers temporaires, ou explicitement via `--approve-taste` poids `2.0`, max `50` entrées dédupliquées par `project_id`).
  * Si `reference_tracks` est vide au départ en É0, `target_fingerprint.py` interroge `query_taste_prior()` pondéré par distance euclidienne sur `(V, T, E, D)` **avant** le repli générique et inscrit `"source": "user_taste_memory"` dans `multi_scale_contract.json`.
