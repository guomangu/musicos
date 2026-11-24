# Phase 8 — Cahier des Charges Technique : Équilibre Inter-Pistes, Espace 3D, Réalisme Timbral & Finitions Musicales (`Q1..Q7`)

> **Destinataire** : Agent de développement autonome (`/home/gamo/Bureau/musicos/`)
> **Objectif** : Lever les carences restantes identifiées après la Phase 7 — en priorité les **écarts de volume entre pistes** (pistes inaudibles ou noyées dans le mix), puis le **manque de profondeur 3D**, la **boucle d'auto-référence spectrale**, le **rendu statique des timbres headless** et la **rigidité des transitions**.
> **Contraintes globales** :
>
> 1. **100 % rétrocompatible** avec les Phases 1 à 7 (`tests/test_gl_dsl.py` jusqu'à `tests/test_phase7_expressivity_autonomy.py` + `tests/regression_multi_genre.py` `7/7 PASS`).
>
> 2. **0 GPU / 0 PyTorch au runtime** pour tous les nouveaux calculs DSP et décisions d'auto-fix.
>
> 3. **Budget CPU** : surcharge totale Phase 8 $\le 1.5\text{ s}$ sur un morceau de 2 minutes.
>
> 4. **Budget Tokens** : la sortie compacte de `scripts/multiscale_pyramid.py` et `audio-analyzer:audit_multiscale_hv` reste strictement bornée à $\le 1500$ caractères ($\le 400$ tokens).

## 0. Plan d'Exécution Ordonné (Par Rentabilité & Dépendances)

Implémenter et valider les chantiers dans cet ordre exact (chaque jalon doit passer ses tests unitaires avant de passer au suivant) :

| Ordre | Chantier | Problème Ciblé | Fichiers Principaux | 
| ----- | ----- | ----- | ----- | 
| **1** | **`Q1` Auto-Gain Staging par Rôle (Sonie Active)** | Pistes inaudibles ou disproportionnées dès la source (peak norm vs loudness perçu, volumes codés en dur) | `scripts/gain_staging_lib.py` (nouveau), `scripts/soundcheck_lib.py`, `scripts/song_builder.py`, `scripts/gl_dsl.py` | 
| **2** | **`Q2` Matrice Multi-Stems Bornée & Auto-Fader (`ADJUST_TRACK_FADER`)** | `lead_vs_harmony_db >= 2.0` récompense une harmonie inaudible ; l'auto-fix ne touche jamais aux faders | `mcp-servers/audio-analyzer/deep_diagnostics.py`, `scripts/vertical_tester.py`, `scripts/macro_structure_audit.py`, `scripts/multiscale_autofix.py`, `scripts/render_and_audit.py` | 
| **3** | **`Q3` Bus FX 3D (`FX Delay 21`, `FX Hall 22`), Audit Stéréo & Garde-Fou Cascade LPF** | Un seul bus `FX Room` court (pas de vraie profondeur 3D), pas d'audit largeur/balance L-R, sur-filtrage LPF | `scripts/studio_lib.py`, `scripts/gl_dsl.py`, `scripts/song_builder.py`, `mcp-servers/audio-analyzer/deep_diagnostics.py`, `scripts/render_and_audit.py` | 
| **4** | **`Q4` Références Pro Externes & Garde-Fou `taste_memory`** | Boucle d'auto-référence sur les rendus internes | `scripts/bootstrap_golden_refs.py`, `scripts/taste_memory.py`, `scripts/target_fingerprint.py`, `golden_references/external_masters/README.md` | 
| **5** | **`Q5` Transitions Musicales Riches & Séparation de Registre (`V < 0`)** | Transitions mécaniques (absence de fills/risers/impacts) ; conflit registre `MIDI 58-67` en mode sombre | `scripts/transitions_lib.py`, `scripts/song_builder.py`, `mcp-servers/audio-analyzer/musicality_metrics.py`, `schemas/arrangement_plan.schema.json` | 
| **6** | **`Q6` Expressivité Timbrale Headless (`synth_voice` v2) & Sampler 3-Zones** | Timbres `gl.*` statiques en headless ; dégradation formantique des `mono_oneshot` transposés au-delà de $\pm 4$ st | `scripts/gl_dsl.py`, `scripts/sample_prep_lib.py`, `scripts/song_builder.py` | 
| **7** | **`Q7` Orchestrateur, Cache SHA-256, Tests & Docs** | Cohérence bout-en-bout, non-régression CI, mise à jour facettes `docs/` | `scripts/run_autonomous_pipeline.py`, `scripts/artifact_cache.py`, `tests/test_phase8_balance_space_expressivity.py`, `docs/00..05` | 

## Chantier `Q1` — Auto-Gain Staging Déterministe par Rôle (Sonie Active)

### 1.1 Diagnostic du Problème

Actuellement :

* Les one-shots $< 2.5\text{ s}$ sont normalisés au pic (`-1.0 dBFS`), ce qui crée des écarts de sonie perçue dépassant $15\text{ dB}$ entre une percussion/pluck à fort facteur de crête et une 808/pad dense.

* `ChannelSpec.volume` est codé en dur dans `build_song.py` sans compensation des écarts de gain natifs des synthés `gl.*` ni de la perte d'énergie provoquée par les LPF (`2.5-7.5 kHz`).

* `audit_track_soundcheck` (É1) valide toute piste dont `RMS > -42 dBFS`, laissant passer des pistes à $-33\text{ dBFS}$ qui deviennent totalement inaudibles dans un mix à $-11\text{ LUFS}$.

### 1.2 Nouveau Module : `scripts/gain_staging_lib.py`

Créer `scripts/gain_staging_lib.py` avec les fonctions et tables suivantes :

#### A. Mesure de Sonie Active (`measure_active_loudness`)

Ne jamais utiliser un RMS/LUFS intégré brut sur toute la durée d'un clip (une cloche ou un snare qui joue 2 fois par mesure aurait un LUFS artificiellement bas). Calculer la sonie **uniquement sur les trames actives** :

```
def measure_active_loudness(
    audio: np.ndarray,
    sr: int,
    frame_ms: float = 50.0,
    hop_ms: float = 25.0,
    abs_gate_dbfs: float = -45.0,
    rel_gate_db: float = -20.0,
) -> dict:
    """
    Retourne :
      - active_rms_db: float        # RMS K-weighted (ou filtre ITU-R BS.1770) sur les trames actives
      - active_lufs: float          # Estimation LUFS sur les trames actives (pyloudnorm ou K-weighting)
      - peak_dbfs: float            # Pic absolu
      - crest_factor_db: float      # peak_dbfs - active_rms_db
      - active_frame_ratio: float   # Proportion de trames actives dans [0, 1]
      - is_effectively_silent: bool # True si active_frame_ratio < 0.005 ou active_rms_db < -50.0
    """

```

Règle de sélection des trames actives :

1. Appliquer la pondération K (`ITU-R BS.1770` — pré-filtre shelf $+4\text{ dB}$ à $1681\text{ Hz}$ + HPF $38\text{ Hz}$) pour que le niveau reflète la sonie perçue (ne pas pénaliser une 808 sub ni surestimer un sub inaudible).

2. Calculer l'énergie RMS par fenêtre de $50\text{ ms}$ (hop $25\text{ ms}$).

3. Filtrer les fenêtres satisfaisant :
   

   $$
   E_{\text{frame, dBFS}} > \max(\text{abs\_gate\_dbfs},\; E_{\text{max\_frame, dBFS}} + \text{rel\_gate\_db})
   $$

4. Calculer la moyenne quadratique (en puissance, pas moyenne arithmétique de dB) sur les fenêtres retenues.

#### B. Mapping des Rôles Canoniques (`normalize_role_for_gain_staging`)

Mapper tout nom de canal / `ChannelSpec.role` vers l'un des 9 rôles canoniques :
`kick`, `bass_sub`, `snare_clap`, `hihat_perc`, `lead`, `chords_pad`, `counter_melody`, `ear_candy_fx`, `vocals`.

#### C. Table Normative des Cibles Relatives par `mix_profile` (`ROLE_TARGET_OFFSETS_DB`)

Chaque valeur est un triplet `(min_db, target_db, max_db)` exprimé en $\text{dB}$ **relatifs à l'ancre du profil** :

* Ancre si `has_drums == True` : **`kick` =** $0.0\text{ dB}$ (ou `lead` si aucun kick présent dans le projet).

* Ancre si `has_drums == False` (`classical_ambient` ou `--no-drums`) : **`lead` =** $0.0\text{ dB}$ (ou `chords_pad` si instrument solo sans lead).

* Niveau absolu de référence cible de l'ancre en pré-master (avant `BUS` compression / `Master` limiter) : `ANCHOR_PREMASTER_ACTIVE_LUFS = -16.0` (laisse $\ge 6\text{ dB}$ de headroom avant `Master`).

| `mix_profile` | `kick` | `bass_sub` | `snare_clap` | `hihat_perc` | `lead` | `chords_pad` | `counter_melody` | `ear_candy_fx` | `vocals` | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **`trap_rap`** | `(-1, 0, +1)` | `(-2.5, -1.0, +1.0)` | `(-3.5, -1.5, +0.5)` | `(-13, -10.0, -7.0)` | `(-3.0, -1.0, +1.5)` | `(-11.0, -8.0, -5.5)` | `(-12.0, -9.0, -6.0)` | `(-16, -12.0, -8.5)` | `(-1.0, +1.0, +3.0)` | 
| **`pop_rnb`** | `(-1, 0, +1)` | `(-4.0, -2.0, 0.0)` | `(-3.0, -1.5, +0.5)` | `(-13, -10.5, -7.5)` | `(-2.0, 0.0, +2.0)` | `(-9.5, -7.0, -4.5)` | `(-11.0, -8.0, -5.5)` | `(-15, -11.5, -8.0)` | `(-0.5, +1.5, +3.5)` | 
| **`edm_club`** | `(-1, 0, +1)` | `(-3.0, -1.5, +0.5)` | `(-4.0, -2.0, 0.0)` | `(-11, -8.5, -6.0)` | `(-2.5, -0.5, +1.5)` | `(-9.0, -6.5, -4.0)` | `(-11.0, -8.5, -6.0)` | `(-14, -10.5, -7.0)` | `(-1.5, +0.5, +2.5)` | 
| **`rock_metal`** | `(-1, 0, +1)` | `(-3.5, -2.0, 0.0)` | `(-2.0, -0.5, +1.5)` | `(-10, -7.5, -5.0)` | `(-1.5, +0.5, +2.5)` | `(-6.5, -4.5, -2.5)` | `(-9.5, -7.0, -4.5)` | `(-14, -11.0, -8.0)` | `(-0.5, +1.5, +3.5)` | 
| **`lofi_chill`** | `(-1, 0, +1)` | `(-3.5, -1.5, +0.5)` | `(-4.0, -2.0, 0.0)` | `(-14, -11.0, -8.0)` | `(-2.5, -0.5, +1.5)` | `(-8.0, -5.5, -3.5)` | `(-10.5, -8.0, -5.5)` | `(-16, -12.5, -9.0)` | `(-1.0, +1.0, +3.0)` | 
| **`jazz_acoustic`** | `(-1, 0, +1)` | `(-3.0, -1.0, +1.0)` | `(-4.5, -2.5, -0.5)` | `(-11, -8.5, -6.0)` | `(-1.0, +1.0, +3.0)` | `(-7.5, -5.0, -3.0)` | `(-9.5, -7.0, -4.5)` | `(-15, -12.0, -8.5)` | `(0.0, +2.0, +4.0)` | 
| **`classical_ambient`** *(ancre = `lead`)* | `(-8, -5, -2)` | `(-7.0, -4.5, -2.0)` | `(-10, -7, -4)` | `(-16, -13, -10)` | `(-1, 0, +1)` | `(-8.5, -5.5, -3.0)` | `(-10.0, -7.0, -4.5)` | `(-15, -11.5, -8.0)` | `(-0.5, +1.5, +3.5)` | 

#### D. Calcul du Trim et Calibration (`compute_gain_staging_plan`)

```
def compute_gain_staging_plan(
    track_measurements: dict[str, dict],  # {track_key: {role, active_lufs, peak_dbfs, current_volume}}
    mix_profile: str,
    has_drums: bool = True,
    anchor_target_lufs: float = -16.0,
    max_delta_db: float = 12.0,
) -> dict:
    """
    Calcule pour chaque piste :
      - target_active_lufs = anchor_target_lufs + target_offset_db
      - raw_delta_db = target_active_lufs - measured_active_lufs
      - clamped_delta_db = clamp(raw_delta_db, -max_delta_db, +max_delta_db)
      - peak_guard: réduire clamped_delta_db si peak_dbfs + clamped_delta_db > -2.0 dBFS
      - recommended_volume: float dans [0.10, 1.15] (converti via gl_dsl.db_to_auto)
      - status: "OK" | "BOOSTED" | "ATTENUATED" | "SOURCE_TOO_QUIET" | "CLIP_GUARD_ACTIVE"
    """

```

### 1.3 Intégration dans É1 (`soundcheck_lib.py`) & `song_builder.py`

1. **`scripts/soundcheck_lib.py`** :

   * Appeler `measure_active_loudness()` sur chaque WAV isolé d'É1.

   * Ajouter dans le JSON retourné : `active_rms_db`, `active_lufs`, `crest_factor_db`, `target_active_lufs`, `recommended_gain_db`, `recommended_volume_auto`.

   * Relever le seuil d'échec silencieux : émettre `FIX` (`BOOST_TRACK_GAIN` ou `CHECK_ROUTING_OR_SWAP`) si `is_effectively_silent == True` ou si `active_lufs < target_active_lufs - 10.0` (et que le pic ne permet pas de compenser sans dépasser $-1.0\text{ dBFS}$).

2. **`scripts/song_builder.py`** :

   * Ajouter le paramètre `auto_gain_stage: bool = True` dans `build_core_loop_project`, `build_gridline_project` et `run_song_cli` (désactivable par `--no-auto-gain-stage`).

   * Avant l'écriture finale des faders (ou via lecture de `<proj>/renders/diagnostics/soundcheck_metrics.json` / `<proj>/renders/gain_staging_report.json`), appliquer automatiquement `recommended_volume_auto` à chaque `ChannelSpec` concerné.

   * Écrire `<proj>/renders/gain_staging_report.json` avec les valeurs avant/après par piste.

## Chantier `Q2` — Matrice de Hiérarchie Multi-Stems Bornée & Auto-Fader (`ADJUST_TRACK_FADER`)

### 2.1 Diagnostic du Problème

* `analyze_stem_hierarchy` ne contrôle qu'une borne inférieure unilatérale (`lead_vs_harmony_db >= 2.0`), ce qui valide un mix où `Chords/Pad` est à $-25\text{ dB}$ sous le `Lead` (inaudible).

* `macro_structure_audit.py` vérifie uniquement le plafond supérieur (`> -28 dBFS`) pour élaguer les pistes excédentaires, jamais le plancher d'audibilité des pistes censées être actives dans une section.

* `multiscale_autofix.py` et `render_and_audit.py --auto-fix` corrigent EQ, Master Gain et Limiter Ceiling, mais ne peuvent jamais monter ou baisser une piste individuelle.

### 2.2 Spécifications : `analyze_stem_hierarchy` v2 (`deep_diagnostics.py`)

Étendre `analyze_stem_hierarchy(stems_dir, mix_profile="trap_rap", has_drums=True)` pour :

1. Mesurer `active_lufs` (via `gain_staging_lib.measure_active_loudness`) sur chaque stem rendu (`kick`, `bass`, `snare`, `hihat`, `chords`, `lead`, `counter`, `vocals`, etc.).

2. Evaluer des **fourchettes bilatérales** `[min_db, max_db]` dérivées directement de `ROLE_TARGET_OFFSETS_DB[mix_profile]` :

   * `lead_vs_harmony_db` : pour `trap_rap`, doit être dans `[+2.5, +9.5] dB` (ni masqué $< +2.5$, ni harmonie inaudible $> +9.5$).

   * `bass_vs_kick_db` *(si `has_drums`)* : dans `[-3.5, +2.0] dB`.

   * `snare_vs_kick_db` *(si `has_drums`)* : dans `[-4.5, +1.5] dB`.

   * `hihat_vs_kick_db` *(si `has_drums`)* : dans `[-14.0, -5.5] dB`.

   * `counter_vs_lead_db` *(si `counter` présent)* : dans `[-12.0, -3.0] dB`.

   * `vocals_vs_lead_db` *(si `vocals` présent)* : dans `[+1.0, +6.0] dB`.

3. Pour chaque déviation hors fourchette, générer une recommandation structurée :

   ```
   {
     "action": "ADJUST_TRACK_FADER",
     "target_role": "chords_pad",
     "target_track": "Chords",
     "measured_diff_db": 13.4,
     "expected_range_db": [2.5, 9.5],
     "delta_db": 4.0,
     "reason": "Harmonie trop faible (-13.4 dB sous Lead) ; relever de +4.0 dB."
   }
   
   ```

### 2.3 Spécifications : Plancher d'Audibilité par Section (`macro_structure_audit.py`)

Dans `scripts/macro_structure_audit.py`, en complément du plafond cognitif (`> -28 dBFS`), ajouter `audit_section_track_audibility()` :

* Pour chaque section déclarée dans `arrangement_plan.json` (ou `song_blueprint.json`), vérifier chaque piste appartenant à `active_roles` :

  * Rôles primaires (`kick`, `bass_sub`, `snare_clap`, `lead`, `vocals`) :
    

    $$
    \text{track\_active\_lufs}_{\text{section}} \ge \text{section\_mix\_lufs} - 13.0\text{ dB}
    $$

  * Rôles secondaires (`chords_pad`, `hihat_perc`, `counter_melody`) :
    

    $$
    \text{track\_active\_lufs}_{\text{section}} \ge \text{section\_mix\_lufs} - 18.0\text{ dB}
    $$

  * Habillage (`ear_candy_fx`) :
    

    $$
    \text{track\_active\_lufs}_{\text{section}} \ge \text{section\_mix\_lufs} - 22.0\text{ dB}
    $$

* En cas de violation sur une piste censée jouer dans la section, émettre le code d'action `ADJUST_TRACK_FADER` avec le `delta_db` borné à `[+1.5, +4.0] dB` (et signaler `TRACK_UNDER_AUDIBLE_FLOOR` dans les diagnostics).

### 2.4 Spécifications : Action Déterministe `ADJUST_TRACK_FADER` (`multiscale_autofix.py` & `render_and_audit.py`)

1. Ajouter le gestionnaire `ADJUST_TRACK_FADER` dans `scripts/multiscale_autofix.py` et `scripts/render_and_audit.py` (`--auto-fix`) :

   * **Paramètres** : `track_name` (ou `insert_idx`), `delta_db: float`.

   * **Sécurités** :

     * Borner chaque passe à $|\Delta\text{dB}| \le 4.0\text{ dB}$.

     * Cumul maximal toutes passes confondues : $|\sum \Delta\text{dB}| \le 8.0\text{ dB}$ (au-delà, escalader `SWAP_SAMPLE` ou `CHECK_SOURCE_GAIN`).

     * Respect strict de l'isolation É2 (`forbidden_tracks` dans `vertical_tester.py` : en palier $V_k$, seul le fader d'une piste appartenant à la couche $k$ peut être modifié ; en fin de `V4_LEAD` ou en `É4`, tous les faders de pistes/bus sont éligibles).

     * En mode déclaratif (`arrangement_plan.json` présent), persister `volume_trim_db` dans `arrangement_plan.json` (ou `gain_staging_overrides.json`) afin que les recompilations suivantes conservent la correction.

     * En mode patch `.gridline` direct (`render_and_audit.py --auto-fix`), mettre à jour le paramètre `volume` de l'insert mixeur via `GridlineMCP` + `gl_dsl.db_to_auto`, et journaliser dans `renders/autofix_history.json`.

## Chantier `Q3` — Espace 3D (`FX Delay 21`, `FX Hall 22`), Audit Stéréo & Garde-Fou LPF

### 3.1 Extension de la Topologie Buses (`studio_lib.py` & `gl_dsl.py`)

Conserver à l'identique les inserts `17..20` pour zéro rupture, et étendre `setup_standard_buses()` jusqu'à l'insert **`22`** :

| Insert | Nom du Bus | Chaîne Native `gl.*` | Paramétrage Déterministe (`BPM`, `mix_profile`, `valence`) | Rôles Autorisés en Send | 
| ----- | ----- | ----- | ----- | ----- | 
| **`17`** | `BUS Drums` | *(inchangé)* | *(inchangé)* | Kick, Snare, Hats, Percs | 
| **`18`** | `BUS Bass` | *(inchangé)* | *(inchangé)* | 808, Bass, Sub | 
| **`19`** | `BUS Music` | *(inchangé + garde-fou LPF §3.3)* | *(inchangé)* | Chords, Pad, Lead, Counter | 
| **`20`** | `FX Room` | *(inchangé)* | Réverb courte cohésion (`0.20-0.42 s` rythmique / `0.7-1.1 s` acoustique) | Snare, Chords, Lead, Counter (jamais Kick/Sub/Hats) | 
| **`21`** | **`FX Delay`** *(nouveau)* | `gl.delay` (ou plugin delay natif vérifié dans `list_plugin_catalog`) -> `gl.eq` | Temps synchro BPM : croche pointée ($0.75 \times \frac{60}{\text{BPM}}$) ou croche ($0.5 \times \frac{60}{\text{BPM}}$) ; Feedback $20\text{-}32\%$ ; `Dry = 0` ; `gl.eq` : `HPF 400 Hz`, `LPF 3500 Hz` (plus sombre $2800\text{ Hz}$ si $V < -0.2$) | Lead, Pluck, Counter, Vocals (`delay_send = 0.08..0.18`). **Interdit** sur Kick, Sub, Basse, Hats | 
| **`22`** | **`FX Hall`** *(nouveau)* | `gl.reverb` -> `gl.eq` | `Decay` $1.6\text{-}2.4\text{ s}$ (`1.8 s` par défaut, `2.3 s` en `classical_ambient`), `Dry = 0`, `Dry (full) = 0`, `HPF 500 Hz`, `LPF 5500 Hz` (`4500 Hz` si $V < -0.2$) | Pads, Strings, Ear Candy, Impacts/Risers (`hall_send = 0.14..0.30`). **Interdit** sur Kick, Sub, Basse, Snare sec | 

> **Note d'implémentation** : avant de coder `delay_send_fx()` dans `scripts/gl_dsl.py`, interroger `gridline` (`list_plugin_catalog` + `get_parameters`) pour vérifier le nom exact du plugin delay natif Gridline `0.7.7` et les unités exactes de ses paramètres (ms vs beats vs normalisé). Écrire un test unitaire dédié dans `tests/test_gl_dsl.py` / `tests/test_phase8_balance_space_expressivity.py`.

Extensions de `ChannelSpec` (`scripts/song_builder.py`) :

* Ajouter `delay_send: float = 0.0` et `hall_send: float = 0.0` (valeurs par défaut `0.0` pour rétrocompatibilité totale).

* Si `delay_send > 0`, router le send vers l'insert `21` (`FX Delay`) ; si `hall_send > 0`, vers l'insert `22` (`FX Hall`).

### 3.2 Audit Stéréo : Balance L/R & Largeur Médiums/Aigus (`deep_diagnostics.py` & `render_and_audit.py`)

Ajouter deux métriques déterministes dans `run_deep_audit()` (`mcp-servers/audio-analyzer/deep_diagnostics.py`) :

1. **Balance Gauche/Droite (`stereo_balance_db`)** :
   

   $$
   \Delta_{\text{LR}} = |\text{LUFS}_L - \text{LUFS}_R|
   $$

   * Seuil Quality Gate : $\Delta_{\text{LR}} \le 1.2\text{ dB}$ (tous profils).

   * Échec -> code `FIX_STEREO_BALANCE` (rééquilibre panoramique des pistes décentrées).

2. **Largeur Stéréo Médiums/Aigus (`mid_high_side_to_mid_db`)** sur la bande $500\text{ Hz} - 8\text{ kHz}$ :
   

   $$
   W_{\text{SM}} = 10 \log_{10}\left(\frac{E_{\text{Side}}[500\text{-}8000\text{ Hz}] + \epsilon}{E_{\text{Mid}}[500\text{-}8000\text{ Hz}] + \epsilon}\right)
   $$

   * Seuils Quality Gate (planche minimale pour éviter les mix « mono plats ») :

     * `trap_rap`, `pop_rnb`, `edm_club`, `rock_metal`, `lofi_chill` : $W_{\text{SM}} \in [-14.0,\; -1.5]\text{ dB}$.

     * `jazz_acoustic`, `classical_ambient` : $W_{\text{SM}} \in [-18.0,\; -1.5]\text{ dB}$.

   * Si $W_{\text{SM}}$ est trop bas (mix trop étroit) -> code `WIDEN_HARMONY_OR_PERC` (appliquer panoramiques complémentaires $\pm 20\text{-}30\%$ sur hats/percs/counter et élargir `Chords/Pad` ou relever `hall_send`/`delay_send` de $+0.06$).

   * Si $W_{\text{SM}} > -1.5\text{ dB}$ (anti-phase excessive) -> code `NARROW_EXCESSIVE_SIDE`.

### 3.3 Garde-Fou Anti-Sur-Filtrage (Budget Cascade LPF)

Pour éviter l'empilement destructeur de filtres passe-bas (`synth LPF` + `insert LPF` + `BUS Music LPF` + `Master LPF`) qui étouffe les mix sombres :

* Dans `studio_lib.setup_standard_buses()`, lorsque `profile` est `trap_rap` ou `pop_rnb` avec `valence < -0.2`, fixer le LPF de `BUS Music` à $\ge 11.5\text{ kHz}$ (et ne pas cumuler en plus un creux systématique $-2\text{ dB @ } 3.2\text{ kHz}$ si les canaux `Chords` et `Lead` disposent déjà de leur propre LPF `synth_voice`).

* Dans `gl_dsl.synth_voice()`, ajouter un champ `makeup_gain_db` dans `SynthVoiceConfig` compensant automatiquement la perte de sonie due au filtrage passe-bas :

  * `cutoff_hz < 3500`: `+2.5 dB`

  * `3500 <= cutoff_hz < 5500`: `+1.5 dB`

  * `cutoff_hz >= 5500`: `0.0 dB`

## Chantier `Q4` — Références Spectrales Externes & Garde-Fou `taste_memory`

### 4.1 Casser la Boucle d'Auto-Référence (`bootstrap_golden_refs.py`)

1. Créer le répertoire `golden_references/external_masters/` avec un `README.md` documentant l'arborescence attendue :

   ```
   golden_references/external_masters/
     trap_rap/          # Déposer 2-3 masters commerciaux WAV/FLAC/MP3
     pop_rnb/
     edm_club/
     rock_metal/
     lofi_chill/
     jazz_acoustic/
     classical_ambient/
   
   ```

   Ajouter `golden_references/external_masters/*/*.{wav,mp3,flac,ogg}` dans `.gitignore` pour ne jamais versionner d'audio commercial protégé, tout en versionnant uniquement les empreintes statistiques anonymes dans `golden_references/spectral_profiles.json`.

2. Étendre `scripts/bootstrap_golden_refs.py` avec `--external-dir golden_references/external_masters` :

   * Pour chaque profil disposant de fichiers dans `external_masters/<profile>/`, extraire : `bark24_mean`, `bark24_std`, `pink_slope_db_per_oct`, `lufs_integrated`, `lra_lu`, `crest_factor_db`, `mid_high_side_to_mid_db`, `band_ratios` (`sub`, `bass`, `low_mid`, `high_mid`, `air`), et marquer `"provenance": "external_pro_reference"`.

   * En l'absence de fichiers externes pour un profil, conserver le repli actuel marqué `"provenance": "internal_bootstrap"` et émettre un avertissement non bloquant dans le rapport.

### 4.2 Garde-Fou Qualité dans `scripts/taste_memory.py` & `scripts/target_fingerprint.py`

* Dans `record_validated_taste()` :

  * Ne pas enregistrer automatiquement un morceau `PASS` dont le score `PQ` est médiocre : exiger `pq_score >= 7.8` (ou absence de score `PQ` uniquement si explicitement validé via `--approve-taste`).

* Dans `target_fingerprint.py` :

  * Lorsque `user_taste_memory` est utilisé et qu'un profil `"external_pro_reference"` existe pour le `mix_profile`, fusionner les deux avec un plafond de dérive : au plus $30\%$ de poids pour la mémoire automatique (ou $60\%$ pour une entrée `--approve-taste`), et au moins $70\%$ (respectivement $40\%$) pour l'ancre `"external_pro_reference"`.

## Chantier `Q5` — Transitions Musicales Riches & Assouplissement Créatif

### 5.1 Enrichissement de `scripts/transitions_lib.py`

Ajouter 3 générateurs déterministes (100 % beats 0-indexed, 0 sample externe obligatoire, compatibles headless) :

1. **`generate_drum_fill(tracks, transition_beat, bpb=4.0, profile="trap_rap", style="auto", bpm=140.0)`** :

   * Insère sur les $2$ derniers temps avant `transition_beat` (ex. fin de mesure 8 ou 16 avant un Drop/Refrain) un fill idiomatique :

     * `trap_rap` / `edm_club` : snare/clap roll accéléré ($1/8 \to 1/16 \to 1/32$) avec crescendo de vélocité ($55 \to 112$) + légère descente de pitch sur les 4 derniers coups (`-1, -2, -3, -4 st` si sampler) + coupe du kick sur le dernier demi-temps.

     * `pop_rnb` / `rock_metal` / `lofi_chill` : descente de toms / snare syncopé avec ghost notes + accent final sur le temps $4.5$.

     * `jazz_acoustic` : pickup brush/rimshot ternaire discret.

     * `classical_ambient` : no-op (pas de drum fill).

2. **`add_transition_fx(tracks, automations, transition_beat, bpb=4.0, kind="riser_impact", fx_track_idx=None)`** :

   * Sur les $4$ temps précédant `transition_beat`, génère un swell/riser (note longue sur un canal noise/pad dédié avec automation montante `Band 7 freq` / `Cutoff` de $300\text{ Hz} \to 9000\text{ Hz}$ et volume croissant), suivi d'un silence de respiration de $0.25$ beat juste avant le temps 1 du Drop, puis un impact/crash filtré sur `transition_beat` envoyé vers `FX Hall (22)`.

3. **`apply_harmonic_anticipation(chords_notes, bass_notes, bar_boundaries, offset_beats=0.5, ratio=0.35)`** :

   * Sur les pivots de phrase (ex. passage vers la mesure 3 ou 5 d'une section), anticipe l'accord et la basse d'une croche ($0.5$ beat) en raccourcissant proprement la note précédente pour éviter tout chevauchement (zéro collision `b2`).

Intégrer ces 3 primitives comme types déclarables dans `schemas/arrangement_plan.schema.json` (`"transitions": [{"bar": 9, "type": "drum_fill_and_riser", "style": "snare_roll"}]`) et dans `song_builder.compile_arrangement_plan()`.

### 5.2 Séparation Automatique de Registre en Mode Sombre (`V < 0`)

Pour éliminer le conflit fréquentiel entre `Chords` (`MIDI 48-67`) et `Lead` (`MIDI 58-67`) lorsque `valence < -0.2` :

* Dans `song_builder.py` (et `optimize_voice_leading` appelé depuis `compile_arrangement_plan`), adopter des bornes adaptatives selon la valence :

  * Si $V \ge -0.2$ : `Chords` dans `[48, 65]`, `Lead` dans `[60, 74]`.

  * Si $V < -0.2$ : `Chords` resserré dans **`[48, 60]`** (chaleur grave-médium sans envahir le médium du Lead), `Lead` dans **`[60, 72]`** (au moins $+5$ demi-tons d'écart moyen entre centroïde MIDI des accords et centroïde MIDI du Lead).

* Ajouter une métrique informative `chords_lead_register_gap_st` dans `musicality_metrics.py` (`warning` si $< 4.0\text{ st}$ lorsque les deux jouent simultanément).

### 5.3 Assouplissement Expressif des Seuils Mélodiques (`musicality_metrics.py`)

Dans `MusicalityThresholds` :

* Autoriser jusqu'à **2 sauts expressifs** ($\ge 7$ demi-tons, ex. saut d'octave ou de sixte sur le hook/climax) par bloc de 8 mesures sans pénaliser `stepwise_ratio`, à condition qu'ils soient suivis d'un retour conjoint (*gap-fill*) ou d'une respiration ($\ge 0.5$ beat).

* Autoriser les **notes anticipées** (une croche avant le temps fort) à être évaluées sur l'accord de la mesure suivante pour `strong_beat_chord_tone_ratio` (évite les faux négatifs lorsque `apply_harmonic_anticipation` est actif).

## Chantier `Q6` — Expressivité Timbrale Headless (`synth_voice` v2) & Sampler 3-Zones

### 6.1 `gl_dsl.synth_voice` v2 (Mouvement Organique Natif `gl.*`)

Avant implémentation, inspecter les paramètres exacts exposés par les 13 synthés de `SYNTH_REGISTRY` en headless via `gridline` (`get_parameters`) afin de ne déclarer que des noms de paramètres 100 % vérifiés.

Ajouter un paramètre `expressive: bool = True` à `synth_voice(plugin, role, valence, cutoff_hz, expressive=True)` :

* **`pad` / `chords`** : activer un léger mouvement lent (LFO filtre $0.15\text{-}0.35\text{ Hz}$ à faible profondeur, ou léger detune/unison $\le 2$, ou sensibilité vélocité -> cutoff) afin de garantir un flux spectral vivant (zéro alerte `STATIC_TIMBRE` en $S1/H$).

* **`lead`** : activer un léger vibrato musical ($4.8\text{-}5.6\text{ Hz}$, profondeur subtile $\le 8\text{ ct}$ pour préserver `|tuning_offset| <= 6.5 ct`) + sensibilité vélocité -> filtre/amplitude.

* **`pluck` / `bell`** : forte sensibilité vélocité -> cutoff (les notes douces sonnent plus feutrées, les accents plus brillants).

* **`bass`** : zéro detune stéréo, zéro vibrato, phase stricte, mais légère coloration harmonique (`drive` doux si disponible) pour soutenir la traduction petits haut-parleurs (`P3`).

### 6.2 Sampler Multi-Zones (`3-Zone Pitch-Split`) pour Instruments Acoustiques

Pour contourner le bug `gl.multisampler` (`.sfz` muet en GUI) sans subir l'effet « chipmunk » d'un seul WAV transposé sur 2 octaves :

1. Dans `scripts/sample_prep_lib.py`, ajouter `build_multizone_samples(src_wav, out_dir, base_midi, zones=[(48, 55), (56, 63), (64, 72)])` :

   * Génère 3 WAV pré-transposés (ou sélectionne 3 samples distincts s'ils existent dans `Samples/`) avec préservation formantique (`librosa` / phase vocoder doux + `stabilize_stereo_low_mid` + accordage vérifié `measure_fft_tuning()` $\le 3\text{ ct}$), de sorte qu'aucune note jouée ne dépasse $\pm 4$ demi-tons de son sample de zone.

2. Dans `scripts/song_builder.py`, ajouter un helper `split_notes_by_pitch_zones(notes, zones)` permettant d'alimenter proprement les sous-canaux de zone tout en les routant vers le **même insert mixeur** (préserve la simplicité du mixeur et la cohérence timbrale sur toute la tessiture).

## Chantier `Q7` — Intégration Pipeline, Cache, Tests & Mise à Jour Documentaire

### 7.1 Cache SHA-256 & Orchestrateur (`artifact_cache.py`, `run_autonomous_pipeline.py`)

* Inclure `gain_staging_overrides`, `delay_send`, `hall_send` et `expressive` dans les clés de hachage `scripts/artifact_cache.py` pour qu'une retouche `ADJUST_TRACK_FADER` ou un ajout de send FX invalide proprement le palier concerné sans casser le cache des autres pistes.

* Mettre à jour `scripts/multiscale_pyramid.py` pour intégrer les nouveaux codes (`ADJUST_TRACK_FADER`, `FIX_STEREO_BALANCE`, `WIDEN_HARMONY_OR_PERC`) dans le Top 3 des déviations tout en maintenant la borne stricte $\le 1500$ caractères.

### 7.2 Suite de Tests Obligatoire (`tests/test_phase8_balance_space_expressivity.py`)

Créer `tests/test_phase8_balance_space_expressivity.py` couvrant au minimum :

1. **`test_q1_active_loudness_and_gain_staging`** : vérifier qu'un pluck court à fort facteur de crête (`-1.0 dBFS` peak, `-26 LUFS` actif) et une 808 saturée (`-1.0 dBFS` peak, `-10 LUFS` actif) sont calibrés à moins de $\pm 1.0\text{ dB}$ de leur cible relative `trap_rap` après `compute_gain_staging_plan()`, sans dépasser le peak guard (`-2.0 dBFS`).

2. **`test_q2_bounded_stem_hierarchy_and_autofader`** : vérifier qu'un stem `Chords` artificiellement atténué de $-18\text{ dB}$ échoue `analyze_stem_hierarchy` (alors qu'il passait auparavant !), déclenche `ADJUST_TRACK_FADER` borné à $+4.0\text{ dB}$/passe, et converge en $\le 2$ passes ; vérifier également `audit_section_track_audibility` et le respect strict de `forbidden_tracks` en É2.

3. **`test_q3_fx_buses_and_stereo_audits`** : vérifier la création headless des bus `17..22` (`FX Delay 21`, `FX Hall 22`) avec `Dry = 0`, l'audit de balance L/R (un signal décalé de $+3\text{ dB}$ à gauche échoue `FIX_STEREO_BALANCE`), et l'audit de largeur médiums/aigus (un signal strictement mono échoue `WIDEN_HARMONY_OR_PERC`).

4. **`test_q4_external_refs_and_taste_guard`** : vérifier le chargement prioritaire d'un profil `"external_pro_reference"` et le filtrage de `record_validated_taste()` lorsque `pq_score < 7.8` (sauf `--approve-taste`).

5. **`test_q5_musical_transitions_and_dark_register_split`** : vérifier `generate_drum_fill`, `add_transition_fx`, `apply_harmonic_anticipation` (zéro note négative, zéro overlap interne, zéro clash `b2`) et la séparation de registre `Chords [48..60]` / `Lead [60..72]` en $V = -0.7$.

6. **`test_q6_expressive_synth_and_multizone`** : vérifier que `synth_voice(..., expressive=True)` produit uniquement des paramètres valides acceptés par Gridline headless (0 erreur MCP) et que `split_notes_by_pitch_zones` garde chaque note dans $\pm 4$ st de sa zone.

7. **`test_q7_performance_and_token_budget`** : vérifier que la surcharge CPU Phase 8 reste $\le 1.5\text{ s}$ sur 2 minutes d'audio synthétique et que le résumé `multiscale_pyramid` reste $\le 1500$ caractères.

### 7.3 Validation Finale & Non-Régression (Critères d'Acceptation)

L'agent ne considère la Phase 8 terminée que lorsque **toutes** les commandes suivantes retournent `0` (`PASS`) :

```
cd /home/gamo/Bureau/musicos
.venv/bin/python -m unittest tests/test_gl_dsl.py -v
.venv/bin/python -m unittest tests/test_phase2_memory_and_midi.py -v
.venv/bin/python -m unittest tests/test_phase3_autofix_and_stems.py -v
.venv/bin/python -m unittest tests/test_phase4_expressivity.py -v
.venv/bin/python -m unittest tests/test_phase5_loop_first.py -v
.venv/bin/python -m unittest tests/test_phase6_multiscale_hv.py -v
.venv/bin/python -m unittest tests/test_phase7_expressivity_autonomy.py -v
.venv/bin/python -m unittest tests/test_phase8_balance_space_expressivity.py -v
.venv/bin/python tests/regression_multi_genre.py -v

```

### 7.4 Mises à Jour Documentaires Requises (Concis, Aligné sur les Facettes)

* `docs/00_SOMMAIRE.md` : ajouter la section **Phase 8 (`Q1..Q7`)** et l'entrée `scripts/gain_staging_lib.py` + `tests/test_phase8_balance_space_expressivity.py`.

* `docs/01_SYSTEME_STACK_ET_GUIDE.md` : documenter `delay_send_fx`, `reverb_hall`, `synth_voice(..., expressive=True)` et les bus `21..22`.

* `docs/02_COMPOSITION_EMOTIONNELLE_ET_THEORIE.md` : mettre à jour les registres adaptatifs `V < -0.2` (`Chords 48-60`, `Lead 60-72`).

* `docs/03_ARCHITECTURE_BLUEPRINT_ET_COHESION.md` : documenter `generate_drum_fill`, `add_transition_fx`, `apply_harmonic_anticipation` et les nouveaux types de transitions dans `arrangement_plan.json`.

* `docs/05_MIXAGE_QUALITY_GATE_ET_APPRENTISSAGE.md` : documenter la topologie `17..22`, l'auto-gain staging É1, la matrice multi-stems bornée, `ADJUST_TRACK_FADER`, l'audit stéréo (`stereo_balance_db`, `mid_high_side_to_mid_db`) et `golden_references/external_masters/`.