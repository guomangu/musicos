# Facette 5 — Mixage Multi-Genres, Topologie Buses, Quality Gate, Table Symptôme→Fix & Playbook

> **Cheatsheet IA (L'essentiel en 14 lignes)**
> - **Mixage adapté au genre** : Toujours passer **`--profile <mix_profile>`** et **`--beats-per-bar <num>`** à `scripts/render_and_audit.py` (et `profile=...` à `studio_lib.setup_standard_buses`). Ne jamais forcer des seuils Trap sur du Lo-Fi, Jazz, Classique ou Ambient !
> - **Boucle rapide CPU** : (1) Itérer sur un extrait du Refrain/Climax avec `--range <b1>-<b2> --no-neural` (~5× plus rapide, 0 PyTorch), (2) corriger en **1 passe** via la **Table Symptôme→Fix (§3.2)**, (3) lancer **1 seul rendu complet** `--mp3 --sections "..."` (avec score Audiobox PQ) et exiger `quality_gate.passed == true`.
> - **Topologie Buses** : `BUS Drums (17)` · `BUS Bass (18, mono <150 Hz)` · `BUS Music (19, gl.dist Slot 1 -> gl.eq Slot 2, Threshold -13.5 dB pour préserver l'arc)` · `FX Room (20, Dry=0, Dry full=0, Decay full en décisecondes)` · `Master 0 (gl.limiter Attack = 0.002 s, Ceiling <= -2.1 dB, safety=1)`.
> - **Règles de confort d'écoute** : `0` sample clippé · `true_peak <= -1.0 dBTP` · `0` micro-glitch · `0` résonance 1/3 oct · `tuning <= ±6.5 ct` · phase grave `>= 0.50` et bas-médiums `>= 0.55` · début sans clic et fin résolue/fade-out propre.

---

## Chapitre 1 — Profils de Mix & Quality Gate Multi-Genres (`--profile`)

Les seuils de `scripts/render_and_audit.py` et `studio_lib.setup_standard_buses()` s'adaptent au profil choisi dans `song_blueprint.json` :

| `mix_profile` | Genres Couverts | Cible LUFS | `low_mids` max | `air` min *(adaptatif `V < -0.2`)* | Sharpness max | Arc Dynamique min | Caractère des Buses (`setup_standard_buses`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`trap_rap`** | Trap, Cloud, Drill, Rap moderne | `-11.0` (`-12..-9.8`) | `<= 24 %` | `>= 3.0 %` *(sombre : **`>= 1.2 %`**)* | `< 1.90` *(sombre `< 1.85`)* | `>= 2.5 dB` | Sidechain Kick→808 `5:1`, `gl.dist` Soft `8 %` (`Tone <= 7.5k`) avant `gl.eq` LPF, `FX Room` court |
| **`pop_rnb`** | Pop, Synthwave, R&B, Afrobeat, Reggaeton | `-10.5` (`-11.5..-9.5`) | `<= 26 %` | `>= 3.0 %` *(mélancolique : **`>= 1.5 %`**)* | `< 1.95` *(sombre `< 1.85`)* | `>= 2.5 dB` | Sidechain modéré `3:1`, `BUS Music` équilibré, poche voix/lead `1.5–2.5 kHz` |
| **`edm_club`** | House, Techno, EDM, DnB, Rage | `-9.0` (`-10.0..-7.8`) | `<= 24 %` | `>= 3.5 %` *(sombre : `>= 1.75 %`)* | `< 2.05` *(sombre `< 1.90`)* | `>= 2.5 dB` | Sidechain punchy `5:1` sur basse & pads, air contrôlé, sub mono verrouillé |
| **`rock_metal`** | Rock, Indie, Punk, Metal | `-10.0` (`-11.0..-8.5`) | `<= 28 %` | `>= 2.5 %` *(sombre : `>= 1.25 %`)* | `< 2.10` *(sombre `< 1.90`)* | `>= 2.0 dB` | Sidechain léger `3:1`, médiums guitares riches, batterie organique punchy |
| **`lofi_chill`** | Lo-Fi, Boom Bap, Chillhop, Downtempo | `-14.0` (`-15.0..-12.5`) | `<= 34 %` | `>= 0.5 %` *(sombre : `>= 0.25 %`)* | `< 1.75` | `>= 1.8 dB` | LPF chaud `8.5–9.5 kHz` sur `BUS Music`, saturation douce `10 %`, 0 boost air agressif |
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

## Chapitre 2 — Topologie des Buses (`studio_lib.setup_standard_buses`)

`setup_standard_buses(mcp, bpm=..., profile=...)` crée automatiquement les pistes mixeur manquantes (au-delà de `0..16`) et configure :
1. **`BUS Drums` (Insert `17` → `Master 0`)** : `gl.eq` (`HPF 34 Hz`, shelf `10 kHz` selon profil) + `gl.comp` doux (`Ratio 2.2`).
2. **`BUS Bass` (Insert `18` → `Master 0`)** : `gl.stereo` (`Width = 0.0`) + `gl.comp` sidechain (`Detector input = 1`, alimenté par Kick `sidechain=True, preFader=True`) + `gl.eq` (`HPF 30 Hz`). *(Ajouter `gl.dist Soft` uniquement si besoin d'audibilité sur petits HP en Trap/EDM/Rock).*
3. **`BUS Music` (Insert `19` → `Master 0`)** :
   * **Slot 1 `gl.dist`** (`Mode = 0 Soft`, `Mix = 0.0` en acoustique/classique, `0.06–0.10` max en moderne/lo-fi, `Tone <= 7500 Hz`, `Remove DC = 1`, `Oversampling = 2`).
   * **Slot 2 `gl.eq`** (`HPF 90–135 Hz`, **`LPF 8.5–13.5 kHz`** selon profil/valence, + optionnel `Band type = 6 Peak` `-2.0 dB @ 3200 Hz Q=1.4` si le mix reste brillant) → coupe proprement toutes les harmoniques générées par la saturation.
   * **Slot 3 `gl.comp`** Glue (**`Threshold = -13.5 dB`**, `Ratio 1.8`, `Makeup 1.5` — ne jamais descendre à `-19 dB` sous peine d'écraser l'arc Intro/Refrain).
   * **Slot 4 `gl.comp`** micro-ducking Kick (`Detector input = 1`, `Ratio 1.2–2.2`).
4. **`FX Room` (Insert `20` → `Master 0`)** : `gl.reverb` (`Model = 1`, `Dry = 0`, `Dry (full) = 0`, `Low cut (full) = 350..480 Hz`, `Decay (full)` en **décisecondes**). Sends : mélodiques `8–20 %`, Snare `3–8 %`, **Kick/Sub/Hats `0 %`**. Couper par automation à `0` sur les temps d'arrêt/pivots.
5. **`Master 0`** : `gl.eq` (`Quality = 1`, `HPF 28 Hz`, `LPF 17 kHz`) → `gl.comp` (`Ratio 1.6`, `Threshold -10.5..-8.5 dB`) → `gl.limiter` (**`Attack (look-ahead) = 0.002 s`**, `Ceiling = -2.6..-2.1 dB`, `0.2 dB safety = 1`).

---

## Chapitre 3 — Quality Gate Bloquant & Table Symptôme → Action Unique Exacte

### 3.1 Les 12 Critères `render_and_audit.py` & Checklists Pré-Rendu (Modules 1 & 2)
* **Checklist Structure & Mix (Module 1)** : (1) **Spectre** `100 % Mono < 120 Hz` · (2) **Rythme** aucune note `< 100 Hz` de durée `< 1/8` (sauf `Release < 50 ms`) · (3) **Densité** `1 à 8` pistes actives simultanées max · (4) **Hiérarchie** 1 seul Lead actif dominant l'harmonie de `>= +3 dB` · (5) **Structure** aucune boucle de 8 mesures répétée à l'identique · (6) **Isolation « 1 sonorité = 1 piste »** chaque sonorité dispose de son propre canal, insert mixeur dédié, pattern dédié et piste Playlist nommée.
* **Checklist Timbrale & Micro-Timing (Module 2)** : (1) **Anti-agressivité** aucun pic étroit isolé `> +4 dB` dans `2,5–4,5 kHz` · (2) **Enveloppes aiguës** attaque `>= 1 ms` et aucune queue chevauchant la frappe suivante (`T_decay < 0,8 × T_note`) · (3) **Variation** écart-type vélocité `> 0` sur tout motif rapide (`>= 1/16`) · (4) **Pente rose** FFT refrain entre `-3` et `-5 dB/oct` sur `60 Hz–12 kHz` · (5) **Séparation transitoires** au moins 2 familles d'instruments (ex. Kick/Basse ou Snare/Hi-hat) avec décalage `>= 5 ms` sur les temps forts.

1. `clipped_samples == 0` · 2. `true_peak_dbtp <= -1.0` · 3. `|lufs - target_lufs| <= 1.2` · 4. `low_end_corr >= 0.50` (optimal `>= 0.75`) · 5. `low_mid_corr >= 0.55` (optimal `>= 0.65`) · 6. `sharpness < profil_max` · 7. `micro_glitches == 0` · 8. `resonances == []` · 9. `low_mids <= profil_max` · 10. `air >= profil_min` · 11. `|tuning_offset| <= 6.5 ct` · 12. `dynamic_arc >= profil_min`.

### 3.2 Table de Décision Déterministe (Corriger en 1 Passe, Zéro Tâtonnement !)

| Symptôme (`quality_gate.failures`) | Cause Probable | Action Unique Exacte dans Gridline / Python |
| :--- | :--- | :--- |
| **`clipped_samples > 0` ou `true_peak > -1.0`** | Gain limiteur trop fort ou Attack trop lent | `Master slot 3 (gl.limiter)` : vérifier `Attack (look-ahead) = 0.002`, `0.2 dB safety = 1`, baisser `Ceiling` de `-0.4 dB` (vers `-2.35..-2.6 dB`). |
| **`lufs` trop bas (`< target - 1.2`)** | Gain limiteur insuffisant ou mix trop faible | Augmenter `Master slot 3 Gain` de `+Δ` dB (`Δ = target - lufs`, plafonné à `+4 dB`/passe) sans dépasser `-1.2 dBTP`. |
| **`lufs` trop haut (`> target + 1.2`)** | Mix surcompressé ou Gain limiteur trop haut | Baisser `Master slot 3 Gain` de `Δ` dB et vérifier que `BUS Music` ne clippe pas. |
| **`low_end_corr < 0.50`** | Basse/Kick stéréo ou réverbe sur grave | Vérifier `BUS Bass` `gl.stereo Width = 0.0`, Kick `pan = 0`, send `FX Room = 0` sur Kick & Basse, `HPF >= 120 Hz` sur `BUS Music`. |
| **`low_mid_corr < 0.55`** | Pad/Chords/Boucle hors phase `200–800 Hz` | Appliquer `studio_lib.stabilize_stereo_low_mid()` sur le sample fautif ou rétrécir `mixer/N/stereo = -0.35` + réduire `FX Room` send/width. |
| **`sharpness >= max`** | Lead trop aigu (`> MIDI 74`) ou aigus agressifs (`2.5–6 kHz`) | **Action n°1** : vérifier d'abord que le Lead ne dépasse pas `MIDI 74` (sinon baisser de `-12 st`), puis abaisser le LPF du synthé/insert Lead vers `5–7 kHz` et poser `gl.eq` **`Band type = 6 (Peak)`** `-3.5 dB @ 3200 Hz Q=1.4`. |
| **`micro_glitches > 0`** | Clics d'attaque/fin de sample ou hats Nyquist | Lancer `audit_project_sources.py --prepare-samples`, ajouter `fade_in >= 2 ms` / `fade_out >= 12 ms`, `LPF 12–14.5 kHz` sur hats/snare, `gl.limiter Attack = 0.002`. |
| **`resonances = [F Hz]`** | Empilement harmonique ou pic étroit à `F Hz` | Sur la piste dominante (`Chords`/`Pad`/`BUS Music`), poser `gl.eq` **`Band type = 6 (Peak)`** à `F Hz`, `Gain = -4.0 dB`, `Q = 2.8` (ou décaler l'octave du Lead). |
| **`low_mids > max %`** | Boue `250–500 Hz` sur Chords/Pad/Basse | Sur `BUS Music slot 2 (gl.eq)`, `HPF = 140 Hz` + **`Band type = 6 (Peak)`** `-3.5 dB @ 280 Hz Q=1.2`. |
| **`air < min %`** | Mix étouffé (LPF trop bas, hats faibles ou mauvais seuil sombre) | **Si `valence >= 0`** : relever `BUS Drums slot 1` `Band 7 (High shelf)` `+1.5 dB @ 10 kHz` ou ouvrir `BUS Music` `LPF` vers `12–13.5 kHz`. **Si `valence < 0`** : ne pas éclaircir les synthés ; ajuster légèrement le niveau des hi-hats/shakers (`+1 dB`) ou vérifier que le seuil `air min` du profil sombre (`1.2 %`) est bien utilisé (`--dark` / `song_blueprint.json`). |
| **`tuning_offset > 6.5 ct`** | Sample désaccordé (≠ `440 Hz`) | Relire `audit_project_sources.py` → corriger `Root note` + `Fine tune` dans `gl.sampler` sur le sample fautif. |
| **`dynamic_arc < min dB`** | Toutes les sections ont la même densité ou compresseur écrase l'Intro | Retirer Basse/Drums de l'Intro & Pont (Matrice Facette 3 §1.3), vérifier `BUS Music gl.comp Threshold >= -13.5 dB`, automatiser `BUS Music` volume (`-2.5 dB` Intro/Pont, `+1.0 dB` Climax). |
| **Qualité perceptuelle `PQ < 6.5` ou fin coupée** | Timbre synthétique sec ou coupure brutale | Vérifier l'équilibre des 3 couches, l'humanisation des vélocités, et ajouter un fade-out / résolution tenue sur les 2 dernières mesures. |

---

## Chapitre 4 — Playbook des Pièges Techniques Vérifiés & Solutions

| Piège Vérifié | Symptôme Observé | Solution Définitive |
| :--- | :--- | :--- |
| **`gl.eq` `Band type = 4` vs `6`** | Trou $-\infty\text{ dB}$ dans le spectre (`4` = `Notch`). | Toujours utiliser **`6` (`Peak`)** pour toute cloche EQ. |
| **`gl.reverb` paramètres `(full)` & unités** | Audio sec qui fuit malgré `Dry = 0`, ou erreur de plage sur `Decay (full)`. | Régler `Dry (full) = 0`, `Low cut (full) >= 350–480`, **`Decay (full)` en décisecondes (`1..200`, ex. `4` = `0.4 s`)**, `Early reflections (full) <= 24`, `Wet (full) = 45..64`. |
| **`TEMPLATE_EMPTY_MIXER` pistes `0..16`** | Crash `no mixer track 17` si on ajoute des plugins sur `17..20`. | `setup_standard_buses()` appelle désormais `add_mixer_track` automatiquement avant d'ajouter les plugins. |
| **`set_send` sans `sidechain=True`** | Le audio brut du Kick passe dans `BUS Bass`/`BUS Music` et sature. | Toujours passer `{"sidechain": True, "preFader": True}`. |
| **`gl.multisampler` (`.sfz`) en GUI** | Silencieux (`-144 LUFS`) dans `gridline-app --attach`. | Utiliser **`gl.sampler`** avec WAV 24-bit calibré `440 Hz` (`Voices = 12`) ou `gl.keys`/`gl.warmsynth`. |
| **`patch` `mixerTrack`** | Route sur le mauvais bus (`id` interne ≠ index piste). | Utiliser exclusivement `add_channel(..., mixer=M)` ou `edit_channel(channel=N, mixer=M)`. |
| **`gl.limiter` `Attack = 0.005 s`** | Overshoots True Peak et micro-clics sur hats. | Toujours fixer **`Attack (look-ahead) = 0.002`**, `Ceiling <= -2.15 dB`, `0.2 dB safety = 1`. |
| **`librosa.effects.pitch_shift` sur 808** | Dérive `-35 ct` et attaque sub ramollie. | Régler nativement `Root note` + `Fine tune` dans `gl.sampler`. |
| **`loudnorm` sur one-shots `< 2.5 s`** | Écrasement des transitoires Kick/Snare/Clap. | Peak norm `-1.0 dBFS` + micro-fades (`2 ms / 15 ms`) via `clean_audio_sample`. |
| **Collision `_infer_roles()` MIDI** | `"Vocal"` fusionné avec `Lead`, ou `"Piano"` arpégé pris pour `Lead`. | Respecter la table Facette 3 §5.2 (`Bass`, `Chords Piano`, `Lead`, `Counter Harmony`). |
| **Timeout `select_best_samples`** | Indexation CLAP CPU synchrone si `sample_index.json` absent. | Utiliser `search_samples` (`samples.db`) + `analyze_sample_dsp`, ou pré-indexer en CLI `--no-clap`. |
| **Empilement `400–630 Hz`** | Résonance 1/3 oct malgré MIDI propre. | Creuser `Chords/Pad` (`gl.eq Peak = 6`, `-3.5 dB @ 450 Hz Q=1.4`) ou répartir les voicings d'accords plus bas (`low_midi=48`), sans jamais monter le Lead au-dessus de `MIDI 74`. |

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
