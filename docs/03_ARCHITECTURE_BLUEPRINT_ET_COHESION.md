# Facette 3 — Blueprint, Arrangement Multi-Genres, Top-Down `master_sketch.mid`, MIDI Gate & Patron `build_song.py`

> **Cheatsheet IA (L'essentiel en 14 lignes)**
> - **Source de vérité unique & Isolation par piste** : Écrire **toutes** les notes en **beats 0-indexed** dans 1 script Python `build_song.py` qui génère : (1) `song_blueprint.json`, (2) `master_sketch.mid` (3 pistes nommées **`Bass`**, **`Chords`**, **`Lead`**), (3) vérifie `verify_midi_gate()`, puis (4) construit tout le `.gridline` en 1 passe via `GridlineMCP` en respectant absolument **`1 sonorité = 1 canal + 1 insert mixeur + patterns dédiés + 1 piste Playlist nommée`**.
> - **Gate MIDI bloquant** : Tant que `verify_midi_gate()` ne retourne pas `PASS`, **0 appel Gridline et 0 rendu audio** — corriger les notes en Python en `< 0.5 s` grâce à la Table de Correction (§5.3).
> - **Piège `_infer_roles()` (§5.2)** : Nommer exactement `Bass`/`808`, `Chords`/`Pad`, `Lead`/`Melody`, `Counter Harmony`. Ne jamais nommer un contre-chant `"Vocal"` ni un accompagnement arpégé `"Piano"` seul.
> - **Arrangement vivant** : Max **3 couches** hors batterie · **Matrice Rôles × Sections (§1.3)** · renouvellement toutes les **4–8 mesures** · transition obligatoire (drum fill / beat-4 mute / LPF sweep) · **fin propre** (résolution + extinction naturelle, jamais coupée sec).
> - **Physique `T_noire = 60/BPM`** : Release 808 `35–55 ms` · Sidechain release `65–80 ms` (`trap_rap`, Ratio `2.4–3.0`) ou `80–110 ms` (autres genres) · `FX Room` Decay `<= T_noire` (genres rythmiques) ou `0.6–1.1 s` (acoustique/ambient).

---

## Chapitre 1 — Spécification `song_blueprint.json`, Structures Multi-Genres & Arrangement

### 1.1 Structure Normative (`schemas/song_blueprint.schema.json`)
Tout projet commence par `song_blueprint.json` valide :
* **Métadonnées** : `title`, `genre`, `mood`, **`mix_profile`** (`trap_rap`, `pop_rnb`, `edm_club`, `rock_metal`, `lofi_chill`, `jazz_acoustic`, `classical_ambient`), `emotion_vector` (`valence`, `tension`, `energy`, `depth`), `bpm`, `time_signature` (`numerator`, `denominator`), `key`, `swing`, `target_lufs`.
* **`harmonic_core` (Obligatoire)** :
  * `key` (ex. `"C"`, `"F#"`), `mode` (tous modes acceptés : `ionian`, `aeolian`, `dorian`, `phrygian`, `phrygian_dominant`, `lydian`, `mixolydian`, `harmonic_minor`, `melodic_minor`, `locrian`, `pentatonic_minor`, `pentatonic_major`, `blues`), `progression_bars` (`1..32`).
  * `chord_progression` : par mesure `bar`, `symbol`, `bass_midi` (**`31–48` recommandé** pour 808/sub `>= 49 Hz` — en `C..F#`, jouer Octave 2 `36–42` ; bornes schéma `24–60`), `voicing_midi` (2–6 notes `48–67` recommandées, calculées par `optimize_voice_leading`), `chord_tones_pc` (`0..11`).
  * `motif_cell` : `intervals_from_tonic` (`2..12` notes), `rhythm_beats`, `apex_midi` (`55–74` par défaut ; `75–81` timbres très doux si `valence > +0.3`).
  * `phrase_plan` : `A-A'-A''-B`, `A-B-A'-C`, `A-A'-B-B'`, `A-A-B-A` ou `free_ambient`.
* **`structure`**, **`frequency_plan`**, **`tracks`**, **`vocals`**.

### 1.2 Templates Structurels Multi-Genres (Choisir selon Durée & Style)

| Famille de Genre | Format ~1 min | Format ~2 min | Format ~3 min |
| :--- | :--- | :--- | :--- |
| **Pop / Rock / Synthwave / R&B** | Intro(4) → Couplet(8) → Refrain(8) → Pont(4) → Refrain(8) → Outro(4) | Intro(4) → Couplet 1(8) → Pré-Refrain(4) → Refrain 1(8) → Couplet 2(8) → Pont(8) → Refrain Final(8) → Outro(4) | Idem avec Couplets(16) + Refrain double(16) |
| **Trap / Rap / Drill** | Intro(4) → Refrain 1(8) → Couplet(12–16) → Pont(4) → Refrain 2(8) → Outro(4) | Intro(4) → Refrain 1(8) → Couplet 1(16 : 8+8) → Pont(8) → Refrain 2(8) → Couplet 2(8) → Climax(8) → Outro(8) | Intro(8) → Drop 1(16) → Verse 1(24) → Break(8) → Drop 2(16) → Climax(16) → Outro(16) |
| **House / Techno / EDM** | Intro(8) → Build(4) → Drop 1(16) → Break(4) → Outro(4) | Intro(8) → Build(8) → Drop 1(16) → Break(8) → Build 2(8) → Drop 2(16) → Outro(8) | Intro(16) → Build(8) → Drop 1(32) → Break(16) → Build(8) → Drop 2(32) → Outro(16) |
| **Lo-Fi / Jazz / Boom Bap** | Intro(4) → Thème A(8) → Variation B(8) → Thème A'(8) → Outro(4) | Intro(4) → A(8) → A'(8) → B/Solo(8) → Break(4) → A''(8) → Outro(4) | Format `AABA` (32 mes.) ou Blues `12×N` avec solos |
| **Classique / Cinématique / Ambient** | Exposition(8) → Développement(8) → Climax(8) → Coda(4–8) | Intro(8) → Thème doux(8) → Montée(8) → Climax orchestral(12) → Retombée/Coda(8) | Arche continue en 4 mouvements contrastés |

### 1.3 Matrice d'Arrangement Obligatoire (Éviter le "Mur de Son Plat")
Cocher quels rôles jouent dans chaque section afin de garantir un **arc dynamique réel (`3–5 dB LUFS`)** :

| Rôle | Intro (`E=0.3`) | Couplet (`E=0.6`) | Pont/Break (`E=0.35`) | Refrain/Drop (`E=0.9`) | Climax (`E=1.0`) | Outro (`E=0.25`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Harmonie (`Chords/Pad`)** | ✅ (doux/LPF) | ✅ (allégé) | ✅ (ouvert/variation) | ✅ (plein) | ✅ (+octave) | ✅ (fade-out) |
| **Basse (`808/Bass`)** | ❌ (ou 1 note fin) | ✅ (pattern sobre) | ❌ (ou longue sub) | ✅ (pattern plein) | ✅ (+glides/var.) | ❌ |
| **Batterie (`Drums`)** | ❌ (ou hats doux) | ✅ (sans open-hat) | ❌ (ou rim/tom seul) | ✅ (complet) | ✅ (+shaker/rolls) | ❌ (ou kick seul 2 mes.) |
| **Lead / Voix** | Motif filtré ou ❌ | Squelette / Couplet | Contre-chant / ❌ | ✅ (thème complet) | ✅ (+octave/contre-chant) | Écho final sur tonique |

### 1.4 Architecture Temporelle (Grille Puissances de 2 : Paliers 4 / 8 / 16–32) & Transitions
La macro- et micro-structure reposent sur une grille binaire en multiples de 4 (`4, 8, 16, 32` mesures) afin d'allier prédictibilité et prévention de l'ennui :
1. **Palier Micro (toutes les 4 mesures)** : variation rythmique mineure sans ajout de piste (roulement de hi-hat au temps 4, note de passage basse, coupure/*mute* du Kick sur le dernier demi-temps).
2. **Palier Méso (toutes les 8 mesures)** : modification obligatoire de la densité instrumentale (ajout ou retrait d'une piste de catégorie Rythmique, Harmonie ou Habillage — **aucune boucle de 8 mesures répétée à l'identique**).
3. **Palier Macro (toutes les 16 ou 32 mesures)** : changement de section structurelle (Couplet → Pré-Refrain → Refrain → Pont) avec événement de transition et saut net de dynamique.

* **Entre deux sections** (mesures 4, 8, 12, 16...) : appliquer au moins **1 levier** parmi :
  1. *Beat-4 Drop-Out* : couper batterie + basse sur le dernier temps (`beat 3.0..4.0`) + automatiser `FX Room` volume à `0` → impact massif sur le temps 1 suivant !
  2. *Drum Fill* : 3–4 frappes de snare/toms croissantes (`vel 70 → 110`) sur les 2 derniers temps.
  3. *Filter Sweep / Riser* : ouvrir progressivement le Cutoff LPF de l'harmonie sur 2–4 mesures avant le Refrain.
* **Début & Fin propres** : jamais de clic à `t=0` ; terminer l'Outro par une **résolution sur l'accord de tonique (`I` ou `i`) qui s'éteint naturellement** (ou fade-out d'automation sur les 2–4 dernières mesures) — jamais une coupure sèche en plein milieu d'une phrase !

---

## Chapitre 2 — Cohérence Multi-Échelles, Plafond Cognitif (5–8 Pistes) & Physique (`T_noire`)

1. **Échantillon (`< 5 ms`)** : zéro clic/DC/clipping, accordage **`A4 = 440.0 Hz ± 5 ct`**.
2. **Transitoire (`5–50 ms`)** : attaques concurrentes `<= 15 %` dans `400 Hz–3 kHz` ; séparation Kick/Basse.
3. **Motif (`1–2 temps`)** : 1 cellule rythmico-mélodique mémorable.
4. **Phrase (`4 mesures`)** : `A - A' - A'' (Apex <= 74) - B (Silence)` (`V >= 0`), ou courbe descendante/hypnotique (`V < 0`).
5. **Section (`8–16 mesures`)** : évolution obligatoire aux paliers **4 / 8 / 16 mesures** + transition.
6. **Macro** : contraste réel `section_lufs` selon le profil (Facette 5 §1).

### Plafond Cognitif & Matrice des 5 Rôles Fonctionnels (Quotas Simultanés)
Le cortex auditif humain ne suit simultanément que **3 à 5 foyers d'attention** distincts (analyse de scène auditive). Ne jamais faire jouer simultanément plus de **5 à 8 pistes dans une même mesure** (entre `1` en Intro/Break et `8` max au Refrain/Climax). **Clarification importante** : cette limite `5–8` concerne exclusivement les pistes **qui jouent simultanément dans une même mesure**, et **non** le nombre total de pistes du projet Gridline — qui peut librement totaliser `9–14` pistes dédiées pour respecter strictement l'isolation `1 sonorité = 1 piste` (Facette 1 §1.2). *Ne jamais fusionner des percussions ou instruments sur un même canal/pattern pour respecter ce plafond.*

| Catégorie Fonctionnelle | Quota Max Simultané | Bande Prioritaire | Règle d'Exclusion & de Priorité |
| :--- | :---: | :---: | :--- |
| **1. Fondation (Low-End)** | **1 à 2** | `20–200 Hz` | **Exclusivité** : Kick & Basse ne frappent pas à pleine puissance au même instant $t$ (Sidechain ou décalage rythmique). |
| **2. Rythmique (Groove)** | **2 à 4** | `150 Hz–16 kHz` | **Imbrication** : cases temporelles complémentaires (Kick `1/3`, Snare `2/4`, Hats contretemps). |
| **3. Harmonie (Tapis)** | **1 à 2** | `200 Hz–3 kHz` | **Discrétion** : mouvements lents, creux EQ obligatoire `1–3 kHz` pour laisser passer le Lead. |
| **4. Élément Principal (Lead)** | **1** (max 2) | `500 Hz–5 kHz` | **Suprématie** : domine l'harmonie de **`>= +3 dB`**. Si 2 Leads coexistent (Lead + Contre-chant), alternance stricte en question-réponse. |
| **5. Habillage (Ear Candy)** | **1 à 2** | Variable (filtré) | **Intermittence** : présence **< 15 %** de la durée totale du morceau (transitions/surprises uniquement). |

* **Calibrage sur `T_noire = 60.0 / BPM`** :
  * Release 808 : `35–55 ms` (mono cut-itself `Voices = 1`).
  * Release Sidechain : **`65–80 ms`** en `trap_rap` (`Ratio 2.4–3.0`, `Threshold -18 dB` pour préserver le corps 808) ; `80–110 ms` (`~1/4` à `1/2` de `T_noire`) pour les autres genres.
  * Decay `FX Room` : `<= 1.0 × T_noire` pour les genres rythmiques (`0.20–0.42 s`, soit `Decay (full) = 2..4 ds`) ; jusqu'à `0.7–1.1 s` (`7..11 ds`) pour Acoustique/Classique/Ambient.

---

## Chapitre 3 — Loi de la Réduction Unique (`master_sketch.mid`) & Voice Leading

1. **Interdiction** d'écrire Basse, Accords et Mélodie séparément dans Gridline sans partition mère validée.
2. Générer **`master_sketch.mid`** (3 pistes : **`Bass`**, **`Chords`**, **`Lead`**) en Python avec `optimize_voice_leading(chord_pcs_list, low_midi=48, high_midi=67)` (`mcp-servers/audio-analyzer/musicality_metrics.py`).
3. Valider via `studio_lib.verify_midi_gate(".../master_sketch.mid", expected_key="C minor", max_pitch=72 if valence < -0.3 else 76, valence=valence, tension=tension)` (défaut `max_pitch=74`).
4. Éclater la **même** partition (en beats) vers les canaux Gridline en respectant strictement la règle **« 1 sonorité = 1 piste dédiée »** : chaque instrument ou élément rythmique (`Kick`, `Clap`, `HiHat`, `808 Bass`, `Chords Pad`, `Lead`, `Counter`…) reçoit son propre canal, son propre insert mixeur dédié (avant les bus `17..20`), ses **propres patterns séparés** (jamais de pattern multi-instruments) et sa **propre piste Playlist nommée** (`project.playlist.setTrackName(tr, nom)`), chaque automation ayant également sa piste dédiée. (Optionnellement revérifier le MIDI orchestré exporté.)

---

## Chapitre 4 — Grammaire Mélodique (`M1–M5`), Verticale (`V1–V6`) & Horizontale (`H1–H3`)

* **M1 (Conjoint + *Gap-Fill*)** : `>= 65 %` des intervalles `<= 4 st` ; `>= 60 %` des sauts `>= 5 st` suivis d'un retour inverse de `1–3 st`. Ambitus recommandé `<= 19 st` (`12–16 st` optimal).
* **M2 (Économie Motivique)** : `A - A' - A'' (Apex <= 74) - B` (`V >= 0`) ou motif descendant/hypnotique (`V < 0`).
* **M3 (Respiration)** : `20 %` à `50 %` de silence réel dans la piste `Lead`.
* **M4 (Gravité Harmonique)** : `>= 80 %` des notes sur temps forts dans l'accord courant en mode consonant (`>= 65 %` en mode sombre/tendu pour autoriser les appoggiatures expressives `b6->5`, `4->b3`, `b2->1`).
* **M5 (*Pocket Alignment*)** : appuis du Lead calés sur temps `1`/`3` (ou anacrouses) en dialogue avec la batterie.
* **V1 (Voice Leading)** : distance moyenne entre accords consécutifs `<= 2.50 st`, maintien des notes communes.
* **V2 (Zéro transposition d'accord samplé)** : jamais transposer un sample d'accord complet ; utiliser un `mono_oneshot` ou un synthé polyphonique `gl.*`.
* **V3 (Mouvement Contraire/Oblique)** : `>= 55 %` des transitions Basse/Lead en sens contraire ou oblique.
* **V4 (Relais d'Attaques)** : opposer attaque sèche et attaque lente ; contre-chant confiné aux silences du Lead (`overlap <= 15 %`).
* **V5 (EQ Miroir)** : creuser `-2` à `-4 dB` (`800 Hz–2.5 kHz`) sur Chords/Pad au profit du Lead/Voix.
* **V6 (Ciment Timbral)** : `BUS Music` + `FX Room` unique adapté au `mix_profile`.
* **H1–H3 (Horizontal)** : dériver le motif entre sections par **4 leviers** (octave `±12 st`, densité, LPF, contre-chant écho) ; ponts harmoniques en mes. 4/8 ; arc dynamique réel.

---

## Chapitre 5 — Gate MIDI `audit_midi_harmony`, Règles `_infer_roles()` & Table de Correction

### 5.1 Seuils Exactement Vérifiés dans le Code (`musicality_metrics.py` & `deep_diagnostics.py`)

| Métrique | Mode Consonant (`V >= -0.3` et `T < 0.5`) | Mode Sombre / Tendu (`V < -0.3` ou `T >= 0.5`) |
| :--- | :---: | :---: |
| **`max_pitch` (Lead) / `min_pitch` (808)** | `<= 76` (`E5`) / `>= 31` (`G1 = 49 Hz`) | **`<= 72–74`** (`C5–D5`) / **`>= 31`** (`G1`, éviter infra-sub `< 42 Hz`) |
| **`strong_beat_chord_tone_ratio`** | `>= 0.80` | **`>= 0.65`** *(autorise les appoggiatures expressives `b6->5`, `4->b3`, `b2->1` sur temps forts)* |
| **`minor_2nd_clashes`** | `== 0` | **`<= 2` par bloc de 4 mesures** *(si tension diatonique contrôlée `b2/1` en Phrygien ou `b6/5` en Mineur, résolue au pas suivant)* |
| **`stepwise_ratio`** (`\|Δp\| <= 4 st`) | `>= 0.65` | `>= 0.65` |
| **`gap_fill_compliance`** (`\|Δp\| >= 5` → retour `1–3 st`) | `>= 0.60` | `>= 0.60` |
| **`melody_rest_ratio`** | `0.20 – 0.50` | `0.20 – 0.50` |
| **`voice_leading_avg_semitones`** (`quantize = 0.08 s`) | `<= 2.50 st` | `<= 2.50 st` |
| **`lead_overlap_conflict`** (`<= 40 ms`, `< 12 st`) | `<= 0.15` | `<= 0.15` |
| **`contrary_or_oblique_ratio`** (`dm * db <= 0`) | `>= 0.55` | `>= 0.55` |
| **Intégrité tonale & sub** | `out_of_scale == 0`, `sub_collisions` (`< MIDI 48`) `== []` | `out_of_scale == 0`, `sub_collisions` (`< MIDI 48`) `== []` |

*(Note 3/4 ou 6/8 : `strong_beats = beats[::2]` est calibré sur 4/4 ; sur un 3/4 ou Ambient pur sans Lead, veiller à aligner les notes sur les temps forts ou vérifier les autres métriques et `out_of_scale == 0`.)*

### 5.2 Table Complète de Nommage `_infer_roles()` (Éviter tout Faux Classement !)

| Rôle détecté par le code | Mots-clés dans le nom de piste (insensible à la casse) | Piège à éviter |
| :--- | :--- | :--- |
| **Ignoré (Perc/FX)** | `kick, snare, clap, hat, hh, perc, rim, cymbal, crash, tom, shaker, drum, triangle, riser, fx, sfx, chant` | Nommer les textures/risers avec `"FX"`. |
| **`bass`** | `808, bass, sub` (ou pitch moyen `< 48`) | Toujours nommer la basse `"Bass"` ou `"808"`. |
| **`lead`** | `lead, melody, topline, vox, vocal` (ou `"piano"` si `poly_ratio < 0.45` !) | **Piège** : jamais `"Vocal"` sur un contre-chant, ni `"Piano"` seul sur un accompagnement arpégé (nommer `"Chords Piano"`). |
| **`chords`** | `chord, pad, string, rhodes, harmony` (ou `poly_ratio >= 0.45`) | Nommer `"Chords"` / `"Pad"` / `"Chords Piano"`. |
| **`counter`** | Tout autre nom (`guitar, bell, pluck, synth, flute, brass...`) | Si aucun `"lead"` n'existe, la piste la plus aiguë est promue en `lead`. |

### 5.3 Table de Correction Déterministe MIDI Gate (`FAIL` → Action Unique)

| Échec rapporté | Cause | Correction Exacte en Python |
| :--- | :--- | :--- |
| `stepwise_ratio < 0.65` | Trop de sauts `>= 5 st` | Ramener les notes erratiques à l'octave la plus proche du point précédent (`±1..3 st`). |
| `gap_fill_compliance < 0.60` | Saut `>= 5 st` non compensé | Après chaque saut `>= 5 st`, forcer la note suivante à revenir de `1` ou `2` degrés en sens opposé. |
| `melody_rest_ratio < 0.20` | Lead trop dense / continu | Raccourcir les durées (`×0.75`) et vider les beats `3.0–4.0` des mesures 2 et 4 (ou couper Lead dans Intro/Pont). |
| `melody_rest_ratio > 0.50` | Lead trop vide | Allonger légèrement les notes tenues ou ajouter 1–2 notes conjointes en mesure 1 et 3. |
| `strong_beat_chord_tone_ratio < min` | Trop de notes de passage sur temps forts (`< 0.80` consonant ou `< 0.65` sombre/tendu) | Sur `beat % 2 == 0`, projeter la note sur la pitch-class la plus proche de l'accord courant. |
| `too_high_count > 0` | Note Lead au-dessus de `max_pitch` (`72–74` sombre, `76` consonant) | Baisser la phrase ou les sommets d'une octave (`-12 st`). |
| `too_low_count > 0` | Basse / 808 en zone infra-sub (`< MIDI 31`, `< 49 Hz`) | Relever les notes 808 d'une octave (`+12 st`, en Octave 2 `MIDI 36–42` pour `C..F#`) ; ou passer `min_pitch=28` uniquement pour basse électrique/contrebasse acoustique. |
| `voice_leading_avg_semitones > 2.50` | Accords déplacés en blocs | Passer la suite d'accords dans `optimize_voice_leading(chord_pcs_list, 48, 67)`. |
| `lead_overlap_conflict > 0.15` | Lead & Counter frappent ensemble | Décaler les attaques du `Counter` dans les silences du `Lead` (ou à `+12 st` / timbre différent). |
| `contrary_or_oblique_ratio < 0.55` | Basse et Lead montent/descendent ensemble | Inverser la direction d'une mesure sur deux du Lead (ou tenir une note pivot quand la basse change). |
| `minor_2nd_clashes` (`1` ou `13 st`) | Frottement non diatonique ou > 2/bloc en mode sombre | Vérifier que la tension appartient à la gamme active (`<= 2` par 4 mes. si `V < -0.3` / `T >= 0.5`), sinon déplacer d'un degré de gamme. |
| `sub_collisions` | Deux pistes jouent `< MIDI 48` en même temps | Garder `Chords` strictement `>= MIDI 48` (`low_midi=48` dans `optimize_voice_leading`). |

---

## Chapitre 6 — Patron Canonique Minimal `build_song.py` (Piloté par `song_blueprint.json`)

> **Modèle de référence (Centralisé via `scripts/song_builder.py`)** : Ne **jamais** réécrire la plomberie (`pretty_midi`, `GridlineMCP`, création des patterns/clips par section, nommage Playlist, EQ tuples). Importer `song_builder.py` (`load_blueprint`, `degree_to_midi`, `ChannelSpec`, `eq_insert`, `eq_music_pocket`, `eq_kick_clean`, `run_song_cli`) afin de n'écrire dans `build_song.py` que la logique musicale du morceau.

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/gamo/Bureau/musicos/scripts")
from song_builder import (
    ChannelSpec, degree_to_midi, eq_insert, eq_kick_clean,
    eq_music_pocket, optimize_voice_leading, run_song_cli,
)

PROJ_DIR = "/home/gamo/Bureau/morceaux/mon_morceau"
SECTIONS = [("Intro", 0, 8, "intro"), ("Drop1", 8, 16, "drop1"), ("Outro", 24, 8, "outro")]

def generate(meta):
    voicings = optimize_voice_leading([[0, 3, 7], [8, 0, 3], [5, 8, 0], [0, 3, 7]], 48, 65)
    bass_roots = [36, 44, 41, 36]
    tracks = {"kick": [], "808": [], "chords": [], "lead": []}
    markers = [{"name": s, "position": float(b * meta["beats_per_bar"])} for s, b, _, _ in SECTIONS]
    # ... remplir tracks en beats 0-indexed selon SECTIONS (A - A' - A'' - B) ...
    return tracks, markers

CHANNELS = [
    ChannelSpec("Kick", "kick", 1, "drums", volume=0.68, sample_file="kick.wav", one_shot=1, sidechain_buses=[(18, 0.72), (19, 0.50)]),
    ChannelSpec("808", "808", 2, "bass", volume=0.46, sample_file="808.wav", root_note=24, fine_tune=None, voices=1, release=0.05),
    ChannelSpec("Chords", "chords", 3, "music", volume=0.76, sample_file="bell_c5.wav", root_note=72, voices=12, attack=0.06, release=0.42, lpf_hz=2800.0, fx_send=0.14),
    ChannelSpec("Lead", "lead", 4, "music", volume=0.96, sample_file="bell_c5.wav", root_note=72, voices=4, attack=0.01, release=0.18, lpf_hz=4800.0, fx_send=0.16),
]

if __name__ == "__main__":
    run_song_cli(
        proj_dir=PROJ_DIR,
        out_gridline=f"{PROJ_DIR}/MonMorceau.gridline",
        sections=SECTIONS,
        channels=CHANNELS,
        generate_fn=generate,
        bass_keys=["808"],
        chords_key="chords",
        lead_key="lead",
        insert_eq_params=eq_kick_clean(1) + eq_music_pocket(3) + eq_insert(4, [(1, 1, 250), (4, 6, 1500, 2.5, 0.45), (7, 3, 5500)]),
    )
```
