# Facette 3 — Blueprint, Arrangement Multi-Genres, Top-Down `master_sketch.mid`, MIDI Gate & Patron `build_song.py`

> **Cheatsheet IA (L'essentiel en 14 lignes)**
> - **Source de vérité unique** : Écrire **toutes** les notes en **beats 0-indexed** dans 1 script Python `build_song.py` qui génère : (1) `song_blueprint.json`, (2) `master_sketch.mid` (3 pistes nommées **`Bass`**, **`Chords`**, **`Lead`**), (3) vérifie `verify_midi_gate()`, puis (4) construit tout le `.gridline` en 1 passe via `GridlineMCP`.
> - **Gate MIDI bloquant** : Tant que `verify_midi_gate()` ne retourne pas `PASS`, **0 appel Gridline et 0 rendu audio** — corriger les notes en Python en `< 0.5 s` grâce à la Table de Correction (§5.3).
> - **Piège `_infer_roles()` (§5.2)** : Nommer exactement `Bass`/`808`, `Chords`/`Pad`, `Lead`/`Melody`, `Counter Harmony`. Ne jamais nommer un contre-chant `"Vocal"` ni un accompagnement arpégé `"Piano"` seul.
> - **Arrangement vivant** : Max **3 couches** hors batterie · **Matrice Rôles × Sections (§1.3)** · renouvellement toutes les **4–8 mesures** · transition obligatoire (drum fill / beat-4 mute / LPF sweep) · **fin propre** (résolution + extinction naturelle, jamais coupée sec).
> - **Physique `T_noire = 60/BPM`** : Release 808 `35–55 ms` · Sidechain release `80–110 ms` · `FX Room` Decay `<= T_noire` (genres rythmiques) ou `0.6–1.1 s` (acoustique/ambient).

---

## Chapitre 1 — Spécification `song_blueprint.json`, Structures Multi-Genres & Arrangement

### 1.1 Structure Normative (`schemas/song_blueprint.schema.json`)
Tout projet commence par `song_blueprint.json` valide :
* **Métadonnées** : `title`, `genre`, `mood`, **`mix_profile`** (`trap_rap`, `pop_rnb`, `edm_club`, `rock_metal`, `lofi_chill`, `jazz_acoustic`, `classical_ambient`), `emotion_vector` (`valence`, `tension`, `energy`, `depth`), `bpm`, `time_signature` (`numerator`, `denominator`), `key`, `swing`, `target_lufs`.
* **`harmonic_core` (Obligatoire)** :
  * `key` (ex. `"C"`, `"F#"`), `mode` (tous modes acceptés : `ionian`, `aeolian`, `dorian`, `phrygian`, `phrygian_dominant`, `lydian`, `mixolydian`, `harmonic_minor`, `melodic_minor`, `locrian`, `pentatonic_minor`, `pentatonic_major`, `blues`), `progression_bars` (`1..32`).
  * `chord_progression` : par mesure `bar`, `symbol`, `bass_midi` (`24–60`), `voicing_midi` (2–6 notes `36–84` calculées par `optimize_voice_leading`), `chord_tones_pc` (`0..11`).
  * `motif_cell` : `intervals_from_tonic` (`2..12` notes), `rhythm_beats`, `apex_midi` (`55–88`).
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

### 1.4 Boîte à Outils Transitions & Fin Propre
* **Entre deux sections** (mesures 4, 8, 12, 16...) : appliquer au moins **1 levier** parmi :
  1. *Beat-4 Drop-Out* : couper batterie + basse sur le dernier temps (`beat 3.0..4.0`) + automatiser `FX Room` volume à `0` → impact massif sur le temps 1 suivant !
  2. *Drum Fill* : 3–4 frappes de snare/toms croissantes (`vel 70 → 110`) sur les 2 derniers temps.
  3. *Filter Sweep / Riser* : ouvrir progressivement le Cutoff LPF de l'harmonie sur 2–4 mesures avant le Refrain.
* **Début & Fin propres** : jamais de clic à `t=0` ; terminer l'Outro par une **résolution sur l'accord de tonique (`I` ou `i`) qui s'éteint naturellement** (ou fade-out d'automation sur les 2–4 dernières mesures) — jamais une coupure sèche en plein milieu d'une phrase !

---

## Chapitre 2 — Cohérence Multi-Échelles, Règle des 3 Couches & Physique (`T_noire`)

1. **Échantillon (`< 5 ms`)** : zéro clic/DC/clipping, accordage **`A4 = 440.0 Hz ± 5 ct`**.
2. **Transitoire (`5–50 ms`)** : attaques concurrentes `<= 15 %` dans `400 Hz–3 kHz` ; séparation Kick/Basse.
3. **Motif (`1–2 temps`)** : 1 cellule rythmico-mélodique mémorable.
4. **Phrase (`4 mesures`)** : `A - A' - A'' (Apex unique) - B (Silence)`.
5. **Section (`8–16 mesures`)** : renouvellement toutes les **4–8 mesures** + transition.
6. **Macro** : contraste réel `section_lufs` selon le profil (Facette 5 §1).

* **Règle des 3 Couches (hors batterie)** : (1) **Basse** (mono `< 150 Hz`), (2) **Lit Harmonique** (`Chords/Pad`, attaque douce ou dynamique feutrée), (3) **Leader Unique** (`Lead` **OU** `Voix`, contre-chant uniquement dans ses silences).
* **Calibrage sur `T_noire = 60.0 / BPM`** :
  * Release 808 : `35–55 ms` (mono cut-itself `Voices = 1`).
  * Release Sidechain : `80–110 ms` (`~1/4` à `1/2` de `T_noire`).
  * Decay `FX Room` : `<= 1.0 × T_noire` pour les genres rythmiques (`0.20–0.42 s`, soit `Decay (full) = 2..4 ds`) ; jusqu'à `0.7–1.1 s` (`7..11 ds`) pour Acoustique/Classique/Ambient.

---

## Chapitre 3 — Loi de la Réduction Unique (`master_sketch.mid`) & Voice Leading

1. **Interdiction** d'écrire Basse, Accords et Mélodie séparément dans Gridline sans partition mère validée.
2. Générer **`master_sketch.mid`** (3 pistes : **`Bass`**, **`Chords`**, **`Lead`**) en Python avec `optimize_voice_leading(chord_pcs_list, low_midi=48, high_midi=72)` (`mcp-servers/audio-analyzer/musicality_metrics.py`).
3. Valider via `studio_lib.verify_midi_gate(".../master_sketch.mid", expected_key="C minor", max_pitch=84)`.
4. Éclater la **même** partition (en beats) vers les canaux Gridline (puis optionnellement revérifier le MIDI orchestré exporté).

---

## Chapitre 4 — Grammaire Mélodique (`M1–M5`), Verticale (`V1–V6`) & Horizontale (`H1–H3`)

* **M1 (Conjoint + *Gap-Fill*)** : `>= 65 %` des intervalles `<= 4 st` ; `>= 60 %` des sauts `>= 5 st` suivis d'un retour inverse de `1–3 st`. Ambitus recommandé `<= 19 st` (`12–16 st` optimal).
* **M2 (Économie Motivique)** : `A - A' - A'' (Apex) - B`.
* **M3 (Respiration)** : `20 %` à `50 %` de silence réel dans la piste `Lead`.
* **M4 (Gravité Harmonique)** : `>= 80 %` des notes sur temps forts dans l'accord courant.
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

| Métrique | Calcul Réel du Code | Seuil `PASS` |
| :--- | :--- | :---: |
| **`stepwise_ratio`** | Proportion de transitions `\|Δp\| <= 4 st` | `>= 0.65` |
| **`gap_fill_compliance`** | Sauts `\|Δp\| >= 5 st` suivis d'un retour inverse `1 <= \|Δp_next\| <= 3 st` | `>= 0.60` |
| **`melody_rest_ratio`** | `1 - (durée_couverte_Lead / durée_totale)` | `0.20 – 0.50` |
| **`strong_beat_chord_tone_ratio`** | Notes Lead à `±0.09 s` d'un temps fort (`beats[::2]`) dans les pitch-classes de l'accord/basse actif | `>= 0.80` |
| **`voice_leading_avg_semitones`** | Déplacement moyen entre tranches d'accords (`quantize = 0.08 s`) | `<= 2.50 st` |
| **`lead_overlap_conflict`** | Attaques simultanées (`<= 40 ms`, `< 12 st`) entre pistes `lead` et `counter` | `<= 0.15` |
| **`contrary_or_oblique_ratio`** | Transitions Basse/Lead avec `dm * db <= 0` | `>= 0.55` |
| **Intégrité tonale & clashs** | `out_of_scale == 0`, `minor_2nd_clashes` (`1` ou `13 st` inter-pistes) `== []`, `sub_collisions` (`< MIDI 50`, pitches différents) `== []` | `0` défaut |

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
| `strong_beat_chord_tone_ratio < 0.80` | Note de passage sur temps fort | Sur `beat % 2 == 0`, projeter la note sur la pitch-class la plus proche de l'accord courant. |
| `voice_leading_avg_semitones > 2.50` | Accords déplacés en blocs | Passer la suite d'accords dans `optimize_voice_leading(chord_pcs_list, 48, 72)`. |
| `lead_overlap_conflict > 0.15` | Lead & Counter frappent ensemble | Décaler les attaques du `Counter` dans les silences du `Lead` (ou à `+12 st` / timbre différent). |
| `contrary_or_oblique_ratio < 0.55` | Basse et Lead montent/descendent ensemble | Inverser la direction d'une mesure sur deux du Lead (ou tenir une note pivot quand la basse change). |
| `minor_2nd_clashes` (`1` ou `13 st`) | Frottement 2nde/9e min inter-pistes | Décaler la note fautive d'un degré de gamme (vers une note de l'accord) ou couper l'extension sur ce temps. |
| `sub_collisions` | Deux pistes jouent `< MIDI 50` en même temps | Garder `Chords` strictement `>= MIDI 50` (`low_midi=50` dans `optimize_voice_leading`). |

---

## Chapitre 6 — Patron Canonique Minimal `build_song.py` (Single Source of Truth)

> **Modèle de référence** : Tout agent (même léger) peut copier-adapter ce patron en **1 seul script Python** : 0 calcul MIDI de tête, 0 décalage entre `master_sketch.mid` et Gridline, 0 erreur d'unités, exécution `< 3 s`.

```python
#!/usr/bin/env python3
import json, os, shutil, sys
import pretty_midi

ROOT = "/home/gamo/Bureau/musicos"
sys.path.extend([f"{ROOT}/scripts", f"{ROOT}/mcp-servers/audio-analyzer"])
from studio_lib import GridlineMCP, TEMPLATE_EMPTY_MIXER, setup_standard_buses, verify_midi_gate, check_pattern_audibility
from musicality_metrics import optimize_voice_leading

PROJ_DIR = f"{ROOT}/projects/active"
os.makedirs(PROJ_DIR, exist_ok=True)
BPM, BEATS_PER_BAR, KEY_STR, PROFILE = 120.0, 4, "C minor", "pop_rnb"

# 1. Harmonie calculée de façon déterministe (P3 en C minor: Cm7 - Abmaj7 - Ebmaj7 - Bb)
chord_pcs = [[0, 3, 7, 10], [8, 0, 3, 7], [3, 7, 10, 2], [10, 2, 5]]
voicings  = optimize_voice_leading(chord_pcs, low_midi=50, high_midi=67)
bass_roots = [36, 44, 39, 46]  # C2, Ab2, Eb2, Bb2

# 2. Notes en BEATS 0-indexed (Single Source of Truth: (start_beat, dur_beat, midi_pitch, velocity))
bass_notes, chord_notes = [], []
# Exemple sur 4 mesures vérifié PASS au Gate MIDI (à étendre selon structure complète)
for bar_idx in range(4):
    b0 = bar_idx * BEATS_PER_BAR
    bass_notes.append((b0, 3.5, bass_roots[bar_idx], 98))
    for k, p in enumerate(voicings[bar_idx]):
        chord_notes.append((b0 + k * 0.02, 3.6 - k * 0.02, p, 78 if k < len(voicings[bar_idx]) - 1 else 86))
# Mélodie A - A' - A''(Apex C5=72) - B(Gap-Fill + silence 14.2..16.0) dans C minor
lead_notes = [
    (0.0, 0.9, 67, 94), (1.0, 0.45, 70, 82), (1.5, 0.9, 67, 88), (2.5, 0.85, 65, 78),    # A
    (4.0, 0.9, 63, 92), (5.0, 0.45, 65, 80), (5.5, 0.9, 67, 86), (6.5, 0.85, 65, 76),    # A'
    (8.0, 0.9, 67, 96), (9.0, 0.45, 70, 102), (9.5, 0.9, 72, 106), (10.5, 0.85, 70, 90), # A'' Apex
    (12.0, 0.9, 65, 92), (13.0, 1.2, 67, 84),                                             # B + silence
]

# 3. Export master_sketch.mid + GATE BLOQUANT (0 appel Gridline avant PASS !)
pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
b2s = 60.0 / BPM
for name, notes in [("Bass", bass_notes), ("Chords", chord_notes), ("Lead", lead_notes)]:
    inst = pretty_midi.Instrument(program=0, name=name)
    for st, dur, p, vel in notes:
        inst.notes.append(pretty_midi.Note(vel, p, st * b2s, (st + dur) * b2s))
    pm.instruments.append(inst)
sketch_path = f"{PROJ_DIR}/master_sketch.mid"
pm.write(sketch_path)
verify_midi_gate(sketch_path, expected_key=KEY_STR, max_pitch=84)  # Lève RuntimeError si FAIL

# 4. Build Gridline atomique (1 session MCP)
proj_path = f"{PROJ_DIR}/projet.gridline"
shutil.copyfile(TEMPLATE_EMPTY_MIXER, proj_path)
mcp = GridlineMCP(proj_path, f"{PROJ_DIR}/exports")
mcp.call("set_song", {"bpm": BPM, "numerator": BEATS_PER_BAR, "denominator": 4})
setup_standard_buses(mcp, bpm=BPM, profile=PROFILE)
# add_channel(..., mixer=1..N) -> edit_mixer_track(routeTo=17/18/19) -> set_notes (en beats !) -> add_clip -> save_project
assert not check_pattern_audibility(mcp, min_lufs=-22.0), "Pattern silencieux ou trop faible!"
mcp.call("save_project", {})
mcp.close()
```
