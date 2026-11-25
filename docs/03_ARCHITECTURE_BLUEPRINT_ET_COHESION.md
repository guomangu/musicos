# Facette 3 — Blueprint, Arrangement, Transitions, Gate / Autofix MIDI & `compile_song.py` (v3.2+, remplace `build_song.py` manuel)

> **Cheatsheet IA (L'essentiel v2.0)**
> - **Source unique & Isolation (`1 sonorité = 1 piste`)** : Écrire toutes les notes en **beats 0-indexed** dans `build_song.py` (`song_builder.py` + `gl_dsl.py` + `groove_lib.py` + `transitions_lib.py`), exporter `master_sketch.mid` (`Bass`, `Chords`, `Lead`), valider via `verify_midi_gate(..., auto_fix=True)`.
> - **Gate MIDI multi-signatures & Autofix** : Supporte `4/4`, `3/4`, `6/8`, `5/4`, `phrase_plan="free_ambient"` et morceaux sans basse (`contrary_or_oblique_ratio=None`). Si échec, **`scripts/midi_autofix.py`** corrige automatiquement avant tout appel Gridline.
> - **Flux v3.2+ (canonique)** : plus de `build_song.py` manuel — `compile_song.py <projet> [--probe]` lit `song_blueprint.json` + `arrangement_plan.json`, applique `blueprint_enricher.py` (champs créatifs omis), `NarrativeHarmony` (`section_progressions`, phrasé 8 mesures, suspensions 4→3/9→8), `OrchestrationEngine` (comping, voice-leading conscient du Lead), `ExpressionArticulations` (`glide_to_next`, vibrato retardé), `FullCoverageCoreValidator` (pré-vol 100 % + sonde `--probe`), produit le `ScoreIR` (groove en `expression`, clamp frontière 20 ms) puis `master_sketch.mid`.
> - **Piège `_infer_roles()` (§5.2)** : Nommer `Bass`/`808`, `Chords`/`Pad` (`poly_ratio >= 0.45` toléré même `< MIDI 48`), `Lead`/`Melody`, `Counter Harmony` (jamais `"Vocal"` sur contre-chant ni `"Piano"` seul sur arpège).
> - **Arrangement & Transitions (`transitions_lib.py`)** : Max **3 couches** hors batterie · paliers **4 / 8 / 16 mes.** · `beat4_dropout`, `filter_sweep`, `section_gain_arc`, `outro_fade` · fin résolue/fondue (jamais coupée net).
> - **Physique `T_noire = 60/BPM`** : Release 808 `35–55 ms` · Sidechain `65–80 ms` (`trap_rap`, `2.4:1`) ou `80–110 ms` · `FX Room` Decay `<= T_noire` rythmique ou `0.7–1.1 s` acoustique/ambient.

---

## Chapitre 1 — `song_blueprint.json`, Structure Multi-Genres & `transitions_lib.py`

### 1.1 Schéma Normatif (`schemas/song_blueprint.schema.json`)
* **Métadonnées** : `title`, `genre`, `mood`, **`mix_profile`** (`trap_rap`, `pop_rnb`, `edm_club`, `rock_metal`, `lofi_chill`, `jazz_acoustic`, `classical_ambient`), **`has_drums`** (`bool`, défaut `true` ; mettre `false` en Classique/Ambient sans batterie pour assouplir le Quality Gate), `emotion_vector` (`valence`, `tension`, `energy`, `depth`), `bpm`, `time_signature` (`numerator`, `denominator`), `key`, `swing`, `target_lufs`.
* **`harmonic_core`** : `key`, `mode`, `progression_bars`, `chord_progression` (`bass_midi` `31–48` ou `36–42` en `C..F#`, `voicing_midi` `48–67` via `optimize_voice_leading`, `chord_tones_pc`), `motif_cell`, `phrase_plan` (`A-A'-A''-B`, `A-B-A'-C`, `A-A'-B-B'`, `A-A-B-A`, `free_ambient`).
* **`structure`**, **`frequency_plan`**, **`tracks`**, **`vocals`**. Carte complète → [`docs/00_SOMMAIRE.md`](00_SOMMAIRE.md).
* **`tracks[]`** : chaque piste sample porte **`root_note`** (MIDI détecté, ex. 57 pour un La3) et **`fine_tune_ct`** (cents) issus de `propose_palettes` → `blueprint_tracks` ; `song_builder` les applique à `gl.sampler` (`channel/N/Root note`, `Fine tune`). Ne jamais laisser un sample non-C sur le root par défaut C4=60 (dissonance systématique).
* **`swing`** : appliqué réellement aux croches off-beat (`groove_lib.apply_swing`, délai = `swing × 0.25` temps) ; les notes syncopées hors grille 8ᵦ restent en place.
* **BPM des boucles** : `propose_palettes(bpm=...)` exclut les boucles dont le BPM détecté (nom de fichier ou estimation librosa dans `index_project_samples`) s'écarte de plus de 2× la tolérance (défaut 6 BPM), pénalise entre 1× et 2×.

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

### 1.4 Architecture Temporelle (Paliers `4 / 8 / 16`) & Transitions Automatisées (`scripts/transitions_lib.py`)
1. **Palier 4 mes.** : variation rythmique mineure (`beat4_dropout(tracks, autos, beat_idx, bpb)` : coupe batterie/basse sur dernier temps + baisse `FX Room`).
2. **Palier 8 mes.** : modification obligatoire de densité instrumentale (**aucune boucle de 8 mesures identique**).
3. **Palier 16 mes.** : saut dynamique + transition automatisée via **`transitions_lib.py`** : **`filter_sweep`** (ouverture/fermeture LPF `gl.eq`/`Cutoff`), **`section_gain_arc`** (arche dynamique `BUS Music`), **`outro_fade`** (résolution sur tonique + extinction propre).
4. **Transitions riches déclaratives (`Q5.1`)** : `generate_drum_fill(tracks, transition_beat, profile, style)` (roll `1/8 -> 1/16 -> 1/32` + crescendo `55 -> 112`, toms/ghost notes, pickup jazz, no-op ambient), `add_transition_fx` (riser `300 Hz -> 9 kHz` + silence `0.25 beat` + impact filtré vers `FX Hall 22`) et `apply_harmonic_anticipation` (accords/basse anticipés d'une croche, note précédente raccourcie, zéro chevauchement). Types JSON acceptés dans `arrangement_plan.json` (top-level `transitions` ou `sections[].transitions[]`) : `drum_fill`, `riser_impact`, `drum_fill_and_riser`, `harmonic_anticipation`, `impact`.

---

## Chapitre 2 — Cohérence Multi-Échelles, Plafond Cognitif & Physique (`T_noire`)

* **Multi-échelles** : Échantillon (`< 5 ms`, zéro clic, `440 Hz ± 5 ct`) → Transitoire (`5–50 ms`, attaques concurrentes `<= 15 %`) → Motif (`1–2 temps`) → Phrase (`4 mes.`) → Section (`8–16 mes.`) → Macro (`section_lufs`).
* **Plafond cognitif (`5–8` pistes simultanées par mesure vs isolation `1 sonorité = 1 piste`)** : le cerveau ne suit que `3–5` foyers ; plafonner à `5–8` pistes **jouant simultanément** (`1` Intro → `8` Climax), indépendamment du nombre total de pistes du projet Gridline (`9–14` pistes dédiées autorisées — ne jamais fusionner des pistes !).

| Catégorie | Quota Simultané | Bande | Règle d'Exclusion & Priorité |
| :--- | :---: | :---: | :--- |
| **1. Fondation (Low-End)** | **1–2** | `20–200 Hz` | Sidechain / décalage Kick–Basse. |
| **2. Rythmique (Groove)** | **2–4** | `150 Hz–16 kHz` | Cases complémentaires (`Kick`, `Snare`, `Hats`). |
| **3. Harmonie (Tapis)** | **1–2** | `200 Hz–3 kHz` | Creux EQ `1–3 kHz` sous le Lead. |
| **4. Élément Principal (Lead)** | **1** (max 2) | `500 Hz–5 kHz` | Domine l'harmonie de `>= +3 dB` (ou question-réponse). |
| **5. Habillage (Ear Candy)** | **1–2** | Filtré | Présence **< 15 %** de la durée totale. |

* **Calibrage sur `T_noire = 60 / BPM`** : Release 808 `35–55 ms` (`Voices=1`) · Sidechain Release `65–80 ms` (`trap_rap`, `2.4:1`, `-18 dB`) ou `80–110 ms` · `FX Room` Decay `<= T_noire` rythmique (`0.20–0.42 s`) ou `0.7–1.1 s` acoustique/ambient (`gl_dsl.reverb_room`).

---

## Chapitre 3 — Loi de la Réduction Unique (`master_sketch.mid`)

1. Ne **jamais** composer directement dans Gridline sans `master_sketch.mid` (3 pistes : `Bass`, `Chords`, `Lead`) généré en beats 0-indexed avec `optimize_voice_leading(chord_pcs, 48, 67)`.
2. Valider via `verify_midi_gate(..., time_signature=(num, den), phrase_plan=..., auto_fix=True)`.
3. Éclater la partition vers Gridline avec **`song_builder.py`** en respectant **`1 sonorité = 1 canal + 1 insert + patterns dédiés + piste Playlist nommée`**.

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

## Chapitre 5 — Gate MIDI Multi-Signatures, `_infer_roles()` & Autofix (`midi_autofix.py`)

### 5.1 Seuils Vérifiés (`musicality_metrics.py` & `deep_diagnostics.py`)
`audit_midi_harmony` et `verify_midi_gate` supportent désormais **`time_signature`** (`4/4`, `3/4`, `6/8`, `5/4`, temps forts & blocs 4 mesures dynamiques) et **`phrase_plan`** :
* Si `phrase_plan == "free_ambient"`, `melody_rest_ratio` tolère `[0.0, 0.92]`.
* Si aucune piste `bass` ou `lead` explicite n'existe (piano/cordes solo), **`contrary_or_oblique_ratio` retourne `None`** sans faux négatif.

| Métrique | Consonant (`V >= -0.3`, `T < 0.5`) | Sombre / Tendu (`V < -0.3` ou `T >= 0.5`) |
| :--- | :---: | :---: |
| **`max_pitch` (Lead) / `min_pitch` (808)** | `<= 76` / `>= 31` (`>= 49 Hz`) | **`<= 72–74`** / **`>= 31`** |
| **`strong_beat_chord_tone_ratio`** *(temps forts `num/den`)* | `>= 0.80` | **`>= 0.65`** *(appoggiatures `b6->5`, `b2->1`)* |
| **`minor_2nd_clashes`** / **`sub_collisions`** | `== 0` / `[]` | **`<= 2` / bloc 4 mes.** / `[]` |
| **`stepwise_ratio`** / **`gap_fill_compliance`** | `>= 0.65` / `>= 0.60` | `>= 0.65` / `>= 0.60` |
| **`melody_rest_ratio`** / **`voice_leading_avg_semitones`** | `0.20–0.50` *(0–0.92 ambient)* / `<= 2.50 st` | `0.20–0.50` / `<= 2.50 st` |
| **`lead_overlap_conflict`** / **`contrary_or_oblique_ratio`** | `<= 0.15` / `>= 0.55` *(ou `None`)* | `<= 0.15` / `>= 0.55` *(ou `None`)* |

### 5.2 Nommage `_infer_roles()` & Correction Déterministe (`scripts/midi_autofix.py`)
* **Noms reconnus** : Perc/FX ignorés (`kick, snare, clap, hat, perc, rim, cymbal, crash, tom, shaker, drum, fx, riser`) · `bass` (`808, bass, sub` ou pitch `< 48` hors accords `poly_ratio >= 0.45`) · `lead` (`lead, melody, topline, vox, vocal` ou `"piano"` si `poly_ratio < 0.45`) · `chords` (`chord, pad, string, rhodes, harmony` ou `poly_ratio >= 0.45`).
* **Autofix** : Appeler `verify_midi_gate(..., time_signature=(num, den), phrase_plan=..., auto_fix=True)` ou `autofix_midi(...)`. Corrige en 1 passe + ré-audit : hors-gamme, bornes sub/apex, collisions `< MIDI 48` dans `Chords`, appuis temps forts et frottements `b2` (`fix_report`). Si un défaut structurel persiste (`melody_rest_ratio`, `stepwise`, `contrary`), ajuster le générateur Python.

---

## Chapitre 6 — Patron Canonique `build_song.py` (`song_builder` + `gl_dsl` + `groove_lib`)

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, "$AI_STUDIO_DIR/scripts")
from gl_dsl import eq_band, eq_params
from groove_lib import apply_anti_machine_gun, apply_haas_pocket, apply_micro_strum, enforce_frequency_inertia
from song_builder import ChannelSpec, degree_to_midi, optimize_voice_leading, run_song_cli
from transitions_lib import add_transition_fx, apply_harmonic_anticipation, beat4_dropout, generate_drum_fill, outro_fade, section_gain_arc

PROJ_DIR = "$AI_STUDIO_DIR/projects/mon_morceau"
SECTIONS = [("Intro", 0, 8, "intro"), ("Drop1", 8, 16, "chorus"), ("Outro", 24, 8, "outro")]  # start_bar 0-idx

def generate(meta):
    voicings = optimize_voice_leading([[0, 3, 7], [8, 0, 3], [5, 8, 0], [0, 3, 7]], 48, 65)
    tracks = {"kick": [], "808": [], "hihat": [], "chords": [], "lead": []}
    # ... remplir en beats 0-indexed, puis humaniser :
    tracks["chords"] = apply_micro_strum(tracks["chords"], step_ms=15.0, bpm=meta["bpm"])
    tracks["hihat"] = apply_anti_machine_gun(tracks["hihat"])
    tracks["808"] = enforce_frequency_inertia(tracks["808"])
    apply_haas_pocket(tracks, bpm=meta["bpm"], profile=meta["mix_profile"])
    markers = [{"name": s, "position": float(b * meta["beats_per_bar"])} for s, b, _, _ in SECTIONS]
    return tracks, markers

CHANNELS = [
    ChannelSpec("Kick", "kick", 1, "drums", volume=0.72, plugin="gl.kick", sidechain_buses=[(18, 0.8), (19, 0.5)]),
    ChannelSpec("808", "808", 2, "bass", role="bass", valence=-0.6, volume=0.52, plugin="gl.warmsynth"),
    ChannelSpec("Chords", "chords", 3, "music", role="chords", valence=-0.6, volume=0.45, plugin="gl.keys", fx_send=0.14),
    ChannelSpec("Lead", "lead", 4, "music", role="lead", valence=-0.6, volume=0.72, plugin="gl.fm", fx_send=0.16),
]

if __name__ == "__main__":
    run_song_cli(
        proj_dir=PROJ_DIR, out_gridline=f"{PROJ_DIR}/projet.gridline", sections=SECTIONS, channels=CHANNELS,
        generate_fn=generate, bass_keys=["808"], chords_key="chords", lead_key="lead",
        insert_eq_params=eq_params(1, [eq_band(1, "hpf", 30)]) + eq_params(3, [eq_band(1, "hpf", 220), eq_band(4, "peak", 1800, -3.0, q=1.4)]),
    )
```


---

## Phase 6 — Contrat Multi-Échelles ($S5 \rightarrow S1$) & Cohérence Macro ($S4$)

1. **Contrat normatif (`multi_scale_contract.json`)** : généré en **É0** via `scripts/target_fingerprint.py` (ou MCP `audio-analyzer:extract_target_fingerprint`) à partir de `0..3` audio de référence et du schéma `schemas/multi_scale_contract.schema.json`.
2. **Synchronisation $S3$ (Syntaxe & Forme Courte)** : `audit_midi_harmony` / `evaluate_midi_musicality(..., strict_s3=True)` vérifie la cohésion motivique (`A-A'-A''-B`, rejetant clones robotiques `> 0.92` et phrases sans cohésion `< 0.08`), l'arc de tension harmonique sur 4 mesures et l'interlocking Accords/Lead (`chords_lead_attack_overlap <= 0.25`). Corrections mécaniques via `midi_autofix.autofix_midi(..., fix_s3=True)`.
3. **Audit Macro-Structurel $S4$ (`scripts/macro_structure_audit.py`)** :
   - **SSM bar-synchronisée** (`0.35 Chroma CQT + 0.40 MFCC 1..13 + 0.25 Rythme 16 pas`) : tout bloc de 8 mesures cloné (`S_bar8 > 0.94`) déclenche `MUTATE_BLOCK_DENSITY_OR_LPF`.
   - **Nouveauté Checkerboard ($L=4$ mes.)** : chaque transition déclarée dans `song_blueprint.json` doit présenter un pic de nouveauté à `±1` mesure (`REINFORCE_SECTION_TRANSITION`).
   - **Arc Dramatique 5D** `(LUFS, Centroïde, Densité d'onsets, Largeur M/S, Flux cognitifs)` : valide l'évolution de genres sans batterie (`has_drums=false` / `classical_ambient`).
   - **Plafond Cognitif sur stems (`> -28 dBFS` RMS/mesure)** : vérifie le respect des quotas d'arrangement par section (`PRUNE_TRACK_IN_SECTION` si dépassement sur `> 2` mesures consécutives).
4. **Catalogue & EP (`scripts/catalog_coherence.py`)** : sur chaque `É4 PASS`, `render_and_audit.py` exporte `<projet>/renders/master_fingerprint.json` ($\approx 1\text{ Ko}$), audité ensuite en lot par mode `ep|album|dj_set` (homogénéité LUFS `<= 1.5 dB`, pente rose `<= 1.2 dB/oct`, diversité BPM/clé/progression, ordre optimal Camelot).

---

## Phase 7 — Plan Déclaratif, Prosodie (`P2`), Synchro Vocale (`P6`) & Checkpoint É2 (`P5`)

1. **Plan d'Arrangement Déclaratif (`schemas/arrangement_plan.schema.json` + `compile_arrangement_plan`)** :
   - L'agent génère `arrangement_plan.json` (sections, `active_roles`, `role_modifiers`, `transitions`, `motifs`), validé puis compilé de manière déterministe par `song_builder.compile_arrangement_plan()`. Exemple de transition riche : `{"transitions": [{"bar": 9, "type": "drum_fill_and_riser", "style": "snare_roll"}, {"bar": 17, "type": "harmonic_anticipation", "offset_beats": 0.5, "ratio": 0.35}]}`.
   - Permet à `multiscale_autofix.py` (`PRUNE_TRACK_IN_SECTION`, `REINFORCE_SECTION_TRANSITION`, `MUTATE_BLOCK_DENSITY_OR_LPF`) de muter localement le plan JSON et de recompiler sans régénération LLM.
2. **Prosodie Instrumentale (`scripts/phrasing_engine.py`)** :
   - `apply_instrumental_prosody()` façonne chaque cycle de 4 mesures (arc de vélocités crescendo→décroissance + courbe d'automation LPF coordonnée), ajoute les notes de passage basse en mesure `4/8` et appoggiatures cadentielles sur le `Lead` (sécurisées par `detect_minor_2nd_clashes`), et décale le micro-pocket de section (`+3.5 ms` couplets / `-2.0 ms` refrains).
3. **Synchronisation Vocale (`scripts/vocal_arrangement_sync.py`)** :
   - En présence d'un fichier `.ustx` ou `samples/vocals.wav`, `extract_vocal_phrases()` isole les phrases chantées, `mute_counter_during_vocals()` restreint les contre-chants aux fenêtres respiratoires `>= 0.5 beat`, et `build_formant_ducking_automation()` automatise un creux `1.85 kHz` (`Band 5 gain` `0 -> -3.5 dB`) sur `BUS Music`.
4. **Checkpoint Humain 15 s (`--checkpoint-e2` / `--resume-from-checkpoint`)** :
   - Exporte `renders/e2_checkpoint_15s.wav` normalisé à `-14 LUFS`, met `pipeline_state.json` en `AWAITING_E2_CHECKPOINT` avec fiche 5 lignes (`<= 300` tokens), puis reprend en `É3/É4` sans recalculer `É1/É2` via le cache SHA-256.
