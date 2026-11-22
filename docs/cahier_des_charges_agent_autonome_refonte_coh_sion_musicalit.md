# Cahier des Charges d'Exécution : Refonte de la Cohésion Inter-Pistes & de la Musicalité Mélodique

**Destinataire :** Agent de Développement Autonome (Maintenance Docs, Schémas, MCP `audio-analyzer` & Refonte Projets Gridline)  
**Périmètre :** Studio IA Linux (`/home/gamo/Bureau/musicos` / `~/Music/AI-Studio`) — Gridline `0.7.7`, MCP `audio-analyzer`, `audio-index`, `schemas/song_blueprint.schema.json`, documentation & projets actifs.  
**Priorité :** Critique (résolution définitive de l'effet « cafouillis / collage de sons » et du manque de musicalité mélodique).

---

## 0. Synthèse du Problème & Objectif de la Mission

### 0.1 Constat Audible
Les morceaux générés jusqu'ici valident les métriques d'ingénierie audio (`LUFS`, `True Peak`, corrélation de phase, absence de clipping, rythme Trap *half-time* cohérent), mais présentent deux défauts majeurs à l'écoute :
1. **Absence de lien vertical et horizontal entre les pistes** : les instruments sonnent comme des couches juxtaposées plutôt que comme un ensemble organique.
2. **Faiblesse mélodique** : les notes restent dans la gamme sans créer de motif chantable, mémorable ou émotionnellement dirigé.

### 0.2 Les 4 Causes Racines Techniques à Corriger
1. **Angle mort de `audit_midi_harmony`** : l'outil ne vérifie que l'appartenance à la gamme, la hauteur maximale (`max_pitch`) et les clashs de seconde/neuvième mineure. Une suite de notes aléatoires dans la gamme passe à `100 %` tout en sonnant anti-musicale.
2. **Composition piste par piste (*Bottom-Up* fragmenté)** : la 808, les accords, le piano et les contre-chants sont générés séparément. Ils partagent une grille théorique mais aucune **conduite de voix** (*voice leading*) ni aucune **dérivation motivique**.
3. **Bug dans `harmonisation_verticale_et_horizontale.md` (Règle V3)** : la règle actuelle autorise la transposition MIDI parallèle d'un one-shot d'accord (ex. un stab d'accord mineur `Em` transposé sur `C`, `G`, `D`). Cela génère des accords mineurs parallèles hors gamme (`Cm`, `Gm`, `Dm`) qui détruisent la tonalité et l'unité harmonique.
4. **Ciment timbral et dynamique insuffisant** : le seul partage d'une `FX Room` et d'un accordage `440 Hz` ne suffit pas à souder des sources hétérogènes (stabs, SFZ, synthés, 808). Il manque une **enveloppe spectrale commune**, une **saturation collective subtile** et un **creusement dynamique** (*ducking* / EQ miroir) piloté par le leader.

### 0.3 Solution Universelle : Le Principe de la « Réduction Unique » (*Single-Core Composition*)
Tout morceau doit désormais être composé en **Top-Down** :
1. Écrire **toute la musique** (basse + harmonie en *voice leading* + mélodie principale) dans une **partition mère unique** (`master_sketch.mid`) jouée sur un seul piano acoustique sec (`Headroom Piano Clean.sfz`, sans batterie ni effets).
2. Faire passer ce `master_sketch.mid` par un **Gate bloquant** dans `audio-analyzer:audit_midi_harmony` (6 nouvelles métriques de musicalité et de cohésion).
3. Une fois le Gate validé (`PASS`), **éclater (orchestrer)** les voix de cette partition unique vers les canaux Gridline (808, Pad/Chords, Lead/Pluck, Contre-chant dérivé), puis appliquer la chaîne de ciment timbral sur `BUS Music` et `FX Room`.

---

## PHASE 1 : Mises à Jour Documentaires Obligatoires

L'agent de développement doit exécuter les opérations documentaires suivantes avant toute modification de morceau.

### 1.1 Créer `docs/regles_melodie_et_cohesion_inter_pistes.md`
Créer ce document de référence contenant les règles normatives suivantes :

#### A. Loi Fondamentale — La Réduction Unique (`master_sketch.mid`)
* **Interdiction absolue** de programmer la basse, les accords et les mélodies dans des scripts indépendants sans partition mère préalable.
* Structure stricte de `master_sketch.mid` (3 pistes MIDI ou 3 couches de registres sur piano sec) :
  * **Voix 1 — Basse (`Bass`)** : monophonique, registre `C2–B2` (MIDI `36–47`). Joue la fondamentale de l'accord actif + notes d'approche en fin de mesure 4/8.
  * **Voix 2 — Harmonie (`Chords`)** : 2 à 3 notes simultanées, registre `C3–B3` (MIDI `48–59`). Conduite des voix par renversements minimaux.
  * **Voix 3 — Top-Line (`Lead`)** : monophonique, registre `C4–G5` (MIDI `60–79`). Mélodie principale chantable.
* **Test de la Nudité** : rendre `master_sketch.mid` sur `Headroom Piano Clean.sfz` seul. Si la réduction ne sonne pas claire et mélodieuse sans batterie, la réécrire avant d'ajouter le moindre drum.

#### B. Grammaire de la Mélodie (Règles M1 à M5)
* **M1 — Mouvement Conjoint & Loi du *Gap-Fill* (70 / 30)** :
  * Au moins **$65\text{ \%}$ à $80\text{ \%}$** des intervalles consécutifs du Lead doivent être $\le 4$ demi-tons (répétition, demi-ton, ton, tierce).
  * Tout saut large ($\ge 5$ demi-tons) doit être suivi d'un retour conjoint ($\le 2$ demi-tons) dans la **direction opposée**.
  * Ambitus maximal du Lead : **$12$ à $19$ demi-tons** (1 à 1,5 octave), comme une voix humaine.
* **M2 — Économie Motivique (`A - A' - A'' - B`)** :
  * Définir une **cellule mère** unique de $3$ à $5$ notes sur $1$ à $2$ temps.
  * **Mesure 1 (`A`)** : exposition de la cellule sur l'accord 1.
  * **Mesure 2 (`A'`)** : même rythme exact, adapté aux notes de l'accord 2.
  * **Mesure 3 (`A''`)** : variation montante atteignant l'**Apex** (la note la plus aiguë de la phrase, atteinte une seule fois sur les 4 mesures).
  * **Mesure 4 (`B`)** : détente / cadence conclusive (résolution vers la tonique/quinte ou suspension sur la 5e/7e).
* **M3 — Respiration Obligatoire (Ratio Son / Silence)** :
  * Maintenir **$20\text{ \%}$ à $50\text{ \%}$** de silence réel dans la piste Lead.
  * Libérer systématiquement le dernier temps ou demi-temps des mesures 2 et 4 pour laisser respirer la batterie et le contre-chant.
* **M4 — Gravité Harmonique (Temps Forts vs Temps Faibles)** :
  * Sur les temps forts (`beat 0.0`, `beat 2.0` et chaque attaque d'accord), au moins **$80\text{ \%}$** des notes du Lead doivent appartenir à la triade ou 7e de l'accord courant.
  * Les notes étrangères à l'accord (notes de passage, broderies) sont confinées aux subdivisions faibles (`0.5`, `1.5`, `2.5`, `3.5`) et doivent se résoudre par mouvement conjoint ($\pm 1$ ou $\pm 2$ demi-tons).
* **M5 — Couplage Rythmique Mélodie / Groove (*Pocket Alignment*)** :
  * Ancrer les attaques clés du Lead sur le temps `1` (ou en anacrouse juste avant `1`) et créer un appui ou une respiration autour du **Clap/Snare sur le temps `3`** (grille Trap *half-time*).

#### C. Harmonisation Verticale Stricte (Règles V1 à V6)
* **V1 — Conduite des Voix Minimale (*Smooth Voice Leading*)** :
  * Interdiction des accords en position fondamentale qui sautent en bloc.
  * **Note Pivot** : toute note commune entre l'accord $N$ et l'accord $N+1$ reste à la même hauteur MIDI.
  * **Moindre Mouvement** : le déplacement moyen des voix entre deux accords consécutifs doit être **$\le 2.5$ demi-tons**.
* **V2 — Interdiction de Transposition Parallèle d'Accords Samplés** :
  * Ne jamais transposer un sample audio contenant déjà un accord complet (majeur ou mineur) sur tous les degrés d'une gamme.
  * Utiliser exclusivement : (a) un instrument polyphonique (`Headroom Piano Clean.sfz`, `jRhodes3c`, `gl.dcosynth`, `gl.analog`), (b) un one-shot **monophonique** rejoué en polyphonie MIDI, ou (c) un *power chord* (fondamentale + quinte sans tierce).
* **V3 — Mouvement Contraire & Oblique (Basse vs Lead)** :
  * Lorsque la basse (`808`) monte, la mélodie (`Lead`) tend à descendre ou rester stable (et inversement). Interdiction de doubler exactement la ligne de 808 à l'octave sur plus de 2 temps consécutifs.
* **V4 — Complémentarité d'Enveloppes & Relais d'Attaques** :
  * Jamais deux attaques percussives (`Attack < 15 ms`) simultanées dans la même octave des médiums (`400 Hz – 3 kHz`).
  * Si deux instruments coexistent : opposer une attaque rapide (*Pluck / Piano staccato*) à une attaque douce (*Pad / Strings*, `Attack > 40 ms`), ou alterner en Question/Réponse (*Call-and-Response*).
  * Le chevauchement d'attaques simultanées entre deux pistes mélodiques dans la même octave doit être **$\le 15\text{ \%}$**.
* **V5 — Égalisation Miroir Dynamique (*Dynamic Carve*)** :
  * Creuser dynamiquement de $-2$ à $-4\text{ dB}$ la zone `800 Hz – 3 kHz` des pistes d'accompagnement (`Chords / Pad`) uniquement lorsque le `Lead` ou la `Voix` joue (via `TDR Nova` / `ZL Equalizer 2` en Live, ou automation / EQ statique complémentaire `gl.eq` en Headless).
* **V6 — Ciment Timbral & Acoustique Partagé (*Sonic Glue*)** :
  * `BUS Music` regroupe toutes les pistes harmoniques/mélodiques et applique :
    1. **Coupe-haut unificateur** : `LPF` doux à `11–14 kHz` ($12\text{ dB/oct}$) pour unifier la texture et laisser l'air $> 12\text{ kHz}$ aux hi-hats.
    2. **Saturation harmonique collective** : très légère saturation (`CHOWTapeModel` en Live ou `gl.dist` mode `Soft`, `Mix 8–12%`, `Oversampling 4x` en natif 100 % portable).
    3. **Compression de bus** : `gl.comp` (`Ratio 2.0:1`, `Attack 20 ms`, `Release 110 ms`, `GR 1.5–2.5 dB`).
    4. **Micro-Ducking rythmique** : léger sidechain du `Kick` vers `BUS Music` (`-1.5` à `-2.5 dB`, `Release = 1/8` de temps) pour faire respirer l'harmonie sur le groove.
    5. **Pièce acoustique unique (`FX Room`)** : tous les instruments mélodiques (`8–14%`) + Snare/Clap (`3–6%`) dans le même `gl.reverb` court (`Decay <= 1 × T_noire`, `HPF >= 450 Hz`, `Dry = 0`, `Dry (full) = 0`).

#### D. Harmonisation Horizontale (Règles H1 à H3)
* **H1 — Dérivation Thématique Unique** : ne jamais inventer une mélodie sans rapport entre Intro, Couplet, Pont et Refrain. Faire évoluer le même motif par 4 leviers uniquement : **octave** (`-12` / `+12`), **densité** (squelette sur temps forts en couplet vs motif complet au refrain), **filtrage** (`LPF` fermé → ouvert), et **écho** (contre-chant reprenant les 2 dernières notes du Lead dans son silence).
* **H2 — Ponts de Conduite de Voix (*Voice-Leading Bridges*)** : sur le 4e temps des mesures 4 et 8, utiliser une note de liaison (5e, 7e ou sensible) dans la basse ou le Lead qui résout par demi-ton ou ton sur le temps `1` suivant.
* **H3 — Contraste Dynamique Inter-Sections** : garantir $3$ à $5\text{ dB}$ d'écart LUFS réel entre Intro/Pont (`-15.5` à `-14.0 LUFS`) et Refrains (`-11.0` à `-10.0 LUFS`).

---

### 1.2 Corriger `harmonisation_verticale_et_horizontale.md`
* **Section 2 — Règle V3** : supprimer la phrase autorisant la transposition MIDI d'un sample d'accord complet. Remplacer par un renvoi explicite à la règle **V2** ci-dessus (interdiction de transposition parallèle d'accords majeurs/mineurs samplés) et à la règle **V1** (*Smooth Voice Leading*).
* **Section 2 — Ajouter V6** : intégrer le ciment timbral (`LPF` collectif `BUS Music`, saturation douce collective, micro-ducking Kick → `BUS Music`).
* **Section 4 — Checklist Anti-Cafouillis** : ajouter les cases de validation `master_sketch.mid` et les 6 métriques de `audit_midi_harmony`.

### 1.3 Mettre à Jour `.rules`, `GEMINI.md`, `AGENTS.md` et `apprentissage.md`
* Ajouter dans le workflow obligatoire :
  1. Remplir `harmonic_core` dans `song_blueprint.json`.
  2. Générer `master_sketch.mid` et lancer `audio-analyzer:audit_midi_harmony(midi_path=".../master_sketch.mid", ...)` avec `strict_musicality=True`.
  3. **Gate Bloquant** : tant que `musicality_verdict != "PASS"`, interdiction d'instancier la batterie ou de passer au mixage.
  4. Éclater `master_sketch.mid` vers les canaux Gridline et appliquer la chaîne `BUS Music` Glue.
* Documenter dans `apprentissage.md` (nouvelle section) les causes du « syndrome du collage » et la nouvelle procédure vérifiée.

---

## PHASE 2 : Extension de `schemas/song_blueprint.schema.json`

Ajouter la propriété requise `harmonic_core` dans `schemas/song_blueprint.schema.json` afin que l'architecture harmonique et mélodique soit figée mathématiquement avant toute génération dans Gridline.

```json
{
  "harmonic_core": {
    "type": "object",
    "required": [
      "key",
      "mode",
      "progression_bars",
      "chord_progression",
      "motif_cell",
      "phrase_plan"
    ],
    "properties": {
      "key": {
        "type": "string",
        "description": "Tonique exacte, ex: 'E', 'D', 'C', 'Eb'"
      },
      "mode": {
        "type": "string",
        "enum": ["aeolian", "phrygian", "harmonic_minor", "dorian"]
      },
      "progression_bars": {
        "type": "integer",
        "enum": [2, 4, 8]
      },
      "chord_progression": {
        "type": "array",
        "minItems": 1,
        "items": {
          "type": "object",
          "required": ["bar", "symbol", "bass_midi", "voicing_midi"],
          "properties": {
            "bar": { "type": "integer", "minimum": 1 },
            "symbol": { "type": "string", "description": "Ex: 'Em7', 'Cmaj7', 'Am9', 'B7'" },
            "bass_midi": {
              "type": "integer",
              "minimum": 28,
              "maximum": 55,
              "description": "Fondamentale 808 (dans la plage 40-78 Hz idéale)"
            },
            "voicing_midi": {
              "type": "array",
              "minItems": 2,
              "maxItems": 4,
              "items": { "type": "integer", "minimum": 48, "maximum": 76 },
              "description": "Notes MIDI déjà renversées pour voice leading minimal (<= 2.5 st moyen)"
            },
            "chord_tones_pc": {
              "type": "array",
              "items": { "type": "integer", "minimum": 0, "maximum": 11 },
              "description": "Pitch classes (0=C..11=B) autorisées sur temps forts"
            }
          }
        }
      },
      "motif_cell": {
        "type": "object",
        "required": ["intervals_from_tonic", "rhythm_beats", "apex_midi"],
        "properties": {
          "intervals_from_tonic": {
            "type": "array",
            "minItems": 3,
            "maxItems": 6,
            "items": { "type": "integer" }
          },
          "rhythm_beats": {
            "type": "array",
            "minItems": 3,
            "maxItems": 6,
            "items": { "type": "number" }
          },
          "apex_midi": {
            "type": "integer",
            "minimum": 60,
            "maximum": 84,
            "description": "Sommet mélodique unique de la phrase (mesure 3)"
          }
        }
      },
      "phrase_plan": {
        "type": "string",
        "enum": ["A-A'-A''-B", "A-B-A'-C", "A-A'-B-B'"],
        "default": "A-A'-A''-B"
      }
    }
  }
}
```

---

## PHASE 3 : Mise à Jour du Serveur MCP `audio-analyzer` (`audit_midi_harmony`)

### 3.1 Spécification des 6 Nouvelles Métriques Bloquantes

Ajouter le support de ces métriques dans l'outil MCP `audit_midi_harmony` (tout en conservant les vérifications existantes de gamme, `max_pitch`, et clashs de 2nde/9ème mineure) :

| Métrique | Définition Mathématique | Seuil `PASS` | Cause si `FAIL` |
| :--- | :--- | :--- | :--- |
| **`stepwise_ratio`** | Proportion d'intervalles consécutifs du Lead avec $|\Delta p| \le 4$ demi-tons | `>= 0.65` | Mélodie erratique / sauts aléatoires. |
| **`gap_fill_compliance`** | Proportion de sauts $|\Delta p| \ge 5$ suivis d'un mouvement conjoint ($\le 2$ st) en sens inverse | `>= 0.60` | Sauts mélodiques non résolus. |
| **`melody_rest_ratio`** | $1 - \frac{\text{durée couverte par le Lead}}{\text{durée totale active}}$ | `0.20` à `0.50` | Mélodie asphyxiante (`< 0.20`) ou vide (`> 0.50`). |
| **`strong_beat_chord_tone_ratio`** | Proportion de notes du Lead sur temps forts (`beat % 2.0 < 0.15`) appartenant à l'accord actif | `>= 0.80` | Mélodie déconnectée de l'harmonie verticale. |
| **`voice_leading_avg_semitones`** | Distance minimale moyenne (en demi-tons) entre les voix de deux accords consécutifs | `<= 2.50` | Accords plaqués qui sautent en bloc sans notes pivots. |
| **`lead_overlap_conflict`** | Proportion d'attaques simultanées ($\pm 40\text{ ms}$) entre 2 pistes mélodiques à moins de $12$ demi-tons d'écart | `<= 0.15` | Cafouillis d'attaques concurrentes (non respect du relais). |
| **`contrary_or_oblique_ratio`** | Proportion de mouvements Bass vs Lead qui sont contraires ($\Delta b \cdot \Delta m < 0$) ou obliques ($\Delta b = 0$ ou $\Delta m = 0$) | `>= 0.55` | Parallélisme mécanique basse/mélodie. |

### 3.2 Module Python Prêt à Intégrer (`musicality_metrics.py`)

L'agent de développement doit intégrer ce module dans `mcp-servers/audio-analyzer/` et l'appeler depuis `audit_midi_harmony`. Il n'introduit aucune dépendance externe nouvelle (utilise uniquement `pretty_midi` et `numpy`, déjà présents dans l'environnement).

```python
"""
Métriques objectives de musicalité mélodique et de cohésion inter-pistes
pour audio-analyzer:audit_midi_harmony.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pretty_midi


@dataclass
class MusicalityThresholds:
    stepwise_ratio_min: float = 0.65
    gap_fill_min: float = 0.60
    rest_ratio_min: float = 0.20
    rest_ratio_max: float = 0.50
    strong_beat_chord_ratio_min: float = 0.80
    voice_leading_max_st: float = 2.50
    lead_overlap_max: float = 0.15
    contrary_oblique_min: float = 0.55


def _infer_roles(pm: pretty_midi.PrettyMIDI) -> Dict[str, List[pretty_midi.Instrument]]:
    """Classe automatiquement les pistes non-percussives par rôle (bass, chords, lead, counter)."""
    melodic_insts = [inst for inst in pm.instruments if not inst.is_drum and len(inst.notes) > 0]
    roles: Dict[str, List[pretty_midi.Instrument]] = {"bass": [], "chords": [], "lead": [], "counter": []}

    for inst in melodic_insts:
        name = (inst.name or "").lower()
        pitches = np.array([n.pitch for n in inst.notes], dtype=float)
        mean_p = float(np.mean(pitches))

        # Estimation de polyphonie moyenne (notes qui se chevauchent)
        notes_sorted = sorted(inst.notes, key=lambda n: n.start)
        overlaps = sum(
            1
            for i in range(len(notes_sorted) - 1)
            if notes_sorted[i + 1].start < notes_sorted[i].end - 0.02
        )
        poly_ratio = overlaps / max(1, len(notes_sorted))

        if any(k in name for k in ("808", "bass", "sub")) or mean_p < 48.0:
            roles["bass"].append(inst)
        elif any(k in name for k in ("chord", "pad", "string", "rhodes", "harmony")) or poly_ratio >= 0.45:
            roles["chords"].append(inst)
        elif any(k in name for k in ("lead", "melody", "topline", "vox", "vocal", "piano")):
            roles["lead"].append(inst)
        else:
            roles["counter"].append(inst)

    # Fallback si aucun lead explicite : prendre la piste non-basse la plus aiguë
    if not roles["lead"] and (roles["counter"] or roles["chords"]):
        candidates = roles["counter"] if roles["counter"] else roles["chords"]
        highest = max(candidates, key=lambda i: np.mean([n.pitch for n in i.notes]))
        roles["lead"].append(highest)
        if highest in roles["counter"]:
            roles["counter"].remove(highest)

    return roles


def _extract_monophonic_topline(notes: List[pretty_midi.Note]) -> List[pretty_midi.Note]:
    """Extrait la ligne supérieure si la piste contient de légers chevauchements."""
    if not notes:
        return []
    ordered = sorted(notes, key=lambda n: (round(n.start, 3), -n.pitch))
    topline: List[pretty_midi.Note] = []
    for n in ordered:
        if topline and abs(n.start - topline[-1].start) < 0.03:
            continue
        topline.append(n)
    return topline


def _compute_stepwise_and_gap_fill(topline: List[pretty_midi.Note]) -> Tuple[float, float, int]:
    if len(topline) < 2:
        return 1.0, 1.0, 0
    diffs = [topline[i + 1].pitch - topline[i].pitch for i in range(len(topline) - 1)]
    stepwise = sum(1 for d in diffs if abs(d) <= 4)
    stepwise_ratio = stepwise / len(diffs)

    leaps = 0
    filled = 0
    for i, d in enumerate(diffs[:-1]):
        if abs(d) >= 5:
            leaps += 1
            nxt = diffs[i + 1]
            # Mouvement contraire conjoint (1 à 3 demi-tons)
            if (d * nxt < 0) and (1 <= abs(nxt) <= 3):
                filled += 1
    gap_fill = (filled / leaps) if leaps > 0 else 1.0
    ambitus = max(n.pitch for n in topline) - min(n.pitch for n in topline)
    return round(stepwise_ratio, 3), round(gap_fill, 3), int(ambitus)


def _compute_rest_ratio(notes: List[pretty_midi.Note], total_duration: float) -> float:
    if not notes or total_duration <= 0:
        return 1.0
    intervals = sorted((n.start, n.end) for n in notes)
    merged: List[List[float]] = []
    for s, e in intervals:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    covered = sum(e - s for s, e in merged)
    return round(max(0.0, min(1.0, 1.0 - covered / total_duration)), 3)


def _extract_chord_slices(chord_notes: List[pretty_midi.Note], quantize_s: float = 0.08) -> List[Tuple[float, List[int]]]:
    """Regroupe les notes d'accords simultanées en tranches (start_time, pitches_sorted)."""
    if not chord_notes:
        return []
    ordered = sorted(chord_notes, key=lambda n: n.start)
    slices: List[Tuple[float, List[int]]] = []
    cur_t = ordered[0].start
    cur_pitches = [ordered[0].pitch]
    for n in ordered[1:]:
        if n.start - cur_t <= quantize_s:
            cur_pitches.append(n.pitch)
        else:
            slices.append((cur_t, sorted(set(cur_pitches))))
            cur_t = n.start
            cur_pitches = [n.pitch]
    slices.append((cur_t, sorted(set(cur_pitches))))
    return [s for s in slices if len(s[1]) >= 2]


def _compute_voice_leading(slices: List[Tuple[float, List[int]]]) -> float:
    """Calcule le déplacement moyen (en demi-tons) entre deux accords consécutifs."""
    if len(slices) < 2:
        return 0.0
    distances: List[float] = []
    for (_, prev_p), (_, curr_p) in zip(slices[:-1], slices[1:]):
        if prev_p == curr_p:
            continue
        # Pour chaque note du nouvel accord, distance à la note la plus proche de l'accord précédent
        d = float(np.mean([min(abs(c - p) for p in prev_p) for c in curr_p]))
        distances.append(d)
    return round(float(np.mean(distances)), 3) if distances else 0.0


def _compute_strong_beat_chord_tone_ratio(
    pm: pretty_midi.PrettyMIDI,
    topline: List[pretty_midi.Note],
    chord_notes: List[pretty_midi.Note],
    bass_notes: List[pretty_midi.Note],
) -> float:
    if not topline or (not chord_notes and not bass_notes):
        return 1.0
    beats = pm.get_beats()
    if len(beats) < 2:
        return 1.0
    strong_beats = beats[::2]  # Temps 1 et 3 (index 0, 2, 4...)

    strong_notes = [
        n for n in topline if np.min(np.abs(strong_beats - n.start)) <= 0.09
    ]
    if not strong_notes:
        return 1.0

    matches = 0
    for n in strong_notes:
        # Pitch classes actifs à l'instant n.start dans les accords + basse
        active_pcs = {
            c.pitch % 12
            for c in (chord_notes + bass_notes)
            if c.start - 0.05 <= n.start <= c.end + 0.05
        }
        if not active_pcs or (n.pitch % 12) in active_pcs:
            matches += 1
    return round(matches / len(strong_notes), 3)


def _compute_lead_overlap_conflict(
    lead_insts: List[pretty_midi.Instrument],
    counter_insts: List[pretty_midi.Instrument],
) -> float:
    """Détecte les attaques simultanées (<= 40 ms) dans la même octave (< 12 st) entre pistes mélodiques."""
    tracks = lead_insts + counter_insts
    if len(tracks) < 2:
        return 0.0
    total_attacks = sum(len(t.notes) for t in tracks)
    if total_attacks == 0:
        return 0.0
    conflicts = 0
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            for n1 in tracks[i].notes:
                for n2 in tracks[j].notes:
                    if abs(n1.start - n2.start) <= 0.04 and abs(n1.pitch - n2.pitch) < 12:
                        conflicts += 1
                        break
    return round(conflicts / max(1, total_attacks), 3)


def _compute_contrary_oblique_ratio(
    topline: List[pretty_midi.Note], bass_notes: List[pretty_midi.Note]
) -> float:
    if len(topline) < 2 or len(bass_notes) < 2:
        return 1.0
    bass_sorted = sorted(bass_notes, key=lambda n: n.start)

    def bass_at(t: float) -> Optional[int]:
        active = [b.pitch for b in bass_sorted if b.start - 0.05 <= t <= b.end + 0.05]
        return active[-1] if active else None

    total = 0
    good = 0
    for n1, n2 in zip(topline[:-1], topline[1:]):
        b1, b2 = bass_at(n1.start), bass_at(n2.start)
        if b1 is None or b2 is None:
            continue
        dm = n2.pitch - n1.pitch
        db = b2 - b1
        if dm == 0 and db == 0:
            continue
        total += 1
        # Contraire (dm * db < 0) ou oblique (un des deux reste stable)
        if (dm * db) <= 0:
            good += 1
    return round(good / total, 3) if total > 0 else 1.0


def evaluate_midi_musicality(
    midi_path: str,
    thresholds: Optional[MusicalityThresholds] = None,
) -> Dict[str, Any]:
    """Point d'entrée principal à appeler depuis audit_midi_harmony."""
    th = thresholds or MusicalityThresholds()
    pm = pretty_midi.PrettyMIDI(midi_path)
    total_dur = pm.get_end_time()
    roles = _infer_roles(pm)

    lead_notes = [n for inst in roles["lead"] for n in inst.notes]
    chord_notes = [n for inst in roles["chords"] for n in inst.notes]
    bass_notes = [n for inst in roles["bass"] for n in inst.notes]
    topline = _extract_monophonic_topline(lead_notes)

    stepwise_ratio, gap_fill, ambitus = _compute_stepwise_and_gap_fill(topline)
    rest_ratio = _compute_rest_ratio(topline, total_dur)
    slices = _extract_chord_slices(chord_notes)
    vl_avg = _compute_voice_leading(slices)
    strong_beat_ratio = _compute_strong_beat_chord_tone_ratio(pm, topline, chord_notes, bass_notes)
    overlap_conflict = _compute_lead_overlap_conflict(roles["lead"], roles["counter"])
    contrary_ratio = _compute_contrary_oblique_ratio(topline, bass_notes)

    failures: List[str] = []
    if stepwise_ratio < th.stepwise_ratio_min:
        failures.append(f"stepwise_ratio={stepwise_ratio} < {th.stepwise_ratio_min} (mélodie trop sautillante)")
    if gap_fill < th.gap_fill_min:
        failures.append(f"gap_fill_compliance={gap_fill} < {th.gap_fill_min} (sauts non résolus en sens inverse)")
    if not (th.rest_ratio_min <= rest_ratio <= th.rest_ratio_max):
        failures.append(f"melody_rest_ratio={rest_ratio} hors [{th.rest_ratio_min}, {th.rest_ratio_max}]")
    if strong_beat_ratio < th.strong_beat_chord_ratio_min:
        failures.append(f"strong_beat_chord_tone_ratio={strong_beat_ratio} < {th.strong_beat_chord_ratio_min}")
    if vl_avg > th.voice_leading_max_st:
        failures.append(f"voice_leading_avg_semitones={vl_avg} > {th.voice_leading_max_st} (accords en blocs)")
    if overlap_conflict > th.lead_overlap_max:
        failures.append(f"lead_overlap_conflict={overlap_conflict} > {th.lead_overlap_max} (cafouillis d'attaques)")
    if contrary_ratio < th.contrary_oblique_min:
        failures.append(f"contrary_or_oblique_ratio={contrary_ratio} < {th.contrary_oblique_min}")

    return {
        "musicality_verdict": "PASS" if not failures else "FAIL",
        "metrics": {
            "stepwise_ratio": stepwise_ratio,
            "gap_fill_compliance": gap_fill,
            "melody_ambitus_semitones": ambitus,
            "melody_rest_ratio": rest_ratio,
            "strong_beat_chord_tone_ratio": strong_beat_ratio,
            "voice_leading_avg_semitones": vl_avg,
            "lead_overlap_conflict": overlap_conflict,
            "contrary_or_oblique_ratio": contrary_ratio,
        },
        "detected_roles": {k: [i.name for i in v] for k, v in roles.items()},
        "failures": failures,
    }
```

---

## PHASE 4 : Algorithme de *Voice Leading* Automatique & Chaîne DSP Gridline

### 4.1 Algorithme d'Optimisation Automatique des Renversements (Python / Lua)
Avant d'écrire `chord_progression` dans `song_blueprint.json` ou d'envoyer les notes dans Gridline, l'agent doit toujours calculer les renversements par minimisation de la distance $L_1$ entre deux accords consécutifs :

```python
from itertools import product

def optimize_voice_leading(
    chord_pcs_list: list[list[int]],
    low_midi: int = 50,
    high_midi: int = 72,
) -> list[list[int]]:
    """
    Prend une liste d'accords définis par leurs pitch classes (0..11)
    et retourne des voicings MIDI renversés minimisant le mouvement moyen (<= 2.5 st).
    """
    def candidates_for_chord(pcs: list[int]) -> list[list[int]]:
        pools = []
        for pc in pcs:
            opts = [m for m in range(low_midi, high_midi + 1) if m % 12 == pc]
            pools.append(opts)
        res = []
        for combo in product(*pools):
            s = sorted(combo)
            # Éviter les clusters < 2 demi-tons dans le grave et les écarts > 16 st
            if s[-1] - s[0] <= 16 and len(set(s)) == len(pcs):
                res.append(s)
        return res

    all_cands = [candidates_for_chord(pcs) for pcs in chord_pcs_list]
    # Premier accord centré autour du milieu de la plage
    mid = (low_midi + high_midi) / 2.0
    chosen = [min(all_cands[0], key=lambda c: abs(sum(c) / len(c) - mid))]

    for cands in all_cands[1:]:
        prev = chosen[-1]
        best = min(
            cands,
            key=lambda c: sum(min(abs(n - p) for p in prev) for n in c) / len(c),
        )
        chosen.append(best)
    return chosen
```

### 4.2 Architecture de Routage et Ciment Timbral dans Gridline `0.7.7`
Respecter impérativement les contraintes de `apprentissage.md` (100 % compatible Live GUI + Headless CLI, sobriété CPU, paramètres `gl.reverb` full model) :

1. **Pistes mélodiques individuelles** (`Piano`, `Pad`, `Pluck/Lead`, `Counter`) :
   * Route exclusivement vers **`BUS Music` (Insert 11)** (jamais directement vers `Master 0`).
   * `HPF` (`gl.eq`, `"Low cut"`) obligatoire : `130–160 Hz` sur Piano/Lead, `180–240 Hz` sur Pad/Strings.
   * Accordage vérifié à `A4 = 440.0 Hz ± 5 cents` sur tous les samplers.
   * **Piano principal** : utiliser `gl.multisampler` avec `soundbanks/sfz/headroom-piano/Headroom Piano Clean.sfz` (SFZv1 une directive par ligne, polyphonie `<= 24`).
2. **`BUS Music` (Insert 11 → `Master 0`) — Chaîne de Ciment Timbral** :
   * **Slot 1 — `gl.eq`** : `Low cut 120 Hz` + `High cut (LPF) 12500 Hz` (`12 dB/oct`) pour unifier le haut du spectre + creusement léger `-1.5 dB @ 350 Hz` si `low_mids > 24%`.
   * **Slot 2 — Saturation collective** : `gl.dist` (natif 100 % portable) en mode `Soft`, `Oversampling = 4x`, `Mix = 0.10` (ou `CHOWTapeModel` subtil en session Live GUI uniquement).
   * **Slot 3 — Compression de bus (`gl.comp`)** : `Ratio = 2.0`, `Attack = 0.020 s`, `Release = 0.110 s`, seuil ajusté pour `1.5–2.5 dB` de GR.
   * **Slot 4 — Micro-Ducking rythmique (`gl.comp`)** : `Detector input = Sidechain` alimenté par un send pré-fader du `Kick`, `Attack = 0.005 s`, `Release = 0.080 s` ($\approx 1/8$ de temps), `GR = 1.5–2.5 dB`.
   * **Largeur stéréo** : `mixer/11/stereo` ajusté entre `-0.20` et `-0.35` si `low_mid_200_800hz_correlation < 0.65`.
3. **`FX Room` (Insert 12 → `Master 0`) — Espace Acoustique Unique** :
   * `gl.reverb` avec vérification obligatoire des paramètres **(full)** (`param 38 = 1`) :
     * `Dry (full) = 0` (`param 33 = 0`)
     * `Low cut (full) >= 450 Hz` (`param 23`)
     * `Decay (full)` calibré à `<= 1 × T_noire` (ex. `0.33–0.45 s`)
     * `Early reflections (full) <= 24` (`param 26`)
   * Sends vers `FX Room` : Pistes mélodiques `8–14%`, Snare/Clap `3–6%`, Kick/808/Hi-Hats **`0%`**.
   * Automation de volume sur `FX Room` pour couper à `0` sur le 4e temps des mesures pivots (`M8, M32, M40, M56, M72`).

---

## PHASE 5 : Plan d'Exécution & Refonte des Projets Musicaux

L'agent de développement doit exécuter cette séquence dans l'ordre strict :

1. **Mise à jour Documentation & Règles** :
   * Créer `docs/regles_melodie_et_cohesion_inter_pistes.md`.
   * Corriger `harmonisation_verticale_et_horizontale.md` (Règle V3 + V6 + Checklist).
   * Mettre à jour `.rules`, `GEMINI.md`, `AGENTS.md` pour imposer le Gate `master_sketch.mid`.
2. **Mise à jour Schéma & Outil MCP** :
   * Ajouter `harmonic_core` dans `schemas/song_blueprint.schema.json`.
   * Intégrer `musicality_metrics.py` dans `mcp-servers/audio-analyzer/` et exposer le verdict `musicality_verdict` dans `audit_midi_harmony`.
   * Écrire et exécuter un test unitaire rapide vérifiant qu'une mélodie aléatoire retourne `FAIL` et qu'une partition respectant `M1–M4` + `V1` retourne `PASS`.
3. **Audit des Projets Existants** :
   * Exécuter le nouveau `audit_midi_harmony` sur les fichiers `.mid` des projets actifs (`projects/active*`) pour mesurer objectivement les scores actuels avant refonte.
4. **Refonte Musicale (pour chaque projet à améliorer)** :
   * Remplir `harmonic_core` dans `song_blueprint.json` avec des voicings calculés via `optimize_voice_leading(...)` et une cellule motivique `A - A' - A'' - B`.
   * Générer `master_sketch.mid` (Piano sec uniquement) et itérer jusqu'à obtenir `musicality_verdict == "PASS"` (`0` note hors gamme, `0` clash, les 7 métriques dans les cibles).
   * Éclater `master_sketch.mid` dans Gridline :
     * Voix Basse → `808` (accordée FFT, monophonique `Voices = 1`, sidechain Kick).
     * Voix Médianes → `Chords / Pad` (attaque douce, jamais de transposition parallèle d'accord samplé).
     * Voix Supérieure → `Lead / Piano` (`Headroom Piano Clean.sfz` ou instrument cible).
     * Contre-chant (uniquement aux Refrains / Couplet 2) → écho des 2 dernières notes du Lead dans ses silences, espacé d'une octave.
   * Appliquer la chaîne `BUS Music` Glue + `FX Room` unique + automation de coupure aux pivots.
   * Lancer la boucle d'auto-mixage complète (`gridline render` → `analyze_mix` + `audit_mix_deep`) jusqu'à validation simultanée de tous les critères :
     * `musicality_verdict == "PASS"`
     * `clipped_samples == 0` et `true_peak_dbtp <= -1.0`
     * `integrated_lufs` à `± 0.5 dB` de la cible, avec contraste Intro/Pont vs Refrain `>= 3.0 dB`
     * `correlation_low_end_150hz >= 0.75` et `low_mid_200_800hz_correlation >= 0.65`
     * `0` résonance 1/3 octave.
5. **Clôture & Capitalisation** :
   * Consigner les mesures Avant/Après (`musicality_metrics`, `PQ/CE/CU`, `LUFS`, `dBTP`) dans `apprentissage.md`.