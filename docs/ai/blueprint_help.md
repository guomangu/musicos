# Guide Universel des Blueprints (`blueprint_help.md`)

Ce guide permet aux agents IA de concevoir rapidement un `song_blueprint.json` solide, validé et adapté à **n'importe quel genre ou configuration de projet** (création complète, remix/flip VGM, ou arrangement à partir de samples fournis par l'utilisateur), en capitalisant sur les réalisations du studio.

---

## 1. Méthode Rapide (Workflow en 5 étapes avant de coder)

1. **Inventorier les sources existantes** :
   - Si le projet contient déjà un `.gridline` initial (ex. `Untitled.gridline`), lister ses canaux et extraire : **durée**, **sample rate**, **fréquence fondamentale (`f0`) / note exacte** (pour 808 et one-shots) et **chroma / tonalité** (pour boucles et voix).
   - Respecter la règle d'or : **zéro ajout de synthé/sample externe non demandé** si l'utilisateur a déjà fourni sa sélection, mais **droit total de dérivation** (time-stretch, découpe de couplets/refrains, version staccato/chopped, riser inversé, couche `+12 st`, filtrage) sauvegardée dans `samples/` du projet.
2. **Fixer la grille temporelle (`bpm` & `time_signature`)** :
   - **Rap / Trap moderne** : si une voix ou boucle est à `70–95 BPM`, passer en **half-time double-grille (`140–190 BPM`)** (ex. voix `90 BPM` → grille **`180 BPM`** dans `active5`). Avantage : `Snare/Clap` verrouillé sur le 3e temps, rolls `1/16` et `1/32` lisibles, et macro-forme naturelle de `64–96 mesures` (`2 min – 2 min 30`).
   - **Boucles à autre tempo** : calculer `rate = target_bpm / source_bpm` (ou `target_half_bpm / source_bpm`) avec `librosa.effects.time_stretch` (préserve le pitch).
3. **Choisir le mode & la tonalité (`key`)** :
   - **Trap sombre / Hard** : Mineur naturel (Éolien) ou Phrygien (`bII -> i`, tension demi-ton).
   - **Cloud / Mélodique** : Mineur naturel ou Dorien.
   - **Règle d'accordage 808** : mesurer `f0` après le transitoire (`50–450 ms`), régler `Root note` (`24` pour `C1`, `36` pour `C2`) et `Fine tune` en cents ; privilégier les fondamentales entre **`41 et 85 Hz`** (`E1–E2`, éviter de rester sous `38 Hz`).
4. **Assembler `structure` + `frequency_plan` + `tracks`** selon les modèles ci-dessous.
5. **Valider immédiatement** contre le schéma officiel :
   ```bash
   /home/gamo/Bureau/musicos/.venv/bin/python -c \
     "import json, jsonschema; jsonschema.validate(json.load(open('projects/<actif>/song_blueprint.json')), json.load(open('schemas/song_blueprint.schema.json'))); print('VALID')"
   ```

---

## 2. Capitalisation des Projets Précédents (Référentiel Studio)

| Projet | Style | BPM / Tonalité | Mesures | Enseignements clés réutilisables |
| :--- | :--- | :--- | :--- | :--- |
| **`active`** | Template | `120` / `C minor` | `12` | Squelette minimal `5` pistes (`kick`, `snare`, `hat`, `808`, `pad`), cible `-14.0 LUFS`. |
| **`active2`** (*Obsidian Midnight*) | Dark Trap | `140` / `C minor` | `36` | Format court percutant (`4m Intro -> 8m Drop -> 8m Couplet -> 4m Pont -> 8m Drop Climax -> 4m Outro`), cible `-11.0 LUFS`. |
| **`active3`** (*Souls in Motion*) | Melodic Trap / Rap | `164` / `D# minor` | `40` | Format VGM/Stems + Rap (`Root note = 63` sur stems `gl.sampler` pour 0 note hors gamme), `19` canaux, cible `-10.5 LUFS`. |
| **`active4`** (*Hyper Ambient Zone*) | Cloud / Progressive Trap | `150` / `D minor` | `76` | Format 2 minutes progressif : boucle ambiance ping-pong (`Base` + copie `High +12` filtrée en 2e minute), `3–4` frappes 808/mesure, `-10.5 LUFS`. |
| **`active5`** (*Stoicville Nights*) | Dark Trap / Rap Flip | `180` (half-time `90`) / `E minor` | `96` | Format complet 2 min 08 : acapella découpé sur grille, accords techno time-stretchés + version *chopped* + *riser* inversé, 2 808 distinctes (`Apollo` couplets / `Blxckout` refrains), protocole multi-échelles (`PQ = 8.11`, `-11.45 LUFS`, `-1.18 dBTP`). |

---

## 3. Modèles de Structure Universels (par durée & objectif)

### Modèle A — Format Court / Démo (`32–40 mesures`, ~1 min – 1 min 15)
Idéal pour valider un groove ou une idée rapidement (`active2`, `active3`) :
- `Intro (4m, E=0.35)` → `Drop / Refrain 1 (8m, E=0.90)` → `Couplet (8m, E=0.75)` → `Pont / Break (4m, E=0.45)` → `Drop 2 Climax (8m, E=0.95)` → `Outro (4m, E=0.25)`.

### Modèle B — Format Progressif 2 Minutes (`64–76 mesures`, ~2 min)
Idéal pour Trap / Cloud / Boom-Bap avec évolution sur la 2e minute (`active4`) :
- `Intro (4m)` → `Couplet 1 (8m)` → `Refrain 1 (8m)` → `Pont 1 (4m)` → `Refrain 2 (8m)` → `Transition (4m)` → `Couplet 2 (variation texture/808, 8m)` → `Pont 2 planant (8m)` → `Refrain 3 Climax (+12 st / couche haute, 16m)` → `Outro (8m)`.

### Modèle C — Format Complet Haute Rétention (`96 mesures`, ~2 min 08 – 3 min 15)
Idéal pour un morceau Rap / Trap complet suivant le protocole multi-échelles (`active5`) :
1. **Intro / Encodage (`M1–8`)** : Thème filtré (`LPF ~1 kHz`), pas de basse, pivot M8 (coupure + riser inversé).
2. **Refrain 1 / Drop Initial (`M9–16`)** : Ouverture du spectre (`808` + `Kick` + `Snare/Clap` sur temps 3 + mélodie haute).
3. **Couplet 1 (`M17–32`, `8 + 8`)** : Soustraction des aigus (`M17–24`) pour libérer la voix, réintroduction d'un motif secondaire (`M25–32`) + pivot M32.
4. **Pont / Pré-Refrain (`M33–40`)** : Coupure complète `808 + Kick`, tension harmonique + buildup.
5. **Refrain 2 / Climax (`M41–56`)** : Retour plein + top-loop hats (`> 4 kHz`) + variations 808 (glides / octaves).
6. **Couplet 2 (`M57–72`, `8 + 8`)** : Changement de texture (sample *chopped*, autre 808 ou pattern alternatif).
7. **Refrain 3 (`M73–88`)** : Énergie maximale.
8. **Outro (`M89–96`)** : Soustraction progressive (coupe 808 à M89, coupe batterie à M93, fin filtrée).

---

## 4. Règles Universelles d'Arrangement, de Durées & de Spectre à inscrire dans le Blueprint

- **Règle des 3 Couches (hors batterie)** : ne jamais dépasser **3 rôles mélodiques/harmoniques simultanés** dans `active_roles` :
  1. *Fondation* (`808 / Sub`)
  2. *Harmonie / Texture* (`Pad / Chords / Guitare`)
  3. *Motif / Voix* (`Lead Rap / Vocal / Contre-chant aigu`)
- **Cohésion Inter-Pistes (Verticale & Horizontale)** (voir [harmonisation_verticale_et_horizontale.md](../harmonisation_verticale_et_horizontale.md)) :
  - **Accordage unifié (`440 Hz ± 5 ct`)** sur tous les samples tonaux (808, accords, pianos).
  - **Horloge unique** : découper toute boucle d'un autre groove en one-shots propres (`t = 0` sans pick-up caché) et re-quantifier sur la grille maîtresse ; verrouiller `Kick + 808 + Accord` sur les temps forts.
  - **Relais *Call-and-Response*** & **grille d'accords partagée** : les stabs d'accords suivent les fondamentales de la 808 en MIDI, et les contre-mélodies répondent dans les vides rythmiques.
- **Dimensionnement temporel & Enveloppes ADSR (`T_noire = 60000 / BPM ms`)** (voir [protocole_universel...md](../protocole_universel_d_architecture_musicale_coh_rence_multi_chelles_mod_le_pr_dictif_orientation_trap.md)) :
  - Calculer la grille en ms (ex. à `180 BPM` : `1 mesure = 1333.3 ms`, `1 noire = 333.3 ms`, `1 croche = 166.7 ms`, `1 double-croche = 83.3 ms`, `1 triple-croche = 41.7 ms`).
  - **Ticks / Hi-Hats** : `20–90 ms` (`Attack 0`, `Decay 30–70 ms`, `Sustain 0%`), extinction avant la subdivision suivante.
  - **Kick / Snare** : Kick court (`80–160 ms`), Snare/Clap (`120–250 ms`, réverbe `< T_noire`).
  - **808 Monophonique (*Cut Itself*) OBLIGATOIRE** : `Voices = 1`, `Release = 30–60 ms` ; longue (`1–4` temps) si mélodie staccato, ou courte/bounce (`1/8–1/4` temps) si pads/voix denses (**Règle d'or Staccato / Legato**).
  - **Queues Reverb / Delay** : `RT60` et delay synchronisés à un multiple/sous-multiple de `T_noire`, et coupés proprement sur les mutes de transition (M4 / M8).
- **Renouvellement 4 / 8 mesures** : noter dans `structure[].notes` les micro-variations toutes les 4 mesures et les **pivots de transition** (coupure beat 4, fill snare, riser inversé, vocal FX) toutes les 8 mesures.
- **Checklist avant rendu** : Soustraction (mélodie+808 sans drums) · Densité (`<= 3` couches) · Durées et Silences (`808` mono + ticks secs) · Marqueurs des 8 mesures · Espace Vocal (`800 Hz – 2.5 kHz` dégagé en couplet).
- **Allocation fréquentielle standard (`frequency_plan`)** :
  - `sub_20_60` : `808 / Sub` uniquement (**mono strict** `gl.stereo Width = 0`, `HPF 28–30 Hz`).
  - `bass_60_250` : `Kick (55–120 Hz)` en sidechain sur `BUS 808` (`15–40 ms`) ; `HPF >= 130–160 Hz` sur tout le reste.
  - `low_mids_250_500` : corps snare/piano/pad (1 seul instrument dominant, coupe douce `~300 Hz` si besoin).
  - `mids_500_4000` : zone vocale prioritaire (`1.5–3 kHz`) ; creuser `-2.5 à -3 dB` à `~1.8 kHz` sur les pads/pianos si voix lead active.
  - `highs_4000_20000` : hi-hats, top-loops (`HPF 4 kHz`), brillance piano, chants, send partagé `FX Room` (`Decay <= 0.85 s`, `HPF 300 Hz`).
- **Cibles `target_lufs`** : `-14.0` (Streaming / Lo-Fi / Boom-Bap), `-10.5` à `-11.0` (Trap / Rap / Drill), `-9.0` (Club).

---

## 5. Squelette JSON Universel (Copier & Adapter)

```json
{
  "title": "Titre du Morceau",
  "genre": "Genre / Sous-genre",
  "mood": "Ambiance en 3-5 mots",
  "bpm": 150,
  "time_signature": { "numerator": 4, "denominator": 4 },
  "key": "E minor",
  "swing": 0.06,
  "target_lufs": -10.5,
  "structure": [
    {
      "section": "Intro",
      "start_bar": 1,
      "bars": 8,
      "energy": 0.35,
      "active_roles": ["harmony_filtered", "motif_pizz"],
      "notes": "Exposition filtrée sans basse ; pivot mesure 8 (riser / mute)."
    }
  ],
  "frequency_plan": {
    "sub_20_60": "Mono strict réservé à la 808/Sub (HPF 28 Hz).",
    "bass_60_250": "Kick sidechain -> BUS 808 ; HPF >= 140 Hz sur mélodies et voix.",
    "low_mids_250_500": "1 seul instrument harmonique dominant, coupe douce ~300 Hz.",
    "mids_500_4000": "Poche vocale 1.5-3 kHz dégagée sur accords/leads.",
    "highs_4000_20000": "Hi-hats > 4 kHz, brillance, 1 send FX Room (HPF 300 Hz)."
  },
  "tracks": [
    {
      "name": "Kick",
      "role": "kick",
      "source_type": "gridline_native",
      "plugin": "gl.kick",
      "mixer_track": 1,
      "freq_lane": "55-120 Hz + click 2-4 kHz",
      "pan": 0.0,
      "notes": "Sidechain vers BUS 808."
    }
  ],
  "vocals": {
    "enabled": false,
    "language": "en",
    "singer": "",
    "lyrics_sections": []
  }
}
```
