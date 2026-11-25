# Facette 2 — Composition Universelle, Émotion `(V, T, E, D)`, Harmonie, Mélodie & Groove

> **Cheatsheet IA (L'essentiel v2.0)**
> - **Règle n°1** : Ne **jamais** calculer notes MIDI, gammes ou accords de tête — importer `SCALES`, `degree_to_midi` (`song_builder.py`) et `optimize_voice_leading` (`musicality_metrics.py`). Carte fichiers → [`docs/00_SOMMAIRE.md`](00_SOMMAIRE.md).
> - **Émotion `(V, T, E, D)`** : Choisir **Mode** (§2), **Progression éprouvée (§4)**, **BPM & Matrice Rythmique (§6)**, **Palette (§7)**.
> - **Mélodie (§5)** : Motif de **3 à 5 notes** varié sur 4 mesures (`A - A' - A'' [Apex <= 74] - B [Silence]` si `V >= 0`, ou courbe descendante/hypnotique si `V < 0`) ; `>= 65 %` pas conjoints (`<= 4 st`), `20–50 %` silence (`[0, 0.92]` en `free_ambient`), temps forts sur accord (`>= 80 %` consonant / `>= 65 %` sombre), **zéro** note sur chaque croche, **zéro** arpège 1-3-5-8 plat ou vélocité constante.
> - **Humanisation automatique** : Appeler **`scripts/groove_lib.py`** (`apply_micro_strum`, `apply_haas_pocket`, `apply_anti_machine_gun`, `enforce_frequency_inertia`) au lieu de coder les décalages à la main.
> - **Registres stricts** : Basse/808 **`31–48`** (`>= 49 Hz`, en `C..F#` jouer `36–42`), Accords `48–67`, Lead `55–72` (Apex `<= 74` ; `75–81` réservé aux timbres doux si `V > +0.3`).

---

## Chapitre 1 — Table MIDI Universelle & Helper Déterministe

> **Interdiction** de convertir notes ↔ numéros MIDI de tête : utiliser `song_builder.SCALES` et `song_builder.degree_to_midi(tonic_midi, mode, degree_0idx, oct_shift=0)`.

| Note | C | C#/Db | D | D#/Eb | E | F | F#/Gb | G | G#/Ab | A | A#/Bb | B |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pc`** | `0` | `1` | `2` | `3` | `4` | `5` | `6` | `7` | `8` | `9` | `10` | `11` |
| **Oct 1** | `24`⚠️ | `25`⚠️ | `26`⚠️ | `27`⚠️ | `28`⚠️ | `29`⚠️ | `30`⚠️ | **`31`** | **`32`** | **`33`** | **`34`** | **`35`** |
| **Oct 2** *(Basse)* | `36` | `37` | `38` | `39` | `40` | `41` | `42` | `43` | `44` | `45` | `46` | `47` |
| **Oct 3** *(Chords)* | `48` | `49` | `50` | `51` | `52` | `53` | `54` | `55` | `56` | `57` | `58` | `59` |
| **Oct 4** *(Lead)* | `60` | `61` | `62` | `63` | `64` | `65` | `66` | `67` | `68` | `69` (`440Hz`) | `70` | `71` |
| **Oct 5** *(Apex)* | `72` | `73` | `74` | `75` | `76` | `77` | `78` | `79` | `80` | `81` | `82` | `83` |

* **Bornes obligatoires** : Basse/808 **`MIDI 31–48`** (en `C..F#`, privilégier Octave 2 `36–42` contre l'infra-sub `< 42 Hz`) · Accords **`48–67`** · Lead **`55–72`** (`<= 74` par défaut ; `75–81` uniquement timbres doux + `V > +0.3`).
* **Registres adaptatifs sombres (`V < -0.2`, Phase 8 `Q5.2`)** : resserrer `Chords/Pad` dans **`48–60`** et `Lead` dans **`60–72`** (au moins **`+5 st`** d'écart moyen entre centroïdes MIDI, vérifié par `chords_lead_register_gap_st` dans `musicality_metrics.py`).
* Gammes couvertes par `song_builder.SCALES` : `ionian`/`major`, `dorian`, `phrygian`, `lydian`, `mixolydian`, `aeolian`/`minor`, `harmonic_minor`, `melodic_minor`, `phrygian_dominant`, `locrian`, `pentatonic_minor`, `pentatonic_major`, `blues`.

---

## Chapitre 2 — Modèle Émotionnel Holistique `(V, T, E, D)` & Modes

Toute demande utilisateur se traduit en 4 axes mesurables :
1. **Valence ($V \in [-1, +1]$)** : `-1.0` tragique/sombre ↔ `0.0` neutre/mystérieux ↔ `+1.0` joyeux/lumineux.
2. **Tension ($T \in [0, 1]$)** : `0.0` apaisé/consonant ↔ `1.0` nerveux/urgent/dramatique.
3. **Énergie ($E \in [0, 1]$)** : `0.0` intimiste/calme ↔ `1.0` puissant/frontal/monumental.
4. **Profondeur ($D \in [0, 1]$)** : `0.0` sec/proche/percutant ↔ `1.0` vaste/éthéré/cinématique.

### 2.1 Modes & Gammes (du plus sombre au plus lumineux)

> **Règle de Verrouillage par Valence ($V$)** :
> - Si **`V <= -0.3`** (triste, sombre, mélancolique, tendu) : **interdiction** d'utiliser Ionien, Lydien, Mixolydien ou Pentatonique Majeure. Choisir exclusivement parmi **Éolien**, **Phrygien**, **Mineur Harmonique**, **Dorien** (uniquement si cool/chic) ou **Pentatonique Mineure**.
> - Si **`V <= -0.7`** (sombre, tragique, menaçant) : privilégier **Phrygien** (`i - bII`) ou **Mineur Harmonique** (`i - bVI - V7`) avec ancrage fort sur la tonique `i`.

| Mode | Intervalles (demi-tons) | $V$ | $T$ | Couleur Émotionnelle & Genres Typiques |
| :--- | :--- | :---: | :---: | :--- |
| **Éolien (Mineur Naturel)** | `[0, 2, 3, 5, 7, 8, 10]` | `-0.7` | `0.3` | Triste, nostalgique, dramatique (Pop triste, Trap/Cloud, Rock, Ballade) |
| **Phrygien** | `[0, 1, 3, 5, 7, 8, 10]` | `-0.9` | `0.8` | Menaçant, sombre, tendu (`i <-> bII`) (Dark Trap, Drill, Metal, Techno) |
| **Mineur Harmonique** | `[0, 2, 3, 5, 7, 8, 11]` | `-0.8` | `0.7` | Tragique, théâtral, classique/épique (Orchestral, Metal, Trap dramatique) |
| **Phrygien Dominant** | `[0, 1, 4, 5, 7, 8, 10]` | `-0.6` | `0.85` | Mystique, oriental/flamenco, intense (Cinéma, Trap, Metal, Electro) |
| **Locrien** | `[0, 1, 3, 5, 6, 8, 10]` | `-1.0` | `1.0` | Horreur, chaos, instabilité (Sound design horreur uniquement) |
| **Pentatonique Mineure / Blues** | `[0, 3, 5, 7, 10]` (+`6`) | `-0.4` | `0.4` | Expressif, accrocheur, solo garanti sans clash (Rock, Blues, Rap, R&B) |
| **Dorien** | `[0, 2, 3, 5, 7, 9, 10]` | `-0.2` | `0.3` | Mineur élégant, cool, résilient (Soul, R&B, Funk, Lo-Fi, House, Cinéma) |
| **Mixolydien** | `[0, 2, 4, 5, 7, 9, 10]` | `+0.5` | `0.3` | Assuré, groove, rock/funk positif (Rock, Funk, Afrobeat, House) |
| **Pentatonique Majeure** | `[0, 2, 4, 7, 9]` | `+0.7` | `0.1` | Mélodique, instantanément chantable, zéro fausse note (Pop, Folk, Lo-Fi) |
| **Ionien (Majeur)** | `[0, 2, 4, 5, 7, 9, 11]` | `+0.8` | `0.1` | Joyeux, franc, solaire (Pop, Folk, Indie, House lumineuse) |
| **Lydien** | `[0, 2, 4, 6, 7, 9, 11]` | `+1.0` | `0.2` | Féérique, spatial, émerveillement (Cinéma, Synthwave, Ambient, VGM) |

---

## Chapitre 3 — Accords, Voicings & Dissociation Spectrale

### 3.1 Formules d'Accords (demi-tons depuis la fondamentale de l'accord)

| Code | Intervalles | $V$ | $T$ | Couleur & Usage |
| :--- | :--- | :---: | :---: | :--- |
| `Maj` / `m` | `[0, 4, 7]` / `[0, 3, 7]` | `+0.9` / `-0.7` | `0.1` / `0.2` | Clarté / gravité franche (Pop, Rock, Classique, Folk) |
| `Maj7` / `m7` | `[0, 4, 7, 11]` / `[0, 3, 7, 10]` | `+0.6` / `-0.5` | `0.3` | Nostalgie douce / mélancolie soyeuse (Lo-Fi, Jazz, R&B, Cloud, House) |
| `Maj9` / `m9` | `[0, 4, 7, 11, 14]` / `[0, 3, 7, 10, 14]` | `+0.7` / `-0.4` | `0.35` | Planant, luxueux, cinématographique (Neo-Soul, R&B, Lo-Fi, Chill) |
| `sus2` / `sus4` | `[0, 2, 7]` / `[0, 5, 7]` | `0.0` | `0.4` / `0.6` | Ouvert, aérien / héroïque suspendu (Ambient, Rock, Cinématique) |
| `5` (Power) | `[0, 7]` | `0.0` | `0.3` | Puissance neutre, zéro boue harmonique (Rock, Metal, Dark Trap) |
| `m(add9)` / `m(b6)` | `[0, 3, 7, 14]` / `[0, 3, 7, 8]` | `-0.6` / `-0.8` | `0.6` | Tristesse nerveuse / drame déchirant (Ballade, Cinéma, Trap sombre) |
| `7` / `dim7` | `[0, 4, 7, 10]` / `[0, 3, 6, 9]` | `+0.2` / `-0.9` | `0.7` / `0.95` | Tension de dominante (appel vers `I`/`i`) / angoisse (Jazz, Blues, Classique) |

* **Voicing Serré (`< 1` oct)** = intime/dense · **Voicing Ouvert (`1.5–2.5` oct)** = épique/aéré · **Toujours passer les pitch-classes dans `optimize_voice_leading()`** pour garder un mouvement moyen `<= 2.5 st` et éviter les blocs parallèles amateurs.
* **Dissociation Spectrale (Émotions Paradoxales)** : pour mêler deux émotions (ex. *triste + nerveux*, *doux + dansant*), affecter une polarité aux médiums/aigus (`> 300 Hz` : accords `m9` éthérés + soupirs descendants `b6 -> 5`) et la polarité opposée aux graves/rythme (`< 150 Hz` : basse sèche saturée + rythme syncopé).

---

## Chapitre 4 — Catalogue Universel de 14 Progressions Harmoniques Éprouvées

> Une IA textuelle ne doit **jamais inventer des accords au hasard**. Choisir une progression éprouvée dans le bloc correspondant à la valence `(V)` cible, la transposer dans la tonalité cible, puis calculer ses renversements via `optimize_voice_leading()`.

### A. Sombre & Triste (`V < -0.3`)

| # | Émotion / Ambiance | Genres | Mode | Progression (4 ou 8 mesures) | Mouvement Basse (demi-tons depuis tonique) |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **P4** | Triste, profond, introspectif | Ballade, Cloud Trap, Piano, Lo-Fi | Éolien | `i7 – VI(maj7) – iv7 – v7` (ou `V7`) | `[0, 8, 5, 7]` |
| **P4b** | Mélancolique, lourd, hypnotique | Trap sombre, Cloud, Piano, Cinéma | Éolien | `i – bVI – v – i` ou `i – bVI – iv – i` | `[0, 8, 7, 0]` ou `[0, 8, 5, 0]` |
| **P5** | Sombre, nerveux, menaçant | Dark Trap, Drill, Metal, Techno | Phrygien | `i – bII – i – bVII` (ou `i – bII`) | `[0, 1, 0, 10]` |
| **P5b** | Sombre, minimaliste, oppressant | Dark Trap, Drill, Ambient sombre | Phrygien / Pédale | `i – bII(maj7) – i – bVI` (pédale de tonique `i` possible) | `[0, 0, 0, 8]` ou `[0, 1, 0, 8]` |
| **P6** | Tragique, dramatique, théâtral | Orchestral, Classique, Rap épique | Min. Harm. | `i – iv – VI – V7` | `[0, 5, 8, 7]` |
| **P3** | Épique, mélancolique-héroïque *(relatif majeur)* | Pop, Rock, Cinéma, Rap, EDM | Éolien | `i – VI – III – VII` *(⚠️ Attention : sonne héroïque/pop — attirance vers le relatif majeur `III`. Ne pas l'utiliser pour une ambiance vraiment sombre, glaciale ou dépressive.)* | `[0, 8, 3, 10]` |

### B. Neutre & Groove (`-0.3 <= V <= +0.3`)

| # | Émotion / Ambiance | Genres | Mode | Progression (4 ou 8 mesures) | Mouvement Basse (demi-tons depuis tonique) |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **P7** | Cool, groove, chic, résilient | House, Funk, Soul, R&B, Lo-Fi | Dorien | `i7 – IV7 – i7 – bVII(maj7)` | `[0, 5, 0, 10]` |
| **P11** | Blues / Groove intemporel (12 mes.) | Blues, Rock, Jazz, Soul | Blues/Mix. | `I7(×4) – IV7(×2) – I7(×2) – V7 – IV7 – I7 – V7` | `[0..0, 5, 5, 0, 0, 7, 5, 0, 7]` |
| **P12** | Escalade émotionnelle (Pont / Pré-refrain) | Tous genres | Éolien/Ion. | `iv7 – III – VI – V(sus4 -> 7)` | `[5, 3, 8, 7]` |

### C. Lumineux & Joyeux (`V > +0.3`)

| # | Émotion / Ambiance | Genres | Mode | Progression (4 ou 8 mesures) | Mouvement Basse (demi-tons depuis tonique) |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **P8** | Chaleureux, jazz, cosy | Lo-Fi, Jazz, Neo-Soul, Bossa | Ionien | `ii7 – V7 – Imaj7 – vi7` | `[2, 7, 0, 9]` |
| **P9** | Rêveur, spatial, contemplatif | Ambient, Cinéma SF, Chill, VGM | Lydien | `Imaj7 – II(add9) – Imaj7 – vii7` | `[0, 2, 0, 11]` |
| **P10** | Énergique, solaire, rock/funk | Rock, Funk, Indie, Afrobeat | Mixolydien | `I – bVII – IV – I` | `[0, 10, 5, 0]` |
| **P2** | Nostalgique, doux-amer, lumineux | Pop, Synthwave, Anime/J-Pop | Ionien | `IVmaj7 – V7 – iii7 – vi7` | `[5, 7, 4, 9]` |
| **P1** | Joyeux, fédérateur, hymne | Pop, Indie, EDM, Rock | Ionien | `I – V – vi – IV` | `[0, 7, 9, 5]` |

* **Astuce Pont / Contraste harmonique** : Si le Couplet/Refrain utilise `P4` / `P4b` ou `P3`, utiliser `P12` (`iv7–III–VI–V7`) au Pont pour créer un soulèvement émotionnel (ou basculer vers `iv -> V7` en mineur harmonique pour appeler le retour sur `i`).

---

## Chapitre 5 — Recette Mélodique, Relais Basse & Expressivité (`scripts/groove_lib.py`)

### 5.1 Générer une Mélodie Agréable sur 4 Mesures (`V >= 0` vs `V < 0`)
1. **Cellule `A` (Mesure 1)** : `3–5` notes sur `4` beats avec `>= 0.75–1.5` beat de silence, ancrée sur l'accord 1, `>= 65 %` pas conjoints (`<= 4 st`, sinon retour inverse de `1–3 st` *Gap-Fill*). Ex. en `Cm7` (`C=0, Eb=2, G=4, Ab=5`) : beats `[0.0, 1.0, 1.5, 2.5]` → `[G4, Ab4, G4, Eb4]`.
2. **Développement selon Valence (`V`)** :
   * **`V >= 0` (`A - A' - A'' [Apex <= 74] - B [Silence]`)** : `A'` adapte le même rythme à l'accord 2 + respiration ; `A''` atteint l'apex unique (`+2..+5 st`, `<= 74`) puis redescend ; `B` résout sur tierce/tonique puis **silence sur beats `2.5..4.0`**.
   * **`V < 0` (Sombre / Hypnotique)** : `A` descendant (`b6 -> 5` ou `b3 -> 2 -> 1`, `MIDI 58–67`) ; `A'` répétition obsédante `-1..-2` degrés ; `A''` appoggiature `b6->5` / `b2->1` (sans apex joyeux) ; `B` retombée grave + `1.5–2.0` beats de silence.
3. **Sculpting & Contre-chant** : varier les vélocités (temps `0.0`: `92–104`, temps `2.0`: `84–94`, contretemps: `68–82`, apex/tension `+6..10`, fin `-10..-15`), légato `0.90–0.98×` / pluck `0.45–0.70×`, mouvement Basse/Lead contraire ou oblique (`>= 55 %`), contre-chant uniquement dans les silences du Lead (`overlap <= 15 %`).

### 5.2 Loi d'Inertie, Haas & Expressivité Automatisée (`groove_lib.py`)
* **Inertie fréquentielle** : `< 100 Hz` toujours long (`>= 1/2` Sub `20–60 Hz`, `>= 1/8` Basse `60–250 Hz` sauf `Release < 50 ms`) ; subdivisions rapides (`1/16`, `1/32`) réservées aux aigus `> 800 Hz`. Imposer automatiquement via **`enforce_frequency_inertia(notes)`**.
* **Humanisation déterministe (`scripts/groove_lib.py`)** :
  * **`apply_micro_strum(chords, step_ms=15.0, bpm=...)`** : décale chaque voix ascendante (`+10–25 ms`) et atténue les voix internes.
  * **`apply_haas_pocket(tracks, bpm=..., profile=...)`** : Kick = `0 ms` (ancrage), Snare/Clap `+0..+10 ms` (urgent `-5 ms`), Hats `+5..+18 ms`, Basse `+10..+20 ms` (préserve l'attaque Kick et le headroom).
  * **`apply_anti_machine_gun(notes)`** : humanise vélo/longueur sur motifs rapides (`>= 1/16`) sans jamais répéter deux frappes identiques.

### 5.3 Bibliothèque Universelle de 11 Matrices Rythmiques Exactes (Beats 0-Indexed)

> Chaque mesure 4/4 vaut `4.0` beats (`0.00`..`3.75` par `0.25`). Choisir la matrice du genre cible et appliquer `groove_lib` + variations en mes. 4 et 8.

| Style / Genre | BPM | Feel | Kick (beats) | Snare / Clap (beats) | Hi-Hat / Ride / Perc (beats & vélocités) | Swing / Micro-timing |
| :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| **1. Trap / Cloud** | `124–152` | Half-time | `0.0, 1.75, 2.75` (var. `0.0, 2.5`) | `2.0` (`vel 108`) | Croches `0.0, 0.5..3.5` (`75–92`) + rolls `1/16`/`1/32` sur `1.5–1.75` ou `3.5–3.875` (pitch `-1..-3 st`) | Snare `+0.01`, Hats `-0.005` |
| **2. Drill** | `138–146` | Half-time | `0.0, 1.5, 3.25` | **`2.0` et `3.5`** (signature Drill) | Syncopé `0.0, 0.75, 1.5, 2.0, 2.75, 3.5` (`72–95`) + glides 808 en mes. 4/8 | Sec, tendu, `swing = 0` |
| **3. Boom Bap / Lo-Fi** | `76–94` | Groove | `0.0, 1.75, 2.5` (var. `0.0, 2.25, 2.75`) | `1.0, 3.0` (`vel 98`, ghost `1.75` `vel 45`) | Croches `0.0, 0.5..3.5` avec accents alternés `88 / 58` | **Swing `0.15–0.30`**, Snare retardé `+10–18 ms` |
| **4. House / Techno / EDM** | `118–134` | 4-on-floor | **`0.0, 1.0, 2.0, 3.0`** | Clap `1.0, 3.0` (`vel 102`) | Open Hat contretemps **`0.5, 1.5, 2.5, 3.5`** (`88`) + Closed `0.25, 1.25...` (`55`) | House swing `0.10–0.20` ; Techno `0.0` |
| **5. Pop / Synthwave** | `102–126` | Binaire | `0.0, 1.5, 2.0` (ou 4-on-floor au Refrain) | `1.0, 3.0` (`vel 104`) | Croches régulières `0.0..3.5` (`82 / 64`) + Shaker `1/16` au Refrain (`48–62`) | Propre, `swing 0–0.05` |
| **6. Rock / Indie / Metal** | `105–165` | Binaire | `0.0, 1.5, 2.0, 2.5` (Metal : double `1/16`) | `1.0, 3.0` (`vel 112`) + crash sur `0.0` de mes. 1 | Croches Hi-Hat fermé (Couplet) → Ride ou Crash-Ride (Refrain) (`90 / 70`) | Humain `±6 ms`, fill toms/snare sur `3.0–3.75` mes. 4/8 |
| **7. Funk / R&B / Neo-Soul** | `88–114` | Pocket | `0.0, 0.75, 2.25, 2.5` | `1.0, 3.0` (`vel 100`) + ghosts `1.75, 2.75` (`42`) | Doubles croches `0.0..3.75` avec forte dynamique (`85, 45, 68, 42`) | **Pocket / Swing `0.12–0.25`** |
| **8. Reggaeton / Afrobeat** | `94–112` | Dembow | `0.0, 1.0, 2.0, 3.0` (Afro : `0.0, 1.5, 2.5`) | **`0.75, 1.5, 2.75, 3.5`** (Dembow) | Shaker `1/16` continu (`55–75`) + rimshot syncopé | Rebonds doux, basse calée sur Kick |
| **9. Drum & Bass / Jungle** | `168–176` | Breakbeat | **`0.0, 2.5`** | **`1.0, 3.0`** (ghosts `1.75, 2.25` `vel 52`) | `1/8` + `1/16` roulants (`82, 55, 74, 58`) | Tendu, sub basse longue |
| **10. Valse `3/4` / Ballade `6/8`** | `72–115` | Ternaire | `3/4` : `0.0` · `6/8` : `0.0, 2.5` | `3/4` : rim/brush `1.0, 2.0` (`58`) · `6/8` : `3.0` (`85`) | `3/4` : `0.0, 1.0, 2.0` (fort-doux-doux) · `6/8` : `0.0..5.0` (fort sur `0` et `3`) | Rubato léger, dynamique douce |
| **11. Ambient / Classique** | `55–90` | Libre | Aucun (ou timbale/sub doux sur `0.0` mes. 1) | Aucun (ou percussion cinématique sur pivots) | Aucun — rythme porté par les attaques lentes (`> 0.3 s`) et arpèges doux | Respiration libre, `FX Room` ample |

---

## Chapitre 6 — Palettes d'Instrumentation Cohérentes par Genre

Registres MIDI disjoints et instruments `gl.*` natifs compatibles GUI + headless :

| Famille de Genre | Profil Mix (`--profile`) | Basse / 808 (`MIDI 31–48`, `>= 49 Hz`) | Harmonie (`MIDI 48–67`) | Lead / Motif (`MIDI 55–72`, Apex `<= 74`) | Percussions / Texture |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Trap / Cloud / Drill** | `trap_rap` | `gl.sampler` 808 mono (`Voices=1`) | `gl.sampler` piano/pad WAV ou `gl.warmsynth` (assombrir par shelf doux, jamais de LPF brique < 6,5 kHz) | `gl.sampler` bell/pluck/flûte `mono_oneshot` ou `gl.pluck` (shelf doux, jamais de LPF brique < 9 kHz) | Samples Trap (Kick, Clap/Snare, Hats) |
| **Pop / Synthwave** | `pop_rnb` | `gl.analog` / `gl.fingerbass` (octave bass) | `gl.dcosynth` / `gl.warmsynth` / `gl.keys` (shelf doux, jamais de LPF brique < 6,5 kHz) | `gl.twinva` / `gl.keys` / `gl.plucker` / Voix (shelf doux, jamais de LPF brique < 9 kHz) | Kick/Snare punchy + Shaker |
| **House / Techno / EDM** | `edm_club` | `gl.acid` / `gl.analog` / `gl.stepbass` | `gl.sawswarm` / `gl.wavetable` / `gl.dcosynth` (shelf doux, jamais de LPF brique < 6,5 kHz) | `gl.wavetable` / `gl.pluck` / `gl.synth` (shelf doux, jamais de LPF brique < 9 kHz) | `gl.kick` / samples 909 + Open Hat |
| **Lo-Fi / Boom Bap** | `lofi_chill` | `gl.fingerbass` / sub rond doux | `gl.keys` (EP/Rhodes) / `gl.sampler` piano doux | `gl.plucker` (guitare douce) / flûte / piano (`<= 72`) | Kick/Snare feutrés (LPF `8–10 kHz`) |
| **Rock / Indie / Metal** | `rock_metal` | `gl.fingerbass` | `gl.plucker` / `gl.twinstring` (+ `gl.dist` suivi de LPF) | `gl.twinstring` / `gl.analog` / Voix | `gl.drumkit` ou samples acoustiques |
| **Jazz / R&B / Soul** | `jazz_acoustic` | `gl.fingerbass` (walking/pocket) | `gl.keys` (Rhodes `m7`/`m9`/`Maj9`) | `gl.keys` / `gl.sampler` sax/flûte/guitare | Kit doux / rimshot / ride |
| **Classique / Cinéma / Ambient** | `classical_ambient` | `gl.string` (contrebasse/cello) / sub doux | `gl.string` / `gl.twinstring` / `gl.warmsynth` (pad lent) | `gl.sampler` piano WAV / `gl.string` / `gl.plucker` | Timbales/percs ciné légères ou aucune |

*(Notes registres : pour les tonalités `C`, `C#`, `D`, `D#`, `E`, `F`, `F#`, écrire la fondamentale 808 en Octave 2 `MIDI 36–42` plutôt qu'en Octave 1 `< 42 Hz`. Côté aigus, `MIDI 75–81` est réservée exclusivement aux timbres très doux/sinusoïdaux — flûte douce, cordes légato avec filtre de synthé musical (pente douce) `<= 5 kHz`, jamais de LPF brique 24 dB/oct — sur morceaux lumineux `valence > +0.3`, et strictement interdite sur synthés riches, cloches FM ou morceaux sombres.)*
