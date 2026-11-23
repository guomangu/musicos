# Facette 2 — Composition Universelle, Émotion `(V, T, E, D)`, Harmonie, Mélodie & Rythmes Multi-Genres

> **Cheatsheet IA (L'essentiel en 14 lignes)**
> - **Règle n°1 anti-erreur LLM** : Ne **JAMAIS** calculer une note MIDI, une gamme ou un accord de tête. Utiliser la **formule Python (§1)** à partir du pitch-class (`C=0..B=11`) et des intervalles ci-dessous.
> - **Émotion → Paramètres** : Extraire `(V, T, E, D)` du prompt → choisir **Mode** (selon `V, T`), **Progression éprouvée (§4)**, **BPM & Matrice Rythmique (§6)**, **Palette d'instruments (§7)**.
> - **Mélodie agréable garantie (§5)** : 1 motif court de **3 à 5 notes** répété et varié sur 4 mesures (`A - A' - A'' [Apex <= 74] - B [Silence]` si `V >= 0`, ou courbe descendante/hypnotique si `V < 0`), `>= 65 %` pas conjoints (`<= 4 st`), `20–50 %` de silence réel, notes de l'accord sur temps forts (`>= 80 %` consonant, `>= 65 %` sombre/tendu), vélocités en courbe (jamais plates), **micro-strum 10–25 ms** sur accords piano/guitare.
> - **5 Anti-Patterns LLM interdits** : (1) note sur chaque croche sans respiration, (2) arpège mécanique 1-3-5-8 en boucle, (3) vélocité uniforme `100` partout, (4) sauts erratiques non compensés, (5) même boucle de 4 mesures répétée sans variation jusqu'à la fin.
> - **Cohérence universelle** : 1 seule idée mélodique forte + max 3 couches hors batterie + registre séparé par instrument (Basse/808 **`31–48`** `G1–C3` `>= 49 Hz`, en `C..F#` jouer en Octave 2 `36–42` ; Accords `48–67` ; Lead `55–72`, Apex absolu `<= 74` ; `75–81` réservé aux timbres doux si `V > +0.3`).

---

## Chapitre 1 — Table MIDI Universelle & Calcul Déterministe Python

> **Interdiction absolue** de convertir notes ↔ numéros MIDI de tête (les LLM se trompent sur les dièses/bémols et les octaves). Toujours utiliser cette table et ce helper Python dans `build_song.py`.

| Note | C | C#/Db | D | D#/Eb | E | F | F#/Gb | G | G#/Ab | A | A#/Bb | B |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pitch Class (`pc`)** | `0` | `1` | `2` | `3` | `4` | `5` | `6` | `7` | `8` | `9` | `10` | `11` |
| **Octave 1 (`31–35` Sub/808 ; ⚠️ `24–30` infra-sub `< 47 Hz`)** | `24`⚠️ | `25`⚠️ | `26`⚠️ | `27`⚠️ | `28`⚠️ | `29`⚠️ | `30`⚠️ | **`31`** | **`32`** | **`33`** | **`34`** | **`35`** |
| **Octave 2 (Basse / 808 en `C..F#`)** | `36` | `37` | `38` | `39` | `40` | `41` | `42` | `43` | `44` | `45` | `46` | `47` |
| **Octave 3 (Accords `48–67`)** | `48` | `49` | `50` | `51` | `52` | `53` | `54` | `55` | `56` | `57` | `58` | `59` |
| **Octave 4 (Lead `55–72`, C4=60)** | `60` | `61` | `62` | `63` | `64` | `65` | `66` | `67` | `68` | `69` (`440Hz`) | `70` | `71` |
| **Octave 5 (Apex `<= 74` ; `75–81` doux `V>+0.3`)** | `72` | `73` | `74` | `75` | `76` | `77` | `78` | `79` | `80` | `81` | `82` | `83` |

> **Plages de Registres MIDI par Défaut (Anti-Stridence & Anti-Infra-Sub)** :
> - **Basse / 808** : **`MIDI 31–48` (`G1–C3`, `>= 49 Hz`)**. *Règle anti-infra-sub* : pour les tonalités `C`, `C#`, `D`, `D#`, `E`, `F`, `F#`, écrire la fondamentale 808 en **Octave 2 (`MIDI 36–42`)** plutôt qu'en Octave 1 (`< 42 Hz`, zone infra-sub qui sature `gl.limiter` tout en bridant le LUFS).
> - **Accords (`Chords`)** : `MIDI 48–67` (`C3–G4`).
> - **Lead / Motif** : **`MIDI 55–72` (`G3–C5`)** par défaut, **Apex absolu `<= MIDI 74` (`D5`)**.
> - *Exception* : La zone `MIDI 75–81` est réservée exclusivement aux timbres très doux/sinusoïdaux (flûte douce, cordes légato avec LPF `<= 5 kHz`) sur des morceaux lumineux (`valence > +0.3`). **Interdite** sur synthés riches, cloches FM ou morceaux sombres.

```python
SCALES = {
    "ionian": [0, 2, 4, 5, 7, 9, 11], "major": [0, 2, 4, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10], "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11], "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10], "minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11], "phrygian_dominant": [0, 1, 4, 5, 7, 8, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "pentatonic_minor": [0, 3, 5, 7, 10], "pentatonic_major": [0, 2, 4, 7, 9],
    "blues": [0, 3, 5, 6, 7, 10],
}
def degree_to_midi(tonic_midi: int, mode: str, degree_0idx: int, oct_shift: int = 0) -> int:
    sc = SCALES[mode]
    oct_add, idx = divmod(degree_0idx, len(sc))
    return tonic_midi + 12 * (oct_add + oct_shift) + sc[idx]
```
*(Ex. en `C minor` avec `tonic_midi=60` : degré 0=`C4(60)`, degré 2=`Eb4(63)`, degré 4=`G4(67)`, degré 5=`Ab4(68)` — 0 erreur de gamme possible !)*

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

## Chapitre 5 — Recette Pas-à-Pas : Générer une Mélodie Agréable (Anti-Chaos LLM)

Pour qu'une IA textuelle crée une mélodie **chantable, émotive et immédiatement agréable** (validant le MIDI Gate dès le 1er essai), appliquer exactement ces **4 étapes** :

### Étape 1 — Créer UNE Cellule Motivique Courte (Mesure 1 = `A`)
* **3 à 5 notes maximum** sur 1 mesure (4 beats), en laissant au moins `0.75 à 1.5 beat` de silence.
* Choisir les hauteurs parmi les **degrés de la gamme** (jamais hors gamme), avec la 1re note (temps `0.0` ou anacrouse `0.5`) appartenant à **l'accord 1** (fondamentale, tierce ou quinte).
* Privilégier des mouvements **conjoints** (`±1` ou `±2` degrés de gamme, soit `1–4 st`). Si on fait un saut expressif (`+5` à `+8 st`), **revenir immédiatement de `1–3 st` en sens inverse** (*Gap-Fill*).
* *Exemple concret en `C minor` (accord `Cm7`, degrés 0-indexed `0=C, 2=Eb, 3=F, 4=G, 5=Ab, 6=Bb`)* :
  * `A` (mes. 1, beats `[0.0, 1.0, 1.5, 2.5]`, durées `[0.9, 0.45, 0.9, 0.9]`, degrés `[4, 5, 4, 2]` → `G4, Ab4, G4, Eb4`, silence `3.4..4.0`).

### Étape 2 — Développer sur 4 Mesures selon `(V, T)` (2 Plans Distincts)

Choisir obligatoirement le plan adapté à la Valence `V` :

* **Plan 1 — Courbe Ascendante (`V >= 0`, Pop / Lumineux / Épique)** : `A - A' - A'' [Apex <= MIDI 74] - B [Cadence + Silence]` :
  * **Mesure 1 (`A`)** : Exposé clair de la cellule (`MIDI 58–70`).
  * **Mesure 2 (`A'`)** : **Même squelette rythmique**, adapté aux notes de l'accord 2 (décalé de `±1` ou `±2` degrés) + silence fin de mesure (`3.0..4.0`).
  * **Mesure 3 (`A''` — Apex Unique)** : Monter vers **le point culminant unique** de la phrase (`+2` à `+5 st` au-dessus de `A`, **`<= MIDI 74`** par défaut), puis redescendre d'un pas conjoint (`1–3 st`).
  * **Mesure 4 (`B` — Cadence & Respiration)** : Redescendre doucement vers la tierce ou la tonique (`beat 0.0` et `1.5`), puis **silence complet sur beats `2.5..4.0`** (place pour la batterie, le fill ou un bref écho du contre-chant).

* **Plan 2 — Courbe Sombre / Mélancolique / Hypnotique (`V < 0`)** :
  * **Mesure 1 (`A`)** : Motif court (3–4 notes) centré sur la quinte/tierce mineure (`MIDI 58–67`), mouvement **descendant** (ex. soupir `b6 -> 5` ou `b3 -> 2 -> 1`).
  * **Mesure 2 (`A'`)** : Répétition presque identique ou légèrement abaissée (`-1` à `-2` degrés) pour créer un effet hypnotique/obsédant + silence en fin de mesure.
  * **Mesure 3 (`A''`)** : **Pas d'apex joyeux obligatoire**. Option A : tension sombre sur `b6` ou `b2` résolvant vers le bas (`b6 -> 5` ou `b2 -> 1`). Option B : maintien bas/statique avec variation rythmique subtile.
  * **Mesure 4 (`B`)** : Retombée grave vers la tonique `1` ou la quinte inférieure, suivie de **1.5 à 2.0 beats de silence réel**.

### Étape 3 — Sculpting Expressif (Vélocité, Durée/Gate, Micro-Strum)
* **Hiérarchie de vélocité** (ne jamais écrire `vel=100` partout) :
  * Temps fort (`0.0`) : `92–104` · Temps 3 (`2.0`) : `84–94` · Contretemps (`0.5, 1.5, 2.5`) : `68–82` · Note d'Apex mes. 3 (`V >= 0`) ou appoggiature `b6`/`b2` (`V < 0`) : `+6 à +10` · Fin de phrase (`B`) : `-10 à -15`.
* **Articulation (*Gate Length*)** :
  * Légato / Chant / Cordes / Pad : `0.90–0.98 × écart` (éviter chevauchement involontaire en mono).
  * Pluck / Guitare / Funk / Pizzicato : `0.45–0.70 × écart`.
* **Micro-Strum sur Accords Piano / Guitare / Harpe** : décaler chaque note d'un accord ascendant de `+0.015 à +0.035 beat` (`10–25 ms`) et réduire la vélocité des voix internes (`-8` par rapport à la note haute) → supprime instantanément le son "MIDI robotique".

### Étape 4 — Relais Basse & Contre-Chant
* **Mouvement contraire/oblique** : quand la basse monte d'un accord à l'autre, faire descendre ou tenir le Lead (garantit `contrary_or_oblique_ratio >= 0.55`).
* **Contre-chant** : placer ses 2–3 notes **uniquement dans les silences du Lead** (ex. beats `3.0–3.75` de mes. 2 et `2.5–3.75` de mes. 4), à une octave ou timbre distinct → `lead_overlap_conflict <= 0.15`.

---

## Chapitre 6 — Loi de l'Inertie Fréquentielle & 10 Matrices Rythmiques Exactes (Beats 0-Indexed)

### 6.1 Loi de l'Inertie Fréquentielle (Relation Hauteur / Durée)
> **Règle physique absolue** : Plus un son est grave, plus ses notes doivent être longues et espacées ; plus il est aigu, plus ses subdivisions peuvent être courtes. Une onde de $40\text{ Hz}$ a une période de $25\text{ ms}$ et requiert plusieurs cycles pour que sa hauteur soit perçue sans clic ni boue (*mud*). **Interdiction de toute figure $\le 1/16$ sous $100\text{ Hz}$** pour les sons maintenus (exception staccato Funk/Trap : imposer **`Release < 50 ms`** ou Gate complet avant la note suivante).

| Registre | Bande Fréquentielle | Valeurs Rythmiques Recommandées | Enveloppe ADSR Type | Instruments |
| :--- | :---: | :--- | :--- | :--- |
| **Sub-Basse** | `20–60 Hz` | Rondes, Blanches (`>= 1/2`) | Attaque lente/moyenne, Sustain élevé, Release contrôlé | Sine Sub, 808 tenue |
| **Basse** | `60–250 Hz` | Blanches, Noires, Croches pointées | Attaque nette, Decay moyen (`Release < 50 ms` si rapide) | Kick, Basse élec., Synth Bass |
| **Bas-Médiums** | `250–800 Hz` | Noires, Croches | Sustain moyen (accords) ou court (rythmique) | Pads, Piano MG, Guitare |
| **Hauts-Médiums** | `800 Hz–4 kHz` | Croches, Double-croches | Attaque précise, articulation dynamique | Voix, Lead synthé, Snare, Cuivres |
| **Aigus / Air** | `4–20 kHz` | Double-croches, Triple-croches, Triolets | Attaque instantanée, Decay & Release très courts | Hi-hats, Shakers, Cymbales |

### 6.2 Micro-Timing, Humanisation (Effet Haas) & Loi Anti-Mitraillette
* **Effet de précédence (Effet Haas)** : si Kick + Basse + Hi-hat frappent à `0.0 ms` exact, leurs crêtes s'additionnent (perte de *headroom*) et masquent les attaques. Un décalage de **`5 ms à 15 ms`** sépare chaque attaque tout en conservant un impact perçu comme unique.
* **Loi Anti-Mitraillette (motifs rapides `>= 1/16`)** : aucune frappe consécutive ne doit être identique — varier **vélocité** (ex. temps fort `100 %`, contretemps `65 %`, subdivisions `40 %`, **`±5 %`** aléatoire), **hauteur** (**`±5` à `±15 ct`**) et **filtre LPF** (coupure proportionnelle à la vélocité).

| Élément | Décalage par rapport à la grille | Effet Musical Produit |
| :--- | :---: | :--- |
| **Kick (Grosse caisse)** | **`0 ms`** (Ancrage absolu) | Référence temporelle fixe pour l'auditeur. |
| **Snare / Clap** | **`-5 ms` à `+10 ms`** | En avance (`-5 ms`) = urgent/nerveux · En retard (`+10 ms`) = lourd/*groovie*. |
| **Hi-hats / "Tics"** | **`+5 ms` à `+20 ms`** (ou Swing `52–58 %`) | Dégage l'attaque Kick/Snare et crée un balancement naturel. |
| **Basse / 808** | **`+10 ms` à `+25 ms`** | Laisse passer le transitoire initial du Kick avant l'arrivée de l'énergie grave. |

### 6.3 Bibliothèque Universelle de 10 Matrices Rythmiques Exactes (Beats 0-Indexed)

> Chaque mesure 4/4 vaut `4.0` beats (`0.00` à `3.75` par pas de `0.25`). Pour éviter tout groove cassé, partir de la matrice du genre cible puis ajouter des variations mineures en fin de bloc (mes. 4 et 8).

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

## Chapitre 7 — Palettes d'Instrumentation Cohérentes par Genre

Choisir une palette où les instruments **appartiennent au même univers acoustique** et occupent des **registres MIDI disjoints** :

| Famille de Genre | Profil Mix (`--profile`) | Basse / 808 (`MIDI 31–48`, `>= 49 Hz`) | Harmonie (`MIDI 48–67`) | Lead / Motif (`MIDI 55–72`, Apex `<= 74`) | Percussions / Texture |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Trap / Cloud / Drill** | `trap_rap` | `gl.sampler` 808 mono (`Voices=1`) | `gl.sampler` piano/pad WAV ou `gl.warmsynth` (LPF `<= 5.5 kHz`) | `gl.sampler` bell/pluck/flûte `mono_oneshot` ou `gl.pluck` (LPF `<= 7 kHz`) | Samples Trap (Kick, Clap/Snare, Hats) |
| **Pop / Synthwave** | `pop_rnb` | `gl.analog` / `gl.fingerbass` (octave bass) | `gl.dcosynth` / `gl.warmsynth` / `gl.keys` (LPF `<= 5.5 kHz`) | `gl.twinva` / `gl.keys` / `gl.plucker` / Voix (LPF `<= 7.5 kHz`) | Kick/Snare punchy + Shaker |
| **House / Techno / EDM** | `edm_club` | `gl.acid` / `gl.analog` / `gl.stepbass` | `gl.sawswarm` / `gl.wavetable` / `gl.dcosynth` (LPF `<= 5.5 kHz`) | `gl.wavetable` / `gl.pluck` / `gl.synth` (LPF `<= 7.5 kHz`) | `gl.kick` / samples 909 + Open Hat |
| **Lo-Fi / Boom Bap** | `lofi_chill` | `gl.fingerbass` / sub rond doux | `gl.keys` (EP/Rhodes) / `gl.sampler` piano doux | `gl.plucker` (guitare douce) / flûte / piano (`<= 72`) | Kick/Snare feutrés (LPF `8–10 kHz`) |
| **Rock / Indie / Metal** | `rock_metal` | `gl.fingerbass` | `gl.plucker` / `gl.twinstring` (+ `gl.dist` suivi de LPF) | `gl.twinstring` / `gl.analog` / Voix | `gl.drumkit` ou samples acoustiques |
| **Jazz / R&B / Soul** | `jazz_acoustic` | `gl.fingerbass` (walking/pocket) | `gl.keys` (Rhodes `m7`/`m9`/`Maj9`) | `gl.keys` / `gl.sampler` sax/flûte/guitare | Kit doux / rimshot / ride |
| **Classique / Cinéma / Ambient** | `classical_ambient` | `gl.string` (contrebasse/cello) / sub doux | `gl.string` / `gl.twinstring` / `gl.warmsynth` (pad lent) | `gl.sampler` piano WAV / `gl.string` / `gl.plucker` | Timbales/percs ciné légères ou aucune |

*(Notes registres : pour les tonalités `C`, `C#`, `D`, `D#`, `E`, `F`, `F#`, écrire la fondamentale 808 en Octave 2 `MIDI 36–42` plutôt qu'en Octave 1 `< 42 Hz`. Côté aigus, `MIDI 75–81` est réservée exclusivement aux timbres très doux/sinusoïdaux — flûte douce, cordes légato avec LPF `<= 5 kHz` — sur morceaux lumineux `valence > +0.3`, et strictement interdite sur synthés riches, cloches FM ou morceaux sombres.)*
