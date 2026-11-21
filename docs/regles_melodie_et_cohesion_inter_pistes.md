# Règles Normatives de Mélodie & Cohésion Inter-Pistes (*Single-Core Top-Down*)

Ce document définit les règles obligatoires de composition mélodique, de conduite des voix (*voice leading*) et de ciment timbral pour éliminer l'effet « collage de sons » et garantir une musicalité chantable et organique.

---

## A. Loi Fondamentale — La Réduction Unique (`master_sketch.mid`)

* **Interdiction absolue** de programmer la basse, les accords et les mélodies dans des scripts indépendants sans partition mère préalable.
* Tout morceau doit d'abord être composé **Top-Down** dans un fichier MIDI unique (`master_sketch.mid`) à 3 voix sur un seul piano acoustique sec (`Headroom Piano Clean.sfz`, sans batterie ni effets) :
  * **Voix 1 — Basse (`Bass`)** : monophonique, registre `C2–B2` (MIDI `36–47`) ou sub `E1–B2` (`28–47`). Joue la fondamentale de l'accord actif + notes d'approche en fin de mesure 4/8.
  * **Voix 2 — Harmonie (`Chords`)** : 2 à 3 notes simultanées, registre `C3–B3` (MIDI `48–59`). Conduite des voix par renversements minimaux (`<= 2.5` demi-tons de mouvement moyen).
  * **Voix 3 — Top-Line (`Lead`)** : monophonique, registre `C4–G5` (MIDI `60–79`). Mélodie principale chantable et mémorisable.
* **Test de la Nudité** : rendre `master_sketch.mid` sur `Headroom Piano Clean.sfz` seul. Si la réduction ne sonne pas claire, cohérente et mélodieuse sans batterie, la réécrire avant d'ajouter le moindre drum.
* **Gate Bloquant (`audit_midi_harmony`)** : `master_sketch.mid` doit obtenir `musicality_verdict == "PASS"` (`strict_musicality=True`) avant toute orchestration multi-pistes ou ajout de batterie.

---

## B. Grammaire de la Mélodie (Règles M1 à M5)

### M1 — Mouvement Conjoint & Loi du *Gap-Fill* (70 / 30)
* Au moins **$65\text{ \%}$ à $80\text{ \%}$** des intervalles consécutifs du `Lead` doivent être conjoints ou petits ($\le 4$ demi-tons : répétition, demi-ton, ton, tierce).
* Tout saut large ($\ge 5$ demi-tons) doit être suivi d'un retour conjoint ($1$ à $3$ demi-tons) dans la **direction opposée** (*gap-fill*, conformité $\ge 60\text{ \%}$).
* Ambitus maximal du `Lead` : **$12$ à $19$ demi-tons** (1 à 1,5 octave), comme une voix humaine.

### M2 — Économie Motivique (`A - A' - A'' - B`)
* Définir une **cellule mère** unique de $3$ à $5$ notes sur $1$ à $2$ temps (`intervals_from_tonic` + `rhythm_beats` dans `harmonic_core`).
* **Mesure 1 (`A`)** : exposition de la cellule sur l'accord 1.
* **Mesure 2 (`A'`)** : même rythme exact, adapté aux notes de l'accord 2.
* **Mesure 3 (`A''`)** : variation montante atteignant l'**Apex** (`apex_midi`, note la plus aiguë de la phrase, atteinte une seule fois sur les 4 mesures).
* **Mesure 4 (`B`)** : détente / cadence conclusive (résolution vers la tonique/quinte ou suspension sur la 5e/7e).

### M3 — Respiration Obligatoire (Ratio Son / Silence)
* Maintenir **$20\text{ \%}$ à $50\text{ \%}$** de silence réel dans la piste `Lead` (`melody_rest_ratio` entre `0.20` et `0.50`).
* Libérer systématiquement le dernier temps ou demi-temps des mesures 2 et 4 pour laisser respirer la batterie et le contre-chant.

### M4 — Gravité Harmonique (Temps Forts vs Temps Faibles)
* Sur les temps forts (`beat 0.0`, `beat 2.0` et chaque attaque d'accord), au moins **$80\text{ \%}$** des notes du `Lead` doivent appartenir à la triade ou 7e de l'accord courant (`strong_beat_chord_tone_ratio >= 0.80`).
* Les notes étrangères à l'accord (notes de passage, broderies) sont confinées aux subdivisions faibles (`0.5`, `1.5`, `2.5`, `3.5`) et doivent se résoudre par mouvement conjoint ($\pm 1$ ou $\pm 2$ demi-tons).

### M5 — Couplage Rythmique Mélodie / Groove (*Pocket Alignment*)
* Ancrer les attaques clés du `Lead` sur le temps `1` (ou en anacrouse juste avant `1`) et créer un appui ou une respiration autour du **Clap/Snare sur le temps `3`** (grille Trap *half-time*).

---

## C. Harmonisation Verticale Stricte (Règles V1 à V6)

### V1 — Conduite des Voix Minimale (*Smooth Voice Leading*)
* Interdiction des accords en position fondamentale qui sautent en bloc.
* **Note Pivot** : toute note commune entre l'accord $N$ et l'accord $N+1$ reste à la même hauteur MIDI.
* **Moindre Mouvement** : le déplacement moyen des voix entre deux accords consécutifs doit être **$\le 2.5$ demi-tons** (`voice_leading_avg_semitones <= 2.50`, calculé par `optimize_voice_leading`).

### V2 — Interdiction de Transposition Parallèle d'Accords Samplés
* Ne jamais transposer un sample audio contenant déjà un accord complet (majeur ou mineur) sur tous les degrés d'une gamme (cela génère des accords parallèles hors gamme, ex. `Cm`, `Gm`, `Dm` en `Mi` mineur).
* Utiliser exclusivement :
  1. Un instrument polyphonique (`Headroom Piano Clean.sfz`, `jRhodes3c`, `gl.dcosynth`, `gl.analog`),
  2. Un one-shot **monophonique** rejoué en polyphonie MIDI, ou
  3. Un *power chord* strict (fondamentale + quinte **sans tierce**).

### V3 — Mouvement Contraire & Oblique (Basse vs Lead)
* Lorsque la basse (`808`) monte, la mélodie (`Lead`) tend à descendre ou rester stable, et inversement (`contrary_or_oblique_ratio >= 0.55`).
* Interdiction de doubler exactement la ligne de 808 à l'octave sur plus de 2 temps consécutifs.

### V4 — Complémentarité d'Enveloppes & Relais d'Attaques
* Jamais deux attaques percussives (`Attack < 15 ms`) simultanées dans la même octave des médiums (`400 Hz – 3 kHz`).
* Opposer une attaque rapide (*Pluck / Piano staccato*) à une attaque douce (*Pad / Strings*, `Attack > 40 ms`), ou alterner en Question/Réponse (*Call-and-Response*).
* Le chevauchement d'attaques simultanées ($\pm 40\text{ ms}$) entre deux pistes mélodiques à moins de $12$ demi-tons doit rester **$\le 15\text{ \%}$** (`lead_overlap_conflict <= 0.15`).

### V5 — Égalisation Miroir Dynamique (*Dynamic Carve*)
* Creuser de $-2$ à $-4\text{ dB}$ la zone `800 Hz – 3 kHz` des pistes d'accompagnement (`Chords / Pad`) lorsque le `Lead` ou la `Voix` joue (via `TDR Nova` / `ZL Equalizer 2` en Live, ou automation / EQ complémentaire `gl.eq` en Headless).

### V6 — Ciment Timbral & Acoustique Partagé (*Sonic Glue*)
* **`BUS Music`** regroupe toutes les pistes harmoniques et mélodiques et applique cette chaîne obligatoire :
  1. **Coupe-haut unificateur (`gl.eq`)** : `Low cut 120–150 Hz` + `High cut (LPF) 11–14 kHz` ($12\text{ dB/oct}$) pour fondre les timbres ensemble et laisser l'air $> 12\text{ kHz}$ aux hi-hats.
  2. **Saturation harmonique collective** : très légère saturation (`gl.dist` mode `Soft`, `Oversampling = 4x`, `Mix = 0.08–0.12` en portable natif, ou `CHOWTapeModel` subtil en Live GUI).
  3. **Compression de bus (`gl.comp`)** : `Ratio = 2.0`, `Attack = 0.020 s`, `Release = 0.110 s`, seuil ajusté pour `1.5–2.5 dB` de réduction de gain.
  4. **Micro-Ducking rythmique (`gl.comp`)** : `Detector input = Sidechain` alimenté par un envoi pré-fader du `Kick`, `Attack = 0.005 s`, `Release = 0.080 s` ($\approx 1/8$ de temps), `1.5–2.5 dB` de GR pour faire respirer l'harmonie sur le groove.
* **Pièce acoustique unique (`FX Room`)** : tous les instruments mélodiques (`8–14%`) + Snare/Clap (`3–6%`) vers le même `gl.reverb` court (`Decay <= 1 × T_noire`, `HPF >= 450 Hz`, `Dry = 0`, `Dry (full) = 0`), coupé à `0` sur les temps de silence/transition.

---

## D. Harmonisation Horizontale (Règles H1 à H3)

* **H1 — Dérivation Thématique Unique** : ne jamais inventer une mélodie sans rapport entre Intro, Couplet, Pont et Refrain. Faire évoluer le même motif par 4 leviers uniquement : **octave** (`-12` / `+12`), **densité** (squelette sur temps forts en couplet vs motif complet au refrain), **filtrage** (`LPF` fermé → ouvert), et **écho** (contre-chant reprenant les 2 dernières notes du Lead dans son silence).
* **H2 — Ponts de Conduite de Voix (*Voice-Leading Bridges*)** : sur le 4e temps des mesures 4 et 8, utiliser une note de liaison (5e, 7e ou sensible) dans la basse ou le Lead qui résout par demi-ton ou ton sur le temps `1` suivant.
* **H3 — Contraste Dynamique Inter-Sections** : garantir **$3$ à $5\text{ dB}$** d'écart LUFS réel entre Intro/Pont (`-15.5` à `-14.0 LUFS`) et Refrains (`-11.0` à `-10.0 LUFS`).
