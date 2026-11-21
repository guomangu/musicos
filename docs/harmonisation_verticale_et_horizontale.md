# Stratégies et Règles Universelles d'Harmonisation Inter-Pistes (Verticale & Horizontale)

Ce document définit la méthode universelle permettant d'éliminer l'effet « collage de sons séparés » ou « cafouillis rythmique/harmonique » lors de la composition multi-pistes assistée par IA dans Gridline. Une piste peut être propre isolément et pourtant ruiner la cohésion globale si elle n'est pas **structurellement liée** aux autres pistes sur les plans **micro-tonal, rythmique, harmonique et acoustique**.

---

## 1. Diagnostic des Causes Racines du « Cafouillis » Multi-Pistes

Lorsque plusieurs pistes sonnent déconnectées ou confuses entre elles, la cause provient presque toujours d'une ou plusieurs des 5 défaillances suivantes :
1. **Dérive micro-tonale entre samples** : deux samples issus de banques différentes (ex. un piano à `+14 cents`, un pizzicato à `-31 cents`, un accord à `-13 cents`) cumulent jusqu'à `45 cents` d'écart (presque un quart de ton). L'oreille perçoit immédiatement des sons « faux » qui ne appartiennent pas au même univers.
2. **Poly-grooves concurrents (Boucles étrangères empilées)** : superposer une boucle brute (ex. boucle Techno 4/4 étirée depuis 125 BPM) par-dessus un groove Trap *half-time* crée deux horloges rythmiques incompatibles.
3. **Transitoires et pick-ups cachés à `t = 0`** : une boucle samplée peut contenir une note d'approche ou une queue d'accord dissonante dans ses premières `20–200 ms` (ex. un `Ré#` juste avant l'accord de `Mi` mineur), créant un clash brutal à chaque premier temps.
4. **Collision d'accords statiques et de mélodies indépendantes** : laisser un sample d'accord unique (`Em`) jouer en continu pendant que la 808 change de basse (`C`, `G`, `D`) ou que deux instruments mélodiques jouent en même temps dans la même fenêtre temporelle et fréquentielle.
5. **Absence de ciment acoustique partagé** : des samples secs issus de studios différents restent perçus comme des éléments juxtaposés s'ils ne partagent ni la même pièce acoustique (`FX Room`), ni une compression de bus cohésive (*Bus Glue*).

---

## 2. Stratégies d'Harmonisation Verticale (Cohésion à l'instant $t$)

La cohérence verticale est prioritaire : à chaque milliseconde, l'ensemble des pistes actives doit fonctionner comme **un seul méta-instrument unifié**.

### V1. Unification Micro-Tonale Stricte (`A4 = 440.0 Hz ± 5 cents`)
* **Règle :** Mesurer systématiquement le *tuning offset* (FFT / `librosa.estimate_tuning`) de **chaque** sample tonal (808, pianos, accords, guitares, voix, synthés) avant composition.
* **Action :** Aligner chaque sample sur le diapason exact `A4 = 440 Hz` via le paramètre `Fine tune` du sampler ou un `pitch_shift` lors de la dérivation dans `samples/`. Ne jamais tolérer plus de **`± 5 cents`** d'écart entre deux pistes harmoniques.
* **Précaution :** Si un one-shot possède un léger *pitch-drop* ou vibrato à l'attaque (ex. pizzicato ou 808), mesurer la hauteur sur le **corps stable** du son (`50–400 ms`), et trimmer toute attaque parasite.

### V2. Horloge Rythmique Unique (*Single Master Grid* & *Transient Locking*)
* **Règle d'exclusion des boucles concurrentes :** Ne jamais empiler une boucle audio brute provenant d'un autre style/tempo par-dessus un pattern programmé si leurs accents ne sont pas identiques.
* **Extraction en One-Shots propres (`t = 0`) :** Découper les boucles sources en **one-shots isolés** (stab d'accord isolé, kick isolé, clap isolé, hat isolé, rim/perc isolé) en supprimant toute queue ou note d'approche avant la transitoire (`start` exact sur le pic d'attaque, `fade-in 1–2 ms`), puis les **re-programmer sur l'unique grille MIDI maîtresse** du morceau.
* **Verrouillage d'Impact Composite (*Composite Hit*) :** Sur les temps forts structurels (ex. temps `1` et contretemps clés `1.5`, `2.75`), verrouiller simultanément (`± 5 ms`) l'attaque du **Kick** + la fondamentale de la **808** + l'attaque de l'**accord / stab harmonique** afin qu'ils fusionnent en un seul impact massif et cohérent.
* **Un seul moteur de hi-hats principal :** Utiliser un seul motif de hi-hats cohérent (éventuellement doublé en stéréo/couches sur les **mêmes** subdivisions exactes), jamais 2 ou 3 boucles de hats aux rythmes indépendants.

### V3. Grille d'Accords Maîtresse, Voice Leading & Zéro Transposition Parallèle d'Accords Samplés
*(Voir les règles normatives détaillées dans [regles_melodie_et_cohesion_inter_pistes.md](regles_melodie_et_cohesion_inter_pistes.md) — Section C, V1 & V2.)*
* **Règle :** Définir `harmonic_core` dans `song_blueprint.json` et générer la partition mère `master_sketch.mid` avant toute orchestration :
  * **Conduite des voix minimale (*Smooth Voice Leading*)** : interdiction des accords en position fondamentale qui sautent en bloc. Calculer des renversements conservant les notes pivots avec un déplacement moyen **`<= 2.5` demi-tons** via `optimize_voice_leading(...)`.
  * **Interdiction de transposition parallèle d'accords complets samplés** : ne jamais transposer un sample audio contenant déjà un accord complet (majeur ou mineur) sur tous les degrés d'une gamme (cela génère des accords parallèles hors gamme). Utiliser exclusivement un instrument polyphonique (`Headroom Piano Clean.sfz`, `gl.dcosynth`, `gl.analog`), un one-shot **monophonique** rejoué en polyphonie MIDI, ou un *power chord* strict (fondamentale + quinte sans tierce).
  * La **808 / Basse** joue la fondamentale de l'accord actif ; les **mélodies** respectent les notes d'accord sur temps forts (`strong_beat_chord_tone_ratio >= 0.80`) et maintiennent un **mouvement contraire ou oblique** face à la basse (`contrary_or_oblique_ratio >= 0.55`).

### V4. Règle du Relais Vertical (*Call-and-Response* — Jamais deux leaders simultanés)
* **Séparation temporelle dans la mesure :** Au lieu de faire jouer `Accords + Piano A + Piano B` à chaque temps, répartir les interventions en relais :
  * L'**Harmonie (`Chords Stab / Pad`)** frappe sur les appuis forts (ex. `beat 0.0` et `beat 2.0` ou `1.5`) puis laisse de l'espace.
  * Le **Contre-chant (`Piano / Pluck`)** répond uniquement dans les **trous rythmiques** laissés libres (ex. `beat 1.5`, `2.5`, `3.5`).
* **Séparation de registre :** Limiter à **1 seul motif mélodique actif par registre** (un motif grave-médium en couplet, un motif aigu en refrain). Si deux instruments mélodiques coexistent, ils doivent être espacés d'au moins **1 octave** et jouer en question/réponse.

### V5. Ciment Acoustique Partagé (*Shared Space, Bus Glue & Mirror EQ*)
* **Pièce acoustique unique (`FX Room`) :** Envoyer toutes les pistes mélodiques (`6–12%`) et le snare/clap/perc (`3–6%`) vers **un seul bus réverb court partagé** (`RT60 <= 1 × T_noire`, `HPF >= 400 Hz`, `Dry = 0%`). Le cerveau perçoit ainsi tous les instruments dans la **même salle physique**.
* **Compression de Bus Cohésive (*Bus Glue*) :**
  * `BUS Drums` (`Ratio 2.5:1`, `Attack 12 ms`, `Release 85 ms`) pour souder Kick, Snare, Clap et Hats.
  * `BUS Music` (`Ratio 2.0:1`, `Attack 20 ms`, `Release 110 ms`) pour souder Accords, Pianos et Textures.
  * `Master Comp` (`Ratio 1.6:1`, `Threshold -10 à -12 dB`, `GR 1–2 dB`) pour que l'instru entière respire sur un même battement.
* **Égalisation Miroir (*Complementary / Dynamic Carve EQ*) :** Chaque fois qu'un instrument leader est accentué dans une bande (`800 Hz – 3 kHz`), creuser symétriquement (`-2 à -4 dB`) sur les accords ou pads d'accompagnement (`TDR Nova` / `ZL Equalizer 2` en Live, ou EQ complémentaire `gl.eq` en Headless).

### V6. Ciment Timbral Collectif (`BUS Music` Sonic Glue & Micro-Ducking)
En complément de `FX Room`, **`BUS Music`** (Insert 11 ou bus harmonique dédié) doit appliquer une chaîne unificatrice complète sur l'ensemble des pistes mélodiques et harmoniques :
1. **Coupe-haut unificateur (`gl.eq`)** : `Low cut 120–150 Hz` + `High cut (LPF) 11–14 kHz` ($12\text{ dB/oct}$) pour fondre la texture dans un même espace fréquentiel et laisser l'air $> 12\text{ kHz}$ aux hi-hats.
2. **Saturation harmonique collective** : saturation très légère (`gl.dist` mode `Soft`, `Oversampling = 4x`, `Mix = 0.08–0.12` en portable natif, ou `CHOWTapeModel` subtil en Live GUI).
3. **Compression de bus (`gl.comp`)** : `Ratio = 2.0`, `Attack = 0.020 s`, `Release = 0.110 s`, seuil ajusté pour `1.5–2.5 dB` de GR.
4. **Micro-Ducking rythmique (`gl.comp`)** : `Detector input = Sidechain` alimenté par un send pré-fader du `Kick`, `Attack = 0.005 s`, `Release = 0.080 s` ($\approx 1/8$ de temps), `1.5–2.5 dB` de GR pour faire respirer l'harmonie sur le groove.

---

## 3. Stratégies d'Harmonisation Horizontale (Continuité Causale dans le Temps)

Un morceau est fluide lorsque chaque mesure et chaque section découlent logiquement de la précédente.

### H1. Fil Conducteur Thématique (*Leitmotiv Transformation*)
* **Règle :** Ne pas changer d'univers sonore entre les sections. Conserver **le même noyau harmonique (les 4 accords) et la même signature mélodique**, en ne transformant que :
  * le **timbre / filtrage** (ex. *Intro* : Pad filtré `LPF 900 Hz` ; *Refrain* : Stabs ouverts + octave haute `+12 st`) ;
  * la **densité rythmique** (*Couplet* : motif pizzicato épuré et 808 bounce courte ; *Refrain* : stabs pleins et 808 sostenuto) ;
  * le **registre** (montée `+12 st` au Refrain 2 / 3 pour élever l'énergie sans ajouter de nouvelles mélodies concurrentes).

### H2. Conduite des Voix & Approches (*Voice Leading*) aux Transitions (M4 / M8)
* Sur les 2 derniers temps de la 4e ou 8e mesure :
  * Utiliser une **note de passage diatonique** sur la 808 ou le piano (ex. quinte `B` ou 7e `D` menant à `E`) pour créer un appel gravitationnel vers le temps `1` suivant.
  * Ne jamais interrompre un accord en suspens : résoudre ou couper proprement sur le temps de respiration.

### H3. Ponts Causaux (*Transition Anchors*) et Nettoyage des Queues
* Toute transition majeure (`Intro -> Refrain`, `Couplet -> Pont`, `Pont -> Refrain`) doit être liée physiquement par un élément pont :
  * *Riser* inversé calibré sur exactement **4 temps** (`1` mesure) se terminant pile sur l'impact du drop.
  * *Fill* de snare/clap/perc sur les 2 derniers temps, ou **silence d'aspiration** (coupure simultanée de la 808, des drums et du volume de `FX Room` sur le 4e temps).

### H4. Arc Dynamique Réel (`3 à 5 dB` de contraste)
* Ne pas écraser l'Intro et le Pont au même niveau LUFS que les Refrains via un compresseur/limiteur Master réglé trop bas :
  * *Intro / Pont / Outro* : `-15.5` à `-14.0 LUFS`
  * *Couplets* : `-12.5` à `-11.5 LUFS`
  * *Refrains / Climax* : `-10.8` à `-9.8 LUFS`

---

## 4. Checklist Anti-Cafouillis & Gate de Musicalité (À valider avant tout rendu final)

* [ ] **Test de la Réduction Unique (`master_sketch.mid` + Test de la Nudité) :** La basse, les accords (en *voice leading*) et le Lead chantable (`A - A' - A'' - B`) ont-ils été validés seuls sur piano sec avant d'ajouter la batterie ?
* [ ] **Gate `audit_midi_harmony` (`strict_musicality=True` -> `PASS`) :**
  * `stepwise_ratio >= 0.65` et `gap_fill_compliance >= 0.60`
  * `melody_rest_ratio` dans `[0.20, 0.50]`
  * `strong_beat_chord_tone_ratio >= 0.80`
  * `voice_leading_avg_semitones <= 2.50`
  * `lead_overlap_conflict <= 0.15`
  * `contrary_or_oblique_ratio >= 0.55`
* [ ] **Test d'Accordage (`± 5 ct`) :** Tous les samples tonaux (808, accords, pianos, synthés) ont-ils été mesurés et recalés à `A4 = 440.0 Hz` ?
* [ ] **Test de l'Horloge Unique :** Toutes les boucles étrangères ont-elles été remplacées par des one-shots propres (`t = 0`, sans pick-up caché) verrouillés sur la grille maîtresse du morceau ?
* [ ] **Test de Verrouillage Harmonique :** À chaque mesure, la 808, les accords (sans transposition parallèle d'accord complet samplé) et les mélodies suivent-ils exactement `harmonic_core` ?
* [ ] **Test du Relais (*Call-and-Response*) :** Les attaques des accords et des contre-mélodies alternent-elles dans le temps au lieu de se chevaucher sur chaque temps (`lead_overlap_conflict <= 0.15`) ?
* [ ] **Test du Ciment Timbral & Acoustique :** `BUS Music` applique-t-il `EQ (LPF 11-14k)` + `gl.dist Soft (Mix 8-12%)` + `gl.comp Glue` + `Kick Sidechain Micro-Ducking`, et toutes les familles partagent-elles `FX Room` (`Dry = 0`, `Dry (full) = 0`) ?
* [ ] **Test d'Arc Dynamique :** Existe-t-il au moins `3 dB` d'écart LUFS réel entre Intro/Pont et Refrains ?
