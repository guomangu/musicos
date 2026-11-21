# Spécifications Universelles : Théorie Harmonique, Rythmique et Sound Design de la Trap

Ce document constitue une base de connaissances formelle et algorithmique destinée à la génération procédurale et à la composition d'instrumentales de rap et de trap (notamment orientées vers des esthétiques *planantes* et *rythmées*). Il établit les corrélations strictes entre la hauteur des notes (harmonie verticale), la progression temporelle (harmonie horizontale), la densité rythmique et l'ingénierie sonore.

---

## 1. Architecture Harmonique : Verticalité et Horizontalité

Dans la musique Trap, l'harmonie est minimaliste mais extrêmement sensible aux tensions d'intervalles en raison de la prédominance des basses fréquences (808) et des espaces fréquentiels laissés libres pour la voix.

### 1.1 Harmonie Verticale (Empilement Fréquentiel et Voicing)
L'harmonie verticale régit la superposition des notes à un instant $t$ précis. Pour éviter le masquage fréquentiel (boue sonore dans le bas du spectre), la distribution doit respecter des règles strictes d'espacement :

*   **Registre Sub-Bass ($20\text{ Hz} - 60\text{ Hz}$) & Bass ($60\text{ Hz} - 120\text{ Hz}$) :** Exclusivement réservé à la **808** (ou basse synthétique). La 808 joue systématiquement la **fondamentale** de l'accord courant. Les renversements (tierce ou quinte à la basse) sont proscrits à 95 % car ils détruisent l'assise gravitationnelle du morceau.
*   **Registre Bas-Médium ($120\text{ Hz} - 350\text{ Hz}$) :** Zone de sécurité ("Mud zone"). Les instruments harmoniques (pads, pianos, synthés) doivent être filtrés (High-Pass Filter à $120\text{ Hz} - 150\text{ Hz}$) pour ne jamais entrer en collision avec les harmoniques supérieures de la 808.
*   **Registre Médium et Aigu ($350\text{ Hz} - 8\text{ kHz}$) :** C'est ici que se déploient les accords et les mélodies.
    *   *Voicing ouvert (Open Voicing) :* Pour un son planant, on écarte les notes de l'accord. Par exemple, au lieu de jouer $Do_3 - Mi\flat_3 - Sol_3$, on joue $Do_3 - Sol_3 - Mi\flat_4$ (la tierce est montée d'une octave).
    *   *Extensions privilégiées :* L'ajout de la **7ème mineure** ($m7$), de la **9ème** ($add9$ ou $m9$) et de la **11ème** ($m11$) est fondamental pour obtenir une couleur "planante" (Cloud / Ambient Trap).

### 1.2 Harmonie Horizontale (Progressions et Mouvement Temporel)
L'harmonie horizontale définit l'enchaînement des accords dans le temps (généralement sur des boucles de 4 ou 8 mesures). Contrairement à la musique pop, la trap n'utilise presque jamais de cadences parfaites classiques ($V \rightarrow I$). Elle repose sur des **mouvements modaux** et des **tensions de demi-tons**.

*   **Les Gammes de Référence :**
    1.  **Mineure Naturelle (Mode Éolien) :** La base standard (ex: $La - Si - Do - Ré - Mi - Fa - Sol$).
    2.  **Mode Phrygien :** Caractérisé par une seconde mineure ($b2$) par rapport à la tonique (ex: $Mi - Fa - Sol - La - Si - Do - Ré$). Crée une tension sombre et immédiate.
    3.  **Mineure Harmonique :** Introduction de la 7ème majeure ($+7$) dans une gamme mineure (ex: $La - Si - Do - Ré - Mi - Fa - Sol\#$). Très utilisé pour des sonorités dramatiques ou orchestrales.
    4.  **Mode Dorien :** Mineur avec une 6ème majeure ($\natural 6$). Utilisé pour une vibe planante, nostalgique ou "soulful" (ex: $Ré - Mi - Fa - Sol - La - Si - Do$).

*   **Les Progressions Types (Boucles de 4 mesures) :**
    *   *Boucle Sombre / Agressive (Tension de demi-ton) :* $i \rightarrow VI$ (ex: $Cm \rightarrow A\flat$) ou $i \rightarrow bII$ (ex: $Cm \rightarrow D\flat$, mouvement phrygien). Le passage d'un intervalle d'un demi-ton entre la 5te de l'accord $i$ ($Sol$) et la fondamentale de l'accord $VI$ ($La\flat$) génère une lourdeur caractéristique.
    *   *Boucle Planante / Mélancolique :* $i7 \rightarrow VI_{maj7} \rightarrow iv7 \rightarrow v7$ ou $vi \rightarrow IV \rightarrow i \rightarrow V$ (en mineur : $i \rightarrow VI \rightarrow III \rightarrow VII$).
    *   *Boucle Hypnotique (Statique) :* Un seul accord mineur (ex: $Fm9$) maintenu sur 4 mesures, où seule la ligne de basse (808) et une mélodie supérieure (arpège ou lead) créent le mouvement horizontal.

---

## 2. Interaction 808, Mélodie et Hi-Hats (Couplage Rythmico-Harmonique)

L'essence du groove Trap réside dans la synchronisation mathématique et fréquentielle entre la 808 (basse) et les Hi-Hats (hautes fréquences), encadrant l'espace mélodique.

### 2.1 Règles d'Agencement des 808
*   **Alignement Harmonique :** La 808 doit impérativement jouer la note fondamentale de l'accord au moment où celui-ci change (généralement sur le temps 1 de la mesure 1 et de la mesure 3).
*   **Mouvements Mélodiques de la 808 :** Entre deux changements d'accords, la 808 peut effectuer des sauts d'intervals pour dynamiser le rythme :
    *   *Saut d'octave ($+12$ demi-tons) :* Utilisé en fin de mesure pour relancer l'énergie sans créer de dissonance.
    *   *Passage par la Quinte ($+7$ demi-tons) :* Crée une tension rythmique forte avant de retomber sur la fondamentale.
    *   *Glissandos (Slides / Portamento) :* Transition continue de hauteur entre deux notes (souvent vers l'octave supérieure ou la tierce mineure), typique de la Drill et de la Trap moderne. La durée du slide doit être indexée sur le tempo (généralement $1/16$ ou $1/8$ de temps).
*   **Relation Kick / 808 :** Le Kick (grosse caisse) est accordé (pitché) pour frapper exactement sur la fondamentale de la gamme ou reste neutre (atonique) avec un coupe-bas à $45\text{ Hz}$ pour laisser passer le sub de la 808 via un *sidechain* rapide.

### 2.2 Règles d'Agencement des Hi-Hats (Charlestons)
Les Hi-Hats dictent la subdivision temporelle et l'intensité perçue du morceau. Ils ne sont pas mélodiques au sens strict, mais leur placement rythmique répond à l'harmonie :
*   **La Grille de Base :** Un motif continu en croches ($1/8$, soit "two-step") ou en doubles croches ($1/16$) établit le pouls.
*   **Les Rolls (Roulements) :** Subdivisions rapides en triolets ($1/16T$, $1/32$, $1/32T$, $1/64$).
    *   *Règle de placement :* Les rolls se placent stratégiquement juste avant les frappes de **Snare/Clap** (temps 3 de chaque mesure en demi-temps) ou juste avant un **changement d'accord / frappe de 808** pour annoncer la résolution harmonique.
*   **Pitch-Shifting des Hi-Hats :** Pour renforcer la dimension planante ou sombre, on peut faire descendre le pitch des Hi-Hats lors d'un roll (ex: descente de $0$ à $-5$ demi-tons sur 4 pas de $1/32$), créant un effet d'aspiration vers le grave (la 808).

---

## 3. Matrice d'Adaptation : Intensité Rythmique vs. Évolution Harmonique

Pour répondre à la problématique centrale — *comment accorder les accords et les éléments selon l'intensité et l'évolution du rythme* — l'agent doit appliquer une **loi de conservation de l'espace fréquentiel et cognitif**. Plus le rythme est complexe et dense, plus l'harmonie doit être simple et statique (et inversement).

| Phase / Section du Morceau | Intensité Rythmique | Évolution Harmonique & Voicing | Comportement 808 & Basse | Densité des Hi-Hats & Percussions |
| :--- | :--- | :--- | :--- | :--- |
| **Intro / Outro** | **Très Faible** (0 à 20 %) | **Riche et Évolutive :** Accords complets ($m9$, $maj7$), arpèges lents, nappes (pads) avec beaucoup de réverbération et filtrage passe-bas (Low-Pass) progressif. | **Absente** ou simple nappe de sub-basse tenue (Reese bass) sans attaque percussive. | **Absents** ou très espacés (ex: un seul hat sur les contretemps, filtré). |
| **Couplet (Partie 1 - Calme)** | **Modérée** (40 à 60 %) | **Espace Aéré :** Suppression des fréquences médiums des accords pour laisser place à la voix. Utilisation d'un seul motif mélodique simple (ostinato) dans les aigus ou les médiums. | **Minimale :** Frappes espacées uniquement sur les temps forts (temps 1 et 3). Peu ou pas de slides. | **Régulière :** Two-step ($1/8$) simple avec 1 ou 2 rolls discrets en fin de boucle de 4 mesures. |
| **Pré-Refrain (Build-up)** | **Croissante** (60 à 80 %) | **Tension Ascendante :** Montée en fréquence (ouverture des filtres), ajout d'une note de tension (7ème ou 9ème) ou montée d'une octave d'un élément mélodique. | **Coupure (Mute) :** Souvent coupée sur les 2 ou 4 derniers temps avant le refrain pour maximiser l'impact du retour. | **Accélération :** Multiplication des rolls ($1/32$), ajout de percussions secondaires (rimshots, open hats). |
| **Refrain (Drop / Chorus)** | **Maximale** (90 à 100 %) | **Directe et Impactante :** Superposition de la mélodie principale + une contre-mélodie ou un lead puissant. Accords plaqués courts (staccato) ou arpèges rapides pour ne pas noyer la rythmique. | **Complexe et Puissante :** Pattern syncopé, sauts d'octaves, saturation accrue (harmoniques hautes audibles), couplée au Kick. | **Dense et Syncopée :** Mélange de $1/8$, $1/16$ et triolets ($1/16T$), variations de vélocité et de panoramique (panning gauche/droite). |
| **Pont (Bridge)** | **Faible / Désynchronisée** (30 %) | **Variation Modale :** Changement de tonalité temporaire (ex: passage au relatif majeur) ou ralentissement de moitié (Half-time) sur les accords. | **Reese Bass** (basse continue modulée) remplaçant la 808 percussive. | **Minimaliste :** Disparition des kicks et snares principaux, maintien d'une texture rythmique légère. |

---

## 4. Typologie des Sous-Genres de Trap et Paramètres Sonores

L'agent doit sélectionner un profil esthétique qui dicte le tempo (BPM), le choix des timbres (sound design) et les règles harmoniques spécifiques.

### 4.1 Cloud Trap / Rap Planant (ex: Travis Scott, Don Toliver, PNL)
*   **Tempo :** $115 - 135\text{ BPM}$ (ou demi-temps ressenti à $60-68\text{ BPM}$).
*   **Harmonie :** Accords mineurs étendus ($m9$, $m11$), modes Dorien et Éolien. Progressions lentes (un accord par mesure ou toutes les 2 mesures).
*   **Sound Design & Sonorités :**
    *   *Synthés :* Pads analogiques chauds, nappes vocales (choeurs synthétiques), guitares électriques noyées dans la réverbération et le delay (ping-pong delay).
    *   *Traitement :* Utilisation massive du *Low-Pass Filter* (effet "sous l'eau"), *Chorus*, *Phaser* et désaccordage léger (*Detune* de $\pm 10$ à $20\text{ cents}$) pour élargir l'image stéréo.
    *   *808 :* Très ronde, longue queue (decay long), peu de distorsion, sub-basse profonde ($35 - 50\text{ Hz}$).
    *   *Drums :* Snares et claps avec une réverbération longue (tail reverb), hi-hats doux (coupés au-dessus de $12\text{ kHz}$ pour éviter l'agressivité).

### 4.2 Dark / Hard Trap (ex: Future, Metro Boomin, 21 Savage)
*   **Tempo :** $130 - 150\text{ BPM}$.
*   **Harmonie :** Mode Phrygien et Mineure Harmonique. Répétition obsessionnelle de boucles de 2 ou 4 notes avec intervalles de demi-tons ($b2$ et $b6$).
*   **Sound Design & Sonorités :**
    *   *Instruments :* Pianos sombres (registre grave), cloches tubulaires (Tubular Bells), cordes orchestrales (violons en staccato/pizzicato), flûtes ethniques ou synthétiques sombres.
    *   *Traitement :* Saturation à bande (Tape saturation), effets de ralentissement (Half-Speed / Gross Beat) qui transposent la mélodie une octave plus bas à demi-vitesse.
    *   *808 :* Type "SpinZ" ou "Zay" avec une attaque franche (punchy) et une saturation moyenne dans les médiums ($150 - 300\text{ Hz}$) pour percer sur petits haut-parleurs.
    *   *Drums :* Claps secs et claquants, hi-hats très brillants et rapides avec des rolls en $1/32$ et $1/64$.

### 4.3 Rage / Hyper-Trap (ex: Playboi Carti, Yeat, Ken Carson)
*   **Tempo :** $140 - 165\text{ BPM}$.
*   **Harmonie :** Très simple, souvent basée sur des **Power Chords** (fondamentale + quinte, sans tierce) ou des arpèges synthétiques ultra-répétitifs sur 2 accords maximum ($i \rightarrow VI$).
*   **Sound Design & Sonorités :**
    *   *Synthés :* Leads numériques agressifs (ondes en dents de scie / Sawtooth, ondes carrées / Square), synthétiseurs type Serum/Vital avec beaucoup d'unisson (SuperSaws) et de modulation de fréquence (FM).
    *   *808 :* Extrêmement distordue (Soft/Hard Clipping), occupant une large bande fréquentielle jusqu'à $1\text{ kHz}$. La 808 agit presque comme un instrument mélodique saturé.
    *   *Drums :* Kicks absents (la saturation de la 808 fournit l'attaque), snares courtes, hi-hats très mécaniques et soutenus.

### 4.4 Melodic / Emotional Trap / Pain Beats (ex: Lil Baby, Rod Wave, Polo G)
*   **Tempo :** $145 - 175\text{ BPM}$ (flux rythmique rapide mais ressenti émotionnel).
*   **Harmonie :** Progressions diatoniques mineures complètes ($i \rightarrow VI \rightarrow III \rightarrow VII$ ou $iv \rightarrow VI \rightarrow i \rightarrow V$). Utilisation fréquente de notes de passage (passing tones) et d'appoggiatures.
*   **Sound Design & Sonorités :**
    *   *Instruments :* Piano acoustique expressif (avec variations de vélocité importantes pour simuler le jeu humain), guitares acoustiques arpégées, voix pitchées (Vocal Chops).
    *   *808 :* Mélodique, suivant fidèlement les renversements ou les notes de passage de la main gauche du piano, avec une attaque douce.
    *   *Drums :* Très dynamiques, utilisation intensive de percussions organiques (rimshots, triangles, open hats sur les temps faibles).

### 4.5 UK / NY Drill (Dérivé rythmique majeur)
*   **Tempo :** $140 - 145\text{ BPM}$.
*   **Harmonie :** Similaire à la Dark Trap (Mineure harmonique, Phrygien), souvent accompagnée de nappes vocales ou de cordes dramatiques.
*   **Spécificités Rythmiques (Crucial pour l'agent) :**
    *   *Placement de la Snare :* Au lieu de frapper sur les temps 3 de chaque mesure, la snare frappe sur le **temps 3 de la première mesure**, puis sur le **temps 4 (ou le "et" du 3) de la deuxième mesure**. Ce décalage crée le rebond caractéristique de la Drill.
    *   *Hi-Hats :* Motif syncopé en groupes de 3 ($3 + 3 + 2$ sur 8 pas), jamais de two-step continu.
    *   *808 :* Glissandos (slides) extrêmes d'une ou deux octaves en fin de mesure.

---

## 5. Protocole d'Exécution pour l'Agent Autonome (Algorithme de Composition)

Pour générer une instrumentale **planante et rythmée** cohérente, l'agent doit exécuter les étapes suivantes dans l'ordre :

1.  **Initialisation des Paramètres Globaux :**
    *   Choisir un tempo entre $125\text{ BPM}$ et $138\text{ BPM}$ (zone optimale pour l'équilibre planant/rythmé).
    *   Sélectionner une tonalité mineure (ex: $Fa\#$ mineur ou $Do$ mineur, zones où les fondamentales de 808 se situent idéalement entre $40\text{ Hz}$ et $55\text{ Hz}$, le *sweet spot* des caissons de basses).
2.  **Génération de la Base Harmonique (L'Atmosphère) :**
    *   Créer une progression de 4 accords en mode Éolien ou Dorien utilisant des accords de 7ème ou 9ème ($i9 \rightarrow VI_{maj7} \rightarrow iv7 \rightarrow v7$).
    *   Appliquer un voile fréquentiel (Low-Pass Filter à $2.5\text{ kHz}$) et une réverbération large (Decay $> 2.5\text{ s}$, Mix $40\%$) sur un Pad ou un Piano électrique.
3.  **Création du Contrepoint Mélodique (L'Accroche) :**
    *   Générer un arpège ou un motif court de 4 à 6 notes dans le registre aigu ($C5 - C6$) avec un timbre cristallin (Bell, Pluck ou Flûte) et un Delay synchronisé au tempo ($1/4$ ou $1/8$ pointé).
4.  **Construction de la Fondation Rythmique (Drums) :**
    *   Placer la **Snare/Clap** sur le temps 3 de chaque mesure (en grille demi-temps).
    *   Générer une piste de **Hi-Hats** en $1/8$, puis injecter des variations de vélocité (alternance fort/faible à $100\% / 65\%$) et insérer des rolls ($1/16T$ ou $1/32$) uniquement sur les temps faibles précédant les snares ou les changements d'accords.
5.  **Génération et Accordage de la 808 :**
    *   Extraire les notes fondamentales des 4 accords générés à l'étape 2.
    *   Placer les notes de 808 dans l'octave $1$ ou $2$ ($C1 - B1$, soit $32\text{ Hz} - 61\text{ Hz}$).
    *   Synchroniser le rythme de la 808 avec des silences stratégiques : laisser respirer la fin de la mesure 2 et la fin de la mesure 4 pour amplifier l'effet planant des réverbérations.
6.  **Mixage Automatisé (Équilibre Spectral) :**
    *   Appliquer un filtre coupe-bas (High-Pass) à $130\text{ Hz}$ sur toutes les pistes mélodiques et harmoniques.
    *   Appliquer un *Soft Clipper* sur le bus Master pour contrôler les transitoires des drums et donner de la densité à la 808 sans écrêtage numérique désagréable.