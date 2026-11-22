# Spécifications de Conception et d'Arrangement Musical
**Document de référence pour Agent de Développement Musical Autonome**

Ce document définit les contraintes physiques, psychoacoustiques et structurelles régissant la composition, l'arrangement et le mixage d'un morceau. L'agent doit traiter ces règles comme des contraintes d'optimisation pour éviter la saturation spectrale, la confusion cognitive et la monotonie structurelle.

---

## 1. Loi de l'Inertie Fréquentielle (Relation Hauteur / Durée)

### Règle
La durée minimale d'une note et sa vitesse de répétition doivent être inversement proportionnelles à sa fréquence fondamentale. Plus un son est grave, plus ses notes doivent être longues et espacées ; plus il est aigu, plus ses subdivisions peuvent être courtes et rapides.

### Justification Physique et Acoustique
* **Longueur d'onde et temps d'établissement :** Une onde grave (ex. $40\text{ Hz}$) a une longueur d'onde d'environ $8,5\text{ mètres}$ et nécessite $25\text{ ms}$ pour accomplir un seul cycle complet. Il faut plusieurs cycles pour que l'oreille identifie une hauteur tonale précise. Des notes trop courtes dans le grave ne produisent qu'un clic percussif sans note définie.
* **Masquage temporel et résonance :** Les basses fréquences possèdent une énergie cinétique élevée et un temps de déclin naturel lent (traînée acoustique). Si deux notes graves se succèdent trop rapidement, leurs queues de résonance se chevauchent, créant des interférences de phase et un brouillard sonore (*mud*).
* **Précision transitoire des aigus :** Une onde aiguë (ex. $8\text{ kHz}$) accomplit un cycle en $0,125\text{ ms}$. L'attaque et le relâchement sont quasi instantanés, permettant au système auditif de distinguer des micro-événements rythmiques très rapprochés (les "tics", hi-hats, shakers).

### Matrice de Contraintes Rythmiques par Registre

| Registre | Bande Fréquentielle | Valeurs Rythmiques Recommandées | Enveloppe ADSR Type | Exemples d'Instruments |
| :--- | :--- | :--- | :--- | :--- |
| **Sub-Basse** | $20\text{ Hz} - 60\text{ Hz}$ | Rondes, Blanches | Attaque lente/moyenne, Sustain élevé, Release contrôlé | Sine Sub, 808 tenue |
| **Basse** | $60\text{ Hz} - 250\text{ Hz}$ | Blanches, Noires, Croches pointées | Attaque nette, Decay moyen | Kick, Basse électrique, Synth Bass |
| **Bas-Médiums** | $250\text{ Hz} - 800\text{ Hz}$ | Noires, Croches | Sustain moyen (accords) ou court (rythmique) | Pads, Piano (main gauche), Guitare rythmique |
| **Hauts-Médiums** | $800\text{ Hz} - 4\text{ kHz}$ | Croches, Double-croches | Attaque précise, articulation dynamique | Voix principale, Lead synthé, Snare, Cuivres |
| **Aigus / Air** | $4\text{ kHz} - 20\text{ kHz}$ | Double-croches, Triple-croches, Triolets | Attaque instantanée, Decay et Release très courts | Hi-hats, Shakers, Cymbales, Percussions fines |

> **Exception algorithmique (Staccato grave) :** Si l'agent génère une ligne de basse rapide (ex. double-croches en Funk ou roulements de 808 en Trap), il doit impérativement réduire le paramètre *Release* (ou appliquer un *Gate*) afin que chaque note s'interrompe totalement avant le déclenchement de la suivante.

---

## 2. Référentiel Mathématique des Durées (Mesures et BPM)

### Règle de Conversion Temporelle
En signature rythmique standard ($4/4$), **1 mesure contient 4 temps** (4 noires). La durée absolue en millisecondes ($\text{ms}$) d'une noire dépend exclusivement du tempo ($\text{BPM}$, battements par minute) selon l'équation :

$$T_{\text{noire (ms)}} = \frac{60\,000}{\text{BPM}}$$

Pour toute figure rythmique de fraction $F$ d'une mesure (ex. $F = \frac{1}{16}$ pour une double-croche) :

$$T_{\text{note (ms)}} = \left( \frac{60\,000}{\text{BPM}} \times 4 \right) \times F$$

* **Modificateur Pointé (`.`) :** Multiplie la durée de base par $1,5$ (ajoute la moitié de sa valeur).
* **Modificateur Triolet (`T`) :** Multiplie la durée de base par $\frac{2}{3}$ (3 notes jouées dans l'espace de 2).

### Table de Correspondance des Figures Rythmiques (Signature $4/4$)

| Figure Rythmique | Notation | Fraction de Mesure | Nombre de Temps | Durée à $90\text{ BPM}$ | Durée à $120\text{ BPM}$ | Durée à $140\text{ BPM}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ronde** | `1/1` | $1$ mesure | $4\text{ temps}$ | $2667\text{ ms}$ | $2000\text{ ms}$ | $1714\text{ ms}$ |
| **Blanche** | `1/2` | $1/2$ mesure | $2\text{ temps}$ | $1333\text{ ms}$ | $1000\text{ ms}$ | $857\text{ ms}$ |
| **Noire** | `1/4` | $1/4$ mesure | $1\text{ temps}$ | $667\text{ ms}$ | $500\text{ ms}$ | $429\text{ ms}$ |
| **Croche** | `1/8` | $1/8$ mesure | $0,5\text{ temps}$ | $333\text{ ms}$ | $250\text{ ms}$ | $214\text{ ms}$ |
| **Triolet de croche** | `1/8T` | $1/12$ mesure | $0,33\text{ temps}$ | $222\text{ ms}$ | $167\text{ ms}$ | $143\text{ ms}$ |
| **Double-croche** | `1/16` | $1/16$ mesure | $0,25\text{ temps}$ | $167\text{ ms}$ | $125\text{ ms}$ | $107\text{ ms}$ |
| **Triolet de double-croche** | `1/16T` | $1/24$ mesure | $0,167\text{ temps}$ | $111\text{ ms}$ | $83\text{ ms}$ | $71\text{ ms}$ |
| **Triple-croche** ("Tics") | `1/32` | $1/32$ mesure | $0,125\text{ temps}$ | $83\text{ ms}$ | $62,5\text{ ms}$ | $53,5\text{ ms}$ |

> **Contrainte physique absolue :** À $120\text{ BPM}$, une triple-croche dure $62,5\text{ ms}$. Une fréquence de $40\text{ Hz}$ (période de $25\text{ ms}$) n'a le temps d'effectuer que $2,5$ cycles complets durant cette note, ce qui est insuffisant pour être perçu comme une note tonale stable. L'agent doit donc interdire toute valeur $\le 1/16$ sous $100\text{ Hz}$ pour les sons maintenus.

---

## 3. Plafond Cognitif et Allocation des Rôles Fonctionnels

### Règle
L'agent ne doit jamais activer simultanément plus de **5 à 8 pistes** (ou groupes de pistes jouant à l'unisson). Ces pistes doivent être allouées selon 5 rôles fonctionnels exclusifs, avec un maximum de **3 à 5 foyers d'attention** perceptibles au même instant $t$.

### Justification Psychoacoustique
* **Limite de la mémoire de travail auditive :** Le cortex auditif humain ne peut isoler et suivre simultanément que 3 à 5 flux sonores distincts (principe de *l'analyse de scène auditive* de Bregman). Au-delà, le cerveau fusionne les sources secondaires en un bruit de fond indistinct.
* **Masquage fréquentiel simultané :** Deux instruments jouant dans la même plage de fréquences au même moment s'annulent ou se masquent mutuellement. L'oreille ne perçoit que le plus fort, rendant l'autre inutile et consommateur de marge dynamique (*headroom*).

### Matrice d'Allocation des Pistes (Quotas Simultanés)

| Catégorie Fonctionnelle | Quota Max | Occupation Spectrale Prioritaire | Règle d'Exclusion et de Priorité |
| :--- | :--- | :--- | :--- |
| **1. Fondation (Low-End)** | **1 à 2** | $20\text{ Hz} - 200\text{ Hz}$ | **Règle d'exclusivité :** Le Kick et la Basse ne doivent pas frapper à pleine puissance au même instant $t$ (appliquer une compression *Sidechain* ou un décalage rythmique). |
| **2. Rythmique (Groove)** | **2 à 4** | $150\text{ Hz} - 16\text{ kHz}$ | **Règle d'imbrication :** Les éléments percussifs doivent occuper des cases temporelles complémentaires (ex. Kick sur temps 1/3, Snare sur temps 2/4, Hi-hats sur les contretemps). |
| **3. Harmonie (Tapis)** | **1 à 2** | $200\text{ Hz} - 3\text{ kHz}$ | **Règle de discrétion :** Mouvements mélodiques lents. Doit présenter un creux fréquentiel (EQ) autour de $1\text{ kHz} - 3\text{ kHz}$ pour ne pas concurrencer le Lead. |
| **4. Élément Principal (Lead)** | **1** (max 2) | $500\text{ Hz} - 5\text{ kHz}$ | **Règle de suprématie :** Élément le plus fort et le plus détaillé du mix. Si 2 Leads coexistent, ils doivent fonctionner en *question-réponse* (alternance temporelle), jamais en superposition continue. |
| **5. Habillage (Ear Candy)** | **1 à 2** | Variable (souvent filtré) | **Règle d'intermittence :** Présence inférieure à $15\%$ de la durée totale du morceau. Utilisé uniquement pour marquer les transitions ou créer de la surprise. |

---

## 4. Architecture Temporelle (La Grille des Multiples de 4)

### Règle
La macro-structure et la micro-structure du morceau doivent être construites sur une grille binaire basée sur des puissances de 2 (4, 8, 16, 32 mesures). L'état du système (nombre de pistes actives ou variation des motifs) doit évoluer à chaque franchissement de palier.

### Justification Cognitive
* **Prédictibilité et récompense :** Le cerveau humain anticipe les cycles musicaux. Une structure régulière (multiples de 4 mesures) permet à l'auditeur de prédire la pulsation, tandis que l'introduction d'une variation exactement à la fin d'un cycle (mesure 4, 8 ou 16) libère de la dopamine (effet de résolution ou de surprise contrôlée).
* **Prévention de l'habituation auditive :** Si une boucle tourne à l'identique pendant plus de 8 mesures sans ajout, retrait ou variation, le cerveau filtre le signal comme un bruit environnemental constant, provoquant l'ennui.

### Règles de Transition par Palier

1. **Palier Micro (Toutes les 4 mesures) :**
   * *Action :* Variation rythmique mineure sans ajout de nouvelle piste.
   * *Exemples :* Roulement de hi-hat sur le 4e temps de la mesure 4, note de passage sur la basse, coupure (*mute*) du Kick sur le dernier demi-temps.
2. **Palier Méso (Toutes les 8 mesures) :**
   * *Action :* Modification de la densité instrumentale (ajout ou retrait d'une piste de catégorie 2, 3 ou 5).
   * *Exemples :* Entrée des hi-hats à la mesure 9, ajout d'un pad harmonique à la mesure 17.
3. **Palier Macro (Toutes les 16 ou 32 mesures) :**
   * *Action :* Changement de section structurelle (Couplet $\rightarrow$ Pré-refrain $\rightarrow$ Refrain $\rightarrow$ Pont) accompagné d'un événement de transition (Riser, Crash, Impact) et d'un changement net de dynamique globale.

---

## 5. Spatialisation 3D (Panoramique, Profondeur et Hauteur)

### Règle
L'agent doit répartir les sources sonores dans un espace tridimensionnel virtuel afin d'éviter l'accumulation d'énergie au centre de l'image stéréo.

### Justification Physique
* **Directivité des fréquences :** Les basses fréquences sont omnidirectionnelles et nécessitent une somme égale sur les deux haut-parleurs pour délivrer un maximum de pression acoustique (d'où le placement en Mono strict). Les hautes fréquences sont très directives et permettent une localisation stéréo précise.
* **Indice de distance acoustique :** Dans la nature, un son lointain perd ses fréquences aiguës (absorbées par l'air), perd son attaque transitoire et présente un ratio élevé entre les réflexions (réverbération) et le son direct.

### Matrice de Placement Spatial

| Axe Spatial | Paramètre Technique | Règle d'Application par Catégorie |
| :--- | :--- | :--- |
| **Largeur (X)**<br>*Gauche / Droite* | **Panoramique (Pan)** | • **Centre strict (0%) :** Kick, Sub/Basse, Snare, Voix Lead.<br>• **Légèrement décalé ($\pm 15\%$ à $35\%$) :** Hi-hats, Percussions secondaires, Arpèges.<br>• **Large ($\pm 50\%$ à $100\%$) :** Pads stéréo, Guitares doublées, Cymbales, Effets Ear Candy. |
| **Profondeur (Y)**<br>*Avant / Arrière* | **Volume + Réverbération + Filtre Passe-Bas** | • **Premier plan (Sec, fort, brillant) :** Lead, Kick, Snare.<br>• **Second plan (Réverbération courte, -3 à -6 dB) :** Rythmique secondaire, Contre-chants.<br>• **Arrière-plan (Réverbération longue, aigus atténués) :** Pads, Nappes, Ambiances. |
| **Hauteur (Z)**<br>*Bas / Haut* | **Égalisation (Fréquences)** | • **Sol (Base) :** Sub et Basse.<br>• **Milieu :** Corps des instruments et Voix.<br>• **Plafond :** Air ($> 10\text{ kHz}$), Cymbales, brillance des synthés. |

---

## 6. Synthèse des Règles de Validation (Checklist pour l'Agent)

Avant de valider le rendu d'une section musicale, l'agent doit vérifier les assertions suivantes :

* `[ ]` **Vérification Spectrale :** Aucune fréquence inférieure à $120\text{ Hz}$ n'est panoramiquée à gauche ou à droite (100% Mono sous $120\text{ Hz}$).
* `[ ]` **Vérification Rythmique :** Aucune note inférieure à $100\text{ Hz}$ n'a une durée inférieure à une croche (sauf si le paramètre *Release* est $< 50\text{ ms}$).
* `[ ]` **Vérification de Densité :** Le nombre de pistes actives simultanément à tout instant $t$ est compris entre $1$ (Intro/Break) et $8$ (Refrain/Climax).
* `[ ]` **Vérification de Hiérarchie :** Un seul élément occupe le rôle de *Lead* à un instant $t$, et son niveau crête/RMS domine les éléments harmoniques d'au moins $3\text{ dB}$.
* `[ ]` **Vérification Structurelle :** Aucune boucle de 8 mesures n'est répétée à l'identique sans qu'au moins un paramètre (ajout/retrait de piste ou variation rythmique) ne soit modifié.