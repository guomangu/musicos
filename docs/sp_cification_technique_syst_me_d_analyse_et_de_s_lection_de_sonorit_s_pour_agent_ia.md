# Spécification Technique : Module d'Analyse Psychoacoustique et de Sélection de Samples pour Agent IA Musical

## 1. Objectif et Contexte Système

Ce document définit l'architecture complète d'un pipeline logiciel permettant à un agent textuel autonome d'évaluer objectivement la **qualité perceptive (confort auditif)**, la **qualité esthétique (grain)** et l'**adéquation fonctionnelle** de samples audio avant de les intégrer dans un morceau.

### Architecture en entonnoir (Pipeline)
L'agent doit implémenter et orchestrer 4 étapes séquentielles :
1. **Pré-indexation (Offline)** : Extraction unique des caractéristiques DSP et des embeddings audio pour toute la banque de samples, stockée dans une base JSON ou SQLite.
2. **Filtre Dur (Hard Filter)** : Élimination immédiate des fichiers présentant des défauts techniques bloquants (clipping, clics, phase inversée, durée invalide).
3. **Scoring Multicritères** : Calcul d'une note composite combinant confort psychoacoustique, adéquation au rôle de la piste et proximité esthétique (CLAP / *Golden References*).
4. **Vérification Contextuelle (Mix-Aware)** : Calcul du masquage fréquentiel et de la compatibilité harmonique vis-à-vis des pistes déjà présentes dans le morceau.

---

## 2. Environnement Technique et Dépendances

L'agent doit générer des scripts compatibles avec **Python 3.10+**.

### Dépendances requises (`requirements.txt`)
* `numpy` : Calcul matriciel et opérations vectorielles.
* `scipy` : Traitement du signal (filtres, FFT, enveloppes, corrélation).
* `librosa` : Extraction de caractéristiques audio (STFT, centroïde, HPSS, chroma, onset).
* `soundfile` : Lecture rapide et fiable des fichiers `.wav`, `.flac`, `.ogg`.
* `torch` et `transformers` (ou `laion-clap`) : Inférence locale du modèle d'embedding audio **LAION-CLAP**.
* `scikit-learn` : Calcul des similarités cosinus et normalisation.

### Normalisation d'entrée obligatoire
Avant toute analyse DSP, chaque fichier audio doit être chargé avec les règles suivantes :
* Conserver le signal stéréo brut $y_{\text{stereo}} \in \mathbb{R}^{2 \times N}$ pour l'analyse de corrélation de phase et de clipping.
* Générer une version mono $y_{\text{mono}} = \frac{1}{2}(y_L + y_R)$ rééchantillonnée à $f_s = 44100\text{ Hz}$ pour l'analyse spectrale.
* Retirer la composante continue (DC offset) : $y_{\text{mono}} \leftarrow y_{\text{mono}} - \text{mean}(y_{\text{mono}})$.
* Pour les métriques spectrales (hors clipping), normaliser le pic à $-1\text{ dBFS}$ afin de rendre les comparaisons indépendantes du volume d'enregistrement.

---

## 3. Couche 1 : Analyse Psychoacoustique et DSP

### 3.1. Défauts bloquants (Hard Rejects)
Un sample est immédiatement marqué `rejected: true` si l'une des conditions suivantes est vérifiée :

1. **Clipping numérique sévère** :
   Compter le nombre maximal d'échantillons consécutifs $C_{\max}$ tels que $|y[n]| \ge 0.99$.
   * Condition de rejet : $C_{\max} \ge 3$ ou proportion totale d'échantillons saturés $> 0.1\%$.
2. **Clics d'enveloppe (Non-Zero Crossing)** :
   Mesurer l'amplitude absolue aux bornes du fichier (les 5 premiers et 5 derniers échantillons).
   * Condition de rejet (ou flag `needs_fade: true` si l'agent sait appliquer un fade-in/out de $5\text{ ms}$) : $\max(|y[0:5]|) > 0.05$ ou $\max(|y[-5:]|) > 0.05$.
3. **Silence ou signal trop faible** :
   * Niveau RMS global $<-55\text{ dBFS}$ ou durée effective $< 0.03\text{ s}$.
4. **Corrélation de phase stéréo ($\rho_{LR}$)** :
   Pour les fichiers stéréo, calculer le coefficient de corrélation de Pearson entre le canal gauche $y_L$ et le canal droit $y_R$ :
   $$\rho_{LR} = \frac{\sum_{n} y_L[n] y_R[n]}{\sqrt{\sum_{n} y_L[n]^2 \sum_{n} y_R[n]^2} + \epsilon}$$
   * Condition de rejet global : $\rho_{LR} < -0.2$ (hors phase destructive).
   * Condition de rejet spécifique pour les rôles `kick` et `bass_sub` : $\rho_{LR} < 0.7$.

### 3.2. Métriques de Confort Auditif
Calculer le spectrogramme de puissance $P(f, t) = |\text{STFT}(y_{\text{mono}})|^2$ avec `n_fft = 2048` et `hop_length = 512`.

1. **Indice de Dureté / Fatigue Auditive ($R_{\text{harsh}}$)** :
   Mesure la concentration d'énergie dans la zone de sensibilité maximale de l'oreille humaine ($2000\text{ Hz}$ à $5000\text{ Hz}$) par rapport à l'énergie spectrale totale :
   $$R_{\text{harsh}} = \frac{\sum_{t} \sum_{f=2000}^{5000} P(f, t)}{\sum_{t} \sum_{f=20}^{20000} P(f, t) + \epsilon}$$
   * **Interprétation** : Un $R_{\text{harsh}} > 0.35$ sur un son continu (pad, lead, basse) indique une forte probabilité d'agressivité.
   * **Détection de pic résonant ($P_{\text{peak\_harsh}}$)** : Calculer le ratio entre le maximum spectral dans $[2000, 5000]\text{ Hz}$ et la médiane spectrale de cette même bande (converti en dB). Si ce pic dépasse $+12\text{ dB}$, le son possède une résonance sifflante désagréable.

2. **Rugosité Spectrale / Dissonance Sensorielle ($D_{\text{rough}}$)** :
   Estimer la rugosité due aux battements de fréquences proches (modèle de Plomp-Levelt simplifié sur les pics spectraux dominants $f_i, f_j$ d'amplitudes $a_i, a_j$ dans chaque trame) :
   $$D_{\text{rough}} = \sum_{i < j} a_i a_j \cdot \left( e^{-3.5 s |f_j - f_i|} - e^{-5.75 s |f_j - f_i|} \right)$$
   où $s = \frac{0.24}{0.021 \min(f_i, f_j) + 19}$.
   * Normaliser $D_{\text{rough}}$ par la somme des énergies $\sum a_i^2$. Une valeur élevée indique un timbre criard, métallique ou mal accordé.

3. **Pureté Harmonique vs Bruit** :
   * **Spectral Flatness (SFM)** : Moyenne géométrique sur moyenne arithmétique du spectre de puissance (`librosa.feature.spectral_flatness`). Proche de $0$ = son tonal pur ; proche de $1$ = bruit blanc.
   * **Ratio Harmonique / Percussif ($R_{\text{harm}}$)** : Via séparation HPSS (`librosa.effects.hpss`), calculer :
     $$R_{\text{harm}} = \frac{\|y_{\text{harm}}\|_2^2}{\|y_{\text{harm}}\|_2^2 + \|y_{\text{perc}}\|_2^2 + \epsilon}$$

4. **Dynamique et Enveloppe Temporelle** :
   * **Facteur de crête (*Crest Factor*)** en dB :
     $$\text{CF}_{\text{dB}} = 20 \log_{10}\left( \frac{\max(|y_{\text{mono}}|)}{\text{RMS}(y_{\text{mono}}) + \epsilon} \right)$$
   * **Temps d'attaque ($T_{\text{attack}}$)** : Temps en millisecondes entre le début effectif du son (premier franchissement de $-40\text{ dB}$ du pic) et le pic d'enveloppe RMS.
   * **Spectral Flux ($\Phi_{\text{spec}}$)** : Moyenne de la norme $L_2$ de la différence entre trames spectrales consécutives. Mesure la stabilité du timbre dans le temps.

5. **Répartition Fréquentielle par Bandes** :
   Calculer le pourcentage d'énergie dans 6 bandes clés :
   * `E_sub` : $[20, 60]\text{ Hz}$
   * `E_bass` : $[60, 250]\text{ Hz}$
   * `E_lowmid` : $[250, 800]\text{ Hz}$
   * `E_mid` : $[800, 2000]\text{ Hz}$
   * `E_harsh` : $[2000, 5000]\text{ Hz}$
   * `E_air` : $[5000, 20000]\text{ Hz}$

### 3.3. Calcul du Score de Confort Auditif ($S_{\text{comfort}} \in [0, 1]$)
Partir d'un score de base de $1.0$ et appliquer des pénalités douces (sigmoïdes ou linéaires bornées) :
* Pénalité de dureté si $R_{\text{harsh}} > \tau_{\text{role}}$ (seuil dépendant du rôle).
* Pénalité de pic résonant si $P_{\text{peak\_harsh}} > 10\text{ dB}$.
* Pénalité de rugosité excessive (hors percussions métalliques).
* Pénalité si présence de clics nécessitant un fade.

---

## 4. Couche 2 : Profils Cibles par Rôle de Piste

Pour calculer le score d'adéquation au rôle $S_{\text{role}} \in [0, 1]$, comparer les métriques extraites aux plages idéales ci-dessous. Pour chaque critère, attribuer $1.0$ si la valeur est dans la plage idéale, et faire décroître linéairement vers $0.0$ à mesure qu'elle s'en éloigne.

| Rôle (`role_id`) | Énergie Dominante | $T_{\text{attack}}$ (ms) | $\text{CF}_{\text{dB}}$ (dB) | $R_{\text{harm}}$ | SFM | Contraintes Spécifiques |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `kick` | `E_sub` + `E_bass` $> 0.65$ | $[0, 15]$ | $[10, 22]$ | $[0.1, 0.5]$ | $< 0.15$ | $\rho_{LR} \ge 0.85$, durée $< 1.0\text{ s}$ |
| `bass_sub` | `E_sub` + `E_bass` $> 0.70$ | $[5, 80]$ | $[3, 12]$ | $> 0.75$ | $< 0.05$ | $\rho_{LR} \ge 0.80$, $R_{\text{harsh}} < 0.10$ |
| `snare_clap` | `E_lowmid` + `E_mid` + `E_harsh` $> 0.60$ | $[0, 20]$ | $[12, 24]$ | $< 0.45$ | $[0.05, 0.45]$ | `E_sub` $< 0.05$, durée $< 1.2\text{ s}$ |
| `hihat_perc` | `E_air` + `E_harsh` $> 0.75$ | $[0, 15]$ | $[10, 25]$ | $< 0.30$ | $> 0.15$ | `E_sub` + `E_bass` $< 0.05$ |
| `lead_melody` | `E_lowmid` + `E_mid` + `E_harsh` $> 0.65$ | $[5, 100]$ | $[4, 14]$ | $> 0.70$ | $< 0.08$ | $R_{\text{harsh}} < 0.32$, $P_{\text{peak\_harsh}} < 12\text{ dB}$ |
| `pad_atmosphere` | `E_lowmid` + `E_mid` $> 0.50$ | $> 120$ | $[3, 10]$ | $> 0.80$ | $< 0.10$ | $\Phi_{\text{spec}}$ faible, `E_sub` $< 0.10$, $R_{\text{harsh}} < 0.25$ |
| `chord_keys` | `E_bass` + `E_lowmid` + `E_mid` $> 0.70$ | $[5, 60]$ | $[6, 16]$ | $> 0.75$ | $< 0.08$ | $R_{\text{harsh}} < 0.28$ |

---

## 5. Couche 3 : Évaluation Esthétique via Embeddings Audio (LAION-CLAP)

Le DSP seul ne distingue pas un timbre « riche et pro » d'un son « plastique / bas de gamme ». L'agent doit utiliser le modèle local **LAION-CLAP** (ex: `laion/larger_clap_music` via HuggingFace `transformers`).

### 5.1. Extraction de l'Embedding
Pour chaque sample, extraire le vecteur normalisé $v_s \in \mathbb{R}^{512}$ tel que $\|v_s\|_2 = 1$.

### 5.2. Méthode A : Scoring Sémantique Zero-Shot ($S_{\text{zs}}$)
Pré-calculer les centroïdes vectoriels de deux ensembles de prompts textuels :
* **Prompts Positifs ($V_{\text{pos}}$)** :
  * `"warm analog studio quality sound, rich harmonics, clean professional mix"`
  * `"pleasant lush musical timbre, high fidelity audio sample"`
  * `"polished modern music production instrument, well balanced tone"`
* **Prompts Négatifs ($V_{\text{neg}}$)** :
  * `"cheap plastic midi sound, amateur harsh digital distortion"`
  * `"unpleasant piercing resonant noise, muddy low quality audio"`
  * `"tinny irritating synthetic beep, poor recording quality"`

Calculer la similarité moyenne positive $\text{sim}_{\text{pos}} = \text{mean}_{p \in V_{\text{pos}}}(v_s \cdot p)$ et négative $\text{sim}_{\text{neg}} = \text{mean}_{n \in V_{\text{neg}}}(v_s \cdot n)$.
Convertir en score $S_{\text{zs}} \in [0, 1]$ via une fonction logistique (température $T = 0.05$) :
$$S_{\text{zs}} = \frac{1}{1 + \exp\left( -\frac{\text{sim}_{\text{pos}} - \text{sim}_{\text{neg}}}{T} \right)}$$

### 5.3. Méthode B : Proximité aux *Golden References* ($S_{\text{ref}}$)
Maintenir un répertoire `golden_references/<role_id>/` contenant 5 à 15 samples validés par l'humain comme étant excellents pour ce rôle.
* Pour un rôle donné, stocker les vecteurs de référence $\{r_1, \dots, r_K\}$.
* Pour un sample candidat $v_s$, calculer la similarité cosinus avec le top-3 le plus proche (plutôt que la moyenne globale, pour autoriser plusieurs sous-styles valides au sein d'un même rôle) :
  $$S_{\text{ref}} = \text{clip}\left( \frac{\text{mean}(\text{top3}(\{v_s \cdot r_k\})) - 0.3}{0.6}, 0, 1 \right)$$

### 5.4. Score Esthétique Combiné ($S_{\text{aesth}}$)
* Si des *Golden References* existent pour le rôle : $S_{\text{aesth}} = 0.7 S_{\text{ref}} + 0.3 S_{\text{zs}}$.
* Sinon : $S_{\text{aesth}} = S_{\text{zs}}$.

---

## 6. Couche 4 : Vérification Contextuelle (Masquage et Tonalité)

Avant de valider un sample pour un morceau en cours, l'agent doit comparer le candidat au mix actuel (export audio des pistes déjà validées $y_{\text{mix}}$).

### 6.1. Calcul du Masquage Fréquentiel ($P_{\text{mask}} \in [0, 1]$)
1. Calculer le spectre moyen normalisé en énergie sur 24 bandes critiques (échelle Bark ou bandes tiers d'octave entre $20\text{ Hz}$ et $20\text{ kHz}$) pour le mix existant $B_{\text{mix}} \in \mathbb{R}^{24}$ et pour le sample candidat $B_{\text{cand}} \in \mathbb{R}^{24}$, tels que $\sum B_i = 1$.
2. Calculer le chevauchement spectral par similarité cosinus (ou produit scalaire de Bhattacharyya) :
   $$P_{\text{mask}} = \sum_{i=1}^{24} \sqrt{B_{\text{mix}}[i] \cdot B_{\text{cand}}[i]}$$
3. **Exception par rôle** : Pour les pistes rythmiques très courtes (`kick`, `snare_clap`, `hihat_perc`), le masquage continu est moins critique que pour les pistes tenues (`lead_melody`, `pad_atmosphere`, `bass_sub`). L'agent doit réduire le poids du masquage pour les percussions.

### 6.2. Compatibilité Tonale (Pour les rôles mélodiques/harmoniques)
Pour `bass_sub`, `lead_melody`, `pad_atmosphere`, `chord_keys` :
* Extraire la fréquence fondamentale médiane $f_0$ via `librosa.yin` ou le profil chromatique (`librosa.feature.chroma_cqt`).
* Identifier la note dominante (ex: `C`, `F#`) et stocker `detected_pitch` dans l'index.
* Si le morceau est dans une tonalité cible, l'agent sait exactement de combien de demi-tons transposer le sample (en évitant les transpositions extrêmes $> \pm 5$ demi-tons qui dégradent le timbre).

---

## 7. Formule de Décision Finale

Pour chaque sample candidat ayant passé le **Hard Filter**, calculer le score final :

$$S_{\text{final}} = w_1 S_{\text{comfort}} + w_2 S_{\text{role}} + w_3 S_{\text{aesth}} - w_4 P_{\text{mask}}$$

### Pondérations recommandées par défaut
* **Pistes mélodiques / harmoniques** (`lead_melody`, `pad_atmosphere`, `bass_sub`, `chord_keys`) :
  $w_1 = 0.25$, $w_2 = 0.30$, $w_3 = 0.45$, $w_4 = 0.20$.
* **Pistes percussives** (`kick`, `snare_clap`, `hihat_perc`) :
  $w_1 = 0.20$, $w_2 = 0.40$, $w_3 = 0.40$, $w_4 = 0.05$.

### Seuil d'acceptation
* $S_{\text{final}} \ge 0.70$ : **Excellent**, sélection directe.
* $0.55 \le S_{\text{final}} < 0.70$ : **Acceptable**, à utiliser si aucun meilleur candidat n'est disponible (un léger EQ peut être requis).
* $S_{\text{final}} < 0.55$ : **Rejeté**, l'agent doit chercher un autre sample ou demander une régénération.

---

## 8. Spécifications des Scripts à Générer par l'Agent

L'agent doit créer une suite de **3 scripts Python modulaires** partageant un fichier utilitaire `audio_metrics.py`.

### 8.1. `audio_metrics.py` (Bibliothèque partagée)
* Fonctions pures pour le chargement audio, le calcul des défauts bloquants, l'extraction de toutes les métriques DSP de la Section 3, le calcul des bandes Bark, la détection de pitch, et le wrapper d'inférence LAION-CLAP.

### 8.2. `index_samples.py` (Pré-indexation)
* **CLI** : `python index_samples.py --input-dir ./samples --output-db ./sample_index.json --device cuda` (fallback automatique sur `cpu`).
* **Comportement** : Parcourt récursivement `--input-dir`, ignore les fichiers déjà indexés (vérification par hash MD5 ou timestamp), calcule DSP + embedding CLAP, sauvegarde dans `--output-db` (les vecteurs CLAP peuvent être sauvegardés dans un fichier compagnon `.npz` pour garder le JSON léger et lisible).

### 8.3. `build_references.py` (Gestion des Golden References)
* **CLI** : `python build_references.py --ref-dir ./golden_references --output-refs ./golden_embeddings.npz`
* **Comportement** : Scanne les sous-dossiers par `role_id` et sauvegarde les matrices d'embeddings CLAP correspondantes.

### 8.4. `select_sample.py` (Moteur de requête pour l'Agent)
* **CLI** :
  ```bash
  python select_sample.py \
    --db ./sample_index.json \
    --refs ./golden_embeddings.npz \
    --role lead_melody \
    --mix-path ./current_mix.wav \
    --target-key "C minor" \
    --top-k 5 \
    --format json
  ```
* **Comportement** : Applique le Hard Filter, calcule les scores $S_{\text{comfort}}$, $S_{\text{role}}$, $S_{\text{aesth}}$, $P_{\text{mask}}$ et $S_{\text{final}}$, trie les résultats et imprime un JSON propre sur `stdout`.

---

## 9. Schémas de Données (JSON)

### 9.1. Entrée dans `sample_index.json`
```json
{
  "id": "a1b2c3d4",
  "filepath": "/samples/synths/warm_analog_lead_C4.wav",
  "duration_sec": 3.45,
  "sample_rate": 44100,
  "hard_reject": false,
  "reject_reasons": [],
  "needs_fade": false,
  "dsp": {
    "phase_corr": 0.92,
    "crest_factor_db": 8.4,
    "attack_ms": 22.5,
    "r_harsh": 0.19,
    "peak_harsh_db": 6.1,
    "roughness": 0.08,
    "sfm": 0.03,
    "r_harm": 0.88,
    "spectral_flux": 0.12,
    "bands": {
      "sub": 0.01,
      "bass": 0.14,
      "lowmid": 0.35,
      "mid": 0.26,
      "harsh": 0.19,
      "air": 0.05
    },
    "bark_24": [0.002, 0.008, "..."],
    "detected_note": "C",
    "detected_f0_hz": 261.63
  },
  "clap_index": 42
}
```

### 9.2. Sortie CLI de `select_sample.py`
```json
{
  "query": {
    "role": "lead_melody",
    "mix_evaluated": true,
    "total_candidates": 1240,
    "passed_hard_filter": 980
  },
  "recommendations": [
    {
      "rank": 1,
      "filepath": "/samples/synths/warm_analog_lead_C4.wav",
      "final_score": 0.84,
      "scores": {
        "comfort": 0.91,
        "role_fit": 0.88,
        "aesthetic": 0.85,
        "masking_penalty": 0.14
      },
      "pitch_info": {
        "detected_note": "C",
        "recommended_semitone_shift": 0
      },
      "warnings": []
    }
  ]
}
```

---

## 10. Protocole Opérationnel pour l'Agent Autonome

L'agent textuel doit suivre strictement cette boucle de travail :

1. **Initialisation** :
   * Vérifier l'existence de `sample_index.json`. S'il est absent ou obsolète, exécuter `index_samples.py`.
   * Vérifier l'existence de `golden_embeddings.npz`. S'il est absent mais que des fichiers existent dans `golden_references/`, exécuter `build_references.py`.
2. **Construction itérative du morceau (Piste par Piste)** :
   * Commencer par les fondations : `kick` puis `bass_sub`.
   * Après l'ajout de chaque piste validée, rendre un export audio temporaire `current_mix.wav`.
   * Pour chaque nouvelle piste (`chord_keys`, `lead_melody`, `pad_atmosphere`, `snare_clap`, `hihat_perc`), appeler `select_sample.py` en passant `--mix-path ./current_mix.wav`.
3. **Gestion des cas limites (Fallback)** :
   * Si le meilleur candidat a un `final_score < 0.55` à cause d'un `masking_penalty` élevé ($> 0.65$), l'agent ne doit pas rejeter le sample aveuglément : il peut soit choisir le candidat #2 avec un masquage plus faible, soit appliquer un filtre passe-haut/EQ sur la piste conflictuelle avant de réévaluer.
   * Si `needs_fade: true` est signalé sur le meilleur candidat, l'agent doit automatiquement appliquer un fondu linéaire de $5\text{ ms}$ en entrée et en sortie lors de l'assemblage.
4. **Apprentissage continu** :
   * Chaque fois que l'utilisateur humain valide un morceau finalisé comme étant particulièrement réussi, l'agent peut proposer de copier les samples utilisés dans `golden_references/<role_id>/` et de relancer `build_references.py` pour affiner continuellement ses goûts.