# Journal d'Apprentissage & Playbook d'Itération pour Agents IA Autonomes

Ce document capitalise les **erreurs réelles rencontrées**, les **pièges techniques de Gridline / MCP / audio**, les **leçons de composition et de mixage**, ainsi que les **méthodes de diagnostic et de correction rapide** vérifiées en production sur ce studio (`/home/gamo/Bureau/musicos`, lien `~/Music/AI-Studio`).

Tout agent IA travaillant sur ce projet doit consulter ce guide **avant** de générer un morceau ou d'exécuter sa boucle d'auto-mixage.

---

## 1. Pièges d'Infrastructure, MCP & Gestion de Projets

### 1.1 Isolation d'un projet hors de `projects/active/`
- **Problème** : Par défaut, [`scripts/gridline-mcp.sh`](../scripts/gridline-mcp.sh) et la configuration MCP pointent sur `projects/active/projet.gridline` et `projects/active/renders/`. De plus, si Gridline GUI est ouvert sur le port `47390`, `gridline mcp --attach` s'y connecte directement et les outils `save_project` / `render_audio` interdisent d'écrire hors du dossier projet ou export actif.
- **Risque** : Écraser le template `projects/active/projet.gridline` ou laisser des rendus dispersés quand l'utilisateur demande un dossier dédié (ex. `projects/active2/`).
- **Solution vérifiée** :
  1. Créer le blueprint, les scripts Lua et les rendus finaux dans `projects/<projet>/`.
  2. Pour sauvegarder le `.gridline` depuis le MCP sans écraser `projects/active/projet.gridline`, utiliser `save_project(path="renders/tmp_projet.gridline")` puis déplacer le fichier vers `projects/<projet>/projet.gridline`.
  3. Valider le fichier autonome sur disque et générer le master final directement via la CLI :
     ```bash
     gridline info projects/<projet>/projet.gridline
     gridline render projects/<projet>/projet.gridline -o projects/<projet>/renders/master_final.wav --bits 24
     ```
  4. Nettoyer les éventuels WAV temporaires laissés dans `projects/active/renders/`.

### 1.2 Volatilité de la session en mode Live GUI (`--attach`)
- **Problème** : Si l'utilisateur recharge un projet ou clique dans la GUI pendant que l'agent opère via `--attach`, l'état en mémoire est réinitialisé (`no channel 1 (the rack has 0)` lors d'un `set_parameters` suivant, avec changement possible du chemin `projectFile` entre `/home/gamo/Bureau/musicos/...` et `~/Music/AI-Studio/...`).
- **Solution** :
  - Construire le socle (tempo, canaux, chargement des samples) en un seul appel `batch`.
  - Si un outil échoue avec une erreur d'objet manquant inattendue, appeler immédiatement `get_song` et `list_channels` pour vérifier l'état réel de la session.
  - Sauvegarder un jalon `.gridline` dès que la structure de base est en place.

### 1.3 Différences de catalogue plugins : Live GUI vs Headless vs CLI
- **Problème** : En `--attach` (GUI active), `list_plugin_catalog` expose ~908 entrées (natifs `gl.*` + plugins tiers CLAP/VST3/LV2), mais certains synthés tiers (ex. `Surge XT`) apparaissent avec `presets: []` et des centaines de paramètres bruts, et ne sont pas disponibles dans le serveur MCP headless (`gridline mcp <fichier>`).
- **Solution** :
  - Toujours vérifier la disponibilité via `list_plugin_catalog` avant d'instancier un plugin.
  - Privilégier les plugins natifs `gl.*` (`gl.sampler`, `gl.analog`, `gl.matrixfm`, `gl.plasma`, `gl.eq`, `gl.comp`, `gl.dist`, `gl.clipper`, `gl.stereo`, `gl.delay`, `gl.hall`, `gl.limiter`) pour garantir une exécution **100 % identique** entre Live GUI, CLI `gridline render` et mode headless.

### 1.4 Pièges des schémas MCP Gridline
- **`get_parameters`** : N'accepte que `channel`, `limit`, `mixer`, `offset`, `query`, `slot` (passer un argument non supporté comme `nonDefaultOnly` provoque une erreur). Toujours lire le schéma dans `~/.gemini/antigravity/mcp/<serveur>/<outil>.json` en cas de doute.
- **`inspect` vs `list_channels` / `edit_mixer_track`** : `inspect(path="channel/N")` expose les IDs internes du fichier projet (ex. `"mixerTrack": 4` pour l'insert logique `2`). Ne jamais utiliser ces IDs internes dans les outils haut niveau : utiliser toujours la numérotation logique (`0 = Master`, `1.. = Inserts`) exposée par `list_channels` et `get_mixer`.
- **`patch` sur `gl.sampler`** : Le champ `plugin.data` attend **impérativement** une chaîne de chemin absolu brute (`{"plugin": {"data": "/chemin/absolu/sample.wav"}}`), jamais un objet JSON.

### 1.5 Pièges des scripts Lua (`run_script`)
- **Structure obligatoire** : Encapsuler tout le code dans `function apply() ... end` et **ne jamais appeler `apply()` au top-level**.
- **`"changed": false` trompeur** : `run_script` peut renvoyer `"changed": false` même lorsque `P.patterns.setNotes(...)` a bien écrit les notes dans le pattern. Toujours vérifier la présence réelle des notes avec `get_pattern(pattern=N)` et inspecter le tableau `log`.
- **Unités de temps et bornes** :
  - Dans `set_notes` (outil MCP), les temps sont en **beats**.
  - Dans Lua (`project.patterns.setNotes`), `time` et `length` sont en **ticks** (`ppq = project.song.ppq()`, soit `96` ticks/beat par défaut) :
    - `1/8` = `ppq / 2` (`48` ticks)
    - `1/16` = `ppq / 4` (`24` ticks)
    - `1/32` = `ppq / 8` (`12` ticks)
    - Triolet `1/16T` = `ppq / 6` (`16` ticks)
    - Triolet `1/32T` = `ppq / 12` (`8` ticks)
  - `pan` dans Lua `setNotes` est borné à `-64..64`, `pitch` et `velocity` à `0..127`.
- **Portabilité** : Rechercher les canaux et patterns par nom (`P.channels.find("Hi-Hat Closed")`, `P.patterns.find("Drop A")`) plutôt que par index codé en dur.

---

## 2. Erreurs de Composition, Accordage & Sound Design

### 2.1 Accordage réel des 808 de la banque (`audio-index` vs réalité physique)
- **Problème** : L'étiquette `musical_key` dans `audio-index` (calculée par Essentia) peut indiquer une tonalité approximative ou ne précise ni l'octave exacte ni la dérive en cents. De plus, `gl.sampler` a par défaut `Root note = 60` (`C4`). Si une 808 samplée en `C1` (`32.7 Hz`) est jouée au MIDI `36` (`C2`) avec `Root note = 60`, elle descend de 2 octaves (`~8 Hz` inaudible).
- **Cas concrets mesurés dans la banque locale** :
  - `06. 808 - bat wing yaiba (zayspinz).wav` : fondamentale stable à `32.96 Hz` (`C1 +13.5 ct`), faible DC (`-0.00074`).
  - `ctm 808.wav` & `808 Bass Cm Last 002.wav` : fondamentale à `65.0–65.3 Hz` (`C2`).
  - `Deep 808.wav` : attaque à `33.65 Hz` descendant à `32.97 Hz`, avec un DC offset notable (`0.027`).
- **Protocole obligatoire avant d'écrire une ligne de 808** :
  1. Mesurer avec Python (`soundfile` + FFT après les premiers `80 ms` d'attaque pitch-drop) la fondamentale réelle (`Hz`), sa stabilité entre `0.05 s` et `1.25 s`, et le DC offset.
  2. Régler `gl.sampler` : `Root note` sur l'octave réelle du sample (`24` pour `C1`, `36` pour `C2`), corriger `Fine tune` en cents (ex. `-14 ct`), activer `Remove DC = On`, passer en monophonique (`Voices = 1`, `One shot = Off`, `Release = 0.035 s`, `Glide = 0.045 s`) pour que deux notes de 808 ne se chevauchent jamais.
  3. **Vérifier sur rendu** : créer un pattern test d'une note sur `C2` (`pitch: 36`), lancer `render_audio(scope="pattern")` et contrôler par FFT que la fondamentale rendue tombe bien à `65.4 Hz` (dans notre cas : `65.27 Hz`, soit `-3.5 ct`, et `L-R diff = 0.000000`).

### 2.2 Rythmique half-time en Trap (130–160 BPM)
- **Piège** : Placer le clap sur les beats `1` et `3` (`start = 1.0, 3.0`) à 140 BPM transforme un beat Trap en rythme House/Dance rapide.
- **Règle** : En Trap à 140 BPM (`4/4`), le beat est ressenti en **half-time (70 BPM)**. Le clap/snare principal frappe uniquement sur le **3e temps de chaque mesure** (`start = 2.0, 6.0, 10.0, 14.0` en indexation 0-based), soutenu par des ghost snares/rims en syncope (`1.75`, `3.5`).

### 2.3 Sélection du couple Kick + 808 sans masquage
- **Piège** : Un kick très long (`> 0.5 s`) ou trop chargé en sub (`20–50 Hz`) entre en conflit de phase avec la 808.
- **Solution** :
  - Choisir un kick court et punchy (`0.22–0.31 s`) dont le gros de l'énergie se situe entre `60` et `140 Hz` (ex. `Kick Drum 046 Tit Tit.wav` : `0.257 s`, `61%` en `60–140 Hz`).
  - Couper sous `35 Hz` sur le kick, creuser `~82 Hz` (`-2.5 dB`) sur la 808, et router un envoi **sidechain pré-fader** du Kick vers `gl.comp` sur la 808 (`Detector input = Sidechain`, `Attack = 2 ms`, `Release = 85 ms`, `Lookahead = 2 ms`).
  - Ajouter une saturation harmonique légère sur la 808 (`gl.dist` mode `Soft`, `Oversampling = 4x`, `Mix ~45%`) pour qu'elle reste audible dans les `80–250 Hz` sur casques/enceintes sans gonfler le sub `< 60 Hz`.

### 2.4 Pièges des presets intégrés (`gl.eq`, `gl.analog`, `gl.delay`)
- **`gl.eq` preset `"Kick punch"`** : Contient un `Low shelf +4 dB` à `55 Hz` qui concurrence directement la 808. Toujours le convertir en `Low cut` à `35 Hz` + pic de punch vers `85 Hz`.
- **`gl.analog` preset `"Velvet Pad"`** : Embarque sa propre `Reverb`, une attaque très lente (`900 ms`), `Unison spread = 100%` + `Chorus = On`. Ajouter un LFO sur le pitch ou un fort `Unison detune` (`0.42`) + `Resonance` (`0.18`) donne un son instable/« étrange » et une bosse résonante nasale vers `500–900 Hz`.
  - **Correction** : Désactiver `LFO 1 target = None`, `Drive = 0`, `Chorus = Off`, limiter à `Unison voices = 2` (`detune 0.12`, `spread 0.45`), filtre doux `LP 12` (`Resonance = 0.02`, `Cutoff ~980 Hz`), et équilibrer l'EQ (`+2.5 dB` à `320 Hz`, `-3.5 dB` à `650 Hz`).
- **`gl.matrixfm` (`Lattice`) `"Glass Bell"` en registre aigu** : Sur des notes `C5–Ab5` (`MIDI 72–80`), les partiels FM métalliques deviennent perçants et agressifs au casque.
  - **Correction** : Pour des mélodies sombres claires et musicales, privilégier un vrai piano acoustique (`gl.keys` `Model = 4` `"Grand piano"`, `Muffle ~3800 Hz`, `Hardness ~0.38`, `Overdrive = 0`, `Tremolo depth = 0`) dans le registre naturel `C4–Ab4` (`MIDI 60–68`).
- **`gl.hall` (`Cloister`) `"Grand hall"` sur plusieurs pistes** : Embarque par défaut un boost `Wet EQ +3.15 dB` à `275 Hz` et un decay de `3.1 s`. Envoyer clap, snare, hats, pad et lead dedans produit une sensation de « grande pièce vide avec résonance ».
  - **Correction** : Laisser les drums Trap **secs et frontaux** (0 % réverbe sur kick, 808, hi-hats, open hat, snare ; au plus `6%` sur le clap) et utiliser un bus `gl.reverb` `"Small room"` court (`Decay time ~0.9 s`, `Low cut 320 Hz`, sends `12–14%` max sur pad/piano).
- **`gl.delay` preset `"Dotted eighth"`** : Utilise `Feedback mode = Ping-pong` par défaut, ce qui peut créer des écarts de phase droite/gauche indésirables. Préférer `Normal` (`Wet 10%`) si le mix souffre d'un manque de compatibilité mono.

---

## 3. Boucle d'Auto-Mixage : Erreurs Types & Diagnostics `audio-analyzer`

L'enchaînement réel observé lors du mixage d'« Obsidian Midnight » illustre les 4 dérives les plus fréquentes d'un agent IA et comment les corriger méthodiquement par itération.

### 3.1 Dérive 1 — Mix noyé dans le Sub (`v1` : `-9.88 LUFS`, `Sub 36.8%`, `Bass 60.1%`, `Air 0.55%`)
- **Symptôme** : La 808 seule atteignait déjà ~`-11.5 LUFS`, écrasant pad, cloche, clap et hi-hats (`Air < 1%`).
- **Diagnostic** : Dans la pondération ITU-R BS.1770 (LUFS), les basses fréquences pèsent moins que les médiums/aigus, ce qui pousse un agent à monter les faders sans voir que le sub sature déjà la plage dynamique.
- **Correction** : Baisser franchement la 808 (`0.82 → 0.54`), remonter les percussions hautes (`Hi-Hat`, `Clap`, `Snare`) avec un `High shelf` vers `8 kHz` (`+3.5 dB` sur les hi-hats), et vérifier dans `analyze_mix` que `air >= 3%` (`6.06%` atteint au final).

### 3.2 Dérive 2 — Opposition de phase stéréo & perte de dynamique (`v2` : `correlation_full = -0.138`, `mono_compatible: false`)
- **Symptôme** : En montant le `Dark Pad` et en poussant `gl.stereo` (`Width = 1.45`) par-dessus `Unison spread 100%` + `Chorus 50%`, l'Intro est devenue presque aussi forte que les Drops (`-12.47 dB` vs `-11.83 dB`) et la bande `500–2000 Hz` a absorbé `61.76%` de l'énergie avec une corrélation négative (`-0.305`).
- **Règle de diagnostic** : Dès que `correlation_full < 0.3` ou `stereo_width_ratio > 0.8`, vérifier par bande et par section quel instrument a un signal Side (`L-R`) supérieur au signal Mid (`L+R`). Ne **jamais** empiler un élargisseur `gl.stereo > 1.1` sur un synthé ayant déjà Unison + Chorus + Réverbe stéréo.
- **Correction** : Réduire `Unison spread` (`0.55`), `Chorus mix` (`0.25`), ramener `gl.stereo` du pad à `Width = 0.90` (`Mono below 220 Hz`), ramener `gl.hall` à `Width = 0.95`, et baisser `BUS Music` (`0.72`). La corrélation est immédiatement remontée à `+0.810` (`mono_compatible: true`).

### 3.3 Dérive 3 — Overshoot True Peak 4x, clipping post-limiteur & grésillements (`v3 → final`)
- **Symptôme A (ISP)** : Un limiteur travaillant à `44.1 kHz` sans oversampling 4x garantit le niveau d'échantillons (`sample_peak_dbfs`), mais des transitoires très raides (`hi-hats` rapides, `gl.clipper` sur bus batterie, aigus jusqu'à 20 kHz) génèrent des **crêtes inter-échantillons (ISP)** de `+1.3 à +2.0 dB` lors de la mesure 4x True Peak (`pyloudnorm` / `scipy.signal.resample_poly`).
- **Symptôme B (Grésillement / Clipping)** :
  1. Monter le fader `Master` (`mixer 0`) au-delà de **`0.8` (`0.0 dB` unity)** (ex. `0.92 = +1.2 dB`) applique du gain **après** `gl.limiter` et provoque un clipping numérique (`776` samples clippés).
  2. Régler `gl.limiter` `Attack (look-ahead)` à `5 ms` peut laisser passer une transitoire rapide, tandis qu'empiler `gl.dist` sur la 808 + un `Gain` limiteur excessif (`+8.6 dB`) génère du grésillement d'intermodulation sur les graves et augmente la charge CPU en lecture live (`Unison = 5`, `gl.hall High density = On`).
- **Correction vérifiée** :
  1. Toujours laisser le fader `Master` à **`0.8` (`0.0 dB`)** et `Attack (look-ahead)` de `gl.limiter` à **`0.002 s` (`2 ms`)** avec `0.2 dB safety = On`.
  2. Désactiver `gl.clipper` sur `BUS Drums` et couper `gl.dist` sur la 808 si le sample possède déjà assez d'harmoniques (`enabled = false`), en activant `Declick = Transient` sur `808` et `Hi-Hat`.
  3. Ajouter un coupe-haut propre à `17.5 kHz` sur l'EQ Master (avant le limiteur) pour supprimer les fréquences proches de Nyquist responsables des rebonds de Gibbs.
  4. Calibrer les bus (`BUS Drums 0.96`, `BUS Music 0.96`, `Master Comp Makeup +2.5 dB`) puis ajuster `gl.limiter` (`Ceiling = -2.5 dB`, `Gain = +6.8 dB`).
  - *Résultat final vérifié (`master_final.wav`)* : **`-11.59 LUFS`**, **`-1.12 dBTP`**, **`0` sample clippé**, `correlation_low_end_150hz = 0.915`, `air = 5.13%`, verdict `audio-analyzer` : *« Mix équilibré : loudness, True Peak, spectre et phase dans les cibles. »*

### 3.4 Dérive 4 — Divergence entre `render_audio` MCP et `gridline render` CLI
- **Constat** : `gridline render` CLI inclut la queue complète (`--tail leave` par défaut, ex. `56.675 s` vs `55.714 s` via MCP) et affiche un LUFS arrondi à 1 décimale (`-11.2 LUFS`).
- **Solution** : Toujours relancer `audio-analyzer:analyze_mix` sur le `.wav` final produit sur disque avant de valider la livraison.

---

## 4. Protocole Standard d'Itération (Checklist pour Tout Agent)

1. **Blueprint d'abord** : Rédiger `song_blueprint.json` dans le dossier projet cible et valider contre `schemas/song_blueprint.schema.json`.
2. **Audit des samples** : Chercher via `audio-index`, puis mesurer en Python `f0`, durée et DC offset des kicks et 808 avant programmation.
3. **Construction modulaire** :
   - Routing strict : chaque canal sur son insert (`>= 1`), groupé vers `BUS Drums (10)`, `BUS Music (11)`, `FX Hall (12)` → `Master (0)`.
   - Vérifier `"silent": false` dès le premier pattern via `analyze_loudness(scope="pattern")`.
   - Vérifier l'accordage 808 par FFT sur un rendu court.
4. **Boucle Auto-Mix itérative (ne jamais livrer au 1er rendu)** :
   - Rendre → `analyze_mix(target_lufs=...)` + `analyze_low_end_masking` → diagnostiquer → ajuster faders/EQ/stéréo/limiteur → re-rendre → `compare_renders`.
   - Critères d'arrêt obligatoires :
     - `clipped_samples == 0`
     - `true_peak_dbtp <= -1.0`
     - `integrated_lufs` à `± 1.2 dB` de la cible (viser `± 0.4 dB`)
     - `correlation_low_end_150hz >= 0.5` (viser `>= 0.75`) et `mono_compatible == true`
     - `air (6–20 kHz) >= 3%` (sauf esthétique lo-fi volontairement filtrée) et `low_mids <= 24%`.
5. **Livraison vérifiée** : Sauvegarder `.gridline`, `.mid` (optionnel) et `.wav` final dans le dossier cible, puis confirmer avec `gridline info` + `analyze_mix` sur le fichier final.
6. **Mise à jour obligatoire de ce journal** : À la fin de **chaque itération** (composition, mixage, débogage ou ajout d'outillage), ajouter ci-dessous les nouveaux problèmes rencontrés, mesures chiffrées et solutions vérifiées afin que l'agent suivant ne répète jamais une erreur déjà résolue.

---

## 5. Playbook des Nouveaux Outils d'Autonomie (Diagnostic Ciblé)

En complément de la boucle de base, tout agent autonome doit appliquer les vérifications suivantes :
- **Avant d'écrire/valider des synthés criards ou faux** : lancer `audio-index:search_synth_presets(query=..., category=...)` pour choisir un preset `gl.*` selon son vrai centroïde/timbre CLAP, puis vérifier le MIDI avec `audio-analyzer:audit_midi_harmony(midi_path, expected_key=..., max_pitch=84)`.
- **En cas d'impression de « grande pièce vide / résonance studio »** : lancer `audio-analyzer:render_project_stems(project_path)` et `audio-analyzer:audit_mix_deep(audio_path, headphone_target="sony_xm6")` pour isoler la piste fautive (`boxy_200_500hz_ratio`), contrôler `low_mid_200_800hz_correlation >= 0.55` et `estimated_rt60_ms < 650`.
- **En cas de grésillements (live ou rendu)** : lancer `audio-analyzer:check_pipewire_health()` (recommande `pw-metadata -n settings 0 clock.force-quantum 1024` si Bluetooth XM6 actif) et surveiller `micro_glitch_or_crackle_events` dans `audit_mix_deep`.
- **Pour caler l'équilibre fréquentiel sur une référence pro** : appeler `audio-analyzer:match_reference_audio(mix_path, reference_path)`.

---

## 6. Leçons Vérifiées — Extension d'Autonomie & Refonte *Obsidian Midnight* (2026-11-21)

### 6.1 Bug critique des moteurs "Reference / Full" dans `gl.reverb` et `gl.delay`
- **Problème réel constaté** : Lors du premier audit sur `projects/active2`, malgré des paramètres `gl.reverb` (`Dry = 0`, `Low cut = 320 Hz`, `Decay = 0.9 s`) et `gl.delay` (`Wet = 10%`, `Feedback = 22%`) apparemment propres, `audit_mix_deep` mesurait toujours `low_mid_200_800hz_correlation = 0.085` et des pics à `500/630/800 Hz`, produisant un écho de grande pièce vide.
- **Cause racine** :
  - `gl.reverb` active par défaut **`param 38 (Full model version) = 1 ("2")`** et lit alors les paramètres `(full)` (`params 23..35`) au lieu des index legacy (`1..14`). Ces index cachés étaient restés à **`Dry (full) = 128 (100%)`** sur un bus send (!), `Early reflections (full) = 128`, `Low cut (full) = 75 Hz`, `Decay (full) = 15 (1.5 s)`.
  - `gl.delay` (« Tracer ») active par défaut **`param 31 (Delay model) = 1 ("Reference")`** et lit `params 50..75` (`73 Wet (stereo ref.) = 80%`, `64 Feedback = 50%`), ignorant complètement `Wet = 10%`.
- **Solution vérifiée** : Toujours inspecter la liste complète avec `gridline plugin-params <plugin>` ou `get_parameters(mixer=M, slot=S)`. Sur `gl.reverb`, fixer impérativement `param 33 = 0` (`Dry full`), `param 23 >= 450 Hz`, `param 26 <= 24`, `param 28 = 3..6` (`0.3–0.6 s`) et `param 34 = 10..20`.

### 6.2 API MCP Gridline : schémas exacts, rate limit & routage
- **`batch`** : Utilise `{"calls": [{"tool": "...", "arguments": {...}}]}` (et **non** `"args"` ni `"label"`).
- **`set_parameters`** : Utilise `{"changes": [{"path": "mixer/M/slot/S/P", "value": V}]}` (jusqu'à 500 changements en un appel).
- **Rate limit (`-32000`)** : Enchaîner plus d'une trentaine d'appels unitaires déclenche `Rate limit: too many tool calls`. Regrouper les EQs dans un unique `set_parameters` et les notes/patterns/clips dans des appels `batch` (avec retry + backoff `0.75 s` dans les scripts Python).
- **Pistes playlist & MP3** : Le renommage de piste playlist passe par `patch(path="track/N", set={"name": "..."})` (pas `playlist/tracks/N`). L'outil MCP `export_project` ne supporte que `midi`, `flp` et `flp_zip` ; l'export MP3 passe par `gridline render <proj> -o <out.mp3> --format mp3 --bitrate 320`.
- **Double routage avec `add_mixer_track`** : Une nouvelle piste insert route par défaut vers `Master (0)`. Toujours passer `routeTo=10` ou `routeTo=11` via `edit_mixer_track` lorsqu'elle passe par un bus (`BUS Drums` / `BUS Music`).

### 6.3 Audition objective, SFZ natif dans `gl.multisampler` & choix de sonorités
- **Vrai piano acoustique headless & GUI** : `gl.multisampler` charge nativement les `.sfz` via `patch(path="channel/N", set={"plugin": {"data": "/.../Headroom Piano (NoFX).sfz"}})` (mesuré justesse `262.1 Hz` sur `C4`, `conf = 0.97`, Audiobox `CE = 5.03`). Toujours ranger les banques SFZ dans `soundbanks/sfz/` et **jamais** dans `samples/` (pour ne pas polluer `search_samples` avec des centaines de notes individuelles).
- **Audition préalable contre les pièges stéréo/timbre** : Un pré-test de candidats a montré que `gl.plucker / Soft harp` avait une corrélation stéréo de **`-0.000`** (totalement déphasé), alors que **`gl.twinstring / Nylon pluck`** atteignait `PQ = 7.40`, `corr = +0.865`, `sharp = 1.04`, et **`gl.dcosynth / Polar Strings`** `CE = 4.74`, `corr = +0.777`, `sharp = 0.69`.
- **`gl.dcosynth` & `gl.twinstring` gain staging** : `gl.twinstring` démarre avec `param 1 (Level) = 0.5`, et `gl.dcosynth` sort nettement plus bas que `gl.sampler` (`-12.9 dB` vs `-3 dB`). De plus, `Polar Strings` a une corrélation `200–800 Hz` de `0.395` seul : régler `stereoSep = 0.55` sur son insert + passe-haut `240 Hz` préserve une corrélation globale `>= 0.65`.

### 6.4 Notes d'interprétation sur `audit_mix_deep`, `audit_midi_harmony`, `PipeWire` et `audiobox_aesthetics`
- **Harmonie multi-pistes** : Sur une basse `Ab -> Eb/G` en fin de mesure 3 (`Ab` vers `G2`), ou `Fm -> Gm` en mesure 4, ne pas laisser traîner un `Ab` sur `G2` ni un `Eb` contre un `D` dans les accords tenus (`Polar Strings` + `Classic Tine` raccourcis avant `beat 11.0`, piano sans `C3` sur les drops avec 808). Résultat vérifié sur `obsidian_midnight.mid` : `0` note hors gamme, `0` clash de 2nde/9ème mineure, `max_pitch = 80` (`Ab5`).
- **`estimated_rt60_ms` sur mix complet avec 808** : `pyroomacoustics.rt60` mesure `~1,3 s` même sur un mix sec dès lors que la 808 tient ses notes sub sur 3–4 temps. Toujours croiser avec `low_mid_200_800hz_correlation` (`0.665`), `slapback_echo_score` (`0.199`) et la sonde CLAP `reverb_vs_dry` (`healthy +0.113` vs `defect -0.103`).
- **Grésillements live sous Sony XM6 vs rendu WAV** : `check_pipewire_health()` a capturé en direct `active_quantum = 256` et `gridline-app ERR=431` sur `bluez_output` alors que `master_final.wav` affichait `clipped_samples = 0`, `micro_glitch_or_crackle_events = 0` et `true_peak_dbtp = -1.25 dBTP`. Solution live : `pw-metadata -n settings 0 clock.force-quantum 1024`.
- **Intersample True Peak (`ISP`) de `gl.limiter`** : Avec des attaques très rapides et un `air shelf` actif, `true_peak_dbtp` dépasse `sample_peak_dbfs` d'environ `+1.5 à +2.0 dB` (un `Ceiling = -2.1 dB` peut laisser passer `-0.55 dBTP`). Viser `Ceiling = -2.75..-3.0 dB` avec `Attack = 1 ms` et `0.2 dB safety = On` pour garantir `true_peak_dbtp <= -1.0 dBTP`.
- **`audiobox_aesthetics` sous `torchaudio >= 2.11` / Python 3.14** : Passer un chemin fichier déclenche `ModuleNotFoundError: torchcodec`. Précharger le tenseur via `soundfile` (`[channels, samples]`) évite complètement `torchcodec`.

### 6.5 SFZ live GUI (`gridline-app --attach`), saturation Master et signe `stereo` (2026-11-21)
- **Piège critique `gl.multisampler` en live GUI (`--attach`) vs CLI headless** : `Headroom Piano (NoFX).sfz` (SFZv2 avec `#define`, `#include`, `<master>`, `locc$C_VOL=1`, `<curve>`) jouait à `-21.69 LUFS` via `gridline mcp <fichier>`, mais rendait **`-144 LUFS` (`silent: true`)** dans l'instance graphique `gridline-app` (`--attach`). De plus, dans un `.sfz` aplati, écrire plusieurs directives sur une seule ligne (`<region> sample=Samples/HEADROOM PIANO ...flac lokey=21`) échoue si le chemin du sample contient des espaces.
  - **Solution vérifiée** : générer un fichier SFZv1 pur avec **une directive par ligne (`\n`)** (`soundbanks/sfz/headroom-piano/Headroom Piano Clean.sfz`, `150` régions Close multi-vélocités avec `install.sh`). Vérifié à parité exacte (`0.00 dB` d'écart) : `PAT 9` Intro = `-17.91 LUFS` et morceau complet = `-11.80 LUFS` aussi bien en live (`--attach`) qu'en rendu CLI/fichier sauvegardé.
- **Écoute et édition manuelle UI / récupération Master (`MIX 0`)** : Lors de manipulations manuelles pour retrouver un instrument silencieux, la désactivation de `gl.comp`/`gl.limiter` (`Gain=+24 dB`, `Ceiling=+2.9 dB`, `mix=20%`), l'activation d'un `gl.convolver` (`Room` à `80%`) sur le Master et des boosts EQ inversés (`+5.7 dB @ 630 Hz`, `+5.8 dB @ 8.8 kHz`) produisent un fort écrêtage (`+9.84 dB peak`, `34 920` samples clippés) et une résonance de pièce. Toujours nettoyer `MIX 0` (`EQ -> Comp -> Limiter Ceiling -4.2 dB`) et éviter le double creux `630 Hz` (`MIX 8` + `MIX 11 = -9.5 dB`) sur le piano.
- **Signe de `mixer/N/stereo` et cible `--export-dir` avec `--attach`** :
  - `mixer/N/stereo` (`Stereo separation`) fonctionne à rebours d'une « largeur » positive : les valeurs **négatives (`-0.32` à `-0.46`) fusionnent vers le centre mono**, et positives élargissent (`Grand Piano -0.32` + `BUS Music -0.35` -> `low_mid_200_800hz_correlation` portée de `0.406` à `0.806`).
  - `gridline mcp --attach` ignore `--export-dir` et exporte dans `projects/active/renders/`. Toujours lancer les exports finaux `.wav`/`.mp3`/`.mid` directement depuis le `.gridline` sauvegardé via `gridline mcp <proj> --export-dir ...` et `gridline render <proj> -o <out.mp3> --format mp3`.


