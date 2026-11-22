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

1. **Blueprint (`harmonic_core`) & Gate `master_sketch.mid` d'abord** : Rédiger `song_blueprint.json` (incluant `harmonic_core` avec voicings calculés par *Smooth Voice Leading* `<= 2.5 st` et cellule `A-A'-A''-B`) et valider contre `schemas/song_blueprint.schema.json`. Générer la réduction unique `master_sketch.mid` et passer le Gate bloquant `audit_midi_harmony(..., strict_musicality=True)` (`musicality_verdict == "PASS"`) **avant** toute batterie ou mixage, conformément à [regles_melodie_et_cohesion_inter_pistes.md](regles_melodie_et_cohesion_inter_pistes.md) et [protocole_universel...md](protocole_universel_d_architecture_musicale_coh_rence_multi_chelles_mod_le_pr_dictif_orientation_trap.md).
2. **Audit des samples, Accordage `440 Hz` & Enveloppes** : Chercher via `audio-index`, mesurer en Python `f0`, tuning en cents (`A4 = 440 Hz ± 5 ct`), durée et DC offset ; dimensionner les enveloppes ADSR (`Voices = 1` *cut itself* sur 808, ticks courts `20–90 ms`, queues reverb/delay multiples de `T_noire` coupées aux transitions). Jamais de transposition parallèle d'un sample d'accord complet.
3. **Construction modulaire & Ciment Timbral `BUS Music`** :
   - Routing strict : chaque canal sur son insert (`>= 1`), groupé vers `BUS Drums (10/17)`, `BUS 808 (18)`, `BUS Music (11/19)` (chaîne `gl.eq` LPF `11–14 kHz` + `gl.dist Soft 4x Mix 8–12%` + `gl.comp Glue` + `gl.comp` Sidechain Micro-Ducking Kick), `FX Room (12/20)` (`Dry = 0`, `Dry (full) = 0`) → `Master (0)`.
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

---

## 7. Correctifs Globaux CPU, PipeWire & Sobriété DSP (`Souls in Motion`, `projects/active3`, 2026-11-21)

### 7.1 Diagnostic réel de saturation CPU (`gridline-app` à 256 % CPU & freezes `libSDL3`)
- **Constat mesuré** sur Intel Core `i5-1240P` (12 cœurs hybrides / 16 threads, profil `powersave` à `1700 MHz`) :
  - `gridline-app` tournait à **`256 %` CPU** avec **`ERR=2850..3207` xruns PipeWire** (`check_pipewire_health`).
  - Causes identifiées dans `~/.config/gridline/settings.json` et PipeWire :
    1. `audio.bufferFrames = 256` et `pulse.min.quantum = 256/48000` (172 callbacks audio/seconde en temps réel + resampling 44.1→48 kHz).
    2. `audio.threads = 0` (auto sur les 16 threads logiques P+E-cores), provoquant contention et throttling thermique.
    3. `freeze.autoHibernate = false` (les plugins silencieux restaient actifs en permanence).
    4. `ui.animations = true`, `ui.smoothScroll = true`, `live.pluginWindowsSeveral = true`, `draw.smooth.* = true` (4 freezes enregistrés dans `~/.config/gridline/crash-reports/freeze-*.txt` sur `libSDL3` / `Mesa Intel Iris Xe`).
- **Correctifs globaux appliqués** :
  1. **Gridline (`~/.config/gridline/settings.json`)** : `audio.bufferFrames = 1024`, `audio.threads = 4`, `freeze.autoHibernate = true` (`sleepSeconds = 15`), `live.pluginWindowsSeveral = false`, `ui.animations = false`, `ui.smoothScroll/Snap/Zoom = false`, `draw.smooth.* = false`.
  2. **PipeWire persistant** : `~/.config/pipewire/pipewire.conf.d/99-ai-studio-lowcpu.conf` (`default.clock.quantum = 1024`, `min-quantum = 1024`, `max-quantum = 2048`) et `~/.config/pipewire/pipewire-pulse.conf.d/99-ai-studio-lowcpu.conf` (`pulse.min.req/quantum = 1024/48000`), complétés par `pw-metadata -n settings 0 clock.force-quantum 1024` dans `scripts/launch-gridline.sh`.
  3. **Règle de sobriété DSP dans les projets IA** : n'instancier `gl.eq` que sur les pistes qui le justifient (économiser les EQs sur percussions propres routées vers `BUS Drums`, passant ici de 22 EQs à 11), garder `Quality = 0` (`Normal`) sur les inserts (`Quality = 1` réservé au `Master`), et limiter `gl.multisampler` à `Polyphony <= 24`.
  4. **Zéro ajout de sons externes / synthés électriques non fournis** : Lorsqu'un projet contient déjà une sélection utilisateur (`Untitled.gridline`), ne **jamais** ajouter de synthés lourds/électriques (`gl.matrixfm` `"Saw Filter Lead"`, `gl.dcosynth`, `gl.twinstring`, etc.) ni de samples non fournis. Enrichir la musicalité uniquement via les stems, 808s, percussions et voix préparés (filtrage doux, découpes, variations de sections, `gl.snare` natif déjà présent).

### 7.2 Leçons de composition & mixage sur `projects/active3` (*Souls in Motion*, 40 mesures @ 164 BPM, `D# minor`)
- **Architecture finale 100 % fidèle aux sons fournis (`19` canaux uniquement)** : Suppression de tout synthé ajouté (`Warm Lead` `gl.matrixfm`, `Polar Strings` `gl.dcosynth`, `Nylon Pluck` `gl.twinstring`, `Headroom Piano` `gl.multisampler`) et regroupement des 5 `808` sur un seul `BUS 808 (20)` (`12` effets DSP légers au total). Benchmark vérifié (`gridline bench --threads 4`) : **`7.50x` temps réel** (`1,43 s` pour `10,73 s` audio), et rendu final **`-11.13 LUFS`**, **`-1.30 dBTP`**, `0` sample clippé, `0` résonance 1/3 octave, `PQ = 7.916`.
- **Tonalité réelle vs nom de fichier du pack** : La boucle du pack `Rossi - SOULS` indique `Db minor - 164 bpm`, mais `2 - Guitar.wav/mid`, `2 - Piano.wav/mid` et `2 - Melo.mid` jouent en réalité en **`D# minor` (`Eb minor`)** (`D#4 = 311 Hz`, `D#5 = 623 Hz`, confirmé par `audit_mix_deep` `detected_key = "Eb minor"`, confiance `0.824`). Toujours mesurer le spectre/MIDI réel avant d'accorder la 808.
- **Note pan dans `set_notes` vs Lua** : Dans l'outil MCP `set_notes`, `pan` est un **entier `-64..64`** (et non un flottant `-1..1` comme sur `add_channel`). Passer `-0.08` fait échouer silencieusement le `batch` si l'on ne vérifie pas `ok: false` dans la réponse JSON.
- **Zéro faux positif harmonique sur stems `gl.sampler`** : Définir `Root note = 63` (`D#4`) et déclencher les stems (`2 - Guitar`, `2 - Piano`), le triangle et les ad-libs sur `pitch = 63` conserve exactement le pitch d'origine tout en garantissant `0` note hors gamme dans `audit_midi_harmony`.
- **Équilibre spectral de `2 - Piano.wav` & `MZ Vox [Grrrrraahhh!]`** : `2 - Piano.wav` concentrait `75,3 %` de son énergie entre `250–500 Hz` (corrélation `+0,495`), et `MZ Vox [Grrrrraahhh!]` avait une corrélation de `-0,035`. Correction vérifiée : HPF `185 Hz` + coupe `-4,5 dB @ 320 Hz` + `stereo = -0,22` sur `2 - Piano`, et `stereo = -0,45` sur `MZ Vox [Grrrrraahhh!]`.

---

## 8. Projet `projects/active4/projyy.gridline` (*Hyper Ambient Zone*, 76 mesures @ 150 BPM = 122,4 s, `D minor`)

### 8.1 Détails techniques MCP & paramètres Gridline 0.7.7
- **Choix `gl.eq` & `gl.limiter`** : Dans `gl.eq`, le type passe-haut s'appelle `"Low cut"` (et non `"High pass"`), et le facteur Q se nomme `"Band N Q (legacy)"`. Dans `gl.limiter`, le preset `"Brickwall -1 dB"` peut laisser des crêtes inter-échantillons (`+0.2 dBTP` en `v1`) ; régler explicitement `Ceiling = -1.6 dB` avec `0.2 dB safety = 1.0` et `Gain = +3.5..+4.0 dB` garantit `true_peak_dbtp = -1.01 dBTP` (`sample_peak_dbfs = -1.81 dBFS`, `0` sample clippé) tout en portant le loudness à `-11.21 LUFS`.
- **Nappe ambiante continue & évolution harmonique (`oddlogic` Base + `oddlogic High (+12)`)** : Pour un sample d'ambiance de 32,58 s (`oddlogic - total hyper ambient zone vibe.wav`, accord naturel `D3-F3-A3-D4`) sur un morceau de 2 minutes, activer `Loop = true`, `Loop mode = "Ping-pong"`, `Loop start = 0.08`, `Loop end = 0.92`, `Loop crossfade = 0.18`, `Attack = 0.35 s`, `Release = 1.2 s`. Pour faire évoluer la 2e minute sans ajouter de sample externe, dupliquer le canal en `oddlogic High (+12)` (joué à l'octave `MIDI 72`, `Cutoff = 4800 Hz`, HPF `420 Hz`, `-3 dB @ 3000 Hz`).
- **Hi-hat Trap brillant (`11. hh - game got told.wav`)** : Son pic naturel vers `6.3–8 kHz` (`+11.87 dB` en `v1`, `sharpness = 2.201 acum`) est ramené dans la zone saine (`sharpness = 1.708 acum`, `air = 5.02%`) par une coupe `-2.5 dB @ 7200 Hz` (`Q = 1.1`) + un léger `High shelf +2.0 dB @ 10.5 kHz`.
- **Rate-limit MCP en script Python stdio** : Lors d'enchaînements rapides de `set_parameters` / `set_send` / `set_notes` sur `gridline mcp <proj>`, espacer les appels de `~50 ms` (ou regrouper via `batch`) et prévoir un retry sur `"Rate limit"`.

### 8.2 Organisation Playlist & Hygiène du Master (retour utilisateur)
- **Une piste Playlist dédiée par instrument / version** : Ne jamais empiler tout l'arrangement sur une seule piste Playlist (`Track 1`). Séparer chaque instrument et chaque variation (`Vox Chant`, `808 Infinite`, `808 Dream`, `Kick`, `Clap`, `Perc`, `HH Main`, `HH Accent`, `Open Hat`, `oddlogic Base`, `oddlogic High (+12)`, `Dream Synth`) sur des pistes dédiées (`Tracks 1..12`) avec des patterns nommés par section pour faciliter la relecture humaine.
- **Respect des mutes/suppressions utilisateur & hygiène `Master`** : Toujours inspecter `list_channels` et `get_mixer` avant d'itérer (ici `Snare` `mixer 5` mutée par l'utilisateur et `tempo = 150 BPM` conservés). Vérifier également qu'aucune `gl.reverb` ou `gl.limiter` en doublon ne reste sur le bus `Master` (`mixer 0`) : la suppression d'une réverbe accidentelle sur le Master a fait remonter `low_mid_200_800hz_correlation` de `0.701` à `0.812` et restauré la clarté du kick/808.

### 8.3 Cadence 808 à 150 BPM & diagnostic des pistes inactives (`master_2min_v3.wav`)
- **Piège de densité 808 à 150 BPM** : Écrire `6` à `7` frappes de `01. 808 - infinite` par mesure (`1/16` ou `1/8` en rafale) à `150 BPM` rend la basse trop rapide et étouffe le groove Trap.
  - **Solution vérifiée** : Limiter la 808 à **3–4 frappes espacées par mesure** (`0.0`, `1.5`, `2.75`, avec glissando `porta` d'octave `+12` ou turnaround sur `3.5` une mesure sur deux) en synchronisant `Kick - M2` sur les mêmes appuis (`master_2min_v3.wav` : `-11.56 LUFS`, `-1.07 dBTP`, `0` clic).
- **Diagnostic des sons inactifs (`silent: true` vs canaux inutilisés)** :
  - Un pattern peut apparaître visuellement sur la Playlist tout en étant silencieux (`analyze_loudness(scope="pattern")` -> `silent: true`) si son **insert Mixer** est muté (`mixer 5 Snare`, `mixer 9 Perc`, `mixer 12 Dream Synth` avaient `muted = true`).
  - Toujours vérifier conjointement `list_channels` (`notes == 0` sur `(SXJ) Kick (7)` et `808 [ OZ ]`) et `get_mixer` (`muted == true`) pour distinguer les mutes volontaires des canaux non utilisés.

### 8.4 Adaptation Hard Trap + Planant (`master_hard_trap_final.wav`) & règles manquantes
Conformément à [sp_cifications_th_oriques_et_techniques_harmonie_rythme_et_sound_design_de_la_trap.md](sp_cifications_th_oriques_et_techniques_harmonie_rythme_et_sound_design_de_la_trap.md) :
- **Mode Phrygien & coupure pré-drop** : Utilisation de la progression `Dm -> Eb -> Bb -> A` (`i -> bII -> VI -> v/i`) pour la tension sombre de demi-ton, avec coupure de la 808 sur les 2 derniers temps avant transition (`M2` mesure 8) et clap sec (`0%` réverbe).
- **Piège 808 sous `40 Hz` + `gl.clipper` Master** : Jouer `01. 808 - infinite` à `D1/Eb1` (`36–39 Hz`) devant un `gl.clipper` agressif (`Threshold -3.0`, `Mix 0.70`) sature la bande `20–60 Hz` (`61.94%`) et étouffe les hauts-médiums (`1.06%`). Placer la fondamentale Phrygienne dans la plage **`55–78 Hz`** (`D2=38`, `Eb2=39`, `Bb1=34`, `A1=33`), avec `gl.clipper` dosé (`Threshold -1.5 dB`, `Mix 0.45`, `4x`) suivi de `gl.limiter` (`Gain +5.4 dB`, `Ceiling -2.0 dB`), donne **`-11.58 LUFS`**, **`-1.03 dBTP`**, `0` sample clippé et `sharpness = 1.68 acum`.
- **Hi-Hats Hard Trap & coupe boxy `160 Hz`** : Alternance de vélocité `108/70` + rolls `1/32` pitch-down (`60 -> 58 -> 57 -> 55`) vers la 808 ; coupe `-3.5 dB @ 160 Hz` (`Q = 2.8`) sur `Ambient Pad` (`mixer 11`) pour éliminer la résonance boxy `160 Hz` (`+5.26 dB` -> résolue).








## 9. Production *Stoicville Nights* (`projects/active5`) — Protocole Multi-Échelles & Samples Dérivés
- **Architecture Trap 180 BPM Half-Time & Protocole Multi-Échelles** :
  - Calage exact de l'acapella `90 BPM - T-Pain - Stoicville.mp3` (grille 90 BPM -> 180 BPM half-time, phase `t0 = 0.2067 s`, `bar90 = 2.6667 s`) en segments dédiés (`Intro`, `Hook`, `Verse 1`, `Verse 2`, `Outro`) dans `projects/active5/samples/`.
  - Dérivation des boucles fournies (`Techno Batch Chords E 125.wav`, `Techno Batch Hats 125.wav`) via time-stretch `rate = 0.72`, découpe rythmique staccato (`techno_chords_e_chop.wav` pour Couplet 2), riser inversé (`techno_chords_e_rev.wav` pour les transitions M7/M39/M71) et filtrage HPF `4000 Hz` sur les hats (`techno_hats_180bpm.wav`).
  - **Règle des 3 couches & pivots de fin de phrase** : Snare/Clap verrouillés sur le 3e temps, coupure basse/drums sur le 4e temps des mesures pivots (M8, M32, M40, M56, M72), et alternance `MZ 808 [Apollo]` (couplets) / `MZ 808 [Blxckout]` (refrains) avec sidechain Kick -> `BUS 808` (`gl.comp` `Detector input = "Sidechain"`).
  - **Paramètres Gridline 0.7.7 vérifiés via `set_parameters`** : Les paramètres d'un canal s'adressent via `channel/N/<Param>` (et non `channel/N/plugin/<Param>`), les slots d'effets via `mixer/M/slot/S/<Param>` (index 1-based), `set_send` prend `mixer`, `to`, `level`, et `gl.hall` utilise `Wet level` / `Dry level`.
  - **Contrôle True Peak ISP avec Air Shelf** : Un `High shelf` sur le `Master` après compression génère des crêtes inter-échantillons ; déplacer le shelf `10 kHz (+1.8 dB)` sur `BUS Drums` avant le limiteur Master (`Gain +5.4 dB`, `Ceiling -3.0 dB`, `0.2 dB safety = true`) a permis d'atteindre **`-11.45 LUFS`**, **`-1.18 dBTP`**, `0` sample clippé et `3.16%` d'air (`6–20 kHz`).
  - **Piège `project.playlist.addAudioClip` (`length: 384` par défaut)** : En Lua, `addAudioClip(ch, track, start, offset)` crée le clip avec une longueur par défaut de `384` ticks (`1` mesure), tronquant les voix/boucles longues. Toujours mettre à jour `clip["length"]` en ticks (`bars * 384`) ou déclencher via un pattern dédié sur une piste Playlist séparée (`1` piste Playlist nommée par instrument/son).

---

## 10. Production *Stoicville Ignition* (`projects/active5/projet.gridline`) — Application du Protocole Section 2 (Durées Acoustiques & Enveloppes, 2026-11-21)
- **Synchronisation des enveloppes à `180 BPM` ($T_{\text{noire}} = 333.3\text{ ms}$, $1/16 = 83.3\text{ ms}$, $1/32 = 41.7\text{ ms}$)** :
  - Extraction de one-shots calibrés depuis les boucles de `Untitled.gridline` dans `projects/active5/samples/` : `techno_kick_oneshot.wav` (`125 ms`, HPF `32 Hz`), `techno_clap_oneshot.wav` (`190 ms`, HPF `220 Hz`), `techno_hat_oneshot.wav` (`65 ms`, HPF `3.5 kHz`), `techno_openhat_oneshot.wav` (`180 ms`), ainsi que `techno_key_hats_180bpm.wav`, `techno_perc_180bpm.wav` et `techno_chords_e_high12.wav` (`+12 st`, HPF `650 Hz`).
  - `gl.kick` réglé à `Decay = 0.13 s`, `gl.hat` à `Decay = 0.045 s`, `gl.snare` (`Body 0.11 s / Noise 0.16 s`), `gl.clap` (`0.18 s`), et `Voices = 1` + `One shot = 0` + `Release = 0.045 s` + `Glide = 0.055 s` sur `MZ 808 [Apollo]` (`Fine -14 ct`) et `MZ 808 [Blxckout]` (`Fine +6 ct`).
  - `FX Room` (`gl.reverb`, `Model = 1 Plate`, `Decay time = 0.333 s` = `1 × T_noire`, `Dry = 0`, `Dry (full) = 0`) couplé à une piste d'automation (`add_automation` sur `mixer/20/volume`) qui coupe la queue de réverbe à `0` sur le 4e temps des mesures pivots (`M8, M32, M40, M56, M72`).
- **Dynamique inter-sections écrasée par `Master Comp / Limiter` (`v1 -> final`)** :
  - En `v1`, un seuil `Master Comp` à `-14 dB` et un gain limiteur à `+4.5 dB` écrasaient `Intro` (`-11.73 LUFS`) presque au niveau de `Refrain 1` (`-10.85 LUFS`).
  - En relevant `Master Comp Threshold` à `-10 dB` (`Ratio = 1.8`) et en baissant les gains des couches d'Intro/Pont/Outro (`Chords E Filtered = 0.44`, `Stoicville Intro = 0.62`, vélocités réduites), la dynamique entre sections s'est rouverte (`Intro -14.11 LUFS` / `Pont -14.45 LUFS` -> `Refrains -11.02 à -10.96 LUFS`), tout en supprimant la résonance à `63 Hz` (`-5.6 dB` sur `BUS 808` + `-2.5 dB` sur `BUS Drums` -> `0` résonance 1/3 octave).
- **Zéro faux clash `audit_midi_harmony` sur samplers de boucles** :
  - Régler `Root note = 64` (`E4`) et déclencher les boucles/voix sur `pitch = 64` (au lieu de `C4 = 60` face à `B3` de `Upright Pizz`) supprime tous les faux positifs `minor_2nd_cluster` (`0` clash, `0` note hors gamme `E minor`).
- **Résultat final vérifié (`Stoicville_Ignition_final.wav`, `128.51 s` = `96 mesures @ 180 BPM`)** : **`-11.48 LUFS`**, **`-1.09 dBTP`**, `0` sample clippé, `0` résonance 1/3 octave, `correlation_low_end_150hz = 1.000`, `low_mid_200_800hz_correlation = 0.942`, `sharpness = 1.776 acum`, `air = 6.25%`, `PQ = 7.945`, `CU = 7.585`, `detected_key = "E minor"`.

---

## 11. Résolution Universelle du « Cafouillis » & Refonte Instrumentale *Obsidian Pulse* (`projects/active5/projet.gridline`, 2026-11-22)
- **Diagnostic racine sur retour utilisateur (*« sons pas liés entre eux, brouillon et cafouillis »*)** :
  1. **Dérive micro-tonale cumulée (`45 cents`)** : `Upright Pizz E` (`-31 ct` + `17.1 ms` de silence pré-transitoire), `Techno Chords E` (`-13 ct`), et `Headroom 93` (`+14 ct`), donnant `tuning_a4_hz = 453.15 Hz (+51 ct)` sur le mix global.
  2. **Pick-up dissonant caché à `t = 0` & poly-grooves** : `Techno Batch Chords E 125.wav` démarre par une queue `B / D#` à `0.023 s` avant son accord principal (`0.221 s`, contretemps techno), et les 3 boucles Techno (`Hats`, `Key Hats`, `Perc`) créaient une horloge 4/4 concurrente du *half-time* Trap.
  3. **Absence de grille harmonique unifiée** : un accord statique `Em` affrontait une ligne `808` mobile (`Em -> C -> G -> B`) et deux pianos jouant simultanément sans relais temporel.
- **Solution universelle formalisée dans [docs/harmonisation_verticale_et_horizontale.md](harmonisation_verticale_et_horizontale.md) et appliquée** :
  - Suppression complète de la piste de rappeur (`Stoicville` + `Bankroll`).
  - Extraction de **one-shots propres calés à `t = 0`** et **accordés à `A4 = 440.0 Hz`** (`techno_chord_stab_e_440.wav` Power Chord `E+B`, `techno_pad_e_440.wav`, `techno_chord_riser_440.wav`, `upright_pizz_e3_440.wav` trimmé des `17.1 ms`, `techno_rim_oneshot.wav`, `techno_shaker_oneshot.wav`), tous re-programmés sur l'unique grille `180 BPM`.
  - **Grille d'accords maîtresse unique (`Em -> Em7 -> Cmaj7 -> Am/Bm`)** : les stabs d'accords sont transposés en MIDI et verrouillés sur `Kick + 808` (`beat 0.0 / 1.5`), tandis que `Upright Pizz` et `Headroom Piano` répondent strictement dans les vides (`1.0 / 2.5 / 3.5`) en notes d'accord.
  - **Ciment acoustique** : `FX Room` unique partagé (`0.333 s`, `Dry = 0`, `HPF 420 Hz`, automation de coupure sur M8/M32/M40/M56/M72) + compression glue sur `BUS Drums`, `BUS Music` (`Attack 20 ms`, `Ratio 2.0`) et `Master` + EQ miroir (`-2.5 dB @ 320 Hz`, `-2.2 dB @ 1.25 kHz`).
- **Résultat final vérifié (`Obsidian_Pulse_final.wav`, `128.675 s` = `96 mesures @ 180 BPM`)** : **`-11.54 LUFS`**, **`-1.08 dBTP`**, `0` sample clippé, `0` résonance 1/3 octave, `correlation_full = 0.901`, `correlation_low_end_150hz = 0.999`, `low_mid_200_800hz_correlation = 0.785`, `sharpness = 1.741 acum`, `detected_key = "E minor"` (`0.819`), `high_mids = 30.24%`, `air = 5.13%`, verdict `audio-analyzer` : *« Mix équilibré : loudness, True Peak, spectre et phase dans les cibles. »*

---

## 12. Refonte Top-Down `master_sketch.mid` & Gate 7 Métriques (`cahier_des_charges`, 2026-11-22)

- **Architecture implémentée (Docs, Schéma JSON, MCP `audio-analyzer`, Scripts Lua & Python)** :
  - Ajout du module [mcp-servers/audio-analyzer/musicality_metrics.py](../mcp-servers/audio-analyzer/musicality_metrics.py) (`MusicalityThresholds`, `optimize_voice_leading` <= 2.5 st, `evaluate_midi_musicality` sur 7 métriques) intégré dans `deep_diagnostics.audit_midi_file(..., strict_musicality=True)` et `audio_analyzer_server.audit_midi_harmony`.
  - `_infer_roles` dans `musicality_metrics.py` exclut explicitement les pistes percussives samplées (`_PERC_TRACK_RE` incluant `open[- ]?hat`, `drum`, etc.) afin que des pistes samplées comme `Techno OpenHat OneShot` ou `Kick Native` ne faussent pas `roles["counter"]` et `lead_overlap_conflict`.
  - Schéma [schemas/song_blueprint.schema.json](../schemas/song_blueprint.schema.json) et blueprints (`projects/active`, `projects/active5`) mis à jour avec `harmonic_core` (`chord_progression`, `motif_cell`, `phrase_plan`).
  - Scripts Lua `projects/active/scripts/` harmonisés : [generate_chord_progression.lua](../projects/active/scripts/generate_chord_progression.lua) (optimisation *Smooth Voice Leading* L1), [setup_mix_buses.lua](../projects/active/scripts/setup_mix_buses.lua) (`BUS Music` + sidechain Kick vers `BUS 808` et `BUS Music`), [humanize_all_patterns.lua](../projects/active/scripts/humanize_all_patterns.lua) (vélocité seule sur `Kick`/`808`/`Snare`/`Clap`, micro-timing + vélocité sur hats/mélodiques).

- **Leçons techniques découvertes pendant la refonte *Obsidian Pulse* (`projects/active5`)** :
  1. **Piège `skeleton` sur l'orchestration du `Lead`** : réduire une topline en mode `skeleton` (`st in (0.0, 2.0)`) lorsque la fin de phrase B saute de `>= 5` demi-tons vers le retour en A (`B4 -> E5`) casse la résolution *gap-fill* (`gap_fill_compliance` chutait à `0.364 < 0.60` sur le MIDI orchestré vs `0.667` sur `master_sketch.mid`). Terminer la phrase B sur la septième (`D5 -> E5`, mouvement conjoint `+2 st`) supprime le saut inter-mesure non résolu (`gap_fill_compliance = 1.000`).
  2. **Zéro cluster de seconde/neuvième mineure entre accords et contre-chant** : remplacer `Cmaj7` (où `B3` frottait à `1 st` contre `C4` de `Chords Pad/Stab`) par `Cadd9` et limiter le sustain des accords (`Chords Piano VoiceLed` / `Chords Pad E`) à `2.85` beats pour laisser beats `3.0–4.0` libres à `Counter Pizz` en notes d'accord garantit `minor_2nd_clashes = []`.
  3. **Discontinuités sample 0 & pointes True Peak Nyquist sur Snare/Clap** :
     - `techno_shaker_oneshot.wav` démarrait à `+0.6831` au sample 0, créant une marche verticale à chaque croche et `2+ dB` d'overshoot inter-échantillons True Peak ; un fade-in/out de `1.5–2.5 ms` sur les one-shots dérivés élimine ce défaut à la source.
     - L'empilement simultané à `t = 0` de `Snare Native`, `Clap Native` et `Techno Clap OneShot` sur le temps 3 produisait un saut sample-à-sample (`-0.366 -> +0.749`) à fréquence de Nyquist. L'ajout d'un `gl.eq` LPF `11.5 kHz` sur `Snare (mixer 4)` et `Clap (mixer 5)`, un micro-décalage de poche (`+4 ms` / `-2 ms`), une attaque compresseur de `12 ms` sur `BUS Drums` et un limiteur Master réglé à `Ceiling = -2.50 dB` (`Look-ahead = 6 ms`) ramène le True Peak à **`-1.09 dBTP`**.

- **Comparatif vérifié Avant / Après Refonte (*Obsidian Pulse*, 100 % sons `projects/active5`, sans voix rap)** :
  - **MIDI pré-refonte (`v1`)** : `FAIL` (`lead_overlap_conflict = 0.991`, `stepwise_ratio = 0.613`, `gap_fill_compliance = 0.345`, `melody_rest_ratio = 0.83`).
  - **`master_sketch.mid` (refonte)** : `PASS` (`stepwise_ratio = 0.935`, `gap_fill_compliance = 1.0`, `melody_rest_ratio = 0.335`, `strong_beat_chord_tone_ratio = 1.0`, `voice_leading_avg_semitones = 0.786`, `lead_overlap_conflict = 0.0`, `contrary_or_oblique_ratio = 0.903`).
  - **`obsidian_pulse.mid` orchestré (20 pistes)** : `PASS` (`stepwise_ratio = 0.948`, `gap_fill_compliance = 1.0`, `melody_rest_ratio = 0.462`, `strong_beat_chord_tone_ratio = 0.923`, `voice_leading_avg_semitones = 0.892`, `lead_overlap_conflict = 0.0`, `contrary_or_oblique_ratio = 0.937`, `minor_2nd_clashes = []`, `low_end_simultaneous_collisions = []`).
  - **Rendu final (`Obsidian_Pulse_final.wav`, `128.0 s`, cible `-10.5 LUFS`)** : **`-11.18 LUFS`** (`.mp3` `-11.1 LUFS`), **`-1.09 dBTP`**, `0` sample clippé, `0` résonance 1/3 octave, `correlation_low_end_150hz = 0.998`, `low_mid_200_800hz_correlation = 0.751`, `sharpness = 1.716 acum`, `CE = 6.026`, `CU = 7.688`, `PC = 5.799`, `PQ = 7.536`, recommandation `audio-analyzer` : *« Mix équilibré : loudness, True Peak, spectre et phase dans les cibles. »* Progression dynamique : `Intro -14.12 LUFS`, `Pont -13.31 LUFS`, `Refrains -10.76 à -10.52 LUFS`, `Outro -13.18 LUFS`.

---

## 13. Production *Airey Horizon* (`projects/active6/projet.gridline`, 33 mesures @ 132 BPM = 60,00 s, `G# minor`) — 2026-11-22
- **Analyse harmonique & recalage temporel d'une nappe évolutive (`NUAN SONAR - Airey Progression EMajor.wav`, 58,94 s)** :
  - Mesure par auto-similarité chroma/RMS : la progression réelle tourne en cycles de `~7.27 s` (`4` mesures à `132 BPM` : `Emaj7 -> C#m9 -> Bmaj7`, et cycle final résolvant sur `G#m7`), mais ses attaques lentes dérivent jusqu'à `±0.24 s` sur `58 s`. Toutes ses notes appartiennent exactement à **`G# minor` (`B major` relatif)** (`0` altération conflictuelle).
  - Découpe aux minima RMS et calage exact par bloc de `4` mesures (`7.2727 s`) + `5` mesures d'outro (`9.0909 s`) dans `projects/active6/samples/` (`nuan_4bar_a_132.wav`, `nuan_4bar_b_132.wav`, `nuan_5bar_outro_132.wav`, `nuan_4bar_high12_132.wav` `+12 st`, `nuan_riser_1bar_132.wav` inversé) -> **`0,00 ms`** de dérive sur 60 s.
- **Pièges d'accordage 808 & nomenclature `_infer_roles` dans `audit_midi_harmony`** :
  - `(SXJ) 808 (5)` présentait une partielle inharmonique à `81 Hz` (`+372 ct`) et tirait le Refrain 2 à `-27 ct`, alors que `(SXJ) 808 (4)` (`Root 26`, `Fine -14 ct`, DC retiré) suit une série harmonique pure (`37.01 / 74.02 / 111.03 Hz`) mesurée à **`+0.0 à +3.0 ct`** sur Intro/Refrain/Couplet/Outro.
  - Ne jamais utiliser `librosa.effects.pitch_shift` sur une 808 sub (peut décaler le pitch de `-35 ct` et ramollir l'attaque) : conserver le WAV DC-nettoyé et régler `Root note` + `Fine tune` nativement dans `gl.sampler`.
  - Dans `musicality_metrics.py` et `deep_diagnostics.py`, `_PERC_TRACK_RE` reconnaît `hi-hat`, `open-hat`, `perc`, `kick`, `snare`, `clap` (mais pas `hh` ni `percs`), et `_infer_roles` classe `"string"` dans `chords` avant `lead`. Nommer les pistes percussives `Hi-Hat ...` / `Perc ...` et le lead `gl.twinstring` en `Lead Pluck (Twin)` garantit une inférence exacte (`Full Orchestrated MIDI Audit`: `PASS`, `stepwise_ratio = 1.0`, `gap_fill = 1.0`, `rest_ratio = 0.42`, `strong_beat_chord_tone = 0.931`, `voice_leading = 0.591 st`, `lead_overlap = 0.0`, `contrary_or_oblique = 0.953`, `0` clash).
- **Nettoyage Nyquist & équilibre spectral (`mix_v1 -> master_final`)** :
  - `06. hh - bat wing yaiba.wav`, `05. hh - bankroll.wav` et `(SXJ) Percs (8).wav` contenaient des sauts impulsionnels sample-à-sample (`1.00–1.21`) générateurs de micro-clics et d'overshoots True Peak. Un lissage `LPF 11.5–12.5 kHz` + fade-in `2 ms` dans `samples/` ramène `micro_glitch_or_crackle_events` à **`0`**.
  - `NUAN SONAR` concentrait `49,7 %` de son énergie entre `150–300 Hz` (`0,02 %` > `4 kHz`) : une coupe `-5.0 dB @ 245 Hz` + `HPF 185 Hz` sur `mixer 8/9` + présence `BUS Music +2.8 dB @ 1.65 kHz` a porté `high_mids` de `3.92 %` (`v1`) à `15.09 %` (`master_final.wav`).
  - **Rendu final vérifié (`master_final.wav`, `60.00 s`)** : **`-11.83 LUFS`** (`.mp3` `-11.8 LUFS`), **`-1.50 dBTP`**, `0` sample clippé, `0` micro-glitch, `sub = 10.39%`, `low_mids = 21.41%`, `air = 3.87%`, `correlation_low_end_150hz = 0.998`, `low_mid_200_800hz_correlation = 0.873`, `sharpness = 1.608 acum`, `CE = 5.887`, `CU = 7.385`, `PC = 5.791`, `PQ = 7.120`, recommandation `audio-analyzer` : *« Mix équilibré : loudness, True Peak, spectre et phase dans les cibles. »* Arc dynamique : `Intro -15.73 LUFS`, `Refrain 1 -10.37 LUFS`, `Couplet -11.16 LUFS`, `Pont -14.41 LUFS`, `Refrain 2 Climax -10.47 LUFS`, `Outro -18.07 LUFS`.

---

## 14. Post-Mortem Workflow & Optimisations Fiabilité / Tokens (2026-11-22)
Analyse des goulots d'étranglement et faux positifs observés durant la production de `projects/active6` (*Airey Horizon*), avec correctifs appliqués dans le code et la documentation :

1. **Bug de mesure du diapason (`+41 ct` fantôme) dans `mcp-servers/audio-analyzer/deep_diagnostics.py`** :
   - *Problème* : `_compute_psychoacoustics` appelait `es.TuningFrequency()` sur un unique bloc de `1,48 s` (`65 536` samples) incluant transitoires de kick/hi-hats, affichant **`450.54 Hz (+41.0 ct)`** alors que chaque section musicale mesurée par CQT était entre **`+0.0` et `+5.0 ct`**.
   - *Solution* : Calculer `tuning_a4_hz` et `tuning_cents_offset` sur `librosa.effects.harmonic(ex44)` via `librosa.estimate_tuning` (`12 s` harmoniques filtrées). Vérifié sur `master_final.wav` : **`440.51 Hz (+2.0 ct)`**.
2. **Faux positifs `_PERC_TRACK_RE` & `_infer_roles` dans `musicality_metrics.py` / `deep_diagnostics.py`** :
   - *Problème* : Des pistes nommées `HH Bankroll`, `Percs (8)`, `FX (4)_2`, `Chants (6)` échappaient à `_PERC_TRACK_RE`, et `"Twin String Lead"` était classé dans `"chords"` parce que `"string"` était testé avant `"lead"`.
   - *Solution* : Extension unifiée de `_PERC_TRACK_RE` (`hh`, `hats?`, `percs?`, `fx`, `sfx`, `riser`, `zap`, `triangle`, `chant`) partagée entre les deux modules, et priorité des mots-clés `lead/melody/topline/vox/vocal` avant `string` dans `_infer_roles`.
3. **Coût en tokens/tours des explorations manuelles et des 6 rendus complets (`v1 -> v6`)** :
   - *Problèmes* : (a) `5` scripts Python ad-hoc pour auditer `newtest.gridline`, les 808 et `NUAN SONAR` ; (b) détection tardive (en `v2` et `v5`) des sauts Nyquist (`1.0–1.21`) sur les hi-hats/`Perc 8`, du partiel inharmonique (`-322.8 ct`) sur `808 (5)`, et de la dérive `-35 ct` de `librosa.effects.pitch_shift` sur sub-bass ; (c) `gridline render` complet (`60 s`) générant ~90 Ko de logs `\r rendering XX%` et bloquant plusieurs tours d'attente ; (d) duplication de `GridlineMCP` et du setup des bus entre `build_active5.py` et `build_active6.py`.
   - *Solutions déployées* :
     - **`scripts/audit_project_sources.py`** : pré-audit en 1 appel (harmoniques `1*f0/2*f0/3*f0`, cycles/BPM/chroma des boucles, sauts Nyquist, DC, énergie `150–300 Hz`, + `--prepare-samples`).
     - **`scripts/studio_lib.py`** : module partagé (`GridlineMCP`, `clean_audio_sample`, `verify_midi_gate`, `setup_standard_buses` pré-calibré `Ceiling = -2.35 dB` + `BUS Music` + `FX Room`).
     - **`scripts/render_and_audit.py`** : rendu silencieux (0 spam `\r`), support `--range BAR1-BAR2` (~4× plus rapide pour les itérations EQ/mixage) et sortie JSON unique (`analyze_mix` + `audit_mix_deep` + `section_lufs` + `max_sample_jump`).
4. **Clarification documentaire (`GEMINI.md`, `ARCHITECTURE_AGENTS.md`, `blueprint_help.md`)** :
   - Préciser explicitement que les synthés déjà inclus dans le `.gridline` initial de l'utilisateur (ex. `gl.twinstring`, `gl.synth`) sont autorisés (seul l'ajout de synthés **non présents** reste interdit).
   - Référencer les 3 scripts partagés pour réduire de ~60–70 % le nombre d'appels d'outils sur les futurs morceaux.


---

## 15. Production *Midnight Drift* (`projects/active7/projet.gridline`, 28 mesures @ 150 BPM = 45,13 s, `Bb minor`) — 2026-11-22
- **Format court 45 secondes rythmé et planant à partir de `Untitled.gridline`** :
  - Analyse de `BFNK - Notes_01.wav` (`16.000 s` @ `120 BPM` exact, `-7 ct`, `75,7 %` d'énergie en `150–300 Hz`, progression `Bbm7 -> Absus2 -> Gbmaj7 -> Gbmaj7/F`) : time-stretch `rate = 1.25` (`12.800 s` = 8 mesures @ `150 BPM`) + recalage `+7 ct` -> `440.0 Hz` exact (`+0.0 ct`).
  - Dérivés sauvegardés dans `projects/active7/samples/` : `bfnk_full_150.wav`, `bfnk_a_150.wav`, `bfnk_b_150.wav`, `bfnk_chop_150.wav` (gate rythmique à la noire pour le Pont), `bfnk_high12_150.wav` (`+12 st` shimmer Climax), `radiate_riser_150.wav` (riser inversé 1 mesure), `vox_virtual_bb4.wav` (`-210 ct`), `vox_reused_f5.wav` (`-30 ct`), `nm_-_perc_14_tuned.wav` (`+40 ct`), `fx_4_3_tuned.wav` (`-55 ct`), et nettoyage LPF `14 kHz` des hats/cymbales (`0` micro-glitch).
- **Piège corrélation `200–800 Hz` sur boucle stéréo large (`BFNK - Notes_01.wav`)** :
  - *Problème* : `BFNK - Notes_01.wav` présentait une corrélation `200–800 Hz` de **`-0,014`** (élargissement stéréo natif extrême), tirant le premier mix de contrôle (`test_ref1.wav`) à `low_mid_corr = 0.177`.
  - *Solution* : Traitement M/S doux (`< 650 Hz`, atténuation du canal Side à `0.18` tout en conservant `0.90` au-dessus de `650 Hz`) dans les dérivés `bfnk_*_150.wav`. Corrélation `200–800 Hz` portée à `0.910` sur le stem et **`0.707`** sur le master final.
- **Piège d'écrasement dynamique sur `Intro` avec `gl.synth` + `BUS Music Comp`** :
  - *Problème* : Avec `BUS Music gl.comp` à `Threshold = -19.5 dB / Makeup +3.6 dB` et `gl.synth` peu sensible à la vélocité, l'`Intro` (`-10.69 LUFS`) était presque au même niveau que `Refrain 1` (`-10.33 LUFS`).
  - *Solution* : Relever `BUS Music gl.comp Threshold` à `-14.0 dB` (`Makeup +1.4 dB`), `Master Comp Threshold` à `-9.5 dB` (`Makeup +1.4 dB`), et automatiser le fader de `Lead Synth (mixer 12)` (`0.54` Intro -> `0.96` Refrain 1 -> `0.56` Pont -> `1.00` Climax -> `0.52` Outro), rétablissant un arc réel de **`3,2 à 3,9 dB`** (`Intro -12.83 LUFS`, `Refrain1 -9.66 LUFS`, `Pont -13.44 LUFS`, `Climax -9.52 LUFS`, `Outro -16.39 LUFS`).
- **Rendu final vérifié (`master_final.wav`, `45.13 s`)** : **`-10.71 LUFS`** (`.mp3` généré 320 kbps), **`-1.40 dBTP`**, `0` sample clippé, `0` micro-glitch, `tuning = 440.0 Hz (0.0 ct)`, `correlation_low_end_150hz = 0.999`, `low_mid_200_800hz_correlation = 0.707`, `air = 4.01%`, `sharpness = 1.678 acum`, `CE = 6.720`, `CU = 7.908`, **`PQ = 8.078`**, sondes CLAP toutes saines, verdict `audio-analyzer` : *« Mix équilibré : loudness, True Peak, spectre et phase dans les cibles. »*


---

## 16. Production *Simba d'Ea'alah* (`projects/active8/projet.gridline`, 64 mesures @ 144 BPM / 72 BPM ressenti = 107,15 s, `E minor`) — 2026-11-22
- **Reverse-engineering d'un morceau MP3 (`Ea'alah (Family).mp3`, `240.9 s`, `166.97 BPM` / boucle `5.744 s`, `G major / E minor`) en Cloud Rap inspiré de PNL (*Simba*)** :
  - **Timeout MCP `split_stems` sur titre long (> 3 min)** : `reverse-engineer:split_stems` a atteint le timeout RPC de 3 min sur un MP3 de `4 min`, alors que le processus enfant `gridline stems --model htdemucs_6s` continuait en arrière-plan (~5 min au total sur CPU) et a produit les 6 stems intacts (`Bass`, `Drums`, `Guitar`, `Other`, `Piano`, `Vocals`). Pour les morceaux `>= 3 min`, lancer `gridline stems` via commande asynchrone ou surveiller sa complétion plutôt que relancer un second `split_stems`.
  - **Extraction 100 % issue des stems de `Ea'alah (Family).mp3` (zéro synthé ajouté)** :
    - `Other` (`102–112 s`, `Guitar=0`, `G3` isolé à `111 s` `197.76 Hz`) -> `eaalah_pad_loop.wav` (nappe sous-marine `LPF 2.8 kHz`, stabilisée M/S `< 650 Hz`) + `eaalah_pad_single.wav` (instrument polyphonique d'accords `Em7 -> Cadd9 -> Am7 -> Bm7` avec `Root=55, Fine=-16 ct`).
    - `Guitar` (`1.89–13.38 s` sans batterie/basse + note isolée `B3` à `230.0 s` `250.02 Hz`) -> `eaalah_guitar_loop.wav` / `_b.wav` (time-stretch `rate = 0.8624` vers `144 BPM`, `-4 ct`), `eaalah_guitar_shimmer.wav` (`+12 st` Climax), `eaalah_guitar_pluck.wav` (`Lead A-A'-A''-B`, `Root=59, Fine=-21 ct`).
    - `Vocals` (`0.13–5.87 s` & `70.40–76.14 s` sans batterie + note `C5` à `5.72 s` `523.43 Hz`) -> `eaalah_vox_cloud_a/b.wav` (nuages spectraux `360–4200 Hz`) et `eaalah_vox_note_c5.wav` (contre-harmonies placées exclusivement dans les silences du Lead).
  - **Index `audio-index` vs dossier `soundbanks/`** : Les chemins indexés dans `samples.db` pointant vers `.../samples/<pack>/...` résident physiquement sous `.../soundbanks/<pack>/...`. Penser à vérifier l'existence sur disque (`Deep 808.wav` `33.49 Hz = C1 +41.3 ct`, `Kick Drum 001 Tight.wav`, `Soft Clap.wav`, `Andromeda Snare.wav`, `Crisp hi-hat.wav`, `Chrome Open Hat.wav`).
  - **Évitement des faux positifs d'audit MIDI sur les déclencheurs de boucles & accords `Cmaj7` (`B3` vs `C4/C5`)** :
    - Nommer les canaux déclencheurs de boucles longues avec le mot isolé `FX` (`Ea'alah Pad Bed FX`, `Ea'alah Guitar Loop A FX`, `Ea'alah Vocal Cloud A FX`) afin que `_PERC_TRACK_RE` ne prenne pas leurs notes de déclenchement tenues pour des notes d'accords/mélodie, et préférer `Cadd9` (`[52, 55, 60, 62]`) à `Cmaj7` (`B3+C4`) sous un contre-chant `C5`. Résultat sur `simba_eaalah.mid` (`64` mesures) : **`PASS`**, `stepwise = 1.0`, `gap_fill = 1.0`, `rest_ratio = 0.321`, `voice_leading = 0.798 st`, `overlap = 0.0`, `contrary_oblique = 1.0`, `0` clash.
  - **Choix de hi-hat pour l'air Cloud Rap sans pic `8 kHz`** : `Cloud hi-hat.wav` (`45 ms`, seulement `4.2 %` > `10 kHz`) créait un pic à `8 kHz (+5.57 dB)` sous shelf ; le remplacer par `Crisp hi-hat.wav` (`69.3 %` > `10 kHz`, coupé à `95 ms` et `LPF 11.8 kHz`, `max_diff = 0.443`) a supprimé la résonance `8 kHz`.
- **Rendu final vérifié (`master_final.wav` & `Simba d'Ea'alah.mp3`, `107.15 s`)** : **`-11.43 LUFS`** (`Intro -15.45`, `Couplet1 -11.23`, `Refrain1 -9.98`, `Pont -14.51`, `Climax -9.89`, `Outro -18.41`), **`-1.82 dBTP`**, `0` sample clippé, `0` micro-glitch, `tuning = 440.0 Hz (0.0 ct)`, `correlation_low_end_150hz = 1.0`, `low_mid_200_800hz_correlation = 0.910`, `sharpness = 1.477 acum`, `CE = 6.69`, `CU = 7.21`, `PQ = 7.457`, toutes les 4 sondes CLAP saines (`shrill_vs_warm = -0.215`).

---

## 17. Extension 3 Minutes & Enrichissement *Glass Horizon* (`projects/active9/projet.gridline`, 129 mesures @ 172 BPM = 180,00 s, `F minor`) — 2026-11-22
- **Respect des choix stylistiques utilisateur dans un `.gridline` modifié à la main** :
  - Avant toute refonte, inspecter les IDs canaux/clips et volumes mixeur : l'utilisateur avait retiré `Lead Synth`, `Noisehats Air FX` et `Glass Shimmer`, remonté la `Deep 808` / `Hi-Hat` / `Tech Snare`, et laissé 3 clips audio sur les pistes 16–18 (`Chant Misc 003 Hi.wav`, `HEADROOM PIANO LEVEL1 CLOSE 99.flac`, `vocal-3.ogg`).
- **Traitement des 3 samples fournis (`A4 = 440 Hz`, anti-aliasing, registre & harmonie)** :
  - `HEADROOM PIANO LEVEL1 CLOSE 99.flac` présentait un *stretch tuning* natif de **`+30.4 ct`** (`2533.18 Hz` vs `Eb7 = 2489.02 Hz`) et un niveau `pp` très faible (`pk = 0.056`) : dérivé `headroom_99_eb7_clean.wav` recalé `-30.4 ct`, normalisé `0.78`, couplé à `Headroom Piano Clean.sfz` (`gl.multisampler`, `Polyphony = 16`, `Cutoff = 6200 Hz`) pour remplacer `Lead Synth` de manière organique.
  - `Chant Misc 003 Hi.wav` (`220 Hz` = `A3`, dissonant à `1 st` de `Ab` en `F minor`, et `max_jump = 0.49`) : dérivés `chant_hi_ab3.wav` (`-1 st`) et `chant_hi_c4.wav` (`+3 st`) filtrés (`HPF 240 Hz`, `LPF 8.5 kHz`).
  - `vocal-3.ogg` (`C4 -> G4`, `391.7 Hz = -1.3 ct`, 2e harmonique dominante à **`783.8 Hz`**) : normalisation à `0.80` sans atténuation avait provoqué un pic de **`+16.59 dB` à `800 Hz`** dans `Break 1` (`-9.98 LUFS` vs `-10.44 LUFS` sur Drop!). Correction vérifiée : volume `mixer 12 = 0.64` + coupe EQ `-7.2 dB @ 784 Hz (Q = 1.6)` + `HPF 230 Hz`.
- **Collision `_infer_roles` (`musicality_metrics.py`) & micro-glitches `gl.hat` + limiteur `5 ms`** :
  - Nommer la contre-mélodie vocale `"Vocal 3 Counter"` déclenchait le mot-clé `"vocal"` de `_infer_roles` et fusionnait sa ligne avec `Headroom Piano Lead` (`gap_fill_compliance = 0.571`). Renommer la piste MIDI en `"Counter Harmony (Voice3)"` a rétabli la séparation `lead / counter` (`gap_fill = 0.955`, `lead_overlap_conflict = 0.0`).
  - En `Break 1`, `gl.hat` (`Tone = 12.2 kHz`) + shelf `BUS Drums +6.2 dB @ 8.2 kHz` + `gl.limiter Attack (look-ahead) = 0.005 s` produisait `7` micro-glitches (`max_jump = 1.136`, `-0.62 dBTP`). Passer `gl.hat Tone = 10.5 kHz`, `BUS Drums Band 7 LPF = 14.2 kHz`, et `gl.limiter Attack = 0.002 s` (`Ceiling = -3.05 dB`) a ramené `micro_glitches` à **`0`** et `true_peak_dbtp` à **`-1.13 dBTP`**.
- **Rendu final vérifié (`master_final.wav` & `Glass_Horizon_3min.mp3`, `180.92 s`)** : **`-12.12 LUFS`** (`Intro -14.41`, `Drops -10.98 à -11.05`, `Breaks -13.06 à -13.71`, `Outro -15.80`), **`-1.13 dBTP`**, `0` sample clippé, `0` micro-glitch, `tuning = 440.25 Hz (+1.0 ct)`, `correlation_low_end_150hz = 1.0`, `low_mid_corr = 0.848`, `air = 3.58%`, `sharpness = 1.68 acum`, `CE = 6.923`, `CU = 7.403`, `PQ = 7.234`, toutes les 4 sondes CLAP saines, verdict `audio-analyzer` : *« Mix équilibré : loudness, True Peak, spectre et phase dans les cibles. »*




---

## 18. Production *Le Serpent de Pigalle (Attaque)* (Captaine Roshi, `68 mesures @ 136 BPM`, `E Phrygian`) & Pipeline Psychoacoustique + APIs en Ligne — 2026-11-22
- **Piège critique `gl.eq` (`Band N type = 4` vs `type = 6`)** :
  - Dans `gl.eq`, l'index `4` correspond à **`Notch`** (coupe infinie $-\infty\text{ dB}$) et non à `Peak` (`type = 6`) — l'ordre exact des choix est `['Off', 'Low cut', 'Band pass', 'High cut', 'Notch', 'Low shelf', 'Peak', 'High shelf']`. Utiliser `type = 4` sur `7.8 kHz` ou `1.8 kHz` creuse un trou total dans le spectre ; passer à `type = 6` (`Peak`) rétablit une cloche douce et maîtrisée.
- **Choix du hi-hat pour éviter la résonance `8 kHz` sur Trap rapide** :
  - `hh_batwing_clean.wav` concentre `82.7 %` de son énergie entre `6–10 kHz` (pic à `7.4 kHz`) et provoque une résonance locale `8 kHz` sous shelf. `hh_bankroll_clean.wav` (`52.2 %` dans `9–16 kHz` contre `42.1 %` dans `6–9 kHz`, filtré LPF `14.2 kHz` dans `hh_roshi_clean.wav`) élimine totalement la résonance `8 kHz` (`resonances_third_oct = []`) avec `PQ = 7.562` et `CE = 6.462`.
- **Intégration du Pipeline Psychoacoustique (`scripts/audio_metrics.py`, `index_samples.py`, `build_references.py`, `select_sample.py`) & APIs en Ligne (`fetch_samples_api.py` + `audio-index` MCP)** :
  - Entonnoir à 4 couches : Hard Filter (clipping consécutif, clics bornes `needs_fade`, silence, corrélation $\rho_{LR}$) → Confort Auditif ($R_{\text{harsh}}$ `2–5 kHz`, pic résonant, rugosité Plomp-Levelt) → Profil de rôle (`kick`, `bass_sub`, `snare_clap`, `hihat_perc`, `lead_melody`, `pad_atmosphere`, `chord_keys`) → LAION-CLAP Zero-Shot & `golden_references/` + Masquage Bhattacharyya sur 24 bandes Bark.
  - Client `fetch_samples_api.py` (Freesound API v2 + Lots of Sounds API, clés dans `.env` / `.env.example`) : en mode démo public `/api/v1/sounds/sample`, Lots of Sounds renvoie un champ relatif `stream_url` qui retourne du JSON contenant l'URL signée CDN ; `ffmpeg` nécessite `-c:a pcm_s24le` (et non `-sample_fmt s24`) pour exporter en WAV 24-bit avec `loudnorm`.
- **Refonte psychoacoustique des sonorités de `roshi_serpent.gridline` (avant/après)** :
  - Audit des anciens samples (`v12`) : 5 pistes étaient **`rejected`** (`riser_1bar_136.wav` `0.071` hors phase `rho=-0.06`, `chant_roshi.wav` `0.138`, `bell_bb6_440.wav` `0.363`, `pluck_b3_440.wav` `0.426`, `vox_c5_440.wav` `0.550`), `Snare` était rêche (`snare_roshi_clean.wav` `0.567`, `r_harsh=0.73`), et `Chords` utilisait une onde scie brute `gl.synth`.
  - **Nouveau kit 100 % `excellent` (`0.788..0.950`)** + piano acoustique `Headroom Piano Clean.sfz` (`gl.multisampler`, `Polyphony = 12`, `LPF 4.6 kHz`) :
    - `Kick` -> `kick_tight_roshi.wav` (`0.859`), `Snare` -> `snare_crisp_roshi.wav` (`0.930`, `r_harsh=0.04`), `Clap` -> `clap_soft_roshi.wav` (`0.950`), `Hi-Hat` -> `hh_ctd_roshi.wav` (`0.923`, EQ `-5.5 dB @ 6.4 kHz`), `Open Hat` -> `oh_chrome_roshi.wav` (`0.815`), `Perc` -> `perc_6ix4our_roshi.wav` (`0.935`), `Crash FX` -> `crash_deccer_roshi.wav` (`0.809`), `Riser FX` -> `riser_glass_roshi.wav` (`0.878`, `Root=63` -> `F Maj` cadence `bII` vers `Em`), `Chant FX` -> `chant_sxj5_roshi.wav` (`0.843`), `808 Pluggg` -> `808_deep_roshi.wav` (`0.927`, recalé à `C1 = 32.70 Hz`), `808 V12` -> `808_v12_roshi.wav` (`0.942`, harmonique 3 à `98.17 Hz`), `Pluck Lead` -> `lead_gtr_roshi.wav` (`0.791`, tranche mono-note `G4` de `12 - Lead guitar.wav`), `Vox Choir` -> `vox_choir_roshi.wav` (`0.873`, `vocal3_sustain_g4.wav`), `Bell Shimmer` -> `counter_bell_roshi.wav` (`0.788`, tranche mono-note `D#5` de `2 - Guitar.wav`).
  - **Vérification sur les tranches mono-note & harmoniques** : Toujours croiser `YIN` + spectre FFT avant de régler `Root note` d'un `gl.sampler` (`808_v12` avait une harmonique 3 à `98.17 Hz = G2`, `vocal3_sustain_g4` une harmonique 2 dominante à `783.5 Hz = G5`) et vérifier qu'une tranche extraite d'un stem (`12 - Lead guitar`, `2 - Guitar`) ne contient qu'une seule hauteur fondamentale (contre `16 - Bells` / `17 - Bells` qui jouaient des accords complets).
  - **Rendu final vérifié (`master_final.wav`, `120.679 s`)** : MIDI gate `PASS` (`stepwise=1.0`, `gap_fill=1.0`, `voice_leading=0.826 st`, `0` clash), **`-11.48 LUFS`** (`Intro -13.25` -> `Refrain1 -10.90` -> `Climax -10.87` -> `Outro -13.39`), **`-1.38 dBTP`**, `0` sample clippé, `0` micro-glitch, `0` résonance 1/3 octave, `tuning = 439.75 Hz (-1.0 ct)`, `correlation_low_end_150hz = 1.0`, `low_mid_corr = 0.946`, `sharpness = 1.615 acum`, `CE = 6.471`, `CU = 7.501`, `PQ = 7.388`, toutes les 4 sondes CLAP négatives (`defect_margin` jusqu'à `-0.2658`).
- **Diagnostic et résolution du bug récurrent « Synth Chords / Dark Piano Chords ne fonctionne pas dans Gridline GUI »** :
  - **Cause 1 (`patch` `mixerTrack` = `id` interne et non index mixeur)** : Dans le fichier `backup 15-57-01` (`v12`), `patch("channel/1", {"name": "Synth Chords", "mixerTrack": 12})` a routé le canal vers l'objet mixeur d'`id = 12`, soit **`mixer 10` (`808 Pluggg` -> `BUS 808` mono `Width = 0` + sidechain `-22 dB / 5:1`)** au lieu de `mixer 12` (`id = 14`), ou sur un `id` inexistant (silence) si des pistes ont été supprimées. **Règle** : toujours router via `edit_channel(channel=N, mixer=M)` et ne **jamais** écrire `mixerTrack` via `patch`.
  - **Cause 2 (`gl.multisampler` + `.sfz` externe muet en GUI `gridline-app --attach` vs rendu headless)** : Inspection directe sur `gridline-app` en cours d'exécution via `gridline mcp --attach` (`task-784`) : `Headroom Piano Clean.sfz` dans `gl.multisampler` retournait **`-144 LUFS` (`silent: true`)** sur `pat 1` et `pat 2` dans `gridline-app`, alors que les mêmes motifs rendaient `-21.42` / `-14.99 LUFS` en CLI headless et que `Pluck Lead` (`gl.sampler`) sortait à `-9.49 LUFS` dans `gridline-app` ! De plus, sous `vel < 60`, la couche `LEVEL1` du SFZ cumulée à `Velocity = 0.70` tombait à `-45.01 LUFS`.
  - **Cause 3 (Seuil de vérification `silent: false` insuffisant)** : Même avec `gl.synth`, des accords sous-mixés (`active6 = -29.22 LUFS`, `active7 = -28.08 LUFS`) passaient `silent: false` tout en étant inaudibles.
  - **Solution définitive 100 % compatible GUI + CLI (appliquée & documentée dans `.rules` et `ARCHITECTURE_AGENTS.md`)** : convertir la note acoustique (`HEADROOM PIANO LEVEL4 CLOSE 57.flac` -> `piano_dark_a3_roshi.wav`, calibrée `A4 = 439.75 Hz`, `Score = 0.812 [excellent]`) et la charger dans **`gl.sampler`** (`Voices = 12`, `Root note = 57`, `One shot = 0`), avec contrôle systématique `analyze_loudness(scope="pattern")` **`lufs >= -22.0`** (ici `-15.41` à `-12.94 LUFS`).


