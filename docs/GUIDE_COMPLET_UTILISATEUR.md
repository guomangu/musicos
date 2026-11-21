# Guide Complet Utilisateur — Composer, Mixer & Flipper des Classics VGM en Rap avec l'IA

Bienvenue dans ton studio musical assisté par intelligence artificielle. Ce tutoriel pas à pas t'explique **comment tout fonctionne ensemble** et **comment passer d'un simple prompt (ou d'une musique de jeu vidéo) à un morceau complet prêt à exporter** dans Gridline.

---

## Chapitre 1 — Comprendre l'architecture en 2 minutes

Quand tu écris un prompt dans **Antigravity** ou **Zed**, l'IA n'invente pas de l'audio à l'aveugle : elle pilote un vrai studio numérique grâce à **5 outils spécialisés (serveurs MCP)** :

1. **`gridline` (le DAW)** : crée les pistes, programme les notes MIDI et scripts Lua, charge tes samples et effets, puis exporte l'audio. Chaque modification de l'IA crée une étape `AI: ...` dans l'historique `Ctrl+Z` de Gridline — tu ne perds jamais ton travail manuel.
2. **`audio-index` (la mémoire de tes samples)** : a analysé tes packs dans `pack/` (1 898 fichiers audio/MIDI dont 1 682 avec vecteurs **LAION CLAP** et analyse **Essentia**). L'IA peut chercher par ambiance (*« kick sec punchy »*, *« guitare lofi sombre »*), BPM ou tonalité.
3. **`reverse-engineer` (le laboratoire VGM → Rap)** : sépare n'importe quel morceau en 4 ou 6 pistes isolées (`drums`, `bass`, `other`, `vocals`, `guitar`, `piano`) avec `htdemucs_6s`, puis convertit les mélodies et accords audio en vraies notes **MIDI** grâce au modèle **Spotify Basic Pitch (ONNX)**.
4. **`audio-analyzer` (les oreilles mathématiques)** : écoute les rendus WAV exportés et mesure les **LUFS** (volume perçu), le **True Peak** (saturation), la répartition sur 5 bandes de fréquences et les conflits Kick/808.
5. **`vocaloid` (la voix et le chant)** : découpe tes paroles en syllabes, écrit des partitions **OpenUTAU / DiffSinger** (`.ustx`) et synthétise un guide vocal `.wav` calé sur ton morceau.

---

## Chapitre 2 — Lancer une session de travail

### Mode Live (voir Gridline se construire sous tes yeux — recommandé)
1. Lance Gridline sur le projet actif avec :
   ```bash
   /home/gamo/Bureau/musicos/scripts/launch-gridline.sh
   ```
   *(Cela ouvre `projects/active/projet.gridline` avec le serveur MCP local actif sur le port `47390`.)*
2. Ouvre ensuite `/home/gamo/Bureau/musicos` dans **Antigravity** ou **Zed**.
   - Grâce au script intelligent `scripts/gridline-mcp.sh`, si Gridline est déjà ouvert au lancement de ton éditeur, l'IA s'attache **en direct** à la fenêtre Gridline ouverte : tu vois apparaître les canaux, notes et plugins en temps réel et tu peux appuyer sur `Espace` pour écouter immédiatement.
   - Si tu as ouvert Zed/Antigravity **avant** Gridline, l'IA modifie le fichier `projects/active/projet.gridline` sur disque (il suffit ensuite de l'ouvrir dans Gridline ou de recharger les serveurs MCP).

### Raccourcis utiles dans Gridline
- **Annuler l'IA (`Ctrl+Z`)** : chaque commande MCP crée une étape nommée `AI: ...`. Tu peux revenir en arrière proprement à tout moment.
- **Scripts Lua personnels (`Tools > Composing > Project scripts`)** : 3 scripts prêts à l'emploi ont été liés directement dans Gridline :
  - `generate_chord_progression.lua` : génère une suite d'accords diatoniques (`Minor`, `Major`, `Dorian`, `Phrygian`).
  - `humanize_all_patterns.lua` : humanise légèrement le timing et les vélocités de tous tes patterns.
  - `setup_mix_buses.lua` : route tes pistes proprement vers `BUS Drums` (10), `BUS Music` (11) et `BUS Vocal` (12).

---

## Chapitre 3 — Créer un morceau de zéro par prompt

Par défaut, l'IA suit un workflow strict défini dans `.rules` et dans la documentation théorique ([règles de mélodie & cohésion inter-pistes](regles_melodie_et_cohesion_inter_pistes.md), [protocole multi-échelles & durées acoustiques](protocole_universel_d_architecture_musicale_coh_rence_multi_chelles_mod_le_pr_dictif_orientation_trap.md), [harmonisation inter-pistes verticale & horizontale](harmonisation_verticale_et_horizontale.md) et [spécifications harmoniques/rythmiques Trap](sp_cifications_th_oriques_et_techniques_harmonie_rythme_et_sound_design_de_la_trap.md)) : **Plan (`song_blueprint.json` + `harmonic_core`) → Réduction `master_sketch.mid` + Gate Musicalité (`PASS`) → Orchestration & Ciment Timbral `BUS Music` → Auto-Mixage**.

### Étape 1 : Demander un morceau
Copie-colle par exemple un de ces prompts dans Antigravity ou Zed :

> **Exemple Trap sombre :**
> « Prépare un blueprint puis génère une instru Trap sombre à 142 BPM en Do mineur (24 mesures : Intro 4, Refrain 8, Couplet 8, Outro 4). Utilise un kick punchy et une 808 accordée depuis ma banque de samples, des hi-hats rapides avec rolls en Lua, un pad sombre sur Surge XT et une mélodie de cloches sur Lattice (`gl.matrixfm`). Fais l'auto-mixage à -11 LUFS. »

> **Exemple Boom-Bap / Lo-Fi avec ta banque :**
> « Cherche une boucle de guitare lofi autour de 80-88 BPM dans ma banque de samples, adapte le projet à son tempo et sa tonalité, ajoute un kick et une caisse claire Boom-Bap humanisés avec du groove, une basse chaude analogique et passe le tout dans CHOW Tape Model pour un rendu vinyle à -14 LUFS. »

### Étape 2 : Valider le Blueprint
L'IA rédige d'abord `projects/active/song_blueprint.json` et te le résume (BPM, gamme, découpage par mesures, répartition des fréquences).
- Si tu veux ajuster un détail (*« passe plutôt à 136 BPM »* ou *« remplace le pad par un piano »*), dis-le avant qu'elle ne construise les pistes.
- Si tu veux qu'elle génère d'un seul coup sans te demander validation, ajoute simplement *« génère tout directement sans attendre ma validation »* dans ton prompt.

### Étape 3 : Écouter et itérer
Tu peux ensuite lui demander des retouches ciblées comme à un vrai ingé son :
- *« Rends les hi-hats plus espacés dans les couplets et ajoute des rolls avant chaque refrain. »*
- *« Creuse un peu les bas-médiums du pad et ajoute de la réverb Dragonfly Hall sur la mélodie. »*
- *« Sauvegarde ce morceau dans `projects/mon-banger/projet.gridline`. »*

---

## Chapitre 4 — Tutoriel Reverse Engineering : Transformer un classique de Jeu Vidéo en Instru Rap

C'est l'un des workflows les plus puissants de ton studio. Voici comment procéder étape par étape.

### Étape 1 : Préparer le fichier source
Crée un dossier pour ton projet et déposes-y le thème de jeu vidéo (`.mp3`, `.wav`, `.flac` ou `.ogg`), par exemple :
```text
projects/zelda-flip/source/gerudo_valley.mp3
```
*(Astuce : Gridline embarque aussi `yt-dlp` et `ffmpeg` dans `/usr/share/gridline/tools/` si tu convertis tes propres fichiers audio locaux.)*

### Étape 2 : Lancer le Reverse Engineering par prompt
Selon le style de rap visé, utilise un prompt comme celui-ci :

> **Option A — Flip Sampled (style Boom-Bap / Griselda / Alchemist / Kanye) :**
> « Analyse `projects/zelda-flip/source/gerudo_valley.mp3` pour un flip Rap Boom-Bap à 90 BPM. Sépare les 6 stems (`htdemucs_6s`) dans `projects/zelda-flip/stems`, extrais les accords et le MIDI du stem guitare/piano, puis crée un beat dans `projects/zelda-flip/projet.gridline` : choppe le stem dans `gl.seltmann`, filtre sous 140 Hz avec TDR Nova/ZL Equalizer 2, ajoute CHOW Tape Model pour le grain vinyle, et pose une batterie Boom-Bap + basse calée sur les accords. Mixe à -13 LUFS. »

> **Option B — Remake / Trap Hybride (style Metro Boomin / Pierre Bourne / Drill) :**
> « Fais du reverse engineering sur `projects/chrono-trap/source/corridors_of_time.wav` pour une instru Trap à 144 BPM. Sépare les stems, transcris la mélodie principale et les accords en MIDI via `transcribe_audio_to_midi`, puis ré-orchestre le thème dans Surge XT et Gridline Keys/Plasma (garde aussi un peu du stem original filtré en intro). Ajoute une 808 de ma banque accordée sur la tonalité détectée avec sidechain sur le kick, des hi-hats rapides, et auto-mixe à -10.5 LUFS. »

### Étape 3 : Ce que l'IA fait automatiquement sous le capot
1. **`analyze_song_for_rap_flip`** détecte le BPM source, la tonalité, la grille d'accords mesure par mesure et calcule le ratio de stretch / repitch vinyle.
2. **`split_stems`** sépare le morceau via `gridline stems` (`htdemucs_6s` → `drums`, `bass`, `other`, `vocals`, `guitar`, `piano`).
3. **`transcribe_audio_to_midi`** extrait les notes polyphoniques MIDI avec Spotify Basic Pitch ONNX et écrit un `.mid` + des notes directes pour le Piano Roll de Gridline.
4. **`gridline`** assemble le sample découpé et/ou le remake synthé, coupe les basses du jeu original (`< 140 Hz`), ajoute la rythmique et accorde la 808 sur les vraies fondamentales du classique.

### Étape 4 : Explorer visuellement dans Gridline avec tes nouveaux plugins
Si tu ouvres le projet dans Gridline pour observer et apprendre :
- Mets **`LSP Spectrum Analyzer x2`** sur le Master pour voir exactement où frappent le kick (50–80 Hz), la 808 et le sample VGM.
- Mets **`LSP Oscilloscope x2`** sur le bus Kick+808 pour voir la forme d'onde et le sidechain en temps réel.
- Ouvre **`ZL Equalizer 2`** ou **`TDR Nova`** sur le stem VGM pour observer la courbe FFT et l'égalisation dynamique qui laisse respirer les drums.
- Ouvre **`Surge XT`** pour voir comment les oscillateurs, filtres et LFO recréent des leads 16-bit ou modernes.
- Ouvre **`Dragonfly Hall Reverb`** / **`Plate`** et **`CHOWTapeModel`** pour sculpter l'espace et la chaleur analogique.

---

## Chapitre 5 — Ajouter du chant, des acapellas ou des voix synthétiques

Tu disposes de **3 niveaux de voix** selon tes besoins :

1. **Acapellas & Vocaux de ta banque locale** :
   - Demande à l'IA : *« Cherche un vocal chop ou acapella compatible avec mon morceau en Do mineur dans ma banque et place-le sur les refrains. »*
   - Elle piochera dans `Vocal Sample Pack AVM` ou `CHRISSA SJE & NIKOS - INFINITE`.
2. **Guide vocal / Vocoder / Voix robotique immédiate (automatique)** :
   - Demande : *« Écris un refrain en français sur cette grille d'accords, génère le `.ustx` et synthétise un guide vocal `.wav`, puis passe-le dans le vocodeur Cipher (`gl.vocoder`) ou `gl.pitcher` avec Dragonfly Plate Reverb. »*
3. **Chant réaliste neuronal (OpenUTAU + DiffSinger)** :
   - OpenUTAU Linux (`0.1.576`) est installé dans `tools/OpenUtau/OpenUtau`.
   - Pour avoir des voix chantées réalistes (anglais, japonais, français, etc.), télécharge une banque de voix **DiffSinger** gratuite (OpenUTAU Singers) et dépose son dossier dans `/home/gamo/Bureau/musicos/voicebanks/`.
   - L'IA générera le `.ustx` prêt à ouvrir dans OpenUTAU (`open_in_openutau`), que tu exporteras en `.wav` puis réimporteras dans Gridline.

---

## Chapitre 6 — L'auto-mixage et l'export final

Avant de te rendre un morceau, l'IA exécute toujours sa **boucle d'auto-mixage** :
1. Elle rend `mix_check.wav` dans `renders/`.
2. Elle lance `audio-analyzer:analyze_mix` et vérifie :
   - `clipped_samples == 0` (aucune saturation numérique)
   - `true_peak_dbtp <= -1.0` (marge propre pour Spotify/YouTube/MP3)
   - `integrated_lufs` à `± 1.2 dB` de la cible (`-14` streaming, `-11` Trap/Rap)
   - `correlation_low_end_150hz >= 0.5` (basses bien centrées en mono)
3. Si un indicateur dévie, elle corrige les faders, l'EQ ou `gl.limiter` et compare avant/après avec `compare_renders`.

Pour exporter ton fichier final prêt à partager (WAV 24-bit, FLAC, MP3, ou projet FL Studio `.flp`), demande simplement à l'IA ou utilise directement en ligne de commande :
```bash
gridline render projects/active/projet.gridline -o projects/active/renders/master_final.wav --bits 24
# ou en MP3 / FLAC / stems séparés :
gridline render projects/active/projet.gridline -o projects/active/renders/master_final.mp3 --bitrate 320
```

---

## Chapitre 7 — Dépannage & Évolutions utiles

- **Tu as ajouté de nouveaux samples dans `pack/` ?**
  Demande à l'IA *« Ré-indexe mes samples avec CLAP »* ou lance :
  ```bash
  /home/gamo/Bureau/musicos/.venv/bin/python /home/gamo/Bureau/musicos/mcp-servers/audio-index/indexer.py --with-clap
  ```
- **Pourquoi certains plugins Windows du web (Voxengo SPAN, PsyScope, Isol8, Youlean, Kilohearts, Valhalla) ne sont pas installés ?**
  Ils n'existent pas en version Linux native : les installer exigerait Wine/Yabridge (lourd, fragile sous Wayland, et inutile pour l'IA). Ils sont tous remplacés par des équivalents **Linux natifs CLAP/VST3** déjà installés et vérifiés sans aucune erreur : **LSP Spectrum Analyzer**, **LSP Oscilloscope**, **LSP Crossover**, **ZL Equalizer 2**, **TDR Nova**, **Dragonfly Reverb**, **CHOW Tape Model**, plus le moteur python `pyloudnorm`.
- **Cas particulier de Vital et NeuralNote :**
  - **NeuralNote** (`v1.1.0` Linux VST3) est incompatible avec la glibc récente de Fedora 44 (`executable stack`). Son moteur exact (**Spotify Basic Pitch ONNX**) est directement intégré côté IA dans le serveur MCP `reverse-engineer` (`transcribe_audio_to_midi`).
  - **Vital** exige une connexion utilisateur sur [vital.audio](https://vital.audio/) : si tu souhaites l'ajouter en plus de **Surge XT** et **Plasma**, télécharge `VitalInstaller.zip` depuis ton compte, extrais `Vital.clap` / `Vital.vst3` dans `~/.clap/` ou `~/.vst3/`, puis lance `gridline scan-plugins --force`.
- **Conserver l'historique propre entre plusieurs morceaux :**
  Crée un sous-dossier par morceau dans `projects/nom-du-morceau/` (avec ses sous-dossiers `source/`, `stems/`, `midi/`, `vocals/`, `renders/`) ou demande à l'IA de travailler directement sur un nouveau fichier `.gridline`.
- **Préserver le processeur (CPU) et éviter les craquements/freezes dans Gridline :**
  - Les réglages globaux Gridline (`~/.config/gridline/settings.json`) et PipeWire (`~/.config/pipewire/pipewire.conf.d/99-ai-studio-lowcpu.conf` + `pipewire-pulse.conf.d/99-ai-studio-lowcpu.conf`) sont calibrés à **`1024` samples** (`~21–23 ms`), **`audio.threads = 4`** (adapté aux cœurs Performance sur Intel hybride), **`freeze.autoHibernate = true`** (mise en veille des plugins inactifs après `15 s`), animations UI et courbes lissées désactivées, et une seule fenêtre plugin ouverte à la fois.
  - Utilise `./scripts/launch-gridline.sh` pour forcer automatiquement `clock.force-quantum = 1024` sous PipeWire avant d'ouvrir Gridline.
  - Si ton ordinateur portable tourne sur batterie en profil `power-saver` (`CPU max 1700 MHz`), passe temporairement en mode équilibré ou performance (`powerprofilesctl set balanced`) lors des grosses sessions audio.
