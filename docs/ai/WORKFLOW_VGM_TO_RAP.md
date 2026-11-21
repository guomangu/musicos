# Protocole IA : Reverse Engineering VGM → Instru Rap

Ce guide définit la chaîne exacte à suivre lorsqu'un utilisateur fournit un classique de jeu vidéo (`.mp3`, `.wav`, `.flac`, `.ogg`) et demande d'en extraire la matière musicale pour en faire une instru Rap (Trap, Boom-Bap, Drill, Lo-Fi, Cloud Rap).

## Phase 1 : Diagnostic musical du morceau source
1. Appelle `reverse-engineer:analyze_song_for_rap_flip(audio_path, target_rap_bpm=...)` (ex. `140` pour Trap/Drill, `90` pour Boom-Bap).
2. Récupère :
   - `detected_bpm`, `key` et `chords_first_bars` (grille d'accords par mesure).
   - `time_stretch_ratio` et `semitones_if_vinyl_repitch` (si l'utilisateur veut un repitch vinyle type Boom-Bap plutôt qu'un time-stretch transparent).

## Phase 2 : Séparation des stems (`gridline stems`)
1. Appelle `reverse-engineer:split_stems(audio_path, output_dir="projects/<nom>/stems", model="htdemucs_6s")`.
   - `htdemucs_6s` produit jusqu'à 6 pistes : `drums`, `bass`, `other`, `vocals`, `guitar`, `piano`.
   - Utilise `model="htdemucs"` si le morceau est très électronique/chiptune 8-bit/16-bit où `guitar`/`piano` sont moins pertinents.

## Phase 3 : Extraction MIDI & Accords (Basic Pitch ONNX)
1. Appelle `reverse-engineer:transcribe_audio_to_midi` sur les stems mélodiques pertinents (`piano.wav`, `guitar.wav`, `other.wav`, et éventuellement `bass.wav`) :
   - Passe `bpm=<detected_bpm>` et `quantize_beats=0.25` (ou `0.5` pour des accords).
   - Sauvegarde aussi un fichier `.mid` dans `projects/<nom>/midi/` si utile.
2. Récupère le tableau `gridline_notes` (`[{start, length, pitch, velocity}]`).
   - Nettoie/filtre si nécessaire (ex. supprimer les notes ghost très courtes < 50 de vélocité ou hors tessiture).

## Phase 4 : Rédaction du `song_blueprint.json` (`harmonic_core`) & Gate `master_sketch.mid`
Avant d'assembler dans Gridline, écris `song_blueprint.json` avec **`harmonic_core`** (voicings optimisés par *Smooth Voice Leading* `<= 2.5 st` et cellule `A-A'-A''-B`), génère la réduction `master_sketch.mid` et passe le Gate `audio-analyzer:audit_midi_harmony(..., strict_musicality=True)` (`PASS`) avant d'ajouter la batterie. Stratégie choisie (ou hybride) :
- **Stratégie A — Sample Flip (Boom-Bap / Trap Sampled)** :
  - Utilise le stem `other.wav` / `piano.wav` dans `gl.seltmann` (chop/beat sync) ou `gl.slicer` (Kerf) + coupe passe-haut à `130–160 Hz` (`gl.eq`, ou `TDR Nova` / `ZL Equalizer 2` si mode Live) + patine analogique (`gl.dist`/`gl.eq` ou `CHOWTapeModel` en mode Live).
- **Stratégie B — Remake / Ré-orchestration MIDI (Trap / Drill / Hybride Moderne)** :
  - Injecte `gridline_notes` directement via `set_notes` dans des instruments fiables (`gl.keys`, `gl.matrixfm`, `gl.plasma`, `gl.string`, ou `Surge XT` en mode Live si présent dans `list_plugin_catalog`).
  - Transpose éventuellement en mineur ou octaves supérieures pour libérer le bas du spectre.
- **Stratégie C — Hybride (le plus puissant)** :
  - Garde le stem VGM filtré en intro/couplet et double/remplace le thème au refrain avec Surge XT / Plasma.

## Phase 5 : Rythmique Rap & Basse / 808 accordée
1. Cherche les one-shots dans `audio-index:search_samples` / `find_compatible_samples` :
   - Kick sec et punchy (`category="kick"`), Snare/Clap (`category="snare"` / `"clap"`), Hi-hat (`category="hat"`), 808 (`category="808"`, compatible avec `key`).
   - Charge les samples dans `gl.sampler` via `patch(path="channel/N", set={"plugin": {"data": "/abs/path"}})` et vérifie `"silent": false`.
2. Programme les notes de la **808 / Sub** sur les **fondamentales exactes des accords** extraits en Phase 1 (`chords_first_bars`) ou du stem `bass` transcrit !
3. Programme les hi-hats Rap en Lua (`run_script` dans `function apply()`) avec rolls en triolets / doubles-croches et vélocités variées.

## Phase 6 : Mixage spécialisé VGM → Rap & Vérification
1. **Nettoyage du sample VGM** :
   - Coupe passe-haut stricte `>= 130 Hz` sur le stem VGM pour qu'aucun résidu de basse du jeu original ne vienne polluer la 808.
   - Si le sample VGM a des médiums agressifs (`1.5–4 kHz`), insère `TDR Nova` ou `ZL Equalizer 2` en EQ dynamique légère pour laisser la place aux futurs couplets rap.
2. **Cohabitation Kick / 808** :
   - Envoi sidechain Kick -> 808 (`set_send(mixer=kick_mix, to=808_mix, sidechain=true)`) + `gl.comp` ou EQ dynamique sur la 808.
3. **Boucle d'auto-mixage** :
   - `render_audio` -> `audio-analyzer:analyze_mix(target_lufs=-11.0)` (Trap/Rap) ou `-14.0` (streaming) + `analyze_low_end_masking`.
   - Ajuste jusqu'à `clipped_samples == 0`, `true_peak_dbtp <= -1.0`, `correlation_low_end_150hz >= 0.5`.
