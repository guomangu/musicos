# AGENT RUNBOOK v3.0 — Création musicale en 3 macro-tours (<= 1 500 tokens de contexte)

Racine du studio : `$AI_STUDIO_DIR` (résolue automatiquement, aucun chemin absolu).
Seul ce fichier est chargé en session de production. L'archive détaillée est dans
[REFERENCE_ARCHIVE.md](REFERENCE_ARCHIVE.md) (jamais lue en production).

## Tour 1 — `audio-analyzer:studio_prepare_session(project_dir, style_intent, reference_paths)`
Indexe `Samples/` (scénario B : `user_samples_locked`, aucun téléchargement), propose 3 palettes
(`root_note`, BPM, boucle `melodic_loop`/`percussive_loop` déjà filtrés), profil cible (valence, LUFS).
Retour JSON <= 1 500 cars.

## Tour 2 — Écrire `song_blueprint.json` + `arrangement_plan.json`, puis `audio-analyzer:studio_build_and_validate_core(project_dir, palette_id)`
- `song_blueprint.json` : schéma `schemas/song_blueprint.schema.json` — toute piste
  `source_type="sample"` exige `sample_path` + `root_note` (= `midi_root` détecté, verrou octave exact).
- `arrangement_plan.json` : schéma `schemas/arrangement_plan.schema.json`, motifs compacts
  `compact_notes: [[beat, degré, dur, vélocité], ...]` + `palette_id`.
- Le gate `blueprint_safety` bloque : root_note faux, S_max > 5/7 st, boucle mélodique hors gamme
  (R_out > 0,22) ou hors tempo (r_eff hors [0,88 ; 1,14]), cloche inharmonique non confinée {1,5,8}.
- Compilation 100 % déclarative : `python3 scripts/compile_song.py <projet> --core-only`
  (ordre canonique MIDI immuable : voice-leading LIL -> phrasing -> transitions -> midi gate
  contour-preserving -> ScoreIR (`score_ir.py` : groove en `expression` + clamp frontière
  harmonique 20 ms) -> `sanitize_note_overlaps`). Si gridline dispo : 1 rendu stems
  8 mesures + évaluation V1..V4 en NumPy pur (`stem_matrix_evaluator`).

## Tour 3 — `audio-analyzer:studio_render_final_master(project_dir, export_mp3)`
`compile_song` complet + **1 seul** rendu master + quality gate (`render_and_audit`, rapport
`musical_safety`, budget de notches 2/piste 3 total -> SWAP_SAMPLE). Fiche finale <= 1 500 cars.

## Tables créatives minimales (non automatisables)
- Modes par valence : V < -0,2 -> aeolian/phrygian/harmonic_minor ; sinon ionian/dorian/mixolydian.
- Progressions (4 mesures) : trap sombre `i - bIImaj7 - i - bVI` (P5b, b9 autorisé) ; pop `I - V - vi - IV` ;
  edm `i - bVI - bVII - i` ; jazz `ii - V7 - I - VI` ; etc. (14 au total en archive).
- Matrices rythmiques 16 pas : ex. trap `1000 0010 1000 0010` (kick) + lead syncopé ; lofi swing 0,08.

## Règles absolues (résumé des 14 invariants, détail en archive)
1. `user_samples_locked` : jamais de sample externe si `Samples/` fourni (auto_fetch off).
2. `mono_oneshot` uniquement en mélodie (jamais accord/séquence) + `root_note` obligatoire.
3. A4 = 440 Hz ±5 ct ; root = `midi_root` détecté (YIN+HPS+anti-partiel), fine tune en cents.
4. 1 sonorité = 1 canal + 1 piste mixeur + 1 pattern + 1 piste Playlist nommée.
5. 5-8 pistes simultanées max ; lead >= harmonie +3 dB ; floors de présence par rôle.
6. 100 % mono < 120 Hz ; pas de note < 100 Hz plus courte qu'une croche.
7-12. Plugins natifs `gl.*` WAV 24-bit ; routage `add_channel(mixer=M)` uniquement ; pièges DSP
   (gl.eq Peak=6, sidechain preFader, set_song "tempo", Lua `function apply() end`).
13. Master doux : limiteur Ceiling -1,2 dBTP / Gain <= 5,5 dB, comp -12 dB, `gl.dist` <= 0,04,
    LPF bus >= 12 kHz, source LPF >= 6,5/9 kHz (jamais de LPF brique < 6,5 kHz -> "mix en U").
14. Évolution aux paliers 4/8/16 mesures ; silences 20-50 % sur le lead ; fin propre.

## Budget visé par morceau
3 tours LLM, <= 8 000 tokens totaux, 0 ligne de `build_song.py` écrite, <= 2 rendus Gridline,
audit NumPy §1+§2+gains < 1,5 s. Source unique des constantes de mix : `config/mix_profiles.json`.

## v3.1 (6 chantiers)
Seuils 100 % via `config/mix_profiles.json` + `mix_config.py` ; autofix transactionnel avec
rollback (`autofix_transaction.py`, journal `renders/autofix_history.json`) ; ScoreIR +
clamp frontière harmonique ; `adapt_voicing_to_timbre` + audit chroma-vs-grille
(`FIX_TRACK_ROOT_NOTE` câblé) ; corridors spectraux versionnés `golden_references/spectral_corridors.json`
(repli 3 niveaux, `reference_tier`/`reference_provenance`) ; hygiène banques `asset_hygiene.py`
(pas de détection de licence — responsabilité utilisateur).
