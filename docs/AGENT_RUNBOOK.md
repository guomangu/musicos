# AGENT RUNBOOK v3.3 — Création musicale en 3 macro-tours (<= 1 500 tokens)

Racine du studio : `$AI_STUDIO_DIR` (résolue auto, 0 chemin absolu).
Seul ce fichier est lu en production. Archive : [REFERENCE_ARCHIVE.md](REFERENCE_ARCHIVE.md).

## Tour 1 — `audio-analyzer:studio_prepare_session(project_dir, style_intent, reference_paths, remix_source_path=None, lyrics_spec=None)`
- Indexe `Samples/` (`user_samples_locked`, 0 téléchargement), propose 3 palettes cohérentes en espace DRR
  (`root_note`, BPM, boucles filtrées, `wetness_score`, archétypes `config/timbre_archetypes.json`).
- Scénario C (`remix_source_path`) : renvoie BPM/clé/grille de départ ; Scénario D (`lyrics_spec`) : syllabes/pauses.
- Retour JSON <= 1 500 cars.

## Tour 2 — Écrire `song_blueprint.json` + `arrangement_plan.json`, puis `audio-analyzer:studio_build_and_validate_core(project_dir, palette_id)`
- `song_blueprint.json` (`schemas/song_blueprint.schema.json`) : inclure `seed`, `mix_profile`, `emotion_vector`,
  `harmonic_core` (supporte `section_progressions` : verse/pre_chorus/chorus/bridge/outro + `bass_mode` :
  `root|stepwise_inversions|tonic_pedal|dominant_pedal`). Toute piste sample exige `sample_path` + `root_note == midi_root`.
- `arrangement_plan.json` (`schemas/arrangement_plan.schema.json`) : `palette_id`, `compact_notes: [[beat, deg, dur, vel]]`,
  phrasé 8 mesures (demi-cadence mes. 4, résolution mes. 8), `comping_style` par section (`sustained_pad|broken_arp_up|rhythmic_comp|pulse_ostinato`),
  `articulation` optionnelle (`glide_to_next`, `legato`, `swell`), suspensions résolues (`4->3`, `9->8`), `groove_style` (`midi_groove_bank`).
- `blueprint_enricher.py` complète proprement tout champ créatif omis sans écraser les choix explicites.
- `blueprint_safety` bloque : root_note faux, S_max > 5/7 st, boucle hors gamme/tempo, cloche hors {1,5,8}.
- Compilation 100 % déclarative : `compile_song.py <proj> --probe` :
  1. `NarrativeHarmony` + `OrchestrationEngine` (voice-leading LIL conscient du Lead) -> `phrasing` -> `transitions`
  2. `midi gate` contour-preserving + tournoi `MelodicAppealIndex` (2–3 variantes, 0 rendu)
  3. `ScoreIR` (`groove` en `expression` + articulations + clamp frontière 20 ms) -> `sanitize_note_overlaps`
  4. `FullCoverageCoreValidator` (pré-vol symbolique 100 % du morceau : clones 8 mes., charge cognitive, arc, outro)
  5. 1 seul rendu stems **sonde composite** (8 mes. noyau + tampon silence Hall + 4 mes. pistes hors-noyau) ->
     `stem_matrix_evaluator` (V1..V4, gains, chroma-vs-grille hors fenêtres de glide, `FIX_TRACK_ROOT_NOTE`).

## Tour 3 — `audio-analyzer:studio_render_final_master(project_dir, export_mp3)`
- `compile_song` complet (`outro_ritardando` sûr) + **1 seul** rendu master + `preview_highlights.wav` (24 s).
- Quality gate (`render_and_audit`) : 100 % seuils depuis `config/mix_profiles.json` + `golden_references/spectral_corridors.json`
  (repli 3 niveaux) + audit translation (mono fold-down, small-speaker, sharpness P95, PLR climax).
- Autofix transactionnel (`autofix_transaction.py` : rollback si Palier 1 régresse, notches 2/3 -> SWAP_SAMPLE).
- Optionnel après écoute humaine : `audio-analyzer:studio_refine_track(project_dir, delta_intent)` (ajustement chirurgical 1 tour).

## Tables créatives minimales & 14 invariants (résumé)
- Modes : V < -0,2 -> aeolian/phrygian/harmonic_minor ; sinon ionian/dorian/mixolydian.
- Progressions : trap sombre `i - bIImaj7 - i - bVI` ; pop `I - V - vi - IV` ; pont `iv7 - III - VI - V7` ; jazz `ii - V7 - I - VI`.
- Invariants : (1) `user_samples_locked`, (2) `mono_oneshot` + `root_note`, (3) A4 = 440 Hz ±5 ct (fondation <= 6 ct),
  (4) 1 sonité = 1 canal/insert/pattern/piste, (5) 5–8 pistes simultanées max, lead >= harmonie +3 dB,
  (6) 100 % mono < 120 Hz, note < 100 Hz >= 1/8, (7–12) natifs `gl.*` WAV 24-bit, `gl_dsl`, `add_channel(mixer=M)`,
  (13) Master doux (`config/mix_profiles.json` : Ceiling -1,2 dBTP, Gain <= 5,5 dB, `gl.dist` <= 0,04, jamais de LPF brique
  < 6,5 kHz harmonie / < 9 kHz lead — utiliser shelf doux/filtre synthé), (14) évolution 4/8/16 mes., silences Lead 20–50 %, fin propre.

## Budget
3 tours LLM, <= 8 000 tokens, 0 ligne `build_song.py`, <= 2 rendus Gridline, audit NumPy < 1,5 s.
