# REFERENCE ARCHIVE v3.0 — Ne JAMAIS charger en session de production

En production, seul [AGENT_RUNBOOK.md](AGENT_RUNBOOK.md) est lu (3 macro-tours, budget tokens).
Ce fichier est l'archive de développement/débogage : historique technique, inventaire des
fonctions internes, playbook des pièges vérifiés, journal des morceaux validés.

## Index des facettes détaillées (archive)

| Facette | Contenu archivé |
| :--- | :--- |
| [00_SOMMAIRE.md](00_SOMMAIRE.md) | Index complet des dossiers/fichiers, ordre d'utilité du workflow |
| [01_SYSTEME_STACK_ET_GUIDE.md](01_SYSTEME_STACK_ET_GUIDE.md) | Stack Fedora/PipeWire, install, venv, commandes exactes (`$AI_STUDIO_DIR`) |
| [02_COMPOSITION_EMOTIONNELLE_ET_THEORIE.md](02_COMPOSITION_EMOTIONNELLE_ET_THEORIE.md) | Composition émotionnelle (V, T, E, D), modes, 14 progressions, mélodie, groove |
| [03_ARCHITECTURE_BLUEPRINT_ET_COHESION.md](03_ARCHITECTURE_BLUEPRINT_ET_COHESION.md) | Blueprint, harmonic_core, schéma JSON, cohérence inter-facettes |
| [04_ANALYSE_SONORE_SAMPLES_APIS_ET_VGM.md](04_ANALYSE_SONORE_SAMPLES_APIS_ET_VGM.md) | Entonnoir DSP 4 couches (YIN+HPS+anti-partiel, HPSS/HNR, chroma CQT, inharmonicité), rôles, CLAP |
| [05_MIXAGE_QUALITY_GATE_ET_APPRENTISSAGE.md](05_MIXAGE_QUALITY_GATE_ET_APPRENTISSAGE.md) | Profils de mix, buses 17..22, quality gate 12+ critères, auto-fix DAG 5 étages, stem matrix |

## Inventaire des modules v3.0 (pointeurs, pas de duplication)

- `scripts/blueprint_safety.py` : audit projet-wide du blueprint (verrou octave exact,
  `MAX_SEMITONE_SHIFT` 5/7 st, boucles `melodic_loop` key-fit CQT + BPM lock [0,88;1,14],
  confinement cloches {1,5,8}, `root_note` obligatoire).
- `scripts/mix_config.py` + `config/mix_profiles.json` : **source unique** des constantes
  (ceiling -1,2 dBTP, gains, notches 2/3, tuning 6 ct fondation / 12 ct peak expressif, LPF sources).
- `scripts/stem_matrix_evaluator.py` : §1+§2 (V1..V4) + solveur de gains depuis 1 rendu stems 8 mesures.
- `scripts/compile_song.py` : compilateur 100 % déclaratif (ordre canonique MIDI immuable).
- `scripts/gain_staging_lib.py` : `solve_analytic_gain_trims` (DAG étage 3, one-shot).
- `scripts/sample_prep_lib.py` : `kick_tuning_guard`, `apply_tail_gate`, `repitch_loop`, `build_multizone_samples`.
- `scripts/groove_lib.py` : `sanitize_note_overlaps` (garde finale anti note-stealing, 10 ms).
- `scripts/transitions_lib.py` : `apply_harmonic_anticipation` synchrone (coupe Lead/Counter).
- `mcp-servers/audio-analyzer/musicality_metrics.py` : Low Interval Limit, `analyze_harmonic_clashes`
  (b2 vs b9 modale, sensible), `cadence_resolution_ok`, `rhythmic_motif_recurrence`,
  `chord_melody_tension_alignment`.
- `scripts/midi_autofix.py` : autofixer contour-preserving (coût J(c), anti quintes parallèles).
- Macro-outils MCP (3 tours) : `studio_prepare_session`, `studio_build_and_validate_core`,
  `studio_render_final_master` dans `mcp-servers/audio-analyzer/audio_analyzer_server.py`.

## Playbook des pièges vérifiés (historique)

- Erreur d'octave `root_note` = varispeed ±12 st -> chipmunk/smudge : verrouiller `root_note == midi_root`.
- Cloche à partiels inharmoniques (2,76x) -> YIN verrouille le partiel : HPS + anti-partial-lock.
- Accord/dyade joué chromatiquement -> `poly_chord_or_dyad` non jouable.
- LPF brique < 6,5 kHz à la source + shelf batterie -> "mix en U" : floors 6,5/9 kHz + high-shelf doux.
- Boucle 140 BPM dans projet 142 BPM -> flam dès 2 mesures : verrou BPM 0,05 + stretch exact.
- Vibrato ±12 ct lu comme désaccordé sur le master -> mesurer la justesse sur stems fondation ≤ 6 ct.
- Notches EQ empilés -> son en peigne : budget 2/piste, 3 total puis SWAP_SAMPLE.
- Chemins personnels absolus en dur dans les docs/scripts -> test `tests/test_docs_hygiene.py` bloquant (0 occurrence de tout chemin `/home/...`).

## Journal des morceaux validés (PQ)

rosh2 PQ 7.96, song1 PQ 8.05 (référence haute) ; song3/4/5 PQ 7.28-7.47 (régression Phases 6-8,
corrigée par v3.0 : cible PQ >= 7.8 sur les 7 profils de `tests/regression_multi_genre.py`).
