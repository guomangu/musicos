# Audit v3.2 — Les 5 Carences Majeures Restantes & Solutions

**Socle vérifié :** v3.1 (6 chantiers intégrés : `mix_config.py`/`config/mix_profiles.json`, `autofix_transaction.py`, `score_ir.py` + clamp 20 ms, `adapt_voicing_to_timbre` + `evaluate_barwise_harmonic_alignment`, `spectral_corridors.json`, `asset_hygiene.py`). Le socle anti-fausse-note / anti-clipping est solide ; ce qui plafonne désormais la qualité **esthétique** est listé ci-dessous.

**Objectif v3.2 :** des morceaux vraiment mélodieux, vivants, cohérents d'un bout à l'autre — en conservant le budget (≤ 3 macro-tours LLM, ≤ 8 000 tokens, ≤ 2 rendus Gridline).

**État d'implémentation (v3.2 — terminé) :** les 5 priorités sont développées, un commit par jalon, et couvertes par la CI documentaire (`tests/test_docs_hygiene.py`) :

| Priorité | Jalon | Commit |
| :---: | :--- | :--- |
| P1 (#5b) | Purge contradictions docs + 3 assertions CI | `2b3b6ae` |
| P2 (#3) | `FullCoverageCoreValidator` + `compile_song --probe` | `38ac66d` |
| P3 (#1) | `NarrativeHarmony` (sections, phrasé 8 mesures, suspensions) | `bd379e0` |
| P4 (#2) | `ExpressionArticulations` (glides 808, vibrato retardé, vélocité→cutoff, ritardando) | `1129f93` |
| P5 (#4 + #5a) | `TimbralCohesion` (wetness/DRR, archétypes `gl.*`, banque grooves MIDI) + `MelodicAppealIndex` (tournoi Tour 2) | `3531038` |

## Feuille de route (ordre de priorité)

| Priorité | Problème | Modules ciblés | Impact audible |
| :---: | :--- | :--- | :--- |
| **P1** | #5b — contradictions doc + CI | `docs/00..05`, `REFERENCE_ARCHIVE.md`, `tests/test_docs_hygiene.py` | Supprime les ordres contradictoires (LPF « mix en U », plafonds) |
| **P2** | #3 — angle mort Tour 2 → Tour 3 | `full_coverage_validator.py`, `compile_song.py --probe`, `stem_matrix_evaluator.py`, `audio_analyzer_server.py` | 100 % des pistes auditées en stems dès le Tour 2 (sonde 12 mesures) |
| **P3** | #1 — narration statique (boucle 4 mesures) | `narrative_harmony.py`, schémas blueprint/plan, `song_builder.py`, `phrasing_engine.py`, `score_ir.py` | Vrais refrains/ponts contrastés, phrasé 8 mesures Q/R, suspensions 4→3/9→8 |
| **P4** | #2 — expressivité mécanique | `score_ir.py` (articulations), `gl_dsl.py`, `song_builder.py`, `transitions_lib.py` | Glides 808, vibrato retardé ≥ 220 ms, vélocité→cutoff, `outro_ritardando` |
| **P5** | #4 + #5a — DRR/espace, archétypes, grooves MIDI, MAI | `audio_metrics.py` (wetness/tail), `palette_builder.py`, `config/timbre_archetypes.json`, `midi_groove_bank.py`, `musicality_metrics.compute_melodic_appeal_index` | Espace sonore unifié (sends pondérés DRR), grooves idiomatiques, sélection active du meilleur motif |

## Les 5 problèmes & solutions

1. **Narration harmonique statique** → `NarrativeHarmony` : `harmonic_core.section_progressions` (verse/pre_chorus/bridge/chorus/outro), phrasé 8 mesures Antécédent/Conséquent (demi-cadence mesure 4, cadence fermée mesure 8), opérateurs motiviques déterministes (`motif_answer`, `rhythmic_lift`, `call_and_response`, `melodic_climax_arch`), suspensions `note_role="suspension"` résolues en 0,5–1,0 beat dans le ScoreIR.
2. **Expressivité mécanique** → `ExpressionArticulations` : `expression.articulation` (`glide_to_next`, `legato`, `swell`...), `glide_ms` 35–90 ms sur 808 (exemption ciblée du coupe-chevauchement), vibrato LFO retardé (0 sur les 220 premières ms, attaque ≤ 6 ct), couplage vélocité→Cutoff, `outro_ritardando` organique (4–8 % sur les 2 dernières mesures).
3. **Angle mort Tour 2/Tour 3** → `FullCoverageCoreValidator` : pré-vol symbolique 100 % du morceau sur ScoreIR dès le Tour 2 (< 100 ms : clones 8 mesures, charge cognitive 1–8 pistes/mesure, arc d'énergie par section, intégrité transitions/outro) + **sonde stems composite 12 mesures** (8 noyau + 4 hors-noyau séparées par 1 mesure de silence) en 1 seul rendu.
4. **Cohérence acoustique des palettes** → `TimbralCohesion` : `wetness_score`/`tail_duration_ms` dans `audio_metrics.py`, terme de cohérence d'espace DRR dans `palette_builder.py`, `fx_send`/`hall_send` pondérés par `(1 - wetness)` dans `song_builder.py`, archétypes `gl.*` calibrés par genre dans `config/timbre_archetypes.json`, banque de grooves extraits des `.mid` locaux via `scripts/midi_groove_bank.py` (jamais de mélodie copiée).
5. **Loi de Goodhart + contradictions doc** → `MelodicAppealIndex` : `compute_melodic_appeal_index(score_ir)` déterministe (< 15 ms, 4 critères positifs : entropie motivique zone d'or, clarté de l'arc de contour, tension→résolution cadentielle mesures 3–4/7→8, complémentarité rythmique lead/basse/counter) + tournoi de 2–3 variantes d'ornementation en Tour 2 (0 rendu audio) ; CI documentaire dans `tests/test_docs_hygiene.py` (0 lien Markdown cassé, 100 % des `scripts/*.py` indexés dans `00_SOMMAIRE.md`, 0 valeur chiffrée divergente vs `config/mix_profiles.json` dans `AGENT_RUNBOOK.md`/`docs/05_*`).

## Règle LPF clarifiée (v3.1+, normative)

Jamais de **LPF brique** (`gl.eq` 24 dB/oct) sous `6.5 kHz` (harmonie) / `9 kHz` (lead) — planchers `source_lpf_floors_hz` de `config/mix_profiles.json`. Pour assombrir un timbre : *high-shelf* doux (−3 dB) ou filtre de synthé musical (pente 6–12 dB/oct). Toute valeur chiffrée de mix/gate est gouvernée exclusivement par `config/mix_profiles.json` via `scripts/mix_config.py`.
