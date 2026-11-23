"""
Métriques objectives de musicalité mélodique et de cohésion inter-pistes
pour audio-analyzer:audit_midi_harmony, plus l'optimiseur de voice leading.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pretty_midi

_PERC_TRACK_RE = re.compile(
    r"\b(kick|snare|clap|open[- ]?hat|hi[- ]?hat|hats?|hh|percs?|rim|cymbal|crash|tom|shaker|drum|808 kick|triangle|zap|riser|fx|sfx|chant|chants)\b"
)


@dataclass
class MusicalityThresholds:
    stepwise_ratio_min: float = 0.65
    gap_fill_min: float = 0.60
    rest_ratio_min: float = 0.20
    rest_ratio_max: float = 0.50
    strong_beat_chord_ratio_min: float = 0.80
    voice_leading_max_st: float = 2.50
    lead_overlap_max: float = 0.15
    contrary_oblique_min: float = 0.55


def optimize_voice_leading(
    chord_pcs_list: list[list[int]],
    low_midi: int = 48,
    high_midi: int = 72,
) -> list[list[int]]:
    """
    Prend une liste d'accords définis par leurs pitch classes (0..11)
    et retourne des voicings MIDI renversés minimisant le mouvement moyen (<= 2.5 st).
    """
    def candidates_for_chord(pcs: list[int]) -> list[list[int]]:
        pools = []
        for pc in pcs:
            opts = [m for m in range(low_midi, high_midi + 1) if m % 12 == pc]
            pools.append(opts)
        res = []
        for combo in product(*pools):
            s = sorted(combo)
            if s[-1] - s[0] <= 16 and len(set(s)) == len(pcs):
                # Éviter les secondes mineures serrées (< 2 st) entre voix graves (< MIDI 57)
                tight_low = any(
                    (s[k + 1] - s[k] <= 1 and s[k] < 57)
                    for k in range(len(s) - 1)
                )
                if not tight_low:
                    res.append(s)
        return res

    all_cands = [candidates_for_chord(pcs) for pcs in chord_pcs_list]
    if any(not c for c in all_cands):
        # Fallback sans filtre tight_low si accord très dense
        def raw_cands(pcs: list[int]) -> list[list[int]]:
            pools = [[m for m in range(low_midi, high_midi + 1) if m % 12 == pc] for pc in pcs]
            return [sorted(c) for c in product(*pools) if sorted(c)[-1] - sorted(c)[0] <= 18 and len(set(c)) == len(pcs)]
        all_cands = [c if c else raw_cands(pcs) for c, pcs in zip(all_cands, chord_pcs_list)]

    mid = (low_midi + high_midi) / 2.0
    chosen = [min(all_cands[0], key=lambda c: abs(sum(c) / len(c) - mid))]

    for cands in all_cands[1:]:
        prev = chosen[-1]
        best = min(
            cands,
            key=lambda c: sum(min(abs(n - p) for p in prev) for n in c) / len(c),
        )
        chosen.append(best)
    return chosen


def _infer_roles(pm: pretty_midi.PrettyMIDI) -> Dict[str, List[pretty_midi.Instrument]]:
    """Classe automatiquement les pistes non-percussives par rôle (bass, chords, lead, counter)."""
    melodic_insts = [
        inst
        for inst in pm.instruments
        if not inst.is_drum
        and len(inst.notes) > 0
        and not _PERC_TRACK_RE.search((inst.name or "").lower())
    ]
    roles: Dict[str, List[pretty_midi.Instrument]] = {"bass": [], "chords": [], "lead": [], "counter": []}

    for inst in melodic_insts:
        name = (inst.name or "").lower()
        pitches = np.array([n.pitch for n in inst.notes], dtype=float)
        mean_p = float(np.mean(pitches))

        notes_sorted = sorted(inst.notes, key=lambda n: n.start)
        overlaps = sum(
            1
            for i in range(len(notes_sorted) - 1)
            if notes_sorted[i + 1].start < notes_sorted[i].end - 0.02
        )
        poly_ratio = overlaps / max(1, len(notes_sorted))

        if any(k in name for k in ("808", "bass", "sub")) or mean_p < 48.0:
            roles["bass"].append(inst)
        elif any(k in name for k in ("lead", "melody", "topline", "vox", "vocal")):
            roles["lead"].append(inst)
        elif any(k in name for k in ("chord", "pad", "string", "rhodes", "harmony")) or poly_ratio >= 0.45:
            roles["chords"].append(inst)
        elif "piano" in name:
            roles["lead"].append(inst)
        else:
            roles["counter"].append(inst)

    if not roles["lead"] and (roles["counter"] or roles["chords"]):
        candidates = roles["counter"] if roles["counter"] else roles["chords"]
        highest = max(candidates, key=lambda i: np.mean([n.pitch for n in i.notes]))
        roles["lead"].append(highest)
        if highest in roles["counter"]:
            roles["counter"].remove(highest)

    return roles


def _extract_monophonic_topline(notes: List[pretty_midi.Note]) -> List[pretty_midi.Note]:
    """Extrait la ligne supérieure si la piste contient de légers chevauchements."""
    if not notes:
        return []
    ordered = sorted(notes, key=lambda n: (round(n.start, 3), -n.pitch))
    topline: List[pretty_midi.Note] = []
    for n in ordered:
        if topline and abs(n.start - topline[-1].start) < 0.03:
            continue
        topline.append(n)
    return topline


def _compute_stepwise_and_gap_fill(topline: List[pretty_midi.Note]) -> Tuple[float, float, int]:
    if len(topline) < 2:
        return 1.0, 1.0, 0
    diffs = [topline[i + 1].pitch - topline[i].pitch for i in range(len(topline) - 1)]
    stepwise = sum(1 for d in diffs if abs(d) <= 4)
    stepwise_ratio = stepwise / len(diffs)

    leaps = 0
    filled = 0
    for i, d in enumerate(diffs[:-1]):
        if abs(d) >= 5:
            leaps += 1
            nxt = diffs[i + 1]
            if (d * nxt < 0) and (1 <= abs(nxt) <= 3):
                filled += 1
    gap_fill = (filled / leaps) if leaps > 0 else 1.0
    ambitus = max(n.pitch for n in topline) - min(n.pitch for n in topline)
    return round(stepwise_ratio, 3), round(gap_fill, 3), int(ambitus)


def _compute_rest_ratio(notes: List[pretty_midi.Note], total_duration: float) -> float:
    if not notes or total_duration <= 0:
        return 1.0
    intervals = sorted((n.start, n.end) for n in notes)
    merged: List[List[float]] = []
    for s, e in intervals:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    covered = sum(e - s for s, e in merged)
    return round(float(max(0.0, min(1.0, 1.0 - covered / float(total_duration)))), 3)


def _extract_chord_slices(chord_notes: List[pretty_midi.Note], quantize_s: float = 0.08) -> List[Tuple[float, List[int]]]:
    """Regroupe les notes d'accords simultanées en tranches (start_time, pitches_sorted)."""
    if not chord_notes:
        return []
    ordered = sorted(chord_notes, key=lambda n: n.start)
    slices: List[Tuple[float, List[int]]] = []
    cur_t = ordered[0].start
    cur_pitches = [ordered[0].pitch]
    for n in ordered[1:]:
        if n.start - cur_t <= quantize_s:
            cur_pitches.append(n.pitch)
        else:
            slices.append((cur_t, sorted(set(cur_pitches))))
            cur_t = n.start
            cur_pitches = [n.pitch]
    slices.append((cur_t, sorted(set(cur_pitches))))
    return [s for s in slices if len(s[1]) >= 2]


def _compute_voice_leading(slices: List[Tuple[float, List[int]]]) -> float:
    """Calcule le déplacement moyen (en demi-tons) entre deux accords consécutifs."""
    if len(slices) < 2:
        return 0.0
    distances: List[float] = []
    for (_, prev_p), (_, curr_p) in zip(slices[:-1], slices[1:]):
        if prev_p == curr_p:
            continue
        d = float(np.mean([min(abs(c - p) for p in prev_p) for c in curr_p]))
        distances.append(d)
    return round(float(np.mean(distances)), 3) if distances else 0.0


def _compute_strong_beat_chord_tone_ratio(
    pm: pretty_midi.PrettyMIDI,
    topline: List[pretty_midi.Note],
    chord_notes: List[pretty_midi.Note],
    bass_notes: List[pretty_midi.Note],
) -> float:
    if not topline or (not chord_notes and not bass_notes):
        return 1.0
    beats = pm.get_beats()
    if len(beats) < 2:
        return 1.0
    strong_beats = beats[::2]

    strong_notes = [
        n for n in topline if np.min(np.abs(strong_beats - n.start)) <= 0.09
    ]
    if not strong_notes:
        return 1.0

    matches = 0
    for n in strong_notes:
        active_pcs = {
            c.pitch % 12
            for c in (chord_notes + bass_notes)
            if c.start - 0.05 <= n.start <= c.end + 0.05
        }
        if not active_pcs or (n.pitch % 12) in active_pcs:
            matches += 1
    return round(matches / len(strong_notes), 3)


def _compute_lead_overlap_conflict(
    lead_insts: List[pretty_midi.Instrument],
    counter_insts: List[pretty_midi.Instrument],
) -> float:
    """Détecte les attaques simultanées (<= 40 ms) dans la même octave (< 12 st) entre pistes mélodiques."""
    tracks = lead_insts + counter_insts
    if len(tracks) < 2:
        return 0.0
    total_attacks = sum(len(t.notes) for t in tracks)
    if total_attacks == 0:
        return 0.0
    conflicts = 0
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            for n1 in tracks[i].notes:
                for n2 in tracks[j].notes:
                    if abs(n1.start - n2.start) <= 0.04 and abs(n1.pitch - n2.pitch) < 12:
                        conflicts += 1
                        break
    return round(conflicts / max(1, total_attacks), 3)


def _compute_contrary_oblique_ratio(
    topline: List[pretty_midi.Note], bass_notes: List[pretty_midi.Note]
) -> float:
    if len(topline) < 2 or len(bass_notes) < 2:
        return 1.0
    bass_sorted = sorted(bass_notes, key=lambda n: n.start)

    def bass_at(t: float) -> Optional[int]:
        active = [b.pitch for b in bass_sorted if b.start - 0.05 <= t <= b.end + 0.05]
        return active[-1] if active else None

    total = 0
    good = 0
    for n1, n2 in zip(topline[:-1], topline[1:]):
        b1, b2 = bass_at(n1.start), bass_at(n2.start)
        if b1 is None or b2 is None:
            continue
        dm = n2.pitch - n1.pitch
        db = b2 - b1
        if dm == 0 and db == 0:
            continue
        total += 1
        if (dm * db) <= 0:
            good += 1
    return round(good / total, 3) if total > 0 else 1.0


def _beat_block_4bars(pm: pretty_midi.PrettyMIDI, time_s: float) -> int:
    """Map a timestamp (seconds) to its 0-indexed 4-measure block."""
    beats = pm.get_beats()
    num = pm.time_signature_changes[0].numerator if pm.time_signature_changes else 4
    bars_beats = max(1, 4 * int(num))
    if len(beats) >= 2:
        step = float(beats[-1] - beats[-2])
        if time_s <= beats[-1]:
            beat_pos = float(np.interp(time_s, beats, np.arange(len(beats), dtype=float)))
        else:
            beat_pos = (len(beats) - 1) + (time_s - float(beats[-1])) / max(step, 1e-4)
        return int(max(0.0, beat_pos) // bars_beats)
    return 0


def _infer_allowed_pcs_from_midi(midi_path: str) -> Optional[set[int]]:
    """Infer scale pitch classes from a MIDI file via music21 when no explicit key is supplied."""
    try:
        import music21

        score = music21.converter.parse(midi_path)
        est_key = score.analyze("key")
        return {p.pitchClass for p in est_key.getScale().getPitches()}
    except Exception:
        return None


def detect_minor_2nd_clashes(
    pm: pretty_midi.PrettyMIDI,
    allowed_pcs: Optional[set[int]] = None,
    valence: float = 0.0,
    tension: float = 0.3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Detect inter-track minor 2nd (1 st) / minor 9th (13 st) clashes.
    When valence < -0.3 or tension >= 0.5, up to 2 diatonic tensions per
    4-bar block are permitted when both pitches belong to `allowed_pcs`.
    Returns (all_clashes, blocking_clashes).
    """
    melodic_notes: List[Tuple[float, float, int, str]] = []
    for inst in pm.instruments:
        name = inst.name or f"Track_{inst.program}"
        if not inst.notes:
            continue
        is_perc = bool(inst.is_drum) or bool(_PERC_TRACK_RE.search(name.lower()))
        if not is_perc:
            for n in inst.notes:
                melodic_notes.append((n.start, n.end, n.pitch, name))

    melodic_notes.sort(key=lambda x: x[0])
    dark_tense = (valence < -0.3) or (tension >= 0.5)
    all_clashes: List[Dict[str, Any]] = []
    blocking_clashes: List[Dict[str, Any]] = []
    diatonic_per_block: Dict[int, int] = {}

    for i, (s1, e1, p1, tr1) in enumerate(melodic_notes):
        for s2, e2, p2, tr2 in melodic_notes[i + 1 :]:
            if s2 >= e1 - 0.01:
                break
            if tr1 == tr2:
                continue
            diff = abs(p1 - p2)
            if diff not in (1, 13):
                continue

            is_diatonic = (
                allowed_pcs is not None
                and (p1 % 12) in allowed_pcs
                and (p2 % 12) in allowed_pcs
            )
            block_idx = _beat_block_4bars(pm, s2)
            entry = {
                "time_s": round(float(s2), 3),
                "block_4bars": block_idx,
                "tracks": [tr1, tr2],
                "notes": [pretty_midi.note_number_to_name(p1), pretty_midi.note_number_to_name(p2)],
                "interval": "minor_2nd_cluster" if diff == 1 else "minor_9th_clash",
                "diatonic": is_diatonic,
            }
            all_clashes.append(entry)

            if dark_tense and is_diatonic:
                diatonic_per_block[block_idx] = diatonic_per_block.get(block_idx, 0) + 1
                if diatonic_per_block[block_idx] > 2:
                    blocking_clashes.append(entry)
            else:
                blocking_clashes.append(entry)

    return all_clashes, blocking_clashes


def evaluate_midi_musicality(
    midi_path: str,
    thresholds: Optional[MusicalityThresholds] = None,
    valence: float = 0.0,
    tension: float = 0.3,
    allowed_pcs: Optional[set[int]] = None,
) -> Dict[str, Any]:
    """Point d'entrée principal à appeler depuis audit_midi_harmony."""
    dark_tense = (valence < -0.3) or (tension >= 0.5)
    if thresholds is not None:
        th = thresholds
    else:
        th = MusicalityThresholds(
            strong_beat_chord_ratio_min=0.65 if dark_tense else 0.80
        )
    pm = pretty_midi.PrettyMIDI(midi_path)
    total_dur = pm.get_end_time()
    roles = _infer_roles(pm)

    lead_notes = [n for inst in roles["lead"] for n in inst.notes]
    chord_notes = [n for inst in roles["chords"] for n in inst.notes]
    bass_notes = [n for inst in roles["bass"] for n in inst.notes]
    topline = _extract_monophonic_topline(lead_notes)

    stepwise_ratio, gap_fill, ambitus = _compute_stepwise_and_gap_fill(topline)
    rest_ratio = _compute_rest_ratio(topline, total_dur)
    slices = _extract_chord_slices(chord_notes)
    vl_avg = _compute_voice_leading(slices)
    strong_beat_ratio = _compute_strong_beat_chord_tone_ratio(pm, topline, chord_notes, bass_notes)
    overlap_conflict = _compute_lead_overlap_conflict(roles["lead"], roles["counter"])
    contrary_ratio = _compute_contrary_oblique_ratio(topline, bass_notes)

    eff_allowed = allowed_pcs
    if eff_allowed is None and dark_tense:
        raw_clashes, _ = detect_minor_2nd_clashes(pm, allowed_pcs=None, valence=valence, tension=tension)
        if raw_clashes:
            eff_allowed = _infer_allowed_pcs_from_midi(midi_path)
    all_clashes, blocking_clashes = detect_minor_2nd_clashes(
        pm, allowed_pcs=eff_allowed, valence=valence, tension=tension
    )

    failures: List[str] = []
    if stepwise_ratio < th.stepwise_ratio_min:
        failures.append(f"stepwise_ratio={stepwise_ratio} < {th.stepwise_ratio_min} (mélodie trop sautillante)")
    if gap_fill < th.gap_fill_min:
        failures.append(f"gap_fill_compliance={gap_fill} < {th.gap_fill_min} (sauts non résolus en sens inverse)")
    if not (th.rest_ratio_min <= rest_ratio <= th.rest_ratio_max):
        failures.append(f"melody_rest_ratio={rest_ratio} hors [{th.rest_ratio_min}, {th.rest_ratio_max}]")
    if strong_beat_ratio < th.strong_beat_chord_ratio_min:
        failures.append(f"strong_beat_chord_tone_ratio={strong_beat_ratio} < {th.strong_beat_chord_ratio_min}")
    if vl_avg > th.voice_leading_max_st:
        failures.append(f"voice_leading_avg_semitones={vl_avg} > {th.voice_leading_max_st} (accords en blocs)")
    if overlap_conflict > th.lead_overlap_max:
        failures.append(f"lead_overlap_conflict={overlap_conflict} > {th.lead_overlap_max} (cafouillis d'attaques)")
    if contrary_ratio < th.contrary_oblique_min:
        failures.append(f"contrary_or_oblique_ratio={contrary_ratio} < {th.contrary_oblique_min}")
    if blocking_clashes:
        failures.append(
            f"minor_2nd_clashes={len(blocking_clashes)} bloquants "
            f"({'<= 2 diatoniques/4 mes. autorisées' if dark_tense else '0 autorisé'})"
        )

    return {
        "musicality_verdict": "PASS" if not failures else "FAIL",
        "emotion_mode": "dark_tense" if dark_tense else "consonant",
        "metrics": {
            "stepwise_ratio": stepwise_ratio,
            "gap_fill_compliance": gap_fill,
            "melody_ambitus_semitones": ambitus,
            "melody_rest_ratio": rest_ratio,
            "strong_beat_chord_tone_ratio": strong_beat_ratio,
            "voice_leading_avg_semitones": vl_avg,
            "lead_overlap_conflict": overlap_conflict,
            "contrary_or_oblique_ratio": contrary_ratio,
            "minor_2nd_clashes_total": len(all_clashes),
            "minor_2nd_clashes_blocking": len(blocking_clashes),
        },
        "minor_2nd_clashes": all_clashes,
        "blocking_minor_2nd_clashes": blocking_clashes,
        "detected_roles": {k: [i.name for i in v] for k, v in roles.items()},
        "failures": failures,
    }
