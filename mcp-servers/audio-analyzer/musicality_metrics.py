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
                # v3.0 Low Interval Limit (ERB bands): no close interval below its absolute floor
                lil_bad = any(low_interval_violation(s[i], s[j]) for i in range(len(s)) for j in range(i + 1, len(s)))
                if not tight_low and not lil_bad:
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


def _voicing_candidates(pcs: list[int], low_midi: int, high_midi: int) -> list[list[int]]:
    """Candidats de voicing d'un accord (mêmes filtres que optimize_voice_leading)."""
    pools = [[m for m in range(low_midi, high_midi + 1) if m % 12 == pc] for pc in pcs]
    res = []
    for combo in product(*pools):
        sv = sorted(combo)
        if sv[-1] - sv[0] <= 16 and len(set(sv)) == len(pcs):
            tight_low = any((sv[k + 1] - sv[k] <= 1 and sv[k] < 57) for k in range(len(sv) - 1))
            lil_bad = any(low_interval_violation(sv[i], sv[j]) for i in range(len(sv)) for j in range(i + 1, len(sv)))
            if not tight_low and not lil_bad:
                res.append(sv)
    return res


def optimize_voice_leading_with_lead(
    chord_pcs_list: list[list[int]],
    lead_contour: list[int | None],
    low_midi: int = 48,
    high_midi: int = 72,
    pad_filtered: bool = False,
) -> list[list[int]]:
    """v3.3 #2 — voice-leading avec terme de coût conscient du contour du Lead :
    pénalise le croisement soprano >= lead_bar_min - 2 (sauf pad très filtré en arrière-plan),
    les octaves/quintes parallèles entre la voix supérieure des accords et le Lead, et
    favorise le mouvement contraire. `lead_contour[i]` = pitch min du Lead sur l'accord i
    (None si silence). Repli sur `optimize_voice_leading` si un accord n'a aucun candidat."""
    all_cands = [_voicing_candidates(pcs, low_midi, high_midi) for pcs in chord_pcs_list]
    if any(not c for c in all_cands):
        return optimize_voice_leading(chord_pcs_list, low_midi=low_midi, high_midi=high_midi)

    def lead_cost(i: int, c: list[int], prev: list[int] | None) -> float:
        lm = lead_contour[i] if i < len(lead_contour) else None
        if lm is None:
            return 0.0
        sop = max(c)
        cost = 0.0
        if not pad_filtered and sop > int(lm) - 3:
            cost += 10.0 * (sop - (int(lm) - 3))  # croisement soprano/Lead (strict : sop > lead-3)
        if i > 0 and prev is not None and i - 1 < len(lead_contour) and lead_contour[i - 1] is not None:
            iv_now = (sop - int(lm)) % 12
            iv_prev = (max(prev) - int(lead_contour[i - 1])) % 12
            if iv_now == iv_prev and iv_now in (0, 7):
                cost += 8.0  # octave/quinte parallèle soprano <-> Lead
            # bonus mouvement contraire
            d_sop = sop - max(prev)
            d_lead = int(lm) - int(lead_contour[i - 1])
            if d_sop * d_lead < 0:
                cost -= 2.0
        return cost

    mid = (low_midi + high_midi) / 2.0
    chosen = [min(all_cands[0], key=lambda c: abs(sum(c) / len(c) - mid) + lead_cost(0, c, None))]
    for i in range(1, len(all_cands)):
        prev = chosen[-1]
        best = min(
            all_cands[i],
            key=lambda c: sum(min(abs(n - p) for p in prev) for n in c) / len(c) + lead_cost(i, c, prev),
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

        if any(k in name for k in ("808", "bass", "sub")):
            roles["bass"].append(inst)
        elif any(k in name for k in ("lead", "melody", "topline", "vox", "vocal")):
            roles["lead"].append(inst)
        elif any(k in name for k in ("chord", "pad", "string", "rhodes", "harmony")) or poly_ratio >= 0.45:
            roles["chords"].append(inst)
        elif mean_p < 48.0:
            roles["bass"].append(inst)
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


def _compute_stepwise_and_gap_fill(
    topline: List[pretty_midi.Note],
    sec_per_beat: Optional[float] = None,
) -> Tuple[float, float, int]:
    if len(topline) < 2:
        return 1.0, 1.0, 0
    diffs = [topline[i + 1].pitch - topline[i].pitch for i in range(len(topline) - 1)]
    stepwise = sum(1 for d in diffs if abs(d) <= 4)

    leaps = 0
    filled = 0
    for i, d in enumerate(diffs[:-1]):
        if abs(d) >= 5:
            leaps += 1
            nxt = diffs[i + 1]
            if (d * nxt < 0) and (1 <= abs(nxt) <= 3):
                filled += 1
    gap_fill = (filled / leaps) if leaps > 0 else 1.0

    # Phase 8 Q5.3 — up to 2 expressive leaps (>= 7 st) per 8 bars are not penalised
    # when resolved by a contrary gap-fill or a >= 0.5 beat breathing rest.
    spb = float(sec_per_beat) if sec_per_beat else None
    expressive = 0
    if spb is not None:
        window_s = 32.0 * spb
        leap_times: List[float] = []
        for i, d in enumerate(diffs):
            if abs(d) < 7:
                continue
            idx_next = i + 1
            nxt = diffs[idx_next] if idx_next < len(diffs) else 0
            gap_filled = (d * nxt < 0) and (1 <= abs(nxt) <= 3)
            rest_after = (
                idx_next + 1 < len(topline)
                and (topline[idx_next + 1].start - topline[idx_next].end) >= 0.5 * spb
            )
            if not (gap_filled or rest_after):
                continue
            t_leap = topline[idx_next].start
            leap_times = [t for t in leap_times if t_leap - t < window_s]
            if len(leap_times) < 2:
                leap_times.append(t_leap)
                expressive += 1

    stepwise_ratio = (stepwise + expressive) / len(diffs)
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


def _parse_time_signature(
    time_signature: Optional[Any],
) -> Optional[tuple[int, int]]:
    """Normalise une signature temporelle (`(3, 4)` ou `'3/4'`) en tuple `(num, den)`."""
    if time_signature is None or time_signature == "":
        return None
    if isinstance(time_signature, str):
        raw = time_signature.strip()
        if "/" in raw:
            parts = raw.split("/", 1)
            return int(parts[0]), int(parts[1])
        return None
    if isinstance(time_signature, (tuple, list)) and len(time_signature) == 2:
        return int(time_signature[0]), int(time_signature[1])
    return None


def _resolve_time_signature(
    pm: pretty_midi.PrettyMIDI,
    time_signature: Optional[Any] = None,
) -> tuple[int, int]:
    """Lit `(numerator, denominator)` depuis l'argument explicite ou les méta-événements MIDI."""
    parsed = _parse_time_signature(time_signature)
    if parsed is not None and parsed[0] > 0 and parsed[1] > 0:
        return parsed
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        if ts.numerator > 0 and ts.denominator > 0:
            return int(ts.numerator), int(ts.denominator)
    return 4, 4


def _bar_quarters(num: int, den: int) -> float:
    """Longueur d'une mesure en noires (quarter notes)."""
    return float(max(1, num)) * (4.0 / float(max(1, den)))


def _strong_beat_offsets_quarters(num: int, den: int) -> tuple[float, ...]:
    """Positions des temps forts intra-mesure exprimées en noires (0-indexed)."""
    if (num, den) == (4, 4):
        return (0.0, 2.0)
    if (num, den) in ((3, 4), (2, 4)):
        return (0.0,)
    if (num, den) == (6, 8):
        return (0.0, 1.5)
    if (num, den) == (9, 8):
        return (0.0, 1.5, 3.0)
    if (num, den) == (12, 8):
        return (0.0, 1.5, 3.0, 4.5)
    if (num, den) == (5, 4):
        return (0.0, 3.0)
    if (num, den) == (7, 8):
        return (0.0, 1.0, 2.0)
    if den == 8 and num >= 6 and num % 3 == 0:
        return tuple(i * 1.5 for i in range(num // 3))
    if num >= 4 and den == 4:
        return (0.0, 2.0)
    return (0.0,)


def _strong_beat_times(
    pm: pretty_midi.PrettyMIDI,
    num: int,
    den: int,
    total_dur: Optional[float] = None,
) -> np.ndarray:
    """Génère les timestamps exacts (en secondes) de tous les temps forts du morceau."""
    end_t = float(total_dur if total_dur is not None else pm.get_end_time())
    if end_t <= 0.0:
        return np.array([0.0], dtype=float)
    ppq = float(pm.resolution or 480)
    end_q = float(pm.time_to_tick(end_t)) / ppq
    bar_q = _bar_quarters(num, den)
    offsets_q = _strong_beat_offsets_quarters(num, den)

    times: List[float] = []
    bar_idx = 0
    while True:
        bar_start_q = bar_idx * bar_q
        if bar_start_q > end_q + 1e-6:
            break
        for off_q in offsets_q:
            q = bar_start_q + off_q
            if q > end_q + 1e-6:
                break
            tick = int(round(q * ppq))
            t = float(pm.tick_to_time(tick))
            if t <= end_t + 1e-6:
                times.append(t)
        bar_idx += 1
    return np.asarray(times if times else [0.0], dtype=float)


def _compute_strong_beat_chord_tone_ratio(
    pm: pretty_midi.PrettyMIDI,
    topline: List[pretty_midi.Note],
    chord_notes: List[pretty_midi.Note],
    bass_notes: List[pretty_midi.Note],
    time_signature: Optional[Any] = None,
) -> float:
    if not topline or (not chord_notes and not bass_notes):
        return 1.0
    num, den = _resolve_time_signature(pm, time_signature)
    strong_beats = _strong_beat_times(pm, num, den)
    if len(strong_beats) == 0:
        return 1.0

    tempos = pm.get_tempo_changes()[1]
    sec_per_beat = 60.0 / float(tempos[0]) if len(tempos) else 0.5

    # Phase 8 Q5.3 — an eighth-note anticipation is credited against the following
    # downbeat chord; unmatched anticipations are ignored (never create false negatives).
    strong_notes: list[pretty_midi.Note] = []
    anticipated_matches = 0
    for n in topline:
        idx = int(np.argmin(np.abs(strong_beats - n.start)))
        dt = float(strong_beats[idx] - n.start)
        if abs(dt) <= 0.09:
            strong_notes.append(n)
        elif 0.0 < dt <= 0.55 * sec_per_beat:
            next_pcs = {
                c.pitch % 12
                for c in (chord_notes + bass_notes)
                if c.start - 0.05 <= float(strong_beats[idx]) <= c.end + 0.05
            }
            if (n.pitch % 12) in next_pcs:
                anticipated_matches += 1
    if not strong_notes:
        return 1.0

    matches = anticipated_matches
    for n in strong_notes:
        active_pcs = {
            c.pitch % 12
            for c in (chord_notes + bass_notes)
            if c.start - 0.05 <= n.start <= c.end + 0.05
        }
        if not active_pcs or (n.pitch % 12) in active_pcs:
            matches += 1
    return round(min(1.0, matches / len(strong_notes)), 3)


def _compute_chords_lead_register_gap(
    chord_notes: List[pretty_midi.Note],
    lead_notes: List[pretty_midi.Note],
) -> Optional[float]:
    """
    Phase 8 Q5.2 — Mean MIDI centroid gap (semitones) between simultaneously playing
    Chords/Pad and Lead notes (informative warning when < 4.0 st).
    """
    if not chord_notes or not lead_notes:
        return None
    gaps: List[float] = []
    for ln in lead_notes:
        active = [c.pitch for c in chord_notes if c.start - 0.05 <= ln.start <= c.end + 0.05]
        if active:
            gaps.append(float(ln.pitch) - float(np.mean(active)))
    if not gaps:
        return None
    return round(float(np.mean(gaps)), 2)


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
) -> Optional[float]:
    if len(topline) < 2 or len(bass_notes) < 2:
        return None
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
    return round(good / total, 3) if total > 0 else None


def _beat_block_4bars(
    pm: pretty_midi.PrettyMIDI,
    time_s: float,
    time_signature: Optional[Any] = None,
) -> int:
    """Map a timestamp (seconds) to its 0-indexed 4-measure block across any time signature."""
    num, den = _resolve_time_signature(pm, time_signature)
    bar_q = _bar_quarters(num, den)
    ppq = float(pm.resolution or 480)
    q = float(pm.time_to_tick(max(0.0, float(time_s)))) / ppq
    bar_idx = int((q + 1e-6) // max(bar_q, 1e-4))
    return max(0, bar_idx // 4)


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
    time_signature: Optional[Any] = None,
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
            block_idx = _beat_block_4bars(pm, s2, time_signature=time_signature)
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


def _levenshtein(seq1: List[int], seq2: List[int]) -> int:
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i, a in enumerate(seq1, start=1):
        curr = [i]
        for j, b in enumerate(seq2, start=1):
            cost = 0 if a == b else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _load_pm(pm_or_path: Any) -> pretty_midi.PrettyMIDI:
    if isinstance(pm_or_path, pretty_midi.PrettyMIDI):
        return pm_or_path
    if isinstance(pm_or_path, str):
        return pretty_midi.PrettyMIDI(pm_or_path)
    raise TypeError(f"Expected PrettyMIDI or file path, got {type(pm_or_path).__name__}")


def _note_quarter_pos(pm: pretty_midi.PrettyMIDI, time_s: float) -> float:
    ppq = float(pm.resolution or 480)
    return float(pm.time_to_tick(max(0.0, float(time_s)))) / ppq


def compute_motif_cohesion(
    pm_or_notes: Any,
    time_signature: Optional[Any] = None,
    phrase_plan: str = "",
    motif_similarity_band: Tuple[float, float] = (0.15, 0.55),
    bpm: float = 120.0,
) -> Dict[str, Any]:
    """
    Échelle S3 / Axe H (Phase 6 §3.3) :
    Segmente la piste Lead par mesure, calcule contour intervallaire + grille 1/16,
    et mesure la distance composite d(A, A'), d(A, A''), d(A, B).
    Rejette `ROBOTIC_LOOP_CLONE` (< 0.05) et `NO_MOTIF_COHESION` (> 0.75).
    """
    is_free_ambient = (phrase_plan or "").strip().lower() == "free_ambient"
    if is_free_ambient:
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "issue": None,
            "d_a_ap": None,
            "d_a_app": None,
            "d_a_b": None,
            "blocks_evaluated": 0,
            "reason": "free_ambient",
        }

    pm: Optional[pretty_midi.PrettyMIDI] = None
    if isinstance(pm_or_notes, (pretty_midi.PrettyMIDI, str)):
        pm = _load_pm(pm_or_notes)
        num, den = _resolve_time_signature(pm, time_signature)
        roles = _infer_roles(pm)
        has_explicit_lead = bool(roles["lead"]) and any(inst not in roles["chords"] for inst in roles["lead"])
        if not has_explicit_lead:
            return {
                "status": "SKIP",
                "verdict": "PASS",
                "score": 1.0,
                "issue": None,
                "d_a_ap": None,
                "d_a_app": None,
                "d_a_b": None,
                "blocks_evaluated": 0,
                "reason": "no_explicit_lead",
            }
        topline = _extract_monophonic_topline([n for i in roles["lead"] for n in i.notes])
    else:
        parsed_ts = _parse_time_signature(time_signature)
        num, den = parsed_ts if parsed_ts else (4, 4)
        topline = _extract_monophonic_topline(list(pm_or_notes or []))

    bar_q = _bar_quarters(num, den)
    sec_per_q = 60.0 / max(40.0, float(bpm))

    by_bar: Dict[int, List[Tuple[float, int]]] = {}
    for n in topline:
        q = _note_quarter_pos(pm, n.start) if pm is not None else (float(n.start) / sec_per_q)
        b_idx = max(0, int((q + 1e-6) // max(bar_q, 1e-4)))
        bar_phase = (q - b_idx * bar_q) / max(bar_q, 1e-4)
        by_bar.setdefault(b_idx, []).append((bar_phase, int(n.pitch)))

    def _bar_sig(events: List[Tuple[float, int]]) -> Tuple[List[int], np.ndarray]:
        ev_sorted = sorted(events, key=lambda x: x[0])
        pitches = [p for _, p in ev_sorted]
        contour = [pitches[k + 1] - pitches[k] for k in range(len(pitches) - 1)]
        rgrid = np.zeros(16, dtype=np.float64)
        for ph, _ in ev_sorted:
            slot = min(15, max(0, int(round(ph * 16.0))))
            rgrid[slot] = 1.0
        return contour, rgrid

    def _dist(sig1: Tuple[List[int], np.ndarray], sig2: Tuple[List[int], np.ndarray]) -> float:
        c1, r1 = sig1
        c2, r2 = sig2
        lev = _levenshtein(c1, c2) / float(max(len(c1), len(c2), 1))
        n1 = float(np.linalg.norm(r1))
        n2 = float(np.linalg.norm(r2))
        r_cos = float(np.dot(r1, r2) / (n1 * n2 + 1e-9)) if (n1 > 0 and n2 > 0) else 0.0
        return round(0.5 * lev + 0.5 * (1.0 - r_cos), 4)

    active_bars = sorted(by_bar.keys())
    if len(active_bars) < 3:
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "issue": None,
            "d_a_ap": None,
            "d_a_app": None,
            "d_a_b": None,
            "blocks_evaluated": 0,
            "reason": "fewer_than_3_active_bars",
        }

    # Evaluate 4-bar phrase blocks aligned to first active phrase or bar-0 grid
    max_bar = active_bars[-1]
    blocks: List[Tuple[int, int, int, int]] = []
    start_bar = 0
    while start_bar <= max_bar:
        if start_bar in by_bar and ((start_bar + 1) in by_bar or (start_bar + 2) in by_bar):
            blocks.append((start_bar, start_bar + 1, start_bar + 2, start_bar + 3))
            start_bar += 4
        else:
            start_bar += 1

    if not blocks:
        # Fallback: take first 4 active bars sequentially
        padded = active_bars[:4]
        while len(padded) < 4:
            padded.append(-1)
        blocks = [(padded[0], padded[1], padded[2], padded[3])]

    lo_opt, hi_opt = float(motif_similarity_band[0]), float(motif_similarity_band[1])
    hi_app_opt = max(0.60, hi_opt + 0.05)

    block_reports: List[Dict[str, Any]] = []
    clone_blocks: List[int] = []
    erratic_blocks: List[int] = []
    d_ap_vals: List[float] = []
    d_app_vals: List[float] = []
    d_b_vals: List[float] = []

    for b_a, b_ap, b_app, b_b in blocks:
        if b_a not in by_bar:
            continue
        sig_a = _bar_sig(by_bar[b_a])
        d_ap = _dist(sig_a, _bar_sig(by_bar[b_ap])) if b_ap in by_bar else None
        d_app = _dist(sig_a, _bar_sig(by_bar[b_app])) if b_app in by_bar else None
        d_b = _dist(sig_a, _bar_sig(by_bar[b_b])) if b_b in by_bar else None
        if d_ap is not None:
            d_ap_vals.append(d_ap)
        if d_app is not None:
            d_app_vals.append(d_app)
        if d_b is not None:
            d_b_vals.append(d_b)

        pairs = [v for v in (d_ap, d_app) if v is not None]
        if len(pairs) >= 2:
            if max(pairs) < 0.05:
                clone_blocks.append(b_a + 1)
            elif min(pairs) > 0.75:
                erratic_blocks.append(b_a + 1)
        elif len(pairs) == 1:
            if pairs[0] < 0.05 and (d_b is None or d_b < 0.05):
                clone_blocks.append(b_a + 1)
            elif pairs[0] > 0.75:
                erratic_blocks.append(b_a + 1)

        block_reports.append({
            "start_bar_1idx": b_a + 1,
            "d_a_ap": d_ap,
            "d_a_app": d_app,
            "d_a_b": d_b,
        })

    med_ap = round(float(np.median(d_ap_vals)), 4) if d_ap_vals else None
    med_app = round(float(np.median(d_app_vals)), 4) if d_app_vals else None
    med_b = round(float(np.median(d_b_vals)), 4) if d_b_vals else None

    issue: Optional[str] = None
    if clone_blocks:
        issue = "ROBOTIC_LOOP_CLONE"
    elif erratic_blocks:
        issue = "NO_MOTIF_COHESION"

    # Score calculation (1.0 inside optimal band, penalized outside / on failure)
    score = 1.0
    if issue == "ROBOTIC_LOOP_CLONE":
        score = 0.25
    elif issue == "NO_MOTIF_COHESION":
        score = 0.30
    else:
        for val, hi_lim in ((med_ap, hi_opt), (med_app, hi_app_opt)):
            if val is None:
                continue
            if val < lo_opt:
                score -= min(0.25, (lo_opt - val) * 1.5)
            elif val > hi_lim:
                score -= min(0.25, (val - hi_lim) * 1.2)

    return {
        "status": "FAIL" if issue else "PASS",
        "verdict": "FAIL" if issue else "PASS",
        "score": round(max(0.0, min(1.0, score)), 3),
        "issue": issue,
        "d_a_ap": med_ap,
        "d_a_app": med_app,
        "d_a_b": med_b,
        "clone_start_bars": clone_blocks,
        "erratic_start_bars": erratic_blocks,
        "blocks_evaluated": len(block_reports),
        "blocks": block_reports,
    }


def compute_harmonic_tension_curve(
    pm_or_path: Any,
    allowed_pcs: Optional[set[int]] = None,
    tonic_pc: Optional[int] = None,
    target_tension: float = 0.35,
    time_signature: Optional[Any] = None,
    phrase_plan: str = "",
) -> Dict[str, Any]:
    """
    Échelle S3 / Axe H x V (Phase 6 §3.3) :
    Calcule T_m in [0, 1] par mesure (degrés tendus b2/4/#4/b6/7, distance tonale de l'accord,
    rugosité verticale) et valide la moyenne (+-0.25) et la dynamique std >= 0.08 sur 4 mesures.
    """
    pm = _load_pm(pm_or_path)
    num, den = _resolve_time_signature(pm, time_signature)
    bar_q = _bar_quarters(num, den)
    roles = _infer_roles(pm)

    bass_notes = [n for inst in roles["bass"] for n in inst.notes]
    chord_notes = [n for inst in roles["chords"] for n in inst.notes]
    lead_notes = [n for inst in roles["lead"] for n in inst.notes]
    counter_notes = [n for inst in roles["counter"] for n in inst.notes]
    all_melodic = bass_notes + chord_notes + lead_notes + counter_notes

    if not all_melodic:
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "curve": [],
            "mean_tension": 0.0,
            "std_tension": 0.0,
            "target_tension": float(target_tension),
            "issues": [],
        }

    # Resolve tonic pitch class deterministically
    eff_tonic = tonic_pc
    if eff_tonic is None and bass_notes:
        first_bass = sorted(bass_notes, key=lambda n: n.start)
        eff_tonic = int(first_bass[0].pitch) % 12
    elif eff_tonic is None and allowed_pcs:
        eff_tonic = min(allowed_pcs)
    elif eff_tonic is None:
        eff_tonic = int(sorted(all_melodic, key=lambda n: n.start)[0].pitch) % 12

    tense_intervals = {1, 5, 6, 8, 11}  # b2, 4, #4/b5, b6, 7
    bar_notes: Dict[int, List[pretty_midi.Note]] = {}
    bar_harm_pcs: Dict[int, List[int]] = {}
    bar_bass_pcs: Dict[int, List[int]] = {}

    for n in all_melodic:
        q = _note_quarter_pos(pm, n.start)
        b_idx = max(0, int((q + 1e-6) // max(bar_q, 1e-4)))
        bar_notes.setdefault(b_idx, []).append(n)
    for n in chord_notes:
        q = _note_quarter_pos(pm, n.start)
        b_idx = max(0, int((q + 1e-6) // max(bar_q, 1e-4)))
        bar_harm_pcs.setdefault(b_idx, []).append(n.pitch % 12)
    for n in bass_notes:
        q = _note_quarter_pos(pm, n.start)
        b_idx = max(0, int((q + 1e-6) // max(bar_q, 1e-4)))
        bar_bass_pcs.setdefault(b_idx, []).append(n.pitch % 12)

    max_b = max(bar_notes.keys())
    curve: List[float] = []
    for b_idx in range(max_b + 1):
        notes_b = bar_notes.get(b_idx, [])
        if not notes_b:
            continue

        # 1) Duration-weighted proportion of tense scale degrees relative to tonic
        tot_dur = 0.0
        tense_dur = 0.0
        for n in notes_b:
            d = max(0.05, float(n.end - n.start))
            tot_dur += d
            rel = (int(n.pitch) - eff_tonic) % 12
            if rel in tense_intervals:
                weight = 1.0 if rel in (1, 6, 11) else 0.75
                tense_dur += d * weight
        tense_prop = min(1.0, tense_dur / max(tot_dur, 1e-6))

        # 2) Chord distance from tonic
        pcs_b = set(bar_harm_pcs.get(b_idx, []) + bar_bass_pcs.get(b_idx, []))
        if not pcs_b:
            pcs_b = {n.pitch % 12 for n in notes_b}
        root_pc = bar_bass_pcs[b_idx][0] if bar_bass_pcs.get(b_idx) else min(pcs_b)
        root_rel = (root_pc - eff_tonic) % 12
        rel_pcs = {(pc - eff_tonic) % 12 for pc in pcs_b}
        has_tritone = any(((p1 - p2) % 12 == 6) for p1 in pcs_b for p2 in pcs_b)
        if has_tritone or root_rel in (1, 6, 11):
            chord_dist = 0.90
        elif root_rel == 7:  # Dominant V
            chord_dist = 0.80
        elif root_rel in (2, 5, 8, 10):  # ii, iv/IV, bVI, bVII
            chord_dist = 0.52
        elif root_rel in (3, 4, 9):  # iii/bIII, vi/VI
            chord_dist = 0.38
        else:  # Tonic I / i
            chord_dist = 0.10 + (0.12 if 11 in rel_pcs or 1 in rel_pcs else 0.0)

        # 3) Vertical roughness across overlapping notes in bar
        rough_pairs = 0.0
        total_pairs = 0
        s_notes = sorted(notes_b, key=lambda x: x.start)
        for i, n1 in enumerate(s_notes):
            for n2 in s_notes[i + 1 :]:
                if n2.start >= n1.end - 0.01:
                    break
                total_pairs += 1
                iv = abs(int(n1.pitch) - int(n2.pitch)) % 12
                if iv in (1, 11):
                    rough_pairs += 1.0
                elif iv == 6:
                    rough_pairs += 0.85
                elif iv in (2, 10):
                    rough_pairs += 0.45
        roughness = min(1.0, rough_pairs / max(1, total_pairs))

        t_m = max(0.0, min(1.0, 0.35 * tense_prop + 0.40 * chord_dist + 0.25 * roughness))
        curve.append(round(t_m, 3))

    mean_t = round(float(np.mean(curve)), 3) if curve else 0.0
    std_t = round(float(np.std(curve)), 3) if len(curve) >= 2 else 0.0
    is_free_ambient = (phrase_plan or "").strip().lower() == "free_ambient"

    issues: List[str] = []
    if len(curve) >= 4 and not is_free_ambient:
        if abs(mean_t - float(target_tension)) > 0.25:
            issues.append(f"harmonic_tension_mean={mean_t} hors cible {target_tension:.2f}±0.25")
        if std_t < 0.08:
            issues.append(f"harmonic_tension_flat std={std_t} < 0.08 (tension harmonique plate)")

    score = 1.0
    if len(curve) >= 4 and not is_free_ambient:
        mean_err = abs(mean_t - float(target_tension))
        if mean_err > 0.25:
            score -= min(0.45, (mean_err - 0.25) * 1.6)
        if std_t < 0.08:
            score -= min(0.45, (0.08 - std_t) * 5.0)

    return {
        "status": "FAIL" if issues else "PASS",
        "verdict": "FAIL" if issues else "PASS",
        "score": round(max(0.0, min(1.0, score)), 3),
        "curve": curve,
        "mean_tension": mean_t,
        "std_tension": std_t,
        "target_tension": float(target_tension),
        "tonic_pc": eff_tonic,
        "issues": issues,
    }


def compute_rhythmic_interlocking(
    pm_or_path: Any,
    time_signature: Optional[Any] = None,
    max_chords_lead_overlap: float = 0.30,
    max_grid_saturation: float = 0.25,
) -> Dict[str, Any]:
    """
    Échelle S3 / Axe V x H (Phase 6 §3.3) :
    Projette les attaques Kick/Snare/Bass/Chords/Lead sur une grille 1/16 par mesure :
      - `chords_lead_attack_overlap` <= 0.30
      - `kick_bass_downbeat_lock` (quand has_drums == True)
      - `grid_saturation_ratio` <= 0.25 (> 3 rôles simultanés sur même pas 1/16)
    """
    pm = _load_pm(pm_or_path)
    num, den = _resolve_time_signature(pm, time_signature)
    bar_q = _bar_quarters(num, den)
    roles = _infer_roles(pm)

    kick_slots: set[Tuple[int, int]] = set()
    drum_other_slots: set[Tuple[int, int]] = set()
    for inst in pm.instruments:
        nm = (inst.name or "").lower()
        is_perc = bool(inst.is_drum) or bool(_PERC_TRACK_RE.search(nm))
        if not is_perc:
            continue
        for n in inst.notes:
            q = _note_quarter_pos(pm, n.start)
            b_idx = max(0, int((q + 1e-6) // max(bar_q, 1e-4)))
            ph = (q - b_idx * bar_q) / max(bar_q, 1e-4)
            slot = min(15, max(0, int(round(ph * 16.0))))
            if "kick" in nm or "bd" in nm or (inst.is_drum and n.pitch in (35, 36)):
                kick_slots.add((b_idx, slot))
            else:
                drum_other_slots.add((b_idx, slot))

    def _collect_slots(insts: List[pretty_midi.Instrument]) -> set[Tuple[int, int]]:
        s: set[Tuple[int, int]] = set()
        for inst in insts:
            for n in inst.notes:
                q = _note_quarter_pos(pm, n.start)
                b_idx = max(0, int((q + 1e-6) // max(bar_q, 1e-4)))
                ph = (q - b_idx * bar_q) / max(bar_q, 1e-4)
                slot = min(15, max(0, int(round(ph * 16.0))))
                s.add((b_idx, slot))
        return s

    bass_slots = _collect_slots(roles["bass"])
    chords_slots = _collect_slots(roles["chords"])
    lead_slots = _collect_slots(roles["lead"])
    counter_slots = _collect_slots(roles["counter"])

    chords_lead_overlap: Optional[float] = None
    if lead_slots and chords_slots:
        shared = len(lead_slots & chords_slots)
        chords_lead_overlap = round(shared / float(len(lead_slots)), 3)

    has_drums = bool(kick_slots or drum_other_slots)
    kick_bass_lock: Optional[float] = None
    if has_drums and kick_slots and bass_slots:
        # Check whether bass attacks anchor cleanly with kick (within ±1 1/16th slot for Haas/sidechain)
        bass_idx_linear = [b * 16 + s for b, s in sorted(bass_slots)]
        kick_idx_set = {b * 16 + s for b, s in kick_slots}
        locked = sum(
            1 for idx in bass_idx_linear
            if any((idx + d) in kick_idx_set for d in (-1, 0, 1))
        )
        kick_bass_lock = round(locked / float(len(bass_idx_linear)), 3)

    all_slots = kick_slots | drum_other_slots | bass_slots | chords_slots | lead_slots | counter_slots
    if all_slots:
        active_bars = {b for b, _ in all_slots}
        total_grid_steps = max(1, len(active_bars) * 16)
        saturated = 0
        for pos in all_slots:
            count = sum(
                1
                for grp in (kick_slots, drum_other_slots, bass_slots, chords_slots, lead_slots, counter_slots)
                if pos in grp
            )
            if count > 3:
                saturated += 1
        grid_saturation = round(saturated / float(total_grid_steps), 3)
    else:
        grid_saturation = 0.0

    issues: List[str] = []
    if chords_lead_overlap is not None and chords_lead_overlap > float(max_chords_lead_overlap):
        issues.append(
            f"chords_lead_attack_overlap={chords_lead_overlap} > {max_chords_lead_overlap}"
        )
    if grid_saturation > float(max_grid_saturation):
        issues.append(
            f"grid_saturation_ratio={grid_saturation} > {max_grid_saturation}"
        )

    score = 1.0
    if chords_lead_overlap is not None and chords_lead_overlap > float(max_chords_lead_overlap):
        score -= min(0.50, (chords_lead_overlap - float(max_chords_lead_overlap)) * 1.5)
    if grid_saturation > float(max_grid_saturation):
        score -= min(0.45, (grid_saturation - float(max_grid_saturation)) * 2.0)

    return {
        "status": "FAIL" if issues else "PASS",
        "verdict": "FAIL" if issues else "PASS",
        "score": round(max(0.0, min(1.0, score)), 3),
        "chords_lead_attack_overlap": chords_lead_overlap,
        "kick_bass_downbeat_lock": kick_bass_lock,
        "grid_saturation_ratio": grid_saturation,
        "has_drums": has_drums,
        "issues": issues,
    }


def evaluate_midi_musicality(
    midi_path: str,
    thresholds: Optional[MusicalityThresholds] = None,
    valence: float = 0.0,
    tension: float = 0.3,
    allowed_pcs: Optional[set[int]] = None,
    time_signature: Optional[Any] = None,
    phrase_plan: str = "",
    strict_s3: bool = False,
    s3_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Point d'entrée principal à appeler depuis audit_midi_harmony."""
    dark_tense = (valence < -0.3) or (tension >= 0.5)
    is_free_ambient = (phrase_plan or "").strip().lower() == "free_ambient"
    if thresholds is not None:
        th = thresholds
    else:
        th = MusicalityThresholds(
            strong_beat_chord_ratio_min=0.65 if (dark_tense or is_free_ambient) else 0.80,
            rest_ratio_min=0.0 if is_free_ambient else 0.20,
            rest_ratio_max=0.92 if is_free_ambient else 0.50,
        )
    pm = pretty_midi.PrettyMIDI(midi_path)
    num, den = _resolve_time_signature(pm, time_signature)
    eff_ts = (num, den)
    total_dur = pm.get_end_time()
    roles = _infer_roles(pm)

    lead_notes = [n for inst in roles["lead"] for n in inst.notes]
    chord_notes = [n for inst in roles["chords"] for n in inst.notes]
    bass_notes = [n for inst in roles["bass"] for n in inst.notes]
    topline = _extract_monophonic_topline(lead_notes)

    has_bass = bool(roles["bass"]) and len(bass_notes) >= 2
    has_explicit_lead = bool(roles["lead"]) and any(
        inst not in roles["chords"] for inst in roles["lead"]
    )

    stepwise_ratio, gap_fill, ambitus = _compute_stepwise_and_gap_fill(
        topline,
        sec_per_beat=(60.0 / float(pm.get_tempo_changes()[1][0])) if len(pm.get_tempo_changes()[1]) else None,
    )
    rest_ratio = _compute_rest_ratio(topline, total_dur)
    slices = _extract_chord_slices(chord_notes)
    vl_avg = _compute_voice_leading(slices)
    strong_beat_ratio = _compute_strong_beat_chord_tone_ratio(
        pm, topline, chord_notes, bass_notes, time_signature=eff_ts
    )
    register_gap = _compute_chords_lead_register_gap(chord_notes, lead_notes)
    overlap_conflict = _compute_lead_overlap_conflict(roles["lead"], roles["counter"])
    contrary_ratio = (
        _compute_contrary_oblique_ratio(topline, bass_notes)
        if (has_bass and has_explicit_lead)
        else None
    )

    eff_allowed = allowed_pcs
    if eff_allowed is None and dark_tense:
        raw_clashes, _ = detect_minor_2nd_clashes(
            pm, allowed_pcs=None, valence=valence, tension=tension, time_signature=eff_ts
        )
        if raw_clashes:
            eff_allowed = _infer_allowed_pcs_from_midi(midi_path)
    all_clashes, blocking_clashes = detect_minor_2nd_clashes(
        pm, allowed_pcs=eff_allowed, valence=valence, tension=tension, time_signature=eff_ts
    )

    # S3 Phase 6 metrics (always computed fast < 50 ms; enforced when strict_s3=True or s3_contract given)
    s3_cfg = s3_contract or {}
    motif_band = tuple(s3_cfg.get("motif_similarity_band") or (0.15, 0.55))
    max_cl_overlap = float(s3_cfg.get("max_chords_lead_attack_overlap", 0.30))
    motif_cohesion = compute_motif_cohesion(
        pm,
        time_signature=eff_ts,
        phrase_plan=phrase_plan,
        motif_similarity_band=(float(motif_band[0]), float(motif_band[1])),
    )
    harmonic_tension = compute_harmonic_tension_curve(
        pm,
        allowed_pcs=eff_allowed,
        target_tension=tension,
        time_signature=eff_ts,
        phrase_plan=phrase_plan,
    )
    rhythmic_interlocking = compute_rhythmic_interlocking(
        pm,
        time_signature=eff_ts,
        max_chords_lead_overlap=max_cl_overlap,
    )

    failures: List[str] = []
    warnings: List[str] = []
    if register_gap is not None and register_gap < 4.0 and has_explicit_lead and chord_notes:
        warnings.append(f"chords_lead_register_gap_st={register_gap} < 4.0 (registres Chords/Lead trop proches)")
    if has_explicit_lead:
        if not is_free_ambient and stepwise_ratio < th.stepwise_ratio_min:
            failures.append(f"stepwise_ratio={stepwise_ratio} < {th.stepwise_ratio_min} (mélodie trop sautillante)")
        if not is_free_ambient and gap_fill < th.gap_fill_min:
            failures.append(f"gap_fill_compliance={gap_fill} < {th.gap_fill_min} (sauts non résolus en sens inverse)")
        if not (th.rest_ratio_min <= rest_ratio <= th.rest_ratio_max):
            failures.append(f"melody_rest_ratio={rest_ratio} hors [{th.rest_ratio_min}, {th.rest_ratio_max}]")
        if strong_beat_ratio < th.strong_beat_chord_ratio_min:
            failures.append(f"strong_beat_chord_tone_ratio={strong_beat_ratio} < {th.strong_beat_chord_ratio_min}")
    if vl_avg > th.voice_leading_max_st:
        failures.append(f"voice_leading_avg_semitones={vl_avg} > {th.voice_leading_max_st} (accords en blocs)")
    if overlap_conflict > th.lead_overlap_max:
        failures.append(f"lead_overlap_conflict={overlap_conflict} > {th.lead_overlap_max} (cafouillis d'attaques)")
    if contrary_ratio is not None and not is_free_ambient and contrary_ratio < th.contrary_oblique_min:
        failures.append(f"contrary_or_oblique_ratio={contrary_ratio} < {th.contrary_oblique_min}")
    if blocking_clashes:
        failures.append(
            f"minor_2nd_clashes={len(blocking_clashes)} bloquants "
            f"({'<= 2 diatoniques/4 mes. autorisées' if dark_tense else '0 autorisé'})"
        )

    if strict_s3 or s3_contract is not None:
        if motif_cohesion.get("issue"):
            failures.append(
                f"motif_cohesion={motif_cohesion['issue']} (d_a_ap={motif_cohesion['d_a_ap']}, d_a_app={motif_cohesion['d_a_app']})"
            )
        failures.extend(harmonic_tension.get("issues") or [])
        failures.extend(rhythmic_interlocking.get("issues") or [])

    return {
        "musicality_verdict": "PASS" if not failures else "FAIL",
        "emotion_mode": "dark_tense" if dark_tense else "consonant",
        "time_signature": f"{num}/{den}",
        "phrase_plan": phrase_plan or "standard",
        "has_bass": has_bass,
        "has_lead": has_explicit_lead,
        "metrics": {
            "stepwise_ratio": stepwise_ratio,
            "gap_fill_compliance": gap_fill,
            "melody_ambitus_semitones": ambitus,
            "melody_rest_ratio": rest_ratio,
            "strong_beat_chord_tone_ratio": strong_beat_ratio,
            "chords_lead_register_gap_st": register_gap,
            "voice_leading_avg_semitones": vl_avg,
            "lead_overlap_conflict": overlap_conflict,
            "contrary_or_oblique_ratio": contrary_ratio,
            "minor_2nd_clashes_total": len(all_clashes),
            "minor_2nd_clashes_blocking": len(blocking_clashes),
            "motif_d_a_ap": motif_cohesion.get("d_a_ap"),
            "motif_d_a_app": motif_cohesion.get("d_a_app"),
            "motif_issue": motif_cohesion.get("issue"),
            "harmonic_tension_mean": harmonic_tension.get("mean_tension"),
            "harmonic_tension_std": harmonic_tension.get("std_tension"),
            "chords_lead_attack_overlap": rhythmic_interlocking.get("chords_lead_attack_overlap"),
            "kick_bass_downbeat_lock": rhythmic_interlocking.get("kick_bass_downbeat_lock"),
            "grid_saturation_ratio": rhythmic_interlocking.get("grid_saturation_ratio"),
        },
        "s3_diagnostics": {
            "motif_cohesion": motif_cohesion,
            "harmonic_tension": harmonic_tension,
            "rhythmic_interlocking": rhythmic_interlocking,
        },
        "minor_2nd_clashes": all_clashes,
        "blocking_minor_2nd_clashes": blocking_clashes,
        "detected_roles": {k: [i.name for i in v] for k, v in roles.items()},
        "failures": failures,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# v3.0 -- Low Interval Limit, modal clash analysis, directional metrics
# ---------------------------------------------------------------------------

# (interval in semitones) -> absolute floor of the LOW note (MIDI) allowed for that interval
LOW_INTERVAL_LIMIT: dict[int, int] = {
    1: 60, 2: 60,      # seconde mineure/majeure: >= C4 (sauf appoggiature <= 0.5 beat)
    3: 52, 4: 52,      # tierce: >= E3
    5: 48, 6: 48,      # quarte / triton: >= C3
    7: 43,             # quinte juste: >= G2
    12: 36,            # octave: >= C2
}


def low_interval_violation(p_low: int, p_high: int) -> bool:
    """True if the simultaneous pair (p_low, p_high) violates the Low Interval Limit."""
    dp = int(p_high) - int(p_low)
    if dp <= 0:
        return False
    floor = None
    if dp <= 2:
        floor = LOW_INTERVAL_LIMIT[1] if dp == 1 else LOW_INTERVAL_LIMIT[2]
    elif dp <= 4:
        floor = LOW_INTERVAL_LIMIT[3] if dp == 3 else LOW_INTERVAL_LIMIT[4]
    elif dp <= 6:
        floor = LOW_INTERVAL_LIMIT[5] if dp == 5 else LOW_INTERVAL_LIMIT[6]
    elif dp == 7:
        floor = LOW_INTERVAL_LIMIT[7]
    elif dp == 12:
        floor = LOW_INTERVAL_LIMIT[12]
    return floor is not None and int(p_low) < floor


_BII_SYMBOLS = ("bII", "bIImaj7", "bIImaj", "N6")
_DOMINANT_SYMBOLS = ("V", "V7", "V7(b9)", "V7sus4", "vii", "vii°", "vii°7", "V7#9")


def _symbol_root_pc(symbol: str) -> Optional[int]:
    m = re.match(r"^\s*([A-Ga-g])([#b]?)", symbol or "")
    if not m:
        return None
    table = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
             "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
    return table.get(m.group(1).upper() + m.group(2))


def _chord_has_third(symbol: str) -> bool:
    """False for power chords (5) and sus2/sus4; True otherwise (third must survive)."""
    sym = (symbol or "").lower()
    if sym.endswith("5") and "maj" not in sym and "m" not in sym.replace("5", "", 1):
        return False
    if "sus2" in sym or "sus4" in sym:
        return False
    return True


def analyze_harmonic_clashes(
    notes: List[Tuple[float, float, int, str]],
    chords: Optional[List[Dict[str, Any]]] = None,
    mode: str = "",
    tonic_pc: Optional[int] = None,
) -> Dict[str, Any]:
    """v3.0 mode- and spacing-aware dissonance analysis (replaces the binary minor-2nd counter).

    `notes`: (start, end, pitch, track). `chords`: [{start, end, symbol}]. Rules:
      1. a 1 st / 11 st interval between two HELD voices (> 0.5 beat) is always forbidden;
         a 13 st (b9) between bass and an upper voice is allowed in phrygian modes on bII
         and in harmonic minor on V7(b9);
      2. the leading tone (+11 st from tonic) is only allowed over a dominant chord and is
         never doubled across simultaneous tracks;
      3. every non-power, non-sus chord must keep its third (+3/+4 st from its root).
    """
    mode_l = (mode or "").lower()
    phrygian_like = mode_l in ("phrygian", "phrygian_dominant")
    harmonic_minor = mode_l == "harmonic_minor"
    hard: List[Dict[str, Any]] = []
    allowed_b9: List[Dict[str, Any]] = []
    leading_tone_issues: List[Dict[str, Any]] = []
    third_issues: List[Dict[str, Any]] = []

    notes = sorted(notes, key=lambda x: (x[0], x[2]))
    for i, (s1, e1, p1, tr1) in enumerate(notes):
        for s2, e2, p2, tr2 in notes[i + 1:]:
            if s2 >= e1 - 0.01:
                break
            if tr1 == tr2:
                continue
            diff = abs(int(p1) - int(p2))
            if diff not in (1, 11, 13):
                continue
            held = (e1 - s1) > 0.5 and (e2 - s2) > 0.5
            entry = {"time_s": round(float(min(s1, s2)), 3), "tracks": [tr1, tr2],
                     "notes": [int(p1), int(p2)], "interval_st": diff, "held": held}
            if diff in (1, 11):
                if held:
                    hard.append({**entry, "rule": "close_minor_2nd_held_forbidden"})
                # non-held (appoggiature <= 0.5 beat) close b2 is tolerated
            elif diff == 13:
                # b9 bass-vs-upper: allowed only in the modal contexts above
                chord = None
                if chords:
                    chord = next((c for c in chords if float(c["start"]) - 0.05 <= s2 <= float(c["end"]) + 0.05), None)
                sym = str((chord or {}).get("symbol") or "")
                is_bII = any(bs in sym for bs in _BII_SYMBOLS)
                is_dom = any(ds in sym for ds in _DOMINANT_SYMBOLS)
                if (phrygian_like and is_bII) or (harmonic_minor and is_dom and "b9" in sym):
                    allowed_b9.append({**entry, "rule": "b9_allowed_modal", "chord": sym})
                else:
                    hard.append({**entry, "rule": "b9_forbidden_outside_modal_context", "chord": sym})

    if tonic_pc is not None:
        lt_pc = (int(tonic_pc) + 11) % 12
        # leading tone doubling across simultaneous tracks
        lt_notes = [(s, e, p, tr) for s, e, p, tr in notes if p % 12 == lt_pc]
        for i, (s1, e1, p1, tr1) in enumerate(lt_notes):
            for s2, e2, p2, tr2 in lt_notes[i + 1:]:
                if s2 >= e1 - 0.01:
                    break
                if tr1 == tr2:
                    continue
                leading_tone_issues.append({"time_s": round(float(s2), 3), "tracks": [tr1, tr2],
                                             "rule": "leading_tone_doubled_forbidden"})
        # leading tone outside dominant chords
        if chords:
            for s, e, p, tr in lt_notes:
                chord = next((c for c in chords if float(c["start"]) - 0.05 <= s <= float(c["end"]) + 0.05), None)
                sym = str((chord or {}).get("symbol") or "")
                if not any(ds in sym for ds in _DOMINANT_SYMBOLS):
                    leading_tone_issues.append({"time_s": round(float(s), 3), "track": tr, "chord": sym,
                                                 "rule": "leading_tone_outside_dominant_forbidden"})

    if chords and tonic_pc is not None:
        for c in chords:
            sym = str(c.get("symbol") or "")
            if not _chord_has_third(sym):
                continue
            root = _symbol_root_pc(sym)
            if root is None:
                continue
            third_pcs = {(root + 3) % 12, (root + 4) % 12}
            s0, e0 = float(c["start"]), float(c["end"])
            voices = [p for s, e, p, tr in notes if float(s) >= s0 - 0.05 and float(s) <= e0 + 0.05 and p % 12 != root % 12]
            if not any(p % 12 in third_pcs for p in voices):
                third_issues.append({"time_s": round(s0, 3), "chord": sym,
                                     "rule": "chord_third_missing"})

    return {
        "hard_clashes": hard,
        "allowed_b9": allowed_b9,
        "leading_tone_issues": leading_tone_issues,
        "chord_third_issues": third_issues,
        "ok": not hard and not leading_tone_issues and not third_issues,
    }


def _bar_grid_attacks(pm: pretty_midi.PrettyMIDI, inst_notes: List[pretty_midi.Note], bar_q: float, n_bars: int, steps: int = 16) -> List[List[float]]:
    grid = [[0.0] * steps for _ in range(n_bars)]
    for n in inst_notes:
        q = _note_quarter_pos(pm, n.start)
        b = int((q + 1e-6) // max(bar_q, 1e-4))
        if 0 <= b < n_bars:
            step = int(round((q - b * bar_q) / bar_q * steps)) % steps
            grid[b][step] = 1.0
    return grid


def rhythmic_motif_recurrence(pm_or_notes: Any, n_bars: int = 4, steps: int = 16, band: Tuple[float, float] = (0.45, 0.85)) -> Dict[str, Any]:
    """v3.0: cosine similarity of the 16-step attack grid, bar 1 vs bars 2..n.

    Below `band[0]` the rhythm has no identity; above `band[1]` it is mechanically cloned.
    """
    pm = _load_pm(pm_or_notes)
    ts = _resolve_time_signature(pm, None)
    bar_q = _bar_quarters(ts[0], ts[1])
    lead = _infer_roles(pm)["lead"]
    notes = [n for inst in lead for n in inst.notes] or [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    if not notes:
        return {"ok": True, "sims": [], "band": list(band)}
    grid = _bar_grid_attacks(pm, notes, bar_q, n_bars, steps)
    sims = []
    for b in range(1, n_bars):
        a, c = grid[0], grid[b]
        na = float(np.linalg.norm(a)); nc = float(np.linalg.norm(c))
        sims.append(round(float(np.dot(a, c) / (na * nc + 1e-9)), 3) if na > 0 and nc > 0 else 0.0)
    ok = all(band[0] <= s <= band[1] for s in sims) if sims else True
    return {"ok": ok, "sims": sims, "band": list(band)}


def cadence_resolution_ok(pm_or_notes: Any, tonic_pc: int = 0, rest_pcs: Optional[set] = None, tension_pcs: Optional[set] = None) -> Dict[str, Any]:
    """v3.0: the last long note (>= 0.5 beat) of the cycle lands on a rest degree {1,3,5},
    and the half-phrase (bar 4 of an 8-bar cycle) stays on a tension degree {2,4,5,b6,7}."""
    pm = _load_pm(pm_or_notes)
    ts = _resolve_time_signature(pm, None)
    bar_q = _bar_quarters(ts[0], ts[1])
    lead = _infer_roles(pm)["lead"]
    notes = sorted([n for inst in lead for n in inst.notes] or [n for inst in pm.instruments if not inst.is_drum for n in inst.notes], key=lambda n: n.start)
    if not notes:
        return {"ok": True, "final_degree_pc": None, "half_degree_pc": None}
    rest = rest_pcs if rest_pcs is not None else {(tonic_pc + 0) % 12, (tonic_pc + 4) % 12, (tonic_pc + 7) % 12}
    tension = tension_pcs if tension_pcs is not None else {(tonic_pc + 2) % 12, (tonic_pc + 5) % 12, (tonic_pc + 7) % 12, (tonic_pc + 8) % 12, (tonic_pc + 10) % 12}
    tempos = pm.get_tempo_changes()
    bpm0 = float(tempos[1][0]) if len(tempos) > 1 and len(tempos[1]) else 120.0
    beat_s = 60.0 / bpm0
    long_notes = [n for n in notes if (n.end - n.start) >= 0.5 * beat_s]
    final = long_notes[-1] if long_notes else notes[-1]
    end_q = _note_quarter_pos(pm, final.start)
    total_bars = max(4, int(end_q // bar_q) + 1)
    half_bar = max(1, total_bars // 2)
    half_notes = [n for n in notes if half_bar * bar_q <= _note_quarter_pos(pm, n.start) < (half_bar + 1) * bar_q and (n.end - n.start) >= 0.5 * beat_s]
    half = half_notes[-1] if half_notes else None
    final_ok = (final.pitch % 12) in rest
    half_ok = True if half is None else (half.pitch % 12) in tension
    return {
        "ok": bool(final_ok and half_ok),
        "final_degree_pc": final.pitch % 12,
        "final_ok": bool(final_ok),
        "half_degree_pc": (half.pitch % 12) if half else None,
        "half_ok": bool(half_ok),
        "rest_pcs": sorted(rest),
        "tension_pcs": sorted(tension),
    }


def chord_melody_tension_alignment(
    pm_or_notes: Any,
    chords: Optional[List[Dict[str, Any]]] = None,
    tonic_pc: int = 0,
) -> Dict[str, Any]:
    """v3.0: the melodic apex (highest note of the cycle) must coincide with a
    non-tonic tension chord (iv, bVI, bII, V7...), not with the final resolution."""
    pm = _load_pm(pm_or_notes)
    lead = _infer_roles(pm)["lead"]
    notes = [n for inst in lead for n in inst.notes] or [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    if not notes:
        return {"ok": True, "apex": None}
    apex = max(notes, key=lambda n: (n.pitch, n.start))
    if chords:
        chord = next((c for c in chords if float(c["start"]) - 0.05 <= apex.start <= float(c["end"]) + 0.05), None)
        sym = str((chord or {}).get("symbol") or "")
        root = _symbol_root_pc(sym)
        tension = root is not None and (root - tonic_pc) % 12 not in (0,) and any(
            t in sym for t in ("iv", "bVI", "bII", "V7", "V", "vii")
        )
        return {"ok": bool(tension), "apex_pitch": apex.pitch, "apex_time_s": round(float(apex.start), 3),
                "chord": sym or None}
    return {"ok": True, "apex_pitch": apex.pitch, "apex_time_s": round(float(apex.start), 3), "chord": None}


# ---------------------------------------------------------------------------
# v3.1 IR-02 — entrée directe sur ScoreIR (court-circuite `_infer_roles`) :
# les rôles, la grille d'accords et les `chord_idx` viennent du compilateur.
# ---------------------------------------------------------------------------

def _scripts_dir() -> str:
    import os as _os
    d = _os.path.dirname(_os.path.abspath(__file__))
    for _ in range(4):
        cand = _os.path.join(d, "scripts")
        if _os.path.isdir(cand):
            return cand
        d = _os.path.dirname(d)
    return _os.path.join(d, "scripts")


def evaluate_score_ir(score_ir: Any, thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """v3.1 IR-02 : évalue un `ScoreIR` directement (rôles explicites, grille harmonique
    immuable, `chord_idx` par note — y compris les anticipations). Ne passe jamais par
    `_infer_roles()`."""
    import sys as _sys
    sd = _scripts_dir()
    if sd not in _sys.path:
        _sys.path.insert(0, sd)
    from score_ir import evaluate_score_ir as _eval_ir
    return _eval_ir(score_ir, thresholds)


def infer_score_ir_from_midi(midi_path: str, expected_key: str = "", bpm: Optional[float] = None) -> Any:
    """v3.1 IR-02 (compat ascendante) : construit un ScoreIR inféré depuis un MIDI externe
    (rôles via `_infer_roles`, grille à segment unique depuis la tonalité attendue/détectée)."""
    import sys as _sys
    sd = _scripts_dir()
    if sd not in _sys.path:
        _sys.path.insert(0, sd)
    import pretty_midi as _pm
    from score_ir import build_score_ir
    pm = _pm.PrettyMIDI(midi_path)
    tempo = float(bpm) if bpm else float(pm.get_tempo_changes()[1][0]) if len(pm.get_tempo_changes()[1]) else 120.0
    b2s = 60.0 / max(40.0, tempo)
    roles = _infer_roles(pm)
    tracks: Dict[str, List[Dict[str, Any]]] = {}
    for role_name, insts in roles.items():
        for inst in insts:
            tracks.setdefault(role_name, [])
            for n in inst.notes:
                tracks[role_name].append({
                    "start": round(float(n.start) / b2s, 5),
                    "length": round(max(0.02, float(n.end - n.start)) / b2s, 5),
                    "pitch": int(n.pitch),
                    "velocity": int(n.velocity),
                })
    total_beats = max(8.0, round(float(pm.get_end_time()) / b2s / 4.0) * 4.0)
    meta = {"bpm": tempo, "beats_per_bar": 4, "denominator": 4, "valence": 0.0,
            "total_bars": max(1, int(round(total_beats / 4.0))), "tonic": expected_key or "C", "key": expected_key or "C"}
    hc = {"key": expected_key or "C", "mode": "minor" if "minor" in (expected_key or "").lower() else "major"}
    return build_score_ir(hc, meta, tracks)


# ---------------------------------------------------------------------------
# v3.1 HARM-03 — couplage Timbre <-> Voicing (rugosité psychoacoustique).
# ---------------------------------------------------------------------------

def _lil_violations(voicing: List[int]) -> List[tuple]:
    out = []
    for i in range(len(voicing) - 1):
        iv = voicing[i + 1] - voicing[i]
        lim = LOW_INTERVAL_LIMIT.get(iv)
        if lim is not None and voicing[i] < lim:
            out.append((i, iv, voicing[i], lim))
    return out


def _enforce_lil(voicing: List[int], high: int = 88) -> List[int]:
    """Low Interval Limit : aucune seconde sous G3, aucune tierce sous C3, etc."""
    v = sorted(set(int(p) for p in voicing))
    for _ in range(10):
        viols = _lil_violations(v)
        if not viols:
            break
        i, _iv, low, _lim = viols[0]
        if low + 12 <= high and (low + 12) not in v:
            v[i] = low + 12
            v = sorted(set(v))
        else:
            del v[i + 1]
            if len(v) < 2:
                break
    return v


def adapt_voicing_to_timbre(
    voicing: List[int],
    timbre_features: Optional[Dict[str, Any]] = None,
    valence: float = 0.0,
    chord_root_pc: int = 0,
) -> Tuple[List[int], Dict[str, Any]]:
    """v3.1 HARM-03 : adapte un voicing d'accord au timbre qui le joue.

    - Timbre **inharmonique** (`inharmonicity > 0.15` ou famille bell/metallic/fm) :
      2 à 3 notes max, confinement `{1, 5, 8}` (offsets {0, 7, 12} vs racine d'accord),
      espacement minimal >= 5 st dans le médium-grave (< MIDI 60).
    - Timbre **riche/brillant** (`centroid > 2200 Hz` ou saw/fm/wavetable) avec >= 4 notes :
      drop-2 (2e note la plus haute abaissée d'une octave) si la LIL le permet, sinon
      shell voicing (quinte redondante retirée quand 7e/9e présentes).
    - La **Low Interval Limit** est toujours respectée en sortie.
    Retourne `(voicing, report)`.
    """
    v = sorted(set(int(p) for p in voicing))
    if len(v) < 2:
        return v, {"changes": [], "inharmonic": False, "bright": False}
    feats = dict(timbre_features or {})
    inhar = float(feats.get("inharmonicity", 0.0) or 0.0)
    family = str(feats.get("family") or feats.get("timbre_class") or feats.get("plugin") or "").lower()
    centroid = float(feats.get("spectral_centroid_hz", feats.get("centroid_hz", 0.0)) or 0.0)
    inharmonic = inhar > 0.15 or any(k in family for k in ("bell", "metallic", "fm", "matrixfm"))
    bright = centroid > 2200.0 or any(k in family for k in ("saw", "fm", "matrixfm", "supersaw", "wavetable"))
    root = int(chord_root_pc) % 12
    changes: List[str] = []

    if inharmonic and len(v) > 3:
        sel = [v[0]]
        for p in v[1:]:
            if len(sel) >= 3:
                break
            if (p - root) % 12 not in (0, 7):
                continue
            if p < 60 and p - sel[-1] < 5:
                continue
            if p - sel[-1] < 1:
                continue
            sel.append(p)
        if len(sel) < 2:
            sel = v[:3]
        v = sorted(sel)
        changes.append("inharmonic_confinement_2_3_notes")

    if bright and len(v) >= 4:
        pcs = {p % 12 for p in v}
        fifth = (root + 7) % 12
        has_ext = any((root + iv) % 12 in pcs for iv in (10, 11, 2, 3))
        if has_ext and any(p % 12 == fifth for p in v):
            v = [p for p in v if p % 12 != fifth]
            changes.append("shell_remove_fifth")
        else:
            v2 = list(v)
            v2[-2] -= 12
            v2 = sorted(set(v2))
            if not _lil_violations(v2):
                v = v2
                changes.append("drop2")
            elif any(p % 12 == fifth for p in v):
                v = [p for p in v if p % 12 != fifth]
                changes.append("shell_remove_fifth")
            else:
                v = v[:-1]
                changes.append("drop_top_note")

    v = _enforce_lil(v)
    if v != sorted(set(int(p) for p in voicing)) and not changes:
        changes.append("lil_enforced")
    return v, {"changes": changes, "inharmonic": inharmonic, "bright": bright}


# ---------------------------------------------------------------------------
# v3.2 #5a — Indice d'Attrait Mélodique (anti-loi de Goodhart) + tournoi de variantes.
# ---------------------------------------------------------------------------

def _lead_notes_of(score_ir: Any) -> list[Any]:
    for t in getattr(score_ir, "tracks", {}).values():
        if getattr(t, "role", "") == "lead" and t.notes:
            return sorted(t.notes, key=lambda n: (n.grid_beat, n.pitch))
    return []


def _entropy_bits(intervals: list[int]) -> float:
    if not intervals:
        return 0.0
    from collections import Counter
    c = Counter(intervals)
    n = len(intervals)
    h = 0.0
    for v in c.values():
        p = v / n
        h -= p * (p and __import__("math").log2(p))
    return h


def compute_melodic_appeal_index(score_ir: Any) -> Dict[str, Any]:
    """v3.2 #5a — `MelodicAppealIndex` : 4 critères POSITIFS sur [0,1], déterministe, < 15 ms.

    1. `motif_entropy_golden` : entropie des intervalles par bloc de 4 mesures dans la zone
       d'or [1.5, 2.6] bits (vrai hook avec variations mesurées ; pénalise monotonie et bruit).
    2. `contour_clarity` : corrélation du contour lead avec un template d'arche (montée puis
       descente) par période de 8 mesures — trajectoire lisible, un sommet clair.
    3. `tension_resolution` : accumulation de notes hors-accord mesures 3-4 et 7 (tension)
       suivie d'une résolution sur note d'accord en mesure 8 de chaque période.
    4. `rhythmic_complementarity` : faible corrélation des histogrammes d'onsets lead/basse
       (emboîtement / call-and-response plutôt que martèlement simultané).
    """
    import math as _math

    lead = _lead_notes_of(score_ir)
    grid = getattr(score_ir, "harmonic_grid", []) or []
    bpb = float(getattr(score_ir, "time_signature", {}).get("numerator", 4) or 4)
    total = max((float(n.grid_beat) + float(n.duration_beats) for n in lead), default=0.0) if lead else 0.0
    if not lead or total <= 0.0:
        return {"mai": 0.0, "criteria": {}, "n_notes": 0}

    # 1) entropie motivique par bloc de 4 mesures
    ent_scores = []
    block = 4 * bpb
    p = 0.0
    while p < total:
        ns = [n for n in lead if p <= float(n.grid_beat) < p + block]
        ivs = [abs(int(b.pitch) - int(a.pitch)) for a, b in zip(ns, ns[1:])]
        h = _entropy_bits(ivs)
        lo, hi = 1.5, 2.6
        if h <= 0:
            ent_scores.append(0.0)
        elif lo <= h <= hi:
            ent_scores.append(1.0)
        elif h < lo:
            ent_scores.append(max(0.0, h / lo))
        else:
            ent_scores.append(max(0.0, 1.0 - (h - hi) / 2.0))
        p += block
    c_entropy = float(sum(ent_scores) / len(ent_scores)) if ent_scores else 0.0

    # 2) clarté d'arc par période de 8 mesures (corrélation avec arche montée-descente)
    period = 8 * bpb
    arch_tpl = [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
    contour_scores = []
    p = 0.0
    while p < total:
        ns = [n for n in lead if p <= float(n.grid_beat) < p + period]
        if len(ns) >= 4:
            xs = [float(n.grid_beat) for n in ns]
            ys = [float(n.pitch) for n in ns]
            # ré-échantillonner le contour sur 8 points
            import numpy as _np
            grid_x = _np.linspace(min(xs), max(xs), 8)
            ys_r = _np.interp(grid_x, xs, ys)
            ys_r = (ys_r - ys_r.min()) / max(1e-9, (ys_r.max() - ys_r.min()))
            a = _np.asarray(arch_tpl, dtype=float)
            b = ys_r
            denom = float(_np.linalg.norm(a) * _np.linalg.norm(b)) + 1e-12
            corr = float(_np.dot(a, b) / denom)
            contour_scores.append(max(0.0, corr))
        p += period
    c_contour = float(sum(contour_scores) / len(contour_scores)) if contour_scores else 0.0

    # 3) tension -> résolution par période de 8 mesures
    tr_scores = []
    p = 0.0
    while p < total:
        pend = p + period
        ns = [n for n in lead if p <= float(n.grid_beat) < pend]
        if ns:
            tension = 0.0
            for n in ns:
                bar_in_period = (float(n.grid_beat) - p) / bpb
                cidx = n.chord_idx if 0 <= getattr(n, "chord_idx", 0) < len(grid) else 0
                seg = grid[cidx] if 0 <= cidx < len(grid) else {}
                chord_pcs = set(seg.get("chord_pcs", []))
                if int(n.pitch) % 12 not in chord_pcs and (2.0 <= bar_in_period <= 4.0 or 6.0 <= bar_in_period <= 7.5):
                    tension += 1.0
            tension_r = min(1.0, tension / 3.0)
            last = max(ns, key=lambda n: float(n.grid_beat))
            cidx = last.chord_idx if 0 <= getattr(last, "chord_idx", 0) < len(grid) else 0
            seg = grid[cidx] if 0 <= cidx < len(grid) else {}
            resolved = 1.0 if int(last.pitch) % 12 in set(seg.get("chord_pcs", [])) else 0.0
            tr_scores.append(0.5 * tension_r + 0.5 * resolved)
        p += period
    c_tension = float(sum(tr_scores) / len(tr_scores)) if tr_scores else 0.0

    # 4) complémentarité rythmique lead/basse (faible corrélation des onsets = emboîtement)
    bass_notes: list[Any] = []
    for t in getattr(score_ir, "tracks", {}).values():
        if getattr(t, "role", "") == "bass":
            bass_notes.extend(t.notes)
    if bass_notes and lead:
        import numpy as _np
        nb = int(total / 0.25) + 1
        lh = _np.zeros(nb)
        bh = _np.zeros(nb)
        for n in lead:
            lh[int(float(n.grid_beat) / 0.25)] = 1.0
        for n in bass_notes:
            bh[int(float(n.grid_beat) / 0.25)] = 1.0
        denom = float(_np.linalg.norm(lh) * _np.linalg.norm(bh)) + 1e-12
        corr = float(_np.dot(lh, bh) / denom) if denom > 1e-12 else 1.0
        c_compl = max(0.0, 1.0 - corr)
    else:
        c_compl = 0.5

    mai = 0.30 * c_entropy + 0.25 * c_contour + 0.25 * c_tension + 0.20 * c_compl
    return {
        "mai": round(float(min(1.0, max(0.0, mai))), 4),
        "criteria": {
            "motif_entropy_golden": round(c_entropy, 4),
            "contour_clarity": round(c_contour, 4),
            "tension_resolution": round(c_tension, 4),
            "rhythmic_complementarity": round(c_compl, 4),
        },
        "n_notes": len(lead),
    }


def select_best_motif_variant(score_ir: Any, variant_scores: list[tuple[Any, Dict[str, Any]]]) -> Dict[str, Any]:
    """v3.2 #5a — tournoi rapide en Tour 2 (0 rendu audio) : parmi les variantes qui passent
    les invariants durs (`evaluate_score_ir`), retenir celle qui maximise le MAI."""
    import sys as _sys
    sd = _scripts_dir()
    if sd not in _sys.path:
        _sys.path.insert(0, sd)
    from score_ir import evaluate_score_ir as _eval
    best = None
    best_mai = -1.0
    for variant, _tag in variant_scores:
        ev = _eval(variant)
        if ev.get("errors"):
            continue  # invariants durs non respectés -> inéligible
        mai = compute_melodic_appeal_index(variant)["mai"]
        if mai > best_mai:
            best_mai = mai
            best = (variant, _tag, mai, ev)
    if best is None and variant_scores:
        best = (variant_scores[0][0], variant_scores[0][1], compute_melodic_appeal_index(variant_scores[0][0])["mai"], _eval(variant_scores[0][0]))
    variant, tag, mai, ev = best
    return {"selected": tag, "mai": mai, "evaluate_passed": not ev.get("errors"), "score_ir": variant}
