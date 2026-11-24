#!/usr/bin/env python3
"""
Pluggable Listening Layer (`ListenerBackend`, RFC §3.4).

Provides two interchangeable implementations sharing the exact same concise JSON contract:
  - `DeterministicSemanticListener` (default: 0 GPU, 0 PyTorch, 100% local CPU)
  - `MultimodalAudioListener` (optional: sends short 4-8s WAV + role prompt to a
    configurable multimodal audio backend such as Gemini / Qwen-Audio / custom callable,
    falling back safely to `DeterministicSemanticListener` on misconfiguration).

Contract returned by `evaluate_track(...)`:
{
  "track": "<track_name>",
  "verdict": "PASS" | "FIX" | "SWAP",
  "score": 0.0..1.0,
  "human_diagnosis": ["...", "...", "..."],   # 1..3 short French sentences
  "next_action": {"code": "...", "params": {...}},
  "backend": "deterministic_semantic" | "multimodal_audio"
}
"""
from __future__ import annotations

import abc
import base64
import json
import os
import urllib.request
from typing import Any, Callable


VALID_VERDICTS = ("PASS", "FIX", "SWAP")


def normalize_listener_response(
    raw: dict[str, Any],
    *,
    track_name: str,
    backend_name: str,
) -> dict[str, Any]:
    """Validate and normalize any backend output to the strict RFC §6.2 contract."""
    verdict = str(raw.get("verdict", "FIX")).upper().strip()
    if verdict not in VALID_VERDICTS:
        verdict = "FIX"
    try:
        score = round(max(0.0, min(1.0, float(raw.get("score", 0.5)))), 4)
    except Exception:
        score = 0.5
    diag_raw = raw.get("human_diagnosis") or []
    if isinstance(diag_raw, str):
        diag_list = [diag_raw.strip()] if diag_raw.strip() else []
    else:
        diag_list = [str(x).strip() for x in diag_raw if str(x).strip()]
    diag_list = diag_list[:3] or ["Évaluation acoustique terminée."]

    na = raw.get("next_action")
    if not isinstance(na, dict) or not na.get("code"):
        na = {"code": "KEEP_TRACK" if verdict == "PASS" else "REVIEW_TRACK", "params": {}}
    else:
        na = {"code": str(na["code"]), "params": dict(na.get("params") or {})}

    return {
        "track": track_name,
        "verdict": verdict,
        "score": score,
        "human_diagnosis": diag_list,
        "next_action": na,
        "backend": backend_name,
    }


class ListenerBackend(abc.ABC):
    """Abstract interface for track-level listening / evaluation."""

    name: str = "base"

    @abc.abstractmethod
    def evaluate_track(
        self,
        wav_path: str,
        track_spec: dict[str, Any],
        dsp_report: dict[str, Any],
        blueprint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return normalized `verdict`, `score`, `human_diagnosis` (<= 3 lines), `next_action`."""


class DeterministicSemanticListener(ListenerBackend):
    """Default 0-GPU / 0-PyTorch deterministic listener synthesizing crisp 1-3 line diagnostics."""

    name = "deterministic_semantic"

    def evaluate_track(
        self,
        wav_path: str,
        track_spec: dict[str, Any],
        dsp_report: dict[str, Any],
        blueprint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del wav_path
        track_name = str(track_spec.get("name") or track_spec.get("role") or "Track")
        role = str(track_spec.get("role") or "lead_melody")
        ev = (blueprint or {}).get("emotion_vector") or {}
        valence = float(track_spec.get("valence", ev.get("valence", 0.0)))

        checks = dsp_report.get("checks") or {}
        metrics = dsp_report.get("metrics") or {}
        timbre_card = str(dsp_report.get("timbre_card") or "")

        # 1) Hard failure: silent / inaudible track -> SWAP / CHECK_ROUTING_OR_SWAP
        if not checks.get("audibility_ok", True):
            rms_db = metrics.get("rms_dbfs", -120.0)
            pk_db = metrics.get("peak_dbfs", -120.0)
            return normalize_listener_response(
                {
                    "verdict": "SWAP",
                    "score": 0.05,
                    "human_diagnosis": [
                        f"Piste muette ou quasi inaudible (RMS {rms_db:.1f} dBFS, crête {pk_db:.1f} dBFS).",
                        "Vérifier le routage mixeur, le chemin du sample (ou .sfz interdit) ou l'octave jouée.",
                    ],
                    "next_action": {
                        "code": "CHECK_ROUTING_OR_SWAP",
                        "params": {"track": track_name, "role": role, "rms_dbfs": rms_db},
                    },
                },
                track_name=track_name,
                backend_name=self.name,
            )

        # Accumulate prioritized findings & score deductions
        score = 1.0
        findings: list[tuple[int, str, dict[str, Any], str]] = []  # (priority, severity, action, msg)

        tuning_ok = checks.get("tuning_ok", True)
        gain_headroom_ok = checks.get("gain_headroom_ok", True)
        cents_err = metrics.get("tuning_error_cents")
        fine_tune_delta = int(metrics.get("recommended_fine_tune_delta_ct") or 0)
        chromatic_ok = checks.get("chromatic_integrity_ok", True)
        spectral_ok = checks.get("spectral_clean_ok", True)
        envelope_ok = checks.get("envelope_ok", True)
        semantic_ok = checks.get("semantic_fit_ok", True)

        centroid_hz = float(metrics.get("spectral_centroid_hz") or 0.0)
        decay_ms = metrics.get("estimated_decay_ms")
        glitches = int(metrics.get("micro_glitches") or 0)
        harsh_peak = metrics.get("worst_2_5khz_peak")

        if not chromatic_ok:
            score -= 0.45
            findings.append((
                1,
                "SWAP",
                {"code": "RESTRICT_RANGE_OR_SWAP", "params": {"track": track_name, "role": role}},
                f"Intégrité chromatique dégradée (glitches={glitches}, r_harm={metrics.get('r_harm')}) : sample inadapté à cet ambitus.",
            ))

        if not gain_headroom_ok:
            act_lufs = metrics.get("active_lufs", -50.0)
            targ_lufs = metrics.get("target_active_lufs", -20.0)
            rec_db = metrics.get("recommended_gain_db", 0.0)
            score -= 0.30
            findings.append((
                2,
                "FIX",
                {
                    "code": "BOOST_TRACK_GAIN",
                    "params": {
                        "track": track_name,
                        "role": role,
                        "active_lufs": act_lufs,
                        "target_active_lufs": targ_lufs,
                        "recommended_gain_db": rec_db,
                    },
                },
                f"Sonie active insuffisante ({act_lufs:.1f} LUFS < {targ_lufs - 10.0:.1f} LUFS) sans marge de crête suffisante -> BOOST_TRACK_GAIN / compresser source.",
            ))

        if not tuning_ok and cents_err is not None:
            abs_ct = abs(float(cents_err))
            if abs_ct > 55.0:
                score -= 0.45
                findings.append((
                    1,
                    "SWAP",
                    {"code": "SWAP_SAMPLE", "params": {"track": track_name, "role": role, "reason": "severe_detune"}},
                    f"Justesse : dérive sévère ({cents_err:+.1f} ct), sample inharmonique ou mauvais Root note.",
                ))
            else:
                score -= min(0.32, 0.14 + abs_ct / 120.0)
                findings.append((
                    2,
                    "FIX",
                    {"code": "ADJUST_FINE_TUNE", "params": {"fine_tune_delta_ct": fine_tune_delta}},
                    f"Justesse : décalage constant de {cents_err:+.1f} ct sur les notes jouées (appliquer Fine tune {fine_tune_delta:+d} ct).",
                ))

        if not semantic_ok:
            score -= 0.28
            target_lpf = int(metrics.get("recommended_lpf_hz") or (4800 if valence < -0.3 else 7500))
            findings.append((
                2,
                "FIX",
                {"code": "APPLY_LPF", "params": {"cutoff_hz": target_lpf}},
                f"Timbre perçu : trop brillant/agressif (centroïde {centroid_hz / 1000.0:.1f} kHz) pour valence {valence:+.2f} -> LPF {target_lpf} Hz.",
            ))

        if not spectral_ok:
            score -= 0.22
            if glitches > 0:
                findings.append((
                    3,
                    "FIX",
                    {"code": "INCREASE_ATTACK", "params": {"attack_ms": 8.0, "release_ms": 40.0, "declick": 2}},
                    f"Propreté : {glitches} micro-clic(s) détecté(s) -> augmenter Attack >= 8 ms et activer Declick.",
                ))
            elif harsh_peak:
                f_hz = float(harsh_peak["freq_hz"])
                ex_db = float(harsh_peak["local_excess_db"])
                cut_db = round(-min(6.0, max(2.5, ex_db - 2.0)), 1)
                findings.append((
                    3,
                    "FIX",
                    {"code": "ADD_NOTCH_EQ", "params": {"freq_hz": f_hz, "gain_db": cut_db, "q": 3.2}},
                    f"Propreté spectrale : résonance étroite à {f_hz:.0f} Hz (+{ex_db:.1f} dB) -> EQ Peak {cut_db:+.1f} dB (Q=3.2).",
                ))

        if not envelope_ok:
            score -= 0.20
            adsr_params = dict(metrics.get("recommended_adsr_params") or {"decay_ms": 220.0, "release_ms": 60.0})
            findings.append((
                4,
                "FIX",
                {"code": "ADJUST_ADSR", "params": adsr_params},
                f"Enveloppe : {metrics.get('envelope_issue', 'queue inadaptée au BPM')} -> ajuster ADSR {adsr_params}.",
            ))

        # Combine APPLY_LPF + ADJUST_FINE_TUNE cleanly when both fire (RFC §3.4 example)
        codes = {f[2]["code"] for f in findings}
        next_action: dict[str, Any]
        if "APPLY_LPF" in codes and "ADJUST_FINE_TUNE" in codes and not any(f[1] == "SWAP" for f in findings):
            lpf_p = next(f[2]["params"] for f in findings if f[2]["code"] == "APPLY_LPF")
            tune_p = next(f[2]["params"] for f in findings if f[2]["code"] == "ADJUST_FINE_TUNE")
            next_action = {
                "code": "APPLY_LPF_AND_TUNE",
                "params": {
                    "cutoff_hz": lpf_p["cutoff_hz"],
                    "fine_tune_delta_ct": tune_p["fine_tune_delta_ct"],
                },
            }
            verdict = "FIX"
        elif findings:
            findings.sort(key=lambda x: x[0])
            verdict = "SWAP" if any(f[1] == "SWAP" for f in findings) else "FIX"
            next_action = findings[0][2]
        else:
            verdict = "PASS"
            next_action = {"code": "KEEP_TRACK", "params": {"track": track_name}}

        # Build concise 1-3 line French diagnosis
        diag: list[str] = []
        if findings:
            for _prio, _sev, _act, msg in sorted(findings, key=lambda x: x[0])[:2]:
                diag.append(msg)
            if len(diag) < 3:
                if tuning_ok and cents_err is not None:
                    diag.append(f"Justesse : conforme ({cents_err:+.1f} ct <= 8 ct).")
                elif envelope_ok and decay_ms is not None:
                    diag.append(f"Enveloppe : bonne décroissance ({decay_ms:.0f} ms), aucun chevauchement.")
                elif timbre_card:
                    diag.append(f"Timbre : {timbre_card}.")
        else:
            diag.append(f"Timbre conforme au rôle '{role}' : {timbre_card or f'centroïde {centroid_hz:.0f} Hz'}.")
            if cents_err is not None:
                diag.append(f"Justesse validée ({cents_err:+.1f} ct), 0 résonance critique ni micro-clic.")
            else:
                diag.append("Spectre propre (0 résonance 2–5 kHz, 0 glitch) et enveloppe calée au BPM.")

        return normalize_listener_response(
            {
                "verdict": verdict,
                "score": max(0.0, min(1.0, score)),
                "human_diagnosis": diag[:3],
                "next_action": next_action,
            },
            track_name=track_name,
            backend_name=self.name,
        )


class MultimodalAudioListener(ListenerBackend):
    """
    Optional multimodal audio-native listener (Gemini / Qwen-Audio / custom callable).
    Accepts either:
      - `callable_fn(wav_path, prompt, track_spec, dsp_report) -> dict` (for SDK/tests)
      - `endpoint_url` + optional `api_key` (HTTP JSON POST sending base64 WAV + prompt)
    Falls back cleanly to `DeterministicSemanticListener` if unconfigured or on error.
    """

    name = "multimodal_audio"

    def __init__(
        self,
        *,
        callable_fn: Callable[[str, str, dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
        endpoint_url: str = "",
        api_key: str = "",
        model: str = "",
        timeout_s: float = 20.0,
    ) -> None:
        self.callable_fn = callable_fn
        self.endpoint_url = endpoint_url.strip() or os.environ.get("AI_STUDIO_AUDIO_LISTENER_URL", "").strip()
        self.api_key = api_key.strip() or os.environ.get("AI_STUDIO_AUDIO_LISTENER_KEY", "").strip()
        self.model = model.strip() or os.environ.get("AI_STUDIO_AUDIO_LISTENER_MODEL", "gemini-2.5-flash")
        self.timeout_s = float(timeout_s)
        self._fallback = DeterministicSemanticListener()

    def build_prompt(self, track_spec: dict[str, Any], dsp_report: dict[str, Any], blueprint: dict[str, Any] | None) -> str:
        role = track_spec.get("role", "lead_melody")
        ev = (blueprint or {}).get("emotion_vector") or {}
        return (
            f"Évalue ce rendu isolé (4-8s) pour le rôle '{role}' "
            f"(genre={(blueprint or {}).get('genre', 'unknown')}, valence={ev.get('valence', 0.0)}, "
            f"key={(blueprint or {}).get('key', 'C minor')}). "
            f"Mesures DSP : {json.dumps(dsp_report.get('metrics', {}), ensure_ascii=False)}. "
            "Réponds STRICTEMENT en JSON avec clés: verdict ('PASS'|'FIX'|'SWAP'), "
            "score (0..1), human_diagnosis (1-3 phrases courtes en français), "
            "next_action ({code, params})."
        )

    def evaluate_track(
        self,
        wav_path: str,
        track_spec: dict[str, Any],
        dsp_report: dict[str, Any],
        blueprint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        track_name = str(track_spec.get("name") or track_spec.get("role") or "Track")
        prompt = self.build_prompt(track_spec, dsp_report, blueprint)

        if self.callable_fn is not None:
            try:
                raw = self.callable_fn(wav_path, prompt, track_spec, dsp_report)
                return normalize_listener_response(raw, track_name=track_name, backend_name=self.name)
            except Exception:
                return self._fallback.evaluate_track(wav_path, track_spec, dsp_report, blueprint)

        if self.endpoint_url and os.path.isfile(wav_path):
            if not self.endpoint_url.lower().startswith("https://"):
                return self._fallback.evaluate_track(wav_path, track_spec, dsp_report, blueprint)
            max_bytes = 10 * 1024 * 1024  # 10 Mo max envoyés en base64 à l'endpoint externe
            if os.path.getsize(wav_path) > max_bytes:
                return self._fallback.evaluate_track(wav_path, track_spec, dsp_report, blueprint)
            try:
                with open(wav_path, "rb") as fh:
                    wav_b64 = base64.b64encode(fh.read()).decode("ascii")
                body = json.dumps({
                    "model": self.model,
                    "prompt": prompt,
                    "audio_wav_base64": wav_b64,
                }).encode("utf-8")
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                req = urllib.request.Request(self.endpoint_url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                return normalize_listener_response(raw, track_name=track_name, backend_name=self.name)
            except Exception:
                return self._fallback.evaluate_track(wav_path, track_spec, dsp_report, blueprint)

        return self._fallback.evaluate_track(wav_path, track_spec, dsp_report, blueprint)


def get_listener_backend(config: dict[str, Any] | None = None) -> ListenerBackend:
    """Instantiate configured `ListenerBackend` (defaults to 0-GPU `DeterministicSemanticListener`)."""
    cfg = config or {}
    kind = str(cfg.get("listener_backend") or cfg.get("backend") or "deterministic").lower().strip()
    if kind in ("multimodal", "multimodal_audio"):
        return MultimodalAudioListener(
            callable_fn=cfg.get("callable_fn"),
            endpoint_url=str(cfg.get("endpoint_url") or ""),
            api_key=str(cfg.get("api_key") or ""),
            model=str(cfg.get("model") or ""),
        )
    return DeterministicSemanticListener()
