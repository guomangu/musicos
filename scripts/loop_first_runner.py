#!/usr/bin/env python3
"""
Deterministic State Machine Orchestrator for Small LLMs (`loop_first_runner.py`, RFC §6).

Guides any AI agent step-by-step through the Loop-First pipeline:
  `E0_PALETTE` -> `E1_SOUNDCHECK` -> `E2_VERTICAL` (`V1_LOW_END -> V2_GROOVE -> V3_HARMONY -> V4_LEAD`)
  -> `E3_HORIZONTAL` -> `E4_FINAL_GATE` -> `DONE`

Persists `<proj>/pipeline_state.json` and enforces:
  - Automatic `user_samples_locked = True` when `<proj>/Samples/` contains user audio files
  - Strict `forbidden_tracks` fault-isolation guardrail during `E2_VERTICAL`
  - Max `2` fix retries per track/stage before mandatory escalation to `SWAP_SAMPLE`
  - Single explicit `next_mandatory_step` payload so modest LLMs never lose track.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
INDEX_DIR = os.path.join(ROOT_DIR, "mcp-servers", "audio-index")
ANALYZER_DIR = os.path.join(ROOT_DIR, "mcp-servers", "audio-analyzer")
for _d in (SCRIPTS_DIR, INDEX_DIR, ANALYZER_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from deep_diagnostics import VERTICAL_STAGES
from blueprint_safety import summarize_musical_safety
from indexer import AUDIO_EXTS

PIPELINE_STAGES = (
    "E0_PALETTE",
    "E1_SOUNDCHECK",
    "E2_VERTICAL",
    "AWAITING_E2_CHECKPOINT",
    "E3_HORIZONTAL",
    "E4_FINAL_GATE",
    "DONE",
)

DEFAULT_ALLOWED_ACTIONS_BY_STAGE: dict[str, list[str]] = {
    "V1_LOW_END": ["ADJUST_SIDECHAIN_OR_POCKET", "MONO_LOW_END", "RAISE_SUB_OCTAVE_OR_HPF", "SWAP_SAMPLE"],
    "V2_GROOVE": ["TAME_DRUMS_OR_HATS", "APPLY_ANTI_MACHINE_GUN", "SWAP_SAMPLE"],
    "V3_HARMONY": ["HPF_180Hz", "PEAK_CUT_300Hz_-3.5dB", "STABILIZE_HARMONY_PHASE", "FIX_HARMONY_CHROMA_CLASH", "SWAP_SAMPLE"],
    "V4_LEAD": ["CARVE_LEAD_POCKET", "ADJUST_LEAD_TONAL_BALANCE", "SWAP_SAMPLE"],
}


def detect_user_samples_locked(project_dir: str) -> bool:
    """Return True if `<project_dir>/Samples/` exists and contains at least one audio file."""
    samples_upper = os.path.join(os.path.abspath(project_dir), "Samples")
    if not os.path.isdir(samples_upper):
        return False
    for _root, _dirs, fns in os.walk(samples_upper):
        for fn in fns:
            if not fn.startswith(".") and os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                return True
    return False


class LoopFirstStateMachine:
    """Persistent state machine stored in `<project_dir>/pipeline_state.json`."""

    def __init__(
        self,
        project_dir: str,
        *,
        max_attempts: int = 2,
        track_stage_map: dict[str, str] | None = None,
        has_drums: bool | None = None,
        e2_checkpoint: bool | None = None,
    ) -> None:
        self.project_dir = os.path.abspath(os.path.expanduser(project_dir))
        self.state_path = os.path.join(self.project_dir, "pipeline_state.json")
        self.max_attempts = max(1, int(max_attempts))

        self._dirty_hashes = False
        default_contract = os.path.join(self.project_dir, "multi_scale_contract.json")
        auto_contract = default_contract if os.path.isfile(default_contract) else None

        if os.path.isfile(self.state_path):
            with open(self.state_path, encoding="utf-8") as fh:
                self.state: dict[str, Any] = json.load(fh)
            self.state.setdefault("artifact_hashes", {})
            self.state.setdefault("contract_path", auto_contract)
            self.state.setdefault("multiscale_scores", {})
            self.state.setdefault("e2_checkpoint_enabled", False)
            self.state.setdefault("e2_checkpoint_approved", False)
            self.state.setdefault("e2_checkpoint_wav", None)
            if e2_checkpoint is not None:
                self.state["e2_checkpoint_enabled"] = bool(e2_checkpoint)
            if detect_user_samples_locked(self.project_dir):
                self.state["user_samples_locked"] = True
        else:
            bp_has_drums = self._infer_has_drums_from_blueprint()
            eff_has_drums = bp_has_drums if has_drums is None else bool(has_drums)
            self.state = {
                "project_dir": self.project_dir,
                "stage": "E0_PALETTE",
                "substage": None,
                "has_drums": eff_has_drums,
                "user_samples_locked": detect_user_samples_locked(self.project_dir),
                "selected_palette_id": None,
                "contract_path": auto_contract,
                "core_loop_locked": False,
                "e2_checkpoint_enabled": bool(e2_checkpoint) if e2_checkpoint is not None else False,
                "e2_checkpoint_approved": False,
                "e2_checkpoint_wav": None,
                "max_attempts": self.max_attempts,
                "track_stage_map": dict(track_stage_map or {}),
                "soundcheck_results": {},
                "vertical_results": {},
                "horizontal_seams": {},
                "multiscale_scores": {},
                "artifact_hashes": {},
                "final_gate_passed": False,
                "next_mandatory_step": {},
            }
            self._recompute_next_step()
            self.save()

    def _infer_has_drums_from_blueprint(self) -> bool:
        bp_path = os.path.join(self.project_dir, "song_blueprint.json")
        if os.path.isfile(bp_path):
            try:
                with open(bp_path, encoding="utf-8") as fh:
                    bp = json.load(fh)
                if "has_drums" in bp:
                    return bool(bp["has_drums"])
                return str(bp.get("mix_profile", "trap_rap")) != "classical_ambient"
            except Exception:
                pass
        return True

    def save(self) -> str:
        os.makedirs(self.project_dir, exist_ok=True)
        if not self._dirty_hashes and os.path.isfile(self.state_path):
            try:
                with open(self.state_path, encoding="utf-8") as fh:
                    disk_state = json.load(fh)
                if isinstance(disk_state, dict) and "artifact_hashes" in disk_state:
                    self.state["artifact_hashes"] = disk_state["artifact_hashes"]
            except Exception:
                pass
        self._dirty_hashes = False
        with open(self.state_path, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, indent=2, ensure_ascii=False)
        return self.state_path

    def _tracks_for_stage(self, stage_name: str) -> list[str]:
        tmap = self.state.get("track_stage_map") or {}
        return [t for t, st in tmap.items() if st == stage_name]

    def _forbidden_tracks_for_vertical(self, current_stage: str) -> list[str]:
        order = {s: i for i, s in enumerate(VERTICAL_STAGES)}
        cur_idx = order.get(current_stage, 0)
        tmap = self.state.get("track_stage_map") or {}
        locked: list[str] = []
        for t, st in tmap.items():
            if order.get(st, 99) < cur_idx:
                locked.append(t)
        return locked

    def _recompute_next_step(self) -> None:
        st = self.state["stage"]
        if st == "E0_PALETTE":
            self.state["substage"] = None
            tool = "index_project_samples_then_propose_palettes" if self.state["user_samples_locked"] else "propose_palettes"
            self.state["next_mandatory_step"] = {
                "tool": tool,
                "target_track": None,
                "allowed_actions": ["SELECT_PALETTE_A", "SELECT_PALETTE_B", "SELECT_PALETTE_C"],
                "forbidden_tracks": [],
            }
            return

        if st == "E1_SOUNDCHECK":
            sc = self.state.get("soundcheck_results") or {}
            # Find any failing or pending track
            pending = [t for t, info in sc.items() if info.get("status") != "PASS"]
            if not pending and sc:
                self.state["stage"] = "E2_VERTICAL"
                self.state["substage"] = "V1_LOW_END"
                self._recompute_next_step()
                return
            target = pending[0] if pending else None
            attempts = int(sc[target]["attempts"]) if target else 0
            escalate = target is not None and attempts >= int(self.state.get("max_attempts", self.max_attempts))
            self.state["substage"] = target
            self.state["next_mandatory_step"] = {
                "tool": "swap_sample_and_soundcheck" if escalate else "audit_track_soundcheck",
                "target_track": target,
                "allowed_actions": ["SWAP_SAMPLE"] if escalate else ["ADJUST_ROOT_NOTE", "ADJUST_FINE_TUNE", "APPLY_LPF", "ADD_NOTCH_EQ", "ADJUST_ADSR", "SWAP_SAMPLE"],
                "forbidden_tracks": [t for t, info in sc.items() if info.get("status") == "PASS"],
            }
            return

        if st == "E2_VERTICAL":
            vr = self.state.get("vertical_results") or {}
            has_drums = bool(self.state.get("has_drums", True))
            next_v: str | None = None
            for vst in VERTICAL_STAGES:
                if vst == "V2_GROOVE" and not has_drums:
                    if vst not in vr:
                        vr[vst] = {"status": "PASS", "attempts": 0, "skipped": True}
                    continue
                if (vr.get(vst) or {}).get("status") != "PASS":
                    next_v = vst
                    break

            if next_v is None:
                self.state["core_loop_locked"] = True
                if self.state.get("e2_checkpoint_enabled") and not self.state.get("e2_checkpoint_approved"):
                    self.state["stage"] = "AWAITING_E2_CHECKPOINT"
                    self.state["substage"] = "HUMAN_LISTEN_15S"
                    self.state["next_mandatory_step"] = {
                        "tool": "await_human_checkpoint",
                        "stage": "AWAITING_E2_CHECKPOINT",
                        "target_track": self.state.get("e2_checkpoint_wav"),
                        "allowed_actions": ["RESUME_FROM_CHECKPOINT", "ADJUST_CORE_LOOP"],
                        "forbidden_tracks": [],
                    }
                    return
                self.state["stage"] = "E3_HORIZONTAL"
                self.state["substage"] = "SEAMS"
                self._recompute_next_step()
                return

            self.state["substage"] = next_v
            cur_info = vr.get(next_v) or {}
            attempts = int(cur_info.get("attempts", 0))
            escalate = cur_info.get("status") in ("FAIL", "FIX", "SWAP") and attempts >= int(self.state.get("max_attempts", self.max_attempts))
            layer_tracks = self._tracks_for_stage(next_v)
            target_track = layer_tracks[0] if layer_tracks else next_v
            forbidden = self._forbidden_tracks_for_vertical(next_v)
            allowed = ["SWAP_SAMPLE"] if escalate else DEFAULT_ALLOWED_ACTIONS_BY_STAGE.get(next_v, ["SWAP_SAMPLE"])

            self.state["next_mandatory_step"] = {
                "tool": "swap_layer_and_retest" if escalate else ("apply_fix_and_retest" if cur_info.get("status") else "audit_vertical_step"),
                "stage": next_v,
                "target_track": target_track,
                "target_layer_tracks": layer_tracks,
                "allowed_actions": allowed,
                "forbidden_tracks": forbidden,
            }
            return

        if st == "AWAITING_E2_CHECKPOINT":
            if self.state.get("e2_checkpoint_approved"):
                self.state["core_loop_locked"] = True
                self.state["stage"] = "E3_HORIZONTAL"
                self.state["substage"] = "SEAMS"
                self._recompute_next_step()
                return
            self.state["substage"] = "HUMAN_LISTEN_15S"
            self.state["next_mandatory_step"] = {
                "tool": "await_human_checkpoint",
                "stage": "AWAITING_E2_CHECKPOINT",
                "target_track": self.state.get("e2_checkpoint_wav"),
                "allowed_actions": ["RESUME_FROM_CHECKPOINT", "ADJUST_CORE_LOOP"],
                "forbidden_tracks": [],
            }
            return

        if st == "E3_HORIZONTAL":
            hs = self.state.get("horizontal_seams") or {}
            failed = [k for k, v in hs.items() if v.get("status") != "PASS"]
            if hs and not failed:
                self.state["stage"] = "E4_FINAL_GATE"
                self.state["substage"] = "RENDER_AND_AUDIT"
                self._recompute_next_step()
                return
            self.state["next_mandatory_step"] = {
                "tool": "audit_horizontal_seam" if not failed else "fix_seam_and_retest",
                "target_track": failed[0] if failed else None,
                "allowed_actions": ["ADD_BEAT4_DROPOUT", "ADD_FILTER_SWEEP", "ADJUST_SECTION_GAIN_ARC", "APPLY_OUTRO_FADE"],
                "forbidden_tracks": list((self.state.get("track_stage_map") or {}).keys()),
            }
            return

        if st == "E4_FINAL_GATE":
            if self.state.get("final_gate_passed"):
                self.state["stage"] = "DONE"
                self.state["substage"] = None
                self.state["next_mandatory_step"] = {
                    "tool": "none",
                    "target_track": None,
                    "allowed_actions": [],
                    "forbidden_tracks": [],
                }
            else:
                self.state["next_mandatory_step"] = {
                    "tool": "render_and_audit",
                    "target_track": None,
                    "allowed_actions": ["RENDER_FINAL_MP3_AND_AUTOFIX"],
                    "forbidden_tracks": [],
                }
            return

        # DONE
        self.state["next_mandatory_step"] = {
            "tool": "none",
            "target_track": None,
            "allowed_actions": [],
            "forbidden_tracks": [],
        }

    # --- State Transitions ---

    def apply_palette_to_blueprint(self, palette: dict[str, Any], blueprint_path: str = "") -> dict[str, Any]:
        """Merge a `propose_palettes` palette's `blueprint_tracks` (carrying detected
        `root_note` / `fine_tune_ct`) into `<projet>/song_blueprint.json` `tracks[]`.

        This is what guarantees a `gl.sampler` track is never played from a wrong C4
        root (systematic dissonance on bells/keys otherwise).
        """
        bp_path = os.path.abspath(os.path.expanduser(blueprint_path)) if blueprint_path else os.path.join(self.project_dir, "song_blueprint.json")
        bp: dict[str, Any] = {}
        if os.path.isfile(bp_path):
            with open(bp_path, encoding="utf-8") as fh:
                bp = json.load(fh)
        tracks = bp.setdefault("tracks", [])
        if not isinstance(tracks, list):
            tracks = []
            bp["tracks"] = tracks
        by_role = {str(t.get("role", "")).lower(): t for t in tracks if isinstance(t, dict)}
        merged = 0
        for tr in (palette.get("blueprint_tracks") or []):
            role = str(tr.get("role", "")).lower()
            if role and role in by_role:
                by_role[role].update(tr)
            else:
                tracks.append(dict(tr))
            merged += 1
        tmp = bp_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(bp, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, bp_path)
        cat_path = os.path.join(os.path.dirname(bp_path), "samples_catalog.json")
        safety = summarize_musical_safety(bp, catalog_path=cat_path if os.path.isfile(cat_path) else None)
        self.state["palette_blueprint_written"] = True
        self.state["palette_blueprint_tracks"] = merged
        self.state["musical_safety"] = safety
        self.save()
        return {"blueprint_path": bp_path, "tracks_merged": merged, "musical_safety": safety}

    def select_palette(
        self,
        palette_id: str,
        tracks: list[str] | None = None,
        track_stage_map: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Record selected palette from E0 and transition to E1_SOUNDCHECK."""
        self.state["selected_palette_id"] = str(palette_id)
        if track_stage_map:
            self.state["track_stage_map"] = dict(track_stage_map)
        if tracks:
            for t in tracks:
                self.state["soundcheck_results"].setdefault(t, {"status": "PENDING", "attempts": 0})
                if t not in self.state["track_stage_map"]:
                    from vertical_tester import classify_channel_stage
                    self.state["track_stage_map"][t] = classify_channel_stage(t)
        self.state["stage"] = "E1_SOUNDCHECK"
        self._recompute_next_step()
        self.save()
        return self.state

    def record_soundcheck(
        self,
        track_name: str,
        verdict: str,
        blocker: str = "",
        stage_hint: str = "",
    ) -> dict[str, Any]:
        """Record an E1 soundcheck attempt (`PASS | FIX | SWAP`); advances to E2 when all tracks PASS."""
        v = str(verdict).upper().strip()
        sc = self.state.setdefault("soundcheck_results", {})
        prev = sc.get(track_name) or {"status": "PENDING", "attempts": 0}
        attempts = int(prev.get("attempts", 0)) + 1
        entry: dict[str, Any] = {"status": "PASS" if v == "PASS" else v, "attempts": attempts}
        if blocker and v != "PASS":
            entry["blocker"] = blocker
        sc[track_name] = entry
        if track_name not in self.state["track_stage_map"]:
            from vertical_tester import classify_channel_stage
            self.state["track_stage_map"][track_name] = stage_hint or classify_channel_stage(track_name)
        self.state["stage"] = "E1_SOUNDCHECK"
        self._recompute_next_step()
        self.save()
        return self.state

    def record_vertical(
        self,
        stage_name: str,
        verdict: str,
        blocker: str = "",
    ) -> dict[str, Any]:
        """Record an E2 vertical stage result (`V1_LOW_END..V4_LEAD`) with retry tracking."""
        st = str(stage_name).upper().strip()
        if st not in VERTICAL_STAGES:
            raise ValueError(f"Unknown vertical stage: {stage_name}")
        v = str(verdict).upper().strip()
        vr = self.state.setdefault("vertical_results", {})
        prev = vr.get(st) or {"attempts": 0}
        attempts = int(prev.get("attempts", 0)) + 1
        entry: dict[str, Any] = {"status": "PASS" if v == "PASS" else "FAIL", "attempts": attempts}
        if blocker and v != "PASS":
            entry["blocker"] = blocker
        vr[st] = entry
        self.state["stage"] = "E2_VERTICAL"
        self._recompute_next_step()
        self.save()
        return self.state

    def record_horizontal(
        self,
        seam_name: str,
        verdict: str,
        blocker: str = "",
    ) -> dict[str, Any]:
        """Record an E3 horizontal seam/outro result."""
        v = str(verdict).upper().strip()
        hs = self.state.setdefault("horizontal_seams", {})
        prev = hs.get(seam_name) or {"attempts": 0}
        entry: dict[str, Any] = {
            "status": "PASS" if v == "PASS" else "FAIL",
            "attempts": int(prev.get("attempts", 0)) + 1,
        }
        if blocker and v != "PASS":
            entry["blocker"] = blocker
        hs[seam_name] = entry
        self.state["stage"] = "E3_HORIZONTAL"
        self._recompute_next_step()
        self.save()
        return self.state

    def record_final_gate(self, passed: bool) -> dict[str, Any]:
        self.state["final_gate_passed"] = bool(passed)
        self.state["stage"] = "E4_FINAL_GATE"
        self._recompute_next_step()
        self.save()
        return self.state

    def set_contract_path(self, contract_path: str) -> dict[str, Any]:
        self.state["contract_path"] = os.path.abspath(os.path.expanduser(contract_path))
        self.save()
        return self.state

    def trigger_e2_checkpoint(self, wav_path: str = "", summary_lines: list[str] | None = None) -> dict[str, Any]:
        """Mark core loop as locked at end of E2 and pause in AWAITING_E2_CHECKPOINT."""
        self.state["e2_checkpoint_enabled"] = True
        self.state["e2_checkpoint_approved"] = False
        self.state["core_loop_locked"] = True
        if wav_path:
            self.state["e2_checkpoint_wav"] = wav_path
        if summary_lines is not None:
            self.state["e2_checkpoint_summary"] = list(summary_lines)[:5]
        vr = self.state.setdefault("vertical_results", {})
        for vst in VERTICAL_STAGES:
            if vst not in vr:
                vr[vst] = {"status": "PASS", "attempts": 1}
        self.state["stage"] = "AWAITING_E2_CHECKPOINT"
        self._recompute_next_step()
        self.save()
        return self.state

    def resume_from_checkpoint(self, adjustment_notes: str = "") -> dict[str, Any]:
        """Approve the E2 15s human checkpoint and advance directly to E3_HORIZONTAL preserving SHA-256 cache."""
        self.state["e2_checkpoint_approved"] = True
        self.state["core_loop_locked"] = True
        if adjustment_notes:
            self.state["e2_checkpoint_notes"] = adjustment_notes
        if self.state.get("stage") in ("E0_PALETTE", "E1_SOUNDCHECK", "E2_VERTICAL", "AWAITING_E2_CHECKPOINT"):
            self.state["stage"] = "E3_HORIZONTAL"
            self.state["substage"] = "SEAMS"
        self._recompute_next_step()
        self.save()
        return self.state

    def record_multiscale_scores(self, stage: str, pyramid_output: dict[str, Any]) -> dict[str, Any]:
        ms = self.state.setdefault("multiscale_scores", {})
        ms[str(stage)] = {
            "verdict": pyramid_output.get("verdict"),
            "blocking_scale": pyramid_output.get("blocking_scale"),
            "global_alignment_score": pyramid_output.get("global_alignment_score"),
            "scale_matrix": pyramid_output.get("scale_matrix"),
            "full_report_path": pyramid_output.get("full_report_path"),
        }
        self.save()
        return self.state

    def set_artifact_hash(self, key: str, hash_val: str, result: dict[str, Any] | None = None) -> dict[str, Any]:
        ah = self.state.setdefault("artifact_hashes", {})
        ah[str(key)] = {
            "hash": str(hash_val),
            "verdict": (result or {}).get("verdict"),
            "result": result or {},
        }
        self._dirty_hashes = True
        self.save()
        return self.state

    def get_artifact_hash(self, key: str) -> dict[str, Any] | None:
        return (self.state.get("artifact_hashes") or {}).get(str(key))

    def validate_edit_target(self, track_name: str, *, raise_on_forbidden: bool = True) -> dict[str, Any]:
        """
        Enforce fault-isolation guardrail: block edits on tracks in `forbidden_tracks`
        during E2_VERTICAL (or already-locked timbres in E3_HORIZONTAL).
        """
        forbidden = list((self.state.get("next_mandatory_step") or {}).get("forbidden_tracks") or [])
        low_forb = {t.lower() for t in forbidden}
        allowed = track_name.lower() not in low_forb
        if not allowed and raise_on_forbidden:
            raise PermissionError(
                f"Fault-isolation guardrail: editing locked track '{track_name}' is FORBIDDEN during "
                f"{self.state['stage']}/{self.state['substage']}. Forbidden tracks: {forbidden}"
            )
        return {
            "track": track_name,
            "allowed": allowed,
            "stage": self.state["stage"],
            "substage": self.state["substage"],
            "forbidden_tracks": forbidden,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Loop-First deterministic state machine runner (`pipeline_state.json`).")
    ap.add_argument("project_dir", help="Project directory containing pipeline_state.json")
    ap.add_argument("--checkpoint-e2", action="store_true", help="Enable pause at end of E2 in AWAITING_E2_CHECKPOINT")
    ap.add_argument("--resume-from-checkpoint", action="store_true", help="Approve E2 checkpoint and advance to E3_HORIZONTAL")
    sub = ap.add_subparsers(dest="cmd", required=False)

    sub.add_parser("status", help="Print current pipeline_state.json")

    p_pal = sub.add_parser("select-palette", help="Record E0 palette selection and move to E1")
    p_pal.add_argument("--palette-id", required=True)
    p_pal.add_argument("--tracks", nargs="*", default=[])

    p_sc = sub.add_parser("record-soundcheck", help="Record E1 soundcheck result for a track")
    p_sc.add_argument("--track", required=True)
    p_sc.add_argument("--verdict", required=True, choices=["PASS", "FIX", "SWAP"])
    p_sc.add_argument("--blocker", default="")

    p_v = sub.add_parser("record-vertical", help="Record E2 vertical stage result")
    p_v.add_argument("--stage", required=True, choices=list(VERTICAL_STAGES))
    p_v.add_argument("--verdict", required=True, choices=["PASS", "FAIL", "FIX", "SWAP"])
    p_v.add_argument("--blocker", default="")

    p_h = sub.add_parser("record-horizontal", help="Record E3 seam/outro result")
    p_h.add_argument("--seam", required=True)
    p_h.add_argument("--verdict", required=True, choices=["PASS", "FAIL", "FIX"])
    p_h.add_argument("--blocker", default="")

    p_chk = sub.add_parser("validate-target", help="Check if editing a track is allowed by isolation guardrail")
    p_chk.add_argument("--track", required=True)

    p_res = sub.add_parser("resume-checkpoint", help="Approve E2 checkpoint and move to E3_HORIZONTAL")
    p_res.add_argument("--notes", default="")

    args = ap.parse_args()
    sm = LoopFirstStateMachine(args.project_dir, e2_checkpoint=True if args.checkpoint_e2 else None)
    if args.resume_from_checkpoint:
        out = sm.resume_from_checkpoint()
    elif args.cmd in (None, "status"):
        out = sm.state
    elif args.cmd == "select-palette":
        out = sm.select_palette(args.palette_id, tracks=args.tracks)
    elif args.cmd == "record-soundcheck":
        out = sm.record_soundcheck(args.track, args.verdict, blocker=args.blocker)
    elif args.cmd == "record-vertical":
        out = sm.record_vertical(args.stage, args.verdict, blocker=args.blocker)
    elif args.cmd == "record-horizontal":
        out = sm.record_horizontal(args.seam, args.verdict, blocker=args.blocker)
    elif args.cmd == "validate-target":
        out = sm.validate_edit_target(args.track, raise_on_forbidden=False)
    elif args.cmd == "resume-checkpoint":
        out = sm.resume_from_checkpoint(adjustment_notes=args.notes)
    else:
        out = sm.state
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
