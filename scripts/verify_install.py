#!/usr/bin/env python3
"""
v3.3 hotfix — Vérification finale post-installation AI-Studio (appelée par `install.sh`, étape 8/8).

Remplace l'ancien heredoc `python3 - <<PY` d'install.sh qui plantait avec un
`FileNotFoundError` sur `projects/active/song_blueprint.json` lors d'une installation
fraîche (aucun projet actif créé pour l'instant) — avec `set -euo pipefail`, toute
l'installation échouait à la dernière étape.

Comportement :
  - `validate_active_blueprint(studio)` : si `projects/active/song_blueprint.json`
    est ABSENT → validation ignorée proprement (normal à l'installation initiale) ;
    s'il est PRÉSENT → validation stricte contre `schemas/song_blueprint.schema.json`
    (échec = exit 1 avec message clair).
  - `handshake_servers(studio)` : handshake JSON-RPC `initialize` + `tools/list`
    sur les 5 serveurs MCP, un rapport par serveur, exit 1 si l'un échoue.

Usage :
  .venv/bin/python scripts/verify_install.py --studio "$STUDIO_DIR"
  .venv/bin/python scripts/verify_install.py --skip-mcp          # tests / CI
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MCP_SERVERS = (
    ("gridline", ("scripts/gridline-mcp.sh",)),
    ("audio-index", ("mcp-servers/audio-index/audio_index_server.py",)),
    ("audio-analyzer", ("mcp-servers/audio-analyzer/audio_analyzer_server.py",)),
    ("vocaloid", ("mcp-servers/vocaloid/vocaloid_server.py",)),
    ("reverse-engineer", ("mcp-servers/reverse-engineer/reverse_engineer_server.py",)),
)


def validate_active_blueprint(studio: str, schema_path: str | None = None) -> tuple[bool, str]:
    """Valide le blueprint actif s'il existe ; sinon signale un saut propre (pas d'exception).

    Retourne (ok, message). Lève jsonschema.ValidationError si le fichier existe mais est invalide.
    """
    import jsonschema

    schema_path = schema_path or os.path.join(studio, "schemas", "song_blueprint.schema.json")
    bp_path = os.path.join(studio, "projects", "active", "song_blueprint.json")
    if not os.path.isfile(bp_path):
        return True, (
            "SKIP: aucun projet actif (projects/active/song_blueprint.json absent) — "
            "validation de schéma ignorée, normale à l'installation initiale "
            "(créer un projet via studio_prepare_session pour la valider)."
        )
    with open(schema_path, encoding="utf-8") as fh:
        schema = json.load(fh)
    with open(bp_path, encoding="utf-8") as fh:
        bp = json.load(fh)
    jsonschema.validate(bp, schema)
    return True, f"OK: song_blueprint.json actif validé contre schemas/song_blueprint.schema.json ({bp_path})"


def handshake_servers(studio: str, venv_py: str | None = None) -> list[str]:
    """Handshake JSON-RPC initialize + tools/list sur les 5 serveurs MCP. Retourne la liste des échecs."""
    venv_py = venv_py or os.path.join(studio, ".venv", "bin", "python")
    if not os.path.isfile(venv_py):
        venv_py = sys.executable

    servers: list[list[str]] = []
    for _name, rel in MCP_SERVERS:
        if rel[0].endswith(".sh"):
            servers.append([os.path.join(studio, rel[0])])
        else:
            servers.append([venv_py, os.path.join(studio, rel[0])])

    init_msg = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                           "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                      "clientInfo": {"name": "install-check", "version": "1.0"}}}) + "\n"
    notif_msg = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
    tools_msg = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}) + "\n"

    failures: list[str] = []
    for cmd in servers:
        label = os.path.basename(cmd[-1])
        try:
            env = os.environ.copy()
            env["AI_STUDIO_DIR"] = studio
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, env=env, text=True, bufsize=1)
            try:
                proc.stdin.write(init_msg)
                proc.stdin.flush()
                r1 = json.loads(proc.stdout.readline())
                if "result" not in r1:
                    raise RuntimeError(f"échec initialize: {r1}")
                proc.stdin.write(notif_msg + tools_msg)
                proc.stdin.flush()
                r2 = json.loads(proc.stdout.readline())
                t_count = len(r2.get("result", {}).get("tools", []))
                if t_count <= 0:
                    raise RuntimeError(f"aucun outil exposé: {r2}")
                server_name = r1["result"].get("serverInfo", {}).get("name", "?")
                print(f"MCP OK: {label} ({server_name}, tools={t_count})")
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                proc.wait(timeout=10)
        except Exception as exc:
            failures.append(f"{label}: {exc}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description="Vérification finale post-installation AI-Studio")
    ap.add_argument("--studio", default=ROOT, help="Racine du studio (défaut: AI_STUDIO_DIR ou parent du script)")
    ap.add_argument("--skip-blueprint", action="store_true", help="Ne pas valider projects/active/song_blueprint.json")
    ap.add_argument("--skip-mcp", action="store_true", help="Ne pas faire le handshake des serveurs MCP")
    args = ap.parse_args()
    studio = os.path.abspath(os.path.expanduser(args.studio))

    rc = 0
    if not args.skip_blueprint:
        try:
            ok, msg = validate_active_blueprint(studio)
            print(msg)
            if not ok:
                rc = 1
        except Exception as exc:
            print(f"ÉCHEC validation song_blueprint.json actif: {exc}")
            rc = 1
    if not args.skip_mcp:
        failures = handshake_servers(studio)
        if failures:
            for f in failures:
                print(f"MCP ÉCHEC: {f}")
            print("Validation handshake & tools/list des serveurs MCP : ÉCHEC")
            rc = 1
        else:
            print("Validation JSON Schema + handshake & tools/list des 5 serveurs MCP : OK")
    return rc


if __name__ == "__main__":
    sys.exit(main())
