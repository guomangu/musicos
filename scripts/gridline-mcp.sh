#!/usr/bin/env bash
set -euo pipefail

STUDIO_DIR="${AI_STUDIO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ACTIVE_PROJECT="${GRIDLINE_PROJECT:-$STUDIO_DIR/projects/active/projet.gridline}"
EXPORT_DIR="${GRIDLINE_EXPORT_DIR:-$STUDIO_DIR/projects/active/renders}"

mkdir -p "$(dirname "$ACTIVE_PROJECT")" "$EXPORT_DIR"

# Si Gridline GUI tourne et expose son port MCP local (127.0.0.1:47390),
# on s'y attache en direct pour modifier la session ouverte à l'écran.
if python3 -c "import socket; s=socket.socket(); s.settimeout(0.3); s.connect(('127.0.0.1', 47390)); s.close()" 2>/dev/null; then
  exec gridline mcp --attach
fi

# Sinon, si le fichier projet n'existe pas encore, on le crée depuis le template propre.
# Ordre de repli : $GRIDLINE_EMPTY_TEMPLATE > template système Gridline > template embarqué (assets/).
if [ ! -f "$ACTIVE_PROJECT" ]; then
  TEMPLATE="${GRIDLINE_EMPTY_TEMPLATE:-}"
  if [ -z "$TEMPLATE" ] || [ ! -f "$TEMPLATE" ]; then
    TEMPLATE="/usr/share/gridline/assets/templates/Basics/Empty with mixer tracks.gridline"
  fi
  if [ ! -f "$TEMPLATE" ]; then
    TEMPLATE="$STUDIO_DIR/assets/empty_with_mixer_tracks.gridline"
  fi
  if [ -f "$TEMPLATE" ]; then
    cp "$TEMPLATE" "$ACTIVE_PROJECT"
  else
    echo "[WARN] Aucun template .gridline trouvé : $ACTIVE_PROJECT sera créé vide." >&2
    : > "$ACTIVE_PROJECT"
  fi
fi

exec gridline mcp "$ACTIVE_PROJECT" --export-dir "$EXPORT_DIR"
