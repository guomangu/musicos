#!/usr/bin/env bash
set -euo pipefail

STUDIO_DIR="${AI_STUDIO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT="${1:-$STUDIO_DIR/projects/active/projet.gridline}"

if [ ! -f "$PROJECT" ]; then
  TEMPLATE="${GRIDLINE_EMPTY_TEMPLATE:-}"
  if [ -z "$TEMPLATE" ] || [ ! -f "$TEMPLATE" ]; then
    TEMPLATE="/usr/share/gridline/assets/templates/Basics/Empty with mixer tracks.gridline"
  fi
  if [ ! -f "$TEMPLATE" ]; then
    TEMPLATE="$STUDIO_DIR/assets/empty_with_mixer_tracks.gridline"
  fi
  if [ -f "$TEMPLATE" ]; then
    mkdir -p "$(dirname "$PROJECT")"
    cp "$TEMPLATE" "$PROJECT"
  fi
fi

# Préserve le CPU et évite les xruns (notamment sous casque Bluetooth)
if command -v pw-metadata >/dev/null 2>&1; then
  pw-metadata -n settings 0 clock.force-quantum 1024 >/dev/null 2>&1 || true
fi

echo "Ouverture de Gridline sur : $PROJECT (PipeWire quantum=1024)"
echo "MCP HTTP local actif sur : http://127.0.0.1:47390/mcp"
exec gridline-app "$PROJECT"
