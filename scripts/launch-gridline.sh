#!/usr/bin/env bash
set -euo pipefail

STUDIO_DIR="${AI_STUDIO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT="${1:-$STUDIO_DIR/projects/active/projet.gridline}"

if [ ! -f "$PROJECT" ]; then
  TEMPLATE="/usr/share/gridline/assets/templates/Basics/Empty with mixer tracks.gridline"
  if [ -f "$TEMPLATE" ]; then
    mkdir -p "$(dirname "$PROJECT")"
    cp "$TEMPLATE" "$PROJECT"
  fi
fi

echo "Ouverture de Gridline sur : $PROJECT"
echo "MCP HTTP local actif sur : http://127.0.0.1:47390/mcp"
exec gridline-app "$PROJECT"
