#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# AI-Studio Musical (Gridline + Zed + Antigravity) — Installeur Fedora flexible
# Détecte automatiquement l'architecture, les paquets présents, les plugins,
# les modèles IA et configure l'environnement de façon idempotente.
# ==============================================================================

STUDIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="$(uname -m)"

SKIP_PLUGINS=0
SKIP_CLAP=0
SKIP_ZED=0
NO_SUDO=0
CHECK_ONLY=0
FORCE_RESCAN=0

usage() {
  cat << 'EOF'
Usage: ./install.sh [options]

Options:
  --check-only     Vérifie l'état actuel sans modifier ni installer quoi que ce soit
  --skip-plugins   N'installe pas les plugins audio tiers (Surge XT, ZL EQ2, TDR Nova, CHOW, Dragonfly, LSP)
  --skip-clap      Ignore le préchargement CLAP et calcule uniquement l'index Essentia rapide
  --skip-zed       N'installe pas Zed automatiquement s'il est absent (génère tout de même la config)
  --no-sudo        N'appelle pas sudo dnf (ignore l'installation des paquets système manquants)
  --force-rescan   Force un rescan complet des plugins dans Gridline
  -h, --help       Affiche cette aide
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)   CHECK_ONLY=1 ;;
    --skip-plugins) SKIP_PLUGINS=1 ;;
    --skip-clap)    SKIP_CLAP=1 ;;
    --skip-zed)     SKIP_ZED=1 ;;
    --no-sudo)      NO_SUDO=1 ;;
    --force-rescan) FORCE_RESCAN=1 ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "[ERREUR] Option inconnue : $1" >&2; usage; exit 1 ;;
  esac
  shift
done

log()  { printf "\033[1;34m[AI-Studio]\033[0m %s\n" "$*"; }
ok()   { printf "\033[1;32m[OK]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[ERREUR]\033[0m %s\n" "$*" >&2; }

have() { command -v "$1" >/dev/null 2>&1; }

check_fedora() {
  if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    if [[ "${ID:-}" != "fedora" && "${ID_LIKE:-}" != *"fedora"* ]]; then
      warn "Système détecté (${PRETTY_NAME:-inconnu}) non basé sur Fedora ; les étapes DNF peuvent nécessiter une adaptation."
    else
      ok "OS compatible Fedora détecté : ${PRETTY_NAME:-Fedora} ($ARCH)"
    fi
  fi
}

install_system_deps() {
  log "1/8 Vérification des dépendances système Fedora..."
  local missing_pkgs=()

  have python3 || missing_pkgs+=(python3)
  python3 -m pip --version >/dev/null 2>&1 || missing_pkgs+=(python3-pip)
  python3 -m venv --help >/dev/null 2>&1 || missing_pkgs+=(python3)
  have git       || missing_pkgs+=(git)
  have curl      || missing_pkgs+=(curl)
  have unzip     || missing_pkgs+=(unzip)
  have xz        || missing_pkgs+=(xz)
  have tar       || missing_pkgs+=(tar)
  have 7z        || missing_pkgs+=(7zip)
  have ar        || missing_pkgs+=(binutils)
  have espeak-ng || missing_pkgs+=(espeak-ng)
  have sox       || missing_pkgs+=(sox)
  have aubiopitch || missing_pkgs+=(aubio)
  rpm -q python3-devel >/dev/null 2>&1 || missing_pkgs+=(python3-devel)

  rpm -q pipewire >/dev/null 2>&1 || missing_pkgs+=(pipewire)
  rpm -q pipewire-pulseaudio >/dev/null 2>&1 || missing_pkgs+=(pipewire-pulseaudio)
  rpm -q pipewire-jack-audio-connection-kit >/dev/null 2>&1 || missing_pkgs+=(pipewire-jack-audio-connection-kit)

  local need_ffmpeg=0
  have ffmpeg || need_ffmpeg=1

  if [[ ${#missing_pkgs[@]} -eq 0 && $need_ffmpeg -eq 0 ]]; then
    ok "Tous les paquets système requis sont déjà installés."
    return 0
  fi

  if [[ $CHECK_ONLY -eq 1 || $NO_SUDO -eq 1 ]]; then
    warn "Paquets manquants : ${missing_pkgs[*]} $([[ $need_ffmpeg -eq 1 ]] && echo ffmpeg)"
    return 0
  fi

  local dnf_bin="dnf"
  have dnf5 && dnf_bin="dnf5"

  if [[ ${#missing_pkgs[@]} -gt 0 ]]; then
    log "Installation via $dnf_bin : ${missing_pkgs[*]}"
    sudo "$dnf_bin" install -y "${missing_pkgs[@]}"
  fi

  if [[ $need_ffmpeg -eq 1 ]]; then
    if ! sudo "$dnf_bin" install -y ffmpeg 2>/dev/null; then
      sudo "$dnf_bin" install -y ffmpeg-free
    fi
  fi
  ok "Dépendances système installées."
}

install_gridline_and_stems() {
  log "2/8 Vérification de Gridline DAW et des modèles Demucs..."
  if ! have gridline; then
    if [[ $CHECK_ONLY -eq 1 ]]; then
      warn "Gridline non installé."
      return 0
    fi
    if [[ $NO_SUDO -eq 1 ]]; then
      err "Gridline est absent et --no-sudo empêche l'installation du RPM."
      return 1
    fi

    local local_rpm=""
    for candidate in "$HOME"/Bureau/Gridline*.rpm "$HOME"/Desktop/Gridline*.rpm "$HOME"/Downloads/Gridline*.rpm "$HOME"/Téléchargements/Gridline*.rpm; do
      if [[ -f "$candidate" ]]; then
        local_rpm="$candidate"
        break
      fi
    done

    if [[ -n "$local_rpm" ]]; then
      log "Installation de Gridline depuis le RPM local : $local_rpm"
      sudo dnf install -y "$local_rpm"
    else
      log "Téléchargement du dernier RPM Gridline pour $ARCH depuis GitHub..."
      local rpm_url
      rpm_url="$(curl -fsSL "https://api.github.com/repos/lolitemaultes/Gridline/releases/latest" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); arch='$ARCH'; print(next((a['browser_download_url'] for a in d.get('assets',[]) if a['name'].endswith(f'linux-{arch}.rpm')), ''))")"
      if [[ -z "$rpm_url" ]]; then
        err "Impossible de trouver un RPM Gridline pour $ARCH sur GitHub."
        return 1
      fi
      local tmp_rpm="/tmp/gridline-latest-$ARCH.rpm"
      curl -fL "$rpm_url" -o "$tmp_rpm"
      sudo dnf install -y "$tmp_rpm"
      rm -f "$tmp_rpm"
    fi
  fi
  ok "Gridline détecté : $(gridline --version | head -n1)"

  local stems_dir="$HOME/.local/share/gridline/stem-models"
  local need_htdemucs=1
  local need_htdemucs_6s=1
  [[ -f "$stems_dir/955717e8-8726e21a.th" ]] && need_htdemucs=0
  [[ -f "$stems_dir/5c90dfd2-34c22ccb.th" ]] && need_htdemucs_6s=0

  if [[ $need_htdemucs -eq 0 && $need_htdemucs_6s -eq 0 ]]; then
    ok "Modèles de stems Demucs (htdemucs & htdemucs_6s) déjà présents."
  elif [[ $CHECK_ONLY -eq 1 ]]; then
    warn "Modèles Demucs manquants dans $stems_dir."
  else
    [[ $need_htdemucs -eq 1 ]] && gridline stems --download htdemucs
    [[ $need_htdemucs_6s -eq 1 ]] && gridline stems --download htdemucs_6s
    ok "Modèles Demucs téléchargés."
  fi
}

check_or_install_zed() {
  log "3/8 Vérification des éditeurs (Zed / Antigravity)..."
  if have zed || [[ -x "$HOME/.local/bin/zed" ]]; then
    ok "Zed détecté."
  elif [[ $CHECK_ONLY -eq 1 || $SKIP_ZED -eq 1 ]]; then
    warn "Zed non détecté (ignoré)."
  else
    log "Installation de Zed via le script officiel zed.dev..."
    curl -fsSL https://zed.dev/install.sh | sh || warn "Installation automatique de Zed échouée ; vous pouvez l'installer manuellement."
  fi

  if have antigravity || [[ -d "$HOME/.gemini/config" ]]; then
    ok "Antigravity détecté."
  else
    warn "Antigravity non détecté ; la config ~/.gemini/config/mcp_config.json sera préparée."
  fi
}

setup_python_venv() {
  log "4/8 Vérification de l'environnement Python (.venv)..."
  local venv_py="$STUDIO_DIR/.venv/bin/python"
  local venv_pip="$STUDIO_DIR/.venv/bin/pip"

  if [[ $CHECK_ONLY -eq 1 ]]; then
    if [[ -x "$venv_py" ]] && "$venv_py" -c "import mcp, pyloudnorm, soundfile, scipy, librosa, essentia, yaml, jsonschema, onnxruntime, mido, torch, torchaudio, transformers, mosqito, music21, pretty_midi, pyroomacoustics, matchering, audiobox_aesthetics" 2>/dev/null; then
      ok "Environnement .venv complet et importable."
    else
      warn "Environnement .venv absent ou dépendances Python incomplètes."
    fi
    return 0
  fi

  if [[ ! -x "$venv_py" ]]; then
    log "Création de $STUDIO_DIR/.venv..."
    python3 -m venv "$STUDIO_DIR/.venv"
  fi

  if "$venv_py" -c "import mcp, pyloudnorm, soundfile, scipy, librosa, essentia, yaml, jsonschema, onnxruntime, mido, torch, torchaudio, transformers, mosqito, music21, pretty_midi, pyroomacoustics, matchering, audiobox_aesthetics" 2>/dev/null; then
    ok "Dépendances Python .venv déjà toutes satisfaites."
  else
    log "Installation des paquets Python (PyTorch/Torchaudio CPU + requirements.txt)..."
    "$venv_py" -m pip install --upgrade pip
    "$venv_pip" install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch torchaudio
    "$venv_pip" install -r "$STUDIO_DIR/requirements.txt"
    ok "Dépendances Python installées dans .venv."
  fi

  if [[ $SKIP_CLAP -eq 0 ]]; then
    if ! HF_HUB_OFFLINE=1 "$venv_py" -c "from transformers import ClapModel, ClapProcessor; ClapProcessor.from_pretrained('laion/clap-htsat-unfused', local_files_only=True); ClapModel.from_pretrained('laion/clap-htsat-unfused', local_files_only=True)" 2>/dev/null; then
      log "Préchargement du modèle sémantique LAION CLAP (laion/clap-htsat-unfused)..."
      "$venv_py" -c "from transformers import ClapModel, ClapProcessor; ClapProcessor.from_pretrained('laion/clap-htsat-unfused'); ClapModel.from_pretrained('laion/clap-htsat-unfused')"
    fi
    ok "Modèle LAION CLAP en cache local."

    if ! HF_HUB_OFFLINE=1 "$venv_py" -c "from audiobox_aesthetics.infer import initialize_predictor; initialize_predictor()" >/dev/null 2>&1; then
      log "Préchargement du modèle perceptuel Meta Audiobox Aesthetics (facebook/audiobox-aesthetics)..."
      "$venv_py" -c "from audiobox_aesthetics.infer import initialize_predictor; initialize_predictor()" >/dev/null 2>&1 || warn "Audiobox Aesthetics non préchargé (réessayera à la première analyse)."
    fi
    ok "Modèle Audiobox Aesthetics prêt en cache local."
  fi
}

setup_models_and_openutau() {
  log "5/8 Vérification des modèles ONNX et d'OpenUTAU..."
  local onnx_path="$STUDIO_DIR/tools/models/nmp.onnx"
  local openutau_bin="$STUDIO_DIR/tools/OpenUtau/OpenUtau"

  if [[ -f "$onnx_path" ]]; then
    ok "Modèle Spotify Basic Pitch ONNX présent."
  elif [[ $CHECK_ONLY -eq 1 ]]; then
    warn "Modèle Basic Pitch ONNX manquant : $onnx_path"
  else
    mkdir -p "$(dirname "$onnx_path")"
    curl -fL "https://raw.githubusercontent.com/spotify/basic-pitch/main/basic_pitch/saved_models/icassp_2022/nmp.onnx" -o "$onnx_path"
    ok "Modèle Basic Pitch ONNX téléchargé."
  fi

  if [[ -x "$openutau_bin" ]]; then
    ok "OpenUTAU déjà installé dans tools/OpenUtau/."
  elif [[ $CHECK_ONLY -eq 1 ]]; then
    warn "OpenUTAU absent dans tools/OpenUtau/."
  else
    local ou_arch="x64"
    [[ "$ARCH" == "aarch64" ]] && ou_arch="arm64"
    local ou_url
    ou_url="$(curl -fsSL "https://api.github.com/repos/openutau/OpenUtau/releases/latest" \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print(next((a['browser_download_url'] for a in d.get('assets',[]) if a['name'] == 'OpenUtau-linux-${ou_arch}.tar.gz'), ''))")"
    if [[ -n "$ou_url" ]]; then
      mkdir -p "$STUDIO_DIR/tools/OpenUtau"
      curl -fL "$ou_url" -o /tmp/OpenUtau-linux.tar.gz
      tar -xzf /tmp/OpenUtau-linux.tar.gz -C "$STUDIO_DIR/tools/OpenUtau"
      rm -f /tmp/OpenUtau-linux.tar.gz
      ok "OpenUTAU ($ou_arch) installé."
    else
      warn "Archive OpenUTAU linux-${ou_arch}.tar.gz introuvable sur GitHub."
    fi
  fi
}

install_native_plugins() {
  log "6/8 Vérification des plugins audio Linux natifs (CLAP / VST3 / LV2)..."
  if [[ $SKIP_PLUGINS -eq 1 ]]; then
    warn "Installation des plugins tiers ignorée (--skip-plugins)."
    return 0
  fi
  if [[ "$ARCH" != "x86_64" ]]; then
    warn "Architecture $ARCH : certains binaires précompilés (TDR Nova, CHOW) ciblent x86_64 ; étape plugins adaptée/ignorée."
    return 0
  fi

  mkdir -p "$HOME/.clap" "$HOME/.vst3" "$HOME/.lv2" "$HOME/.local/share/surge-xt"
  local tmp_dl="/tmp/ai_studio_plugins_$$"
  local changed=0

  ensure_dl_dir() { mkdir -p "$tmp_dl"; }

  # 1. TDR Nova (VST3)
  if [[ -d "$HOME/.vst3/TDR Nova.vst3" ]]; then
    ok "TDR Nova déjà installé."
  elif [[ $CHECK_ONLY -eq 1 ]]; then
    warn "TDR Nova manquant."
  else
    ensure_dl_dir
    log "Installation de TDR Nova 2.2.3 (VST3)..."
    curl -fL "https://www.tokyodawn.net/labs/Nova/2.2.3/TDR%20Nova.tar.gz" -o "$tmp_dl/nova.tar.gz"
    mkdir -p "$tmp_dl/nova"
    tar -xzf "$tmp_dl/nova.tar.gz" -C "$tmp_dl/nova"
    find "$tmp_dl/nova" -name "*.vst3" -exec cp -r {} "$HOME/.vst3/" \;
    changed=1
  fi

  # 2. ZL Equalizer 2 (VST3 / CLAP si fourni)
  if [[ -d "$HOME/.vst3/ZL Equalizer 2.vst3" ]]; then
    ok "ZL Equalizer 2 déjà installé."
  elif [[ $CHECK_ONLY -eq 1 ]]; then
    warn "ZL Equalizer 2 manquant."
  else
    ensure_dl_dir
    local zl_pattern="Linux-x86-64.zip"
    grep -m1 -q avx2 /proc/cpuinfo && zl_pattern="Linux-x86-64-AVX2.zip"
    local zl_url
    zl_url="$(curl -fsSL "https://api.github.com/repos/ZL-Audio/ZLEqualizer/releases/latest" \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print(next((a['browser_download_url'] for a in d.get('assets',[]) if '$zl_pattern' in a['name']), ''))")"
    if [[ -n "$zl_url" ]]; then
      log "Installation de ZL Equalizer 2 ($zl_pattern)..."
      curl -fL "$zl_url" -o "$tmp_dl/zl.zip"
      unzip -q -o "$tmp_dl/zl.zip" -d "$tmp_dl/zl"
      find "$tmp_dl/zl" -name "*.vst3" -exec cp -r {} "$HOME/.vst3/" \;
      find "$tmp_dl/zl" -name "*.clap" -exec cp -r {} "$HOME/.clap/" \;
      changed=1
    fi
  fi

  # 3. CHOW Tape Model (CLAP + VST3 + LV2)
  if [[ -f "$HOME/.clap/CHOWTapeModel.clap" && -d "$HOME/.vst3/CHOWTapeModel.vst3" ]]; then
    ok "CHOWTapeModel déjà installé."
  elif [[ $CHECK_ONLY -eq 1 ]]; then
    warn "CHOWTapeModel manquant."
  else
    ensure_dl_dir
    local chow_url
    chow_url="$(curl -fsSL "https://api.github.com/repos/jatinchowdhury18/AnalogTapeModel/releases/latest" \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print(next((a['browser_download_url'] for a in d.get('assets',[]) if 'Linux-x64' in a['name'] and a['name'].endswith('.deb')), ''))")"
    if [[ -n "$chow_url" ]]; then
      log "Installation de CHOWTapeModel..."
      curl -fL "$chow_url" -o "$tmp_dl/chow.deb"
      mkdir -p "$tmp_dl/chow"
      (cd "$tmp_dl/chow" && ar x ../chow.deb && tar -xf data.tar.*)
      find "$tmp_dl/chow" -name "*.clap" -exec cp -r {} "$HOME/.clap/" \;
      find "$tmp_dl/chow" -name "*.vst3" -exec cp -r {} "$HOME/.vst3/" \;
      find "$tmp_dl/chow" -name "*.lv2" -exec cp -r {} "$HOME/.lv2/" \;
      changed=1
    fi
  fi

  # 4. Dragonfly Reverb (VST3 + LV2)
  if [[ -d "$HOME/.vst3/DragonflyHallReverb.vst3" ]]; then
    ok "Dragonfly Reverb déjà installé."
  elif [[ $CHECK_ONLY -eq 1 ]]; then
    warn "Dragonfly Reverb manquant."
  else
    ensure_dl_dir
    local df_url
    df_url="$(curl -fsSL "https://api.github.com/repos/michaelwillis/dragonfly-reverb/releases/latest" \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print(next((a['browser_download_url'] for a in d.get('assets',[]) if 'linux-x86_64.tar.xz' in a['name']), ''))")"
    if [[ -n "$df_url" ]]; then
      log "Installation de Dragonfly Reverb..."
      curl -fL "$df_url" -o "$tmp_dl/df.tar.xz"
      mkdir -p "$tmp_dl/df"
      tar -xJf "$tmp_dl/df.tar.xz" -C "$tmp_dl/df"
      find "$tmp_dl/df" -name "*.vst3" -exec cp -r {} "$HOME/.vst3/" \;
      find "$tmp_dl/df" -name "*.lv2" -exec cp -r {} "$HOME/.lv2/" \;
      changed=1
    fi
  fi

  # 5. Surge XT (CLAP + VST3 + données factory)
  if [[ -f "$HOME/.clap/Surge XT.clap" && -d "$HOME/.local/share/surge-xt/patches_factory" ]]; then
    ok "Surge XT déjà installé."
  elif [[ $CHECK_ONLY -eq 1 ]]; then
    warn "Surge XT manquant."
  else
    ensure_dl_dir
    local surge_url
    surge_url="$(curl -fsSL "https://api.github.com/repos/surge-synthesizer/releases-xt/releases/latest" \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print(next((a['browser_download_url'] for a in d.get('assets',[]) if 'linux-x86_64' in a['name'] and a['name'].endswith('.tar.gz')), ''))")"
    if [[ -n "$surge_url" ]]; then
      log "Installation de Surge XT (CLAP + VST3 + banques)..."
      curl -fL "$surge_url" -o "$tmp_dl/surge.tar.gz"
      mkdir -p "$tmp_dl/surge"
      tar -xzf "$tmp_dl/surge.tar.gz" -C "$tmp_dl/surge"
      find "$tmp_dl/surge" -name "*.clap" -exec cp -r {} "$HOME/.clap/" \;
      find "$tmp_dl/surge" -name "*.vst3" -exec cp -r {} "$HOME/.vst3/" \;
      find "$tmp_dl/surge" -type d -name "surge-xt" -exec cp -r {}/. "$HOME/.local/share/surge-xt/" \; || true
      changed=1
    fi
  fi

  # 6. LSP Plugins (CLAP + VST3)
  if [[ -f "$HOME/.clap/lsp-plugins.clap" && -d "$HOME/.vst3/lsp-plugins.vst3" ]]; then
    ok "LSP Plugins déjà installé."
  elif [[ $CHECK_ONLY -eq 1 ]]; then
    warn "LSP Plugins manquant."
  else
    ensure_dl_dir
    local lsp_url
    lsp_url="$(curl -fsSL "https://api.github.com/repos/lsp-plugins/lsp-plugins/releases/latest" \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print(next((a['browser_download_url'] for a in d.get('assets',[]) if 'Linux-x86_64.7z' in a['name']), ''))")"
    if [[ -n "$lsp_url" ]]; then
      log "Installation de LSP Plugins..."
      curl -fL "$lsp_url" -o "$tmp_dl/lsp.7z"
      mkdir -p "$tmp_dl/lsp"
      7z x -y -o"$tmp_dl/lsp" "$tmp_dl/lsp.7z" >/dev/null
      find "$tmp_dl/lsp" -name "*.clap" -exec cp -r {} "$HOME/.clap/" \;
      find "$tmp_dl/lsp" -name "*.vst3" -exec cp -r {} "$HOME/.vst3/" \;
      changed=1
    fi
  fi

  # sfizz VST3 (lecteur d'instruments SFZ multi-vélocités)
  if [[ -d "/usr/lib64/vst3/sfizz.vst3" || -d "$HOME/.vst3/sfizz.vst3" ]]; then
    ok "sfizz.vst3 déjà installé."
  elif [[ $CHECK_ONLY -eq 1 || $SKIP_PLUGINS -eq 1 ]]; then
    warn "sfizz.vst3 absent (ignoré en mode --check-only ou --skip-plugins)."
  else
    if [[ $NO_SUDO -eq 0 ]] && have dnf && run_sudo dnf copr enable -y lexridge/sfizz-ui >/dev/null 2>&1; then
      log "Installation de sfizz-ui-vst3 via Fedora COPR (lexridge/sfizz-ui)..."
      if run_sudo dnf install -y sfizz-ui-vst3 >/dev/null 2>&1; then
        changed=1
      fi
    fi
    if [[ ! -d "/usr/lib64/vst3/sfizz.vst3" && ! -d "$HOME/.vst3/sfizz.vst3" ]] && have rpm2cpio && have cpio; then
      log "Extraction user-space de sfizz.vst3 dans ~/.vst3..."
      mkdir -p "$HOME/.vst3"
      local sfz_rpm
      sfz_rpm="https://download.copr.fedorainfracloud.org/results/lexridge/sfizz-ui/fedora-44-x86_64/09866468-sfizz-ui/sfizz-ui-vst3-1.2.3-2.20251130git.fc44.x86_64.rpm"
      if curl -fsSL "$sfz_rpm" -o "$tmp_dl/sfizz-ui-vst3.rpm"; then
        mkdir -p "$tmp_dl/sfizz_extract"
        (cd "$tmp_dl/sfizz_extract" && rpm2cpio "$tmp_dl/sfizz-ui-vst3.rpm" | cpio -idm >/dev/null 2>&1)
        if [[ -d "$tmp_dl/sfizz_extract/usr/lib64/vst3/sfizz.vst3" ]]; then
          cp -r "$tmp_dl/sfizz_extract/usr/lib64/vst3/sfizz.vst3" "$HOME/.vst3/"
          changed=1
          ok "sfizz.vst3 installé dans ~/.vst3/."
        fi
      else
        warn "Téléchargement sfizz.vst3 échoué (ignoré)."
      fi
    fi
  fi

  rm -rf "$tmp_dl"

  if have gridline && [[ $CHECK_ONLY -eq 0 ]]; then
    if [[ $changed -eq 1 || $FORCE_RESCAN -eq 1 || ! -f "$HOME/.config/gridline/plugin-cache.json" ]]; then
      log "Scan des plugins dans Gridline..."
      gridline scan-plugins --force
    fi
  fi

  # Courbes AutoEq cibles casques (Sony WH-1000XM6 / XM5 / XM4)
  local targets_dir="$STUDIO_DIR/mcp-servers/audio-analyzer/targets"
  local targets_base="https://raw.githubusercontent.com/jaakkopasanen/AutoEq/master/results"
  local -A autoeq_urls=(
    ["Sony_WH-1000XM6.csv"]="$targets_base/Kuulokenurkka/over-ear/Sony%20WH-1000XM6/Sony%20WH-1000XM6.csv"
    ["Sony_WH-1000XM5.csv"]="$targets_base/oratory1990/over-ear/Sony%20WH-1000XM5/Sony%20WH-1000XM5.csv"
    ["Sony_WH-1000XM4.csv"]="$targets_base/oratory1990/over-ear/Sony%20WH-1000XM4/Sony%20WH-1000XM4.csv"
  )
  local missing_targets=0
  for csv_name in "${!autoeq_urls[@]}"; do
    local csv_dest="$targets_dir/$csv_name"
    if [[ ! -f "$csv_dest" ]]; then
      if [[ $CHECK_ONLY -eq 1 ]]; then
        missing_targets=1
      else
        mkdir -p "$targets_dir"
        curl -fsSL "${autoeq_urls[$csv_name]}" -o "$csv_dest" || warn "Courbe AutoEq $csv_name indisponible."
      fi
    fi
  done
  if [[ $missing_targets -eq 0 ]]; then
    ok "Courbes cibles casques AutoEq (XM6/XM5/XM4) prêtes."
  else
    warn "Courbes AutoEq manquantes dans $targets_dir."
  fi

  # Banques SFZ acoustiques CC0/libres (piano, Rhodes, choeurs/dialogue — hors samples/ pour ne pas polluer search_samples)
  local -A sfz_repos=(
    ["jRhodes3c"]="https://github.com/sfzinstruments/jlearman.jRhodes3c/archive/refs/heads/master.tar.gz"
    ["merry-orks"]="https://github.com/sfzinstruments/karoryfer.272-merry-orks/archive/refs/heads/master.tar.gz"
    ["headroom-piano"]="https://github.com/sfzinstruments/BengtNilsson.HeadroomPiano/archive/refs/heads/master.tar.gz"
  )
  local missing_sfz=0
  for sfz_dir in "${!sfz_repos[@]}"; do
    local target_sfz="$STUDIO_DIR/soundbanks/sfz/$sfz_dir"
    if [[ ! -d "$target_sfz" || -z "$(find "$target_sfz" -maxdepth 2 -name '*.sfz' 2>/dev/null)" ]]; then
      if [[ $CHECK_ONLY -eq 1 ]]; then
        missing_sfz=1
      else
        log "Téléchargement banque SFZ $sfz_dir..."
        local tgz="/tmp/sfz_${sfz_dir}.tar.gz"
        if curl -fsSL "${sfz_repos[$sfz_dir]}" -o "$tgz"; then
          mkdir -p "$target_sfz"
          tar -xzf "$tgz" -C "$target_sfz" --strip-components=1
          rm -f "$tgz"
        else
          warn "Banque SFZ $sfz_dir non téléchargée."
        fi
      fi
    fi
  done
  if [[ $missing_sfz -eq 0 ]]; then
    local hr_dir="$STUDIO_DIR/soundbanks/sfz/headroom-piano"
    local hr_clean="$hr_dir/Headroom Piano Clean.sfz"
    if [[ -d "$hr_dir/Samples" && ! -f "$hr_clean" ]]; then
      if [[ $CHECK_ONLY -eq 1 ]]; then
        warn "Preset SFZ v1 aplati Headroom Piano Clean.sfz absent."
      else
        python3 - "$hr_clean" <<'PY'
import sys
out_path = sys.argv[1]
keys = list(range(21, 109, 3))
levels = [("LEVEL1", 1, 59, 13.3), ("LEVEL2", 60, 89, 6.2), ("LEVEL3", 90, 105, 3.3), ("LEVEL4", 106, 119, 1.1), ("LEVEL5", 120, 127, 0.0)]
lines = []
for vname, lo_v, hi_v, gvol in levels:
    lines += ["<group>", f"lovel={lo_v}", f"hivel={hi_v}", f"volume={gvol:.1f}", "ampeg_attack=0.002", "ampeg_release=0.48"]
    for k in keys:
        lo_k, hi_k = (21 if k == 21 else k - 1), (108 if k == 108 else k + 1)
        lines += ["<region>", f"sample=Samples/HEADROOM PIANO {vname} CLOSE {k}.flac", f"lokey={lo_k}", f"hikey={hi_k}", f"pitch_keycenter={k}"]
with open(out_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
PY
        ok "Preset SFZ v1 Headroom Piano Clean.sfz généré pour gl.multisampler."
      fi
    fi
    ok "Banques d'instruments SFZ prêtes dans soundbanks/sfz/."
  else
    warn "Banques d'instruments SFZ manquantes dans soundbanks/sfz/."
  fi

  # Voicebank DiffSinger FR/EN pour OpenUTAU (TIGER_DS v1.06)
  if [[ -f "$STUDIO_DIR/voicebanks/TIGER_DS_v106/dsconfig.yaml" ]]; then
    ok "Voicebank DiffSinger TIGER_DS_v106 déjà installée."
  elif [[ $CHECK_ONLY -eq 1 ]]; then
    warn "Voicebank DiffSinger TIGER_DS_v106 absente dans voicebanks/."
  else
    log "Téléchargement de la voicebank DiffSinger FR/EN (TIGER_DS v1.06)..."
    local ds_zip="/tmp/TIGER_DS_v106_PACK.zip"
    if curl -fsSL "https://github.com/spicytigermeat/tiger_diffsinger/releases/download/v106/TIGER_DS_v106_PACK.zip" -o "$ds_zip"; then
      mkdir -p "$STUDIO_DIR/voicebanks" /tmp/tiger_ds_unpack "$HOME/.local/share/OpenUtau/Plugins"
      unzip -q -o "$ds_zip" -d /tmp/tiger_ds_unpack
      if [[ -f "/tmp/tiger_ds_unpack/TIGER_DS_v106_PACK/Voice Library/TIGER_DS_v106.zip" ]]; then
        unzip -q -o "/tmp/tiger_ds_unpack/TIGER_DS_v106_PACK/Voice Library/TIGER_DS_v106.zip" -d "$STUDIO_DIR/voicebanks/TIGER_DS_v106"
      fi
      if [[ -d "/tmp/tiger_ds_unpack/TIGER_DS_v106_PACK/OpenUTAU Plugins" ]]; then
        cp -f /tmp/tiger_ds_unpack/TIGER_DS_v106_PACK/OpenUTAU\ Plugins/*.dll "$HOME/.local/share/OpenUtau/Plugins/" 2>/dev/null || true
      fi
      rm -rf "$ds_zip" /tmp/tiger_ds_unpack
      ok "Voicebank DiffSinger TIGER_DS_v106 installée."
    else
      warn "Téléchargement TIGER_DS_v106 échoué (réessayer plus tard)."
    fi
  fi
}

configure_workspace_and_mcps() {
  log "7/8 Configuration du workspace, des liens et des serveurs MCP (Gridline / Zed / Antigravity)..."
  if [[ $CHECK_ONLY -eq 1 ]]; then
    ok "Vérification des fichiers de configuration ignorée en mode --check-only."
    return 0
  fi

  mkdir -p "$STUDIO_DIR/.zed" \
           "$STUDIO_DIR/voicebanks" \
           "$STUDIO_DIR/renders" \
           "$STUDIO_DIR/projects/active/scripts" \
           "$STUDIO_DIR/projects/active/vocals" \
           "$STUDIO_DIR/projects/active/renders" \
           "$HOME/Music" \
           "$HOME/.config/gridline/scripts/project" \
           "$HOME/.config/zed" \
           "$HOME/.gemini/config"

  chmod +x "$STUDIO_DIR/scripts/"*.sh "$STUDIO_DIR/install.sh"

  ln -sf .rules "$STUDIO_DIR/GEMINI.md"
  ln -sf .rules "$STUDIO_DIR/AGENTS.md"

  if [[ -L "$STUDIO_DIR/samples" && ! -e "$STUDIO_DIR/samples" ]]; then
    rm -f "$STUDIO_DIR/samples"
  fi
  if [[ -d "$STUDIO_DIR/soundbanks" && ! -e "$STUDIO_DIR/samples" ]]; then
    ln -s "$STUDIO_DIR/soundbanks" "$STUDIO_DIR/samples"
  elif [[ -d "$STUDIO_DIR/pack" && ! -e "$STUDIO_DIR/samples" ]]; then
    ln -s "$STUDIO_DIR/pack" "$STUDIO_DIR/samples"
  elif [[ ! -e "$STUDIO_DIR/samples" ]]; then
    mkdir -p "$STUDIO_DIR/samples"
  fi

  mkdir -p "$STUDIO_DIR/golden_references/"{kick,bass_sub,snare_clap,hihat_perc,lead_melody,pad_atmosphere,chord_keys}
  if [[ -f "$STUDIO_DIR/.env.example" && ! -f "$STUDIO_DIR/.env" ]]; then
    cp "$STUDIO_DIR/.env.example" "$STUDIO_DIR/.env"
  fi

  if [[ ! -e "$HOME/Music/AI-Studio" ]]; then
    ln -s "$STUDIO_DIR" "$HOME/Music/AI-Studio"
  fi

  for lua_f in "$STUDIO_DIR/projects/active/scripts/"*.lua; do
    [[ -f "$lua_f" ]] && ln -sf "$lua_f" "$HOME/.config/gridline/scripts/project/$(basename "$lua_f")"
  done

  if [[ ! -f "$STUDIO_DIR/projects/active/projet.gridline" ]]; then
    local tpl="/usr/share/gridline/assets/templates/Basics/Empty with mixer tracks.gridline"
    [[ -f "$tpl" ]] && cp "$tpl" "$STUDIO_DIR/projects/active/projet.gridline"
  fi

  STUDIO_DIR="$STUDIO_DIR" python3 - << 'PY'
import json, os, shutil

studio = os.environ["STUDIO_DIR"]
venv_py = os.path.join(studio, ".venv/bin/python")
active_proj = os.path.join(studio, "projects/active/projet.gridline")
renders_dir = os.path.join(studio, "projects/active/renders")

# 1. Gridline settings.json (préserve les réglages existants)
gl_cfg = os.path.expanduser("~/.config/gridline/settings.json")
gl_data = {"format": "gridline-settings", "version": 1, "values": {}}
if os.path.isfile(gl_cfg) and os.path.getsize(gl_cfg) > 0:
    shutil.copy2(gl_cfg, gl_cfg + ".bak")
    with open(gl_cfg, "r", encoding="utf-8") as f:
        gl_data = json.load(f)
gl_data.setdefault("values", {})
gl_data["values"].update({
    "mcp.enabled": True,
    "mcp.allowEdits": True,
    "mcp.propose": False,
    "mcp.port": 47390,
    "mcp.listen": "loopback",
    "mcp.exportFolder": renders_dir,
})
with open(gl_cfg, "w", encoding="utf-8") as f:
    json.dump(gl_data, f, indent=2, ensure_ascii=False)
    f.write("\n")

# 2. Antigravity ~/.gemini/config/mcp_config.json
agy_cfg = os.path.expanduser("~/.gemini/config/mcp_config.json")
agy_data = {}
if os.path.isfile(agy_cfg) and os.path.getsize(agy_cfg) > 0:
    shutil.copy2(agy_cfg, agy_cfg + ".bak")
    with open(agy_cfg, "r", encoding="utf-8") as f:
        agy_data = json.load(f)
agy_data.setdefault("mcpServers", {})
agy_data["mcpServers"].update({
    "gridline": {
        "command": os.path.join(studio, "scripts/gridline-mcp.sh"),
        "args": [],
        "env": {"GRIDLINE_PROJECT": active_proj, "GRIDLINE_EXPORT_DIR": renders_dir},
    },
    "audio-index": {"command": venv_py, "args": [os.path.join(studio, "mcp-servers/audio-index/audio_index_server.py")]},
    "audio-analyzer": {"command": venv_py, "args": [os.path.join(studio, "mcp-servers/audio-analyzer/audio_analyzer_server.py")]},
    "vocaloid": {"command": venv_py, "args": [os.path.join(studio, "mcp-servers/vocaloid/vocaloid_server.py")]},
    "reverse-engineer": {"command": venv_py, "args": [os.path.join(studio, "mcp-servers/reverse-engineer/reverse_engineer_server.py")]},
})
with open(agy_cfg, "w", encoding="utf-8") as f:
    json.dump(agy_data, f, indent=2, ensure_ascii=False)
    f.write("\n")

# 3. Zed workspace (.zed/settings.json) et global (~/.config/zed/settings.json)
zed_servers = {
    "gridline": {
        "command": {
            "path": os.path.join(studio, "scripts/gridline-mcp.sh"),
            "args": [],
            "env": {"GRIDLINE_PROJECT": active_proj, "GRIDLINE_EXPORT_DIR": renders_dir},
        }
    },
    "audio-index": {"command": {"path": venv_py, "args": [os.path.join(studio, "mcp-servers/audio-index/audio_index_server.py")]}},
    "audio-analyzer": {"command": {"path": venv_py, "args": [os.path.join(studio, "mcp-servers/audio-analyzer/audio_analyzer_server.py")]}},
    "vocaloid": {"command": {"path": venv_py, "args": [os.path.join(studio, "mcp-servers/vocaloid/vocaloid_server.py")]}},
    "reverse-engineer": {"command": {"path": venv_py, "args": [os.path.join(studio, "mcp-servers/reverse-engineer/reverse_engineer_server.py")]}},
}
with open(os.path.join(studio, ".zed/settings.json"), "w", encoding="utf-8") as f:
    json.dump({"context_servers": zed_servers}, f, indent=2, ensure_ascii=False)
    f.write("\n")

zed_global = os.path.expanduser("~/.config/zed/settings.json")
zed_data = {}
if os.path.isfile(zed_global) and os.path.getsize(zed_global) > 0:
    shutil.copy2(zed_global, zed_global + ".bak")
    with open(zed_global, "r", encoding="utf-8") as f:
        zed_data = json.load(f)
zed_data.setdefault("context_servers", {}).update(zed_servers)
with open(zed_global, "w", encoding="utf-8") as f:
    json.dump(zed_data, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
  ok "Configurations Gridline, Zed et Antigravity synchronisées sur $STUDIO_DIR."
}

# Résumé humain d'un rapport d'indexation (lit stdin : JSON ou dict Python, affiche 1 ligne).
# v3.3 hotfix : rend explicite "nouveaux / déjà à jour / rejetés (raisons)" pour ne plus
# laisser croire que la phase d'indexation "n'a rien fait" quand tout est déjà en base.
summarize_index_report() {
  local kind="$1"
  "$STUDIO_DIR/.venv/bin/python" - "$kind" << 'PYEOF'
import ast, json, sys
kind = sys.argv[1]
raw = sys.stdin.read()
try:
    r = json.loads(raw)
except Exception:
    r = ast.literal_eval(raw)
if kind == "presets":
    keys = ("total_presets", "indexed", "skipped", "errors", "clap_embedded")
    print("[AI-Studio] Résumé presets gl.* : {total_presets} au total, {indexed} nouveaux, {skipped} déjà à jour (incrémental), {errors} erreurs, CLAP {clap_embedded}".format(**{k: r.get(k, 0) for k in keys}))
else:
    rr = r.get("rejected_reasons") or {}
    det = " (" + ", ".join(f"{k}: {v}" for k, v in sorted(rr.items())) + ")" if rr else ""
    err = f", {r.get('errors', 0)} erreurs" if r.get("errors") else ""
    keys = ("scanned", "total", "indexed", "skipped", "rejected", "dsp_total", "clap_embedded", "cards_total")
    print("[AI-Studio] Résumé samples : {scanned} scannés, {total} en base, {indexed} nouveaux, {skipped} déjà à jour (incrémental), {rejected} rejetés{d}{e}, DSP {dsp_total}, CLAP {clap_embedded}, cartes {cards_total}".format(d=det, e=err, **{k: r.get(k, 0) for k in keys}))
PYEOF
}

index_samples_and_verify() {
  log "8/8 Indexation des samples & validation finale..."
  local venv_py="$STUDIO_DIR/.venv/bin/python"

  if [[ $CHECK_ONLY -eq 1 ]]; then
    ok "Vérification terminée (--check-only)."
    return 0
  fi

  local sample_count
  # Même périmètre que l'indexeur (AUDIO_EXTS + MIDI_EXTS, hors sfz/ __MACOSX/ répertoires cachés)
  sample_count="$(find -L "$STUDIO_DIR/samples" -type d \( -name sfz -o -name __MACOSX -o -name '.*' \) -prune -o -type f ! -name '.*' \( -iname '*.wav' -o -iname '*.ogg' -o -iname '*.flac' -o -iname '*.mp3' -o -iname '*.aiff' -o -iname '*.aif' -o -iname '*.mid' -o -iname '*.midi' \) -print 2>/dev/null | wc -l)"
  if [[ "$sample_count" -gt 0 ]]; then
    log "Indexation incrémentale de $sample_count fichiers dans samples/..."
    local out
    if [[ $SKIP_CLAP -eq 0 ]]; then
      out="$(HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error "$venv_py" "$STUDIO_DIR/mcp-servers/audio-index/indexer.py" --with-clap)"
      printf '%s\n' "$out"; printf '%s' "$out" | summarize_index_report samples
      out="$(HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error "$venv_py" "$STUDIO_DIR/scripts/index_samples.py" --input-dir "$STUDIO_DIR/samples")"
      printf '%s\n' "$out"; printf '%s' "$out" | summarize_index_report samples
    else
      out="$("$venv_py" "$STUDIO_DIR/mcp-servers/audio-index/indexer.py")"
      printf '%s\n' "$out"; printf '%s' "$out" | summarize_index_report samples
      out="$("$venv_py" "$STUDIO_DIR/scripts/index_samples.py" --input-dir "$STUDIO_DIR/samples" --no-clap)"
      printf '%s\n' "$out"; printf '%s' "$out" | summarize_index_report samples
    fi
  else
    warn "Aucun sample audio détecté dans $STUDIO_DIR/samples/ (dépose tes packs ici puis relance install.sh ou indexer.py)."
  fi

  if have gridline; then
    log "Indexation spectrale des presets synthés natifs Gridline (gl.*)..."
    local pout
    if [[ $SKIP_CLAP -eq 0 ]]; then
      pout="$(HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error "$venv_py" "$STUDIO_DIR/mcp-servers/audio-index/preset_indexer.py" --with-clap)"
    else
      pout="$("$venv_py" "$STUDIO_DIR/mcp-servers/audio-index/preset_indexer.py")"
    fi
    printf '%s\n' "$pout"; printf '%s' "$pout" | summarize_index_report presets
  fi

  # v3.3 hotfix : verify_install.py ignore proprement l'absence de projects/active/song_blueprint.json
  # (installation fraîche) au lieu de planter avec FileNotFoundError sous `set -euo pipefail`.
  "$venv_py" "$STUDIO_DIR/scripts/verify_install.py" --studio "$STUDIO_DIR"

  ok "Installation et vérification complète terminées !"
  echo ""
  echo "Pour démarrer une session :"
  echo "  1. (Optionnel Live GUI) : $STUDIO_DIR/scripts/launch-gridline.sh"
  echo "  2. Ouvrir $STUDIO_DIR dans Antigravity ou Zed."
  echo "  3. Consulter le tutoriel : $STUDIO_DIR/docs/01_SYSTEME_STACK_ET_GUIDE.md"
}

check_fedora
install_system_deps
install_gridline_and_stems
check_or_install_zed
setup_python_venv
setup_models_and_openutau
install_native_plugins
configure_workspace_and_mcps
index_samples_and_verify
