# Stack Technologique : Workflow de Production Musicale Assistée par IA

Ce document définit l'architecture technique complète permettant de générer, arranger, chanter et mixer un morceau de musique de bout en bout à partir d'un simple prompt, sur **Fedora 44**, en combinant **Gridline**, **Zed** et **Antigravity**.

---

## 1. Vue d'ensemble de l'architecture

Le système repose sur une architecture modulaire pilotée par **MCP (Model Context Protocol)**. L'agent IA dans Zed agit comme chef d'orchestre et communique avec quatre outils spécialisés :

```text
[ Utilisateur : Prompt & Direction Artistique ]
                     │
                     ▼
   ┌───────────────────────────────────────┐
   │    Zed IDE + Extension Antigravity    │  ◄── Règles projet (.rules) & Blueprint JSON
   │      (Cerveau : Gemini / Local)       │
   └───────────────────┬───────────────────┘
                       │ Protocole MCP (JSON-RPC / stdio / HTTP)
       ┌───────────────┼───────────────┬───────────────────┐
       ▼               ▼               ▼                   ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐     ┌─────────────┐
│  Gridline   │ │ Audio Index │ │  Vocaloid   │     │   Audio     │
│ (MCP natif) │ │ (MCP Python)│ │ (MCP Python)│     │  Analyzer   │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘     └──────┬──────┘
       │               │               │                   │
  DAW & Lua       CLAP2 +         OpenUTAU /          pyloudnorm +
  (175+ DSP)      Essentia        DiffSinger          Essentia DSP
       │               │               │                   │
       └───────────────┴───────────────┴───────────────────┘
                               │
                               ▼
              [ Fedora 44 : PipeWire & Wayland ]
```

---

## 2. Détail des couches technologiques

### Couche 1 : Système & Moteur Audio (Fedora 44)
* **OS :** Fedora 44 (environnement Wayland natif).
* **Serveur Audio :** **PipeWire** (avec passerelles `pipewire-jack` et `pipewire-pulse`).
* **Objectif :** Garantir une faible latence sans configuration JACK complexe, tout en permettant à Gridline, aux synthés natifs et aux rendus vocaux de cohabiter proprement.

### Couche 2 : Cerveau & Orchestration (Zed + Antigravity)
* **Éditeur / Workspace :** **Zed** (ultra-léger, natif Rust, support MCP intégré).
* **Agent IA :** **Antigravity** (propulsé par Gemini ou un modèle local en secours).
* **Artefacts de contrôle :**
  * **Fichier `.rules` (ou instructions système) :** Impose à l'IA les bonnes pratiques musicales (gestion des fréquences, respect de la tonalité, étape de validation du blueprint avant génération).
  * **`song_blueprint.json` :** Contrat de structure généré en début de session (BPM, gamme, découpage par mesures, répartition fréquentielle des pistes, paroles).

### Couche 3 : Station Audionumérique (Gridline)
* **Logiciel :** **Gridline** (binaire Linux précompilé, support Wayland dédié).
* **Interface IA native :** Serveur **MCP intégré** (`gridline mcp --attach` ou endpoint HTTP local `http://127.0.0.1:47390/mcp`) avec mode *headless* possible pour le rendu en arrière-plan.
* **Sécurité d'édition :** Historique `undo` / `redo` isolé pour les actions de l'IA (ne détruit jamais tes modifications manuelles).
* **Moteur de scripting :** **Lua sandboxé** pour la génération algorithmique (accords, arpèges, humanisation MIDI, courbes d'automation).
* **Moteur sonore :**
  * **Natif :** Plus de 175 instruments et effets intégrés (dont *Plasma*, *Seltmann*, *Multisampler SFZ*, *Speech*) pour garantir une fiabilité totale sans dépendance externe.
  * **Tiers (Formats VST3, CLAP, VST2, LV2) :** Vital, Surge XT, ZL Equalizer, suite LSP Plugins, Dragonfly Reverb, CHOW Tape Model, Decent Sampler.
* **Interopérabilité :** Import et export natif au format **`.flp`** (FL Studio).

### Couche 4 : Mémoire Sonore (`mcp-audio-index`)
Serveur MCP local écrit en **Python** (via `mcp` / `FastMCP`) qui évite le tri manuel des samples et empêche l'IA de choisir des sons incompatibles.
* **Compréhension sémantique :** **LAION CLAP2** (`clap2-xxs` ou `clap2-xs`) transforme chaque fichier `.wav` en vecteur pour rechercher par description (*« kick sec punchy »*, *« pad sombre ambiant »*).
* **Analyse musicale :** **Essentia** extrait automatiquement la tonalité (*key*), le BPM et le centroïde spectral (grave/médium/aigu).
* **Stockage :** Base locale légère (**SQLite** + recherche vectorielle ou **Qdrant/FAISS** local) interrogée en quelques millisecondes par l'agent.

### Couche 5 : Synthèse Vocale & Chant (`mcp-vocaloid`)
Serveur MCP dédié à l'automatisation complète des voix chantées.
* **Niveau rapide (intégré) :** Plugin natif **Speech** de Gridline pour les voix robotiques, vocodeurs et tags.
* **Niveau chant réaliste / Vocaloid :** **OpenUTAU** + moteur neuronal **DiffSinger** (open-source, compatible Linux).
* **Mécanisme :** L'agent écrit la mélodie et les paroles syllabées dans un fichier **`.ustx`** (format texte YAML natif d'OpenUTAU), déclenche la synthèse audio en ligne de commande vers `vocals.wav`, puis importe la piste directement dans Gridline.

### Couche 6 : Boucle d'Auto-Mixage (`mcp-audio-analyzer`)
Donne des « oreilles mathématiques » à l'IA pour corriger le mixage sans intervention humaine.
* **Bibliothèques Python :** `pyloudnorm` (normes broadcast ITU-R BS.1770) + `essentia` / `librosa`.
* **Métriques extraites sur les rendus offline de Gridline :**
  * **Loudness intégré (LUFS)** et **True Peak (dBTP)** (détection de saturation/clipping).
  * **Répartition spectrale par bandes** (Sub, Bass, Low-Mids, High-Mids, Air).
  * **Corrélation de phase et masquage** dans le bas du spectre (conflits Kick/808).
* **Boucle fermée :** `Render → Analyse JSON → Ajustement faders/EQ/Limiteur dans Gridline → Re-vérification`.

---

## 3. Arborescence recommandée du Workspace

```text
~/Music/AI-Studio/
├── .zed/
│   └── settings.json            # Déclaration des serveurs MCP pour Zed
├── .rules                       # Règles de production musicale pour Antigravity
├── mcp-servers/
│   ├── audio-index/             # Serveur MCP CLAP2 + Essentia + base SQLite
│   ├── vocaloid/                # Serveur MCP OpenUTAU / DiffSinger CLI
│   └── audio-analyzer/          # Serveur MCP pyloudnorm + analyse spectrale
├── samples/                     # Banque de samples bruts (indexée automatiquement)
├── voicebanks/                  # Banques de voix DiffSinger / OpenUTAU
└── projects/
    └── mon-morceau/
        ├── song_blueprint.json  # Plan validé (BPM, gamme, structure, paroles)
        ├── projet.gridline      # Session Gridline active
        ├── scripts/             # Scripts Lua générés par l'IA
        ├── vocals/              # Fichiers .ustx et exports vocals.wav
        └── renders/             # Exports temporaires pour la boucle d'auto-mixage
```

---

## 4. Feuille de route de mise en place (par phases)

Pour éviter toute friction technique, la stack se déploie en 4 étapes progressives :

| Phase | Briques activées | Capacité débloquée |
| :--- | :--- | :--- |
| **Phase 1 (MVP)** | Fedora 44 + Gridline + Zed/Antigravity (`gridline mcp`) | Création complète d'instrumentales par prompt en utilisant les 175+ sons natifs et Lua. |
| **Phase 2 (Samples)** | Ajout de `mcp-audio-index` (CLAP2 + Essentia) | L'IA pioche intelligemment dans ta propre banque de samples sans fausse note. |
| **Phase 3 (Mixage)** | Ajout de `mcp-audio-analyzer` (`pyloudnorm`) | L'IA écoute ses rendus, corrige les volumes/EQ et évite le clipping automatiquement. |
| **Phase 4 (Chant)** | Ajout de `mcp-vocaloid` (OpenUTAU + DiffSinger) | Génération complète de chansons avec paroles, mélodie vocale et mixage voix/instru. |