# Banques de voix OpenUTAU / DiffSinger

Dépose ici tes dossiers de banques de voix **OpenUTAU** ou **DiffSinger** (contenant `character.txt` / `character.yaml` / `dsconfig.yaml`).

Le serveur MCP `vocaloid` scanne automatiquement :
1. `/home/gamo/Bureau/musicos/voicebanks/`
2. `~/.local/share/OpenUtau/Singers/`

OpenUTAU Linux (`0.1.576`) est installé dans :
`/home/gamo/Bureau/musicos/tools/OpenUtau/OpenUtau`

Même sans voicebank DiffSinger installée, l'outil MCP `synthesize_vocal_preview` génère des guides vocaux chantés en `.wav` calés sur le BPM et les notes MIDI du morceau, prêts à être chargés dans Gridline (ou passés dans `gl.vocoder` Cipher / `gl.pitcher` Plumb).
