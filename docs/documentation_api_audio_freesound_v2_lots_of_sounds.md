# Guide & Documentation Technique : Freesound API v2 & Lots of Sounds API

Ce document détaille le fonctionnement, l'authentification, les endpoints clés et les bonnes pratiques pour intégrer **Freesound API v2** et **Lots of Sounds API** dans un workflow de recherche et téléchargement automatisé de samples.

---

## 1. Comparatif Rapide

| Critère | Freesound API v2 | Lots of Sounds API |
| :--- | :--- | :--- |
| **Base URL** | `https://freesound.org/apiv2` | `https://api.lotsofsounds.com` |
| **Catalogue** | +600 000 sons (communautaire, très vaste) | +12 000 sons (curaté, orientation SFX/Lo-Fi/Beats) |
| **Licences** | Mixte (`CC0`, `CC-BY`, `CC-BY-NC`, `Sampling+`) | **100 % CC0** (Domaine public, zéro attribution) |
| **Formats audio** | Previews : MP3/OGG (HQ) • Originaux : WAV, AIFF, FLAC | MP3 uniquement |
| **Authentification** | Clé API (recherche/previews) • OAuth2 (WAV originaux) | Clé API (`x-api-key`) • Endpoint `/sample` public |
| **Tarification API** | Gratuit (soumis à quota de requêtes) | Clé API requise à partir du plan Pro (15 $/mois) |
| **Support IA / MCP** | Via wrappers communautaires | Serveur MCP natif (`https://api.lotsofsounds.com/mcp`) |

---

## 2. Freesound API v2

Freesound est la base de données ouverte la plus riche pour la recherche acoustique avancée (filtrage par note, BPM, durée, licence).

### A. Authentification
1. Crée un compte sur [freesound.org](https://freesound.org).
2. Demande une clé API sur [freesound.org/apiv2/apply](https://freesound.org/apiv2/apply).
3. Passe ton token dans l'URL (`?token=TA_CLE`) ou via le header HTTP :
   ```http
   Authorization: Token TA_CLE_API
   ```
*(Note : Le téléchargement des fichiers **originaux non compressés** nécessite un flux OAuth2 complet, mais les `previews` HQ en MP3/OGG à 128-320 kbps sont accessibles directement avec le simple Token).*

### B. Endpoints Essentiels

#### 1. Recherche textuelle et filtrée
`GET /apiv2/search/text/`

**Paramètres clés :**
* `query` : Mots-clés (ex: `808 bass`, `kick punchy`).
* `filter` : Syntaxe Apache Solr très puissante pour isoler de vrais samples exploitables.
* `sort` : `score` (pertinence), `rating_desc` (mieux notés), `downloads_desc` (plus populaires).
* `fields` : Liste des champs à retourner (indispensable pour obtenir les URLs de téléchargement dès la recherche et éviter une requête par son).
* `page_size` : Nombre de résultats (max `150`).

**Exemple de requête cURL (chercher 4 basses 808 libres de droits, courtes et bien notées) :**
```bash
curl -G "https://freesound.org/apiv2/search/text/" \
  -H "Authorization: Token TA_CLE_API" \
  --data-urlencode "query=808 bass" \
  --data-urlencode "filter=duration:[0.2 TO 4.0] license:\"Creative Commons 0\" type:(wav OR aiff)" \
  --data-urlencode "sort=rating_desc" \
  --data-urlencode "fields=id,name,duration,tags,previews,license" \
  --data-urlencode "page_size=4"
```

#### 2. Détails d'un son spécifique
`GET /apiv2/sounds/{sound_id}/`

Permet de récupérer l'intégralité des métadonnées, y compris les analyses acoustiques si activées.

#### 3. Recherche par similarité acoustique
`GET /apiv2/sounds/{sound_id}/similar/`

Une fois que tu as trouvé un bon kick ou une bonne 808 (`sound_id`), cet endpoint te renvoie les sons qui ont une signature spectrale proche.

---

## 3. Lots of Sounds API

Lots of Sounds est conçu pour la rapidité d'intégration, les agents IA et l'assurance juridique (tout est en CC0).

### A. Authentification
* **Mode Test (sans compte) :** L'endpoint `/api/v1/sounds/sample` est ouvert à tous et renvoie 6 sons exemples.
* **Accès complet :** Génère une clé API (préfixée par `los_`) depuis ton tableau de bord et passe-la dans le header :
  ```http
  x-api-key: los_ta_cle_api
  ```

### B. Endpoints Essentiels

#### 1. Recherche dans le catalogue
`GET /api/v1/sounds`

**Paramètres clés :**
* `q` : Mot-clé court (ex: `808`, `snare`, `vinyl`).
* `category` : `sfx`, `music`, `lofi`, `ambient`.
* `tag` : Filtrer par tag précis.
* `min_duration` / `max_duration` : Durée en secondes (ex: `max_duration=3`).
* `limit` : Nombre de résultats par page.

**Exemple de requête cURL :**
```bash
curl -G "https://api.lotsofsounds.com/api/v1/sounds" \
  -H "x-api-key: los_ta_cle_api" \
  -d "q=808" \
  -d "max_duration=4" \
  -d "limit=4"
```

#### 2. Récupérer le lien de téléchargement
`GET /api/v1/sounds/{sound_id}/download`

Contrairement à Freesound qui expose les liens directement dans la recherche, Lots of Sounds génère une **URL signée temporaire** (valide 1 heure) à la demande.

```bash
curl "https://api.lotsofsounds.com/api/v1/sounds/SOUND_ID/download" \
  -H "x-api-key: los_ta_cle_api"
```
**Réponse JSON :**
```json
{
  "data": {
    "download_url": "https://cdn.lotsofsounds.com/..."
  }
}
```

### C. Intégration MCP (Pour Claude, Cursor, etc.)
Si tu pilotes ta recherche via un assistant IA, inutile d'écrire un client HTTP : connecte directement le serveur MCP hébergé :
* **URL MCP :** `https://api.lotsofsounds.com/mcp`
* **Header :** `x-api-key: los_ta_cle_api`

---

## 4. Conseils & Bonnes Pratiques pour la Production Musicale

### 1. Évite le piège des "Longs Enregistrements" sur Freesound
Freesound regorge d'enregistrements de 10 minutes où quelqu'un teste un synthétiseur.
* **Règle d'or :** Utilise **toujours** le filtre `duration:[0.1 TO 5.0]` pour tes *one-shots* (kicks, snares, 808) ou `duration:[4.0 TO 32.0]` pour des boucles.

### 2. Nettoie tes requêtes de recherche
* **Freesound** supporte les opérateurs booléens : `query=808 (sub OR bass) -loop -demo` permet d'exclure les démos de morceaux complets.
* **Lots of Sounds** utilise une correspondance simple sur les titres/descriptions : privilégie des **mots-clés très courts** (`q=808` ou `q=sub` plutôt que `q=deep distorted 808 bass`).

### 3. Optimisation du Rate-Limiting et des appels réseau
* Sur **Freesound**, spécifie toujours le paramètre `fields=id,name,previews`. Par défaut, l'API ne renvoie que l'`id`, le `name` et les `tags`, ce qui t'obligerait à faire 4 appels supplémentaires à `/sounds/{id}` pour récupérer les URLs audio.
* Sur **Lots of Sounds**, surveille le header `X-RateLimit-Remaining` (limite de 2 500 requêtes/jour sur le plan Pro).

### 4. Gestion des licences en prod commerciale
* Si tu comptes sortir tes morceaux sur Spotify/Apple Music sans te soucier des crédits :
  * Sur **Freesound**, verrouille le filtre `license:"Creative Commons 0"`. Sinon, une licence `CC-BY` t'oblige légalement à créditer l'auteur du sample dans les métadonnées du morceau.
  * Sur **Lots of Sounds**, tu n'as aucun filtre de licence à gérer, tout est CC0 par défaut.

### 5. Normalisation automatique post-téléchargement
Les samples issus de banques communautaires ont des niveaux de volume très hétérogènes. Si tu automatises le téléchargement via un script (Node.js/Python), passe un coup de `ffmpeg` à la volée pour convertir les MP3 en WAV 24-bit/44.1kHz et normaliser le pic à `-1 dB` avant de les glisser dans ton DAW (Ableton, FL Studio, Logic) :
```bash
ffmpeg -i input.mp3 -af "loudnorm=I=-14:TP=-1.0:LRA=11" -ar 44100 -sample_fmt s24 output_808.wav
```