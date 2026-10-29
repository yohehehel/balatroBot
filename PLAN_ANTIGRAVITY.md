# Spécification Technique : Développement d'un Bot Heuristique Balatro (Objectif > 50% Winrate)

## 1. Contexte et Règles d'Or pour l'Agent
Tu es chargé de concevoir et coder un bot Python modulaire et déterministe capable de gagner au moins 50 % de ses parties sur Balatro via l'API JSON-RPC locale du dépôt `coder/balatrobot`.

### Directives strictes d'ingénierie :
1. **Inspection préalable obligatoire :** Avant d'écrire la moindre ligne de code, lis intégralement le dossier `docs/` — et tout particulièrement le fichier **`docs/api.md`** qui contient l'intégralité des identifiants API et des effets associés pour tous les Jokers, les consommables (Tarots, Planètes, Spectrales) ainsi que les modificateurs de cartes (Enhancements, Editions, Seals). Lis également `CLAUDE.md`, `balatrobot.lua`, et inspecte les dossiers `src/` et `tests/`. Réutilise au maximum les schémas, énumérations, modèles de données ou clients déjà présents dans `src/`.
2. **Isolation du code :** Ne modifie **jamais** les fichiers existants dans `src/` ou `docs/`. Tout ton code doit résider exclusivement dans le dossier `bots/` et ses tests dans `bots/tests/`.
3. **Gestionnaire d'environnement :** Utilise exclusivement `uv` (`uv run pytest`, `uv run python ...`) avec Python 3.13.
4. **Exécution par silos :** Tu dois implémenter le projet étape par étape (Silo 1 à Silo 6). À la fin de chaque silo, exécute les tests unitaires hors-ligne (`pytest`), affiche le résultat, et donne à l'utilisateur la commande exacte pour effectuer le test en direct sur le jeu avant de passer au silo suivant.

---

## 2. Paramètres de Jeu et Stratégie Cible
* **Configuration de départ fixe :** `Red Deck` (+1 défausse par manche) | `White Stake` (Mise Blanche).
* **Politique sur les Blindes :** Jouer **100 % des blindes** (Small, Big, Boss). Interdiction formelle d'utiliser l'action `skip` afin de maximiser les intérêts économiques et le nombre de boutiques visitées.
* **Condition de fin de partie :** 
  * Par défaut : Arrêt et enregistrement d'une **Victoire** dès que le Boss de l'**Ante 8** est battu.
  * Optionnel : Si le flag CLI `--endless` est activé, continuer la partie après l'Ante 8 jusqu'à la défaite pour mesurer le score maximal.
* **Exploitation directe du référentiel `docs/api.md` :** L'ensemble des noms d'API et des effets des Jokers, consommables (Tarots, Planètes, Spectrales) et modificateurs de cartes étant déjà intégralement documenté dans `docs/api.md` (et typé dans `src/`), il ne faut créer aucun fichier JSON externe de référence. Structure directement en Python (via des dictionnaires de mapping propres, des dataclasses ou les énumérations existantes de `src/`) les catégories, priorités (Tier List) et fonctions d'évaluation des éléments de jeu en te basant strictement sur `docs/api.md`.

---

## 3. Architecture Cible du Dossier `bots/`

```text
bots/
├── __init__.py
├── cli.py                      # Point d'entrée CLI (--delay, --fast, --runs, --endless, --port)
├── client_wrapper.py           # Client JSON-RPC (POST http://localhost:12346/), gestion du délai et retries
├── state_parser.py             # Extraction et normalisation de la réponse RPC `gamestate`
├── combat_solver.py            # Évaluateur de score exact (Chips x Mult) + Heuristique de défausse
├── shop_manager.py             # Gestion économique (intérêts), achats, ventes et tri des Jokers
├── pack_consumable_manager.py  # Ouverture des boosters et utilisation des Tarots/Planètes/Spectrales
├── runner.py                   # Machine à états principale orchestrant une partie de bout en bout
├── telemetry.py                # Enregistrement dans `bots/runs_history.csv` et statistiques de winrate
└── tests/
    ├── fixtures/               # Captures JSON réelles de `gamestate` (menu, blind, combat, shop, pack)
    ├── test_silo1_state.py
    ├── test_silo2_combat.py
    ├── test_silo3_discard.py
    ├── test_silo4_shop.py
    ├── test_silo5_packs.py
    └── test_silo6_telemetry.py
```

---

## 4. Plan d'Implémentation Siloté (Étapes 1 à 6)

### SILO 1 : Fondations, Client RPC, Parser d'État & Navigation (Blindes)
**Objectif :** Lire la documentation du repo (`docs/api.md`, `CLAUDE.md`, `src/`), construire le client RPC avec contrôle de vitesse (`--delay 0.8` par défaut, `--fast` pour `0.0`), parser l'état `gamestate`, démarrer une partie en `Red Deck` / `White Stake` et sélectionner automatiquement la blinde active sans jamais la passer.

* **Livrables :** `bots/client_wrapper.py`, `bots/state_parser.py`, `bots/cli.py` (squelette), `bots/tests/test_silo1_state.py`.
* **Validation Hors-Ligne (`pytest`) :** Tester le parsing d'une fixture JSON de `gamestate` conforme à `docs/api.md` et vérifier que la décision sur l'écran de sélection des blindes renvoie toujours l'action de sélection de la blinde courante.
* **Validation En Direct (Live Test) :** Une commande permettant de capturer l'état actuel du jeu dans `bots/tests/fixtures/`, de lancer une nouvelle run `Red Deck / White Stake` depuis le menu et d'entrer dans le premier combat.

---

### SILO 2 : Solveur Mathématique de Mains (Évaluation du Score Exact)
**Objectif :** Construire le moteur de calcul de score pour la phase de combat en s'appuyant sur les spécifications de `docs/api.md`.
1. Générer toutes les sous-combinaisons jouables de 1 à 5 cartes parmi les cartes en main ($\sum_{k=1}^{5} \binom{N}{k}$, soit au plus 381 combinaisons pour 8 cartes).
2. Identifier la main de poker correspondante (Carte Haute, Paire, Double Paire, Brelan, Suite, Couleur, Full, Carré, Quinte Flush) et récupérer ses `Base Chips` et `Base Mult` (à trouver dans `docs/api.md`) selon son niveau actuel dans l'état du jeu (`gamestate`).
3. Calculer les `Chips`, `+Mult` et `xMult` des cartes jouées (en incluant leurs modificateurs : Enhancements, Editions, Seals tels que définis dans `docs/api.md`), des cartes tenues en main (ex: Acier) et des Jokers actifs dans l'ordre exact de gauche à droite.
4. Choisir et jouer la combinaison qui maximise le score (ou qui bat la blinde en un coup en économisant les cartes de valeur).

* **Livrables :** `bots/combat_solver.py` (partie évaluation mathématique et prise en compte des Jokers/modificateurs), `bots/tests/test_silo2_combat.py`.
* **Validation Hors-Ligne (`pytest`) :** Vérifier sur au moins 5 mains de test (avec et sans Jokers/modificateurs de cartes) que le score calculé par `combat_solver.py` correspond au point près à la formule officielle de Balatro et que la meilleure combinaison d'indices de cartes est sélectionnée.
* **Validation En Direct (Live Test) :** Lancer le bot au milieu d'un combat et vérifier dans la console qu'il affiche le score prédit avant de jouer la main, et que ce score correspond au score réel affiché dans le jeu.

---

### SILO 3 : Heuristique de Défausse Ciblée (Exploitation du Red Deck)
**Objectif :** Enrichir `combat_solver.py` pour décider s'il faut **Jouer** ou **Défausser** (jusqu'à 5 cartes à la fois).
* **Règles décisionnelles :**
  1. **Kill Shot immédiat :** Si la meilleure main actuelle suffit à atteindre les jetons requis pour battre la blinde (`chips_needed <= predicted_score`), **JOUER immédiatement**.
  2. **Plus de défausses (`discards_left == 0`) :** **JOUER** la meilleure main (ou jouer une "main poubelle" en guise de défausse déguisée s'il reste plusieurs mains à jouer et que le score actuel est dérisoire).
  3. **Évaluation du potentiel de tirage (si `discards_left > 0` et que la main actuelle est faible) :**
     * *Chasse à la Couleur (Flush) :* Si 4 cartes de la même couleur sont en main (et qu'il reste assez de cartes de cette couleur dans le deck restant), garder ces 4 cartes et défausser jusqu'à 5 des autres cartes.
     * *Chasse à la Suite (Straight) :* Si 4 cartes consécutives (tirage par les deux bouts) sont en main, garder ces 4 cartes et défausser le reste.
     * *Chasse au Full / Carré / Brelan :* Si une Paire haute ou un Brelan est présent, garder les cartes appariées et défausser jusqu'à 5 cartes isolées de faible valeur (en évitant de défausser les cartes dotées de modificateurs précieux comme les Sceaux Bleus/Violets sauf si pertinent).
  4. **Adaptation au Boss Blind :** Désactiver la chasse à une couleur si le Boss actuel annule cette couleur (*The Club, The Goad, The Window, The Head*), ou forcer 5 cartes si le Boss est *The Psychic* (selon les identifiants de Boss définis dans `docs/api.md`).

* **Livrables :** Mise à jour de `bots/combat_solver.py` et `bots/tests/test_silo3_discard.py`.
* **Validation Hors-Ligne (`pytest`) :** Tester des mains à 4 cartes d'une couleur (doit défausser les autres), des mains gagnantes en 1 coup (doit jouer sans défausser), et le comportement face au Boss *The Psychic*.
* **Validation En Direct (Live Test) :** Faire jouer au bot l'intégralité de l'Ante 1 (3 blindes d'affilée) en mode `--delay 0.8` et observer ses choix de défausses et de mains jouées.

---

### SILO 4 : Moteur d'Économie, Boutique & Positionnement des Jokers
**Objectif :** Gérer la phase `CASH_OUT` et `SHOP` pour construire un moteur de score durable à partir des Jokers répertoriés dans `docs/api.md`.
* **Gestion des intérêts :** 1$ d'intérêt est gagné par tranche de 5$ possédés en fin de manche (plafonné à 25$ par défaut, soit 5$ d'intérêts).
  * *Ante 1 & début Ante 2 (Phase de survie) :* Autoriser à descendre sous les paliers d'intérêts pour acheter un premier Joker de score (`+Mult` ou `+Chips`).
  * *À partir de l'Ante 2 stabilisé :* Protéger les paliers de 5$ (ne dépenser que l'argent au-dessus du palier de 5$, 10$, 15$, 20$, 25$, sauf si un Joker Tier S ou `xMult` critique apparaît).
  * *Rerolls :* N'utiliser le Reroll que si l'argent restant après reroll reste $\ge 25\$$ (ou en cas d'urgence absolue avant un Boss en fin de partie).
* **Évaluation et Achat des Jokers :** Structurer dans `bots/shop_manager.py` la classification des Jokers (d'après leurs noms et effets exacts dans `docs/api.md`) par rôle (`Economy`, `Chips`, `+Mult`, `xMult`, `Retrigger/Copy`) et par score de priorité. Éviter les doublons de rôle inutiles ; viser un équilibre typique : 1 Joker Economy/Chips, 1-2 Jokers `+Mult`, 2-3 Jokers `xMult`.
* **Tri automatique des Jokers (Crucial) :** Après chaque achat ou avant chaque combat, réordonner systématiquement les Jokers possédés de gauche à droite :
  1. Jokers d'économie / utilitaires ou `+Chips` à gauche.
  2. Jokers de `+Mult` additif au centre.
  3. Jokers de `xMult` multiplicatif **tout à droite** (et s'assurer que *Blueprint* ou *Brainstorm* est placé de manière à copier le meilleur Joker de score).

* **Livrables :** `bots/shop_manager.py`, `bots/tests/test_silo4_shop.py`.
* **Validation Hors-Ligne (`pytest`) :** Tester la fonction de respect des paliers d'intérêts, la décision d'achat/remplacement d'un Joker faible, et l'algorithme de tri gauche-droite (`+Chips` -> `+Mult` -> `xMult`).
* **Validation En Direct (Live Test) :** Placer le jeu dans une boutique, lancer le script de test du Silo 4 et vérifier que le bot achète les éléments pertinents, réordonne ses Jokers et quitte la boutique proprement.

---

### SILO 5 : Gestion des Boosters & Consommables (Planètes, Tarots, Spectrales)
**Objectif :** Gérer l'achat/ouverture des Boosters (`Arcana`, `Celestial`, `Buffoon`, `Standard`, `Spectral`) et l'utilisation des cartes consommables en s'appuyant sur leurs identifiants et effets dans `docs/api.md`.
* **Planètes (`Celestial`) :** Prioriser l'amélioration des mains les plus faciles à stabiliser (Couleur / *Jupiter*, Paire / *Mercury*, Double Paire / *Uranus*, Brelan / *Venus*, Suite / *Saturn*) ou de la main la plus jouée durant la run en cours. Utiliser les cartes Planètes dès leur acquisition.
* **Tarots (`Arcana`) & Spectrales (`Spectral`) :**
  * Priorité 1 (Économie / Jokers) : *The Hermit* (double l'argent), *Temperance* (valeur de vente des Jokers), *The Soul* / *Wraith* / *Ankh* / * Ectoplasm* selon la situation.
  * Priorité 2 (Modificateurs de cartes et épuration du deck) : *The Empress* (+4 Mult), *The Hierophant* (+30 Chips), *The Chariot* (Steel), *Justice* (Glass), *Death* (copier la meilleure carte sur la plus faible), *The Hanged Man* / *Immolate* (détruire des cartes basses), Sceaux (*Deja Vu*, *Trance*, *Medium*, *Talisman*).
* **Sélection de cartes cibles :** Respecter strictement le nombre de cartes cibles requis par chaque Tarot ou carte Spectrale (tel que documenté dans `docs/api.md`) en sélectionnant les cartes appropriées dans la main avant d'envoyer l'action d'utilisation, ou vendre le consommable s'il est inutilisable et bloque un emplacement.

* **Livrables :** `bots/pack_consumable_manager.py`, `bots/tests/test_silo5_packs.py`.
* **Validation Hors-Ligne (`pytest`) :** Tester le choix de la meilleure carte dans un booster *Celestial*, *Arcana* et *Spectral* simulé, ainsi que la sélection préalable des cartes en main avant l'utilisation d'un consommable ciblé.
* **Validation En Direct (Live Test) :** Ouvrir un Booster Pack en jeu et laisser le bot choisir et appliquer la carte automatiquement.

---

### SILO 6 : Boucle Complète Multi-Runs, Mode Endless & Télémétrie CSV
**Objectif :** Assembler tous les modules dans `bots/runner.py` et `bots/telemetry.py` pour enchaîner $N$ parties sans intervention humaine, gérer les erreurs ou états transitoires, et mesurer le taux de victoire.
* **Gestion de la victoire (Ante 8) vs Endless :**
  * Dès que l'Ante passe à 9 (ou que l'état indique que le Boss de l'Ante 8 est vaincu), enregistrer `win = True` dans la télémétrie.
  * Si `--endless` est absent, arrêter la run proprement et lancer la suivante. Si `--endless` est présent, continuer jusqu'au `GAME_OVER` tout en conservant `win = True` et en notant l'`endless_max_ante`.
* **Fichier `bots/runs_history.csv` :** Ajouter une ligne à la fin de chaque partie avec les colonnes :
  `timestamp, seed, deck, stake, win, max_ante, final_round, fatal_boss, max_money, most_played_hand, final_jokers, duration_sec`
* **Affichage Console :** Afficher à la fin de la session le bilan global (`Total Runs`, `Wins`, `Winrate %`, `Ante Moyen`, `Top 3 Boss Fatals`).

* **Livrables :** `bots/runner.py`, `bots/telemetry.py`, `bots/cli.py` finalisé, `bots/tests/test_silo6_telemetry.py`.
* **Validation Hors-Ligne (`pytest`) :** Tester la machine à états complète sur une séquence de transitions simulées et vérifier la création et le formatage exact du fichier `runs_history.csv`.
* **Validation En Direct (Live Benchmark) :** Lancer 5 parties complètes avec `uv run python -m bots.cli --runs 5 --fast` pour valider la stabilité de bout en bout.