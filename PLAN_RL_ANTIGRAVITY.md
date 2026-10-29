# Spécification Technique : Agent Neuro-Symbolique Macro-RL pour BalatroBot

## 1. Vision Architecturale : Paradigme Neuro-Symbolique (Macro-RL + Micro-Solveur)
Le bot heuristique actuel dispose d'un moteur de combat mathématiquement optimal (`combat_solver.py`), mais plafonne à ~4.3% de winrate à cause de sa rigidité macro-stratégique (refus de skipper les blindes, thésaurisation excessive des 25$, absence de synergies d'archétype).

Nous implémentons une architecture **Neuro-Symbolique** où :
1. **L'Agent Macro-RL (`MaskablePPO`)** prend exclusivement les décisions stratégiques (~20 à 35 décisions par partie) :
   - Jouer ou Skipper une blinde (pour récupérer un Tag).
   - Gérer la boutique (Acheter Joker/Voucher/Booster, Vendre Joker, Reroll, Quitter).
   - Choisir quelle carte sélectionner à l'intérieur de TOUS les Boosters ouverts (`Celestial`, `Arcana`, `Spectral`, `Buffoon`, `Standard`) ou les skipper, afin de construire de vraies synergies (ex: prendre les Planètes et les Tarots de conversion de couleur alignés avec ses Jokers).
   - Décider quand utiliser ou vendre un consommable gardé en inventaire.
2. **Le Micro-Solveur Déterministe (Code existant Silos 1 à 5)** exécute automatiquement la tactique de bas niveau :
   - Dès que l'agent RL choisit `PLAY_BLIND`, `combat_solver.py` joue l'intégralité de la manche de combat (sélection des mains et défausses optimales) en une seule transition d'environnement `step()`.
   - À chaque modification des Jokers, `shop_manager.compute_optimal_joker_order` trie automatiquement les Jokers de gauche à droite (`+Chips` -> `+Mult` -> `xMult` + placement de *Blueprint*/*Brainstorm*).
   - Lorsqu'un Tarot ou une carte Spectrale nécessite de cibler 1 à 3 cartes de la main (ex: *Death*, *The Hanged Man*, *The Star* pour convertir en Carreau), l'agent RL choisit le Tarot et délègue à `pack_consumable_manager.select_targets_for_consumable` la sélection mathématique des cartes cibles optimales dans la main.

### Directives strictes d'ingénierie pour l'Agent :
1. **Inspection préalable obligatoire :** Avant de coder, inspecte `docs/api.md`, `bots/state_parser.py`, `bots/combat_solver.py`, `bots/shop_manager.py`, `bots/pack_consumable_manager.py`, `bots/runner.py` et `bots/instance_pool.py`.
2. **Non-régression absolue :** Les 141 tests unitaires existants dans `bots/tests/` doivent continuer à passer à 100 %. Le mode `--agent heuristic` doit rester pleinement fonctionnel.
3. **Isolation du module RL :** Tout le nouveau code RL doit être placé dans le sous-package `bots/rl/` et ses tests dans `bots/tests/rl/`.
4. **Dépendances Python (via `uv`) :** Ajoute proprement `gymnasium`, `stable-baselines3`, `sb3-contrib`, `torch` et `tensorboard` (ou `rich`/`tqdm`) via `uv add`.
5. **Exécution par Silos (RL-1 à RL-5) :** Implémente un seul silo à la fois, valide-le avec `uv run pytest`, affiche les résultats et attends la validation de l'utilisateur avant de passer au silo suivant.

---

## 2. Arborescence Cible du Module `bots/rl/`

```text
bots/
├── rl/
│   ├── __init__.py
│   ├── constants.py          # Dimensions fixes (MAX_JOKERS=10, vocabulaires d'IDs, etc.)
│   ├── features.py           # Extracteur d'état -> Dict de tenseurs + `estimated_power_ratio`
│   ├── actions.py            # Espace Discrete(26), Action Masking et traducteur vers commandes RPC/Micro
│   ├── reward.py             # Fonction de Reward Shaping (Survie Ante, Marge de score, Économie)
│   ├── env.py                # `BalatroMacroEnv(gymnasium.Env)` encapsulant le Micro-Solveur
│   ├── network.py            # `BalatroFeaturesExtractor(BaseFeaturesExtractor)` (Self-Attention + MLP)
│   ├── train.py              # Script d'entraînement parallèle (`MaskablePPO` sur 6 instances par défaut)
│   └── eval.py               # Wrapper d'inférence pour `--agent rl` et benchmarking
└── tests/
    └── rl/
        ├── test_silo_rl1_features.py
        ├── test_silo_rl2_actions.py
        ├── test_silo_rl3_env_reward.py
        ├── test_silo_rl4_network.py
        └── test_silo_rl5_integration.py
```

---

## 3. Spécification Mathématique du MDP (Markov Decision Process)

### A. Constantes de Dimensionnement & Gestion de l'Overflow (`bots/rl/constants.py`)
- `MAX_JOKERS = 10` (couvre 5 slots de base + jusqu'à 5 Jokers Négatifs/Antimatter).
- `MAX_CONSUMABLES = 4`
- `MAX_SHOP_JOKERS = 4` (cartes individuelles en boutique)
- `MAX_SHOP_BOOSTERS = 2`
- `MAX_PACK_CHOICES = 5`
- **Règle d'Overflow (> 10 Jokers) :** Le Micro-Solveur (`combat_solver.py`) évalue toujours 100 % des Jokers possédés sans limite. Pour les tenseurs RL et l'action de vente, si `len(jokers) > 10`, on trie les Jokers pour exposer dans les 10 slots les Jokers vendables/pertinents sans jamais lever d'`IndexError`, et on renseigne le compte réel dans le vecteur scalaire de contexte.

### B. Espace d'Observation $\mathcal{S}$ (`gymnasium.spaces.Dict`)
L'état observé à chaque point de décision macro est un dictionnaire de tenseurs `float32` / `int64` normalisés :
1. **`context_ features` (`Box`, dim ~45) :**
   - Progression : `ante / 8.0`, `round_num / 24.0`, phase courante en one-hot (`BLIND_SELECT`, `SHOP`, `PACK_ARCANA`, `PACK_CELESTIAL`, `PACK_SPECTRAL`, `PACK_BUFFOON`, `PACK_STANDARD`, `SELECTING_HAND_CONSUMABLE`).
   - Économie : `money / 100.0`, `interest_cap / 100.0`, `current_interest_earned / 20.0`, `reroll_cost / 20.0`, `spendable_above_interest / 50.0`.
   - Capacités : `hands_per_round / 10.0`, `discards_per_round / 10.0`, `hand_size / 12.0`, `joker_slots_used / 10.0`, `joker_slots_max / 10.0`, `free_joker_slots / 5.0`, `consumable_slots_free / 4.0`.
   - **Feature clé anti-"Mort Riche" (`estimated_power_ratio`) :** Ratio $\log_{10}(\text{Estimated Score Capacity} / \text{Next Blind Target Score})$ borné dans $[-2.0, +2.0]$, où `Estimated Score Capacity` est estimé par `combat_solver` en simulant le score du build actuel sur la main type dominante du deck multiplié par `hands_per_round`.
   - Boss de l'Ante courant : ID du Boss et flags indiquant si ce Boss debuff notre couleur dominante, nos figures, ou notre main la plus jouée.
   - Tag proposé (si en `BLIND_SELECT`) : ID / catégorie du Tag de skip.
2. **`deck_features` (`Box`, dim 32) :**
   - Histogramme normalisé des 13 rangs (2 à As : proportion de chaque rang dans le deck complet actuel).
   - Histogramme normalisé des 4 couleurs (`Hearts`, `Diamonds`, `Clubs`, `Spades`) + couleur dominante concentration max.
   - Proportions des améliorations (7), éditions (3), sceaux (4) et ratio `total_cards / 52.0`.
3. **`hand_levels` (`Box`, dim 24) :**
   - Pour les 12 mains de poker : `level / 20.0` et proportion d'utilisation historique dans la run en cours.
4. **`held_jokers` (`Box`, shape `(10, D_entity)`) :**
   - Jusqu'à 10 Jokers possédés (paddés avec des 0) : `[joker_id_index, is_present, role_one_hot(6), edition_one_hot(5), sell_value_norm, dynamic_scaling_value_norm, synergy_suit_or_hand_id]`.
5. **`shop_and_pack_offers` (`Box`, shape `(12, D_entity)`) :**
   - Encode dans le même espace de représentation les offres courantes : les 4 cartes de la boutique, le Voucher, les 2 Boosters, ou les 5 cartes à l'intérieur d'un Booster ouvert, ainsi que les 4 consommables détenus en inventaire.

### C. Espace d'Actions $\mathcal{A}$ (`Discrete(26)`) & Masque Booléen (`action_masks`)
Un espace d'actions plat de 26 actions avec masquage strict des actions invalides :
- **Écran Blind Select (0-1) :**
  - `0` : `PLAY_BLIND` (valide en `BLIND_SELECT` ; lance la blinde et exécute tout le combat via `combat_solver.py`).
  - `1` : `SKIP_BLIND` (valide uniquement sur `Small Blind` et `Big Blind`, jamais sur `Boss Blind`).
- **Actions Boutique (2-10) :**
  - `2..5` : `BUY_SHOP_CARD(0..3)` (valide en `SHOP` si carte présente, argent suffisant, et slot disponible).
  - `6` : `BUY_VOUCHER` (valide en `SHOP` si voucher présent et argent suffisant).
  - `7..8` : `BUY_BOOSTER(0..1)` (valide en `SHOP` si booster présent et argent suffisant).
  - `9` : `REROLL_SHOP` (valide en `SHOP` si `money >= reroll_cost`).
  - `10` : `LEAVE_SHOP` (valide en `SHOP` ; quitte la boutique vers `BLIND_SELECT`).
- **Actions Booster Ouvert (11-16) :**
  - `11..15` : `SELECT_PACK_CARD(0..4)` (valide dans TOUS les boosters ouverts : Planète, Tarot, Spectrale, Buffoon, Standard si l'indice existe et est légal). Si la carte choisie est un Tarot/Spectrale nécessitant des cibles en main, appelle automatiquement `select_targets_for_consumable` avant d'envoyer la commande RPC.
  - `16` : `SKIP_PACK` (valide lorsqu'un booster est ouvert).
- **Gestion des Consommables en Inventaire (17-20) :**
  - `17..20` : `USE_OR_SELL_CONSUMABLE(0..3)` (si le consommable au slot $i$ est utilisable utilement maintenant, l'utilise avec ciblage automatique ; s'il est inutilisable et bloque le slot, le vend).
- **Vente de Jokers (21-25) :**
  - `21..25` : `SELL_JOKER(0..4)` (vend l'un des 5 Jokers principaux les moins prioritaires/non-éternels ; masqué en début de run tant que `free_joker_slots > 0` et qu'aucun besoin urgent de liquidité n'existe, pour éviter le suicide aléatoire en début d'entraînement).

---

## 4. Plan d'Implémentation Siloté (Silos RL-1 à RL-5)

### SILO RL-1 : Espace d'Observation, Feature Engineering & `estimated_power_ratio`
**Objectif :** Construire `bots/rl/constants.py` et `bots/rl/features.py` pour transformer n'importe quel `ParsedGameState` en un dictionnaire de tenseurs NumPy/PyTorch normalisés, incluant l'estimateur de puissance `estimated_power_ratio`.

* **Livrables :** `bots/rl/constants.py`, `bots/rl/features.py`, `bots/tests/rl/test_silo_rl1_features.py`.
* **Validation Hors-Ligne (`pytest`) :**
  1. Vérifier que toutes les fixtures JSON existantes (`menu`, `blind_select`, `shop`, `pack`, `combat`) sont converties en dictionnaires NumPy conformes au `gymnasium.spaces.Dict` sans `NaN` ni `Inf`.
  2. Tester le cas d'overflow avec un état contenant 11 Jokers (avec Négatifs) et vérifier que le tenseur `held_jokers` reste de forme `(10, D_entity)` sans erreur.
  3. Vérifier que `estimated_power_ratio` augmente lorsqu'on ajoute un Joker `xMult` ou qu'on améliore le niveau d'une main de poker, et diminue lorsque l'Ante augmente.

---

### SILO RL-2 : Espace d'Actions (`Discrete(26)`), Action Masking & Délégation Micro
**Objectif :** Construire `bots/rl/actions.py` qui calcule le masque booléen `action_masks(state) -> np.ndarray[bool]` de taille 26 et traduit l'entier d'action choisi par l'agent RL en appels concrets au client RPC et au Micro-Solveur (ciblage automatique des cartes de la main pour les Tarots/Spectrales, tri automatique des Jokers).

* **Livrables :** `bots/rl/actions.py`, `bots/tests/rl/test_silo_rl2_actions.py`.
* **Validation Hors-Ligne (`pytest`) :**
  1. Vérifier qu'en `BLIND_SELECT` sur une `Boss Blind`, seule l'action `0` (`PLAY_BLIND`) est `True` dans le masque (`SKIP_BLIND` est `False`).
  2. Vérifier qu'en `SHOP`, les achats trop chers ou dépassant la limite de slots de Jokers (hors Joker `Negative`) sont masqués à `False`, et que `LEAVE_SHOP` est toujours `True` (garantie qu'au moins 1 action est toujours valide).
  3. Vérifier que dans un `Arcana Pack` ouvert, choisir `SELECT_PACK_CARD(i)` correspondant à un Tarot ciblé (ex: *The Star* ou *Death*) appelle automatiquement `select_targets_for_consumable` pour fournir les indices des cartes de la main à l'API.

---

### SILO RL-3 : Environnement Gymnasium `BalatroMacroEnv` & Reward Shaping
**Objectif :** Construire `bots/rl/reward.py` et `bots/rl/env.py` implémentant l'interface standard `gymnasium.Env` (`reset()`, `step(action)`, `action_masks()`).
* **Fonctionnement de `step(action)` :**
  - Exécute l'action macro choisie.
  - Si l'action déclenche un combat (`PLAY_BLIND`), la méthode `step()` fait tourner automatiquement la boucle interne de combat via `combat_solver.py` (et `cash_out` en cas de victoire de manche) jusqu'au prochain point de décision macro (`SHOP`, `BLIND_SELECT` suivant) ou `GAME_OVER` / Victoire Ante 8.
  - Si un consommable en main doit être joué au début d'une manche avant le premier coup de `combat_solver`, l'environnement le gère proprement.
* **Fonction de Récompense (`bots/rl/reward.py`) :**
  - **Victoire de manche (Round gagné) :** $+0.5 + 0.2 \times \text{Ante} + 0.15 \times \text{clamp}(\log_{10}(\text{Score Obtenu} / \text{Score Requis}), 0, 2.0)$.
  - **Boss d'Ante vaincu :** Bonus additionnel de $+1.0 \times \text{Ante}$.
  - **Victoire Finale (Boss Ante 8 vaincu) :** $+50.0$ (`terminated = True`, `info["is_success"] = True`).
  - **Défaite (`GAME_OVER`) :** $-10.0$, modulé par une pénalité supplémentaire de **"Mort Riche"** : $-0.15 \times \max(0, \text{money} - 15)$ si le bot meurt avec beaucoup d'argent non dépensé en banque !
  - **Discipline de boutique :** $-0.05$ par `REROLL_SHOP` pour éviter les rerolls compulsifs inutiles ; léger shaping potentiel sur l'augmentation du `estimated_power_ratio`.

* **Livrables :** `bots/rl/reward.py`, `bots/rl/env.py`, `bots/tests/rl/test_silo_rl3_env_reward.py`.
* **Validation Hors-Ligne (`pytest`) :**
  1. Passer le validateur `check_env` de Gymnasium/Stable-Baselines3 sur un environnement mocké.
  2. Vérifier le calcul exact de la pénalité de "Mort Riche" (mourir avec 75$ en banque doit infliger une pénalité nettement plus forte que mourir après avoir tout dépensé pour tenter de survivre).
* **Validation En Direct (Live Test) :** Lancer 1 épisode complet avec un agent aléatoire masqué (`env.action_space.sample(mask)`) sur une instance locale pour vérifier que la boucle Macro-RL + Micro-Combat tourne jusqu'à la fin de la partie sans blocage.

---

### SILO RL-4 : Réseau de Neurones Hybride (Entity Self-Attention + MLP) & `MaskablePPO`
**Objectif :** Construire `bots/rl/network.py` (`BalatroFeaturesExtractor`) et le pipeline d'entraînement `bots/rl/train.py`.
* **Architecture PyTorch (`BalatroFeaturesExtractor`) :**
  1. **Encodeur d'Entités (Jokers & Offres Boutique/Pack) :**
     - Couche `nn.Embedding` pour les identifiants de Jokers/Consommables/Tags + projection linéaire des attributs continus vers $d_{\text{model}} = 64$.
     - Bloc `nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)` permettant aux Jokers possédés et aux offres de la boutique/booster d'interagir directement (détection des synergies de build).
     - Pooling moyen masqué (Mean + Max Pooling sur les entités présentes) produisant un vecteur de synergie de dimension 128.
  2. **Encodeur de Contexte & Deck (MLP) :**
     - Concaténation de `context_features`, `deck_features` et `hand_levels` passée dans un MLP `Linear -> LayerNorm -> SiLU` de dimension 128.
  3. **Fusion :**
     - Concaténation (`128 + 128 = 256`) suivie d'une couche `Linear(256, 256) -> LayerNorm -> SiLU` envoyée aux têtes `pi` (Acteur) et `vf` (Critique) de `MaskablePPO`.

* **Livrables :** `bots/rl/network.py`, `bots/rl/train.py` (mode 1 instance + checkpoints + logs TensorBoard/CSV), `bots/tests/rl/test_silo_rl4_network.py`.
* **Validation Hors-Ligne (`pytest`) :**
  1. Tester le passage *forward* d'un batch d'observations dans `BalatroFeaturesExtractor` et dans `MaskablePPO` avec `action_masks`.
  2. Vérifier que les gradients se propagent bien dans la couche d'Attention et le MLP lors d'un `model.learn(total_timesteps=64)` sur environnement mocké.

---

### SILO RL-5 : Entraînement Parallèle (6 Instances Headless), Mode `--agent rl` & Benchmark
**Objectif :** Connecter `bots/instance_pool.py` à un `SubprocVecEnv` / `DummyVecEnv` sous `MaskablePPO` avec **6 instances parallèles par défaut (`--num-envs 6`)**, créer le wrapper d'inférence `bots/rl/eval.py` et mettre à jour `bots/cli.py`.
* **Entraînement Multi-Instances (`bots/rl/train.py`) :**
  - Lance automatiquement 6 instances Balatro headless (ports `12346` à `12351`) via `instance_pool.py`.
  - Sauvegarde périodique des checkpoints dans `bots/rl/checkpoints/` et du meilleur modèle (`best_model.zip`) selon le winrate et l'Ante moyen glissant sur les 50 dernières runs.
  - Enregistre toutes les runs d'entraînement et d'évaluation dans la télémétrie CSV pour suivre la progression en direct.
* **Intégration CLI (`bots/cli.py`) :**
  - Ajoute `--agent [heuristic|rl]` (défaut : `heuristic`) et `--model-path <path>` (défaut : `bots/rl/checkpoints/best_model.zip`).
  - Lorsque `--agent rl` est sélectionné, `runner.py` (ou `eval.py`) utilise le modèle `MaskablePPO` pour toutes les décisions macro et délègue automatiquement le combat et le ciblage micro au solveur déterministe.

* **Livrables :** `bots/rl/eval.py`, mise à jour de `bots/rl/train.py` et `bots/cli.py`, `bots/tests/rl/test_silo_rl5_integration.py`.
* **Validation Hors-Ligne (`pytest`) :**
  1. Vérifier que les 141 tests historiques + tous les nouveaux tests `bots/tests/rl/` passent à 100 %.
  2. Tester le chargement d'un checkpoint `MaskablePPO` sauvegardé et son exécution en mode `--agent rl` sur une partie mockée.
* **Validation En Direct (Live Cluster Test) :**
  1. Commande pour lancer un court test d'entraînement parallèle sur 6 instances headless (`--num-envs 6 --timesteps 2000`).
  2. Commande pour observer 1 partie jouée par le modèle RL (`uv run python -m bots.cli --agent rl --runs 1 --delay 0.5`).