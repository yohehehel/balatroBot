"""Balatro Macro-RL subpackage.

Neuro-symbolic Macro-RL architecture combining high-level strategic decision-making
with low-level deterministic combat solving and targeting.
"""

from __future__ import annotations

from typing import Any

from bots.rl.actions import (
    ActionID,
    NUM_ACTIONS,
    compute_action_mask,
    execute_macro_action,
    get_action_space,
)
from bots.rl.constants import (
    D_CTX,
    D_DECK,
    D_ENTITY,
    D_HAND,
    MAX_CONSUMABLES,
    MAX_JOKERS,
    MAX_OFFERS,
    MAX_PACK_CHOICES,
    MAX_SHOP_BOOSTERS,
    MAX_SHOP_JOKERS,
)
from bots.rl.env import BalatroMacroEnv
from bots.rl.features import (
    compute_estimated_power_ratio,
    extract_features,
    get_observation_space,
)
from bots.rl.network import BalatroFeaturesExtractor
from bots.rl.reward import calculate_step_reward


def __getattr__(name: str) -> Any:
    if name in ("evaluate_rl_agent", "load_model", "run_rl_episode"):
        from bots.rl import eval as _eval
        return getattr(_eval, name)
    if name in ("create_model", "train"):
        from bots.rl import train as _train
        return getattr(_train, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Constants
    "MAX_JOKERS",
    "MAX_CONSUMABLES",
    "MAX_SHOP_JOKERS",
    "MAX_SHOP_BOOSTERS",
    "MAX_PACK_CHOICES",
    "MAX_OFFERS",
    "D_CTX",
    "D_ENTITY",
    "D_DECK",
    "D_HAND",
    # Features (RL-1)
    "get_observation_space",
    "extract_features",
    "compute_estimated_power_ratio",
    # Actions (RL-2)
    "ActionID",
    "NUM_ACTIONS",
    "get_action_space",
    "compute_action_mask",
    "execute_macro_action",
    # Environment & Reward (RL-3)
    "BalatroMacroEnv",
    "calculate_step_reward",
    # Network & Training (RL-4)
    "BalatroFeaturesExtractor",
    "create_model",
    "train",
    # Inference & Evaluation (RL-5)
    "load_model",
    "run_rl_episode",
    "evaluate_rl_agent",
]
