"""Macro Reward Shaping for Balatro Macro-RL (Silo RL-3).

Guides the RL agent towards survival, archetype building, shop discipline,
and Ante 8 victory while heavily penalizing game over (with a 'rich death' penalty).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from bots.rl.actions import ActionID
from bots.rl.features import compute_estimated_power_ratio
from bots.state_parser import BlindType, GameScreen, ParsedGameState


def calculate_step_reward(
    prev_state: ParsedGameState | None,
    action: int,
    next_state: ParsedGameState,
    round_won: bool | None = None,
    boss_defeated: bool | None = None,
    game_won: bool | None = None,
    game_over: bool | None = None,
    score_obtained: int | None = None,
    score_required: int | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute the macro reward and detailed breakdown for the transition.

    Formula from PLAN_RL_ANTIGRAVITY.md:
    1. Round Victory (Shop reached):
       +0.5 + 0.2 * ante + 0.15 * clip(log10(score_obtained / score_required), 0.0, 2.0)
    2. Boss Defeated:
       +1.0 * ante
    3. Final Victory (Ante 8 Boss defeated):
       +50.0
    4. Game Over (Defeat):
       -10.0 - 0.15 * max(0.0, money - 15.0)  # Rich Death penalty
    5. Shop Discipline & Shaping:
       -0.05 per REROLL_SHOP
       + 0.05 * clip(delta_power_ratio, -1.0, 1.0)

    Returns:
        total_reward (float): The total scalar reward for this transition.
        breakdown (dict[str, float]): Component breakdown for logging/telemetry.
    """
    breakdown: dict[str, float] = {
        "round_base": 0.0,
        "round_ante_bonus": 0.0,
        "high_ante_bonus": 0.0,
        "round_margin_bonus": 0.0,
        "boss_bonus": 0.0,
        "victory_bonus": 0.0,
        "defeat_penalty": 0.0,
        "rich_death_penalty": 0.0,
        "reroll_penalty": 0.0,
        "power_ratio_shaping": 0.0,
    }

    # 1. Deduce flags if not explicitly passed
    is_game_over = game_over if game_over is not None else (next_state.state == GameScreen.GAME_OVER)
    is_game_won = game_won if game_won is not None else (next_state.won or next_state.ante_num > 8)

    ante = max(1, prev_state.ante_num if prev_state else next_state.ante_num)

    # Round victory detection: prev_state was in combat / blind_select and next_state reached shop/round_eval
    if round_won is None:
        if prev_state and prev_state.state in (GameScreen.BLIND_SELECT, GameScreen.SELECTING_HAND):
            is_round_won = next_state.state in (GameScreen.SHOP, GameScreen.ROUND_EVAL, "CASH_OUT")
        else:
            is_round_won = False
    else:
        is_round_won = round_won

    # Boss defeat detection
    if boss_defeated is None:
        if is_round_won and prev_state and prev_state.active_blind:
            is_boss = (prev_state.active_blind.type or "").upper() == BlindType.BOSS
        else:
            is_boss = False
    else:
        is_boss = boss_defeated

    # 2. Defeat (GAME_OVER)
    if is_game_over:
        breakdown["defeat_penalty"] = -10.0
        money = float(next_state.money)
        # Rich Death penalty: punish dying with unspent hoarded money > $15
        if money > 15.0:
            breakdown["rich_death_penalty"] = -0.15 * (money - 15.0)

    # 3. Final Victory
    elif is_game_won:
        breakdown["victory_bonus"] = 50.0

    # 4. Round Won
    if is_round_won and not is_game_over:
        breakdown["round_base"] = 0.5
        breakdown["round_ante_bonus"] = 0.2 * float(ante)

        # Progressive exponential bonus for high antes (Antes 6, 7, 8)
        # Guides policy to prioritize xMult builds to clear the late-game barrier
        if ante >= 6:
            high_ante_map = {6: 2.0, 7: 5.0, 8: 10.0}
            breakdown["high_ante_bonus"] = high_ante_map.get(ante, 10.0)

        req = score_required
        if req is None and prev_state and prev_state.active_blind and prev_state.active_blind.score > 0:
            req = prev_state.active_blind.score

        obtained = score_obtained
        if obtained is None and prev_state and next_state:
            obtained = max(next_state.round.chips, req or 0)

        if req is not None and req > 0 and obtained is not None and obtained > 0:
            margin = math.log10(max(1.0, float(obtained)) / float(req))
            breakdown["round_margin_bonus"] = 0.15 * float(np.clip(margin, 0.0, 2.0))

        if is_boss:
            boss_ante_mult = 1.5 if ante >= 6 else 1.0
            breakdown["boss_bonus"] = boss_ante_mult * float(ante)

    # 5. Shop Discipline: penalize REROLL_SHOP to avoid compulsive rerolls
    if action == ActionID.REROLL_SHOP:
        breakdown["reroll_penalty"] = -0.05

    # 6. Power ratio shaping (only during active play, not on terminal step)
    if prev_state and not is_game_over and not is_game_won:
        try:
            pr_prev = compute_estimated_power_ratio(prev_state)
            pr_next = compute_estimated_power_ratio(next_state)
            delta = float(np.clip(pr_next - pr_prev, -1.0, 1.0))
            breakdown["power_ratio_shaping"] = 0.05 * delta
        except Exception:
            pass

    total_reward = sum(breakdown.values())
    return total_reward, breakdown

