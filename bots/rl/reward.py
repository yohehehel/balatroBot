"""Macro Reward Shaping for Balatro Macro-RL (Silo RL-3).

Guides the RL agent towards survival, archetype building, shop discipline,
and Ante 8 victory while heavily penalizing game over (with a 'rich death' penalty).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from bots.pack_consumable_manager import PLANET_TO_HAND
from bots.rl.actions import ActionID
from bots.rl.features import compute_estimated_power_ratio
from bots.shop_manager import JokerRole, get_joker_role
from bots.state_parser import BlindType, Card, GameScreen, ParsedGameState


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
        "emperor_reward": 0.0,
        "hermit_reward": 0.0,
        "planet_archetype_reward": 0.0,
        "celestial_pack_penalty": 0.0,
        "debt_penalty": 0.0,
        "hoarding_penalty": 0.0,
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

    # 6. The Emperor usage shaping
    if ActionID.USE_OR_SELL_CONSUMABLE_0 <= action <= ActionID.USE_OR_SELL_CONSUMABLE_3:
        idx = action - ActionID.USE_OR_SELL_CONSUMABLE_0
        if prev_state and prev_state.consumables and idx < len(prev_state.consumables.cards):
            c = prev_state.consumables.cards[idx]
            if c.key.lower() == "c_emperor":
                max_c = prev_state.consumables.limit or 2
                c_held = len(prev_state.consumables.cards)
                effective_free_slots = max_c - (c_held - 1)
                if effective_free_slots >= 2:
                    breakdown["emperor_reward"] = 0.05
                elif effective_free_slots == 1:
                    breakdown["emperor_reward"] = -0.20
                else:
                    breakdown["emperor_reward"] = -0.45
    elif ActionID.SELECT_PACK_CARD_0 <= action <= ActionID.SELECT_PACK_CARD_4:
        idx = action - ActionID.SELECT_PACK_CARD_0
        if prev_state and prev_state.pack and idx < len(prev_state.pack.cards):
            c = prev_state.pack.cards[idx]
            if c.key.lower() == "c_emperor":
                max_c = (prev_state.consumables.limit if prev_state.consumables else 2) or 2
                c_held = len(prev_state.consumables.cards) if (prev_state and prev_state.consumables) else 0
                effective_free_slots = max_c - c_held
                if effective_free_slots >= 2:
                    breakdown["emperor_reward"] = 0.05
                elif effective_free_slots == 1:
                    breakdown["emperor_reward"] = -0.20
                else:
                    breakdown["emperor_reward"] = -0.45

    # 7. The Hermit usage shaping
    is_hermit_used = False
    if ActionID.USE_OR_SELL_CONSUMABLE_0 <= action <= ActionID.USE_OR_SELL_CONSUMABLE_3:
        idx = action - ActionID.USE_OR_SELL_CONSUMABLE_0
        if prev_state and prev_state.consumables and idx < len(prev_state.consumables.cards):
            if prev_state.consumables.cards[idx].key.lower() == "c_hermit":
                is_hermit_used = True
    elif ActionID.SELECT_PACK_CARD_0 <= action <= ActionID.SELECT_PACK_CARD_4:
        idx = action - ActionID.SELECT_PACK_CARD_0
        if prev_state and prev_state.pack and idx < len(prev_state.pack.cards):
            if prev_state.pack.cards[idx].key.lower() == "c_hermit":
                is_hermit_used = True

    if is_hermit_used and prev_state:
        money_before = float(prev_state.money)
        if money_before >= 20.0:
            breakdown["hermit_reward"] = 0.15
        else:
            penalty = -0.35 * ((20.0 - max(0.0, money_before)) / 20.0) ** 2
            breakdown["hermit_reward"] = float(penalty)

    # 8. Planet Card archetype coherence & Celestial Pack checks
    selected_card: Card | None = None
    if ActionID.BUY_SHOP_CARD_0 <= action <= ActionID.BUY_SHOP_CARD_3:
        idx = action - ActionID.BUY_SHOP_CARD_0
        if prev_state and prev_state.shop and idx < len(prev_state.shop.cards):
            selected_card = prev_state.shop.cards[idx]
    elif ActionID.SELECT_PACK_CARD_0 <= action <= ActionID.SELECT_PACK_CARD_4:
        idx = action - ActionID.SELECT_PACK_CARD_0
        if prev_state and prev_state.pack and idx < len(prev_state.pack.cards):
            selected_card = prev_state.pack.cards[idx]

    if selected_card and prev_state:
        card_k = selected_card.key.lower()
        card_s = (selected_card.set or "").lower()
        is_planet = card_k in PLANET_TO_HAND or card_s == "planet" or card_k == "c_black_hole"

        if is_planet and card_k != "c_black_hole":
            jokers = prev_state.jokers.cards if prev_state.jokers else []
            has_synergy = any(
                j.key in ("j_constellation", "j_astronomer") and not j.state.debuff
                for j in jokers
            )
            if not has_synergy:
                target_hand = PLANET_TO_HAND.get(card_k)
                if target_hand:
                    hands = prev_state.hands if prev_state.hands else {}
                    total_played = sum(h.played for h in hands.values())
                    played_sorted = sorted(
                        [(name, h.played) for name, h in hands.items() if h.played > 0],
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    top_2_hands = {name for name, _ in played_sorted[:2]}

                    if ante == 1:
                        if target_hand in top_2_hands or target_hand in ("Pair", "Two Pair", "Flush", "Straight"):
                            breakdown["planet_archetype_reward"] = 0.06
                    else:
                        if target_hand in top_2_hands:
                            breakdown["planet_archetype_reward"] = 0.06
                        else:
                            hand_played = hands[target_hand].played if target_hand in hands else 0
                            hand_pct = (hand_played / total_played) if total_played > 0 else 0.0
                            if hand_pct < 0.05:
                                breakdown["planet_archetype_reward"] = -0.18

    # Celestial Pack purchase check in Ante 1-2
    if ActionID.BUY_BOOSTER_0 <= action <= ActionID.BUY_BOOSTER_1:
        idx = action - ActionID.BUY_BOOSTER_0
        if prev_state and prev_state.packs and idx < len(prev_state.packs.cards):
            pack = prev_state.packs.cards[idx]
            pkey = (pack.key or "").lower()
            plabel = (getattr(pack, "label", None) or "").lower()
            if ("celestial" in pkey or "celestial" in plabel) and ante in (1, 2):
                scoring_roles = {JokerRole.CHIPS, JokerRole.ADD_MULT, JokerRole.X_MULT, JokerRole.RETRIGGER_COPY}
                jokers = prev_state.jokers.cards if prev_state.jokers else []
                scoring_jokers = [
                    j for j in jokers
                    if not j.state.debuff and get_joker_role(j.key) in scoring_roles
                ]
                if len(scoring_jokers) < 2:
                    breakdown["celestial_pack_penalty"] = -0.15

    # 9. Credit Card debt penalty when leaving shop
    if action == ActionID.LEAVE_SHOP:
        money = next_state.money if next_state else (prev_state.money if prev_state else 0)
        if money < 0:
            tranches = math.ceil(abs(float(money)) / 5.0)
            breakdown["debt_penalty"] = -0.10 * float(tranches)

    # 10. Passive Hoarding Penalty (outside Bull / Bootstraps) when leaving shop in Ante 3+
    if action == ActionID.LEAVE_SHOP and ante >= 3:
        jokers = (prev_state.jokers.cards if prev_state and prev_state.jokers else []) or (next_state.jokers.cards if next_state and next_state.jokers else [])
        has_bull_or_bootstraps = any(j.key in ("j_bull", "j_bootstraps") and not j.state.debuff for j in jokers)
        money = prev_state.money if prev_state else next_state.money
        if not has_bull_or_bootstraps and money > 45:
            st = prev_state or next_state
            pr = compute_estimated_power_ratio(st) if st else 1.0
            max_slots = (st.jokers.limit if st and st.jokers else 5) or 5
            has_free_slot = len(jokers) < max_slots
            if pr < 0.35 or has_free_slot:
                breakdown["hoarding_penalty"] = -0.15

    # 11. Power ratio shaping (only during active play, not on terminal step)
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

