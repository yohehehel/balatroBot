"""Action Space, Action Masking, and Micro-Solver Delegation for Balatro Macro-RL (Silo RL-2).

Implements the flat Discrete(26) action space, computes valid boolean action masks
for each decision state, and delegates low-level tactical execution (Tarot targeting,
Joker left-to-right sorting) to deterministic micro-solvers.
"""

from __future__ import annotations

from enum import IntEnum
import logging
from typing import Any

import gymnasium
import httpx
import numpy as np

from bots.client_wrapper import BalatroClient, RPCError
from bots.pack_consumable_manager import (
    ZERO_TARGET_CONSUMABLES,
    select_targets_for_consumable,
)
from bots.shop_manager import (
    compute_optimal_joker_order,
    evaluate_joker_value,
    get_interest_cap,
    get_protected_interest_floor,
)
from bots.state_parser import Card, GameScreen, ParsedGameState

logger = logging.getLogger(__name__)


# ── Action Space Definition ───────────────────────────────────────────────────

class ActionID(IntEnum):
    # ── Écran BLIND_SELECT (0..1) ──
    PLAY_BLIND = 0
    SKIP_BLIND = 1

    # ── Écran SHOP — Achats & Navigation (2..10) ──
    BUY_SHOP_CARD_0 = 2
    BUY_SHOP_CARD_1 = 3
    BUY_SHOP_CARD_2 = 4
    BUY_SHOP_CARD_3 = 5
    BUY_VOUCHER = 6
    BUY_BOOSTER_0 = 7
    BUY_BOOSTER_1 = 8
    REROLL_SHOP = 9
    LEAVE_SHOP = 10

    # ── Écran SMODS_BOOSTER_OPENED / Pack Ouvert (11..16) ──
    SELECT_PACK_CARD_0 = 11
    SELECT_PACK_CARD_1 = 12
    SELECT_PACK_CARD_2 = 13
    SELECT_PACK_CARD_3 = 14
    SELECT_PACK_CARD_4 = 15
    SKIP_PACK = 16

    # ── Gestion des Consommables en Inventaire (17..20) ──
    USE_OR_SELL_CONSUMABLE_0 = 17
    USE_OR_SELL_CONSUMABLE_1 = 18
    USE_OR_SELL_CONSUMABLE_2 = 19
    USE_OR_SELL_CONSUMABLE_3 = 20

    # ── Vente de Jokers en Boutique (21..25) ──
    SELL_JOKER_0 = 21
    SELL_JOKER_1 = 22
    SELL_JOKER_2 = 23
    SELL_JOKER_3 = 24
    SELL_JOKER_4 = 25


NUM_ACTIONS: int = 26


def get_action_space() -> gymnasium.spaces.Discrete:
    """Return the Gymnasium Discrete(26) action space."""
    return gymnasium.spaces.Discrete(NUM_ACTIONS)


# ── Helpers for Vendable Jokers & Purchasing ──────────────────────────────────

def _get_max_spendable_money(state: ParsedGameState) -> int:
    """Return spendable money taking Credit Card debt allowance into account."""
    has_credit_card = any(j.key == "j_credit_card" for j in (state.jokers.cards if state.jokers else []))
    return state.money + (20 if has_credit_card else 0)


def _get_vendable_jokers(state: ParsedGameState) -> list[Card]:
    """Return up to 5 non-eternal Jokers exposed for sale, sorted ascending by value."""
    if not state.jokers or not state.jokers.cards:
        return []

    vendable = [j for j in state.jokers.cards if not j.modifier.eternal]
    if len(vendable) > 5:
        # Expose in priority the 5 weakest Jokers
        vendable.sort(key=lambda j: (evaluate_joker_value(j, state), j.cost.sell))
        return vendable[:5]
    return vendable


def _has_strategic_reason_to_sell(joker: Card, state: ParsedGameState) -> bool:
    """Anti-suicide guardrail: ensures a plausible strategic reason exists before selling a Joker."""
    # 1. Joker slots are full or overflowing (need room for upgrades/purchases)
    max_slots = state.jokers.limit or 5
    if len(state.jokers.cards) >= max_slots:
        return True

    # 2. Debuffed perishable joker (useless dead weight)
    if joker.modifier.perishable == 0 or joker.state.debuff:
        return True

    # 3. Rental joker draining economy each round
    if joker.modifier.rental:
        return True

    # 4. Consumable/Utility Jokers specifically designed to be sold
    key = joker.key
    if key in ("j_luchador", "j_diet_cola", "j_invisible"):
        return True
    if key == "j_egg" and joker.cost.sell >= 12:
        return True

    # 5. Emergency liquidity: selling allows buying an unaffordable shop item
    spendable = _get_max_spendable_money(state)
    needed_for_item = False

    # Check shop cards
    if state.shop and state.shop.cards:
        for c in state.shop.cards:
            if spendable < c.cost.buy <= spendable + joker.cost.sell:
                needed_for_item = True
                break

    # Check vouchers
    if not needed_for_item and state.vouchers and state.vouchers.cards:
        v = state.vouchers.cards[0]
        if spendable < v.cost.buy <= spendable + joker.cost.sell:
            needed_for_item = True

    # Check packs
    if not needed_for_item and state.packs and state.packs.cards:
        for p in state.packs.cards:
            if spendable < p.cost.buy <= spendable + joker.cost.sell:
                needed_for_item = True
                break

    return needed_for_item


def _can_use_fool(state: ParsedGameState) -> bool:
    """Check if The Fool is valid to use per Balatro rules.

    The Fool copies the last Tarot or Planet card used this run, excluding The Fool.
    It cannot be used if no Tarot/Planet was used yet, or if the last card was The Fool.
    """
    last_tp = getattr(state, "last_tarot_planet", None) or (state.raw.get("last_tarot_planet") if state.raw else None)
    if last_tp:
        clean_tp = str(last_tp).strip().lower()
        if clean_tp in ("c_fool", "the fool", "none", ""):
            return False
        return True

    usage = state.consumeable_usage_total if isinstance(state.consumeable_usage_total, dict) else {}
    tarot_planet_cnt = usage.get("tarot", 0) + usage.get("planet", 0)
    return tarot_planet_cnt > 0


# ── Action Mask Computation ───────────────────────────────────────────────────

def compute_action_mask(state: ParsedGameState) -> np.ndarray:
    """Compute the boolean action mask of shape (26,) for the current game state.

    Guarantees:
    - dtype is bool, shape is (26,)
    - Invariant: mask.any() is ALWAYS True (at least one valid action exists).
    """
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    screen = state.state

    # ── 1. Écran BLIND_SELECT ─────────────────────────────────────────────
    if screen == GameScreen.BLIND_SELECT:
        # PLAY_BLIND is always valid
        mask[ActionID.PLAY_BLIND] = True

        # SKIP_BLIND is only valid on Small Blind and Big Blind, NEVER on Boss Blind
        b = state.active_blind
        if not b:
            # Check blinds dictionary
            for name in ("small", "big"):
                cand = state.blinds.get(name)
                if cand and cand.status in ("SELECT", "CURRENT"):
                    b = cand
                    break

        blind_type = (b.type if b else "").upper()
        if blind_type in ("SMALL", "BIG"):
            mask[ActionID.SKIP_BLIND] = True

    # ── 2. Écran SHOP — Achats & Navigation ────────────────────────────────
    elif screen == GameScreen.SHOP:
        # LEAVE_SHOP is always valid
        mask[ActionID.LEAVE_SHOP] = True

        spendable = _get_max_spendable_money(state)
        jokers = state.jokers.cards if state.jokers else []
        max_joker_slots = state.jokers.limit or 5
        consumables = state.consumables.cards if state.consumables else []
        max_consumable_slots = state.consumables.limit or 2

        # Shop Cards (BUY_SHOP_CARD_0..3)
        shop_cards = state.shop.cards if state.shop else []
        for i in range(min(4, len(shop_cards))):
            c = shop_cards[i]
            cost = c.cost.buy if c.cost.buy > 0 else 0
            if spendable < cost:
                continue

            card_set = (c.set or "").upper()
            is_joker = card_set == "JOKER" or c.key.startswith("j_")
            is_consumable = card_set in ("TAROT", "PLANET", "SPECTRAL") or c.key.startswith("c_")
            is_negative = (c.modifier.edition or "").upper() in ("NEGATIVE", "E_NEGATIVE")

            if is_joker:
                if len(jokers) < max_joker_slots or is_negative:
                    mask[ActionID.BUY_SHOP_CARD_0 + i] = True
            elif is_consumable:
                if len(consumables) < max_consumable_slots:
                    mask[ActionID.BUY_SHOP_CARD_0 + i] = True
            else:
                # Standard playing card: always room in deck
                mask[ActionID.BUY_SHOP_CARD_0 + i] = True

        # Buy Voucher
        vouchers = state.vouchers.cards if state.vouchers else []
        if vouchers:
            v_cost = vouchers[0].cost.buy if vouchers[0].cost.buy > 0 else 10
            if spendable >= v_cost:
                mask[ActionID.BUY_VOUCHER] = True

        # Buy Boosters (BUY_BOOSTER_0..1)
        packs = state.packs.cards if state.packs else []
        for i in range(min(2, len(packs))):
            p_cost = packs[i].cost.buy if packs[i].cost.buy > 0 else 4
            if spendable >= p_cost:
                mask[ActionID.BUY_BOOSTER_0 + i] = True

        # Reroll Shop
        reroll_cost = state.round.reroll_cost if state.round else 5
        if spendable >= reroll_cost:
            mask[ActionID.REROLL_SHOP] = True

        # Sell Jokers (SELL_JOKER_0..4)
        vendable_jokers = _get_vendable_jokers(state)
        for i, j in enumerate(vendable_jokers):
            if _has_strategic_reason_to_sell(j, state):
                mask[ActionID.SELL_JOKER_0 + i] = True

        # Held Consumables in SHOP (USE_OR_SELL_CONSUMABLE_0..3)
        for i in range(min(4, len(consumables))):
            c = consumables[i]
            key = c.key.lower()

            if key in ZERO_TARGET_CONSUMABLES:
                # Safe immediate usage without target cards
                if key.startswith("c_mercury") or key in (
                    "c_venus", "c_earth", "c_mars", "c_jupiter", "c_saturn",
                    "c_uranus", "c_neptune", "c_pluto", "c_planet_x", "c_ceres", "c_eris",
                    "c_black_hole", "c_hermit", "c_temperance",
                ):
                    mask[ActionID.USE_OR_SELL_CONSUMABLE_0 + i] = True
                elif key == "c_judgement" and len(jokers) < max_joker_slots:
                    mask[ActionID.USE_OR_SELL_CONSUMABLE_0 + i] = True
                elif key == "c_wheel_of_fortune" and any(not j.modifier.edition for j in jokers):
                    mask[ActionID.USE_OR_SELL_CONSUMABLE_0 + i] = True
                elif key in ("c_emperor", "c_high_priestess") and len(consumables) <= max_consumable_slots:
                    mask[ActionID.USE_OR_SELL_CONSUMABLE_0 + i] = True
                elif key == "c_fool":
                    if _can_use_fool(state):
                        mask[ActionID.USE_OR_SELL_CONSUMABLE_0 + i] = True
                    elif len(consumables) >= max_consumable_slots:
                        mask[ActionID.USE_OR_SELL_CONSUMABLE_0 + i] = True
                elif key == "c_ankh" and len(jokers) == 1:
                    mask[ActionID.USE_OR_SELL_CONSUMABLE_0 + i] = True
                elif key == "c_hex" and len(jokers) == 1 and any(not j.modifier.edition for j in jokers):
                    mask[ActionID.USE_OR_SELL_CONSUMABLE_0 + i] = True
                elif key == "c_ectoplasm" and len(jokers) >= 1 and (state.hand.limit or 8) > 6:
                    mask[ActionID.USE_OR_SELL_CONSUMABLE_0 + i] = True
                elif key == "c_wraith" and spendable <= 4 and len(jokers) < max_joker_slots:
                    mask[ActionID.USE_OR_SELL_CONSUMABLE_0 + i] = True
                elif key == "c_soul" and len(jokers) < max_joker_slots:
                    mask[ActionID.USE_OR_SELL_CONSUMABLE_0 + i] = True
                else:
                    # If slots are full, allow selling blocked consumable
                    if len(consumables) >= max_consumable_slots:
                        mask[ActionID.USE_OR_SELL_CONSUMABLE_0 + i] = True

            else:
                # Consumable requires target cards in hand:
                # In SHOP, hand is empty: preserve card for next round, UNLESS slots are full
                if len(consumables) >= max_consumable_slots:
                    # Sell to free slot
                    mask[ActionID.USE_OR_SELL_CONSUMABLE_0 + i] = True

    # ── 3. Écran SMODS_BOOSTER_OPENED / Pack Ouvert ───────────────────────
    elif screen in GameScreen.BOOSTER_STATES:
        # SKIP_PACK is always valid
        mask[ActionID.SKIP_PACK] = True

        pack_cards = state.pack.cards if state.pack else []
        jokers = state.jokers.cards if state.jokers else []
        max_joker_slots = state.jokers.limit or 5
        consumables = state.consumables.cards if state.consumables else []
        max_consumable_slots = state.consumables.limit or 2
        hand_cards = state.hand.cards if state.hand else []

        for i in range(min(5, len(pack_cards))):
            c = pack_cards[i]
            key = c.key.lower()
            card_set = (c.set or "").upper()
            is_joker = card_set == "JOKER" or key.startswith("j_")
            is_negative = (c.modifier.edition or "").upper() in ("NEGATIVE", "E_NEGATIVE")

            if is_joker:
                # Buffoon Pack: requires free joker slot (or negative joker)
                if len(jokers) < max_joker_slots or is_negative:
                    mask[ActionID.SELECT_PACK_CARD_0 + i] = True
            elif key in ZERO_TARGET_CONSUMABLES:
                # 0-target consumable used immediately
                editionless_jokers = [j for j in jokers if not j.modifier.edition]
                if key == "c_hex" and (len(editionless_jokers) == 0 or len([j for j in jokers if not j.modifier.eternal]) >= 2):
                    # Hex strictly requires at least 1 editionless Joker, and avoid suicidal wiping of multi-joker build
                    continue
                if key == "c_ankh" and (len(jokers) == 0 or len([j for j in jokers if not j.modifier.eternal]) >= 2):
                    # Ankh strictly requires at least 1 Joker, and avoid suicidal wiping of multi-joker build
                    continue
                if key == "c_ectoplasm" and (len([j for j in jokers if (j.modifier.edition or "").upper() not in ("NEGATIVE", "E_NEGATIVE")]) == 0 or (state.hand.limit or 8) <= 6):
                    # Ectoplasm strictly requires at least 1 non-negative Joker, and avoid catastrophic hand size reduction
                    continue
                if key == "c_wraith" and (len(jokers) >= max_joker_slots or state.money >= 15):
                    # Requires free joker slot, and avoid discarding large bankroll
                    continue
                if key == "c_judgement" and len(jokers) >= max_joker_slots:
                    continue
                if key == "c_fool" and (not _can_use_fool(state) or len(consumables) >= max_consumable_slots):
                    continue
                mask[ActionID.SELECT_PACK_CARD_0 + i] = True
            else:
                # Consumable with hand targeting or standard card
                if hand_cards:
                    targets = select_targets_for_consumable(c, hand_cards, state)
                    if targets:
                        mask[ActionID.SELECT_PACK_CARD_0 + i] = True
                elif card_set in ("DEFAULT", "ENHANCED") or (c.value.rank and c.value.suit):
                    # Standard playing card added to deck
                    mask[ActionID.SELECT_PACK_CARD_0 + i] = True

    # ── 4. Écran SELECTING_HAND (Combat) ──────────────────────────────────
    elif screen == GameScreen.SELECTING_HAND:
        # Micro-solver will play the combat round, but RL can trigger pre-combat consumables
        consumables = state.consumables.cards if state.consumables else []
        hand_cards = state.hand.cards if state.hand else []
        jokers = state.jokers.cards if state.jokers else []
        max_joker_slots = state.jokers.limit or 5

        for i in range(min(4, len(consumables))):
            c = consumables[i]
            ckey = c.key.lower()
            if ckey in ZERO_TARGET_CONSUMABLES:
                if ckey in ("c_ankh", "c_hex", "c_ectoplasm") and len(jokers) == 0:
                    continue
                if ckey in ("c_judgement", "c_wraith") and len(jokers) >= max_joker_slots:
                    continue
                mask[ActionID.USE_OR_SELL_CONSUMABLE_0 + i] = True
            elif hand_cards and select_targets_for_consumable(c, hand_cards, state):
                mask[ActionID.USE_OR_SELL_CONSUMABLE_0 + i] = True

        # Safe fallback to allow advancing / playing
        mask[ActionID.PLAY_BLIND] = True

    # ── 5. Autres écrans (MENU, ROUND_EVAL, GAME_OVER) ────────────────────
    else:
        mask[ActionID.PLAY_BLIND] = True

    # ── Invariant critique : Au moins une action valide garantie ──────────
    if not mask.any():
        if screen == GameScreen.SHOP:
            mask[ActionID.LEAVE_SHOP] = True
        elif screen in GameScreen.BOOSTER_STATES:
            mask[ActionID.SKIP_PACK] = True
        else:
            mask[ActionID.PLAY_BLIND] = True

    return mask


# ── Execution and Micro-Solver Delegation ─────────────────────────────────────

def _auto_rearrange_jokers(state: ParsedGameState, client: BalatroClient) -> bool:
    """Check left-to-right sorting of Jokers and trigger rearrange RPC if needed."""
    jokers = state.jokers.cards if state.jokers else []
    if len(jokers) <= 1:
        return False

    optimal_order = compute_optimal_joker_order(jokers, state.ante_num)
    identity_order = list(range(len(jokers)))
    if optimal_order != identity_order:
        try:
            client.call("rearrange", {"jokers": optimal_order})
            return True
        except Exception:
            return False
    return False


def execute_macro_action(
    action: int,
    state: ParsedGameState,
    client: BalatroClient,
) -> dict[str, Any]:
    """Execute the Macro-RL action on the Balatro JSON-RPC client.

    Automatically delegates tactical micro-targeting (Tarots/Spectrals in packs
    or inventory) and left-to-right Joker reordering.
    Returns a dictionary describing the execution.
    """
    action_enum = ActionID(action)
    action_name = action_enum.name
    rpc_method = ""
    rpc_params: dict[str, Any] = {}
    targets: list[int] = []
    auto_rearranged = False
    transaction: dict[str, str] | None = None

    try:
        # 1. Écran BLIND_SELECT
        if action == ActionID.PLAY_BLIND:
            if state.state == GameScreen.MENU:
                rpc_method = "start"
                rpc_params = {"deck": "RED", "stake": "WHITE"}
                client.call("start", rpc_params)
            elif state.state in GameScreen.BOOSTER_STATES:
                # Fallback if booster is open
                rpc_method = "pack"
                rpc_params = {"skip": True}
                client.call("pack", rpc_params)
            elif state.state == GameScreen.SHOP:
                # Fallback if shop is open
                rpc_method = "next_round"
                client.call("next_round")
            else:
                rpc_method = "select"
                client.call("select")

        elif action == ActionID.SKIP_BLIND:
            if state.state in GameScreen.BOOSTER_STATES:
                rpc_method = "pack"
                rpc_params = {"skip": True}
                client.call("pack", rpc_params)
            elif state.state == GameScreen.SHOP:
                rpc_method = "next_round"
                client.call("next_round")
            else:
                rpc_method = "skip"
                client.call("skip")

        # 2. Achats en Boutique
        elif ActionID.BUY_SHOP_CARD_0 <= action <= ActionID.BUY_SHOP_CARD_3:
            idx = action - ActionID.BUY_SHOP_CARD_0
            if state.state == GameScreen.SHOP:
                card = state.shop.cards[idx] if (state.shop and idx < len(state.shop.cards)) else None
                if card:
                    set_name = (card.set or "").lower()
                    key = card.key or ""
                    if "joker" in set_name or key.startswith("j_"):
                        cat = "joker"
                    elif "tarot" in set_name or key.startswith("c_"):
                        cat = "tarot"
                    elif "planet" in set_name:
                        cat = "planet"
                    elif "spectral" in set_name:
                        cat = "spectral"
                    else:
                        cat = "playing_card"
                    transaction = {
                        "action": "BUY",
                        "category": cat,
                        "item_key": key,
                        "item_name": getattr(card, "label", None) or key,
                    }
                rpc_method = "buy"
                rpc_params = {"card": idx}
                client.call("buy", rpc_params)
                auto_rearranged = _auto_rearrange_jokers(state, client)
            else:
                rpc_method = "noop"

        elif action == ActionID.BUY_VOUCHER:
            if state.state == GameScreen.SHOP:
                voucher = state.vouchers.cards[0] if (state.vouchers and state.vouchers.cards) else None
                if voucher:
                    transaction = {
                        "action": "BUY",
                        "category": "voucher",
                        "item_key": voucher.key or "",
                        "item_name": getattr(voucher, "label", None) or voucher.key or "Voucher",
                    }
                rpc_method = "buy"
                rpc_params = {"voucher": 0}
                client.call("buy", rpc_params)
            else:
                rpc_method = "noop"

        elif action in (ActionID.BUY_BOOSTER_0, ActionID.BUY_BOOSTER_1):
            idx = action - ActionID.BUY_BOOSTER_0
            if state.state == GameScreen.SHOP:
                pack = state.packs.cards[idx] if (state.packs and idx < len(state.packs.cards)) else None
                if pack:
                    transaction = {
                        "action": "BUY",
                        "category": "pack",
                        "item_key": pack.key or "",
                        "item_name": getattr(pack, "label", None) or pack.key or "Booster Pack",
                    }
                rpc_method = "buy"
                rpc_params = {"pack": idx}
                client.call("buy", rpc_params)
            else:
                rpc_method = "noop"

        elif action == ActionID.REROLL_SHOP:
            if state.state == GameScreen.SHOP:
                rpc_method = "reroll"
                client.call("reroll")
            else:
                rpc_method = "noop"

        elif action == ActionID.LEAVE_SHOP:
            if state.state == GameScreen.SHOP:
                auto_rearranged = _auto_rearrange_jokers(state, client)
                rpc_method = "next_round"
                client.call("next_round")
            else:
                rpc_method = "noop"

        # 3. Choix dans un Booster Ouvert
        elif ActionID.SELECT_PACK_CARD_0 <= action <= ActionID.SELECT_PACK_CARD_4:
            idx = action - ActionID.SELECT_PACK_CARD_0
            if state.state in GameScreen.BOOSTER_STATES:
                pack_cards = state.pack.cards if state.pack else []
                selected_card = pack_cards[idx] if idx < len(pack_cards) else None
                if selected_card:
                    key = selected_card.key or ""
                    set_name = (selected_card.set or "").lower()
                    if "joker" in set_name or key.startswith("j_"):
                        cat = "joker"
                    elif "tarot" in set_name:
                        cat = "tarot"
                    elif "planet" in set_name:
                        cat = "planet"
                    elif "spectral" in set_name:
                        cat = "spectral"
                    else:
                        cat = "playing_card"
                    transaction = {
                        "action": "BUY",
                        "category": cat,
                        "item_key": key,
                        "item_name": getattr(selected_card, "label", None) or key,
                    }

                if selected_card and selected_card.key.lower() == "c_fool" and not _can_use_fool(state):
                    rpc_method = "pack"
                    rpc_params = {"skip": True}
                    client.call("pack", rpc_params)
                else:
                    rpc_method = "pack"
                    if selected_card and selected_card.key.lower() not in ZERO_TARGET_CONSUMABLES and state.hand.cards:
                        targets = select_targets_for_consumable(selected_card, state.hand.cards, state)

                    if targets:
                        rpc_params = {"card": idx, "targets": targets}
                    else:
                        rpc_params = {"card": idx}

                    client.call("pack", rpc_params)
                if selected_card and (selected_card.set == "Joker" or selected_card.key.startswith("j_")):
                    auto_rearranged = _auto_rearrange_jokers(state, client)
            else:
                rpc_method = "noop"

        elif action == ActionID.SKIP_PACK:
            if state.state in GameScreen.BOOSTER_STATES:
                if not state.pack or not state.pack.cards:
                    time.sleep(0.1)
                rpc_method = "pack"
                rpc_params = {"skip": True}
                client.call("pack", rpc_params)
            else:
                rpc_method = "noop"

        # 4. Consommables en Inventaire
        elif ActionID.USE_OR_SELL_CONSUMABLE_0 <= action <= ActionID.USE_OR_SELL_CONSUMABLE_3:
            idx = action - ActionID.USE_OR_SELL_CONSUMABLE_0
            consumables = state.consumables.cards if state.consumables else []
            consumable = consumables[idx] if idx < len(consumables) else None

            max_slots = state.consumables.limit or 2
            is_unusable_fool = (consumable and consumable.key.lower() == "c_fool" and not _can_use_fool(state))
            # Sell if in SHOP with slots full and either (unusable Fool) or (requires hand cards when hand is empty)
            should_sell = (
                state.state == GameScreen.SHOP
                and len(consumables) >= max_slots
                and (
                    (consumable and consumable.key.lower() not in ZERO_TARGET_CONSUMABLES and not state.hand.cards)
                    or is_unusable_fool
                )
            )

            if should_sell:
                rpc_method = "sell"
                rpc_params = {"consumable": idx}
                if consumable:
                    transaction = {
                        "action": "SELL",
                        "category": "consumable",
                        "item_key": consumable.key or "",
                        "item_name": getattr(consumable, "label", None) or consumable.key or "Consumable",
                    }
                client.call("sell", rpc_params)
            elif is_unusable_fool:
                # Unusable Fool with room in inventory: do not attempt invalid use
                rpc_method = "noop"
            else:
                # Use consumable
                rpc_method = "use"
                if consumable:
                    key = consumable.key or ""
                    set_name = (consumable.set or "").lower()
                    cat = "planet" if "planet" in set_name else ("spectral" if "spectral" in set_name else "tarot")
                    transaction = {
                        "action": "USE",
                        "category": cat,
                        "item_key": key,
                        "item_name": getattr(consumable, "label", None) or key,
                    }

                if consumable and consumable.key.lower() not in ZERO_TARGET_CONSUMABLES and state.hand.cards:
                    targets = select_targets_for_consumable(consumable, state.hand.cards, state)

                if targets:
                    rpc_params = {"consumable": idx, "cards": targets}
                else:
                    rpc_params = {"consumable": idx}

                client.call("use", rpc_params)
                auto_rearranged = _auto_rearrange_jokers(state, client)

        # 5. Vente de Jokers
        elif ActionID.SELL_JOKER_0 <= action <= ActionID.SELL_JOKER_4:
            idx = action - ActionID.SELL_JOKER_0
            vendable_jokers = _get_vendable_jokers(state)
            target_joker = vendable_jokers[idx] if idx < len(vendable_jokers) else None

            if target_joker and state.jokers:
                real_idx = state.jokers.cards.index(target_joker)
                transaction = {
                    "action": "SELL",
                    "category": "joker",
                    "item_key": target_joker.key or "",
                    "item_name": getattr(target_joker, "label", None) or target_joker.key or "Joker",
                }
                rpc_method = "sell"
                rpc_params = {"joker": real_idx}
                client.call("sell", rpc_params)
                auto_rearranged = _auto_rearrange_jokers(state, client)

        return {
            "action_id": action,
            "action_name": action_name,
            "rpc_method": rpc_method,
            "rpc_params": rpc_params,
            "targets": targets,
            "auto_rearranged": auto_rearranged,
            "transaction": transaction,
            "success": True,
        }

    except Exception as exc:
        is_transport = isinstance(exc, (httpx.HTTPError, ConnectionError, OSError))
        if is_transport:
            logger.error(
                f"Transport failure executing action {action_name} ({rpc_method}): {exc}"
            )
        else:
            logger.warning(
                f"RPC execution error for action {action_name} ({rpc_method}) on screen {state.state}: {exc}"
            )
        return {
            "action_id": action,
            "action_name": action_name,
            "rpc_method": rpc_method,
            "rpc_params": rpc_params,
            "targets": targets,
            "auto_rearranged": auto_rearranged,
            "success": False,
            "is_transport_error": is_transport,
            "error": str(exc),
        }

