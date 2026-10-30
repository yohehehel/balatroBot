"""Observation space extractor and feature engineering for Balatro Macro-RL (Silo RL-1).

Transforms any ParsedGameState into normalized NumPy float32 tensors strictly matching
the gymnasium.spaces.Dict observation specification from PLAN_RL_ANTIGRAVITY.md.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import gymnasium
import numpy as np

from bots.combat_solver import calculate_hand_score
from bots.pack_consumable_manager import get_dominant_deck_suit
from bots.rl.constants import (
    BOSS_TO_IDX,
    D_CTX,
    D_DECK,
    D_ENTITY,
    D_HAND,
    EDITION_TO_IDX,
    ENHANCEMENT_TO_IDX,
    GLOBAL_ITEM_TO_IDX,
    HAND_TYPE_LIST,
    HAND_TYPE_TO_IDX,
    ITEM_TYPE_TO_IDX,
    JOKER_ROLE_TO_IDX,
    MAX_CONSUMABLES,
    MAX_JOKERS,
    MAX_OFFERS,
    SEAL_TO_IDX,
    SUIT_TO_IDX,
    TAG_TO_IDX,
)
from bots.shop_manager import (
    evaluate_joker_value,
    get_interest_cap,
    get_joker_role,
    get_protected_interest_floor,
    is_survival_mode,
)
from bots.state_parser import Card, CardModifier, CardValue, GameScreen, ParsedGameState


# ── Observation Space Definition ──────────────────────────────────────────────

def get_observation_space() -> gymnasium.spaces.Dict:
    """Return the exact Gymnasium observation space for Balatro Macro-RL."""
    return gymnasium.spaces.Dict({
        "context_features": gymnasium.spaces.Box(
            low=-5.0, high=5.0, shape=(D_CTX,), dtype=np.float32
        ),
        "deck_features": gymnasium.spaces.Box(
            low=0.0, high=2.0, shape=(D_DECK,), dtype=np.float32
        ),
        "hand_levels": gymnasium.spaces.Box(
            low=0.0, high=5.0, shape=(D_HAND,), dtype=np.float32
        ),
        "held_jokers": gymnasium.spaces.Box(
            low=-5.0, high=500.0, shape=(MAX_JOKERS, D_ENTITY), dtype=np.float32
        ),
        "shop_and_pack_offers": gymnasium.spaces.Box(
            low=-5.0, high=500.0, shape=(MAX_OFFERS, D_ENTITY), dtype=np.float32
        ),
    })


# ── Entity Encoding ───────────────────────────────────────────────────────────

def _normalize_edition_str(edition: str | None) -> str:
    if not edition:
        return "NONE"
    clean = edition.strip().upper()
    if clean.startswith("E_"):
        clean = clean[2:]
    return clean if clean in EDITION_TO_IDX else "NONE"


def _normalize_enhancement_str(enhancement: str | None) -> str:
    if not enhancement:
        return "NONE"
    clean = enhancement.strip().upper()
    return clean if clean in ENHANCEMENT_TO_IDX else "NONE"


def _normalize_seal_str(seal: str | None) -> str:
    if not seal:
        return "NONE"
    clean = seal.strip().upper()
    return clean if clean in SEAL_TO_IDX else "NONE"


def _encode_entity(
    card: Card | None,
    is_offer: bool = False,
    override_buy_cost: int | None = None,
) -> np.ndarray:
    """Encode a single Joker, Consumable, Voucher, Booster or Playing Card into D_ENTITY (30 floats)."""
    vec = np.zeros(D_ENTITY, dtype=np.float32)
    if card is None:
        return vec

    # 1. entity_id_index (lookup in global catalog)
    key = card.key or ""
    entity_id = GLOBAL_ITEM_TO_IDX.get(key, 0)
    if entity_id == 0 and card.value.rank and card.value.suit:
        entity_id = GLOBAL_ITEM_TO_IDX.get(f"{card.value.suit}_{card.value.rank}", 0)
    vec[0] = float(entity_id)

    # 2. is_present
    vec[1] = 1.0

    # 3. item_type_one_hot (8 classes: NONE, JOKER, TAROT, PLANET, SPECTRAL, VOUCHER, BOOSTER, PLAYING_CARD)
    card_set = (card.set or "").upper()
    item_type = "NONE"
    if key.startswith("j_") or card_set == "JOKER":
        item_type = "JOKER"
    elif key.startswith("v_") or card_set == "VOUCHER":
        item_type = "VOUCHER"
    elif key.startswith("p_") or card_set == "BOOSTER":
        item_type = "BOOSTER"
    elif key.startswith("c_"):
        if key in ("c_mercury", "c_venus", "c_earth", "c_mars", "c_jupiter",
                   "c_saturn", "c_uranus", "c_neptune", "c_pluto", "c_planet_x",
                   "c_ceres", "c_eris"):
            item_type = "PLANET"
        elif key in ("c_familiar", "c_grim", "c_incantation", "c_talisman", "c_aura",
                     "c_wraith", "c_sigil", "c_ouija", "c_ectoplasm", "c_immolate",
                     "c_ankh", "c_deja_vu", "c_hex", "c_trance", "c_medium",
                     "c_cryptid", "c_soul", "c_black_hole"):
            item_type = "SPECTRAL"
        else:
            item_type = "TAROT"
    elif card.value.suit and card.value.rank or card_set in ("DEFAULT", "ENHANCED"):
        item_type = "PLAYING_CARD"
    else:
        item_type = "JOKER" if card.cost.sell > 0 else "NONE"

    type_idx = ITEM_TYPE_TO_IDX.get(item_type, 0)
    # indices 2..9 (8 classes)
    if 0 <= type_idx < 8:
        vec[2 + type_idx] = 1.0

    # 4. role_one_hot (6 classes: ECONOMY, CHIPS, ADD_MULT, X_MULT, RETRIGGER_COPY, RULE_UTILITY)
    # indices 10..15
    if item_type == "JOKER":
        role = get_joker_role(key)
        role_idx = JOKER_ROLE_TO_IDX.get(role, 0)
        # 1-based in JOKER_ROLE_TO_IDX (0 is NONE)
        if 1 <= role_idx <= 6:
            vec[10 + (role_idx - 1)] = 1.0

    # 5. edition_one_hot (5 classes: NONE, FOIL, HOLO, POLYCHROME, NEGATIVE)
    # indices 16..20
    ed_str = _normalize_edition_str(card.modifier.edition)
    ed_idx = EDITION_TO_IDX.get(ed_str, 0)
    if 0 <= ed_idx < 5:
        vec[16 + ed_idx] = 1.0

    # 6. cost_norm
    if override_buy_cost is not None:
        cost = override_buy_cost
    elif is_offer and card.cost.buy > 0:
        cost = card.cost.buy
    else:
        cost = card.cost.sell
    vec[21] = float(np.clip(cost / 20.0, 0.0, 10.0))

    # 7. sell_value_norm
    vec[22] = float(np.clip(card.cost.sell / 20.0, 0.0, 10.0))

    # 8. dynamic_scaling_norm
    dyn_val = 0.0
    if card.value.x_mult is not None and card.value.x_mult > 1.0:
        dyn_val = float(card.value.x_mult)
    elif card.value.mult is not None and card.value.mult > 0:
        dyn_val = float(card.value.mult) / 20.0
    elif card.value.chips is not None and card.value.chips > 0:
        dyn_val = float(card.value.chips) / 100.0
    elif isinstance(card.value.extra, (int, float)) and card.value.extra > 0:
        dyn_val = float(card.value.extra) / 20.0
    vec[23] = float(np.clip(dyn_val, 0.0, 50.0))

    # 9. synergy_suit_or_hand_id
    synergy_id = 0.0
    if card.value.suit:
        synergy_id = float(SUIT_TO_IDX.get(card.value.suit, 0))
    elif key in ("j_lusty_joker", "j_bloodstone"):
        synergy_id = float(SUIT_TO_IDX.get("H", 0))
    elif key in ("j_greedy_joker", "j_rough_gem"):
        synergy_id = float(SUIT_TO_IDX.get("D", 0))
    elif key in ("j_gluttenous_joker", "j_onyx_agate"):
        synergy_id = float(SUIT_TO_IDX.get("C", 0))
    elif key in ("j_wrathful_joker", "j_arrowhead"):
        synergy_id = float(SUIT_TO_IDX.get("S", 0))
    vec[24] = synergy_id

    # 10. modifier flags (4 floats)
    # indices 25..28
    vec[25] = 1.0 if card.modifier.eternal else 0.0
    vec[26] = 1.0 if card.modifier.rental else 0.0
    vec[27] = float(card.modifier.perishable / 5.0) if card.modifier.perishable is not None else 0.0
    vec[28] = 1.0 if card.state.debuff else 0.0
    # index 29: padding / reserved slot
    vec[29] = 0.0

    return vec


# ── Deck Features Extraction ──────────────────────────────────────────────────

def _extract_deck_features(state: ParsedGameState) -> np.ndarray:
    """Extract normalized histogram and modifier proportions of full deck (shape 32)."""
    vec = np.zeros(D_DECK, dtype=np.float32)

    deck_cards = state.deck_cards.cards
    if not deck_cards and state.hand.cards:
        deck_cards = state.hand.cards

    total = len(deck_cards)
    if total == 0:
        return vec

    rank_order = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    suits = ["H", "D", "C", "S"]
    enhancements = ["BONUS", "MULT", "WILD", "GLASS", "STEEL", "STONE", "GOLD", "LUCKY"]
    editions = ["FOIL", "HOLO", "POLYCHROME"]
    seals = ["RED", "BLUE", "GOLD", "PURPLE"]

    # 13 Ranks: indices 0..12
    for c in deck_cards:
        r = c.value.rank
        if r == "10":
            r = "T"
        if r in rank_order:
            vec[rank_order.index(r)] += 1.0

    # 4 Suits: indices 13..16
    for c in deck_cards:
        s = c.value.suit
        if s in suits:
            vec[13 + suits.index(s)] += 1.0

    # 8 Enhancements: indices 17..24
    for c in deck_cards:
        enh = _normalize_enhancement_str(c.modifier.enhancement)
        if enh in enhancements:
            vec[17 + enhancements.index(enh)] += 1.0

    # 3 Editions: indices 25..27
    for c in deck_cards:
        ed = _normalize_edition_str(c.modifier.edition)
        if ed in editions:
            vec[25 + editions.index(ed)] += 1.0

    # 4 Seals: indices 28..31
    for c in deck_cards:
        seal = _normalize_seal_str(c.modifier.seal)
        if seal in seals:
            vec[28 + seals.index(seal)] += 1.0

    # Normalize by total cards in deck
    vec /= float(total)
    return np.clip(vec, 0.0, 2.0).astype(np.float32)


# ── Hand Levels Extraction ────────────────────────────────────────────────────

def _extract_hand_levels(state: ParsedGameState) -> np.ndarray:
    """Extract level and play frequency for the 12 standard poker hands (shape 24)."""
    vec = np.zeros(D_HAND, dtype=np.float32)
    hands_dict = state.hands or {}

    total_played = sum(h.played for h in hands_dict.values()) if hands_dict else 0

    for i, hand_name in enumerate(HAND_TYPE_LIST[1:13]):
        hinfo = hands_dict.get(hand_name)
        if hinfo:
            vec[2 * i] = float(np.clip(hinfo.level / 20.0, 0.0, 5.0))
            share = (float(hinfo.played) / float(max(1, total_played))) if total_played > 0 else 0.0
            vec[2 * i + 1] = float(np.clip(share, 0.0, 1.0))
        else:
            vec[2 * i] = 1.0 / 20.0
            vec[2 * i + 1] = 0.0

    return vec.astype(np.float32)


# ── Estimated Power Ratio Simulation ──────────────────────────────────────────

_ANTE_DEFAULT_TARGETS: dict[int, int] = {
    1: 300,
    2: 800,
    3: 2000,
    4: 5000,
    5: 11000,
    6: 20000,
    7: 35000,
    8: 50000,
}


def _get_target_chips(state: ParsedGameState) -> int:
    """Return the active or upcoming blind score target."""
    if state.active_blind and state.active_blind.score > 0:
        return state.active_blind.score

    for b in state.blinds.values():
        if b.status in ("SELECT", "UPCOMING") and b.score > 0:
            return b.score

    ante = max(1, state.ante_num)
    if ante in _ANTE_DEFAULT_TARGETS:
        return _ANTE_DEFAULT_TARGETS[ante]
    return int(50000 * (1.5 ** (ante - 8)))


def compute_estimated_power_ratio(state: ParsedGameState) -> float:
    """Estimate build scoring capacity relative to target chips via combat_solver.

    Fast (< 2 ms) and 100% deterministic.
    Returns:
        np.clip(log10(max(1.0, estimated_capacity) / max(1.0, target_chips)), -2.0, 2.0)
    """
    target = float(max(1, _get_target_chips(state)))

    # Determine dominant archetype / poker hand
    dominant_hand = "Pair"
    best_hand_score = -1.0
    if state.hands:
        for name, h in state.hands.items():
            # Score formula favoring hands leveled or heavily played
            sc = float(h.played) * 1000.0 + float(h.level) * 100.0 + float(h.chips)
            if sc > best_hand_score:
                best_hand_score = sc
                dominant_hand = name

    # Determine dominant suit
    dom_suit = get_dominant_deck_suit(state) or "H"

    # Synthesize representative played cards for archetype
    def _c(rank: str, suit: str, enh: str | None = None) -> Card:
        return Card(
            id=999,
            key=f"{suit}_{rank}",
            set="Default",
            value=CardValue(suit=suit, rank=rank),
            modifier=CardModifier(enhancement=enh),
        )

    # Check for enhancements in deck
    has_steel = False
    has_glass = False
    if state.deck_cards and state.deck_cards.cards:
        for dc in state.deck_cards.cards:
            if dc.modifier.enhancement == "STEEL":
                has_steel = True
            elif dc.modifier.enhancement == "GLASS":
                has_glass = True

    played_cards: list[Card] = []
    scoring_cards: list[Card] = []
    held_cards: list[Card] = []

    if dominant_hand == "Flush":
        played_cards = [_c("A", dom_suit), _c("K", dom_suit), _c("Q", dom_suit),
                        _c("J", dom_suit), _c("T", dom_suit)]
        scoring_cards = list(played_cards)
    elif dominant_hand == "Straight":
        played_cards = [_c("A", "H"), _c("K", "D"), _c("Q", "C"), _c("J", "S"), _c("T", "H")]
        scoring_cards = list(played_cards)
    elif dominant_hand == "Full House":
        played_cards = [_c("A", "H"), _c("A", "D"), _c("A", "C"), _c("K", "S"), _c("K", "H")]
        scoring_cards = list(played_cards)
    elif dominant_hand == "Three of a Kind":
        played_cards = [_c("A", "H"), _c("A", "D"), _c("A", "C")]
        scoring_cards = list(played_cards)
    elif dominant_hand == "Two Pair":
        played_cards = [_c("A", "H"), _c("A", "D"), _c("K", "C"), _c("K", "S")]
        scoring_cards = list(played_cards)
    elif dominant_hand == "Four of a Kind":
        played_cards = [_c("A", "H"), _c("A", "D"), _c("A", "C"), _c("A", "S")]
        scoring_cards = list(played_cards)
    elif dominant_hand in ("Five of a Kind", "Flush Five"):
        played_cards = [_c("A", dom_suit) for _ in range(5)]
        scoring_cards = list(played_cards)
    elif dominant_hand == "Flush House":
        played_cards = [_c("A", dom_suit), _c("A", dom_suit), _c("A", dom_suit),
                        _c("K", dom_suit), _c("K", dom_suit)]
        scoring_cards = list(played_cards)
    elif dominant_hand == "Straight Flush":
        played_cards = [_c("A", dom_suit), _c("K", dom_suit), _c("Q", dom_suit),
                        _c("J", dom_suit), _c("T", dom_suit)]
        scoring_cards = list(played_cards)
    elif dominant_hand == "High Card":
        played_cards = [_c("A", "H")]
        scoring_cards = list(played_cards)
    else:  # Pair
        played_cards = [_c("A", "H"), _c("A", "D")]
        scoring_cards = list(played_cards)

    if has_glass and scoring_cards:
        scoring_cards[0].modifier.enhancement = "GLASS"

    # Held cards (fill to typical hand size ~8)
    hand_size = state.hand.limit if state.hand and state.hand.limit > 0 else 8
    held_needed = max(0, hand_size - len(played_cards))
    for i in range(held_needed):
        enh = "STEEL" if (has_steel and i == 0) else None
        held_cards.append(_c("K", "S" if i % 2 == 0 else "C", enh=enh))

    # Evaluate single-hand score
    try:
        _, _, _, predicted_score, _ = calculate_hand_score(
            played_cards=played_cards,
            held_cards=held_cards,
            hand_type=dominant_hand,
            scoring_cards=scoring_cards,
            state=state,
        )
    except Exception:
        predicted_score = 50

    # Scale by hands per round (factoring in Card Sharp x3 on repeat hands)
    hands_capacity = state.round.hands_left if state.round.hands_left > 0 else 4
    jkeys = {j.key for j in (state.jokers.cards if state.jokers else []) if not j.state.debuff and j.key}

    if "j_card_sharp" in jkeys and hands_capacity >= 2 and dominant_hand in (
        "High Card", "Pair", "Two Pair", "Flush", "Straight", "Three of a Kind"
    ):
        # First hand scores 1x, subsequent (hands_capacity - 1) hands score 3x with Card Sharp
        effective_hands = 1.0 + float(hands_capacity - 1) * 3.0
        estimated_capacity = float(max(1, predicted_score)) * effective_hands
    else:
        estimated_capacity = float(max(1, predicted_score)) * float(max(1, hands_capacity))

    ratio = math.log10(max(1.0, estimated_capacity) / target)
    return float(np.clip(ratio, -2.0, 2.0))


# ── Context Features Extraction ───────────────────────────────────────────────

def _extract_context_features(state: ParsedGameState) -> np.ndarray:
    """Extract normalized context vector of shape D_CTX (45 floats)."""
    vec = np.zeros(D_CTX, dtype=np.float32)

    # 1. Progression (indices 0..1)
    vec[0] = float(state.ante_num) / 8.0
    vec[1] = float(state.round_num) / 24.0

    # 2. Game Phase one-hot (indices 2..10, 9 classes)
    # [BLIND_SELECT, SHOP, PACK_ARCANA, PACK_CELESTIAL, PACK_SPECTRAL, PACK_BUFFOON, PACK_STANDARD, SELECTING_HAND, OTHER]
    st = state.state
    phase_idx = 8  # OTHER
    if st == GameScreen.BLIND_SELECT:
        phase_idx = 0
    elif st == GameScreen.SHOP:
        phase_idx = 1
    elif st in GameScreen.BOOSTER_STATES:
        pack_cards = state.pack.cards if state.pack else []
        if any(c.set == "Joker" or c.key.startswith("j_") for c in pack_cards):
            phase_idx = 5  # PACK_BUFFOON
        elif any(c.key.startswith("c_") and c.key in (
            "c_mercury", "c_venus", "c_earth", "c_mars", "c_jupiter",
            "c_saturn", "c_uranus", "c_neptune", "c_pluto", "c_planet_x",
            "c_ceres", "c_eris"
        ) for c in pack_cards):
            phase_idx = 3  # PACK_CELESTIAL
        elif any(c.key.startswith("c_") and c.key in (
            "c_familiar", "c_grim", "c_incantation", "c_talisman", "c_aura",
            "c_wraith", "c_sigil", "c_ouija", "c_ectoplasm", "c_immolate",
            "c_ankh", "c_deja_vu", "c_hex", "c_trance", "c_medium",
            "c_cryptid", "c_soul", "c_black_hole"
        ) for c in pack_cards):
            phase_idx = 4  # PACK_SPECTRAL
        elif any(c.value.rank and c.value.suit for c in pack_cards):
            phase_idx = 6  # PACK_STANDARD
        else:
            phase_idx = 2  # PACK_ARCANA
    elif st == GameScreen.SELECTING_HAND:
        phase_idx = 7

    vec[2 + phase_idx] = 1.0

    # 3. Economy (indices 11..15, 5 floats)
    money = float(state.money)
    interest_cap = float(get_interest_cap(state))
    floor_val = float(get_protected_interest_floor(state))
    vec[11] = float(np.clip(money / 100.0, -1.0, 5.0))
    vec[12] = interest_cap / 100.0
    current_interest = min(money // 5.0, interest_cap // 5.0)
    vec[13] = float(np.clip(current_interest / 20.0, 0.0, 2.0))
    vec[14] = float(state.round.reroll_cost) / 20.0
    spendable = max(0.0, money - floor_val)
    vec[15] = float(np.clip(spendable / 50.0, 0.0, 5.0))

    # 4. Capacities & Limits (indices 16..23, 8 floats)
    vec[16] = float(state.round.hands_left) / 10.0
    vec[17] = float(state.round.discards_left) / 10.0
    vec[18] = float(state.round.hands_played) / 10.0
    vec[19] = float(state.round.discards_used) / 10.0
    hand_limit = state.hand.limit if state.hand and state.hand.limit > 0 else 8
    vec[20] = float(hand_limit) / 12.0

    # Overflow encoding: real number of jokers owned divided by 10.0
    real_joker_count = len(state.jokers.cards) if state.jokers else 0
    vec[21] = float(real_joker_count) / 10.0
    joker_limit = state.jokers.limit if state.jokers and state.jokers.limit > 0 else 5
    vec[22] = float(joker_limit) / 10.0
    non_negative_jokers = sum(
        1 for j in (state.jokers.cards if state.jokers else [])
        if _normalize_edition_str(j.modifier.edition) != "NEGATIVE"
    )
    free_joker_slots = max(0, joker_limit - non_negative_jokers)
    vec[23] = float(free_joker_slots) / 5.0

    # 5. Consumable slots (indices 24..25, 2 floats)
    cons_count = len(state.consumables.cards) if state.consumables else 0
    cons_limit = state.consumables.limit if state.consumables and state.consumables.limit > 0 else 2
    vec[24] = float(cons_count) / 4.0
    vec[25] = float(max(0, cons_limit - cons_count)) / 4.0

    # 6. Estimated Power Ratio (index 26, 1 float)
    vec[26] = compute_estimated_power_ratio(state)

    # 7. Target & Score (indices 27..28, 2 floats)
    target = float(max(1, _get_target_chips(state)))
    vec[27] = float(math.log10(target) / 6.0)
    current_chips = float(state.round.chips)
    vec[28] = float(np.clip(current_chips / target, 0.0, 2.0))

    # 8. Blind Type one-hot (indices 29..32, 4 classes: SMALL, BIG, BOSS, NONE)
    b_type = (state.active_blind.type if state.active_blind else "").upper()
    if b_type == "SMALL":
        vec[29] = 1.0
    elif b_type == "BIG":
        vec[30] = 1.0
    elif b_type == "BOSS":
        vec[31] = 1.0
    else:
        vec[32] = 1.0

    # 9. Boss Features (indices 33..37, 5 floats)
    boss_name = ""
    boss_is_active = 0.0
    if state.active_blind and state.active_blind.type == "BOSS":
        boss_name = state.active_blind.name
        boss_is_active = 1.0
    elif "boss" in state.blinds and state.blinds["boss"].name:
        boss_name = state.blinds["boss"].name

    boss_idx = BOSS_TO_IDX.get(boss_name, 0)
    vec[33] = float(boss_idx) / 30.0
    vec[34] = boss_is_active

    dom_suit = get_dominant_deck_suit(state) or "H"
    boss_debuffs_suit = 0.0
    if (boss_name == "The Goad" and dom_suit == "S") or \
       (boss_name == "The Head" and dom_suit == "H") or \
       (boss_name == "The Club" and dom_suit == "C") or \
       (boss_name == "The Window" and dom_suit == "D"):
        boss_debuffs_suit = 1.0
    vec[35] = boss_debuffs_suit

    vec[36] = 1.0 if boss_name == "The Plant" else 0.0  # Debuffs face cards
    vec[37] = 1.0 if boss_name in ("The Eye", "The Mouth", "The Arm", "The Flint") else 0.0

    # 10. Tag Features (indices 38..39, 2 floats)
    tag_name = state.active_blind.tag_name if state.active_blind else None
    tag_idx = TAG_TO_IDX.get(tag_name or "", 0)
    vec[38] = float(tag_idx) / 30.0
    vec[39] = 1.0 if tag_name else 0.0

    # 11. Deck Summary (indices 40..42, 3 floats)
    deck_cards = state.deck_cards.cards if state.deck_cards else []
    total_cards = len(deck_cards)
    vec[40] = float(total_cards) / 52.0

    suit_counts = {"H": 0, "D": 0, "C": 0, "S": 0}
    for dc in deck_cards:
        if dc.value.suit in suit_counts:
            suit_counts[dc.value.suit] += 1
    max_suit_count = max(suit_counts.values()) if total_cards > 0 else 0
    vec[41] = float(max_suit_count) / float(max(1, total_cards))
    vec[42] = float(SUIT_TO_IDX.get(dom_suit, 0)) / 4.0

    # 12. Strategic Flags (indices 43..44, 2 floats)
    vec[43] = 1.0 if is_survival_mode(state) else 0.0
    used_v = state.raw.get("used_vouchers", {})
    v_count = len(used_v) if isinstance(used_v, (dict, list, set)) else 0
    vec[44] = float(v_count) / 16.0

    return np.clip(vec, -5.0, 5.0).astype(np.float32)


# ── Feature Extraction Main Function ──────────────────────────────────────────

def extract_features(state: ParsedGameState) -> dict[str, np.ndarray]:
    """Convert any ParsedGameState into a dictionary of normalized float32 NumPy arrays.

    Strictly satisfies get_observation_space() constraints without NaNs or Infs.
    Handles Joker overflow (> 10 Jokers) deterministically by selecting the 10 most
    strategic / vendable Jokers without IndexError.
    """
    obs_space = get_observation_space()

    # 1. Held Jokers (shape: 10, D_ENTITY)
    held_jokers = np.zeros((MAX_JOKERS, D_ENTITY), dtype=np.float32)
    jokers = list(state.jokers.cards) if state.jokers else []

    if len(jokers) > MAX_JOKERS:
        # Overflow management: sort to prioritize vendable / non-eternal and highest tier Jokers
        sorted_jokers = sorted(
            jokers,
            key=lambda j: (
                not j.modifier.eternal,
                evaluate_joker_value(j, state),
                j.cost.sell,
            ),
            reverse=True,
        )
        selected_jokers = sorted_jokers[:MAX_JOKERS]
    else:
        selected_jokers = jokers

    for i, j in enumerate(selected_jokers):
        held_jokers[i] = _encode_entity(j, is_offer=False)

    # 2. Shop and Pack Offers (shape: 12, D_ENTITY)
    # Slots:
    # In SHOP:
    #   0..3: shop cards
    #   4: voucher
    #   5..6: booster packs
    #   7..10: held consumables
    # In BOOSTER_OPENED:
    #   0..4: pack choices
    #   5..8: held consumables
    # In other screens:
    #   0..3: held consumables
    offers = np.zeros((MAX_OFFERS, D_ENTITY), dtype=np.float32)

    if state.state == GameScreen.SHOP:
        slot = 0
        # Shop cards (0..3)
        shop_cards = state.shop.cards if state.shop else []
        for c in shop_cards[:4]:
            offers[slot] = _encode_entity(c, is_offer=True)
            slot += 1
        slot = 4

        # Voucher (4)
        vouchers = state.vouchers.cards if state.vouchers else []
        if vouchers:
            offers[slot] = _encode_entity(vouchers[0], is_offer=True)
        slot = 5

        # Boosters (5..6)
        packs = state.packs.cards if state.packs else []
        for p in packs[:2]:
            offers[slot] = _encode_entity(p, is_offer=True)
            slot += 1
        slot = 7

        # Held Consumables (7..10)
        consumables = state.consumables.cards if state.consumables else []
        for c in consumables[:4]:
            offers[slot] = _encode_entity(c, is_offer=False)
            slot += 1

    elif state.state in GameScreen.BOOSTER_STATES:
        slot = 0
        # Pack choices (0..4)
        pack_cards = state.pack.cards if state.pack else []
        for c in pack_cards[:5]:
            offers[slot] = _encode_entity(c, is_offer=True)
            slot += 1
        slot = 5

        # Held Consumables (5..8)
        consumables = state.consumables.cards if state.consumables else []
        for c in consumables[:4]:
            offers[slot] = _encode_entity(c, is_offer=False)
            slot += 1

    else:
        # In BLIND_SELECT, SELECTING_HAND, etc.: expose held consumables (0..3)
        slot = 0
        consumables = state.consumables.cards if state.consumables else []
        for c in consumables[:4]:
            offers[slot] = _encode_entity(c, is_offer=False)
            slot += 1

    # 3. Deck Features (shape: 32)
    deck_features = _extract_deck_features(state)

    # 4. Hand Levels (shape: 24)
    hand_levels = _extract_hand_levels(state)

    # 5. Context Features (shape: 45)
    context_features = _extract_context_features(state)

    obs = {
        "context_features": context_features,
        "deck_features": deck_features,
        "hand_levels": hand_levels,
        "held_jokers": held_jokers,
        "shop_and_pack_offers": offers,
    }

    # Ensure no NaN or Inf
    for k, v in obs.items():
        if np.isnan(v).any() or np.isinf(v).any():
            obs[k] = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    return obs
