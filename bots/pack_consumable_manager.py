"""Booster Packs and Consumables Management (Silo 5).

Responsibilities:
- Booster pack purchasing policy in shop (integration with shop_manager.py).
- Opened booster pack decision loop (decide_pack_action):
  - Buffoon Pack: choose best Joker via evaluate_joker_value or skip if full/weak.
  - Celestial Pack: Black Hole first, or planet for most played hand (or Jupiter/Mercury/Uranus).
  - Arcana Pack & Spectral Pack: prioritize Economy/Legendaries, Thinning/Duplication,
    Enhancements/Seals, Suit conversion. Anti-suicide safety on Wraith, Ankh, Hex, Ectoplasm.
  - Standard Pack: Hologram synergy, editions/seals/enhancements, or skip.
- Intelligent target selection for consumables requiring hand cards:
  - 0-target cards: Planets, Hermit, Temperance, Fool, Soul, Wraith, Ankh, Hex, Immolate, etc.
  - Hanged Man: 1 to 2 weakest non-protected cards.
  - Death: exactly 2 cards [worst_idx, best_idx] with worst to the left of best.
  - Enhancements for scoring cards: Empress, Hierophant, Justice, Deja Vu, Talisman, Aura.
  - Enhancements for held cards: Chariot (Steel), Devil (Gold), Trance (Blue Seal).
  - Discard seal: Medium (Purple Seal) on low disposable cards.
- Held consumable inventory management (manage_held_consumables):
  - Planets: immediate use in SHOP or SELECTING_HAND.
  - 0-target tarots/spectrals: use when conditions optimal.
  - Targeted tarots/spectrals: use during SELECTING_HAND with card selection.
  - Sell dangerous/unusable consumables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bots.shop_manager import (
    evaluate_joker_value,
    get_protected_interest_floor,
    get_spendable_budget,
    is_survival_mode,
)
from bots.state_parser import Card, ParsedGameState

# ── Action Dataclasses ─────────────────────────────────────────────────────

@dataclass
class PackAction:
    """Action to take inside an opened booster pack."""
    action_type: str  # "select" or "skip"
    rpc_method: str  # "pack"
    rpc_params: dict[str, Any]  # {"card": int, "targets": [int, ...]} or {"skip": True}
    card_index: int | None
    card_name: str
    target_indices: list[int] = field(default_factory=list)
    score: float = 0.0
    reason: str = ""
    skip: bool = False


@dataclass
class ConsumableAction:
    """Action to take with a held consumable in inventory."""
    action_type: str  # "use" or "sell"
    rpc_method: str  # "use" or "sell"
    rpc_params: dict[str, Any]  # {"consumable": int, "cards": [int, ...]} or {"consumable": int}
    consumable_index: int
    card_name: str
    target_indices: list[int] = field(default_factory=list)
    reason: str = ""


# ── Constants & Registries ─────────────────────────────────────────────────

# Rank values for card strength evaluation
RANK_STRENGTH: dict[str, int] = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}

# Planet hand mappings
PLANET_TO_HAND: dict[str, str] = {
    "c_mercury": "Pair",
    "c_venus": "Three of a Kind",
    "c_earth": "Full House",
    "c_mars": "Four of a Kind",
    "c_jupiter": "Flush",
    "c_saturn": "Straight",
    "c_uranus": "Two Pair",
    "c_neptune": "Straight Flush",
    "c_pluto": "High Card",
    "c_planet_x": "Five of a Kind",
    "c_ceres": "Flush House",
    "c_eris": "Flush Five",
}

# Baseline priority for planets when no hand is heavily played
PLANET_BASE_PRIORITY: dict[str, float] = {
    "c_black_hole": 100.0,
    "c_jupiter": 76.0,  # Flush
    "c_mercury": 75.0,  # Pair
    "c_uranus": 74.0,   # Two Pair
    "c_venus": 73.0,    # Three of a Kind
    "c_saturn": 72.0,   # Straight
    "c_earth": 71.0,    # Full House
    "c_mars": 68.0,     # Four of a Kind
    "c_pluto": 62.0,    # High Card
    "c_neptune": 55.0,  # Straight Flush
    "c_planet_x": 50.0, # Five of a Kind
    "c_ceres": 50.0,    # Flush House
    "c_eris": 50.0,     # Flush Five
}

# Zero-target consumables (can be used without selecting hand cards)
ZERO_TARGET_CONSUMABLES: set[str] = {
    "c_mercury", "c_venus", "c_earth", "c_mars", "c_jupiter", "c_saturn",
    "c_uranus", "c_neptune", "c_pluto", "c_planet_x", "c_ceres", "c_eris",
    "c_black_hole", "c_hermit", "c_temperance", "c_fool", "c_high_priestess",
    "c_emperor", "c_judgement", "c_wheel_of_fortune", "c_soul", "c_wraith",
    "c_ankh", "c_hex", "c_ectoplasm", "c_immolate", "c_familiar", "c_grim",
    "c_incantation", "c_sigil",
}

# Consumables that require target cards in hand
TARGET_REQUIREMENTS: dict[str, tuple[int, int]] = {
    "c_hanged_man": (1, 2),
    "c_death": (2, 2),
    "c_empress": (1, 2),
    "c_heirophant": (1, 2),
    "c_strength": (1, 2),
    "c_magician": (1, 2),
    "c_star": (1, 3),
    "c_moon": (1, 3),
    "c_sun": (1, 3),
    "c_world": (1, 3),
    "c_chariot": (1, 1),
    "c_justice": (1, 1),
    "c_devil": (1, 1),
    "c_tower": (1, 1),
    "c_lovers": (1, 1),
    "c_aura": (1, 1),
    "c_talisman": (1, 1),
    "c_deja_vu": (1, 1),
    "c_trance": (1, 1),
    "c_medium": (1, 1),
    "c_cryptid": (1, 1),
}


# ── Card Strength Evaluation & Helpers ─────────────────────────────────────

def get_dominant_deck_suit(state: ParsedGameState) -> str:
    """Identify the primary/dominant suit of the deck or Joker synergies."""
    jkeys = {j.key for j in state.jokers.cards}
    if "j_bloodstone" in jkeys or "j_lusty_joker" in jkeys:
        return "H"
    if "j_greedy_joker" in jkeys or "j_rough_gem" in jkeys:
        return "D"
    if "j_wrathful_joker" in jkeys or "j_arrowhead" in jkeys:
        return "S"
    if "j_gluttenous_joker" in jkeys or "j_onyx_agate" in jkeys:
        return "C"

    # Count suits in hand if available
    if state.hand.cards:
        counts: dict[str, int] = {}
        for c in state.hand.cards:
            s = c.value.suit or ""
            if s:
                counts[s] = counts.get(s, 0) + 1
        if counts:
            return max(counts, key=counts.get)  # type: ignore[arg-type]

    # Default to Spades
    return "S"


def evaluate_card_strength(card: Card, state: ParsedGameState) -> float:
    """Deterministically score a playing card to identify the weakest and strongest."""
    if card.state.debuff:
        return -50.0

    score = 0.0

    # Base Rank score (2 -> 2.0, A -> 14.0)
    rank_str = card.value.rank or ""
    score += RANK_STRENGTH.get(rank_str, 5)

    # Dominant suit bonus
    dominant_suit = get_dominant_deck_suit(state)
    if card.value.suit == dominant_suit:
        score += 8.0

    # Edition bonus
    ed = (card.modifier.edition or "").lower()
    if "polychrome" in ed:
        score += 50.0
    elif "holographic" in ed:
        score += 35.0
    elif "foil" in ed:
        score += 25.0

    # Seal bonus
    seal = (card.modifier.seal or "").lower()
    if "red" in seal:
        score += 40.0
    elif "blue" in seal:
        score += 35.0
    elif "purple" in seal:
        score += 30.0
    elif "gold" in seal:
        score += 25.0

    # Enhancement bonus
    enh = (card.modifier.enhancement or "").lower()
    if "steel" in enh:
        score += 35.0
    elif "glass" in enh:
        score += 30.0
    elif "mult" in enh:
        score += 18.0
    elif "bonus" in enh:
        score += 15.0
    elif "gold" in enh:
        score += 15.0
    elif "lucky" in enh:
        score += 12.0
    elif "wild" in enh:
        score += 8.0
    elif "stone" in enh:
        score += 5.0

    return score


def is_card_protected_from_destruction(card: Card) -> bool:
    """Returns True if card has valuable modifications that should not be destroyed."""
    if card.modifier.edition:
        return True
    if card.modifier.seal:
        return True
    enh = (card.modifier.enhancement or "").lower()
    if any(k in enh for k in ("steel", "glass", "gold", "mult", "bonus")):
        return True
    return False


# ── Booster Pack Purchase Policy (in Shop) ─────────────────────────────────

def evaluate_pack_purchase(pack_card: Card, state: ParsedGameState) -> float:
    """Evaluate whether to purchase a booster pack available in the shop.

    Respects interest floor protection policy unless in early-game survival mode.
    """
    cost = pack_card.cost.buy or 4
    if state.money < cost:
        return 0.0

    floor = get_protected_interest_floor(state)
    spendable = get_spendable_budget(state)
    survival = is_survival_mode(state)

    key = pack_card.key.lower()
    jkeys = {j.key for j in state.jokers.cards}
    n_jokers = len(state.jokers.cards)
    max_jokers = state.jokers.limit or 5

    # 1. Buffoon Pack (Jokers)
    if "buffoon" in key:
        # Very high priority in Ante 1-3 if open slots or weak jokers
        if state.ante_num <= 3 and (n_jokers < max_jokers or survival):
            return 95.0
        # Otherwise, check spendable budget
        if cost <= spendable:
            return 75.0
        if survival and state.money >= cost:
            return 72.0
        return 0.0

    # 2. Celestial Pack (Planets)
    if "celestial" in key:
        bonus = 5.0 if "mega" in key or "jumbo" in key else 0.0
        if cost <= spendable:
            return 78.0 + bonus
        if survival and state.money - cost >= 10:
            return 65.0 + bonus
        return 0.0

    # 3. Arcana Pack (Tarots)
    if "arcana" in key:
        bonus = 5.0 if "mega" in key or "jumbo" in key else 0.0
        if cost <= spendable:
            return 80.0 + bonus
        if survival and state.money - cost >= 10:
            return 68.0 + bonus
        return 0.0

    # 4. Spectral Pack
    if "spectral" in key:
        bonus = 5.0 if "mega" in key or "jumbo" in key else 0.0
        if cost <= spendable:
            return 82.0 + bonus
        if state.ante_num <= 4 and state.money - cost >= 10:
            return 74.0 + bonus
        return 0.0

    # 5. Standard Pack (Playing cards)
    if "standard" in key:
        # Exceptional synergy: Hologram gains x0.25 Mult per added card!
        if "j_hologram" in jkeys:
            return 93.0
        # Driver's license synergy if under 16 enhanced cards
        if "j_drivers_license" in jkeys:
            enhanced_count = sum(1 for c in state.deck_cards.cards if c.modifier.enhancement)
            if enhanced_count < 16 and cost <= spendable:
                return 75.0
        # Default policy: low priority to avoid deck dilution
        if cost <= spendable and state.money >= floor + 15:
            return 30.0
        return 0.0

    # Generic pack fallback
    return 60.0 if cost <= spendable else 0.0


# ── Consumable Target Selection ───────────────────────────────────────────

def select_targets_for_consumable(
    consumable: Card,
    hand_cards: list[Card],
    state: ParsedGameState,
) -> list[int]:
    """Select the exact indices of cards in hand to target for a consumable.

    Returns an empty list if the consumable requires 0 targets or if no valid
    targets exist.
    """
    key = consumable.key.lower()

    # Case 0: Consumables requiring 0 targets
    if key in ZERO_TARGET_CONSUMABLES:
        return []

    if not hand_cards:
        return []

    dominant_suit = get_dominant_deck_suit(state)

    # Case 1: The Hanged Man (Destroys 1 to 2 cards)
    if key == "c_hanged_man":
        # Candidates: non-protected cards, sorted by strength ascending (weakest first)
        candidates = [
            (idx, card, evaluate_card_strength(card, state))
            for idx, card in enumerate(hand_cards)
            if not is_card_protected_from_destruction(card)
        ]
        if not candidates:
            return []
        candidates.sort(key=lambda item: item[2])
        # Pick up to 2 weakest cards
        return [item[0] for item in candidates[:2]]

    # Case 2: Death (Convert left card into right card: exactly 2 cards)
    if key == "c_death":
        if len(hand_cards) < 2:
            return []

        # We must find a pair (i, j) with i < j (left card i converted into right card j)
        # where card i is weak and card j is strong.
        scored_cards = [
            (idx, card, evaluate_card_strength(card, state))
            for idx, card in enumerate(hand_cards)
        ]

        best_pair: tuple[int, int] | None = None
        max_gain = -999.0

        for i in range(len(scored_cards)):
            for j in range(i + 1, len(scored_cards)):
                left_idx, _, left_val = scored_cards[i]
                right_idx, _, right_val = scored_cards[j]
                # Gain of converting left into right
                gain = right_val - left_val
                if gain > max_gain:
                    max_gain = gain
                    best_pair = (left_idx, right_idx)

        # Only execute if right card is strictly better than left card
        if best_pair and max_gain > 0:
            return [best_pair[0], best_pair[1]]

        # Fallback: if no natural left-to-right pair gives a gain (e.g. best card is at index 0),
        # return the first 2 indices or empty
        return []

    # Case 3: Scoring Enhancement Cards (The Empress, The Hierophant, Justice, Aura, Talisman, Deja Vu)
    if key in ("c_empress", "c_heirophant"):
        # Enhances 1 to 2 cards with Mult (+4) or Bonus (+30 chips)
        # Pick best cards that do not yet have a major enhancement
        eligible = [
            (idx, card, evaluate_card_strength(card, state))
            for idx, card in enumerate(hand_cards)
            if not card.modifier.enhancement
        ]
        eligible.sort(key=lambda item: item[2], reverse=True)
        return [item[0] for item in eligible[:2]]

    if key == "c_justice":
        # Enhances 1 card to Glass (x2 Mult)
        eligible = [
            (idx, card, evaluate_card_strength(card, state))
            for idx, card in enumerate(hand_cards)
            if not card.modifier.enhancement
        ]
        eligible.sort(key=lambda item: item[2], reverse=True)
        return [eligible[0][0]] if eligible else []

    if key == "c_aura":
        # Adds Foil, Holo, or Polychrome to 1 card
        eligible = [
            (idx, card, evaluate_card_strength(card, state))
            for idx, card in enumerate(hand_cards)
            if not card.modifier.edition
        ]
        eligible.sort(key=lambda item: item[2], reverse=True)
        return [eligible[0][0]] if eligible else []

    if key == "c_deja_vu":
        # Adds Red Seal (retrigger) to 1 card
        eligible = [
            (idx, card, evaluate_card_strength(card, state))
            for idx, card in enumerate(hand_cards)
            if not card.modifier.seal
        ]
        eligible.sort(key=lambda item: item[2], reverse=True)
        return [eligible[0][0]] if eligible else []

    if key == "c_talisman":
        # Adds Gold Seal ($3 when scored) to 1 card
        eligible = [
            (idx, card, evaluate_card_strength(card, state))
            for idx, card in enumerate(hand_cards)
            if not card.modifier.seal
        ]
        eligible.sort(key=lambda item: item[2], reverse=True)
        return [eligible[0][0]] if eligible else []

    if key == "c_cryptid":
        # Spectral: create 2 copies of 1 selected card in hand
        # Target the absolute strongest card in hand
        scored = [
            (idx, card, evaluate_card_strength(card, state))
            for idx, card in enumerate(hand_cards)
        ]
        scored.sort(key=lambda item: item[2], reverse=True)
        return [scored[0][0]] if scored else []

    # Case 4: Held-in-hand enhancements (The Chariot / Steel, The Devil / Gold, Trance / Blue Seal)
    jkeys = {j.key for j in state.jokers.cards}

    if key == "c_chariot":
        # Steel (+1.5x Mult held in hand)
        # Check Baron synergy (Kings) or Shoot the Moon (Queens)
        if "j_baron" in jkeys:
            kings = [i for i, c in enumerate(hand_cards) if (c.value.rank or "") == "K"]
            if kings:
                return [kings[0]]
        if "j_shoot_the_moon" in jkeys:
            queens = [i for i, c in enumerate(hand_cards) if (c.value.rank or "") == "Q"]
            if queens:
                return [queens[0]]
        # Otherwise: a card that will stay held in hand (off-suit or low card)
        off_cards = [
            (i, c, evaluate_card_strength(c, state))
            for i, c in enumerate(hand_cards)
            if c.value.suit != dominant_suit and not c.modifier.enhancement
        ]
        if off_cards:
            off_cards.sort(key=lambda item: item[2])
            return [off_cards[0][0]]
        # Fallback to any unenhanced card
        unenh = [i for i, c in enumerate(hand_cards) if not c.modifier.enhancement]
        return [unenh[0]] if unenh else [0]

    if key == "c_devil":
        # Gold Card ($3 held at end of round)
        off_cards = [
            (i, c)
            for i, c in enumerate(hand_cards)
            if not c.modifier.enhancement
        ]
        return [off_cards[0][0]] if off_cards else [0]

    if key == "c_trance":
        # Blue Seal (creates planet card of final hand played if held at round end)
        unsealed = [
            (i, c, evaluate_card_strength(c, state))
            for i, c in enumerate(hand_cards)
            if not c.modifier.seal
        ]
        if unsealed:
            return [unsealed[0][0]]
        return [0]

    if key == "c_medium":
        # Purple Seal (creates tarot when discarded)
        # Target a low card likely to be discarded
        low_cards = [
            (i, c, evaluate_card_strength(c, state))
            for i, c in enumerate(hand_cards)
            if not c.modifier.seal
        ]
        if low_cards:
            low_cards.sort(key=lambda item: item[2])
            return [low_cards[0][0]]
        return [0]

    if key == "c_strength":
        # +1 rank on up to 2 cards
        # Target cards below Ace, preferably pairs or high ranks (e.g. Q -> K, J -> Q)
        upgradeable = [
            (i, c)
            for i, c in enumerate(hand_cards)
            if (c.value.rank or "") != "A"
        ]
        return [item[0] for item in upgradeable[:2]]

    # Case 5: Suit conversion (The Star, The Moon, The Sun, The World)
    target_suit_map = {
        "c_star": "D",
        "c_moon": "C",
        "c_sun": "H",
        "c_world": "S",
    }
    if key in target_suit_map:
        target_suit = target_suit_map[key]
        off_suit = [
            i for i, c in enumerate(hand_cards)
            if c.value.suit != target_suit
        ]
        return off_suit[:3]

    # Default fallback for any remaining targeted consumable: up to max required
    req = TARGET_REQUIREMENTS.get(key)
    if req:
        min_targets, max_targets = req
        return list(range(min(len(hand_cards), max_targets)))

    return []


# ── Decision Inside Opened Booster Pack (decide_pack_action) ───────────────

def decide_pack_action(state: ParsedGameState) -> PackAction:
    """Choose the best card or skip inside an opened booster pack.

    Evaluates:
    - Buffoon Pack: best Joker value vs slots.
    - Celestial Pack: Black Hole, then most played hand planet or base priority.
    - Arcana Pack / Spectral Pack: prioritized scoring with anti-suicide checks.
    - Standard Pack: Hologram synergy, editions/seals, or skip.
    """
    pack_cards = state.pack.cards
    if not pack_cards:
        return PackAction(
            action_type="skip",
            rpc_method="pack",
            rpc_params={"skip": True},
            card_index=None,
            card_name="None",
            score=0.0,
            reason="No cards available in pack; skipping",
            skip=True,
        )

    # ── 1. Buffoon Pack (Jokers) ──────────────────────────────────────────
    is_buffoon = any(c.set.lower() == "joker" for c in pack_cards)
    if is_buffoon:
        n_jokers = len(state.jokers.cards)
        max_slots = state.jokers.limit or 5

        # Evaluate all Jokers in pack
        scored_jokers = [
            (idx, card, evaluate_joker_value(card, state))
            for idx, card in enumerate(pack_cards)
        ]
        scored_jokers.sort(key=lambda item: item[2], reverse=True)
        best_idx, best_card, best_score = scored_jokers[0]

        # If slots are full: Balatro forbids taking non-negative Jokers from packs
        is_negative = "negative" in (best_card.modifier.edition or "").lower()
        if n_jokers >= max_slots and not is_negative:
            return PackAction(
                action_type="skip",
                rpc_method="pack",
                rpc_params={"skip": True},
                card_index=None,
                card_name=best_card.label or best_card.key,
                score=0.0,
                reason=f"Joker slots full ({n_jokers}/{max_slots}); cannot take {best_card.label or best_card.key}",
                skip=True,
            )

        if best_score > 0.0:
            return PackAction(
                action_type="select",
                rpc_method="pack",
                rpc_params={"card": best_idx},
                card_index=best_idx,
                card_name=best_card.label or best_card.key,
                score=best_score,
                reason=f"Chose Joker {best_card.label or best_card.key} (Score: {best_score:.1f})",
                skip=False,
            )

        return PackAction(
            action_type="skip",
            rpc_method="pack",
            rpc_params={"skip": True},
            card_index=None,
            card_name="None",
            score=0.0,
            reason="No valuable Jokers in Buffoon pack; skipping",
            skip=True,
        )

    # ── 2. Celestial Pack (Planets) ───────────────────────────────────────
    is_celestial = any(c.key.lower() == "c_black_hole" or c.key.lower() in PLANET_TO_HAND for c in pack_cards)
    if is_celestial:
        # Check for Black Hole first
        for idx, card in enumerate(pack_cards):
            if card.key.lower() == "c_black_hole":
                return PackAction(
                    action_type="select",
                    rpc_method="pack",
                    rpc_params={"card": idx},
                    card_index=idx,
                    card_name=card.label or card.key,
                    score=100.0,
                    reason="Black Hole! Level up all hands by 1",
                    skip=False,
                )

        # Check most played hand in run
        most_played_hand = ""
        max_played_count = 0
        for hname, hinfo in state.hands.items():
            if hinfo.played > max_played_count:
                max_played_count = hinfo.played
                most_played_hand = hname

        scored_planets: list[tuple[int, Card, float]] = []
        for idx, card in enumerate(pack_cards):
            ckey = card.key.lower()
            hand_name = PLANET_TO_HAND.get(ckey, "")
            base_score = PLANET_BASE_PRIORITY.get(ckey, 60.0)

            # Massive bonus if planet matches our most played hand
            if hand_name and hand_name == most_played_hand and max_played_count > 0:
                score = 90.0 + min(max_played_count * 2.0, 10.0)
            else:
                score = base_score

            scored_planets.append((idx, card, score))

        scored_planets.sort(key=lambda item: item[2], reverse=True)
        best_idx, best_card, best_score = scored_planets[0]

        return PackAction(
            action_type="select",
            rpc_method="pack",
            rpc_params={"card": best_idx},
            card_index=best_idx,
            card_name=best_card.label or best_card.key,
            score=best_score,
            reason=f"Chose Planet {best_card.label or best_card.key} ({PLANET_TO_HAND.get(best_card.key.lower(), '')}) (Score: {best_score:.1f})",
            skip=False,
        )

    # ── 3. Arcana Pack (Tarots) & Spectral Pack ────────────────────────────
    is_consumable_pack = any(
        c.set.lower() in ("tarot", "spectral") or c.key.lower() in ZERO_TARGET_CONSUMABLES or c.key.lower() in TARGET_REQUIREMENTS
        for c in pack_cards
    )
    if is_consumable_pack:
        scored_cards: list[tuple[int, Card, float, list[int], str]] = []

        for idx, card in enumerate(pack_cards):
            ckey = card.key.lower()
            score = 0.0
            reason_desc = ""

            # ── Anti-Suicide & Safety Rules for Spectral cards ─────────
            if ckey == "c_wraith":
                # Sets money to $0 for a Rare Joker
                if state.money <= 6 and len(state.jokers.cards) < (state.jokers.limit or 5):
                    score = 85.0
                    reason_desc = "Wraith safe: money <= $6 and open Joker slot"
                else:
                    score = -100.0
                    reason_desc = f"Wraith refused: money is ${state.money} (would reset to $0)"

            elif ckey in ("c_ankh", "c_hex"):
                # Destroys all other Jokers! Requires strictly 1 Joker owned to target/copy.
                if len(state.jokers.cards) == 1:
                    score = 88.0
                    reason_desc = f"{card.label or ckey} safe: exactly 1 Joker owned"
                else:
                    score = -100.0
                    reason_desc = f"{card.label or ckey} refused: {len(state.jokers.cards)} Jokers owned (requires exactly 1)"

            elif ckey == "c_ectoplasm":
                # Permanent -1 hand size, requires at least 1 joker!
                hand_limit = state.hand.limit or 8
                if len(state.jokers.cards) >= 1 and hand_limit >= 8:
                    score = 80.0
                    reason_desc = f"Ectoplasm taken: hand limit is {hand_limit} (>= 8) and >= 1 Joker owned"
                else:
                    score = -100.0
                    reason_desc = f"Ectoplasm refused: {len(state.jokers.cards)} Jokers owned, hand limit is {hand_limit}"

            elif ckey == "c_ouija":
                # Converts hand to single rank, -1 hand size
                if state.ante_num <= 1:
                    score = 70.0
                    reason_desc = "Ouija taken in Ante 1"
                else:
                    score = -50.0
                    reason_desc = "Ouija refused: -1 hand size too costly"

            # ── Priority 1: Economy & Legendaries ─────────────────────
            elif ckey == "c_soul":
                score = 99.0
                reason_desc = "The Soul: Legendary Joker!"

            elif ckey == "c_hermit":
                # Doubles money up to $20
                if state.money >= 6:
                    score = 75.0 + min(state.money, 20.0)
                    reason_desc = f"The Hermit: +${min(state.money, 20)} gain"
                else:
                    score = 55.0
                    reason_desc = f"The Hermit: low gain (+${state.money})"

            elif ckey == "c_temperance":
                total_sell = sum(j.cost.sell for j in state.jokers.cards)
                if total_sell >= 6:
                    score = 75.0 + min(total_sell, 20.0)
                    reason_desc = f"Temperance: +${total_sell} gain"
                else:
                    score = 50.0
                    reason_desc = f"Temperance: low gain (+${total_sell})"

            elif ckey == "c_immolate":
                score = 88.0
                reason_desc = "Immolate: destroys 5 cards and gives +$20"

            # ── Priority 2: Deck Thinning & Duplication ───────────────
            elif ckey == "c_hanged_man":
                score = 86.0
                reason_desc = "The Hanged Man: deck thinning"

            elif ckey == "c_death":
                score = 87.0
                reason_desc = "Death: duplicate best card over weakest"

            elif ckey == "c_cryptid":
                score = 89.0
                reason_desc = "Cryptid: create 2 copies of best card"

            # ── Priority 3: Enhancements & Seals ──────────────────────
            elif ckey == "c_deja_vu":
                score = 83.0
                reason_desc = "Deja Vu: Red Seal on best card"

            elif ckey == "c_trance":
                score = 82.0
                reason_desc = "Trance: Blue Seal on held card"

            elif ckey == "c_medium":
                score = 81.0
                reason_desc = "Medium: Purple Seal on discard card"

            elif ckey == "c_aura":
                score = 80.0
                reason_desc = "Aura: Edition on best card"

            elif ckey == "c_chariot":
                score = 79.0
                reason_desc = "The Chariot: Steel card"

            elif ckey == "c_justice":
                score = 78.0
                reason_desc = "Justice: Glass card"

            elif ckey == "c_empress":
                score = 76.0
                reason_desc = "The Empress: +4 Mult on 2 cards"

            elif ckey == "c_heirophant":
                score = 75.0
                reason_desc = "The Hierophant: +30 Chips on 2 cards"

            elif ckey == "c_talisman":
                score = 74.0
                reason_desc = "Talisman: Gold Seal"

            elif ckey == "c_devil":
                score = 73.0
                reason_desc = "The Devil: Gold card"

            elif ckey == "c_strength":
                score = 70.0
                reason_desc = "Strength: +1 rank on 2 cards"

            # ── Priority 4: Generation & Minor Tarots ─────────────────
            elif ckey in ("c_emperor", "c_high_priestess"):
                score = 66.0
                reason_desc = f"{card.label or ckey}: generate consumables"

            elif ckey == "c_judgement":
                if len(state.jokers.cards) < (state.jokers.limit or 5):
                    score = 70.0
                    reason_desc = "Judgement: create random Joker"
                else:
                    score = 20.0
                    reason_desc = "Judgement: Joker slots full"

            elif ckey == "c_wheel_of_fortune":
                score = 62.0
                reason_desc = "Wheel of Fortune: 1 in 4 chance for edition"

            elif ckey in ("c_star", "c_moon", "c_sun", "c_world"):
                score = 64.0
                reason_desc = f"{card.label or ckey}: suit conversion"

            else:
                score = 50.0
                reason_desc = f"{card.label or ckey}"

            # Calculate target cards in hand
            targets = select_targets_for_consumable(card, state.hand.cards, state)
            req = TARGET_REQUIREMENTS.get(ckey)
            if req:
                min_req, _ = req
                if len(targets) < min_req:
                    score = -50.0
                    reason_desc += f" (Insufficient targets: needed {min_req}, found {len(targets)})"

            scored_cards.append((idx, card, score, targets, reason_desc))

        scored_cards.sort(key=lambda item: item[2], reverse=True)
        best_idx, best_card, best_score, best_targets, best_reason = scored_cards[0]

        # If best option is negative (dangerous / suicide / invalid targets): skip
        if best_score <= 0.0:
            return PackAction(
                action_type="skip",
                rpc_method="pack",
                rpc_params={"skip": True},
                card_index=None,
                card_name=best_card.label or best_card.key,
                score=best_score,
                reason=f"All cards in pack are dangerous or unusable ({best_reason}); skipping",
                skip=True,
            )

        rpc_params: dict[str, Any] = {"card": best_idx}
        if best_targets:
            rpc_params["targets"] = best_targets

        return PackAction(
            action_type="select",
            rpc_method="pack",
            rpc_params=rpc_params,
            card_index=best_idx,
            card_name=best_card.label or best_card.key,
            target_indices=best_targets,
            score=best_score,
            reason=f"Chose {best_card.label or best_card.key} ({best_reason})",
            skip=False,
        )

    # ── 4. Standard Pack (Playing cards) ───────────────────────────────────
    jkeys = {j.key for j in state.jokers.cards}
    has_hologram = "j_hologram" in jkeys
    dominant_suit = get_dominant_deck_suit(state)

    scored_standards: list[tuple[int, Card, float, str]] = []
    for idx, card in enumerate(pack_cards):
        str_val = evaluate_card_strength(card, state)
        reasons: list[str] = []

        has_edition = bool(card.modifier.edition)
        has_seal = bool(card.modifier.seal)
        enh = (card.modifier.enhancement or "").lower()
        has_strong_enh = any(k in enh for k in ("steel", "glass", "mult", "bonus"))
        is_high_dominant = (card.value.suit == dominant_suit) and ((card.value.rank or "") in ("A", "K", "Q", "J"))

        if has_hologram:
            score = 90.0 + str_val
            reasons.append("Hologram synergy (+x0.25 Mult)")
        elif has_edition or has_seal or has_strong_enh or is_high_dominant:
            score = 65.0 + str_val
            if has_edition:
                reasons.append(f"Edition {card.modifier.edition}")
            if has_seal:
                reasons.append(f"Seal {card.modifier.seal}")
            if has_strong_enh:
                reasons.append(f"Enhancement {card.modifier.enhancement}")
            if is_high_dominant:
                reasons.append(f"Face/Ace of {dominant_suit}")
        else:
            score = -10.0
            reasons.append("Base card (would dilute deck)")

        scored_standards.append((idx, card, score, ", ".join(reasons)))

    scored_standards.sort(key=lambda item: item[2], reverse=True)
    best_idx, best_card, best_score, best_reason = scored_standards[0]

    if best_score > 0.0:
        return PackAction(
            action_type="select",
            rpc_method="pack",
            rpc_params={"card": best_idx},
            card_index=best_idx,
            card_name=f"{best_card.value.rank}{best_card.value.suit}",
            score=best_score,
            reason=f"Chose playing card {best_card.value.rank}{best_card.value.suit} ({best_reason})",
            skip=False,
        )

    return PackAction(
        action_type="skip",
        rpc_method="pack",
        rpc_params={"skip": True},
        card_index=None,
        card_name="None",
        score=0.0,
        reason="No quality card in Standard pack (avoiding deck dilution); skipping",
        skip=True,
    )


# ── Inventory Consumables Management (manage_held_consumables) ────────────

def manage_held_consumables(state: ParsedGameState) -> ConsumableAction | None:
    """Evaluate held consumables in inventory and determine if one should be used or sold.

    Rules:
    - Planets: use immediately in SHOP or SELECTING_HAND (unless Observatory voucher).
    - 0-target tarots/spectrals: use when conditions optimal.
    - Targeted tarots/spectrals: use in SELECTING_HAND with card selection.
    - Sell dangerous/unusable cards if slots clogged.
    """
    held = state.consumables.cards
    if not held:
        return None

    # Check for Observatory voucher
    used_vouchers = state.raw.get("used_vouchers", {})
    has_observatory = "v_observatory" in used_vouchers

    for idx, card in enumerate(held):
        ckey = card.key.lower()

        # ── 1. Planets in Inventory ────────────────────────────────────────
        if ckey == "c_black_hole" or ckey in PLANET_TO_HAND:
            # If Observatory is active and this planet matches our primary hand, we can hold it
            if has_observatory and ckey in PLANET_TO_HAND:
                # hold if flush/pair
                pass
            else:
                return ConsumableAction(
                    action_type="use",
                    rpc_method="use",
                    rpc_params={"consumable": idx},
                    consumable_index=idx,
                    card_name=card.label or ckey,
                    reason=f"Using Planet {card.label or ckey} immediately to upgrade hand",
                )

        # ── 2. 0-Target Tarots & Spectral Cards (usable in SHOP or SELECTING_HAND)
        if ckey == "c_hermit":
            if state.money >= 15 or (is_survival_mode(state) and state.money >= 6):
                return ConsumableAction(
                    action_type="use",
                    rpc_method="use",
                    rpc_params={"consumable": idx},
                    consumable_index=idx,
                    card_name="The Hermit",
                    reason=f"Using The Hermit to gain +${min(state.money, 20)}",
                )

        elif ckey == "c_temperance":
            total_sell = sum(j.cost.sell for j in state.jokers.cards)
            if total_sell >= 6:
                return ConsumableAction(
                    action_type="use",
                    rpc_method="use",
                    rpc_params={"consumable": idx},
                    consumable_index=idx,
                    card_name="Temperance",
                    reason=f"Using Temperance to gain +${total_sell}",
                )

        elif ckey == "c_soul":
            if len(state.jokers.cards) < (state.jokers.limit or 5):
                return ConsumableAction(
                    action_type="use",
                    rpc_method="use",
                    rpc_params={"consumable": idx},
                    consumable_index=idx,
                    card_name="The Soul",
                    reason="Using The Soul to spawn Legendary Joker",
                )

        elif ckey in ("c_high_priestess", "c_emperor"):
            # Use if consumable area has room
            consumable_limit = state.consumables.limit or 2
            if len(held) < consumable_limit:
                return ConsumableAction(
                    action_type="use",
                    rpc_method="use",
                    rpc_params={"consumable": idx},
                    consumable_index=idx,
                    card_name=card.label or ckey,
                    reason=f"Using {card.label or ckey} to generate consumables",
                )

        elif ckey == "c_judgement":
            if len(state.jokers.cards) < (state.jokers.limit or 5):
                return ConsumableAction(
                    action_type="use",
                    rpc_method="use",
                    rpc_params={"consumable": idx},
                    consumable_index=idx,
                    card_name="Judgement",
                    reason="Using Judgement to create a random Joker",
                )

        elif ckey == "c_wheel_of_fortune":
            non_edition_jokers = [j for j in state.jokers.cards if not j.modifier.edition]
            if non_edition_jokers:
                return ConsumableAction(
                    action_type="use",
                    rpc_method="use",
                    rpc_params={"consumable": idx},
                    consumable_index=idx,
                    card_name="Wheel of Fortune",
                    reason="Using Wheel of Fortune for chance at Joker edition",
                )

        # ── 3. Targeted Tarots / Spectrals (ONLY in SELECTING_HAND) ─────────
        if state.state == "SELECTING_HAND" and state.hand.cards:
            targets = select_targets_for_consumable(card, state.hand.cards, state)
            req = TARGET_REQUIREMENTS.get(ckey)
            min_req = req[0] if req else 1

            if len(targets) >= min_req:
                return ConsumableAction(
                    action_type="use",
                    rpc_method="use",
                    rpc_params={"consumable": idx, "cards": targets},
                    consumable_index=idx,
                    card_name=card.label or ckey,
                    target_indices=targets,
                    reason=f"Using {card.label or ckey} targeting hand cards {targets}",
                )

        # ── 4. Cleanup / Selling dangerous or clogging consumables ────────
        # If Ankh or Hex held while we have multiple valuable Jokers
        if ckey in ("c_ankh", "c_hex") and len(state.jokers.cards) >= 2:
            return ConsumableAction(
                action_type="sell",
                rpc_method="sell",
                rpc_params={"consumable": idx},
                consumable_index=idx,
                card_name=card.label or ckey,
                reason=f"Selling dangerous {card.label or ckey} ({len(state.jokers.cards)} Jokers owned)",
            )

        # If Wraith held while rich
        if ckey == "c_wraith" and state.money > 10:
            return ConsumableAction(
                action_type="sell",
                rpc_method="sell",
                rpc_params={"consumable": idx},
                consumable_index=idx,
                card_name="Wraith",
                reason=f"Selling Wraith to prevent losing ${state.money}",
            )

    return None
