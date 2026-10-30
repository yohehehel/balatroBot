"""Mathematical hand evaluation and combat solver for Balatro (Silo 2).

Responsibilities:
- Generate all 1-to-5 card play combinations from hand.
- Detect exact poker hand type taking into account rule-modifying Jokers
  (Four Fingers, Shortcut, Smeared Joker, Splash, Pareidolia).
- Identify which played cards actually score.
- Compute predicted score using Balatro's exact scoring pipeline:
  Step A: Base chips & mult from state.hands (or defaults), adjusted for The Flint / boss debuffs.
  Step B: Played scoring cards (left to right): rank, enhancement, edition, on-score Jokers, retriggers.
  Step C: Held-in-hand cards (left to right): Steel, Baron, Shoot the Moon, Raised Fist, retriggers.
  Step D: Global Jokers (left to right): editions, base bonuses, multipliers, Blueprint/Brainstorm.
- Select best play: Kill Shot with card conservation or score maximisation.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from bots.state_parser import Card, ParsedGameState

if TYPE_CHECKING:
    pass


# ── Poker hand names & constants ──────────────────────────────────────────

class PokerHand:
    FLUSH_FIVE = "Flush Five"
    FLUSH_HOUSE = "Flush House"
    FIVE_OF_A_KIND = "Five of a Kind"
    STRAIGHT_FLUSH = "Straight Flush"
    FOUR_OF_A_KIND = "Four of a Kind"
    FULL_HOUSE = "Full House"
    FLUSH = "Flush"
    STRAIGHT = "Straight"
    THREE_OF_A_KIND = "Three of a Kind"
    TWO_PAIR = "Two Pair"
    PAIR = "Pair"
    HIGH_CARD = "High Card"


# Default base values for level 1 hands (fallback if not found in gamestate)
BASE_HAND_STATS: dict[str, tuple[int, int]] = {
    PokerHand.FLUSH_FIVE: (160, 16),
    PokerHand.FLUSH_HOUSE: (140, 14),
    PokerHand.FIVE_OF_A_KIND: (120, 12),
    PokerHand.STRAIGHT_FLUSH: (100, 8),
    PokerHand.FOUR_OF_A_KIND: (60, 7),
    PokerHand.FULL_HOUSE: (40, 4),
    PokerHand.FLUSH: (35, 4),
    PokerHand.STRAIGHT: (30, 4),
    PokerHand.THREE_OF_A_KIND: (30, 3),
    PokerHand.TWO_PAIR: (20, 2),
    PokerHand.PAIR: (10, 2),
    PokerHand.HIGH_CARD: (5, 1),
}

# Level up stats per poker hand: (l_chips, l_mult)
HAND_LEVEL_STATS: dict[str, tuple[int, int]] = {
    PokerHand.FLUSH_FIVE: (50, 3),
    PokerHand.FLUSH_HOUSE: (40, 4),
    PokerHand.FIVE_OF_A_KIND: (35, 3),
    PokerHand.STRAIGHT_FLUSH: (40, 4),
    PokerHand.FOUR_OF_A_KIND: (30, 3),
    PokerHand.FULL_HOUSE: (25, 2),
    PokerHand.FLUSH: (15, 2),
    PokerHand.STRAIGHT: (30, 3),
    PokerHand.THREE_OF_A_KIND: (20, 2),
    PokerHand.TWO_PAIR: (20, 1),
    PokerHand.PAIR: (15, 1),
    PokerHand.HIGH_CARD: (10, 1),
}

# Chips contributed by rank alone
RANK_CHIPS: dict[str, int] = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "10": 10,
    "J": 10,
    "Q": 10,
    "K": 10,
    "A": 11,
}

# Numeric comparison order for ranks (2 .. 14)
RANK_ORDER: dict[str, int] = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "10": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

# Standard suits
SUITS = ("H", "D", "C", "S")

# Fibonacci ranks
FIBONACCI_RANKS = frozenset({"A", "2", "3", "5", "8"})

# Even ranks
EVEN_RANKS = frozenset({"2", "4", "6", "8", "T"})

# Odd ranks
ODD_RANKS = frozenset({"3", "5", "7", "9", "A"})

# Uncommon Jokers in Balatro (for Baseball Card)
UNCOMMON_JOKERS: frozenset[str] = frozenset({
    "j_four_fingers", "j_mime", "j_ceremonial", "j_marble", "j_loyalty_card",
    "j_dusk", "j_fibonacci", "j_steel_joker", "j_hack", "j_pareidolia",
    "j_space", "j_burglar", "j_blackboard", "j_runner", "j_dna",
    "j_sixth_sense", "j_constellation", "j_hiker", "j_card_sharp", "j_madness",
    "j_square", "j_seance", "j_riff_raff", "j_vampire", "j_shortcut",
    "j_hologram", "j_vagabond", "j_cloud_9", "j_rocket", "j_midas_mask",
    "j_luchador", "j_gift", "j_turtle_bean", "j_erosion", "j_to_the_moon",
    "j_stone", "j_lucky_cat", "j_bull", "j_diet_cola", "j_trading",
    "j_flash", "j_trousers", "j_ancient", "j_ramen", "j_castle",
    "j_ticket", "j_acrobat", "j_sock_and_buskin", "j_troubadour", "j_certificate",
    "j_smeared", "j_throwback", "j_rough_gem", "j_bloodstone", "j_arrowhead",
    "j_onyx_agate", "j_glass", "j_ring_master", "j_seeing_double", "j_matador",
    "j_hit_the_road", "j_duo", "j_trio", "j_family", "j_order",
    "j_tribe", "j_invisible", "j_satellite", "j_shoot_the_moon", "j_drivers_license",
    "j_cartomancer", "j_astronomer", "j_burnt", "j_bootstraps", "j_baseball",
    "j_selzer",
})


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class HandCandidate:
    """A scored combination of cards to play."""

    indices: tuple[int, ...]
    cards: list[Card]
    hand_type: str
    scoring_cards: list[Card]
    base_chips: int
    base_mult: float
    final_chips: int
    final_mult: float
    score: int
    debuffed: bool = False
    conservation_score: float = 0.0

    @property
    def card_labels(self) -> list[str]:
        return [c.label or f"{c.value.suit}_{c.value.rank}" for c in self.cards]


# ── Helper functions for poker hand detection ─────────────────────────────

def _is_face_card(card: Card, pareidolia: bool = False) -> bool:
    """Return True if card is a face card (Jack, Queen, King or Pareidolia active)."""
    if card.modifier.enhancement == "STONE":
        return False
    if pareidolia:
        return True
    return card.value.rank in ("J", "Q", "K")


def _is_suit(card: Card, suit: str, smeared: bool = False) -> bool:
    """Check if a card matches target suit (handles Stone, Wild, and Smeared Joker)."""
    if card.modifier.enhancement == "STONE":
        return False
    if card.modifier.enhancement == "WILD" and not card.state.debuff:
        return True
    c_suit = card.value.suit
    if not c_suit:
        return False
    if smeared:
        # Hearts/Diamonds interchangeable, Spades/Clubs interchangeable
        if suit in ("H", "D"):
            return c_suit in ("H", "D")
        if suit in ("S", "C"):
            return c_suit in ("S", "C")
    return c_suit == suit


def _find_flush_cards(cards: list[Card], min_cards: int, smeared: bool) -> list[Card] | None:
    """Return the list of cards forming a flush if one exists, else None."""
    for suit in SUITS:
        matched = [c for c in cards if _is_suit(c, suit, smeared)]
        if len(matched) >= min_cards:
            return matched
    return None


def _find_straight_cards(
    cards: list[Card], min_cards: int, shortcut: bool
) -> list[Card] | None:
    """Return the list of cards forming a straight if one exists, else None.

    Matches Balatro's ``get_straight`` algorithm exactly:
    iterates rank id from 1 to 14, mapping 1 to Ace (for A-2-3-4-5 low straights).
    """
    # Group non-stone cards by rank id (2..14)
    ids: dict[int, list[Card]] = {}
    for c in cards:
        if c.modifier.enhancement == "STONE":
            continue
        r = c.value.rank
        if r and r in RANK_ORDER:
            val = RANK_ORDER[r]
            ids.setdefault(val, []).append(c)

    straight_cards: list[Card] = []
    straight_length = 0
    straight_found = False
    skipped_rank = False

    for j in range(1, 15):
        target_rank = 14 if j == 1 else j
        if target_rank in ids:
            straight_length += 1
            skipped_rank = False
            straight_cards.extend(ids[target_rank])
        elif shortcut and not skipped_rank and j != 14:
            skipped_rank = True
        else:
            straight_length = 0
            skipped_rank = False
            if not straight_found:
                straight_cards = []
            if straight_found:
                break

        if straight_length >= min_cards:
            straight_found = True

    if straight_found:
        return straight_cards
    return None


def identify_poker_hand(
    cards: Sequence[Card],
    joker_keys: set[str] | None = None,
) -> tuple[str, list[Card]]:
    """Identify the poker hand and the scoring cards from a played combination.

    Returns:
        (hand_name, scoring_cards)
        where scoring_cards are ordered left-to-right as they appear in ``cards``.
    """
    if not cards:
        return PokerHand.HIGH_CARD, []

    cards_list = list(cards)
    jkeys = joker_keys or set()
    four_fingers = "j_four_fingers" in jkeys
    shortcut = "j_shortcut" in jkeys
    smeared = "j_smeared" in jkeys
    splash = "j_splash" in jkeys

    req_len = 4 if four_fingers else 5

    # Group non-stone cards by rank
    rank_groups: dict[str, list[Card]] = {}
    stone_cards: list[Card] = []
    for c in cards_list:
        if c.modifier.enhancement == "STONE":
            stone_cards.append(c)
        elif c.value.rank:
            rank_groups.setdefault(c.value.rank, []).append(c)

    # Sort groups by count descending, then by rank order descending
    sorted_groups = sorted(
        rank_groups.values(),
        key=lambda g: (len(g), RANK_ORDER.get(g[0].value.rank or "", 0)),
        reverse=True,
    )
    counts = [len(g) for g in sorted_groups]

    # Flush & Straight candidates
    flush_candidate = _find_flush_cards(cards_list, req_len, smeared)
    straight_candidate = _find_straight_cards(cards_list, req_len, shortcut)

    hand_name: str
    scoring_subset: list[Card]

    # 1. Flush Five: 5 cards of same rank and all same suit
    if counts and counts[0] >= 5 and flush_candidate:
        # Check if the 5 same rank cards are also the flush cards
        top_group_ids = {id(c) for c in sorted_groups[0]}
        flush_ids = {id(c) for c in flush_candidate}
        if len(top_group_ids & flush_ids) >= 5:
            hand_name = PokerHand.FLUSH_FIVE
            scoring_subset = sorted_groups[0][:5]
        else:
            # Fallback to Five of a Kind
            hand_name = PokerHand.FIVE_OF_A_KIND
            scoring_subset = sorted_groups[0][:5]

    # 2. Flush House: 3 of a kind + pair and all 5 form a flush
    elif (
        len(counts) >= 2
        and counts[0] >= 3
        and counts[1] >= 2
        and flush_candidate
        and len(cards_list) >= 5
    ):
        fh_cards = sorted_groups[0][:3] + sorted_groups[1][:2]
        fh_ids = {id(c) for c in fh_cards}
        flush_ids = {id(c) for c in flush_candidate}
        if len(fh_ids & flush_ids) == 5:
            hand_name = PokerHand.FLUSH_HOUSE
            scoring_subset = fh_cards
        else:
            hand_name = PokerHand.FULL_HOUSE
            scoring_subset = fh_cards

    # 3. Five of a Kind
    elif counts and counts[0] >= 5:
        hand_name = PokerHand.FIVE_OF_A_KIND
        scoring_subset = sorted_groups[0][:5]

    # 4. Straight Flush
    elif flush_candidate and straight_candidate:
        # In Balatro: both parts exist, check if intersection of straight & flush has >= req_len
        s_ids = {id(c) for c in straight_candidate}
        sf_cards = [c for c in flush_candidate if id(c) in s_ids]
        if len(sf_cards) >= req_len:
            hand_name = PokerHand.STRAIGHT_FLUSH
            scoring_subset = sf_cards
        else:
            # If not pure straight flush, Flush takes priority over Straight
            hand_name = PokerHand.FLUSH
            scoring_subset = flush_candidate

    # 5. Four of a Kind
    elif counts and counts[0] >= 4:
        hand_name = PokerHand.FOUR_OF_A_KIND
        scoring_subset = sorted_groups[0][:4]

    # 6. Full House (3 of a kind + pair)
    elif len(counts) >= 2 and counts[0] >= 3 and counts[1] >= 2:
        hand_name = PokerHand.FULL_HOUSE
        scoring_subset = sorted_groups[0][:3] + sorted_groups[1][:2]

    # 7. Flush
    elif flush_candidate:
        hand_name = PokerHand.FLUSH
        scoring_subset = flush_candidate

    # 8. Straight
    elif straight_candidate:
        hand_name = PokerHand.STRAIGHT
        scoring_subset = straight_candidate

    # 9. Three of a Kind
    elif counts and counts[0] >= 3:
        hand_name = PokerHand.THREE_OF_A_KIND
        scoring_subset = sorted_groups[0][:3]

    # 10. Two Pair
    elif len(counts) >= 2 and counts[0] >= 2 and counts[1] >= 2:
        hand_name = PokerHand.TWO_PAIR
        scoring_subset = sorted_groups[0][:2] + sorted_groups[1][:2]

    # 11. Pair
    elif counts and counts[0] >= 2:
        hand_name = PokerHand.PAIR
        scoring_subset = sorted_groups[0][:2]

    # 12. High Card
    else:
        hand_name = PokerHand.HIGH_CARD
        if sorted_groups:
            # Highest nominal card
            scoring_subset = [sorted_groups[0][0]]
        elif stone_cards:
            scoring_subset = [stone_cards[0]]
        else:
            scoring_subset = []

    # Final scoring cards assembly:
    # 1. Base scoring cards from figure
    # 2. Add all Stone Cards played
    # 3. If Splash is active: EVERY played card scores!
    if splash:
        final_scoring = list(cards_list)
    else:
        scoring_id_set = {id(c) for c in scoring_subset}
        for sc in stone_cards:
            scoring_id_set.add(id(sc))
        # Keep original left-to-right order of played cards
        final_scoring = [c for c in cards_list if id(c) in scoring_id_set]

    return hand_name, final_scoring


# ── Scoring pipeline ──────────────────────────────────────────────────────

def _get_joker_keys(state: ParsedGameState) -> set[str]:
    """Return all non-debuffed Joker keys in the player's possession."""
    return {j.key for j in state.jokers.cards if not j.state.debuff and j.key}


def _is_boss_active(state: ParsedGameState) -> bool:
    """Return True if the current active blind is an enabled Boss Blind."""
    b = state.active_blind
    if not b or b.type.upper() != "BOSS":
        return False
    # Chicot disables all Boss Blind abilities
    jkeys = _get_joker_keys(state)
    if "j_chicot" in jkeys:
        return False
    return True


def _is_psychic_active(state: ParsedGameState) -> bool:
    """Return True if The Psychic boss blind is active and not disabled."""
    if not _is_boss_active(state):
        return False
    b = state.active_blind
    if not b:
        return False
    bname = (b.name or "").lower()
    bkey = (getattr(b, "key", None) or "").lower()
    return "psychic" in bname or "psychic" in bkey


def _is_ox_active(state: ParsedGameState) -> bool:
    """Return True if The Ox boss blind is active and not disabled."""
    if not _is_boss_active(state):
        return False
    b = state.active_blind
    if not b:
        return False
    bname = (b.name or "").lower()
    bkey = (getattr(b, "key", None) or "").lower()
    return "ox" in bname or "ox" in bkey


def _get_most_played_hand(state: ParsedGameState) -> str | None:
    """Return the name of the most played poker hand this run."""
    if not state.hands:
        return None
    max_p = 0
    most_played = None
    for hname, hinfo in state.hands.items():
        if hinfo.played > max_p:
            max_p = hinfo.played
            most_played = hname
    return most_played


def _ensure_five_cards_for_psychic(
    decision: CombatDecision,
    state: ParsedGameState,
) -> CombatDecision:
    """Guarantee that PLAY decisions against The Psychic have exactly 5 cards."""
    if decision.action != CombatAction.PLAY:
        return decision
    hand_cards = state.hand.cards
    if len(hand_cards) < 5:
        return decision
    if len(decision.indices) >= 5:
        return decision

    current_indices = list(decision.indices)
    needed = 5 - len(current_indices)
    available = [i for i in range(len(hand_cards)) if i not in set(current_indices)]
    # Pick kickers that are best to get rid of (highest discard priority)
    available.sort(key=lambda i: discard_card_priority(hand_cards[i], state), reverse=True)
    completed_indices = current_indices + available[:needed]
    completed_indices.sort()

    played = [hand_cards[i] for i in completed_indices]
    held = [hand_cards[i] for i in range(len(hand_cards)) if i not in set(completed_indices)]
    jkeys = _get_joker_keys(state)
    hand_type, scoring_cards = identify_poker_hand(played, jkeys)
    base_c, fin_c, fin_m, score, debuffed = calculate_hand_score(
        played, held, hand_type, scoring_cards, state
    )
    cons_val = sum(card_conservation_value(c) for c in held)

    cand = HandCandidate(
        indices=tuple(completed_indices),
        cards=played,
        hand_type=hand_type,
        scoring_cards=scoring_cards,
        base_chips=base_c,
        base_mult=fin_m,
        final_chips=fin_c,
        final_mult=fin_m,
        score=score,
        debuffed=debuffed,
        conservation_score=cons_val,
    )
    return CombatDecision(
        action=CombatAction.PLAY,
        indices=completed_indices,
        cards=played,
        reason=f"{decision.reason} [The Psychic: completed to 5 cards]",
        candidate=cand,
        predicted_score=score,
    )


def _get_joker_dynamic_mult(j: Card) -> float:
    """Extract dynamic/accumulated +Mult from card.value (mult or extra)."""
    # For scaling jokers whose current mult is tracked in card.ability.mult:
    # card.ability.extra is the increment rate (e.g. +3 for Red Card, +2 for Flash),
    # NEVER the accumulated mult. When mult is 0 (or None), their current mult is 0.0!
    if j.key in (
        "j_red_card",
        "j_flash",
        "j_ceremonial",
        "j_ride_the_bus",
        "j_green_joker",
        "j_trousers",
        "j_popcorn",
        "j_swashbuckler",
        "j_fortune_teller",
        "j_erosion",
    ):
        return float(j.value.mult) if j.value.mult is not None else 0.0

    if j.value.mult is not None:
        return float(j.value.mult)
    if isinstance(j.value.extra, (int, float)):
        return float(j.value.extra)
    if isinstance(j.value.extra, dict):
        m = (
            j.value.extra.get("mult")
            or j.value.extra.get("s_mult")
            or j.value.extra.get("bonus")
            or 0
        )
        if isinstance(m, (int, float)):
            return float(m)
    return 0.0


def _get_joker_dynamic_xmult(j: Card) -> float:
    """Extract dynamic/accumulated XMult from card.value (x_mult or extra)."""
    if j.key in (
        "j_constellation",
        "j_hologram",
        "j_madness",
        "j_campfire",
        "j_glass",
        "j_lucky_cat",
        "j_caino",
        "j_yorick",
        "j_throwback",
        "j_hit_the_road",
        "j_obelisk",
    ):
        return float(j.value.x_mult) if j.value.x_mult is not None else 1.0

    if j.value.x_mult is not None and j.value.x_mult > 1.0:
        return float(j.value.x_mult)
    if isinstance(j.value.extra, (int, float)) and j.value.extra > 1.0:
        return float(j.value.extra)
    if isinstance(j.value.extra, dict):
        xm = (
            j.value.extra.get("x_mult")
            or j.value.extra.get("xmult")
            or j.value.extra.get("Xmult")
            or 1.0
        )
        if isinstance(xm, (int, float)) and xm > 1.0:
            return float(xm)
    return 1.0


def _get_joker_dynamic_chips(j: Card) -> int:
    """Extract dynamic/accumulated Chips from card.value (chips or extra)."""
    if j.value.chips is not None:
        return int(j.value.chips)
    if isinstance(j.value.extra, dict):
        c = (
            j.value.extra.get("chips")
            or j.value.extra.get("chip")
            or j.value.extra.get("s_chips")
            or 0
        )
        if isinstance(c, (int, float)):
            return int(c)
    if isinstance(j.value.extra, (int, float)) and j.value.extra > 0:
        return int(j.value.extra)
    return 0


def _resolve_effective_joker(j: Card, idx: int, jokers: Sequence[Card]) -> Card | None:
    """Resolve Blueprint and Brainstorm copying behavior."""
    if j.state.debuff:
        return None
    eff = j
    if j.key == "j_blueprint":
        if idx + 1 < len(jokers):
            eff = jokers[idx + 1]
    elif j.key == "j_brainstorm":
        if jokers and jokers[0] is not j:
            eff = jokers[0]
    if eff.state.debuff:
        return None
    return eff


def calculate_hand_score(
    played_cards: Sequence[Card],
    held_cards: Sequence[Card],
    hand_type: str,
    scoring_cards: Sequence[Card],
    state: ParsedGameState,
) -> tuple[int, int, float, int, bool]:
    """Compute (base_chips, final_chips, final_mult, predicted_score, debuffed).

    Follows Balatro's exact execution sequence:
    Step A: Base chips & mult
    Step B: Scored cards (left to right)
    Step C: Held in hand cards (left to right)
    Step D: Global Jokers (left to right)
    """
    played = list(played_cards)
    held = list(held_cards)
    scoring = list(scoring_cards)

    boss_active = _is_boss_active(state)
    boss_name = state.active_blind.name if (boss_active and state.active_blind) else ""

    # Boss hand debuff checks (The Psychic, The Eye, The Mouth)
    if boss_active:
        if boss_name == "The Psychic" and len(played) < 5:
            return (0, 0, 0.0, 0, True)
        if boss_name == "The Eye":
            level_info = state.hands.get(hand_type)
            if level_info and level_info.played_this_round > 0:
                return (0, 0, 0.0, 0, True)
        if boss_name == "The Mouth":
            # Only 1 hand type allowed this round: if another was played, debuffed
            for name, hinfo in state.hands.items():
                if hinfo.played_this_round > 0 and name != hand_type:
                    return (0, 0, 0.0, 0, True)

    # ── Step A: Base Chips & Mult ──────────────────────────────────────────
    if hand_type in state.hands:
        base_chips = state.hands[hand_type].chips
        base_mult = float(state.hands[hand_type].mult)
    else:
        c, m = BASE_HAND_STATS.get(hand_type, (5, 1))
        base_chips = c
        base_mult = float(m)

    # Boss: The Flint (halves base chips and mult)
    if boss_active and boss_name == "The Flint":
        base_mult = float(max(math.floor(base_mult * 0.5 + 0.5), 1))
        base_chips = max(math.floor(base_chips * 0.5 + 0.5), 0)

    # Boss: The Arm (decreases level of played poker hand by 1 before scoring)
    if boss_active and boss_name == "The Arm" and hand_type in state.hands:
        lvl_info = state.hands[hand_type]
        if lvl_info.level > 1:
            l_chips, l_mult = HAND_LEVEL_STATS.get(hand_type, (15, 2))
            base_chips = max(0, base_chips - l_chips)
            base_mult = max(1.0, base_mult - float(l_mult))

    chips = base_chips
    mult = base_mult

    # Joker setup & key cache
    jokers = state.jokers.cards
    jkeys = _get_joker_keys(state)
    effective_jokers = [_resolve_effective_joker(j, i, jokers) for i, j in enumerate(jokers)]
    eff_keys = [ej.key for ej in effective_jokers if ej is not None]

    pareidolia = "j_pareidolia" in jkeys
    sock_and_buskin_count = sum(1 for k in eff_keys if k == "j_sock_and_buskin")
    hack_count = sum(1 for k in eff_keys if k == "j_hack")
    hanging_chad_count = sum(1 for k in eff_keys if k == "j_hanging_chad")
    dusk_count = sum(1 for k in eff_keys if k == "j_dusk") if state.round.hands_left == 1 else 0
    selzer_count = sum(1 for k in eff_keys if k == "j_selzer")

    # ── Step B: Scored cards (left to right) ────────────────────────────────
    first_face_scored = False

    for idx, card in enumerate(scoring):
        if card.state.debuff:
            continue

        is_face = _is_face_card(card, pareidolia)
        is_first_face = is_face and not first_face_scored
        if is_face:
            first_face_scored = True

        # Retrigger count
        reps = 1
        if card.modifier.seal == "RED":
            reps += 1
        if idx == 0 and hanging_chad_count > 0:
            reps += 2 * hanging_chad_count
        if is_face and sock_and_buskin_count > 0:
            reps += sock_and_buskin_count
        if card.value.rank in ("2", "3", "4", "5") and hack_count > 0:
            reps += hack_count
        reps += dusk_count
        reps += selzer_count

        for _ in range(reps):
            # Rank chips
            if card.modifier.enhancement == "STONE":
                chips += 50
            elif card.value.rank:
                chips += RANK_CHIPS.get(card.value.rank, 0)

            # Permanent bonus chips (e.g. from Hiker, Hierophant)
            if card.value.perma_bonus > 0:
                chips += card.value.perma_bonus

            # Enhancement
            enh = card.modifier.enhancement
            if enh == "BONUS":
                chips += 30
            elif enh == "MULT":
                mult += 4.0
            elif enh == "GLASS":
                mult *= 2.0
            # Lucky: conservative = 0 mult; expectation = +4 (20 * 0.2)
            # Using conservative 0 for deterministic floor guarantee

            # Edition
            edi = (card.modifier.edition or "").upper()
            if edi.startswith("E_"):
                edi = edi[2:]
            if edi == "FOIL":
                chips += 50
            elif edi == "HOLO":
                mult += 10.0
            elif edi == "POLYCHROME":
                mult *= 1.5

            # On-score Jokers (evaluated per card trigger)
            for eff_j in effective_jokers:
                if eff_j is None or eff_j.state.debuff:
                    continue
                eff_k = eff_j.key
                r = card.value.rank or ""

                if eff_k == "j_fibonacci" and r in FIBONACCI_RANKS:
                    mult += 8.0
                elif eff_k == "j_scholar" and r == "A":
                    chips += 20
                    mult += 4.0
                elif eff_k == "j_even_steven" and r in EVEN_RANKS:
                    mult += 4.0
                elif eff_k == "j_odd_todd" and r in ODD_RANKS:
                    chips += 31
                elif eff_k == "j_walkie_talkie" and r in ("T", "4"):
                    chips += 10
                    mult += 4.0
                elif eff_k == "j_scary_face" and is_face:
                    chips += 30
                elif eff_k == "j_smiley" and is_face:
                    mult += 5.0
                elif eff_k == "j_photograph" and is_first_face:
                    mult *= 2.0
                elif eff_k == "j_triboulet" and r in ("K", "Q"):
                    mult *= 2.0
                elif eff_k == "j_wee" and r == "2":
                    chips += 8
                elif eff_k == "j_hiker":
                    chips += 5
                elif eff_k == "j_greedy_joker" and _is_suit(card, "D", "j_smeared" in jkeys):
                    mult += 3.0
                elif eff_k == "j_lusty_joker" and _is_suit(card, "H", "j_smeared" in jkeys):
                    mult += 3.0
                elif eff_k == "j_wrathful_joker" and _is_suit(card, "S", "j_smeared" in jkeys):
                    mult += 3.0
                elif eff_k == "j_gluttenous_joker" and _is_suit(card, "C", "j_smeared" in jkeys):
                    mult += 3.0
                elif eff_k == "j_arrowhead" and _is_suit(card, "S", "j_smeared" in jkeys):
                    chips += 50
                elif eff_k == "j_onyx_agate" and _is_suit(card, "C", "j_smeared" in jkeys):
                    mult += 7.0
                elif eff_k == "j_bloodstone" and _is_suit(card, "H", "j_smeared" in jkeys):
                    # 1 in 2 chance (guaranteed 100% with Oops All 6s)
                    mult *= 1.5
                elif eff_k == "j_ancient":
                    target_suit = None
                    if isinstance(eff_j.value.extra, dict):
                        target_suit = eff_j.value.extra.get("suit")
                    elif isinstance(eff_j.value.extra, str):
                        target_suit = eff_j.value.extra
                    if target_suit:
                        st_letter = target_suit[0].upper()
                        if _is_suit(card, st_letter, "j_smeared" in jkeys):
                            mult *= 1.5
                elif eff_k == "j_idol":
                    target_rank = None
                    target_suit = None
                    if isinstance(eff_j.value.extra, dict):
                        target_rank = eff_j.value.extra.get("rank")
                        target_suit = eff_j.value.extra.get("suit")
                    if target_rank and target_suit:
                        rank_map = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J", "10": "T", "T": "T"}
                        norm_rank = rank_map.get(target_rank, target_rank)
                        st_letter = target_suit[0].upper()
                        if card.value.rank == norm_rank and _is_suit(card, st_letter, "j_smeared" in jkeys):
                            mult *= 2.0

    # ── Step C: Held in hand cards (left to right) ─────────────────────────
    mime_count = sum(1 for k in eff_keys if k == "j_mime")
    baron_count = sum(1 for k in eff_keys if k == "j_baron")
    shoot_moon_count = sum(1 for k in eff_keys if k == "j_shoot_the_moon")
    raised_fist_count = sum(1 for k in eff_keys if k == "j_raised_fist")

    for hcard in held:
        if hcard.state.debuff:
            continue
        hreps = 1
        if hcard.modifier.seal == "RED":
            hreps += 1
        hreps += mime_count

        for _ in range(hreps):
            if hcard.modifier.enhancement == "STEEL":
                mult *= 1.5
            if baron_count > 0 and hcard.value.rank == "K":
                mult *= (1.5 ** baron_count)
            if shoot_moon_count > 0 and hcard.value.rank == "Q":
                mult += 13.0 * shoot_moon_count

    # Raised Fist: 2x nominal rank of lowest non-debuff card held in hand
    if raised_fist_count > 0:
        non_debuff_held = [
            c for c in held
            if not c.state.debuff and c.value.rank and c.modifier.enhancement != "STONE"
        ]
        if non_debuff_held:
            lowest = min(non_debuff_held, key=lambda c: RANK_ORDER.get(c.value.rank or "", 99))
            lowest_nom = RANK_CHIPS.get(lowest.value.rank or "", 0)
            l_reps = 1 + (1 if lowest.modifier.seal == "RED" else 0) + mime_count
            mult += 2.0 * lowest_nom * raised_fist_count * l_reps

    # ── Step D: Global Jokers (strictly left to right) ─────────────────────
    num_jokers = len(jokers)
    discards_left = state.round.discards_left

    for i, j in enumerate(jokers):
        if j.state.debuff:
            continue

        # 1. Edition of the joker itself (Foil / Holo trigger before ability)
        j_edi = (j.modifier.edition or "").upper()
        if j_edi.startswith("E_"):
            j_edi = j_edi[2:]
        if j_edi == "FOIL":
            chips += 50
        elif j_edi == "HOLO":
            mult += 10.0

        eff_j = effective_jokers[i]
        if eff_j is None or eff_j.state.debuff:
            if j_edi == "POLYCHROME":
                mult *= 1.5
            continue
        k = eff_j.key

        # Chips Jokers
        if k == "j_banner":
            chips += 30 * discards_left
        elif k == "j_bull":
            chips += 2 * max(0, state.money)
        elif k == "j_ice_cream":
            dyn = _get_joker_dynamic_chips(eff_j)
            chips += dyn if dyn > 0 else (100 if eff_j.value.chips is None and eff_j.value.extra is None else 0)
        elif k == "j_stuntman":
            chips += 250
        elif k == "j_blue_joker":
            rem = state.deck_cards.count if state.deck_cards and state.deck_cards.count else len(getattr(state.deck_cards, "cards", []))
            chips += 2 * rem
        elif k == "j_stone":
            dyn = _get_joker_dynamic_chips(eff_j)
            if dyn > 0:
                chips += dyn
            else:
                deck_list = state.deck_cards.cards if state.deck_cards else []
                stone_cnt = sum(1 for c in deck_list if c.modifier.enhancement == "STONE")
                chips += 25 * stone_cnt
        elif k == "j_castle":
            chips += _get_joker_dynamic_chips(eff_j)
        elif k == "j_wee":
            chips += _get_joker_dynamic_chips(eff_j)
        elif k == "j_runner":
            c = _get_joker_dynamic_chips(eff_j)
            if "Straight" in hand_type:
                c += 15
            chips += c
        elif k == "j_square":
            c = _get_joker_dynamic_chips(eff_j)
            if len(played) == 4:
                c += 4
            chips += c
        elif k == "j_sly" and PokerHand.PAIR in hand_type:
            chips += 50
        elif k == "j_wily" and (
            PokerHand.THREE_OF_A_KIND in hand_type
            or hand_type in (PokerHand.FULL_HOUSE, PokerHand.FOUR_OF_A_KIND, PokerHand.FIVE_OF_A_KIND)
        ):
            chips += 100
        elif k == "j_clever" and (PokerHand.TWO_PAIR in hand_type or hand_type == PokerHand.FULL_HOUSE):
            chips += 80
        elif k == "j_devious" and "Straight" in hand_type:
            chips += 100
        elif k == "j_crafty" and "Flush" in hand_type:
            chips += 80

        # +Mult Jokers
        elif k == "j_joker":
            mult += 4.0
        elif k == "j_half" and len(played) <= 3:
            mult += 20.0
        elif k == "j_gros_michel":
            mult += 15.0
        elif k == "j_popcorn":
            dyn = _get_joker_dynamic_mult(eff_j)
            mult += dyn if dyn > 0 else (20.0 if eff_j.value.mult is None and eff_j.value.extra is None else 0.0)
        elif k == "j_abstract":
            mult += 3.0 * num_jokers
        elif k == "j_mystic_summit" and discards_left == 0:
            mult += 15.0
        elif k == "j_swashbuckler":
            dyn = _get_joker_dynamic_mult(eff_j)
            if dyn > 0:
                mult += dyn
            else:
                mult += sum(float(other_j.cost.sell) for other_j in jokers if other_j is not eff_j)
        elif k == "j_erosion":
            dyn = _get_joker_dynamic_mult(eff_j)
            if dyn > 0:
                mult += dyn
            else:
                total_cards = state.deck_cards.count if state.deck_cards and state.deck_cards.count else len(getattr(state.deck_cards, "cards", []))
                if total_cards and total_cards < 52:
                    mult += 4.0 * (52 - total_cards)
        elif k == "j_misprint":
            mult += 11.0
        elif k == "j_green_joker":
            mult += _get_joker_dynamic_mult(eff_j) + 1.0
        elif k == "j_ride_the_bus":
            has_scoring_face = any(_is_face_card(sc, pareidolia) for sc in scoring)
            if not has_scoring_face:
                mult += _get_joker_dynamic_mult(eff_j) + 1.0
        elif k == "j_trousers":
            m = _get_joker_dynamic_mult(eff_j)
            if PokerHand.TWO_PAIR in hand_type or hand_type == PokerHand.FULL_HOUSE:
                m += 2.0
            mult += m
        elif k == "j_fortune_teller":
            m = _get_joker_dynamic_mult(eff_j)
            if (m <= 1.0 or eff_j.value.mult is None) and state.consumeable_usage_total:
                tarot_cnt = state.consumeable_usage_total.get("tarot", 0)
                if tarot_cnt > 0:
                    m = float(tarot_cnt)
            mult += m
        elif k in ("j_red_card", "j_flash", "j_ceremonial"):
            mult += _get_joker_dynamic_mult(eff_j)
        elif k == "j_jolly" and PokerHand.PAIR in hand_type:
            mult += 8.0
        elif k == "j_zany" and (
            PokerHand.THREE_OF_A_KIND in hand_type
            or hand_type in (PokerHand.FULL_HOUSE, PokerHand.FOUR_OF_A_KIND, PokerHand.FIVE_OF_A_KIND)
        ):
            mult += 12.0
        elif k == "j_mad" and (PokerHand.TWO_PAIR in hand_type or hand_type == PokerHand.FULL_HOUSE):
            mult += 10.0
        elif k == "j_crazy" and "Straight" in hand_type:
            mult += 12.0
        elif k == "j_droll" and "Flush" in hand_type:
            mult += 10.0
        elif k == "j_supernova":
            lvl = state.hands.get(hand_type)
            mult += float((lvl.played + 1) if lvl else 1)
        elif k == "j_bootstraps":
            mult += float((max(0, state.money) // 5) * 2)

        # xMult Jokers
        elif k == "j_stencil":
            dyn = _get_joker_dynamic_xmult(eff_j)
            if dyn > 1.0:
                mult *= dyn
            else:
                empty_slots = max(0, state.jokers.limit - num_jokers)
                mult *= max(1.0, 1.0 + float(empty_slots))
        elif k == "j_cavendish":
            mult *= 3.0
        elif k == "j_card_sharp":
            lvl = state.hands.get(hand_type)
            if lvl and lvl.played_this_round > 0:
                mult *= 3.0
        elif k == "j_blackboard":
            # All cards held in hand must be Spades or Clubs (or Stone). Empty held also passes.
            if all(
                c.modifier.enhancement == "STONE" or c.value.suit in ("S", "C")
                for c in held
            ):
                mult *= 3.0
        elif k == "j_flower_pot":
            has_smeared = "j_smeared" in jkeys
            suits_found = {"H": 0, "D": 0, "S": 0, "C": 0}
            wild_cards = []
            for sc in scoring:
                if sc.state.debuff or sc.modifier.enhancement == "STONE":
                    continue
                if sc.modifier.enhancement == "WILD":
                    wild_cards.append(sc)
                else:
                    for s in ("H", "D", "S", "C"):
                        if _is_suit(sc, s, has_smeared) and suits_found[s] == 0:
                            suits_found[s] += 1
                            break
            for _ in wild_cards:
                for s in ("H", "D", "S", "C"):
                    if suits_found[s] == 0:
                        suits_found[s] += 1
                        break
            if all(v > 0 for v in suits_found.values()):
                mult *= 3.0
        elif k == "j_seeing_double":
            has_smeared = "j_smeared" in jkeys
            suits_found = {"H": 0, "D": 0, "S": 0, "C": 0}
            wild_cards = []
            for sc in scoring:
                if sc.state.debuff or sc.modifier.enhancement == "STONE":
                    continue
                if sc.modifier.enhancement == "WILD":
                    wild_cards.append(sc)
                else:
                    if _is_suit(sc, "C", has_smeared):
                        suits_found["C"] += 1
                    if _is_suit(sc, "H", has_smeared):
                        suits_found["H"] += 1
                    if _is_suit(sc, "D", has_smeared):
                        suits_found["D"] += 1
                    if _is_suit(sc, "S", has_smeared):
                        suits_found["S"] += 1
            for _ in wild_cards:
                if suits_found["C"] == 0:
                    suits_found["C"] += 1
                elif suits_found["D"] == 0:
                    suits_found["D"] += 1
                elif suits_found["S"] == 0:
                    suits_found["S"] += 1
                elif suits_found["H"] == 0:
                    suits_found["H"] += 1
            if suits_found["C"] > 0 and (suits_found["H"] > 0 or suits_found["D"] > 0 or suits_found["S"] > 0):
                mult *= 2.0
        elif k == "j_steel_joker":
            dyn = _get_joker_dynamic_xmult(eff_j)
            if dyn > 1.0:
                mult *= dyn
            else:
                deck_list = state.deck_cards.cards if state.deck_cards else []
                steel_cnt = sum(1 for c in deck_list if c.modifier.enhancement == "STEEL")
                if steel_cnt > 0:
                    mult *= (1.0 + 0.2 * steel_cnt)
        elif k == "j_drivers_license":
            deck_list = state.deck_cards.cards if state.deck_cards else []
            enh_cnt = sum(1 for c in deck_list if c.modifier.enhancement)
            if enh_cnt >= 16:
                mult *= 3.0
        elif k == "j_baseball":
            for other_j in jokers:
                if other_j is not eff_j and not other_j.state.debuff and other_j.key in UNCOMMON_JOKERS:
                    mult *= 1.5
        elif k == "j_ramen":
            dyn = _get_joker_dynamic_xmult(eff_j)
            mult *= dyn if dyn > 1.0 else (2.0 if eff_j.value.x_mult is None and eff_j.value.extra is None else max(1.0, 2.0 - 0.01 * (state.round.discards_used * 5)))
        elif k == "j_loyalty_card":
            if (isinstance(eff_j.value.extra, dict) and eff_j.value.extra.get("loyalty_remaining") == 0) or state.round.hands_played % 6 == 5:
                mult *= 4.0
        elif k == "j_vampire":
            xm = _get_joker_dynamic_xmult(eff_j)
            enh_count = sum(1 for sc in scoring if not sc.state.debuff and sc.modifier.enhancement and sc.modifier.enhancement != "STONE")
            mult *= (xm + 0.1 * enh_count)
        elif k == "j_obelisk":
            current_played = (state.hands[hand_type].played + 1) if hand_type in state.hands else 1
            max_other_played = max((h.played for name, h in state.hands.items() if name != hand_type), default=0)
            if current_played > max_other_played:
                mult *= 1.0
            else:
                mult *= (_get_joker_dynamic_xmult(eff_j) + 0.2)
        elif k in ("j_constellation", "j_hologram", "j_madness", "j_campfire", "j_glass", "j_lucky_cat", "j_caino", "j_yorick", "j_throwback", "j_hit_the_road"):
            mult *= _get_joker_dynamic_xmult(eff_j)
        elif k == "j_duo" and PokerHand.PAIR in hand_type:
            mult *= 2.0
        elif k == "j_trio" and (
            PokerHand.THREE_OF_A_KIND in hand_type
            or hand_type in (PokerHand.FULL_HOUSE, PokerHand.FOUR_OF_A_KIND, PokerHand.FIVE_OF_A_KIND)
        ):
            mult *= 3.0
        elif k == "j_family" and (
            PokerHand.FOUR_OF_A_KIND in hand_type or hand_type == PokerHand.FIVE_OF_A_KIND
        ):
            mult *= 4.0
        elif k == "j_order" and "Straight" in hand_type:
            mult *= 3.0
        elif k == "j_tribe" and "Flush" in hand_type:
            mult *= 2.0
        elif k == "j_acrobat" and state.round.hands_left == 1:
            mult *= 3.0

        # Polychrome triggers directly after the Joker's ability
        if j_edi == "POLYCHROME":
            mult *= 1.5

    total_score = math.floor(chips * mult)
    return (base_chips, chips, mult, total_score, False)


# ── Card conservation heuristic ───────────────────────────────────────────

def card_conservation_value(card: Card) -> float:
    """Assess the strategic value of keeping this card in hand for future turns."""
    val = 0.0
    seal = card.modifier.seal
    enh = card.modifier.enhancement
    edi = card.modifier.edition
    rank = card.value.rank

    # Seals
    if seal in ("RED", "BLUE", "PURPLE"):
        val += 25.0
    elif seal == "GOLD":
        val += 10.0

    # Enhancements
    if enh == "STEEL":
        val += 30.0  # Passively gives x1.5 in hand!
    elif enh == "GLASS":
        val += 20.0
    elif enh in ("GOLD", "LUCKY"):
        val += 15.0

    # Editions
    c_edi = (edi or "").upper()
    if c_edi.startswith("E_"):
        c_edi = c_edi[2:]
    if c_edi == "POLYCHROME":
        val += 15.0
    elif c_edi == "HOLO":
        val += 10.0
    elif c_edi == "FOIL":
        val += 8.0

    # High ranks (Aces / Kings / Queens for Baron, Shoot the Moon, or high chips)
    if rank in ("A", "K", "Q"):
        val += 5.0

    return val


# ── Main API: Candidate Generation & Selection ─────────────────────────────

def evaluate_all_plays(state: ParsedGameState) -> list[HandCandidate]:
    """Generate and score all 1-to-5 card combinations from the player's hand.

    Returns a list of :class:`HandCandidate` sorted by score descending.
    """
    hand_cards = state.hand.cards
    n = len(hand_cards)
    if n == 0:
        return []

    jkeys = _get_joker_keys(state)
    candidates: list[HandCandidate] = []

    # Max cards to play is bounded by 5 and hand highlighted_limit
    max_k = min(5, n, state.hand.highlighted_limit or 5)

    for k in range(1, max_k + 1):
        for combo_indices in itertools.combinations(range(n), k):
            played = [hand_cards[i] for i in combo_indices]
            held = [hand_cards[i] for i in range(n) if i not in combo_indices]

            hand_type, scoring_cards = identify_poker_hand(played, jkeys)
            base_c, fin_c, fin_m, score, debuffed = calculate_hand_score(
                played, held, hand_type, scoring_cards, state
            )

            # Sum conservation value of cards that remain in hand
            cons_val = sum(card_conservation_value(c) for c in held)

            candidates.append(
                HandCandidate(
                    indices=combo_indices,
                    cards=played,
                    hand_type=hand_type,
                    scoring_cards=scoring_cards,
                    base_chips=base_c,
                    base_mult=fin_m,
                    final_chips=fin_c,
                    final_mult=fin_m,
                    score=score,
                    debuffed=debuffed,
                    conservation_score=cons_val,
                )
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def select_best_play(state: ParsedGameState) -> HandCandidate | None:
    """Select the optimal play combination for the current state.

    Decision rules:
    1. If one or more plays beat the remaining chips needed:
       - Choose the winning play that uses the fewest cards (unless Blackboard activates 100% black held cards).
       - Protect economy against The Ox (avoid most played hand if safe alternative wins).
       - Break ties by preserving maximum conservation value in hand.
       - Further break ties by highest score.
    2. Otherwise, select the combination that maximises predicted score.
    """
    candidates = evaluate_all_plays(state)
    if not candidates:
        return None

    # Calculate remaining chips needed to defeat current blind
    active_blind = state.active_blind
    target_score = active_blind.score if active_blind else 300
    chips_needed = max(0, target_score - state.round.chips)

    hand_cards = state.hand.cards
    n = len(hand_cards)
    jkeys = _get_joker_keys(state)

    winning_candidates = [c for c in candidates if not c.debuffed and c.score >= chips_needed]

    # Mr. Bones safety net: on final hand, reaching >= 25% of target prevents death
    if not winning_candidates and state.round.hands_left == 1:
        if any(not j.state.debuff and j.key == "j_mr_bones" for j in state.jokers.cards):
            bones_threshold = max(0, math.ceil(target_score * 0.25) - state.round.chips)
            winning_candidates = [c for c in candidates if not c.debuffed and c.score >= bones_threshold]

    # The Ox avoidance: if money > 10, avoid resetting bankroll to $0 if alternative exists
    is_ox = _is_ox_active(state)
    most_played_hand = _get_most_played_hand(state) if is_ox else None
    avoid_ox = is_ox and most_played_hand is not None and state.money > 10

    if avoid_ox and winning_candidates:
        safe_winning = [c for c in winning_candidates if c.hand_type != most_played_hand]
        if safe_winning:
            winning_candidates = safe_winning

    if winning_candidates:
        has_blackboard = "j_blackboard" in jkeys

        def _winning_sort_key(c: HandCandidate) -> tuple:
            held_is_black = False
            if has_blackboard:
                held = [hand_cards[i] for i in range(n) if i not in set(c.indices)]
                held_is_black = all(
                    cd.modifier.enhancement == "STONE" or cd.value.suit in ("S", "C")
                    for cd in held
                )

            # Normally fewest cards played, but if Blackboard achieves 100% black held cards,
            # we do not penalize using extra kickers to purge red cards!
            card_count_penalty = 0 if held_is_black else len(c.indices)
            is_ox_hand = 1 if (avoid_ox and c.hand_type == most_played_hand) else 0

            return (
                is_ox_hand,
                -int(held_is_black),
                card_count_penalty,
                -c.conservation_score,
                -c.score,
            )

        winning_candidates.sort(key=_winning_sort_key)
        return winning_candidates[0]

    if avoid_ox and candidates:
        safe_candidates = [c for c in candidates if not c.debuffed and c.hand_type != most_played_hand]
        if safe_candidates:
            return safe_candidates[0]

    # No play wins in 1 shot: maximise predicted score
    return candidates[0]


# ── Silo 3: Combat Decision & Targeted Discard ────────────────────────────

class CombatAction:
    PLAY = "play"
    DISCARD = "discard"


@dataclass
class CombatDecision:
    """Represents a discrete combat decision: PLAY or DISCARD."""

    action: str  # "play" or "discard"
    indices: list[int]  # 0-based card indices in state.hand.cards
    cards: list[Card]  # The Card objects corresponding to indices
    reason: str  # Descriptive explanation for logs and telemetry
    candidate: HandCandidate | None = None  # Populated when action == PLAY
    predicted_score: int = 0  # Predicted score when action == PLAY, 0 when DISCARD


MAJOR_HANDS = frozenset({
    PokerHand.STRAIGHT,
    PokerHand.FLUSH,
    PokerHand.FULL_HOUSE,
    PokerHand.FOUR_OF_A_KIND,
    PokerHand.STRAIGHT_FLUSH,
    PokerHand.FIVE_OF_A_KIND,
    PokerHand.FLUSH_HOUSE,
    PokerHand.FLUSH_FIVE,
})


def get_debuffed_suits(state: ParsedGameState) -> set[str]:
    """Return the set of card suits currently debuffed by the active Boss Blind."""
    if not _is_boss_active(state):
        return set()
    b = state.active_blind
    if not b:
        return set()
    name = b.name
    debuffed: set[str] = set()
    if name == "The Club":
        debuffed.add("C")
    elif name == "The Goad":
        debuffed.add("S")
    elif name == "The Window":
        debuffed.add("D")
    elif name == "The Head":
        debuffed.add("H")
    return debuffed


def count_remaining_cards_by_suit(state: ParsedGameState, suit: str) -> int:
    """Count how many cards of a given suit remain unplayed in the deck."""
    if state.deck_cards and state.deck_cards.cards:
        return sum(1 for c in state.deck_cards.cards if c.value.suit == suit)
    # Estimate based on a standard 52-card deck minus cards currently in hand
    in_hand = sum(1 for c in state.hand.cards if c.value.suit == suit)
    return max(0, 13 - in_hand)


def count_remaining_cards_by_rank(state: ParsedGameState, rank: str) -> int:
    """Count how many cards of a given rank remain unplayed in the deck."""
    if state.deck_cards and state.deck_cards.cards:
        return sum(1 for c in state.deck_cards.cards if c.value.rank == rank)
    # Estimate based on standard 4 copies per rank minus cards currently in hand
    in_hand = sum(1 for c in state.hand.cards if c.value.rank == rank)
    return max(0, 4 - in_hand)


def discard_card_priority(card: Card, state: ParsedGameState) -> float:
    """Calculate the priority of discarding this card.

    Higher value -> Higher priority to discard (discard first).
    Negative value -> Strongly protected, keep in hand (do not discard).
    """
    seal = (card.modifier.seal or "").upper()
    enh = (card.modifier.enhancement or "").upper()
    edi = (card.modifier.edition or "").upper()
    rank = card.value.rank or ""
    suit = card.value.suit or ""

    # 1. Strongly protected cards (negative priority = KEEP in hand)
    if enh in ("STEEL", "M_STEEL"):
        return -1000.0  # Passively grants x1.5 Mult held in hand
    if seal == "BLUE":
        return -900.0  # Creates a Planet card at round end if held in hand
    if enh in ("GOLD", "M_GOLD"):
        return -500.0  # Gives $3 at round end held in hand
    if edi in ("POLYCHROME", "HOLO", "FOIL"):
        base_priority = -50.0
    else:
        base_priority = 0.0

    # 2. High priority discard triggers
    if seal == "PURPLE":
        return 1000.0  # Creates a Tarot card immediately when discarded!

    jkeys = _get_joker_keys(state)

    # Joker synergies
    if "j_hit_the_road" in jkeys and rank == "J":
        base_priority += 200.0
    if ("j_faceless" in jkeys or "j_faceless_joker" in jkeys) and _is_face_card(card, "j_pareidolia" in jkeys):
        base_priority += 60.0
    if "j_mail" in jkeys:
        for j in state.jokers.cards:
            if not j.state.debuff and j.key == "j_mail":
                target_r = j.value.extra.get("rank") if isinstance(j.value.extra, dict) else (j.value.extra if isinstance(j.value.extra, str) else None)
                if target_r:
                    rank_map = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J", "10": "T", "T": "T"}
                    norm_target_r = rank_map.get(target_r, target_r)
                    if rank == norm_target_r:
                        base_priority += 200.0

    # Blackboard synergy: x3 Mult if all held cards are Spades or Clubs
    if "j_blackboard" in jkeys:
        if suit in ("H", "D"):
            base_priority += 300.0  # Strongly discard Red cards to keep only Black cards in hand
        else:
            base_priority -= 150.0  # Preserve Spades and Clubs

    # Suit-specific Jokers synergies
    if "j_bloodstone" in jkeys or "j_lusty_joker" in jkeys:
        if suit != "H":
            base_priority += 80.0
        else:
            base_priority -= 50.0
    if "j_arrowhead" in jkeys or "j_wrathful_joker" in jkeys:
        if suit != "S":
            base_priority += 80.0
        else:
            base_priority -= 50.0
    if "j_onyx_agate" in jkeys or "j_gluttonous_joker" in jkeys:
        if suit != "C":
            base_priority += 80.0
        else:
            base_priority -= 50.0
    if "j_rough_gem" in jkeys or "j_greedy_joker" in jkeys:
        if suit != "D":
            base_priority += 80.0
        else:
            base_priority -= 50.0

    # Face card jokers synergy (Photograph, Smiley Face, Scary Face, Triboulet)
    has_face_synergy = any(k in jkeys for k in ("j_photograph", "j_smiley", "j_scary_face", "j_triboulet"))
    if has_face_synergy:
        is_face = _is_face_card(card, "j_pareidolia" in jkeys)
        if not is_face:
            base_priority += 40.0
        else:
            base_priority -= 30.0

    # Rank protection synergies (Wee Joker, Hack, Fibonacci)
    if "j_wee" in jkeys and rank == "2":
        base_priority -= 100.0
    if "j_fibonacci" in jkeys and rank in ("A", "2", "3", "5", "8"):
        base_priority -= 40.0
    if "j_hack" in jkeys and rank in ("2", "3", "4", "5"):
        base_priority -= 40.0

    # Debuffed cards are useless for scoring -> discard priority boost
    debuffed_suits = get_debuffed_suits(state)
    if card.state.debuff or (suit in debuffed_suits):
        base_priority += 100.0

    # Lower rank cards have less chip value -> prefer discarding them
    # Rank order: 2 (lowest) gives +13, Ace (highest) gives +1
    rank_val = RANK_ORDER.get(rank, 0)
    base_priority += float(15 - rank_val)

    return base_priority


def pick_discard_indices(
    state: ParsedGameState,
    keep_indices: set[int],
    max_count: int = 5,
) -> list[int]:
    """Select between 1 and 5 card indices to discard from hand.

    Applies smart filtering:
    - Excludes protected cards (Steel, Blue Seal, Gold) whenever other cards can be discarded.
    - Prioritises Purple Seal, Jokers synergies, and low/debuffed cards.
    - Strictly returns between 1 and 5 unique indices (never empty, never > 5).
    """
    hand_cards = state.hand.cards
    candidate_indices = [i for i in range(len(hand_cards)) if i not in keep_indices]

    if not candidate_indices:
        # If all cards were protected/kept, pick the least valuable card from all hand cards
        candidate_indices = list(range(len(hand_cards)))

    # Sort candidates by discard_card_priority descending (highest = discard first)
    candidate_indices.sort(
        key=lambda i: discard_card_priority(hand_cards[i], state),
        reverse=True,
    )

    # Filter out strongly protected cards (priority < 0, like Steel/Blue Seal) if we have enough unprotected cards
    unprotected = [i for i in candidate_indices if discard_card_priority(hand_cards[i], state) >= 0]

    if unprotected:
        selected = unprotected[:max_count]
    else:
        # All candidates are protected: discard the least protected one
        selected = candidate_indices[:max_count]

    if not selected and candidate_indices:
        selected = [candidate_indices[0]]

    return selected[:5]


def add_kickers_for_recycle(
    state: ParsedGameState,
    candidate: HandCandidate,
) -> list[int]:
    """Add up to 5 cards (candidate scoring + weak kickers) to cycle hand when no discards remain."""
    hand_cards = state.hand.cards
    current_indices = list(candidate.indices)
    current_set = set(current_indices)

    available_indices = [i for i in range(len(hand_cards)) if i not in current_set]
    available_indices.sort(
        key=lambda i: discard_card_priority(hand_cards[i], state),
        reverse=True,
    )

    needed = min(5 - len(current_indices), len(available_indices))
    kickers: list[int] = []

    # First pass: only take non-protected cards
    for idx in available_indices:
        if len(kickers) >= needed:
            break
        c = hand_cards[idx]
        enh = (c.modifier.enhancement or "").upper()
        seal = (c.modifier.seal or "").upper()
        if enh in ("STEEL", "M_STEEL") or seal == "BLUE":
            continue
        kickers.append(idx)

    # Second pass: if boss is The Psychic (requires 5 cards) and we need more cards
    boss_name = state.active_blind.name if (state.active_blind and _is_boss_active(state)) else ""
    if boss_name == "The Psychic" and len(current_indices) + len(kickers) < 5:
        for idx in available_indices:
            if len(kickers) >= needed:
                break
            if idx not in kickers:
                kickers.append(idx)

    return sorted(current_indices + kickers)


def _find_flush_draw_cards(
    state: ParsedGameState,
    smeared: bool,
    debuffed_suits: set[str],
) -> tuple[list[int], str] | None:
    """Find a flush draw (4 cards, or 3 cards if discards_left >= 2 & high deck density).

    Returns (matching_card_indices, suit) or None.
    """
    hand_cards = state.hand.cards
    discards_left = state.round.discards_left
    best_candidate: tuple[list[int], str] | None = None
    best_len = 0
    best_outs = 0

    jkeys = _get_joker_keys(state)
    preferred_suits: set[str] = set()
    if "j_blackboard" in jkeys:
        preferred_suits.update(["S", "C"])
    if "j_bloodstone" in jkeys or "j_lusty_joker" in jkeys:
        preferred_suits.add("H")
    if "j_arrowhead" in jkeys or "j_wrathful_joker" in jkeys:
        preferred_suits.add("S")
    if "j_onyx_agate" in jkeys or "j_gluttonous_joker" in jkeys:
        preferred_suits.add("C")
    if "j_rough_gem" in jkeys or "j_greedy_joker" in jkeys:
        preferred_suits.add("D")

    for s in SUITS:
        if s in debuffed_suits:
            continue

        matching = [
            i for i, c in enumerate(hand_cards)
            if _is_suit(c, s, smeared) and not c.state.debuff
        ]
        count = len(matching)
        outs = count_remaining_cards_by_suit(state, s)
        is_pref = s in preferred_suits
        score_weight = count * 100 + outs + (50 if is_pref else 0)

        # 4-card flush draw: always qualifies
        if count == 4:
            if best_len < 4 or score_weight > best_outs:
                best_len = count
                best_outs = score_weight
                best_candidate = (matching, s)
        # 3-card flush draw: requires >= 2 discards and high density in deck (>= 7 outs)
        elif count == 3 and discards_left >= 2 and outs >= 7:
            if best_len < 4 and score_weight > best_outs:
                best_len = count
                best_outs = score_weight
                best_candidate = (matching, s)

    return best_candidate


def _is_straight_id_sequence(ids: list[int], shortcut: bool) -> bool:
    """Check if a sorted list of 5 rank IDs forms a valid straight."""
    if len(ids) != 5:
        return False
    if shortcut:
        return all(1 <= ids[i + 1] - ids[i] <= 2 for i in range(4))
    return all(ids[i + 1] - ids[i] == 1 for i in range(4))


def _find_straight_draw_cards(
    state: ParsedGameState,
    shortcut: bool,
) -> tuple[list[int], int] | None:
    """Find a 4-card straight draw in hand having >= 2 outs in remaining deck.

    Returns (matching_card_indices, total_outs) or None.
    """
    hand_cards = state.hand.cards
    valid: list[tuple[int, Card]] = [
        (i, c) for i, c in enumerate(hand_cards)
        if c.value.rank in RANK_ORDER and (c.modifier.enhancement or "").upper() not in ("STONE", "M_STONE")
    ]
    if len(valid) < 4:
        return None

    # Group by rank
    by_rank: dict[str, list[int]] = {}
    for i, c in valid:
        r = c.value.rank
        if r:
            by_rank.setdefault(r, []).append(i)

    distinct_ranks = list(by_rank.keys())
    if len(distinct_ranks) < 4:
        return None

    best_draw: tuple[list[int], int] | None = None
    max_outs = 0

    # Test all 4-rank subsets
    for r4 in itertools.combinations(distinct_ranks, 4):
        r4_indices = [by_rank[r][0] for r in r4]
        base_ids = [RANK_ORDER[r] for r in r4]

        # Check missing ranks 1..14 that can complete the straight
        outs_ranks: set[str] = set()
        for missing_id in range(1, 15):
            if missing_id in base_ids or (missing_id == 1 and 14 in base_ids):
                continue

            test_ids = sorted(base_ids + [missing_id])
            variants = [test_ids]
            if 14 in test_ids:
                ace_low_ids = sorted([1 if x == 14 else x for x in test_ids])
                variants.append(ace_low_ids)

            if any(_is_straight_id_sequence(v, shortcut) for v in variants):
                if missing_id in (1, 14):
                    r_str = "A"
                elif missing_id == 10:
                    r_str = "T"
                elif missing_id == 11:
                    r_str = "J"
                elif missing_id == 12:
                    r_str = "Q"
                elif missing_id == 13:
                    r_str = "K"
                else:
                    r_str = str(missing_id)
                outs_ranks.add(r_str)

        total_outs = sum(count_remaining_cards_by_rank(state, r) for r in outs_ranks)
        if total_outs >= 2 and total_outs > max_outs:
            max_outs = total_outs
            best_draw = (r4_indices, total_outs)

    return best_draw


def _find_pair_or_set_draw(
    state: ParsedGameState,
) -> tuple[list[int], str] | None:
    """Find 3OAK, Two Pair, or Pair to keep while discarding remaining cards."""
    hand_cards = state.hand.cards
    by_rank: dict[str, list[int]] = {}
    for i, c in enumerate(hand_cards):
        r = c.value.rank
        if r and (c.modifier.enhancement or "").upper() not in ("STONE", "M_STONE"):
            by_rank.setdefault(r, []).append(i)

    # Three of a Kind
    three_of_a_kinds = [indices for r, indices in by_rank.items() if len(indices) == 3]
    if three_of_a_kinds:
        return (three_of_a_kinds[0], PokerHand.THREE_OF_A_KIND)

    # Two Pair
    pairs = [indices for r, indices in by_rank.items() if len(indices) >= 2]
    if len(pairs) >= 2:
        return (pairs[0] + pairs[1], PokerHand.TWO_PAIR)

    # Pair
    if pairs:
        pairs.sort(key=lambda idxs: RANK_ORDER.get(hand_cards[idxs[0]].value.rank or "", 0), reverse=True)
        return (pairs[0], PokerHand.PAIR)

    return None


def _find_fallback_keep_cards(
    state: ParsedGameState,
    debuffed_suits: set[str],
) -> list[int]:
    """Fallback: keep 2-3 best cards of the most represented non-debuffed suit."""
    hand_cards = state.hand.cards
    if not hand_cards:
        return []

    # Non-debuffed suits count in remaining deck
    deck_counts = {s: count_remaining_cards_by_suit(state, s) for s in SUITS if s not in debuffed_suits}
    best_suit = max(deck_counts.keys(), key=lambda s: deck_counts[s], default="H")

    suit_cards = [
        (i, c) for i, c in enumerate(hand_cards)
        if c.value.suit == best_suit and not c.state.debuff
    ]
    if suit_cards:
        suit_cards.sort(key=lambda item: RANK_ORDER.get(item[1].value.rank or "", 0), reverse=True)
        return [i for i, _ in suit_cards[:3]]

    # If no suit cards, keep top 2 highest rank cards
    ranked = [(i, c) for i, c in enumerate(hand_cards) if not c.state.debuff]
    ranked.sort(key=lambda item: RANK_ORDER.get(item[1].value.rank or "", 0), reverse=True)
    return [i for i, _ in ranked[:2]]


def decide_combat_action(state: ParsedGameState) -> CombatDecision:
    """Determine the optimal combat action (PLAY vs DISCARD) for the current hand.

    Implements the 5-rule decision tree in strict priority order:
    1. Immediate Kill Shot: Play winning hand (using card conservation).
    2. Zero discards left / Discards blocked: Play (recycle with kickers if hand is small & weak).
    3. Excellent hand already held (Tempo): Play major hand or pace-covering hand.
    4. Targeted Discards: Trading Card, Flush Draw, Straight Draw, Pair/Set draw, or Fallback.
    5. Smart Discard Priority: Protect Steel/Blue Seal/Gold, prioritize Purple Seal/Joker triggers.
    """
    hand_cards = state.hand.cards
    if not hand_cards:
        return CombatDecision(
            action=CombatAction.PLAY,
            indices=[],
            cards=[],
            reason="No cards in hand",
        )

    active_blind = state.active_blind
    target_score = active_blind.score if active_blind else 300
    chips_needed = max(0, target_score - state.round.chips)
    hands_left = max(1, state.round.hands_left)
    discards_left = state.round.discards_left
    required_pace = math.ceil(chips_needed / hands_left)

    best_play = select_best_play(state)
    jkeys = _get_joker_keys(state)

    # ── Rule 1: Immediate Kill Shot (Guaranteed win on this play) ────────────
    if best_play and not best_play.debuffed and best_play.score >= chips_needed:
        dec = CombatDecision(
            action=CombatAction.PLAY,
            indices=list(best_play.indices),
            cards=best_play.cards,
            reason=f"Kill Shot - Predicted {best_play.score} >= {chips_needed} needed",
            candidate=best_play,
            predicted_score=best_play.score,
        )
        if _is_psychic_active(state):
            dec = _ensure_five_cards_for_psychic(dec, state)
        return dec

    # ── Rule 1.B: Absolute Survival Law on Final Hand (hands_left == 1 / The Needle) ─
    # If only 1 hand remains, playing a hand that does NOT reach chips_needed guarantees immediate DEFEAT.
    # If discards remain, it is strictly forbidden to play: we MUST discard to dig for a win or Blackboard purge!
    final_hand_survival = (
        hands_left == 1
        and discards_left > 0
        and (not best_play or best_play.debuffed or best_play.score < chips_needed)
    )

    # ── Rule 2: Zero discards left or discards blocked ──────────────────────
    has_anti_discard = ("j_green_joker" in jkeys) or ("j_ramen" in jkeys)
    anti_discard_blocking = has_anti_discard and best_play and (best_play.score >= required_pace)

    if not final_hand_survival and (discards_left <= 0 or anti_discard_blocking):
        if (
            best_play
            and hands_left > 1
            and best_play.score < chips_needed
            and len(best_play.cards) < 5
            and best_play.score < required_pace
        ):
            recycle_indices = add_kickers_for_recycle(state, best_play)
            dec = CombatDecision(
                action=CombatAction.PLAY,
                indices=recycle_indices,
                cards=[state.hand.cards[i] for i in recycle_indices],
                reason=f"Recycle Play - No discards left, played {best_play.hand_type} + kickers to cycle hand",
                candidate=best_play,
                predicted_score=best_play.score,
            )
            if _is_psychic_active(state):
                dec = _ensure_five_cards_for_psychic(dec, state)
            return dec

        block_reason = "Anti-discard Joker active" if anti_discard_blocking else "No discards left"
        play_candidate = best_play or HandCandidate(
            indices=(0,),
            cards=[hand_cards[0]],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[hand_cards[0]],
            base_chips=5,
            base_mult=1.0,
            final_chips=5,
            final_mult=1.0,
            score=5,
        )
        dec = CombatDecision(
            action=CombatAction.PLAY,
            indices=list(play_candidate.indices),
            cards=play_candidate.cards,
            reason=f"Forced Play - {block_reason} (Score: {play_candidate.score}, Hands: {state.round.hands_left})",
            candidate=play_candidate,
            predicted_score=play_candidate.score,
        )
        if _is_psychic_active(state):
            dec = _ensure_five_cards_for_psychic(dec, state)
        return dec

    # ── Rule 3: Current hand already excellent (Tempo sufficient) ───────────
    is_ox = _is_ox_active(state)
    most_played_hand = _get_most_played_hand(state) if is_ox else None
    avoid_ox = is_ox and most_played_hand is not None and state.money > 10

    if not final_hand_survival and best_play and not best_play.debuffed:
        is_major_hand = best_play.hand_type in MAJOR_HANDS
        has_tempo_score = best_play.score >= required_pace * 1.25
        ox_penalty_hit = avoid_ox and best_play.hand_type == most_played_hand

        if (is_major_hand or has_tempo_score) and not (ox_penalty_hit and discards_left > 0):
            reason = (
                f"Tempo Play - Major hand ({best_play.hand_type}) with score {best_play.score}"
                if is_major_hand
                else f"Tempo Play - Score {best_play.score} covers pace ({required_pace * 1.25:.0f})"
            )
            dec = CombatDecision(
                action=CombatAction.PLAY,
                indices=list(best_play.indices),
                cards=best_play.cards,
                reason=reason,
                candidate=best_play,
                predicted_score=best_play.score,
            )
            if _is_psychic_active(state):
                dec = _ensure_five_cards_for_psychic(dec, state)
            return dec

    # ── Rule 4: Targeted Discard Heuristics (discards_left > 0 & weak hand) ─
    # Special case: Trading Card (j_trading) on first discard of the round
    if "j_trading" in jkeys and state.round.discards_used == 0:
        plain_cards = [
            i for i, c in enumerate(hand_cards)
            if not c.modifier.seal
            and (c.modifier.enhancement or "").upper() not in ("STEEL", "M_STEEL", "GOLD", "M_GOLD", "GLASS", "M_GLASS")
            and not c.modifier.edition
        ]
        # Sort by rank ascending to destroy the lowest rank card
        plain_cards.sort(key=lambda i: RANK_ORDER.get(hand_cards[i].value.rank or "", 0))
        target_idx = plain_cards[0] if plain_cards else 0
        target_card = hand_cards[target_idx]
        return CombatDecision(
            action=CombatAction.DISCARD,
            indices=[target_idx],
            cards=[target_card],
            reason=f"Trading Card - Discarding 1 weak card ({target_card.value.rank}{target_card.value.suit}) to destroy for $3",
            predicted_score=0,
        )

    smeared = "j_smeared" in jkeys
    shortcut = "j_shortcut" in jkeys
    debuffed_suits = get_debuffed_suits(state)

    # 4.A Flush Draw
    flush_draw = _find_flush_draw_cards(state, smeared, debuffed_suits)
    if flush_draw:
        kept_indices, suit = flush_draw
        discard_indices = pick_discard_indices(state, set(kept_indices), max_count=5)
        if discard_indices:
            return CombatDecision(
                action=CombatAction.DISCARD,
                indices=discard_indices,
                cards=[hand_cards[i] for i in discard_indices],
                reason=f"Flush Draw ({suit}) - {len(kept_indices)}/5 kept, discarding {len(discard_indices)}",
                predicted_score=0,
            )

    # 4.B Straight Draw
    straight_draw = _find_straight_draw_cards(state, shortcut)
    if straight_draw:
        kept_indices, outs = straight_draw
        discard_indices = pick_discard_indices(state, set(kept_indices), max_count=5)
        if discard_indices:
            return CombatDecision(
                action=CombatAction.DISCARD,
                indices=discard_indices,
                cards=[hand_cards[i] for i in discard_indices],
                reason=f"Straight Draw - {len(kept_indices)}/5 kept ({outs} outs), discarding {len(discard_indices)}",
                predicted_score=0,
            )

    # 4.C Pair / Set Draw
    pair_set = _find_pair_or_set_draw(state)
    if pair_set:
        kept_indices, htype = pair_set
        discard_indices = pick_discard_indices(state, set(kept_indices), max_count=5)
        if discard_indices:
            return CombatDecision(
                action=CombatAction.DISCARD,
                indices=discard_indices,
                cards=[hand_cards[i] for i in discard_indices],
                reason=f"Pair/Set Draw ({htype}) - {len(kept_indices)} kept, discarding {len(discard_indices)}",
                predicted_score=0,
            )

    # 4.D Default Discard Fallback
    fallback_kept = _find_fallback_keep_cards(state, debuffed_suits)
    discard_indices = pick_discard_indices(state, set(fallback_kept), max_count=5)
    if discard_indices:
        return CombatDecision(
            action=CombatAction.DISCARD,
            indices=discard_indices,
            cards=[hand_cards[i] for i in discard_indices],
            reason=f"Fallback Discard - {len(fallback_kept)} kept, discarding {len(discard_indices)}",
            predicted_score=0,
        )

    # Ultimate fallback: play best available combination, UNLESS we are in final_hand_survival!
    if discards_left > 0 and (final_hand_survival or (best_play and best_play.score < chips_needed and hands_left == 1)):
        force_discard = pick_discard_indices(state, set(), max_count=min(5, len(hand_cards)))
        if force_discard:
            return CombatDecision(
                action=CombatAction.DISCARD,
                indices=force_discard,
                cards=[hand_cards[i] for i in force_discard],
                reason=f"Survival Discard - Final hand insufficient score ({best_play.score if best_play else 0} < {chips_needed})",
                predicted_score=0,
            )

    fallback_play = best_play or HandCandidate(
        indices=(0,),
        cards=[hand_cards[0]],
        hand_type=PokerHand.HIGH_CARD,
        scoring_cards=[hand_cards[0]],
        base_chips=5,
        base_mult=1.0,
        final_chips=5,
        final_mult=1.0,
        score=5,
    )
    dec = CombatDecision(
        action=CombatAction.PLAY,
        indices=list(fallback_play.indices),
        cards=fallback_play.cards,
        reason=f"Fallback Play - No discard possible ({fallback_play.hand_type})",
        candidate=fallback_play,
        predicted_score=fallback_play.score,
    )
    if _is_psychic_active(state):
        dec = _ensure_five_cards_for_psychic(dec, state)
    return dec
