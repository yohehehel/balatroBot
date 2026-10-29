"""Unit tests for Silo 2 – Combat Solver (bots/combat_solver.py).

Test coverage:
1. Poker hand identification: Pair, Two Pair, Straight (low A and high A),
   Flush, Full House, Four of a Kind, Straight Flush, Four Fingers, Shortcut, Smeared, Splash.
2. Exact score calculation without Jokers (base, enhancements, editions).
3. Held-in-hand card effects (Steel x1.5, Baron, Shoot the Moon, Mime, Raised Fist).
4. Joker ordering (strictly left-to-right: +Mult before xMult vs xMult before +Mult, Blueprint, Brainstorm).
5. Debuffed cards behavior and Kill Shot card-conservation heuristic.
"""

from __future__ import annotations

import pytest

from bots.combat_solver import (
    PokerHand,
    calculate_hand_score,
    card_conservation_value,
    identify_poker_hand,
    select_best_play,
)
from bots.state_parser import (
    AreaInfo,
    BlindInfo,
    Card,
    CardCost,
    CardModifier,
    CardState,
    CardValue,
    HandLevel,
    ParsedGameState,
    RoundInfo,
)

# ── Helpers for test card creation ─────────────────────────────────────────

def _make_card(
    rank: str,
    suit: str,
    enhancement: str | None = None,
    edition: str | None = None,
    seal: str | None = None,
    debuff: bool = False,
    card_id: int = 1,
) -> Card:
    return Card(
        id=card_id,
        key=f"{suit}_{rank}",
        label=f"{rank} of {suit}",
        value=CardValue(suit=suit, rank=rank),
        modifier=CardModifier(
            enhancement=enhancement,
            edition=edition,
            seal=seal,
        ),
        state=CardState(debuff=debuff),
    )


def _make_joker(
    key: str,
    edition: str | None = None,
    debuff: bool = False,
    joker_id: int = 100,
    sell_cost: int = 1,
) -> Card:
    return Card(
        id=joker_id,
        key=key,
        label=key,
        modifier=CardModifier(edition=edition),
        state=CardState(debuff=debuff),
        cost=CardCost(sell=sell_cost),
    )


def _make_base_state(
    hand_cards: list[Card] | None = None,
    jokers: list[Card] | None = None,
    hands_played: int = 0,
    discards_left: int = 3,
    hands_left: int = 4,
    chips: int = 0,
    blind_score: int = 300,
    blind_name: str = "Small Blind",
    blind_type: str = "SMALL",
) -> ParsedGameState:
    return ParsedGameState(
        state="SELECTING_HAND",
        round_num=1,
        ante_num=1,
        money=4,
        round=RoundInfo(
            hands_left=hands_left,
            hands_played=hands_played,
            discards_left=discards_left,
            chips=chips,
        ),
        blinds={
            "active": BlindInfo(
                type=blind_type,
                status="CURRENT",
                name=blind_name,
                effect="",
                score=blind_score,
            )
        },
        hand=AreaInfo(count=len(hand_cards or []), cards=hand_cards or []),
        jokers=AreaInfo(count=len(jokers or []), cards=jokers or []),
        hands={
            PokerHand.HIGH_CARD: HandLevel(order=1, level=1, chips=5, mult=1),
            PokerHand.PAIR: HandLevel(order=2, level=1, chips=10, mult=2),
            PokerHand.TWO_PAIR: HandLevel(order=3, level=1, chips=20, mult=2),
            PokerHand.THREE_OF_A_KIND: HandLevel(order=4, level=1, chips=30, mult=3),
            PokerHand.STRAIGHT: HandLevel(order=5, level=1, chips=30, mult=4),
            PokerHand.FLUSH: HandLevel(order=6, level=1, chips=35, mult=4),
            PokerHand.FULL_HOUSE: HandLevel(order=7, level=1, chips=40, mult=4),
            PokerHand.FOUR_OF_A_KIND: HandLevel(order=8, level=1, chips=60, mult=7),
            PokerHand.STRAIGHT_FLUSH: HandLevel(order=9, level=1, chips=100, mult=8),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. Poker Hand Identification
# ═══════════════════════════════════════════════════════════════════════════

class TestPokerHandIdentification:
    """Verify exact hand detection and scoring card isolation."""

    def test_pair(self):
        c1 = _make_card("A", "H")
        c2 = _make_card("A", "D")
        c3 = _make_card("K", "S")
        c4 = _make_card("7", "C")
        hand_name, scoring = identify_poker_hand([c1, c2, c3, c4])

        assert hand_name == PokerHand.PAIR
        assert scoring == [c1, c2]  # Only the two Aces score

    def test_two_pair(self):
        c1 = _make_card("A", "H")
        c2 = _make_card("A", "D")
        c3 = _make_card("K", "S")
        c4 = _make_card("K", "C")
        c5 = _make_card("2", "H")
        hand_name, scoring = identify_poker_hand([c1, c2, c3, c4, c5])

        assert hand_name == PokerHand.TWO_PAIR
        assert scoring == [c1, c2, c3, c4]  # The 4 pair cards score

    def test_three_of_a_kind(self):
        c1 = _make_card("J", "H")
        c2 = _make_card("J", "D")
        c3 = _make_card("J", "C")
        c4 = _make_card("4", "S")
        hand_name, scoring = identify_poker_hand([c1, c2, c3, c4])

        assert hand_name == PokerHand.THREE_OF_A_KIND
        assert scoring == [c1, c2, c3]

    def test_straight_low_ace(self):
        """A-2-3-4-5 must form a Straight."""
        c1 = _make_card("A", "H")
        c2 = _make_card("2", "D")
        c3 = _make_card("3", "S")
        c4 = _make_card("4", "C")
        c5 = _make_card("5", "H")
        hand_name, scoring = identify_poker_hand([c1, c2, c3, c4, c5])

        assert hand_name == PokerHand.STRAIGHT
        assert len(scoring) == 5

    def test_straight_high_ace(self):
        """10-J-Q-K-A must form a Straight."""
        c1 = _make_card("T", "H")
        c2 = _make_card("J", "D")
        c3 = _make_card("Q", "S")
        c4 = _make_card("K", "C")
        c5 = _make_card("A", "H")
        hand_name, scoring = identify_poker_hand([c1, c2, c3, c4, c5])

        assert hand_name == PokerHand.STRAIGHT
        assert len(scoring) == 5

    def test_flush(self):
        cards = [
            _make_card("2", "H"),
            _make_card("5", "H"),
            _make_card("8", "H"),
            _make_card("J", "H"),
            _make_card("A", "H"),
        ]
        hand_name, scoring = identify_poker_hand(cards)

        assert hand_name == PokerHand.FLUSH
        assert len(scoring) == 5

    def test_full_house(self):
        cards = [
            _make_card("K", "H"),
            _make_card("K", "D"),
            _make_card("K", "S"),
            _make_card("Q", "C"),
            _make_card("Q", "H"),
        ]
        hand_name, scoring = identify_poker_hand(cards)

        assert hand_name == PokerHand.FULL_HOUSE
        assert len(scoring) == 5

    def test_four_of_a_kind(self):
        c1 = _make_card("7", "H")
        c2 = _make_card("7", "D")
        c3 = _make_card("7", "C")
        c4 = _make_card("7", "S")
        c5 = _make_card("2", "H")
        hand_name, scoring = identify_poker_hand([c1, c2, c3, c4, c5])

        assert hand_name == PokerHand.FOUR_OF_A_KIND
        assert scoring == [c1, c2, c3, c4]

    def test_straight_flush(self):
        cards = [
            _make_card("9", "D"),
            _make_card("T", "D"),
            _make_card("J", "D"),
            _make_card("Q", "D"),
            _make_card("K", "D"),
        ]
        hand_name, scoring = identify_poker_hand(cards)

        assert hand_name == PokerHand.STRAIGHT_FLUSH
        assert len(scoring) == 5

    def test_four_fingers_flush_and_straight(self):
        """Four Fingers enables 4-card Flushes and Straights."""
        jkeys = {"j_four_fingers"}
        # 4 Hearts
        h_cards = [
            _make_card("2", "H"),
            _make_card("5", "H"),
            _make_card("8", "H"),
            _make_card("K", "H"),
        ]
        h_name, h_scoring = identify_poker_hand(h_cards, jkeys)
        assert h_name == PokerHand.FLUSH
        assert len(h_scoring) == 4

        # 4 consecutive ranks: 7-8-9-10
        s_cards = [
            _make_card("7", "C"),
            _make_card("8", "D"),
            _make_card("9", "S"),
            _make_card("T", "H"),
        ]
        s_name, s_scoring = identify_poker_hand(s_cards, jkeys)
        assert s_name == PokerHand.STRAIGHT
        assert len(s_scoring) == 4

    def test_shortcut_straight_with_gaps(self):
        """Shortcut allows gaps of 1 rank in Straights."""
        jkeys = {"j_shortcut"}
        # 4-6-7-9-T (skip 5 and skip 8: 1 gap each)
        cards = [
            _make_card("4", "H"),
            _make_card("6", "D"),
            _make_card("7", "S"),
            _make_card("9", "C"),
            _make_card("T", "H"),
        ]
        hand_name, scoring = identify_poker_hand(cards, jkeys)
        assert hand_name == PokerHand.STRAIGHT
        assert len(scoring) == 5

    def test_smeared_joker_flush(self):
        """Smeared Joker merges Hearts/Diamonds and Spades/Clubs."""
        jkeys = {"j_smeared"}
        cards = [
            _make_card("2", "H"),
            _make_card("5", "D"),
            _make_card("8", "H"),
            _make_card("T", "D"),
            _make_card("A", "H"),
        ]
        hand_name, scoring = identify_poker_hand(cards, jkeys)
        assert hand_name == PokerHand.FLUSH
        assert len(scoring) == 5

    def test_splash_scores_all_played_cards(self):
        """Splash makes all played cards count in scoring."""
        jkeys = {"j_splash"}
        c1 = _make_card("A", "H")
        c2 = _make_card("A", "D")
        c3 = _make_card("3", "S")
        c4 = _make_card("5", "C")
        c5 = _make_card("8", "H")
        hand_name, scoring = identify_poker_hand([c1, c2, c3, c4, c5], jkeys)

        assert hand_name == PokerHand.PAIR
        assert scoring == [c1, c2, c3, c4, c5]  # All 5 cards score


# ═══════════════════════════════════════════════════════════════════════════
# 2. Exact Score Calculation without Jokers
# ═══════════════════════════════════════════════════════════════════════════

class TestExactScoreCalculationWithoutJokers:
    """Exact point-by-point verification with base cards, enhancements, and editions."""

    def test_simple_pair_of_aces(self):
        """Level 1 Pair: base 10 chips, 2 mult.

        Played: Ace of Hearts (11 chips) + Ace of Diamonds (11 chips).
        Total chips = 10 (base) + 11 + 11 = 32.
        Total mult = 2.
        Score = 32 * 2 = 64.
        """
        c1 = _make_card("A", "H")
        c2 = _make_card("A", "D")
        state = _make_base_state(hand_cards=[c1, c2])

        base_c, fin_c, fin_m, score, debuffed = calculate_hand_score(
            played_cards=[c1, c2],
            held_cards=[],
            hand_type=PokerHand.PAIR,
            scoring_cards=[c1, c2],
            state=state,
        )

        assert not debuffed
        assert base_c == 10
        assert fin_c == 32
        assert fin_m == 2.0
        assert score == 64

    def test_pair_with_bonus_foil_mult_polychrome(self):
        """Card 1: Ace of Hearts + BONUS (+30 chips) + FOIL (+50 chips)
        Card 2: Ace of Diamonds + MULT (+4 mult) + POLYCHROME (x1.5 mult)

        Base Pair: 10 chips, 2 mult.
        Card 1: 11 rank + 30 bonus + 50 foil = 91 chips.
        Card 2: 11 rank chips, +4 mult, then x1.5 mult.
        Total chips = 10 + 91 + 11 = 112.
        Total mult = (2 + 4) * 1.5 = 6 * 1.5 = 9.0.
        Score = 112 * 9 = 1008.
        """
        c1 = _make_card("A", "H", enhancement="BONUS", edition="FOIL")
        c2 = _make_card("A", "D", enhancement="MULT", edition="POLYCHROME")
        state = _make_base_state(hand_cards=[c1, c2])

        base_c, fin_c, fin_m, score, debuffed = calculate_hand_score(
            played_cards=[c1, c2],
            held_cards=[],
            hand_type=PokerHand.PAIR,
            scoring_cards=[c1, c2],
            state=state,
        )

        assert not debuffed
        assert fin_c == 112
        assert fin_m == 9.0
        assert score == 1008

    def test_glass_card_multiplication(self):
        """Glass card doubles the Mult when scored."""
        c1 = _make_card("K", "H")
        c2 = _make_card("K", "S", enhancement="GLASS")
        state = _make_base_state(hand_cards=[c1, c2])

        # Base Pair: 10 chips, 2 mult
        # Chips = 10 + 10 (K) + 10 (K) = 30
        # Mult = 2 * 2 (glass) = 4.0
        # Score = 30 * 4 = 120
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c1, c2],
            held_cards=[],
            hand_type=PokerHand.PAIR,
            scoring_cards=[c1, c2],
            state=state,
        )

        assert fin_c == 30
        assert fin_m == 4.0
        assert score == 120

    def test_stone_card_scores_flat_50_chips(self):
        """Stone card gives +50 chips and no rank value."""
        c1 = _make_card("A", "H")
        c_stone = _make_card("2", "S", enhancement="STONE")
        state = _make_base_state(hand_cards=[c1, c_stone])

        # High card: base 5 chips, 1 mult
        # Chips = 5 + 11 (Ace) + 50 (Stone) = 66 chips
        # Mult = 1.0
        # Score = 66 * 1 = 66
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c1, c_stone],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c1, c_stone],
            state=state,
        )

        assert fin_c == 66
        assert fin_m == 1.0
        assert score == 66


# ═══════════════════════════════════════════════════════════════════════════
# 3. Held-in-Hand Cards (Steel & Held Jokers)
# ═══════════════════════════════════════════════════════════════════════════

class TestHeldInHandEffects:
    """Held cards like Steel must apply x1.5 before global Jokers."""

    def test_steel_card_held_in_hand_applies_before_global_joker(self):
        """Played: 2 of Hearts (2) + 2 of Diamonds (2). Base Pair: 10 chips, 2 mult.
        Chips = 10 + 2 + 2 = 14 chips.

        Held in hand: 5 of Clubs with STEEL (x1.5 mult).
        Mult after Step C: 2.0 * 1.5 = 3.0 mult.

        Global Joker: j_joker (+4 mult).
        Mult after Step D: 3.0 + 4 = 7.0 mult.
        Total score = floor(14 * 7.0) = 98.

        If Steel was applied after the Joker, mult would be (2 + 4) * 1.5 = 9.0 (126 pts).
        This test proves the exact Balatro ordering.
        """
        c1 = _make_card("2", "H")
        c2 = _make_card("2", "D")
        c_steel = _make_card("5", "C", enhancement="STEEL")
        joker = _make_joker("j_joker")

        state = _make_base_state(
            hand_cards=[c1, c2, c_steel],
            jokers=[joker],
        )

        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c1, c2],
            held_cards=[c_steel],
            hand_type=PokerHand.PAIR,
            scoring_cards=[c1, c2],
            state=state,
        )

        assert fin_c == 14
        assert fin_m == 7.0
        assert score == 98

    def test_baron_and_shoot_the_moon_held(self):
        """Baron gives x1.5 per held King. Shoot the Moon gives +13 per held Queen."""
        c1 = _make_card("A", "H")
        c_k = _make_card("K", "S")
        c_q = _make_card("Q", "D")
        j_baron = _make_joker("j_baron")
        j_stm = _make_joker("j_shoot_the_moon")

        state = _make_base_state(
            hand_cards=[c1, c_k, c_q],
            jokers=[j_baron, j_stm],
        )

        # High Card Ace: 5 chips, 1 mult
        # Chips = 5 + 11 = 16
        # Held: King (x1.5 via Baron) -> 1 * 1.5 = 1.5
        # Held: Queen (+13 via Shoot the Moon) -> 1.5 + 13 = 14.5
        # Score = floor(16 * 14.5) = 232
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c1],
            held_cards=[c_k, c_q],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c1],
            state=state,
        )

        assert fin_c == 16
        assert fin_m == 14.5
        assert score == 232


# ═══════════════════════════════════════════════════════════════════════════
# 4. Joker Ordering (Strictly Left to Right)
# ═══════════════════════════════════════════════════════════════════════════

class TestJokerOrdering:
    """Mathematical proof that Joker order matters and is respected."""

    def test_plus_mult_before_times_mult_gives_higher_score(self):
        """Order A: [j_joker (+4), j_cavendish (x3)]
        Order B: [j_cavendish (x3), j_joker (+4)]

        High Card Ace: base 5 chips, 1 mult.
        Scored card Ace: +11 chips. Total chips = 16.

        Order A mult: (1 + 4) * 3 = 15 mult.
        Score A = 16 * 15 = 240 pts.

        Order B mult: (1 * 3) + 4 = 7 mult.
        Score B = 16 * 7 = 112 pts.

        Mathematical proof: 240 > 112.
        """
        c1 = _make_card("A", "H")
        j_plus = _make_joker("j_joker", joker_id=1)
        j_times = _make_joker("j_cavendish", joker_id=2)

        # Test Order A
        state_a = _make_base_state(hand_cards=[c1], jokers=[j_plus, j_times])
        _, _, fin_m_a, score_a, _ = calculate_hand_score(
            [c1], [], PokerHand.HIGH_CARD, [c1], state_a
        )
        assert fin_m_a == 15.0
        assert score_a == 240

        # Test Order B
        state_b = _make_base_state(hand_cards=[c1], jokers=[j_times, j_plus])
        _, _, fin_m_b, score_b, _ = calculate_hand_score(
            [c1], [], PokerHand.HIGH_CARD, [c1], state_b
        )
        assert fin_m_b == 7.0
        assert score_b == 112

        assert score_a > score_b

    def test_blueprint_copies_right_joker(self):
        """Blueprint copies the joker to its right."""
        c1 = _make_card("A", "H")
        j_bp = _make_joker("j_blueprint", joker_id=1)
        j_base = _make_joker("j_joker", joker_id=2)  # +4 mult

        state = _make_base_state(hand_cards=[c1], jokers=[j_bp, j_base])
        # High Card Ace: 5 chips, 1 mult + 11 = 16 chips.
        # Blueprint copies j_joker (+4), then j_joker triggers (+4)
        # Mult = 1 + 4 + 4 = 9.0
        # Score = 16 * 9 = 144
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            [c1], [], PokerHand.HIGH_CARD, [c1], state
        )
        assert fin_m == 9.0
        assert score == 144

    def test_brainstorm_copies_leftmost_joker(self):
        """Brainstorm copies the leftmost joker in the lineup."""
        c1 = _make_card("A", "H")
        j_left = _make_joker("j_joker", joker_id=1)  # +4 mult
        j_mid = _make_joker("j_cavendish", joker_id=2)  # x3 mult
        j_bs = _make_joker("j_brainstorm", joker_id=3)  # copies leftmost (j_joker)

        state = _make_base_state(hand_cards=[c1], jokers=[j_left, j_mid, j_bs])
        # Chips = 16
        # Left (j_joker): 1 + 4 = 5
        # Mid (j_cavendish): 5 * 3 = 15
        # Brainstorm copies j_joker: 15 + 4 = 19
        # Score = 16 * 19 = 304
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            [c1], [], PokerHand.HIGH_CARD, [c1], state
        )
        assert fin_m == 19.0
        assert score == 304


# ═══════════════════════════════════════════════════════════════════════════
# 5. Debuffed Cards & Kill Shot Optimization
# ═══════════════════════════════════════════════════════════════════════════

class TestDebuffAndKillShot:
    """Debuffed cards participate in the hand shape but yield 0 chips/effects.
    Kill shot selects the minimal winning combination.
    """

    def test_debuffed_card_gives_zero_chips_but_forms_pair(self):
        """Pair of 8s: 8 of Hearts (debuffed) + 8 of Diamonds (normal).

        Base Pair: 10 chips, 2 mult.
        Debuffed 8: 0 chips.
        Normal 8: 8 chips.
        Total chips = 10 + 8 = 18.
        Mult = 2.
        Score = 18 * 2 = 36 (instead of 52 if normal).
        """
        c1 = _make_card("8", "H", debuff=True)
        c2 = _make_card("8", "D", debuff=False)
        state = _make_base_state(hand_cards=[c1, c2])

        hand_name, scoring = identify_poker_hand([c1, c2])
        assert hand_name == PokerHand.PAIR
        assert scoring == [c1, c2]

        _, fin_c, fin_m, score, _ = calculate_hand_score(
            [c1, c2], [], hand_name, scoring, state
        )
        assert fin_c == 18
        assert fin_m == 2.0
        assert score == 36

    def test_kill_shot_conserves_cards_when_small_score_needed(self):
        """When blind needs 40 chips:
        A Pair of Aces gives 64 pts (2 cards).
        A Full House of Kings/Queens gives ~150 pts (5 cards).
        The solver MUST pick the 2-card Pair to save the other cards!
        """
        hand_cards = [
            _make_card("A", "H", card_id=1),
            _make_card("A", "D", card_id=2),
            _make_card("K", "S", card_id=3),
            _make_card("K", "C", card_id=4),
            _make_card("K", "H", card_id=5),
            _make_card("Q", "S", card_id=6),
            _make_card("Q", "D", card_id=7),
            _make_card("2", "C", card_id=8),
        ]
        # Target 40 chips needed
        state = _make_base_state(hand_cards=hand_cards, blind_score=40, chips=0)

        best = select_best_play(state)
        assert best is not None
        assert best.score >= 40

        # Should choose Pair (2 cards) rather than Full House (5 cards)
        assert len(best.indices) == 2
        assert best.hand_type == PokerHand.PAIR

    def test_kill_shot_preserves_steel_and_glass_in_hand(self):
        """Between two equal-sized winning plays, the solver must conserve
        the one keeping Steel / Glass in hand.
        """
        # Pair of 2s (normal)
        c1 = _make_card("2", "H", card_id=1)
        c2 = _make_card("2", "D", card_id=2)

        # Pair of Aces with STEEL (precious!)
        c3 = _make_card("A", "S", enhancement="STEEL", card_id=3)
        c4 = _make_card("A", "C", card_id=4)

        # Target needed: 25 chips
        # Pair of 2s: 10 base + 4 = 14 chips * 2 = 28 pts (beats 25 pts!)
        # Pair of Aces: beats 25 pts too, but playing the Steel card consumes it
        hand_cards = [c1, c2, c3, c4]
        state = _make_base_state(hand_cards=hand_cards, blind_score=25, chips=0)

        best = select_best_play(state)
        assert best is not None
        assert best.score >= 25

        # Must play the pair of 2s to keep the Steel Ace in hand!
        assert c3 not in best.cards
        assert set(best.indices) == {0, 1}

    def test_psychic_boss_requires_five_cards(self):
        """The Psychic debuffs plays with fewer than 5 cards."""
        c1 = _make_card("A", "H")
        c2 = _make_card("A", "D")
        state = _make_base_state(
            hand_cards=[c1, c2],
            blind_name="The Psychic",
            blind_type="BOSS",
        )

        _, _, _, score, debuffed = calculate_hand_score(
            [c1, c2], [], PokerHand.PAIR, [c1, c2], state
        )
        assert debuffed
        assert score == 0


class TestJokerAndCardEditionCalculations:
    """Validate that editions on Jokers and playing cards are accurately scored."""

    def test_foil_joker_adds_50_chips(self):
        """A Joker with FOIL edition must add +50 chips during Step D scoring."""
        c = _make_card("2", "H")
        j_foil = _make_joker("j_joker", edition="FOIL")
        state = _make_base_state(hand_cards=[c], jokers=[j_foil])

        # High card base: 5 chips, 1 mult.
        # Played 2 of Hearts: +2 chips -> 7 chips.
        # Step D Joker edition FOIL: +50 chips -> 57 chips.
        # Step D j_joker: +4 mult -> 5.0 mult.
        # Score = 57 * 5 = 285.
        _, fin_c, fin_m, score, debuffed = calculate_hand_score(
            played_cards=[c],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c],
            state=state,
        )

        assert not debuffed
        assert fin_c == 57
        assert fin_m == 5.0
        assert score == 285

    def test_foil_joker_case_insensitive_and_prefix(self):
        """Editions formatted as 'e_foil' or 'foil' should also add +50 chips."""
        c = _make_card("2", "H")
        j_foil = _make_joker("j_joker", edition="e_foil")
        state = _make_base_state(hand_cards=[c], jokers=[j_foil])

        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c],
            state=state,
        )

        assert fin_c == 57
        assert fin_m == 5.0
        assert score == 285

    def test_holo_joker_adds_10_mult(self):
        """A Joker with HOLO edition adds +10 mult."""
        c = _make_card("2", "H")
        j_holo = _make_joker("j_joker", edition="HOLO")
        state = _make_base_state(hand_cards=[c], jokers=[j_holo])

        # Chips: 5 base + 2 card = 7 chips.
        # Mult: 1 base + 10 holo + 4 j_joker = 15.0 mult.
        # Score = 7 * 15 = 105.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c],
            state=state,
        )

        assert fin_c == 7
        assert fin_m == 15.0
        assert score == 105

    def test_polychrome_joker_multiplies_by_1_5(self):
        """A Joker with POLYCHROME edition multiplies current mult by 1.5."""
        c = _make_card("2", "H")
        j_poly = _make_joker("j_joker", edition="POLYCHROME")
        state = _make_base_state(hand_cards=[c], jokers=[j_poly])

        # Chips: 5 base + 2 card = 7 chips.
        # Step D j_joker: mult = 1.0 + 4.0 = 5.0 mult.
        # Step D Polychrome (after ability): mult = 5.0 * 1.5 = 7.5 mult.
        # Score = floor(7 * 7.5) = 52.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c],
            state=state,
        )

        assert fin_c == 7
        assert fin_m == 7.5
        assert score == 52

    def test_negative_joker_no_scoring_change(self):
        """A Negative Joker does not change chips or mult directly."""
        c = _make_card("2", "H")
        j_neg = _make_joker("j_joker", edition="NEGATIVE")
        state = _make_base_state(hand_cards=[c], jokers=[j_neg])

        # Chips: 7, Mult: 1 + 4 = 5, Score: 35.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c],
            state=state,
        )

        assert fin_c == 7
        assert fin_m == 5.0
        assert score == 35

    def test_foil_played_card_adds_50_chips(self):
        """A played card with FOIL edition adds +50 chips during Step B."""
        c = _make_card("2", "H", edition="FOIL")
        state = _make_base_state(hand_cards=[c])

        # Base High Card: 5 chips, 1 mult.
        # Played 2 of Hearts (FOIL): 2 rank + 50 foil = 52 chips.
        # Total chips = 57, Mult = 1.
        # Score = 57.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c],
            state=state,
        )

        assert fin_c == 57
        assert fin_m == 1.0
        assert score == 57

    def test_card_conservation_values_for_editions(self):
        """Editions increase conservation priority: Polychrome > Holo > Foil > Base."""
        c_poly = _make_card("2", "H", edition="POLYCHROME")
        c_holo = _make_card("2", "H", edition="HOLO")
        c_foil = _make_card("2", "H", edition="FOIL")
        c_plain = _make_card("2", "H")

        assert card_conservation_value(c_poly) > card_conservation_value(c_holo)
        assert card_conservation_value(c_holo) > card_conservation_value(c_foil)
        assert card_conservation_value(c_foil) > card_conservation_value(c_plain)


class TestGroupAJokersCalculations:
    """Validate accurate scoring for Group A static/deterministic Jokers."""

    def test_stencil_joker_with_empty_slots(self):
        """Stencil gives 1 + empty_slots XMult. 1 joker in 5 limit = 4 empty -> X5 Mult."""
        c = _make_card("2", "H")
        j_stencil = _make_joker("j_stencil")
        state = _make_base_state(hand_cards=[c], jokers=[j_stencil])
        state.jokers.limit = 5

        # Chips: 5 base + 2 card = 7.
        # Mult: 1.0 * (1 + 4) = 5.0.
        # Score = 7 * 5 = 35.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c],
            state=state,
        )
        assert fin_c == 7
        assert fin_m == 5.0
        assert score == 35

    def test_blue_joker_scales_with_remaining_deck(self):
        """Blue Joker gives +2 chips per remaining card in deck."""
        c = _make_card("2", "H")
        j_blue = _make_joker("j_blue_joker")
        state = _make_base_state(hand_cards=[c], jokers=[j_blue])
        state.deck_cards.count = 40

        # Chips: 5 base + 2 card + (40 * 2) = 87 chips.
        # Mult: 1.0. Score = 87.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c],
            state=state,
        )
        assert fin_c == 87
        assert fin_m == 1.0
        assert score == 87

    def test_swashbuckler_adds_other_jokers_sell_value(self):
        """Swashbuckler adds sell value of all other owned Jokers to Mult."""
        c = _make_card("2", "H")
        j_swash = _make_joker("j_swashbuckler", sell_cost=1)
        j_other1 = _make_joker("j_ice_cream", sell_cost=3)
        j_other2 = _make_joker("j_banner", sell_cost=4)
        state = _make_base_state(hand_cards=[c], jokers=[j_swash, j_other1, j_other2])

        # Swashbuckler adds 3 + 4 = 7 Mult.
        # Base High card: 5 chips, 1 mult.
        # Ice cream adds 100 chips. Banner adds 30 * 3 = 90 chips. Card adds 2 chips.
        # Total chips = 5 + 2 + 100 + 90 = 197.
        # Total mult = 1 + 7 = 8.0.
        # Score = 197 * 8 = 1576.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c],
            state=state,
        )
        assert fin_c == 197
        assert fin_m == 8.0
        assert score == 1576

    def test_flower_pot_triggers_when_all_four_suits_score(self):
        """Flower Pot gives X3 Mult when scored cards contain D, C, H, S."""
        c_h = _make_card("2", "H")
        c_d = _make_card("3", "D")
        c_c = _make_card("4", "C")
        c_s = _make_card("5", "S")
        j_pot = _make_joker("j_flower_pot")
        state = _make_base_state(hand_cards=[c_h, c_d, c_c, c_s], jokers=[j_pot])

        # High card / played cards:
        # Base High Card: 5 chips, 1 mult.
        # Chips: 5 + 2 + 3 + 4 + 5 = 19 chips.
        # Mult: 1.0 * 3.0 = 3.0 mult.
        # Score = 19 * 3 = 57.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c_h, c_d, c_c, c_s],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c_h, c_d, c_c, c_s],
            state=state,
        )
        assert fin_c == 19
        assert fin_m == 3.0
        assert score == 57

    def test_seeing_double_triggers_with_club_and_other_suit(self):
        """Seeing Double gives X2 Mult when hand has scoring Club and another suit."""
        c_c = _make_card("2", "C")
        c_d = _make_card("3", "D")
        j_sd = _make_joker("j_seeing_double")
        state = _make_base_state(hand_cards=[c_c, c_d], jokers=[j_sd])

        # Base High Card: 5 chips, 1 mult.
        # Chips = 5 + 2 + 3 = 10.
        # Mult = 1.0 * 2.0 = 2.0.
        # Score = 20.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c_c, c_d],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c_c, c_d],
            state=state,
        )
        assert fin_c == 10
        assert fin_m == 2.0
        assert score == 20

    def test_erosion_adds_mult_below_52_cards(self):
        """Erosion adds +4 Mult per card below 52 in full deck."""
        c = _make_card("2", "H")
        j_ero = _make_joker("j_erosion")
        state = _make_base_state(hand_cards=[c], jokers=[j_ero])
        state.deck_cards.count = 42  # 10 below 52 -> +40 Mult

        # Base High card: 5 chips, 1 mult.
        # Chips = 7. Mult = 1 + 40 = 41.0.
        # Score = 7 * 41 = 287.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c],
            state=state,
        )
        assert fin_c == 7
        assert fin_m == 41.0
        assert score == 287

    def test_steel_joker_and_stone_joker(self):
        """Steel Joker scales X0.2 per steel card; Stone Joker gives +25 chips per stone card."""
        c = _make_card("2", "H")
        j_steel = _make_joker("j_steel_joker")
        j_stone = _make_joker("j_stone")

        # Full deck with 2 Steel and 2 Stone cards
        c_st1 = _make_card("5", "S", enhancement="STEEL")
        c_st2 = _make_card("6", "S", enhancement="STEEL")
        c_sto1 = _make_card("7", "D", enhancement="STONE")
        c_sto2 = _make_card("8", "D", enhancement="STONE")

        state = _make_base_state(
            hand_cards=[c],
            jokers=[j_stone, j_steel],
        )
        state.deck_cards.cards = [c_st1, c_st2, c_sto1, c_sto2]

        # Base High Card: 5 chips, 1 mult.
        # Chips: 5 + 2 (card) + (25 * 2 stone) = 57 chips.
        # Mult: 1.0 * (1.0 + 0.2 * 2 steel) = 1.4 mult.
        # Score = floor(57 * 1.4) = floor(79.8) = 79.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c],
            state=state,
        )
        assert fin_c == 57
        assert fin_m == 1.4
        assert score == 79

    def test_baseball_card_multiplies_for_uncommon_jokers(self):
        """Baseball Card gives X1.5 Mult for each other Uncommon Joker."""
        c = _make_card("7", "H")
        j_bb = _make_joker("j_baseball")
        j_unc1 = _make_joker("j_mime")       # Uncommon
        j_unc2 = _make_joker("j_fibonacci")  # Uncommon
        j_com = _make_joker("j_joker")       # Common (+4 mult)

        state = _make_base_state(
            hand_cards=[c],
            jokers=[j_com, j_bb, j_unc1, j_unc2],
        )

        # Base High Card: 5 chips, 1 mult.
        # Chips: 5 + 7 = 12 chips.
        # Joker adds +4 mult -> 5.0 mult.
        # Baseball Card gives X1.5 for j_unc1 and X1.5 for j_unc2 -> 5.0 * 1.5 * 1.5 = 11.25 mult.
        # Total score = floor(12 * 11.25) = 135.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c],
            state=state,
        )
        assert fin_c == 12
        assert fin_m == 11.25
        assert score == 135

    def test_bloodstone_scores_on_hearts(self):
        """Bloodstone multiplies mult by X1.5 on played Heart cards."""
        c_h = _make_card("2", "H")
        j_bs = _make_joker("j_bloodstone")
        state = _make_base_state(hand_cards=[c_h], jokers=[j_bs])

        # Base High Card: 5 chips, 1 mult.
        # Heart card scores: Bloodstone gives X1.5 Mult -> 1.5 mult.
        # Chips = 7. Score = floor(7 * 1.5) = 10.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c_h],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c_h],
            state=state,
        )
        assert fin_c == 7
        assert fin_m == 1.5
        assert score == 10


class TestGroupBScalingJokersCalculations:
    """Validate accurate scoring for Group B scaling & dynamic Jokers using ability.extra."""

    def test_constellation_with_dynamic_xmult(self):
        """Constellation applies its dynamic x_mult extracted from ability."""
        c = _make_card("2", "H")
        j_const = _make_joker("j_constellation")
        j_const.value.x_mult = 2.1
        state = _make_base_state(hand_cards=[c], jokers=[j_const])

        # High card base: 5 chips, 1 mult.
        # Card adds 2 chips -> 7 chips.
        # Constellation multiplies mult by 2.1 -> 2.1 mult.
        # Score = floor(7 * 2.1) = 14.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c],
            state=state,
        )
        assert fin_c == 7
        assert fin_m == 2.1
        assert score == 14

    def test_green_joker_with_dynamic_mult(self):
        """Green Joker applies its dynamic mult extracted from ability."""
        c = _make_card("2", "H")
        j_green = _make_joker("j_green_joker")
        j_green.value.mult = 18.0
        state = _make_base_state(hand_cards=[c], jokers=[j_green])

        # In Balatro, Green Joker gains +1 Mult in context.before (before scoring)
        # Chips: 7. Mult = 1 base + 18 dynamic + 1 hand_add = 20.0.
        # Score = 7 * 20 = 140.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c],
            state=state,
        )
        assert fin_c == 7
        assert fin_m == 20.0
        assert score == 140

    def test_runner_with_dynamic_chips(self):
        """Runner applies its accumulated chips from ability."""
        c = _make_card("2", "H")
        j_run = _make_joker("j_runner")
        j_run.value.chips = 60
        state = _make_base_state(hand_cards=[c], jokers=[j_run])

        # Chips: 5 base + 2 card + 60 runner = 67 chips.
        # Mult: 1.0. Score = 67.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c],
            state=state,
        )
        assert fin_c == 67
        assert fin_m == 1.0
        assert score == 67

    def test_ancient_joker_matching_suit(self):
        """Ancient Joker gives X1.5 Mult on scored cards matching target suit in ability.extra."""
        c_s = _make_card("2", "S")
        j_anc = _make_joker("j_ancient")
        j_anc.value.extra = {"suit": "Spades"}
        state = _make_base_state(hand_cards=[c_s], jokers=[j_anc])

        # 2 of Spades scores: matches Spades -> X1.5 Mult.
        # Chips = 7. Mult = 1.5. Score = 10.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c_s],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c_s],
            state=state,
        )
        assert fin_c == 7
        assert fin_m == 1.5
        assert score == 10

    def test_idol_joker_matching_rank_and_suit(self):
        """The Idol gives X2 Mult on scored cards matching exact target rank and suit."""
        c_kh = _make_card("K", "H")
        j_idol = _make_joker("j_idol")
        j_idol.value.extra = {"rank": "K", "suit": "Hearts"}
        state = _make_base_state(hand_cards=[c_kh], jokers=[j_idol])

        # King of Hearts scores: matches K of Hearts -> X2.0 Mult.
        # Chips = 5 base + 10 card = 15 chips.
        # Mult = 1.0 * 2.0 = 2.0.
        # Score = 30.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c_kh],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c_kh],
            state=state,
        )
        assert fin_c == 15
        assert fin_m == 2.0
        assert score == 30

    def test_ice_cream_decayed_chips(self):
        """Ice cream with 85 remaining chips gives +85 chips, not 100."""
        c = _make_card("2", "H")
        j_ice = _make_joker("j_ice_cream")
        j_ice.value.extra = {"chips": 85, "chip_mod": 5}
        state = _make_base_state(hand_cards=[c], jokers=[j_ice])

        # Base 5 + card 2 + 85 = 92 chips
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c],
            state=state,
        )
        assert fin_c == 92
        assert score == 92

    def test_flower_pot_requires_four_distinct_suits(self):
        """Flower pot does NOT trigger on a single wild card."""
        c_wild = _make_card("A", "H", enhancement="WILD")
        j_fp = _make_joker("j_flower_pot")
        state = _make_base_state(hand_cards=[c_wild], jokers=[j_fp])

        # Single card cannot satisfy 4 suits
        _, _, fin_m, _, _ = calculate_hand_score(
            played_cards=[c_wild],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c_wild],
            state=state,
        )
        assert fin_m == 1.0  # Does NOT get X3 Mult!

    def test_seeing_double_requires_club_and_other(self):
        """Seeing Double does NOT trigger on a single wild card."""
        c_wild = _make_card("A", "C", enhancement="WILD")
        j_sd = _make_joker("j_seeing_double")
        state = _make_base_state(hand_cards=[c_wild], jokers=[j_sd])

        # Single wild card only fills Club, cannot also fill Other
        _, _, fin_m, _, _ = calculate_hand_score(
            played_cards=[c_wild],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c_wild],
            state=state,
        )
        assert fin_m == 1.0  # Does NOT get X2 Mult!

    def test_blackboard_empty_held_hand(self):
        """Blackboard triggers even when all 5 cards were played (0 cards held)."""
        cards = [_make_card(str(r), "H") for r in range(2, 7)]
        j_bb = _make_joker("j_blackboard")
        state = _make_base_state(hand_cards=cards, jokers=[j_bb])

        _, _, fin_m, _, _ = calculate_hand_score(
            played_cards=cards,
            held_cards=[],
            hand_type=PokerHand.STRAIGHT,
            scoring_cards=cards,
            state=state,
        )
        # Empty held passes (0 non-black cards held) -> X3.0 Mult
        assert fin_m == 12.0  # Base straight mult 4.0 * 3.0 = 12.0

    def test_blueprint_copies_mime_held_cards(self):
        """Blueprint copying Mime applies double retrigger to held Steel card."""
        c_play = _make_card("2", "H")
        c_held = _make_card("K", "S", enhancement="STEEL")
        j_bp = _make_joker("j_blueprint")
        j_mime = _make_joker("j_mime")
        state = _make_base_state(hand_cards=[c_play, c_held], jokers=[j_bp, j_mime])

        # Steel gives 1.5x. Mime + Blueprint = 2 extra retriggers = 3 total triggers = 1.5^3 = 3.375x
        _, _, fin_m, _, _ = calculate_hand_score(
            played_cards=[c_play],
            held_cards=[c_held],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c_play],
            state=state,
        )
        assert pytest.approx(fin_m, 0.001) == 1.0 * (1.5 ** 3)

    def test_castle_flush_and_baron_full_house(self):
        """Regression test for user run: Castle + Greedy Joker + Onyx Agate + Baron."""
        j_delayed = _make_joker("j_delayed_grat")
        j_castle = _make_joker("j_castle")
        j_castle.value.extra = {"chips": 6, "chip_mod": 3}
        j_greedy = _make_joker("j_greedy_joker")
        j_onyx = _make_joker("j_onyx_agate")
        j_baron = _make_joker("j_baron")
        jokers = [j_delayed, j_castle, j_greedy, j_onyx, j_baron]

        hands = {
            "Flush": HandLevel(level=3, played=0, played_this_round=0, chips=65, mult=8),
            "Full House": HandLevel(level=1, played=0, played_this_round=0, chips=40, mult=4),
        }
        state = _make_base_state(hand_cards=[], jokers=jokers)
        state.hands = hands

        # Hand 1: Flush ['AS', 'TS', '9S', '6S', '5S']
        flush_cards = [_make_card("A", "S"), _make_card("T", "S"), _make_card("9", "S"), _make_card("6", "S"), _make_card("5", "S")]
        held_flush = [_make_card("2", "H"), _make_card("3", "C"), _make_card("7", "D")]
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            flush_cards, held_flush, PokerHand.FLUSH, flush_cards, state
        )
        assert fin_c == 112
        assert fin_m == 8.0
        assert score == 896

        # Hand 2: Full House ['JS', 'JD', '4S', '4H', '4D'] with 2 Kings held in hand
        fh_cards = [_make_card("J", "S"), _make_card("J", "D"), _make_card("4", "S"), _make_card("4", "H"), _make_card("4", "D")]
        held_fh = [_make_card("K", "H"), _make_card("K", "S"), _make_card("2", "C")]
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            fh_cards, held_fh, PokerHand.FULL_HOUSE, fh_cards, state
        )
        assert fin_c == 78
        assert fin_m == 22.5
        assert score == 1755

    def test_fortune_teller_with_tarot_count(self):
        """Fortune Teller gains +1 Mult per Tarot card used this run."""
        c = _make_card("2", "H")
        j_ft = _make_joker("j_fortune_teller")
        j_ft.value.mult = 7.0  # From gamestate extraction
        state = _make_base_state(hand_cards=[c], jokers=[j_ft])
        state.consumeable_usage_total = {"tarot": 7}

        # Chips: 5 base + 2 card = 7. Mult = 1.0 base + 7.0 = 8.0. Score = 56.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            played_cards=[c],
            held_cards=[],
            hand_type=PokerHand.HIGH_CARD,
            scoring_cards=[c],
            state=state,
        )
        assert fin_c == 7
        assert fin_m == 8.0
        assert score == 56

    def test_the_arm_levels_down_before_scoring(self):
        """The Arm decreases level of played hand by 1 before scoring if level > 1."""
        c = _make_card("2", "H")
        b_arm = BlindInfo(type="BOSS", status="CURRENT", name="The Arm", effect="", score=1000)
        state = _make_base_state(hand_cards=[c], jokers=[])
        state.blinds = {"boss": b_arm}
        state.hands = {
            "Flush": HandLevel(level=3, played=0, played_this_round=0, chips=65, mult=8)
        }

        # Flush lvl 3 (65 chips, 8 mult) drops by 15 chips and 2 mult to lvl 2 (50 chips, 6 mult)
        flush_cards = [_make_card("A", "H"), _make_card("K", "H"), _make_card("Q", "H"), _make_card("J", "H"), _make_card("9", "H")]
        base_c, fin_c, fin_m, score, debuffed = calculate_hand_score(
            flush_cards, [], PokerHand.FLUSH, flush_cards, state
        )
        assert base_c == 50
        assert fin_m == 6.0
        assert not debuffed

    def test_perma_bonus_on_scoring_cards(self):
        """Cards with perma_bonus (e.g. from Hiker) score their extra bonus chips."""
        c = _make_card("A", "S")
        c.value.perma_bonus = 15
        state = _make_base_state(hand_cards=[c], jokers=[])

        # High Card: base 5 + card 11 + perma_bonus 15 = 31 chips. Mult = 1.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            [c], [], PokerHand.HIGH_CARD, [c], state
        )
        assert fin_c == 31
        assert score == 31

    def test_red_card_starts_at_zero_mult(self):
        """Red Card starts at 0 mult (ability.mult = 0, ability.extra = 3) and must NOT add +3."""
        c = _make_card("2", "H")
        j_red = _make_joker("j_red_card")
        j_red.value.mult = 0.0
        j_red.value.extra = 3
        state = _make_base_state(hand_cards=[c], jokers=[j_red])

        # High Card lvl 1: base 5 chips + 2 card = 7 chips. Base mult = 1.0.
        # Red card adds +0.0 mult -> total mult = 1.0. Score = 7.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            [c], [], PokerHand.HIGH_CARD, [c], state
        )
        assert fin_c == 7
        assert fin_m == 1.0
        assert score == 7

    def test_red_card_with_accumulated_mult(self):
        """Red Card with skipped booster packs adds accumulated mult."""
        c = _make_card("2", "H")
        j_red = _make_joker("j_red_card")
        j_red.value.mult = 6.0
        j_red.value.extra = 3
        state = _make_base_state(hand_cards=[c], jokers=[j_red])

        # High Card lvl 1: 7 chips. Base mult 1.0 + 6.0 = 7.0 mult. Score = 49.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            [c], [], PokerHand.HIGH_CARD, [c], state
        )
        assert fin_c == 7
        assert fin_m == 7.0
        assert score == 49

    def test_hack_and_lusty_joker_retrigger(self):
        """Hack retriggers 2, 3, 4, 5, also repeating Lusty Joker (+3 mult per Heart)."""
        c_4h = _make_card("4", "H")
        c_4s = _make_card("4", "S")
        j_hack = _make_joker("j_hack")
        j_lusty = _make_joker("j_lusty_joker")
        state = _make_base_state(hand_cards=[c_4h, c_4s], jokers=[j_hack, j_lusty])

        # Pair lvl 1: 10 chips, 2 mult.
        # 4H (4 chips, +3 mult) triggers twice with Hack: +8 chips, +6 mult.
        # 4S (4 chips, 0 mult) triggers twice with Hack: +8 chips, +0 mult.
        # Total chips: 10 + 8 + 8 = 26.
        # Total mult: 2 + 6 = 8.
        # Total score: 26 * 8 = 208.
        hand_name, scoring = identify_poker_hand([c_4h, c_4s])
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            [c_4h, c_4s], [], hand_name, scoring, state
        )
        assert fin_c == 26
        assert fin_m == 8.0
        assert score == 208

    def test_obelisk_reset_vs_scaling(self):
        """Obelisk resets when playing most played hand, scales when playing another."""
        c = _make_card("A", "S")
        j_obelisk = _make_joker("j_obelisk")
        j_obelisk.value.x_mult = 1.6
        j_obelisk.value.extra = 0.2

        state = _make_base_state(hand_cards=[c], jokers=[j_obelisk])
        state.hands["Pair"].played = 5
        state.hands["High Card"].played = 2

        # 1. Playing High Card (not most played: 2+1=3 <= 5) -> gains +0.2 -> 1.8x mult
        # High Card lvl 1: 16 chips, 1 mult * 1.8 = 1.8 mult. Score = floor(16 * 1.8) = 28.
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            [c], [], PokerHand.HIGH_CARD, [c], state
        )
        assert fin_m == 1.8
        assert score == 28

        # 2. Playing Pair (most played: 5+1=6 > 2) -> resets to 1.0x mult
        # Pair lvl 1: 10 + 11*2 = 32 chips, 2 mult * 1.0 = 2.0 mult. Score = 64.
        c2 = _make_card("A", "D")
        _, fin_c, fin_m, score, _ = calculate_hand_score(
            [c, c2], [], PokerHand.PAIR, [c, c2], state
        )
        assert fin_m == 2.0
        assert score == 64





