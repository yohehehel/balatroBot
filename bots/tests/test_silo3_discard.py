"""Unit tests for Silo 3: Targeted Discard Heuristics & Combat Decision Loop.

Validates the 5-rule decision tree in bots/combat_solver.py:
1. Kill Shot immediate victory -> PLAY
2. Zero discards left / Discards blocked -> PLAY (recycles hand with kickers)
3. Flush Draw -> Keep 4 matching suit cards, DISCARD others (up to 5)
4. Boss Blind suit debuff adaptation -> Do NOT chase debuffed suit (The Head)
5. Straight Draw -> Keep 4 sequence cards, DISCARD others (up to 5)
6. Steel card protection & Purple Seal priority during discard
7. Trading Card special 1-card discard destruction
8. Anti-discard Jokers (Green Joker / Ramen) preservation
"""

from __future__ import annotations

from bots.combat_solver import (
    CombatAction,
    PokerHand,
    decide_combat_action,
    discard_card_priority,
    get_debuffed_suits,
    pick_discard_indices,
    select_best_play,
)
from bots.state_parser import (
    AreaInfo,
    BlindInfo,
    Card,
    CardModifier,
    CardState,
    CardValue,
    HandLevel,
    ParsedGameState,
    RoundInfo,
)

# ── Helpers ───────────────────────────────────────────────────────────────

def _make_card(
    rank: str,
    suit: str,
    card_id: int = 1,
    enhancement: str | None = None,
    edition: str | None = None,
    seal: str | None = None,
    debuff: bool = False,
) -> Card:
    return Card(
        id=card_id,
        key=f"c_{suit}_{rank}",
        set="Default",
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
) -> Card:
    return Card(
        id=joker_id,
        key=key,
        label=key,
        modifier=CardModifier(edition=edition),
        state=CardState(debuff=debuff),
    )


def _make_combat_state(
    hand_cards: list[Card],
    jokers: list[Card] | None = None,
    discards_left: int = 3,
    discards_used: int = 0,
    hands_left: int = 4,
    chips: int = 0,
    blind_score: int = 300,
    blind_name: str = "Small Blind",
    blind_type: str = "SMALL",
    blind_effect: str = "",
    money: int = 4,
    deck_cards: list[Card] | None = None,
) -> ParsedGameState:
    return ParsedGameState(
        state="SELECTING_HAND",
        round_num=1,
        ante_num=1,
        money=money,
        round=RoundInfo(
            hands_left=hands_left,
            discards_left=discards_left,
            discards_used=discards_used,
            chips=chips,
        ),
        blinds={
            "active": BlindInfo(
                type=blind_type,
                status="CURRENT",
                name=blind_name,
                effect=blind_effect,
                score=blind_score,
            )
        },
        hand=AreaInfo(count=len(hand_cards), cards=hand_cards),
        jokers=AreaInfo(count=len(jokers or []), cards=jokers or []),
        deck_cards=AreaInfo(count=len(deck_cards or []), cards=deck_cards or []),
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


# ── Test 1: Kill Shot immediate victory ───────────────────────────────────

def test_kill_shot_priority_over_discard():
    """Rule 1: If current hand can win immediately, PLAY is chosen even with 4 discards left."""
    # Hand contains a Flush (5 Hearts: A, K, Q, J, 9) scoring ~35 chips * 4 mult + card ranks > 300 pts
    # Let's set blind_score to 100, which any Flush easily exceeds (~130 pts)
    cards = [
        _make_card("A", "H", 1),
        _make_card("K", "H", 2),
        _make_card("Q", "H", 3),
        _make_card("J", "H", 4),
        _make_card("9", "H", 5),
        _make_card("2", "C", 6),
        _make_card("3", "D", 7),
        _make_card("4", "S", 8),
    ]
    state = _make_combat_state(
        hand_cards=cards,
        discards_left=4,
        blind_score=100,  # Flush easily beats 100
    )

    decision = decide_combat_action(state)

    assert decision.action == CombatAction.PLAY
    assert "Kill Shot" in decision.reason
    assert decision.predicted_score >= 100
    # Winner cards should be the flush cards
    assert len(decision.indices) == 5


# ── Test 2: Zero discards left (Rule 2) ───────────────────────────────────

def test_zero_discards_recycles_with_kickers():
    """Rule 2: When discards_left == 0 and hand is weak, PLAY with kickers to recycle cards."""
    # Hand contains a Pair of 2s and trash cards, blind_score is 1000 (cannot win in 1 shot)
    cards = [
        _make_card("2", "H", 1),
        _make_card("2", "D", 2),
        _make_card("4", "C", 3),
        _make_card("6", "S", 4),
        _make_card("8", "H", 5),
        _make_card("T", "D", 6),
        _make_card("K", "C", 7),
        _make_card("A", "S", 8),
    ]
    state = _make_combat_state(
        hand_cards=cards,
        discards_left=0,
        hands_left=3,
        blind_score=1000,
    )

    decision = decide_combat_action(state)

    assert decision.action == CombatAction.PLAY
    # Should play 5 cards (the pair + 3 kickers) to draw fresh cards
    assert len(decision.indices) == 5
    assert "Recycle" in decision.reason or "No discards left" in decision.reason


def test_zero_discards_single_hand_left_forced_play():
    """Rule 2: When discards_left == 0 and hands_left == 1, forced best play."""
    cards = [
        _make_card("K", "H", 1),
        _make_card("K", "D", 2),
        _make_card("4", "C", 3),
    ]
    state = _make_combat_state(
        hand_cards=cards,
        discards_left=0,
        hands_left=1,
        blind_score=1000,
    )

    decision = decide_combat_action(state)

    assert decision.action == CombatAction.PLAY
    assert "No discards left" in decision.reason or "Forced Play" in decision.reason


# ── Test 3: Flush Draw (Rule 4.A) ─────────────────────────────────────────

def test_flush_draw_keeps_suit_discards_others():
    """Rule 4.A: With 4 cards of one suit, keep them and discard up to 5 non-suit cards."""
    # 4 Hearts: A, K, T, 7 and 4 off-suits: 2C, 3D, 4S, 5C
    cards = [
        _make_card("A", "H", 1),
        _make_card("K", "H", 2),
        _make_card("T", "H", 3),
        _make_card("7", "H", 4),
        _make_card("2", "C", 5),
        _make_card("4", "D", 6),
        _make_card("6", "S", 7),
        _make_card("8", "C", 8),
    ]
    state = _make_combat_state(
        hand_cards=cards,
        discards_left=3,
        blind_score=1000,  # High blind to prevent kill shot
    )

    decision = decide_combat_action(state)

    assert decision.action == CombatAction.DISCARD
    assert "Flush Draw (H)" in decision.reason
    # Kept cards must be indices 0, 1, 2, 3 (the 4 Hearts)
    assert set(decision.indices) == {4, 5, 6, 7}
    assert len(decision.indices) == 4


# ── Test 4: Boss Blind suit debuff adaptation (The Head) ──────────────────

def test_boss_blind_the_head_avoids_hearts_flush():
    """Rule 4.A: Face to The Head (Hearts debuffed), do NOT chase Hearts flush."""
    # Hand has 4 Hearts (A, K, T, 7) and 4 off-suits
    # Normal blind would chase Hearts flush, but The Head debuffs Hearts
    cards = [
        _make_card("A", "H", 1),
        _make_card("K", "H", 2),
        _make_card("T", "H", 3),
        _make_card("7", "H", 4),
        _make_card("9", "C", 5),
        _make_card("9", "D", 6),
        _make_card("4", "S", 7),
        _make_card("2", "C", 8),
    ]
    state = _make_combat_state(
        hand_cards=cards,
        discards_left=3,
        blind_score=1000,
        blind_name="The Head",
        blind_type="BOSS",
    )

    # Verify get_debuffed_suits detects Hearts
    assert get_debuffed_suits(state) == {"H"}

    decision = decide_combat_action(state)

    # Must NOT be a Flush Draw on Hearts!
    assert "Flush Draw (H)" not in decision.reason


# ── Test 5: Straight Draw (Rule 4.B) ──────────────────────────────────────

def test_straight_draw_keeps_sequence_discards_trash():
    """Rule 4.B: 6, 7, 8, 9 across different suits forms an open-ended straight draw."""
    # Straight draw cards: 6H, 7D, 8C, 9S. Trash cards: 2C, 2D, 3H, KSpades
    cards = [
        _make_card("6", "H", 1),
        _make_card("7", "D", 2),
        _make_card("8", "C", 3),
        _make_card("9", "S", 4),
        _make_card("2", "C", 5),
        _make_card("2", "D", 6),
        _make_card("3", "H", 7),
        _make_card("K", "S", 8),
    ]
    state = _make_combat_state(
        hand_cards=cards,
        discards_left=2,
        blind_score=2000,
    )

    decision = decide_combat_action(state)

    assert decision.action == CombatAction.DISCARD
    assert "Straight Draw" in decision.reason
    # 6, 7, 8, 9 are at indices 0, 1, 2, 3 -> must be KEPT
    for idx in (0, 1, 2, 3):
        assert idx not in decision.indices
    # The 4 trash cards should be discarded
    assert set(decision.indices) == {4, 5, 6, 7}


# ── Test 6: Steel Protection & Purple Seal Priority ───────────────────────

def test_steel_card_protected_purple_seal_prioritised():
    """Rule 5: Purple Seal is discarded first for Tarot; Steel card is preserved in hand."""
    # 4 Hearts kept for flush draw: AH, KH, QH, JH (indices 0, 1, 2, 3)
    # 4 candidates for discard:
    # - idx 4: 2 of Clubs with PURPLE seal -> Tarot trigger (+1000 priority)
    # - idx 5: King of Spades with STEEL enhancement -> x1.5 Mult held (-1000 priority)
    # - idx 6: 4 of Spades plain (+11 priority)
    # - idx 7: 8 of Diamonds plain (+7 priority)
    cards = [
        _make_card("A", "H", 1),
        _make_card("K", "H", 2),
        _make_card("Q", "H", 3),
        _make_card("J", "H", 4),
        _make_card("2", "C", 5, seal="PURPLE"),
        _make_card("K", "S", 6, enhancement="STEEL"),
        _make_card("4", "S", 7),
        _make_card("8", "D", 8),
    ]
    state = _make_combat_state(
        hand_cards=cards,
        discards_left=2,
        blind_score=5000,
    )

    decision = decide_combat_action(state)

    assert decision.action == CombatAction.DISCARD
    # Purple seal card (idx 4) MUST be in discard list
    assert 4 in decision.indices
    # Steel card (idx 5) MUST NOT be in discard list (protected!)
    assert 5 not in decision.indices
    # Discard count must be between 1 and 5
    assert 1 <= len(decision.indices) <= 5


def test_blue_seal_and_gold_card_protected():
    """Rule 5: Blue Seal and Gold enhancement have negative priority and are protected."""
    card_blue = _make_card("5", "C", seal="BLUE")
    card_gold = _make_card("7", "D", enhancement="GOLD")
    card_steel = _make_card("8", "H", enhancement="STEEL")
    card_purple = _make_card("2", "S", seal="PURPLE")

    state = _make_combat_state(hand_cards=[card_blue, card_gold, card_steel, card_purple])

    assert discard_card_priority(card_purple, state) > 500.0
    assert discard_card_priority(card_blue, state) < 0.0
    assert discard_card_priority(card_gold, state) < 0.0
    assert discard_card_priority(card_steel, state) < 0.0


# ── Test 7: Trading Card Special Case ─────────────────────────────────────

def test_trading_card_first_discard_discards_single_weak_card():
    """Rule 4: Trading Card on first discard of round destroys exactly 1 card for $3."""
    cards = [
        _make_card("A", "H", 1),
        _make_card("K", "H", 2),
        _make_card("Q", "H", 3),
        _make_card("9", "H", 4),
        _make_card("2", "C", 5),  # Plain weak card (lowest rank in hand)
        _make_card("4", "D", 6),
        _make_card("6", "S", 7),
        _make_card("8", "C", 8),
    ]
    jokers = [_make_joker("j_trading")]
    state = _make_combat_state(
        hand_cards=cards,
        jokers=jokers,
        discards_left=3,
        discards_used=0,  # 1st discard of round
        blind_score=5000,
    )

    decision = decide_combat_action(state)

    assert decision.action == CombatAction.DISCARD
    assert "Trading Card" in decision.reason
    assert decision.indices == [4]  # 2 of Clubs (lowest rank plain card)


# ── Test 8: Anti-Discard Jokers (Green Joker / Ramen) ─────────────────────

def test_green_joker_blocks_discard_when_pace_is_met():
    """Rule 2: Green Joker blocks discarding when current hand meets required pace."""
    # Hand contains a Pair of Aces scoring ~35 pts
    # Blind score is 60, hands_left = 3 -> required pace is 20 pts per hand
    # Since 35 >= 20, discarding would penalize Green Joker (-1 Mult), so PLAY is chosen!
    cards = [
        _make_card("A", "H", 1),
        _make_card("A", "D", 2),
        _make_card("4", "C", 3),
        _make_card("6", "S", 4),
        _make_card("8", "H", 5),
    ]
    jokers = [_make_joker("j_green_joker")]
    state = _make_combat_state(
        hand_cards=cards,
        jokers=jokers,
        discards_left=3,
        hands_left=3,
        blind_score=60,
    )

    decision = decide_combat_action(state)

    assert decision.action == CombatAction.PLAY
    assert "Anti-discard" in decision.reason or "Tempo" in decision.reason or "Kill Shot" in decision.reason


# ── Test 9: Safety limits ─────────────────────────────────────────────────

def test_pick_discard_indices_never_empty_and_max_5():
    """Safety: pick_discard_indices strictly returns 1 to 5 indices."""
    cards = [_make_card(str(r), "H", i) for i, r in enumerate(range(2, 10))]
    state = _make_combat_state(hand_cards=cards)

    # With all 8 cards available for discard:
    indices = pick_discard_indices(state, keep_indices=set())
    assert 1 <= len(indices) <= 5

    # With 7 cards kept:
    indices2 = pick_discard_indices(state, keep_indices={0, 1, 2, 3, 4, 5, 6})
    assert len(indices2) == 1
    assert indices2[0] == 7


# ── Test 10: The Needle survival discard ──────────────────────────────────

def test_the_needle_forces_discard_when_score_insufficient_and_discards_left():
    """Rule 1.B: When hands_left == 1 and score < chips_needed, PLAY is forbidden if discards remain."""
    cards = [
        _make_card("A", "H", 1),
        _make_card("K", "H", 2),
        _make_card("Q", "H", 3),
        _make_card("J", "H", 4),
        _make_card("9", "H", 5),
    ]
    # Blind needs 50,000 chips; the Flush only scores ~150 chips.
    state = _make_combat_state(
        hand_cards=cards,
        discards_left=2,
        hands_left=1,
        blind_score=50000,
        blind_name="The Needle",
        blind_effect="bl_needle",
        blind_type="BOSS",
    )

    decision = decide_combat_action(state)

    # Even though it's a Flush (Major hand), playing would guarantee instant loss
    assert decision.action == CombatAction.DISCARD


# ── Test 11: The Psychic strict 5-card requirement ────────────────────────

def test_the_psychic_strictly_requires_5_cards_on_play():
    """The Psychic boss debuffs plays with < 5 cards; solver must pad to exactly 5 cards."""
    cards = [
        _make_card("A", "S", 1),
        _make_card("A", "H", 2),
        _make_card("4", "C", 3),
        _make_card("6", "D", 4),
        _make_card("8", "S", 5),
        _make_card("9", "C", 6),
        _make_card("K", "D", 7),
    ]
    # Kill shot with Pair of Aces (needs 40 chips, Pair scores ~50)
    state = _make_combat_state(
        hand_cards=cards,
        discards_left=2,
        hands_left=3,
        blind_score=40,
        blind_name="The Psychic",
        blind_effect="bl_psychic",
        blind_type="BOSS",
    )

    decision = decide_combat_action(state)

    assert decision.action == CombatAction.PLAY
    assert len(decision.indices) == 5
    assert 0 in decision.indices and 1 in decision.indices


# ── Test 12: The Ox economy protection ────────────────────────────────────

def test_the_ox_avoids_most_played_hand_when_wealthy():
    """Against The Ox, if player has money > 10, avoid playing most played hand if safe alternative exists."""
    # 5 Spades cards with a Pair of Aces:
    # A of Spades, A of Clubs, K of Spades, Q of Spades, J of Spades
    cards = [
        _make_card("A", "S", 1),
        _make_card("A", "C", 2),
        _make_card("K", "S", 3),
        _make_card("Q", "S", 4),
        _make_card("J", "S", 5),
    ]
    # Flush played 25 times; Pair played 2 times
    state = _make_combat_state(
        hand_cards=cards,
        discards_left=2,
        hands_left=3,
        blind_score=30,  # Both Flush and Pair will win immediately (Kill shot)
        blind_name="The Ox",
        blind_effect="bl_ox",
        blind_type="BOSS",
        money=35,
    )
    state.hands["Flush"].played = 25
    state.hands["Pair"].played = 2

    best = select_best_play(state)
    assert best is not None
    # Flush is most played: Ox would set money to 0! Solver must choose Pair instead
    assert best.hand_type != "Flush"
    assert best.hand_type == "Pair"

    decision = decide_combat_action(state)
    assert decision.action == CombatAction.PLAY
    assert decision.candidate.hand_type == "Pair"


# ── Test 13: Blackboard red kicker purge ──────────────────────────────────

def test_blackboard_purges_red_cards_via_kickers():
    """Blackboard x3 Mult triggers when all held cards are black; solver includes red cards as kickers."""
    # 2 Black cards (Pair of 2s), 2 Red cards (4H, 5D), and 2 Black cards (8S, 9C)
    cards = [
        _make_card("2", "S", 1),
        _make_card("2", "C", 2),
        _make_card("4", "H", 3),
        _make_card("5", "D", 4),
        _make_card("8", "S", 5),
        _make_card("9", "C", 6),
    ]
    j_bb = _make_joker("j_blackboard")
    state = _make_combat_state(
        hand_cards=cards,
        jokers=[j_bb],
        discards_left=2,
        hands_left=3,
        blind_score=50,  # 2-card Pair scores 28 (<50); 4-card Pair with Blackboard scores 84 (>=50)
    )

    best = select_best_play(state)
    assert best is not None
    # Must play 4 cards (including the 2 red cards: indices 2 and 3)
    assert 2 in best.indices and 3 in best.indices
    # Remaining held cards must all be black (8S, 9C)
    held = [cards[i] for i in range(len(cards)) if i not in set(best.indices)]
    assert all(c.value.suit in ("S", "C") for c in held)
    assert best.score >= 50

