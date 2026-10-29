"""Unit tests for Silo RL-2: Action Space, Action Masking & Micro-Solver Delegation.

Verifies:
1. Action space dimensions (Discrete(26)) and invariant non-empty mask across all fixtures.
2. BLIND_SELECT masking (PLAY_BLIND and SKIP_BLIND on Small/Big; only PLAY_BLIND on Boss).
3. SHOP masking:
   - Budget constraints ($8 item False at $5, True at $10).
   - Joker slot limits (normal Joker False at 5/5 slots, Negative Joker True).
   - Anti-suicide guardrail on SELL_JOKER (False at 1/5 slots without urgency, True at 5/5 slots).
4. Pack delegation:
   - Auto-targeting with The Star / The Hanged Man in opened Arcana Pack.
5. Targeted Tarot conservation in SHOP:
   - Death / The Star kept in inventory during SHOP, usable in SELECTING_HAND.
6. Automatic left-to-right Joker reordering:
   - Triggers rearrange RPC when Jokers are misordered [xMult, +Chips, +Mult] on LEAVE_SHOP or buy.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bots.rl.actions import (
    ActionID,
    NUM_ACTIONS,
    compute_action_mask,
    execute_macro_action,
    get_action_space,
)
from bots.state_parser import (
    AreaInfo,
    BlindInfo,
    Card,
    CardCost,
    CardModifier,
    CardValue,
    GameScreen,
    ParsedGameState,
    RoundInfo,
    parse_gamestate,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


# ── Mock Client for RPC Delegation Verification ───────────────────────────────

class MockBalatroClient:
    """Mock BalatroClient recording all RPC calls."""

    def __init__(self):
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((method, params))
        return {"result": {"status": "ok"}}


# ── Helpers for Synthetic States ──────────────────────────────────────────────

def _make_card(
    key: str,
    suit: str | None = None,
    rank: str | None = None,
    set_name: str = "Default",
    enhancement: str | None = None,
    edition: str | None = None,
    seal: str | None = None,
    eternal: bool = False,
    rental: bool = False,
    perishable: int | None = None,
    debuff: bool = False,
    sell_cost: int = 2,
    buy_cost: int = 6,
    mult: float | None = None,
    x_mult: float | None = None,
    chips: int | None = None,
    card_id: int = 1,
) -> Card:
    c = Card(
        id=card_id,
        key=key,
        set=set_name,
        label=key,
        value=CardValue(suit=suit, rank=rank, mult=mult, x_mult=x_mult, chips=chips),
        modifier=CardModifier(
            enhancement=enhancement,
            edition=edition,
            seal=seal,
            eternal=eternal,
            rental=rental,
            perishable=perishable,
        ),
        cost=CardCost(sell=sell_cost, buy=buy_cost),
    )
    c.state.debuff = debuff
    return c


def _make_standard_deck() -> list[Card]:
    suits = ["H", "D", "C", "S"]
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    deck = []
    cid = 1
    for s in suits:
        for r in ranks:
            deck.append(_make_card(f"{s}_{r}", suit=s, rank=r, card_id=cid))
            cid += 1
    return deck


def _make_shop_state(money: int = 10, num_jokers: int = 2) -> ParsedGameState:
    jokers = [
        _make_card("j_banner", set_name="Joker", chips=30, sell_cost=3, card_id=101 + i)
        for i in range(num_jokers)
    ]
    return ParsedGameState(
        state=GameScreen.SHOP,
        round_num=2,
        ante_num=1,
        money=money,
        round=RoundInfo(hands_left=4, discards_left=3, reroll_cost=5),
        jokers=AreaInfo(count=len(jokers), limit=5, cards=jokers),
        consumables=AreaInfo(count=0, limit=2, cards=[]),
        shop=AreaInfo(
            count=2,
            limit=4,
            cards=[
                _make_card("j_cavendish", set_name="Joker", x_mult=3.0, buy_cost=8, sell_cost=4, card_id=201),
                _make_card("c_mercury", set_name="Tarot", buy_cost=3, sell_cost=1, card_id=202),
            ],
        ),
        vouchers=AreaInfo(
            count=1,
            limit=1,
            cards=[_make_card("v_grabber", set_name="Voucher", buy_cost=10, card_id=301)],
        ),
        packs=AreaInfo(
            count=1,
            limit=2,
            cards=[_make_card("p_buffoon_normal_1", set_name="Booster", buy_cost=4, card_id=401)],
        ),
        deck_cards=AreaInfo(count=52, cards=_make_standard_deck()),
    )


# ── Test 1: Action Space & Invariant Non-Vide ──────────────────────────────────

def test_action_space_definition_and_invariant():
    """Verify ActionSpace has 26 discrete actions and all fixtures have valid actions."""
    space = get_action_space()
    assert space.n == 26
    assert NUM_ACTIONS == 26

    # Test all JSON fixtures in bots/tests/fixtures/
    fixture_files = list(FIXTURES_DIR.glob("*.json"))
    assert len(fixture_files) >= 4

    for fpath in fixture_files:
        with open(fpath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        state = parse_gamestate(raw)
        mask = compute_action_mask(state)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (26,)
        assert mask.any(), f"Action mask is empty for fixture {fpath.name}"


# ── Test 2: Masquage BLIND_SELECT ─────────────────────────────────────────────

def test_blind_select_masking():
    """Verify Small/Big blinds allow Play and Skip, whereas Boss Blind forbids Skip."""
    # 1. Small Blind
    small_state = ParsedGameState(
        state=GameScreen.BLIND_SELECT,
        round_num=1,
        ante_num=1,
        blinds={
            "small": BlindInfo(type="SMALL", status="SELECT", name="Small Blind", effect="", score=300),
            "big": BlindInfo(type="BIG", status="UPCOMING", name="Big Blind", effect="", score=450),
            "boss": BlindInfo(type="BOSS", status="UPCOMING", name="The Club", effect="", score=600),
        },
    )
    mask_small = compute_action_mask(small_state)
    assert mask_small[ActionID.PLAY_BLIND]
    assert mask_small[ActionID.SKIP_BLIND]
    assert not mask_small[2:].any(), "Only actions 0 and 1 should be active in BLIND_SELECT"

    # 2. Big Blind
    big_state = ParsedGameState(
        state=GameScreen.BLIND_SELECT,
        round_num=2,
        ante_num=1,
        blinds={
            "small": BlindInfo(type="SMALL", status="DEFEATED", name="Small Blind", effect="", score=300),
            "big": BlindInfo(type="BIG", status="SELECT", name="Big Blind", effect="", score=450),
            "boss": BlindInfo(type="BOSS", status="UPCOMING", name="The Club", effect="", score=600),
        },
    )
    mask_big = compute_action_mask(big_state)
    assert mask_big[ActionID.PLAY_BLIND]
    assert mask_big[ActionID.SKIP_BLIND]
    assert not mask_big[2:].any()

    # 3. Boss Blind
    boss_state = ParsedGameState(
        state=GameScreen.BLIND_SELECT,
        round_num=3,
        ante_num=1,
        blinds={
            "small": BlindInfo(type="SMALL", status="DEFEATED", name="Small Blind", effect="", score=300),
            "big": BlindInfo(type="BIG", status="DEFEATED", name="Big Blind", effect="", score=450),
            "boss": BlindInfo(type="BOSS", status="SELECT", name="The Club", effect="", score=600),
        },
    )
    mask_boss = compute_action_mask(boss_state)
    assert mask_boss[ActionID.PLAY_BLIND]
    assert not mask_boss[ActionID.SKIP_BLIND]
    assert not mask_boss[2:].any()


# ── Test 3: Masquage SHOP (Budget, Slots & Garde-fou Vente) ─────────────────────

def test_shop_budget_masking():
    """Verify that an item costing $8 is False at $5 and True at $10."""
    state_poor = _make_shop_state(money=5)
    # Card 0 is Cavendish (buy_cost=8)
    mask_poor = compute_action_mask(state_poor)
    assert not mask_poor[ActionID.BUY_SHOP_CARD_0]
    # Card 1 is Mercury (buy_cost=3)
    assert mask_poor[ActionID.BUY_SHOP_CARD_1]

    state_rich = _make_shop_state(money=10)
    mask_rich = compute_action_mask(state_rich)
    assert mask_rich[ActionID.BUY_SHOP_CARD_0]
    assert mask_rich[ActionID.BUY_SHOP_CARD_1]


def test_shop_joker_slot_limits_and_negative():
    """Verify that 5/5 slots masks normal Jokers to False, but Negative Jokers to True."""
    state_full = _make_shop_state(money=30, num_jokers=5)
    # Shop card 0 is normal Cavendish ($8, slot needed)
    # Shop card 1 is Negative Cavendish ($8, 0 net slot needed)
    state_full.shop.cards = [
        _make_card("j_cavendish", set_name="Joker", buy_cost=8, card_id=501),
        _make_card("j_cavendish", set_name="Joker", edition="NEGATIVE", buy_cost=8, card_id=502),
    ]

    mask = compute_action_mask(state_full)
    assert not mask[ActionID.BUY_SHOP_CARD_0]
    assert mask[ActionID.BUY_SHOP_CARD_1]


def test_shop_sell_joker_guardrail():
    """Verify anti-suicide guardrail on SELL_JOKER in early game vs full build."""
    # 1. Early game with 1 joker and 4 free slots: SELL_JOKER is False to prevent random suicide
    state_early = _make_shop_state(money=10, num_jokers=1)
    mask_early = compute_action_mask(state_early)
    assert not any(mask_early[ActionID.SELL_JOKER_0:ActionID.SELL_JOKER_4 + 1])

    # 2. When slots are full (5/5), SELL_JOKER becomes True for vendable jokers
    state_full = _make_shop_state(money=10, num_jokers=5)
    mask_full = compute_action_mask(state_full)
    assert mask_full[ActionID.SELL_JOKER_0]


# ── Test 4: Délégation du Ciblage dans les Packs (SELECT_PACK_CARD) ────────────

def test_pack_targeting_delegation_the_star():
    """Verify that selecting The Star in an Arcana pack automatically targets up to 3 non-Diamond cards."""
    # Hand with 5 Spades and 3 Diamonds
    hand_cards = [
        _make_card("S_2", suit="S", rank="2", card_id=1),
        _make_card("S_3", suit="S", rank="3", card_id=2),
        _make_card("S_4", suit="S", rank="4", card_id=3),
        _make_card("S_5", suit="S", rank="5", card_id=4),
        _make_card("S_6", suit="S", rank="6", card_id=5),
        _make_card("D_A", suit="D", rank="A", card_id=6),
        _make_card("D_K", suit="D", rank="K", card_id=7),
        _make_card("D_Q", suit="D", rank="Q", card_id=8),
    ]

    pack_state = ParsedGameState(
        state=GameScreen.BOOSTER_OPENED,
        round_num=2,
        ante_num=1,
        money=10,
        hand=AreaInfo(count=len(hand_cards), limit=8, cards=hand_cards),
        pack=AreaInfo(
            count=2,
            limit=3,
            cards=[
                _make_card("c_star", set_name="Tarot", card_id=801),  # Converts up to 3 cards to Diamonds
                _make_card("c_hermit", set_name="Tarot", card_id=802),
            ],
        ),
        deck_cards=AreaInfo(count=52, cards=_make_standard_deck()),
    )

    mask = compute_action_mask(pack_state)
    assert mask[ActionID.SELECT_PACK_CARD_0]

    # Execute action via mock client
    client = MockBalatroClient()
    result = execute_macro_action(ActionID.SELECT_PACK_CARD_0, pack_state, client)

    assert result["rpc_method"] == "pack"
    assert len(result["targets"]) == 3
    # Targets should be non-diamond indices (0, 1, 2)
    assert set(result["targets"]).issubset({0, 1, 2, 3, 4})
    assert client.calls[0] == ("pack", {"card": 0, "targets": result["targets"]})


def test_pack_targeting_delegation_hanged_man():
    """Verify that selecting The Hanged Man automatically targets the 2 weakest non-protected cards."""
    hand_cards = [
        _make_card("S_2", suit="S", rank="2", card_id=1),  # weak 1
        _make_card("H_3", suit="H", rank="3", card_id=2),  # weak 2
        _make_card("D_A", suit="D", rank="A", edition="POLYCHROME", card_id=3),  # protected
        _make_card("C_K", suit="C", rank="K", card_id=4),
    ]

    pack_state = ParsedGameState(
        state=GameScreen.BOOSTER_OPENED,
        round_num=2,
        ante_num=1,
        hand=AreaInfo(count=len(hand_cards), limit=8, cards=hand_cards),
        pack=AreaInfo(
            count=1,
            limit=2,
            cards=[_make_card("c_hanged_man", set_name="Tarot", card_id=901)],
        ),
        deck_cards=AreaInfo(count=52, cards=_make_standard_deck()),
    )

    client = MockBalatroClient()
    result = execute_macro_action(ActionID.SELECT_PACK_CARD_0, pack_state, client)

    assert result["rpc_method"] == "pack"
    assert len(result["targets"]) == 2
    # Indices 0 and 1 are the weakest non-protected cards
    assert set(result["targets"]) == {0, 1}
    assert client.calls[0] == ("pack", {"card": 0, "targets": result["targets"]})


# ── Test 5: Conservation des Tarots Ciblés en SHOP ─────────────────────────────

def test_targeted_tarot_conservation_in_shop():
    """Verify targeted Tarots (Death) are not consumed/sold in SHOP with free slots, but usable in combat."""
    death_card = _make_card("c_death", set_name="Tarot", card_id=701)

    # 1. In SHOP with 1/2 consumable slots: Death is conserved (False)
    shop_state = _make_shop_state(money=10, num_jokers=2)
    shop_state.consumables = AreaInfo(count=1, limit=2, cards=[death_card])
    shop_state.hand = AreaInfo(count=0, limit=8, cards=[])

    mask_shop = compute_action_mask(shop_state)
    assert not mask_shop[ActionID.USE_OR_SELL_CONSUMABLE_0]

    # 2. In SELECTING_HAND with cards in hand: Death becomes usable (True)
    combat_hand = [
        _make_card("S_2", suit="S", rank="2", card_id=1),  # weak
        _make_card("D_A", suit="D", rank="A", card_id=2),  # strong
    ]
    combat_state = ParsedGameState(
        state=GameScreen.SELECTING_HAND,
        round_num=3,
        ante_num=1,
        hand=AreaInfo(count=2, limit=8, cards=combat_hand),
        consumables=AreaInfo(count=1, limit=2, cards=[death_card]),
    )
    mask_combat = compute_action_mask(combat_state)
    assert mask_combat[ActionID.USE_OR_SELL_CONSUMABLE_0]


# ── Test 6: Tri Automatique des Jokers ─────────────────────────────────────────

def test_automatic_joker_reordering_on_leave_shop():
    """Verify that LEAVE_SHOP automatically rearranges misordered Jokers [xMult, Chips, AddMult]."""
    # Misordered jokers: Cavendish (xMult), Banner (+Chips), Abstract (+Mult)
    misordered_jokers = [
        _make_card("j_cavendish", set_name="Joker", x_mult=3.0, card_id=1),
        _make_card("j_banner", set_name="Joker", chips=30, card_id=2),
        _make_card("j_abstract", set_name="Joker", mult=3.0, card_id=3),
    ]

    shop_state = _make_shop_state(money=10)
    shop_state.jokers = AreaInfo(count=3, limit=5, cards=misordered_jokers)

    client = MockBalatroClient()
    result = execute_macro_action(ActionID.LEAVE_SHOP, shop_state, client)

    assert result["auto_rearranged"] is True
    assert result["rpc_method"] == "next_round"

    # Verify rearrange was called with optimal order [1, 2, 0] (Banner, Abstract, Cavendish)
    assert len(client.calls) == 2
    rearrange_call = client.calls[0]
    next_round_call = client.calls[1]

    assert rearrange_call[0] == "rearrange"
    assert rearrange_call[1] == {"jokers": [1, 2, 0]}
    assert next_round_call[0] == "next_round"


def test_spectral_cards_zero_jokers_masked_out():
    """Verify that Hex, Ankh, and Ectoplasm are strictly masked out when player has 0 Jokers."""
    hex_card = _make_card("c_hex", set_name="Spectral", card_id=101)
    ankh_card = _make_card("c_ankh", set_name="Spectral", card_id=102)
    ecto_card = _make_card("c_ectoplasm", set_name="Spectral", card_id=103)

    # 1. Inside a booster pack with 0 Jokers (and held consumable in inventory)
    tarot_held = _make_card("c_magician", set_name="Tarot", card_id=201)
    booster_state_0_jokers = ParsedGameState(
        state=GameScreen.BOOSTER_OPENED,
        pack=AreaInfo(count=3, limit=3, cards=[hex_card, ankh_card, ecto_card]),
        consumables=AreaInfo(count=1, limit=2, cards=[tarot_held]),
        jokers=AreaInfo(count=0, limit=5, cards=[]),
    )
    mask = compute_action_mask(booster_state_0_jokers)
    assert not mask[ActionID.SELECT_PACK_CARD_0]  # Hex masked out!
    assert not mask[ActionID.SELECT_PACK_CARD_1]  # Ankh masked out!
    assert not mask[ActionID.SELECT_PACK_CARD_2]  # Ectoplasm masked out!
    assert not mask[ActionID.USE_OR_SELL_CONSUMABLE_0]  # Held consumables CANNOT be used during booster!
    assert mask[ActionID.SKIP_PACK]  # Skip pack remains valid

    # 2. Inside a booster pack with 1 Joker already having Polychrome edition
    poly_joker = _make_card("j_joker", set_name="Joker", card_id=1, edition="Polychrome")
    booster_state_poly_joker = ParsedGameState(
        state=GameScreen.BOOSTER_OPENED,
        pack=AreaInfo(count=3, limit=3, cards=[hex_card, ankh_card, ecto_card]),
        jokers=AreaInfo(count=1, limit=5, cards=[poly_joker]),
        hand=AreaInfo(count=8, limit=8, cards=[]),
    )
    mask_poly = compute_action_mask(booster_state_poly_joker)
    assert not mask_poly[ActionID.SELECT_PACK_CARD_0]  # Hex masked out because Joker already has edition!
    assert mask_poly[ActionID.SELECT_PACK_CARD_1]  # Ankh valid!

    # 3. Inside a booster pack with 1 editionless Joker
    joker = _make_card("j_joker", set_name="Joker", card_id=2)
    booster_state_1_joker = ParsedGameState(
        state=GameScreen.BOOSTER_OPENED,
        pack=AreaInfo(count=3, limit=3, cards=[hex_card, ankh_card, ecto_card]),
        jokers=AreaInfo(count=1, limit=5, cards=[joker]),
        hand=AreaInfo(count=8, limit=8, cards=[]),
    )
    mask_1 = compute_action_mask(booster_state_1_joker)
    assert mask_1[ActionID.SELECT_PACK_CARD_0]  # Hex valid!
    assert mask_1[ActionID.SELECT_PACK_CARD_1]  # Ankh valid!
    assert mask_1[ActionID.SELECT_PACK_CARD_2]  # Ectoplasm valid!


def test_fool_tarot_masking_logic() -> None:
    """Verify The Fool is strictly masked out when unusable (no Tarot used or last was The Fool)."""
    fool_card = _make_card("c_fool", set_name="Tarot", card_id=301)

    # 1. In SHOP with NO tarot used yet and slots not full
    state_no_usage = ParsedGameState(
        state=GameScreen.SHOP,
        consumables=AreaInfo(count=1, limit=2, cards=[fool_card]),
        last_tarot_planet=None,
        consumeable_usage_total={},
    )
    mask = compute_action_mask(state_no_usage)
    assert not mask[ActionID.USE_OR_SELL_CONSUMABLE_0]  # Unusable, not vendable because 1/2 slots

    # 2. In SHOP with last_tarot_planet == 'c_fool'
    state_last_fool = ParsedGameState(
        state=GameScreen.SHOP,
        consumables=AreaInfo(count=1, limit=2, cards=[fool_card]),
        last_tarot_planet="c_fool",
        consumeable_usage_total={"tarot": 1},
    )
    mask_last_fool = compute_action_mask(state_last_fool)
    assert not mask_last_fool[ActionID.USE_OR_SELL_CONSUMABLE_0]

    # 3. In SHOP with last_tarot_planet == 'c_hermit' -> Valid to use!
    state_valid_fool = ParsedGameState(
        state=GameScreen.SHOP,
        consumables=AreaInfo(count=1, limit=2, cards=[fool_card]),
        last_tarot_planet="c_hermit",
        consumeable_usage_total={"tarot": 1},
    )
    mask_valid = compute_action_mask(state_valid_fool)
    assert mask_valid[ActionID.USE_OR_SELL_CONSUMABLE_0]

    # 4. In BOOSTER pack with no prior tarot -> Fool is masked out!
    state_booster_unusable = ParsedGameState(
        state=GameScreen.BOOSTER_OPENED,
        pack=AreaInfo(count=1, limit=1, cards=[fool_card]),
        consumables=AreaInfo(count=0, limit=2, cards=[]),
        last_tarot_planet=None,
    )
    mask_pack = compute_action_mask(state_booster_unusable)
    assert not mask_pack[ActionID.SELECT_PACK_CARD_0]
    assert mask_pack[ActionID.SKIP_PACK]

    # 5. In BOOSTER pack with valid prior tarot and room in inventory -> Valid!
    state_booster_valid = ParsedGameState(
        state=GameScreen.BOOSTER_OPENED,
        pack=AreaInfo(count=1, limit=1, cards=[fool_card]),
        consumables=AreaInfo(count=0, limit=2, cards=[]),
        last_tarot_planet="c_mercury",
    )
    mask_pack_valid = compute_action_mask(state_booster_valid)
    assert mask_pack_valid[ActionID.SELECT_PACK_CARD_0]


def test_joker_synergy_discard_priorities() -> None:
    """Verify combat solver discards synergize with active Jokers (Blackboard, Bloodstone, Photograph, etc.)."""
    from bots.combat_solver import discard_card_priority

    heart_card = _make_card("H_4", suit="H", rank="4")
    spade_card = _make_card("S_4", suit="S", rank="4")
    face_card = _make_card("H_K", suit="H", rank="K")
    two_card = _make_card("D_2", suit="D", rank="2")

    # Blackboard synergy: Hearts (Red) must have MUCH higher discard priority than Spades (Black)
    blackboard_joker = _make_card("j_blackboard", set_name="Joker")
    state_blackboard = ParsedGameState(
        state=GameScreen.SELECTING_HAND,
        jokers=AreaInfo(count=1, limit=5, cards=[blackboard_joker]),
    )
    p_heart = discard_card_priority(heart_card, state_blackboard)
    p_spade = discard_card_priority(spade_card, state_blackboard)
    assert p_heart > p_spade + 300.0  # Hearts discarded in strong priority over Spades

    # Bloodstone synergy: Non-hearts discarded in priority over Hearts
    bloodstone_joker = _make_card("j_bloodstone", set_name="Joker")
    state_bloodstone = ParsedGameState(
        state=GameScreen.SELECTING_HAND,
        jokers=AreaInfo(count=1, limit=5, cards=[bloodstone_joker]),
    )
    assert discard_card_priority(spade_card, state_bloodstone) > discard_card_priority(heart_card, state_bloodstone)

    # Photograph synergy: Non-face cards discarded in priority over Face cards
    photo_joker = _make_card("j_photograph", set_name="Joker")
    state_photo = ParsedGameState(
        state=GameScreen.SELECTING_HAND,
        jokers=AreaInfo(count=1, limit=5, cards=[photo_joker]),
    )
    assert discard_card_priority(heart_card, state_photo) > discard_card_priority(face_card, state_photo)

    # Wee Joker synergy: Rank 2 is protected (much lower discard priority)
    wee_joker = _make_card("j_wee", set_name="Joker")
    state_wee = ParsedGameState(
        state=GameScreen.SELECTING_HAND,
        jokers=AreaInfo(count=1, limit=5, cards=[wee_joker]),
    )
    assert discard_card_priority(two_card, state_wee) < discard_card_priority(heart_card, state_wee)

