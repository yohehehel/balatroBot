"""Unit tests for Silo RL-1: Observation Space, Feature Engineering & estimated_power_ratio.

Verifies:
1. Gymnasium conformity (observation_space.contains(obs)) across all fixtures (menu, blind_select, combat, shop, pack)
   with zero NaNs or Infs.
2. Joker overflow handling (> 10 Jokers, with Negative and Eternal jokers):
   shape (10, D_entity) is preserved without IndexError, and real count is exposed in context_features.
3. Monotonic sensitivity of estimated_power_ratio:
   - Increasing with strong xMult joker (j_cavendish).
   - Increasing with dominant hand level-up.
   - Decreasing when target chips scale from Ante 1 (300) to Ante 6 (20,000).
4. Sensitivity of deck_features:
   - Converting 3 cards to Diamonds increases the Diamonds proportion by exactly 3 / total_cards.
   - Converting 3 cards to Steel increases the Steel proportion by exactly 3 / total_cards.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from bots.rl.constants import (
    D_CTX,
    D_DECK,
    D_ENTITY,
    D_HAND,
    MAX_JOKERS,
    MAX_OFFERS,
)
from bots.rl.features import (
    compute_estimated_power_ratio,
    extract_features,
    get_observation_space,
)
from bots.state_parser import (
    AreaInfo,
    BlindInfo,
    Card,
    CardCost,
    CardModifier,
    CardValue,
    GameScreen,
    HandLevel,
    ParsedGameState,
    RoundInfo,
    parse_gamestate,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


# ── Helpers to build synthetic states ─────────────────────────────────────────

def _make_card(
    key: str,
    suit: str | None = None,
    rank: str | None = None,
    set_name: str = "Default",
    enhancement: str | None = None,
    edition: str | None = None,
    seal: str | None = None,
    eternal: bool = False,
    sell_cost: int = 1,
    buy_cost: int = 2,
    mult: float | None = None,
    x_mult: float | None = None,
    chips: int | None = None,
    card_id: int = 1,
) -> Card:
    return Card(
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
        ),
        cost=CardCost(sell=sell_cost, buy=buy_cost),
    )


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


def _make_synthetic_shop_state() -> ParsedGameState:
    return ParsedGameState(
        state=GameScreen.SHOP,
        round_num=3,
        ante_num=1,
        money=28,
        round=RoundInfo(hands_left=4, discards_left=3, reroll_cost=5),
        blinds={
            "small": BlindInfo(type="SMALL", status="DEFEATED", name="Small Blind", effect="", score=300),
            "big": BlindInfo(type="BIG", status="SELECT", name="Big Blind", effect="", score=450),
            "boss": BlindInfo(type="BOSS", status="UPCOMING", name="The Club", effect="", score=600),
        },
        jokers=AreaInfo(
            count=2,
            limit=5,
            cards=[
                _make_card("j_joker", set_name="Joker", mult=4.0, sell_cost=2, card_id=101),
                _make_card("j_banner", set_name="Joker", chips=30, sell_cost=3, card_id=102),
            ],
        ),
        consumables=AreaInfo(
            count=1,
            limit=2,
            cards=[_make_card("c_mercury", set_name="Tarot", sell_cost=1, card_id=201)],
        ),
        shop=AreaInfo(
            count=2,
            limit=4,
            cards=[
                _make_card("j_cavendish", set_name="Joker", x_mult=3.0, buy_cost=5, sell_cost=3, card_id=301),
                _make_card("c_empress", set_name="Tarot", buy_cost=3, sell_cost=1, card_id=302),
            ],
        ),
        vouchers=AreaInfo(
            count=1,
            limit=1,
            cards=[_make_card("v_grabber", set_name="Voucher", buy_cost=10, card_id=401)],
        ),
        packs=AreaInfo(
            count=2,
            limit=2,
            cards=[
                _make_card("p_buffoon_normal_1", set_name="Booster", buy_cost=4, card_id=501),
                _make_card("p_arcana_normal_1", set_name="Booster", buy_cost=4, card_id=502),
            ],
        ),
        deck_cards=AreaInfo(count=52, cards=_make_standard_deck()),
        hands={"Pair": HandLevel(order=2, level=2, chips=25, mult=3, played=3)},
    )


def _make_synthetic_pack_state() -> ParsedGameState:
    return ParsedGameState(
        state=GameScreen.BOOSTER_OPENED,
        round_num=3,
        ante_num=1,
        money=20,
        round=RoundInfo(hands_left=4, discards_left=3),
        pack=AreaInfo(
            count=3,
            limit=3,
            cards=[
                _make_card("c_magician", set_name="Tarot", card_id=601),
                _make_card("c_star", set_name="Tarot", card_id=602),
                _make_card("c_hermit", set_name="Tarot", card_id=603),
            ],
        ),
        consumables=AreaInfo(count=0, limit=2, cards=[]),
        jokers=AreaInfo(count=1, limit=5, cards=[_make_card("j_joker", set_name="Joker", card_id=101)]),
        deck_cards=AreaInfo(count=52, cards=_make_standard_deck()),
    )


# ── Test 1: Conformité Gymnasium sur Toutes les Fixtures ──────────────────────

def test_observation_space_dimensions_and_types():
    """Verify exact shapes and dtypes in get_observation_space()."""
    space = get_observation_space()
    assert space["context_features"].shape == (D_CTX,)
    assert space["deck_features"].shape == (D_DECK,)
    assert space["hand_levels"].shape == (D_HAND,)
    assert space["held_jokers"].shape == (MAX_JOKERS, D_ENTITY)
    assert space["shop_and_pack_offers"].shape == (MAX_OFFERS, D_ENTITY)

    for k, v in space.spaces.items():
        assert v.dtype == np.float32, f"Space {k} has dtype {v.dtype}, expected float32"


def test_gymnasium_conformity_on_json_fixtures():
    """Extract features from all fixture JSON files and verify observation_space.contains()."""
    space = get_observation_space()

    fixture_files = list(FIXTURES_DIR.glob("*.json"))
    assert len(fixture_files) >= 4, f"Found only {len(fixture_files)} fixtures in {FIXTURES_DIR}"

    for fpath in fixture_files:
        with open(fpath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        state = parse_gamestate(raw)
        obs = extract_features(state)

        for key, arr in obs.items():
            assert not np.isnan(arr).any(), f"NaN found in {key} for fixture {fpath.name}"
            assert not np.isinf(arr).any(), f"Inf found in {key} for fixture {fpath.name}"
            assert arr.dtype == np.float32, f"Invalid dtype {arr.dtype} in {key} for fixture {fpath.name}"

        assert space.contains(obs), f"Gymnasium space does not contain obs for fixture {fpath.name}"


def test_gymnasium_conformity_on_shop_and_pack_screens():
    """Verify Gymnasium space containment on synthetic SHOP and PACK screens."""
    space = get_observation_space()

    shop_state = _make_synthetic_shop_state()
    shop_obs = extract_features(shop_state)
    assert space.contains(shop_obs)
    # Verify offers are filled for shop
    assert shop_obs["shop_and_pack_offers"][0][1] == 1.0  # Shop card 1 present
    assert shop_obs["shop_and_pack_offers"][4][1] == 1.0  # Voucher present
    assert shop_obs["shop_and_pack_offers"][5][1] == 1.0  # Booster present

    pack_state = _make_synthetic_pack_state()
    pack_obs = extract_features(pack_state)
    assert space.contains(pack_obs)
    # Verify pack choices are filled
    assert pack_obs["shop_and_pack_offers"][0][1] == 1.0
    assert pack_obs["shop_and_pack_offers"][1][1] == 1.0
    assert pack_obs["shop_and_pack_offers"][2][1] == 1.0


# ── Test 2: Gestion de l'Overflow (> 10 Jokers) ────────────────────────────────

def test_joker_overflow_management():
    """Verify that > 10 Jokers preserves shape (10, D_entity) and encodes real count in context."""
    space = get_observation_space()
    base_state = _make_synthetic_shop_state()

    # Create 12 Jokers:
    # 2 Eternal Jokers
    # 3 Negative Jokers
    # 7 standard Jokers with varying sell costs
    twelve_jokers = [
        _make_card("j_ceremonial", set_name="Joker", eternal=True, sell_cost=4, card_id=1),
        _make_card("j_madness", set_name="Joker", eternal=True, sell_cost=5, card_id=2),
        _make_card("j_blueprint", set_name="Joker", edition="NEGATIVE", sell_cost=10, card_id=3),
        _make_card("j_brainstorm", set_name="Joker", edition="NEGATIVE", sell_cost=10, card_id=4),
        _make_card("j_mime", set_name="Joker", edition="NEGATIVE", sell_cost=8, card_id=5),
        _make_card("j_cavendish", set_name="Joker", x_mult=3.0, sell_cost=3, card_id=6),
        _make_card("j_banner", set_name="Joker", chips=30, sell_cost=3, card_id=7),
        _make_card("j_abstract", set_name="Joker", mult=3.0, sell_cost=2, card_id=8),
        _make_card("j_popcorn", set_name="Joker", mult=20.0, sell_cost=2, card_id=9),
        _make_card("j_gros_michel", set_name="Joker", mult=15.0, sell_cost=2, card_id=10),
        _make_card("j_ice_cream", set_name="Joker", chips=100, sell_cost=2, card_id=11),
        _make_card("j_half", set_name="Joker", mult=20.0, sell_cost=1, card_id=12),
    ]

    overflow_state = copy.deepcopy(base_state)
    overflow_state.jokers = AreaInfo(count=12, limit=5, cards=twelve_jokers)

    obs = extract_features(overflow_state)

    # 1. Shape must strictly be (MAX_JOKERS, D_ENTITY) = (10, 30)
    assert obs["held_jokers"].shape == (MAX_JOKERS, D_ENTITY)
    assert space.contains(obs)

    # 2. Context features index 21 must reflect the real 12 jokers: 12 / 10.0 = 1.2
    assert pytest.approx(obs["context_features"][21], rel=1e-4) == 1.2


# ── Test 3: Sensibilité Monotone de estimated_power_ratio ──────────────────────

def test_estimated_power_ratio_monotone_sensitivity():
    """Verify that estimated_power_ratio increases with xMult / hand level and decreases with target."""
    base_state = _make_synthetic_shop_state()
    base_state.jokers.cards = [_make_card("j_joker", set_name="Joker", mult=4.0)]
    base_state.hands = {"Pair": HandLevel(order=2, level=1, chips=10, mult=2, played=2)}
    base_state.ante_num = 1
    base_state.blinds["big"].score = 300

    r_base = compute_estimated_power_ratio(base_state)
    assert -2.0 <= r_base <= 2.0

    # 1. Adding a strong xMult joker (Cavendish x3) MUST strictly increase ratio
    cavendish_state = copy.deepcopy(base_state)
    cavendish_state.jokers.cards.append(
        _make_card("j_cavendish", set_name="Joker", x_mult=3.0)
    )
    r_cavendish = compute_estimated_power_ratio(cavendish_state)
    assert r_cavendish > r_base, f"Expected {r_cavendish} > {r_base} after adding Cavendish"

    # 2. Leveling up dominant hand from level 1 to level 5 MUST strictly increase ratio
    leveled_state = copy.deepcopy(base_state)
    leveled_state.hands["Pair"] = HandLevel(order=2, level=5, chips=70, mult=6, played=5)
    r_leveled = compute_estimated_power_ratio(leveled_state)
    assert r_leveled > r_base, f"Expected {r_leveled} > {r_base} after leveling Pair"

    # 3. Scaling target chips from Ante 1 (300) to Ante 6 (20,000) MUST strictly decrease ratio
    ante6_state = copy.deepcopy(base_state)
    ante6_state.ante_num = 6
    ante6_state.blinds["big"].score = 20000
    r_ante6 = compute_estimated_power_ratio(ante6_state)
    assert r_ante6 < r_base, f"Expected {r_ante6} < {r_base} when target increases to 20,000"


# ── Test 4: Sensibilité de deck_features ───────────────────────────────────────

def test_deck_features_sensitivity():
    """Verify that converting 3 cards to Diamonds or Steel shifts deck_features proportions by 3/52."""
    state = _make_synthetic_shop_state()
    deck = _make_standard_deck()
    state.deck_cards = AreaInfo(count=len(deck), cards=deck)

    obs1 = extract_features(state)
    deck_f1 = obs1["deck_features"]

    # Index 14 is Diamonds ("D" in ["H", "D", "C", "S"] -> 13 + 1 = 14)
    # In standard deck of 52 cards, exactly 13 are Diamonds: 13/52 = 0.25
    assert pytest.approx(deck_f1[14], abs=1e-4) == 13.0 / 52.0

    # Convert 3 Spades cards into Diamonds
    modified_suit_state = copy.deepcopy(state)
    spades = [c for c in modified_suit_state.deck_cards.cards if c.value.suit == "S"]
    for c in spades[:3]:
        c.value.suit = "D"

    obs2 = extract_features(modified_suit_state)
    deck_f2 = obs2["deck_features"]
    expected_diamonds = (13.0 + 3.0) / 52.0
    assert pytest.approx(deck_f2[14], abs=1e-4) == expected_diamonds
    assert pytest.approx(deck_f2[14] - deck_f1[14], abs=1e-4) == 3.0 / 52.0

    # Index 21 is STEEL ("STEEL" in enhancements -> 17 + 4 = 21)
    # Initially 0 steel cards: 0.0
    assert pytest.approx(deck_f1[21], abs=1e-4) == 0.0

    # Convert 3 cards into Steel cards
    modified_steel_state = copy.deepcopy(state)
    for c in modified_steel_state.deck_cards.cards[:3]:
        c.modifier.enhancement = "STEEL"

    obs3 = extract_features(modified_steel_state)
    deck_f3 = obs3["deck_features"]
    expected_steel = 3.0 / 52.0
    assert pytest.approx(deck_f3[21], abs=1e-4) == expected_steel
    assert pytest.approx(deck_f3[21] - deck_f1[21], abs=1e-4) == 3.0 / 52.0
