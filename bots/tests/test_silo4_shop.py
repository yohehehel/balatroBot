"""Unit tests for Silo 4: Economy Engine, Shop Management, and Joker Positioning.

Tests:
1. Interest Protection vs Survival Mode:
   - In Ante 1 with 0 Jokers and $7, bot buys a $6 +Mult joker (survival mode drops money to $1).
   - In Ante 4 with 3 good Jokers and $22, bot refuses a mediocre $5 joker that drops bank below $20 interest floor.
2. Replacing weak Joker when slots are full (5/5):
   - In Ante 5 with 5 Jokers (including a weak +4 Mult / basic econ), bot sells the weakest to buy an S-tier xMult (Cavendish).
3. Left-to-Right Joker Sorting:
   - [Cavendish (xMult), Banner (+Chips), Abstract Joker (+Mult)] -> [Banner, Abstract Joker, Cavendish].
4. Blueprint Placement:
   - [Abstract Joker (+Mult), Blueprint, Cavendish (xMult)] -> [Abstract Joker, Blueprint, Cavendish].
5. Reroll Discipline:
   - With $18 and reroll cost $5, bot refuses reroll (protects interest).
   - With $35 and reroll cost $5, bot accepts reroll on an empty/uninteresting shop.
6. Negative Joker exception:
   - Buys a Negative Joker even when slots are 5/5 because it takes 0 slots.
7. Voucher evaluation:
   - Buys Grabber (+1 hand) when spendable budget allows.
"""

from __future__ import annotations

from bots.shop_manager import (
    ShopActionType,
    compute_optimal_joker_order,
    decide_shop_action,
    get_protected_interest_floor,
    get_spendable_budget,
    is_survival_mode,
    reorder_jokers,
    should_reroll,
)
from bots.state_parser import (
    AreaInfo,
    Card,
    CardCost,
    CardModifier,
    ParsedGameState,
    RoundInfo,
)

# ── Helpers ───────────────────────────────────────────────────────────────

def _make_joker(
    key: str,
    label: str = "",
    buy_cost: int = 6,
    sell_cost: int = 3,
    edition: str | None = None,
    eternal: bool = False,
    joker_id: int = 1,
) -> Card:
    return Card(
        id=joker_id,
        key=key,
        set="Joker",
        label=label or key,
        modifier=CardModifier(edition=edition, eternal=eternal),
        cost=CardCost(buy=buy_cost, sell=sell_cost),
    )


def _make_voucher(
    key: str,
    label: str = "",
    buy_cost: int = 10,
    voucher_id: int = 200,
) -> Card:
    return Card(
        id=voucher_id,
        key=key,
        set="Voucher",
        label=label or key,
        cost=CardCost(buy=buy_cost, sell=0),
    )


def _make_shop_state(
    money: int = 20,
    ante: int = 1,
    jokers: list[Card] | None = None,
    shop_cards: list[Card] | None = None,
    vouchers: list[Card] | None = None,
    reroll_cost: int = 5,
    max_jokers: int = 5,
    used_vouchers: dict | list | None = None,
) -> ParsedGameState:
    jokers_list = jokers or []
    return ParsedGameState(
        state="SHOP",
        round_num=1,
        ante_num=ante,
        money=money,
        round=RoundInfo(reroll_cost=reroll_cost),
        jokers=AreaInfo(count=len(jokers_list), limit=max_jokers, cards=jokers_list),
        shop=AreaInfo(count=len(shop_cards or []), cards=shop_cards or []),
        vouchers=AreaInfo(count=len(vouchers or []), cards=vouchers or []),
        raw={"used_vouchers": used_vouchers or {}},
    )


# ── Test 1: Interest Protection vs Survival Mode ──────────────────────────

def test_survival_mode_buys_scoring_joker_in_ante_1():
    """In Ante 1 with $7 and 0 jokers, bot spends $6 on +Mult joker (drops to $1)."""
    joker_in_shop = _make_joker("j_abstract", label="Abstract Joker", buy_cost=6)
    state = _make_shop_state(
        money=7,
        ante=1,
        jokers=[],
        shop_cards=[joker_in_shop],
    )

    assert is_survival_mode(state) is True
    assert get_spendable_budget(state) == 7

    action = decide_shop_action(state)
    assert action.action_type == ShopActionType.BUY_CARD
    assert action.rpc_params == {"card": 0}
    assert action.cost == 6


def test_cruise_mode_protects_interest_floor_in_ante_4():
    """In Ante 4 with 3 good jokers and $22, bot refuses a mediocre $5 joker to protect $20 floor."""
    jokers = [
        _make_joker("j_banner", buy_cost=6, sell_cost=3, joker_id=1),
        _make_joker("j_abstract", buy_cost=6, sell_cost=3, joker_id=2),
        _make_joker("j_fibonacci", buy_cost=8, sell_cost=4, joker_id=3),
    ]
    # Mediocre joker in shop
    mediocre_joker = _make_joker("j_popcorn", label="Popcorn", buy_cost=5, joker_id=10)
    state = _make_shop_state(
        money=22,
        ante=4,
        jokers=jokers,
        shop_cards=[mediocre_joker],
    )

    assert is_survival_mode(state) is False
    assert get_protected_interest_floor(state) == 20
    assert get_spendable_budget(state) == 2  # Only $2 is spendable above $20 floor

    action = decide_shop_action(state)
    # Since cost is $5 > $2 spendable and popcorn is not an S-tier emergency, bot does NOT buy
    assert action.action_type == ShopActionType.NEXT_ROUND


# ── Test 2: Replacing Weak Joker (Slots Full 5/5) ─────────────────────────

def test_replace_weak_joker_with_s_tier_xmult_in_ante_5():
    """In Ante 5 with 5 Jokers, bot sells weakest joker to buy Cavendish (x3 Mult)."""
    jokers = [
        _make_joker("j_abstract", label="Abstract", buy_cost=6, sell_cost=3, joker_id=1),
        _make_joker("j_fibonacci", label="Fibonacci", buy_cost=8, sell_cost=4, joker_id=2),
        _make_joker("j_banner", label="Banner", buy_cost=6, sell_cost=3, joker_id=3),
        _make_joker("j_blue_joker", label="Blue Joker", buy_cost=6, sell_cost=3, joker_id=4),
        _make_joker("j_joker", label="Joker (+4 Mult)", buy_cost=4, sell_cost=2, joker_id=5),  # Weakest
    ]
    cavendish = _make_joker("j_cavendish", label="Cavendish", buy_cost=8, joker_id=20)
    state = _make_shop_state(
        money=7,  # With $7 + $2 sell value = $9 >= $8 buy cost
        ante=5,
        jokers=jokers,
        shop_cards=[cavendish],
        max_jokers=5,
    )

    action = decide_shop_action(state)

    # Bot should identify that Cavendish is an upgrade over j_joker and sell j_joker
    assert action.action_type == ShopActionType.SELL_JOKER
    assert action.rpc_params == {"joker": 4}  # index of j_joker
    assert "Selling" in action.reason


# ── Test 3: Left-to-Right Joker Sorting ───────────────────────────────────

def test_left_to_right_sorting_chips_mult_xmult():
    """Disordered [Cavendish (xMult), Banner (+Chips), Abstract Joker (+Mult)] -> [Banner, Abstract, Cavendish]."""
    j1 = _make_joker("j_cavendish", label="Cavendish")  # X_MULT
    j2 = _make_joker("j_banner", label="Banner")        # CHIPS
    j3 = _make_joker("j_abstract", label="Abstract")    # ADD_MULT
    jokers = [j1, j2, j3]

    order = compute_optimal_joker_order(jokers, ante=3)

    # Banner (idx 1), Abstract (idx 2), Cavendish (idx 0)
    assert order == [1, 2, 0]

    reordered = reorder_jokers(jokers, order)
    labels = [j.label for j in reordered]
    assert labels == ["Banner", "Abstract", "Cavendish"]


# ── Test 4: Blueprint Placement ───────────────────────────────────────────

def test_blueprint_placed_left_of_cavendish():
    """In mid/late game, Blueprint is placed immediately to the left of the best xMult (Cavendish)."""
    j_abs = _make_joker("j_abstract", label="Abstract")      # ADD_MULT
    j_blue = _make_joker("j_blueprint", label="Blueprint")   # RETRIGGER_COPY
    j_cav = _make_joker("j_cavendish", label="Cavendish")    # X_MULT
    jokers = [j_abs, j_blue, j_cav]

    order = compute_optimal_joker_order(jokers, ante=4)

    # Optimal order: Abstract (0), Blueprint (1), Cavendish (2)
    reordered = reorder_jokers(jokers, order)
    labels = [j.label for j in reordered]
    assert labels == ["Abstract", "Blueprint", "Cavendish"]


def test_blueprint_moved_from_left_to_before_cavendish():
    """Blueprint at index 0 is moved immediately before Cavendish."""
    j_blue = _make_joker("j_blueprint", label="Blueprint")   # RETRIGGER_COPY (0)
    j_abs = _make_joker("j_abstract", label="Abstract")      # ADD_MULT (1)
    j_cav = _make_joker("j_cavendish", label="Cavendish")    # X_MULT (2)
    jokers = [j_blue, j_abs, j_cav]

    order = compute_optimal_joker_order(jokers, ante=4)

    # Abstract (1), Blueprint (0), Cavendish (2)
    assert order == [1, 0, 2]
    reordered = reorder_jokers(jokers, order)
    labels = [j.label for j in reordered]
    assert labels == ["Abstract", "Blueprint", "Cavendish"]


# ── Test 5: Reroll Discipline ─────────────────────────────────────────────

def test_reroll_discipline_refuses_when_under_interest_floor():
    """With $18 and reroll cost $5, bot refuses to reroll (would drop below interest floor)."""
    state = _make_shop_state(
        money=18,
        ante=3,
        reroll_cost=5,
        shop_cards=[],
    )

    assert should_reroll(state) is False
    action = decide_shop_action(state)
    assert action.action_type == ShopActionType.NEXT_ROUND


def test_reroll_discipline_accepts_when_affluent():
    """With $35 and reroll cost $5, bot accepts to reroll an uninteresting shop."""
    state = _make_shop_state(
        money=35,
        ante=3,
        reroll_cost=5,
        shop_cards=[],
    )

    assert should_reroll(state) is True
    action = decide_shop_action(state)
    assert action.action_type == ShopActionType.REROLL
    assert action.cost == 5


# ── Test 6: Negative Joker Exception ──────────────────────────────────────

def test_negative_joker_requires_free_slot():
    """A Negative Joker cannot be bought if slots are 5/5. Bot must sell weakest joker first."""
    jokers = [
        _make_joker("j_abstract", joker_id=1),
        _make_joker("j_fibonacci", joker_id=2),
        _make_joker("j_banner", joker_id=3),
        _make_joker("j_blue_joker", joker_id=4),
        _make_joker("j_joker", joker_id=5),
    ]
    negative_joker = _make_joker("j_popcorn", label="Negative Popcorn", buy_cost=7, edition="NEGATIVE", joker_id=10)
    state = _make_shop_state(
        money=25,
        ante=3,
        jokers=jokers,
        shop_cards=[negative_joker],
        max_jokers=5,
    )

    # 1. When slots are 5/5, bot sells weakest joker (j_joker) to free a slot
    action = decide_shop_action(state)
    assert action.action_type == ShopActionType.SELL_JOKER
    assert action.rpc_params == {"joker": 4}

    # 2. When slots are 4/5, bot buys the negative joker
    state_4_slots = _make_shop_state(
        money=26,
        ante=3,
        jokers=jokers[:4],
        shop_cards=[negative_joker],
        max_jokers=5,
    )
    buy_action = decide_shop_action(state_4_slots)
    assert buy_action.action_type == ShopActionType.BUY_CARD
    assert buy_action.rpc_params == {"card": 0}

    # 3. If card 0 is excluded (e.g. blacklisted after RPC error), bot does not buy it
    excluded_action = decide_shop_action(state_4_slots, exclude_card_indices={0})
    assert excluded_action.action_type != ShopActionType.BUY_CARD


# ── Test 7: Voucher Purchase ──────────────────────────────────────────────

def test_buy_high_impact_voucher():
    """Bot purchases Grabber voucher (+1 hand) when spendable budget allows."""
    voucher = _make_voucher("v_grabber", label="Grabber", buy_cost=10)
    state = _make_shop_state(
        money=35,  # $35 has $10 spendable above $25 interest cap
        ante=2,
        shop_cards=[],
        vouchers=[voucher],
    )

    action = decide_shop_action(state)
    assert action.action_type == ShopActionType.BUY_VOUCHER
    assert action.rpc_params == {"voucher": 0}
    assert action.cost == 10
