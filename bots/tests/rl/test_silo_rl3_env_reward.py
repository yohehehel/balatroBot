"""Unit tests for Silo RL-3: Gymnasium Environment BalatroMacroEnv & Reward Shaping.

Verifies:
1. Gymnasium environment conformity via check_env(BalatroMacroEnv(client=mock)).
2. Rich Death penalty calculation on GAME_OVER ($75 vs $2 in bank).
3. Round victory rewards (base, ante bonus, margin bonus, boss bonus).
4. Shop discipline penalty (-0.05 per reroll).
5. Step macro transition with micro-solver combat loop delegation.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from bots.rl.actions import ActionID, NUM_ACTIONS
from bots.rl.env import BalatroMacroEnv
from bots.rl.reward import calculate_step_reward
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


# ── Mock Client Simulating Balatro State Transitions ──────────────────────────

class MockBalatroStateMachine:
    """Mock Balatro RPC client maintaining realistic state transitions for tests."""

    def __init__(self, initial_state: str = "BLIND_SELECT"):
        self.current_screen = initial_state
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

        # Load fixtures
        self.raw_blind_select = json.loads(
            (FIXTURES_DIR / "blind_select_small.json").read_text(encoding="utf-8")
        )
        self.raw_combat = json.loads(
            (FIXTURES_DIR / "selecting_hand.json").read_text(encoding="utf-8")
        )
        self.raw_menu = json.loads(
            (FIXTURES_DIR / "menu.json").read_text(encoding="utf-8")
        )

        # Build raw shop & round_eval
        self.raw_round_eval = copy.deepcopy(self.raw_combat)
        self.raw_round_eval["state"] = "ROUND_EVAL"
        self.raw_round_eval["round"]["chips"] = 400

        self.raw_shop = copy.deepcopy(self.raw_combat)
        self.raw_shop["state"] = "SHOP"
        self.raw_shop["round"]["chips"] = 400

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((method, params))

        if method == "menu":
            self.current_screen = "MENU"
            return self.raw_menu
        elif method == "start":
            self.current_screen = "BLIND_SELECT"
            return self.raw_blind_select
        elif method == "select":
            self.current_screen = "SELECTING_HAND"
            return self.raw_combat
        elif method == "play":
            # Transition to ROUND_EVAL after play
            self.current_screen = "ROUND_EVAL"
            return self.raw_round_eval
        elif method == "discard":
            self.raw_combat["round"]["discards_left"] = 0
            return self.raw_combat
        elif method == "cash_out":
            self.current_screen = "SHOP"
            return self.raw_shop
        elif method == "next_round":
            self.current_screen = "BLIND_SELECT"
            return self.raw_blind_select
        elif method == "rearrange":
            return self._current_raw()
        elif method == "reroll":
            return self.raw_shop
        elif method == "buy":
            return self.raw_shop
        elif method == "sell":
            return self.raw_shop
        elif method == "pack":
            return self.raw_shop
        elif method == "use":
            return self._current_raw()
        elif method == "gamestate":
            return self._current_raw()

        return self._current_raw()

    def _current_raw(self) -> dict[str, Any]:
        if self.current_screen == "MENU":
            return self.raw_menu
        elif self.current_screen == "BLIND_SELECT":
            return self.raw_blind_select
        elif self.current_screen == "SELECTING_HAND":
            return self.raw_combat
        elif self.current_screen == "ROUND_EVAL":
            return self.raw_round_eval
        elif self.current_screen == "SHOP":
            return self.raw_shop
        return self.raw_blind_select


# ── Test 1: Conformité Gymnasium via check_env ────────────────────────────────

def test_gymnasium_check_env_conformity():
    """Verify that BalatroMacroEnv strictly satisfies Gymnasium specification via check_env."""
    mock_client = MockBalatroStateMachine(initial_state="BLIND_SELECT")
    env = BalatroMacroEnv(client=mock_client, max_steps=20)

    # gymnasium.utils.env_checker.check_env verifies spaces, reset, step, dtypes, and shapes
    check_env(env)


def test_environment_reset_and_action_masks():
    """Verify reset returns valid observation and action_masks returns boolean shape (26,)."""
    mock_client = MockBalatroStateMachine(initial_state="BLIND_SELECT")
    env = BalatroMacroEnv(client=mock_client)

    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert info["state"] == "BLIND_SELECT"

    mask = env.action_masks()
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (NUM_ACTIONS,)
    assert mask.dtype == bool
    assert mask[ActionID.PLAY_BLIND]


# ── Test 2: Calcul de la Pénalité de Mort Riche ───────────────────────────────

def test_rich_death_penalty_calculation():
    """Verify GAME_OVER with $75 is severely more penalized than GAME_OVER with $2."""
    prev_state = ParsedGameState(
        state=GameScreen.SELECTING_HAND,
        round_num=3,
        ante_num=2,
        money=2,
        round=RoundInfo(chips=100),
    )

    # Poor Death ($2 bank)
    poor_death_state = copy.deepcopy(prev_state)
    poor_death_state.state = GameScreen.GAME_OVER
    poor_death_state.money = 2

    r_poor, breakdown_poor = calculate_step_reward(
        prev_state=prev_state,
        action=ActionID.PLAY_BLIND,
        next_state=poor_death_state,
        game_over=True,
    )

    # Rich Death ($75 bank)
    rich_death_state = copy.deepcopy(prev_state)
    rich_death_state.state = GameScreen.GAME_OVER
    rich_death_state.money = 75

    r_rich, breakdown_rich = calculate_step_reward(
        prev_state=prev_state,
        action=ActionID.PLAY_BLIND,
        next_state=rich_death_state,
        game_over=True,
    )

    # Poor death: defeat_penalty = -10.0, rich_death = 0.0
    assert breakdown_poor["defeat_penalty"] == -10.0
    assert breakdown_poor["rich_death_penalty"] == 0.0
    assert pytest.approx(r_poor, abs=1e-3) == -10.0

    # Rich death: defeat_penalty = -10.0, rich_death = -0.15 * (75 - 15) = -9.0
    assert breakdown_rich["defeat_penalty"] == -10.0
    expected_rich_penalty = -0.15 * (75.0 - 15.0)  # -9.0
    assert pytest.approx(breakdown_rich["rich_death_penalty"], abs=1e-3) == expected_rich_penalty
    assert pytest.approx(r_rich, abs=1e-3) == -19.0

    # Verification: Dying rich is 9 points worse
    assert r_rich < r_poor
    assert pytest.approx(r_poor - r_rich, abs=1e-3) == 9.0


# ── Test 3: Récompenses de Victoire et Discipline de Reroll ───────────────────

def test_round_victory_and_boss_bonus():
    """Verify round victory rewards scale with Ante and score margin, and Boss gives extra bonus."""
    prev_state = ParsedGameState(
        state=GameScreen.BLIND_SELECT,
        round_num=3,
        ante_num=3,
        blinds={"small": BlindInfo(type="SMALL", status="SELECT", name="Small Blind", effect="", score=1000)},
    )
    next_state = ParsedGameState(
        state=GameScreen.SHOP,
        round_num=3,
        ante_num=3,
        round=RoundInfo(chips=2000),
    )

    # Small Blind Won (score 2000 vs 1000 required -> margin log10(2) = 0.3010)
    reward, breakdown = calculate_step_reward(
        prev_state=prev_state,
        action=ActionID.PLAY_BLIND,
        next_state=next_state,
        round_won=True,
        score_obtained=2000,
        score_required=1000,
    )

    # Base: 0.5, Ante bonus: 0.2 * 3 = 0.6
    assert breakdown["round_base"] == 0.5
    assert pytest.approx(breakdown["round_ante_bonus"], abs=1e-4) == 0.6
    assert breakdown["round_margin_bonus"] > 0.0
    assert breakdown["boss_bonus"] == 0.0

    # Boss Blind Won
    prev_boss_state = copy.deepcopy(prev_state)
    prev_boss_state.blinds["boss"] = BlindInfo(type="BOSS", status="SELECT", name="The Club", effect="", score=1000)

    reward_boss, breakdown_boss = calculate_step_reward(
        prev_state=prev_boss_state,
        action=ActionID.PLAY_BLIND,
        next_state=next_state,
        round_won=True,
        boss_defeated=True,
        score_obtained=2000,
        score_required=1000,
    )
    # Boss bonus: 1.0 * ante = 3.0
    assert pytest.approx(breakdown_boss["boss_bonus"], abs=1e-4) == 3.0
    assert reward_boss > reward


def test_high_ante_progressive_bonus():
    """Verify progressive exponential bonus is awarded for round victories at Ante 6, 7, and 8."""
    # Ante 5: no high ante bonus
    prev_ante5 = ParsedGameState(state=GameScreen.SELECTING_HAND, ante_num=5)
    next_ante5 = ParsedGameState(state=GameScreen.SHOP, ante_num=5)
    r5, bd5 = calculate_step_reward(prev_state=prev_ante5, action=ActionID.PLAY_BLIND, next_state=next_ante5, round_won=True)
    assert bd5["high_ante_bonus"] == 0.0

    # Ante 6: +2.0
    prev_ante6 = ParsedGameState(state=GameScreen.SELECTING_HAND, ante_num=6)
    next_ante6 = ParsedGameState(state=GameScreen.SHOP, ante_num=6)
    r6, bd6 = calculate_step_reward(prev_state=prev_ante6, action=ActionID.PLAY_BLIND, next_state=next_ante6, round_won=True)
    assert bd6["high_ante_bonus"] == 2.0

    # Ante 7: +5.0
    prev_ante7 = ParsedGameState(state=GameScreen.SELECTING_HAND, ante_num=7)
    next_ante7 = ParsedGameState(state=GameScreen.SHOP, ante_num=7)
    r7, bd7 = calculate_step_reward(prev_state=prev_ante7, action=ActionID.PLAY_BLIND, next_state=next_ante7, round_won=True)
    assert bd7["high_ante_bonus"] == 5.0

    # Ante 8: +10.0, and Boss at Ante 8 gets 1.5x multiplier
    prev_ante8 = ParsedGameState(
        state=GameScreen.SELECTING_HAND,
        ante_num=8,
        blinds={"boss": BlindInfo(type="BOSS", status="SELECT", name="Violet Vessel", effect="", score=100000)},
    )
    next_ante8 = ParsedGameState(state=GameScreen.SHOP, ante_num=8)
    r8, bd8 = calculate_step_reward(prev_state=prev_ante8, action=ActionID.PLAY_BLIND, next_state=next_ante8, round_won=True, boss_defeated=True)
    assert bd8["high_ante_bonus"] == 10.0
    assert bd8["boss_bonus"] == 1.5 * 8.0  # 12.0
    assert r8 > r7 > r6 > r5


def test_reroll_shop_penalty():
    """Verify REROLL_SHOP applies a -0.05 penalty."""
    state = ParsedGameState(state=GameScreen.SHOP, round_num=2, ante_num=1)
    reward, breakdown = calculate_step_reward(
        prev_state=state,
        action=ActionID.REROLL_SHOP,
        next_state=state,
    )
    assert breakdown["reroll_penalty"] == -0.05
    assert reward == -0.05


def test_final_victory_bonus():
    """Verify Ante 8 Victory gives +50.0 bonus."""
    prev_state = ParsedGameState(state=GameScreen.SELECTING_HAND, ante_num=8)
    win_state = ParsedGameState(state=GameScreen.SHOP, ante_num=8, won=True)

    reward, breakdown = calculate_step_reward(
        prev_state=prev_state,
        action=ActionID.PLAY_BLIND,
        next_state=win_state,
        game_won=True,
    )
    assert breakdown["victory_bonus"] == 50.0
    assert reward >= 50.0


# ── Test 4: Validation de l'Étape step avec Micro-Solveur ──────────────────────

def test_step_macro_transition_executes_combat_and_cashes_out():
    """Verify step(PLAY_BLIND) triggers combat solving, cashing out, and delivers round reward."""
    mock_client = MockBalatroStateMachine(initial_state="BLIND_SELECT")
    env = BalatroMacroEnv(client=mock_client)
    env.reset()

    assert env.state.state == "BLIND_SELECT"

    # Step: PLAY_BLIND (0)
    obs, reward, terminated, truncated, info = env.step(ActionID.PLAY_BLIND)

    # State machine should have progressed through combat -> round_eval -> shop
    assert env.state.state == "SHOP"
    assert not terminated
    assert reward > 0.0, f"Expected positive reward for winning round, got {reward}"
    assert info["state"] == "SHOP"
    assert info["reward_breakdown"]["round_base"] == 0.5
    assert env.observation_space.contains(obs)


def test_emperor_reward_tiers():
    """Verify The Emperor's 3 reward tiers (+0.05 for 2 free slots, -0.20 for 1 free slot, -0.45 for 0 free slots)."""
    emperor = Card(key="c_emperor", set="Tarot")
    dummy = Card(key="c_mercury", set="Planet")
    next_s = ParsedGameState(state=GameScreen.SHOP)

    # Pack select: 0/2 consumables held -> effective_free = 2 -> +0.05
    prev_0 = ParsedGameState(
        state=GameScreen.BOOSTER_OPENED,
        pack=AreaInfo(cards=[emperor]),
        consumables=AreaInfo(cards=[], limit=2),
    )
    r0, bd0 = calculate_step_reward(prev_state=prev_0, action=ActionID.SELECT_PACK_CARD_0, next_state=next_s)
    assert bd0["emperor_reward"] == 0.05

    # Pack select: 1/2 consumables held -> effective_free = 1 -> -0.20
    prev_1 = ParsedGameState(
        state=GameScreen.BOOSTER_OPENED,
        pack=AreaInfo(cards=[emperor]),
        consumables=AreaInfo(cards=[dummy], limit=2),
    )
    r1, bd1 = calculate_step_reward(prev_state=prev_1, action=ActionID.SELECT_PACK_CARD_0, next_state=next_s)
    assert bd1["emperor_reward"] == -0.20

    # Pack select: 2/2 consumables held -> effective_free = 0 -> -0.45
    prev_2 = ParsedGameState(
        state=GameScreen.BOOSTER_OPENED,
        pack=AreaInfo(cards=[emperor]),
        consumables=AreaInfo(cards=[dummy, dummy], limit=2),
    )
    r2, bd2 = calculate_step_reward(prev_state=prev_2, action=ActionID.SELECT_PACK_CARD_0, next_state=next_s)
    assert bd2["emperor_reward"] == -0.45


def test_hermit_reward_penalty_curve():
    """Verify The Hermit reward curve at $2, $10, and $25."""
    hermit = Card(key="c_hermit", set="Tarot")
    next_s = ParsedGameState(state=GameScreen.SHOP)

    # 1. Money = $2: -0.35 * ((20 - 2) / 20) ** 2 = -0.35 * 0.81 = -0.2835
    prev_2 = ParsedGameState(
        state=GameScreen.SHOP,
        money=2,
        consumables=AreaInfo(cards=[hermit], limit=2),
    )
    r2, bd2 = calculate_step_reward(prev_state=prev_2, action=ActionID.USE_OR_SELL_CONSUMABLE_0, next_state=next_s)
    assert pytest.approx(bd2["hermit_reward"], abs=1e-4) == -0.2835

    # 2. Money = $10: -0.35 * ((20 - 10) / 20) ** 2 = -0.35 * 0.25 = -0.0875
    prev_10 = ParsedGameState(
        state=GameScreen.SHOP,
        money=10,
        consumables=AreaInfo(cards=[hermit], limit=2),
    )
    r10, bd10 = calculate_step_reward(prev_state=prev_10, action=ActionID.USE_OR_SELL_CONSUMABLE_0, next_state=next_s)
    assert pytest.approx(bd10["hermit_reward"], abs=1e-4) == -0.0875

    # 3. Money = $25: bonus +0.15
    prev_25 = ParsedGameState(
        state=GameScreen.SHOP,
        money=25,
        consumables=AreaInfo(cards=[hermit], limit=2),
    )
    r25, bd25 = calculate_step_reward(prev_state=prev_25, action=ActionID.USE_OR_SELL_CONSUMABLE_0, next_state=next_s)
    assert bd25["hermit_reward"] == 0.15


def test_planet_archetype_reward_and_dead_planet_penalty():
    """Verify bonus on archetype planet (Mercury for Pair) vs penalty on marginal dead planet (Neptune)."""
    mercury = Card(key="c_mercury", set="Planet")
    neptune = Card(key="c_neptune", set="Planet")
    next_s = ParsedGameState(state=GameScreen.SHOP, ante_num=3)

    # Run in Ante 3: Pair played 30 times, Two Pair 10 times, Straight Flush 0 times
    hands = {
        "Pair": HandLevel(played=30),
        "Two Pair": HandLevel(played=10),
        "Straight Flush": HandLevel(played=0),
    }

    # 1. Buying Mercury (Pair) in Shop -> matches top-1 played hand -> +0.06
    prev_mercury = ParsedGameState(
        state=GameScreen.SHOP,
        ante_num=3,
        shop=AreaInfo(cards=[mercury]),
        hands=hands,
    )
    rm, bdm = calculate_step_reward(prev_state=prev_mercury, action=ActionID.BUY_SHOP_CARD_0, next_state=next_s)
    assert bdm["planet_archetype_reward"] == 0.06

    # 2. Buying Neptune (Straight Flush) in Shop -> marginal hand (0% < 5%) -> -0.18
    prev_neptune = ParsedGameState(
        state=GameScreen.SHOP,
        ante_num=3,
        shop=AreaInfo(cards=[neptune]),
        hands=hands,
    )
    rn, bdn = calculate_step_reward(prev_state=prev_neptune, action=ActionID.BUY_SHOP_CARD_0, next_state=next_s)
    assert bdn["planet_archetype_reward"] == -0.18


def test_credit_card_debt_penalty_on_leave_shop():
    """Verify leaving shop with negative money applies -0.10 per $5 tranche."""
    prev_s = ParsedGameState(state=GameScreen.SHOP, ante_num=2, money=-20)
    next_s = ParsedGameState(state=GameScreen.BLIND_SELECT, ante_num=2, money=-20)

    r, bd = calculate_step_reward(prev_state=prev_s, action=ActionID.LEAVE_SHOP, next_state=next_s)
    assert bd["debt_penalty"] == -0.40  # 4 tranches of $5 at -$20


def test_passive_hoarding_penalty_on_leave_shop():
    """Verify hoarding penalty applies on LEAVE_SHOP in Ante 3+ when rich (>45$) without Bull/Bootstraps."""
    # 1. Ante 3, $60, no jokers -> applies -0.15 penalty
    prev_hoard = ParsedGameState(
        state=GameScreen.SHOP,
        ante_num=3,
        money=60,
        jokers=AreaInfo(cards=[], limit=5),
    )
    next_s = ParsedGameState(state=GameScreen.BLIND_SELECT, ante_num=3, money=60)
    r1, bd1 = calculate_step_reward(prev_state=prev_hoard, action=ActionID.LEAVE_SHOP, next_state=next_s)
    assert bd1["hoarding_penalty"] == -0.15

    # 2. Ante 3, $60, but owns Bull (j_bull) -> exempt (0.0)
    bull = Card(key="j_bull", set="Joker")
    prev_bull = ParsedGameState(
        state=GameScreen.SHOP,
        ante_num=3,
        money=60,
        jokers=AreaInfo(cards=[bull], limit=5),
    )
    r2, bd2 = calculate_step_reward(prev_state=prev_bull, action=ActionID.LEAVE_SHOP, next_state=next_s)
    assert bd2["hoarding_penalty"] == 0.0

    # 3. Ante 2 (Early Ante), $60, no jokers -> exempt (0.0)
    prev_ante2 = ParsedGameState(
        state=GameScreen.SHOP,
        ante_num=2,
        money=60,
        jokers=AreaInfo(cards=[], limit=5),
    )
    r3, bd3 = calculate_step_reward(prev_state=prev_ante2, action=ActionID.LEAVE_SHOP, next_state=next_s)
    assert bd3["hoarding_penalty"] == 0.0


