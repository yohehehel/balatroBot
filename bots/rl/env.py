"""Gymnasium Environment BalatroMacroEnv for Balatro Macro-RL (Silo RL-3).

Encapsulates Balatro's high-level macro decision points (Shop, Blind Select, Booster Packs,
Inventory Consumables) and automatically delegates combat rounds to the micro-solver
(combat_solver.py) in a single macro environment step.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import logging
import time
from typing import Any

import gymnasium
import httpx
import numpy as np

from bots.client_wrapper import BalatroClient, RPCError
from bots.combat_solver import decide_combat_action
from bots.pack_consumable_manager import manage_held_consumables
from bots.rl.actions import (
    ActionID,
    compute_action_mask,
    execute_macro_action,
    get_action_space,
)
from bots.rl.features import extract_features, get_observation_space
from bots.rl.progress import log_message
from bots.rl.reward import calculate_step_reward
from bots.shop_manager import compute_optimal_joker_order
from bots.state_parser import (
    BlindType,
    GameScreen,
    ParsedGameState,
    fetch_gamestate,
    parse_gamestate,
    start_run,
)
from bots.telemetry import (
    PurchaseRecord,
    RoundRecord,
    RunRecord,
    record_purchase,
    record_round,
    record_run,
)

logger = logging.getLogger(__name__)

TRANSIENT_STATES = {
    "HAND_PLAYED",
    "DRAW_TO_HAND",
    "NEW_ROUND",
    "PLAY_TAROT",
}


class BalatroMacroEnv(gymnasium.Env):
    """Gymnasium Environment wrapping Balatro for Macro-RL.

    Observations:
        Dict space with 5 keys (context_features, deck_features, hand_levels,
        held_jokers, shop_and_pack_offers).
    Actions:
        Discrete(26) representing high-level strategic actions.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        client: BalatroClient | None = None,
        instance_id: int | None = None,
        deck: str = "RED",
        stake: str = "WHITE",
        seed: str | None = None,
        max_steps: int = 150,
        delay: float = 0.0,
        max_consecutive_actions_per_state: int = 25,
        endless: bool = False,
    ) -> None:
        super().__init__()
        self.client = client or BalatroClient()
        self.instance_id = instance_id
        self.deck = deck
        self.stake = stake
        self.seed = seed
        self.max_steps = max_steps
        self.delay = delay
        self.max_consecutive_actions_per_state = max_consecutive_actions_per_state
        self.endless = endless
        self.won_flag: bool = False

        self.action_space = get_action_space()
        self.observation_space = get_observation_space()

        self.state: ParsedGameState = ParsedGameState()
        self.prev_state: ParsedGameState | None = None
        self.step_count: int = 0
        self.consecutive_state_actions: int = 0
        self.last_seen_state: str = ""
        self.consecutive_transport_errors: int = 0
        self.is_disconnected: bool = False

        # Telemetry tracking for runs_history.csv, rounds_history.csv and purchases_history.csv
        self.record_telemetry: bool = True
        self.run_start_time: float = time.time()
        self.run_hands_breakdown: Counter[str] = Counter()
        self.run_hands_played: int = 0
        self.run_discards_used: int = 0
        self.round_hand_types: list[str] = []
        self.round_hand_scores: list[int] = []
        self.round_hands_played: int = 0
        self.round_discards_used: int = 0
        self.run_jokers_bought: list[str] = []
        self.run_consumables_bought: list[str] = []
        self.run_cards_bought: list[str] = []
        self.run_vouchers_bought: list[str] = []
        self.run_packs_opened: list[str] = []
        self.run_consumables_used: list[str] = []
        self.run_blinds_played: int = 0
        self.run_blinds_skipped: int = 0
        self.run_rerolls: int = 0
        self.run_max_money: int = 4
        self.run_max_ante: int = 1
        self.run_total_reward: float = 0.0

    def _log(self, msg: str) -> None:
        prefix = f"[Inst {self.instance_id}] " if self.instance_id is not None else ""
        try:
            log_message(f"{prefix}{msg}")
        except UnicodeEncodeError:
            clean_msg = msg.encode("ascii", errors="replace").decode("ascii")
            log_message(f"{prefix}{clean_msg}")

    def action_masks(self) -> np.ndarray:
        """Return the boolean action mask of shape (26,) for the current state."""
        return compute_action_mask(self.state)

    def _sleep_delay(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)

    def _poll_stable_state(self, max_polls: int = 8) -> ParsedGameState:
        """Poll the game state, waiting for any transient animations or pack openings to finish."""
        try:
            state = fetch_gamestate(self.client)
            self.consecutive_transport_errors = 0
        except (httpx.HTTPError, ConnectionError, OSError) as err:
            self.consecutive_transport_errors += 1
            if self.consecutive_transport_errors >= 2:
                self._log(f"  [ERROR] Instance transport failure: {err}")
                self.is_disconnected = True
            return self.state or ParsedGameState()
        except Exception:
            return self.state or ParsedGameState()

        polls = 0
        is_real = isinstance(self.client, BalatroClient)
        while polls < max_polls:
            is_transient = state.state in TRANSIENT_STATES
            is_opening_pack = (
                state.state in GameScreen.BOOSTER_STATES
                and (not state.pack or not state.pack.cards)
            )
            if not (is_transient or is_opening_pack):
                break

            if is_real:
                time.sleep(0.08)
            try:
                state = fetch_gamestate(self.client)
            except Exception:
                break
            polls += 1
        return state

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Start a fresh Balatro game run and return the initial observation."""
        super().reset(seed=seed)

        if self.is_disconnected:
            # Check if health endpoint responds before trying to restart
            try:
                self.client.call("health")
                self.is_disconnected = False
                self.consecutive_transport_errors = 0
            except Exception as e:
                base_url = getattr(self.client, "_config", None) and self.client._config.base_url or "unknown"
                msg = f"[Inst {self.instance_id}] Balatro instance on {base_url} is dead/unresponsive: {e}"
                logger.error(msg)
                raise RuntimeError(msg)

        seed_str = str(seed) if seed is not None else self.seed
        if self.record_telemetry and self.step_count > 0:
            try:
                self._record_run_telemetry(terminated=False, truncated=True)
            except Exception:
                pass

        self.step_count = 0
        self.consecutive_state_actions = 0
        self.last_seen_state = ""
        self.prev_state = None
        self.won_flag = False

        self.run_start_time = time.time()
        self.run_hands_breakdown = Counter()
        self.run_hands_played = 0
        self.run_discards_used = 0
        self.round_hand_types = []
        self.round_hand_scores = []
        self.round_hands_played = 0
        self.round_discards_used = 0
        self.run_jokers_bought = []
        self.run_consumables_bought = []
        self.run_cards_bought = []
        self.run_vouchers_bought = []
        self.run_packs_opened = []
        self.run_consumables_used = []
        self.run_blinds_played = 0
        self.run_blinds_skipped = 0
        self.run_rerolls = 0
        self.run_max_money = 4
        self.run_max_ante = 1
        self.run_total_reward = 0.0

        try:
            self.state = start_run(
                self.client,
                deck=self.deck,
                stake=self.stake,
                seed=seed_str,
            )
            self.consecutive_transport_errors = 0
        except (httpx.HTTPError, ConnectionError, OSError) as e:
            self.consecutive_transport_errors += 1
            self._log(f"  [ERROR] Failed to start_run (instance connection lost): {e}")
            self.is_disconnected = True
            raise RuntimeError(f"[Inst {self.instance_id}] Instance died during start_run: {e}")
        except Exception as e:
            logger.warning(f"Failed to start_run on client, fetching current gamestate: {e}")
            try:
                self.state = self._poll_stable_state()
            except Exception:
                self.state = ParsedGameState()

        self.state = self._poll_stable_state()
        self._log(f"-- Fresh run started (Deck: {self.deck}, Stake: {self.stake}, Seed: {seed_str or 'RANDOM'}) --")
        obs = extract_features(self.state)
        info = {
            "state": self.state.state,
            "ante": self.state.ante_num,
            "round": self.state.round_num,
            "money": self.state.money,
        }
        return obs, info

    def _run_micro_combat(self) -> tuple[bool, bool, int, int]:
        """Execute the full combat round using combat_solver.py and cash out to shop.

        Returns:
            round_won (bool): True if blind was successfully defeated.
            boss_defeated (bool): True if the defeated blind was a Boss Blind.
            score_obtained (int): Total chips scored during combat.
            score_required (int): Required chip target.
        """
        round_won = False
        boss_defeated = False
        score_obtained = 0
        score_required = 300

        if self.state.active_blind:
            score_required = self.state.active_blind.score
            boss_defeated = ((self.state.active_blind.type or "").upper() == BlindType.BOSS)

        combat_actions = 0
        max_combat_actions = 20

        while self.state.state == GameScreen.SELECTING_HAND and combat_actions < max_combat_actions:
            combat_actions += 1

            # 1. Check if held consumable can be used
            held_act = manage_held_consumables(self.state)
            if held_act and held_act.action_type == "use":
                try:
                    self.client.call(held_act.rpc_method, held_act.rpc_params)
                    self.state = self._poll_stable_state()
                    self._sleep_delay()
                    continue
                except Exception:
                    pass

            # 2. Check and reorder Jokers if needed before playing
            if len(self.state.jokers.cards) > 1:
                optimal_order = compute_optimal_joker_order(self.state.jokers.cards, self.state.ante_num)
                if optimal_order != list(range(len(self.state.jokers.cards))):
                    try:
                        self.client.call("rearrange", {"jokers": optimal_order})
                        self.state = self._poll_stable_state()
                        self._sleep_delay()
                    except Exception:
                        pass

            # 3. Micro Combat Solver (Play or Discard)
            decision = decide_combat_action(self.state)
            c_names = [f"{c.value.rank}{c.value.suit}" for c in decision.cards]
            target_chips = self.state.active_blind.score if self.state.active_blind else 300
            chips_remaining = max(0, target_chips - self.state.round.chips)
            if decision.action == "play":
                htype = decision.candidate.hand_type if decision.candidate else "Hand"
                self.run_hands_breakdown[htype] += 1
                self.round_hand_types.append(htype)
                self.round_hands_played += 1
                self.run_hands_played += 1
                self._log(f"  [Combat PLAY] {htype.upper()} ({', '.join(c_names)}) | Need {chips_remaining:,} pts | Hands left: {self.state.round.hands_left}")
            else:
                self.round_discards_used += 1
                self.run_discards_used += 1
                self._log(f"  [Combat DISCARD] {len(c_names)} cards ({', '.join(c_names)}) | Discards left: {self.state.round.discards_left}")

            try:
                self.client.call(decision.action, {"cards": decision.indices})
                self.state = self._poll_stable_state()
            except Exception as e:
                logger.error(f"Error during combat decision {decision.action}: {e}")
                if isinstance(e, (httpx.HTTPError, ConnectionError, OSError)):
                    self.consecutive_transport_errors += 1
                    self.is_disconnected = True
                break

            self._sleep_delay()

        # Check combat outcome
        score_obtained = max(self.state.round.chips, score_obtained)

        # Cash out if round was won
        if self.state.state in (GameScreen.ROUND_EVAL, "CASH_OUT"):
            round_won = True
            self._log(f"  [Combat WON] Score: {score_obtained:,} / {score_required:,} pts")
            if boss_defeated:
                self._log(f"  [BOSS DEFEATED] Boss Blind defeated at Ante {self.state.ante_num}!")
            try:
                self.client.call("cash_out")
                self.state = self._poll_stable_state()
            except Exception as e:
                logger.warning(f"Error cashing out: {e}")

        elif self.state.state == GameScreen.SHOP:
            round_won = True
            self._log(f"  [Combat WON] Score: {score_obtained:,} / {score_required:,} pts")

        round_summary = None
        if self.record_telemetry:
            round_summary = self._record_round_telemetry(
                round_won=round_won,
                score_obtained=score_obtained,
                score_required=score_required,
            )

        return round_won, boss_defeated, score_obtained, score_required, round_summary

    def _record_round_telemetry(
        self,
        round_won: bool,
        score_obtained: int,
        score_required: int,
        outcome_override: str | None = None,
    ) -> dict[str, Any]:
        """Record fine-grained round metrics to rounds_history.csv."""
        outcome = outcome_override if outcome_override else ("WON" if round_won else "LOST")
        jokers_str = "|".join(c.key for c in self.state.jokers.cards) if self.state.jokers else ""
        blind_name = self.state.active_blind.name if self.state.active_blind else "Blind"
        record = RoundRecord(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            seed=self.state.seed or str(self.seed or "RANDOM"),
            ante=self.state.ante_num,
            round_num=self.state.round_num,
            blind_name=blind_name,
            target_score=score_required,
            hands_played=self.round_hands_played,
            discards_used=self.round_discards_used,
            hand_types_played=";".join(self.round_hand_types),
            hand_scores=str(score_obtained),
            total_score=score_obtained,
            outcome=outcome,
            jokers_held=jokers_str,
            money=self.state.money,
            hands_left=getattr(self.state.round, "hands_left", 0) if hasattr(self.state, "round") else 0,
            discards_left=getattr(self.state.round, "discards_left", 0) if hasattr(self.state, "round") else 0,
        )
        try:
            record_round(record)
        except Exception as e:
            logger.warning(f"Failed to record round telemetry: {e}")

        summary = {
            "round_won": round_won,
            "outcome": outcome,
            "hands_played": self.round_hands_played,
            "discards_used": self.round_discards_used,
            "score_obtained": score_obtained,
            "target_score": score_required,
        }
        self.round_hand_types = []
        self.round_hand_scores = []
        self.round_hands_played = 0
        self.round_discards_used = 0
        return summary

    def _record_run_telemetry(self, terminated: bool, truncated: bool) -> dict[str, Any]:
        """Record comprehensive game-level run metrics to runs_history.csv."""
        is_game_won = bool(self.state.won or self.state.ante_num > 8 or self.won_flag)
        is_game_over = (self.state.state == GameScreen.GAME_OVER)
        duration = time.time() - self.run_start_time
        most_played = self.run_hands_breakdown.most_common(1)[0][0] if self.run_hands_breakdown else "None"
        jokers_str = "|".join(c.key for c in self.state.jokers.cards) if self.state.jokers else ""
        fatal_boss = (self.state.active_blind.name if (is_game_over and self.state.active_blind) else ("NONE" if is_game_won else "Small Blind"))
        breakdown_str = "|".join(f"{k}:{v}" for k, v in self.run_hands_breakdown.items())

        rec = RunRecord(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            seed=self.state.seed or str(self.seed or "RANDOM"),
            deck=self.deck,
            stake=self.stake,
            win=is_game_won,
            max_ante=self.run_max_ante,
            final_round=self.state.round_num,
            fatal_boss=fatal_boss,
            max_money=self.run_max_money,
            most_played_hand=most_played,
            final_jokers=jokers_str,
            duration_sec=round(duration, 2),
            exact_prediction_pct=0.0,
            total_hands_played=self.run_hands_played,
            total_discards_used=self.run_discards_used,
            hands_breakdown=breakdown_str,
            jokers_bought="|".join(self.run_jokers_bought),
            consumables_bought="|".join(self.run_consumables_bought),
            cards_bought="|".join(self.run_cards_bought),
            vouchers_bought="|".join(self.run_vouchers_bought),
            packs_opened="|".join(self.run_packs_opened),
            consumables_used="|".join(self.run_consumables_used),
            blinds_played=self.run_blinds_played,
            blinds_skipped=self.run_blinds_skipped,
            shop_rerolls=self.run_rerolls,
            total_reward=round(self.run_total_reward, 2),
        )
        try:
            record_run(rec)
        except Exception as e:
            logger.warning(f"Failed to record run telemetry: {e}")

        summary = {
            "win": is_game_won,
            "max_ante": self.run_max_ante,
            "final_round": self.state.round_num,
            "final_money": self.state.money,
            "max_money": self.run_max_money,
            "total_reward": self.run_total_reward,
            "hands_played": self.run_hands_played,
            "discards_used": self.run_discards_used,
            "blinds_played": self.run_blinds_played,
            "blinds_skipped": self.run_blinds_skipped,
            "rerolls": self.run_rerolls,
            "jokers_bought": len(self.run_jokers_bought),
            "consumables_bought": len(self.run_consumables_bought),
            "cards_bought": len(self.run_cards_bought),
            "vouchers_bought": len(self.run_vouchers_bought),
            "packs_opened": len(self.run_packs_opened),
            "consumables_used": len(self.run_consumables_used),
            "hands_breakdown": dict(self.run_hands_breakdown),
        }
        return summary

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute a Macro-RL action and return (obs, reward, terminated, truncated, info)."""
        self.step_count += 1

        # Synchronize with live game state in case background animations completed
        fresh_state = self._poll_stable_state()
        if fresh_state.state != "UNKNOWN":
            self.state = fresh_state

        mask = self.action_masks()

        action_int = int(action)
        invalid_penalty = 0.0

        # Safety: Check action validity against mask
        if not mask[action_int]:
            invalid_penalty = -2.0
            # Fallback to the first valid action to prevent crashing
            valid_indices = np.where(mask)[0]
            if len(valid_indices) > 0:
                action_int = int(valid_indices[0])
            else:
                if self.state.state == GameScreen.SHOP:
                    action_int = int(ActionID.LEAVE_SHOP)
                elif self.state.state in GameScreen.BOOSTER_STATES:
                    action_int = int(ActionID.SKIP_PACK)
                else:
                    action_int = int(ActionID.PLAY_BLIND)

        act_enum = ActionID(action_int)
        self._log(f"  [Macro Action] {act_enum.name} (Screen: {self.state.state}, Ante: {self.state.ante_num}, Money: ${self.state.money})")

        self.prev_state = self.state
        round_won = False
        boss_defeated = False
        score_obtained = 0
        score_required = 0
        round_summary: dict[str, Any] | None = None

        if action_int == ActionID.PLAY_BLIND:
            self.run_blinds_played += 1
        elif action_int == ActionID.REROLL_SHOP:
            self.run_rerolls += 1

        # 1. Execute Macro Action
        exec_info = execute_macro_action(action_int, self.state, self.client)
        if not exec_info.get("success", True):
            invalid_penalty -= 2.0
            if exec_info.get("is_transport_error"):
                self.consecutive_transport_errors += 1
                if self.consecutive_transport_errors >= 2:
                    self._log(f"  [ERROR] Instance transport failure: {exec_info.get('error')}")
                    self.is_disconnected = True
            else:
                # If an RPC logical error occurred (e.g. action rejected because screen transitioned),
                # sleep briefly to allow the game engine to settle before the next observation
                if isinstance(self.client, BalatroClient):
                    time.sleep(0.1)

        # Process granular transaction telemetry
        tx = exec_info.get("transaction")
        if tx and self.record_telemetry:
            try:
                record_purchase(PurchaseRecord(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    worker_id=self.instance_id or 0,
                    seed=self.state.seed or str(self.seed or "RANDOM"),
                    ante=self.state.ante_num,
                    round_num=self.state.round_num,
                    money=self.state.money,
                    action=tx["action"],
                    category=tx["category"],
                    item_key=tx["item_key"],
                    item_name=tx["item_name"],
                ))
            except Exception as e:
                logger.warning(f"Failed to record purchase telemetry: {e}")

            if tx["action"] == "BUY":
                if tx["category"] == "joker":
                    self.run_jokers_bought.append(tx["item_key"])
                elif tx["category"] == "voucher":
                    self.run_vouchers_bought.append(tx["item_key"])
                elif tx["category"] == "pack":
                    self.run_packs_opened.append(tx["item_key"])
                elif tx["category"] in ("tarot", "planet", "spectral"):
                    self.run_consumables_bought.append(tx["item_key"])
                else:
                    self.run_cards_bought.append(tx["item_key"])
            elif tx["action"] == "USE":
                self.run_consumables_used.append(tx["item_name"])

        self._sleep_delay()
        self.state = self._poll_stable_state()

        # If a blind was skipped, wait briefly for any tag booster pack to open and record skipped round
        if action_int == ActionID.SKIP_BLIND:
            self.run_blinds_skipped += 1
            if self.record_telemetry:
                round_summary = self._record_round_telemetry(
                    round_won=False,
                    score_obtained=0,
                    score_required=0,
                    outcome_override="SKIPPED",
                )
            if isinstance(self.client, BalatroClient):
                time.sleep(0.15)
            self.state = self._poll_stable_state()

        # 2. Combat Transition Loop
        # If the action triggered combat, execute micro-solver until next macro decision point
        if action_int == ActionID.PLAY_BLIND or self.state.state == GameScreen.SELECTING_HAND:
            r_won, b_def, s_obt, s_req, r_sum = self._run_micro_combat()
            round_won = r_won
            boss_defeated = b_def
            score_obtained = s_obt
            score_required = s_req
            round_summary = r_sum

        # 3. Watchdog: Anti-infinite loop protection
        if self.state.state == self.last_seen_state:
            self.consecutive_state_actions += 1
        else:
            self.last_seen_state = self.state.state
            self.consecutive_state_actions = 1

        if self.consecutive_state_actions > self.max_consecutive_actions_per_state:
            logger.warning(
                f"[Watchdog] Stuck in state {self.state.state} ({self.consecutive_state_actions} actions). Forcing escape."
            )
            try:
                if self.state.state == GameScreen.SHOP:
                    self.client.call("next_round")
                elif self.state.state in GameScreen.BOOSTER_STATES:
                    self.client.call("pack", {"skip": True})
                self.state = self._poll_stable_state()
            except Exception:
                pass
            self.consecutive_state_actions = 0

        # Update running game stats
        self.run_max_money = max(self.run_max_money, self.state.money)
        self.run_max_ante = max(self.run_max_ante, self.state.ante_num)

        # 4. Terminal State Detection
        is_game_over = (self.state.state == GameScreen.GAME_OVER)
        is_game_won = bool(self.state.won or self.state.ante_num > 8)
        if is_game_won:
            self.won_flag = True
            self._log(f"  [VICTORY] Ante 8 Boss defeated!")
        elif is_game_over:
            self._log(f"  [GAME OVER] Ante {self.state.ante_num}, Round {self.state.round_num}.")

        if self.is_disconnected:
            terminated = True
            truncated = True
        elif self.endless:
            terminated = is_game_over
            truncated = (self.step_count >= self.max_steps)
        else:
            terminated = is_game_over or is_game_won
            truncated = (self.step_count >= self.max_steps)

        # 5. Reward Calculation
        reward, breakdown = calculate_step_reward(
            prev_state=self.prev_state,
            action=action_int,
            next_state=self.state,
            round_won=round_won,
            boss_defeated=boss_defeated,
            game_won=is_game_won,
            game_over=is_game_over,
            score_obtained=score_obtained,
            score_required=score_required,
        )
        total_reward = float(reward + invalid_penalty)
        self.run_total_reward += total_reward

        episode_summary: dict[str, Any] | None = None
        if self.record_telemetry and (terminated or truncated):
            episode_summary = self._record_run_telemetry(terminated=terminated, truncated=truncated)

        # 6. Feature Extraction
        obs = extract_features(self.state)

        info = {
            "state": self.state.state,
            "ante": self.state.ante_num,
            "round": self.state.round_num,
            "money": self.state.money,
            "is_success": is_game_won or self.won_flag,
            "is_disconnected": self.is_disconnected,
            "reward_breakdown": breakdown,
            "invalid_penalty": invalid_penalty,
            "exec_info": exec_info,
        }
        if round_summary:
            info["round_summary"] = round_summary
        if episode_summary:
            info["episode_summary"] = episode_summary

        return obs, total_reward, terminated, truncated, info

