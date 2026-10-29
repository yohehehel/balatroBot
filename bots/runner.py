"""Full Multi-Run Autonomous Orchestrator and State Machine (Silo 6).

Responsibilities:
- State machine routing across all game phases:
  * MENU / GAME_OVER -> Start new run (Red Deck, White Stake, optional seed)
  * BLIND_SELECT -> Always select active blind (never skip)
  * SELECTING_HAND -> Combat loop (held consumables -> joker positioning -> play/discard)
  * ROUND_EVAL -> Cash out to shop
  * SHOP -> Shop decision loop (consumables, jokers, vouchers, packs, reroll, joker positioning)
  * SMODS_BOOSTER_OPENED / Pack states -> Pack card selection or skip
  * Transient states (HAND_PLAYED, DRAW_TO_HAND) -> Backoff polling
- Victory detection (Ante 8 Boss defeated / transition to Ante 9+ / state.won)
  * Default mode: stop immediately on win, record win=True, start next run
  * Endless mode: flag win=True, continue playing until GAME_OVER, record final max_ante
- Anti-infinite loop watchdogs (forced next_round or skip if stuck in an area)
- Multi-run batch execution (run_session) with telemetry recording and summary.
- Detailed round-by-round and run-by-run telemetry recording.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from bots.client_wrapper import BalatroClient, RPCError
from bots.combat_solver import decide_combat_action
from bots.pack_consumable_manager import decide_pack_action, manage_held_consumables
from bots.shop_manager import (
    ShopActionType,
    compute_optimal_joker_order,
    decide_shop_action,
)
from bots.state_parser import (
    GameScreen,
    fetch_gamestate,
    parse_gamestate,
    select_blind,
    start_run,
)
from bots.telemetry import (
    RoundRecord,
    RunRecord,
    print_summary,
    record_round,
    record_run,
)

logger = logging.getLogger(__name__)

# Transient animation states where the bot should wait
TRANSIENT_STATES = {
    "HAND_PLAYED",
    "DRAW_TO_HAND",
    "NEW_ROUND",
    "PLAY_TAROT",
}

# Maximum consecutive actions inside the same state before watchdog fires
MAX_CONSECUTIVE_ACTIONS_PER_STATE = 25


class BotRunner:
    """Autonomous multi-run orchestrator for Balatro."""

    def __init__(
        self,
        client: BalatroClient,
        endless: bool = False,
        delay: float = 0.8,
        seed: str | None = None,
        deck: str = "RED",
        stake: str = "WHITE",
        csv_path: Path | str | None = None,
        rounds_csv_path: Path | str | None = None,
        instance_id: int | None = None,
    ) -> None:
        self.client = client
        self.endless = endless
        self.delay = delay
        self.seed = seed
        self.deck = deck
        self.stake = stake
        self.csv_path = csv_path
        self.rounds_csv_path = rounds_csv_path
        self.instance_id = instance_id

    def _log(self, msg: str) -> None:
        prefix = f"[Inst {self.instance_id}] " if self.instance_id is not None else ""
        print(f"{prefix}{msg}")

    def _sleep_delay(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)

    def run_session(self, num_runs: int = 1) -> list[RunRecord]:
        """Execute a batch of N complete runs sequentially and print telemetry summary."""
        session_records: list[RunRecord] = []

        self._log("\n=======================================================")
        self._log(f"  STARTING AUTONOMOUS BALATRO BOT SESSION ({num_runs} RUNS)")
        self._log(f"  Mode: {'ENDLESS' if self.endless else 'STANDARD (Win at Ante 8)'} | Deck: {self.deck} | Stake: {self.stake}")
        if self.seed:
            self._log(f"  Fixed Seed: {self.seed}")
        self._log("=======================================================\n")

        for run_idx in range(1, num_runs + 1):
            self._log(f"\n── [Run {run_idx}/{num_runs}] Starting run... ────────────────────────────")
            record = self.run_single_game(run_index=run_idx)
            session_records.append(record)

            # Record to CSV
            record_run(record, self.csv_path)

            outcome = "VICTORY" if record.win else "DEFEAT"
            self._log(f"── [Run {run_idx}/{num_runs}] Finished: {outcome} (Ante: {record.max_ante}, Bank: ${record.max_money}, Pred Acc: {record.exact_prediction_pct:.1f}%, Duration: {record.duration_sec:.1f}s) ──")

        print_summary(session_records, self.csv_path)
        return session_records

    def run_single_game(self, run_index: int = 1) -> RunRecord:
        """Play a single run from start to finish."""
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Telemetry tracking variables
        max_ante = 1
        final_round = 0
        max_money = 0
        won = False
        fatal_boss = "NONE"
        most_played_hand = "High Card"
        final_jokers = ""
        seed_used = self.seed or ""
        total_played_hands = 0
        total_discards_used = 0
        exact_predictions = 0

        # Round tracking variables
        curr_round_ante = 1
        curr_round_num = 1
        curr_round_blind = "Small Blind"
        curr_round_target = 300
        curr_round_hands_played = 0
        curr_round_discards_used = 0
        curr_round_hand_types: list[str] = []
        curr_round_hand_scores: list[int] = []
        round_recorded = False

        failed_shop_cards: set[int] = set()
        failed_shop_packs: set[int] = set()
        failed_shop_vouchers: set[int] = set()
        last_shop_act: Any = None

        # Watchdog counters
        last_seen_state = ""
        consecutive_state_actions = 0
        total_game_steps = 0
        max_game_steps = 2500  # Hard circuit breaker

        # 1. Start fresh run (or resume only on run 1 if mid-game and not won/over)
        state = fetch_gamestate(self.client)
        needs_fresh_run = (
            run_index > 1
            or bool(self.seed)
            or state.state in (GameScreen.MENU, GameScreen.GAME_OVER)
            or state.state == "UNKNOWN"
            or state.won
            or state.ante_num > 8
        )
        if needs_fresh_run:
            self._log("  Starting fresh run via start_run...")
            state = start_run(self.client, deck=self.deck, stake=self.stake, seed=self.seed)
            self._sleep_delay()

        # Capture seed if not already fixed
        if not seed_used and state.seed:
            seed_used = state.seed

        # 2. Main Game Loop
        while total_game_steps < max_game_steps:
            total_game_steps += 1

            # Update metrics
            if state.ante_num > max_ante:
                max_ante = state.ante_num
            if state.round_num > final_round:
                final_round = state.round_num
            if state.money > max_money:
                max_money = state.money
            if state.seed and not seed_used:
                seed_used = state.seed

            # Track most played hand
            if state.state != GameScreen.SHOP:
                failed_shop_cards.clear()
                failed_shop_packs.clear()
                failed_shop_vouchers.clear()
                last_shop_act = None
            if state.hands:
                played_hands = [(h, info.played) for h, info in state.hands.items() if info.played > 0]
                if played_hands:
                    played_hands.sort(key=lambda item: item[1], reverse=True)
                    most_played_hand = played_hands[0][0]

            # Track final jokers
            if state.jokers.cards:
                final_jokers = "|".join([j.key for j in state.jokers.cards])

            # Check Victory Condition (Ante 8 Boss defeated / reached Ante 9 / state.won)
            has_beaten_ante8 = state.won or (state.ante_num > 8)
            if has_beaten_ante8:
                if not won:
                    won = True
                    self._log("\n★ VICTORY! Ante 8 Boss defeated! (Run marked as WON)")

                if not self.endless:
                    # In standard mode, victory at Ante 8 ends the run immediately
                    self._log(f"  Standard mode: Stopping run at victory (Ante {state.ante_num}).")
                    fatal_boss = "NONE"
                    try:
                        self.client.call("menu")
                    except Exception as e:
                        logger.warning(f"Failed to return to menu after victory: {e}")
                    break

            # Check Defeat / Game Over
            if state.state == GameScreen.GAME_OVER:
                self._log(f"\n✗ Game Over reached at Ante {state.ante_num}, Round {state.round_num}.")
                active_b = state.active_blind
                b_name = active_b.name if active_b and active_b.name else (
                    state.blinds.get("boss").name if "boss" in state.blinds and state.blinds["boss"].name else "Unknown Blind"
                )
                if won:
                    fatal_boss = f"Endless - {b_name}"
                else:
                    fatal_boss = b_name

                # Record final losing round if not already recorded
                if not round_recorded and (curr_round_hands_played > 0 or curr_round_discards_used > 0):
                    record_round(
                        RoundRecord(
                            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            seed=seed_used,
                            ante=curr_round_ante,
                            round_num=curr_round_num,
                            blind_name=curr_round_blind,
                            target_score=curr_round_target,
                            hands_played=curr_round_hands_played,
                            discards_used=curr_round_discards_used,
                            hand_types_played="|".join(curr_round_hand_types),
                            hand_scores="|".join(str(s) for s in curr_round_hand_scores),
                            total_score=sum(curr_round_hand_scores),
                            outcome="LOST",
                        ),
                        self.rounds_csv_path,
                    )
                    round_recorded = True

                try:
                    self.client.call("menu")
                except Exception as e:
                    logger.warning(f"Failed to return to menu after game over: {e}")
                break

            # Watchdog: Track consecutive actions in identical state
            if state.state == last_seen_state:
                consecutive_state_actions += 1
            else:
                last_seen_state = state.state
                consecutive_state_actions = 1

            # Watchdog Trigger: Prevent infinite loop if stuck
            if consecutive_state_actions > MAX_CONSECUTIVE_ACTIONS_PER_STATE:
                logger.warning(
                    f"Watchdog triggered: {consecutive_state_actions} actions in state '{state.state}'. Forcing escape."
                )
                if state.state == GameScreen.SHOP:
                    self._log("  [Watchdog] Forcing next_round to exit shop.")
                    raw = self.client.call("next_round")
                    state = parse_gamestate(raw)
                    consecutive_state_actions = 0
                    continue
                elif state.state in GameScreen.BOOSTER_STATES:
                    self._log("  [Watchdog] Forcing skip to close pack.")
                    raw = self.client.call("pack", {"skip": True})
                    state = parse_gamestate(raw)
                    consecutive_state_actions = 0
                    continue
                elif state.state == GameScreen.SELECTING_HAND and state.hand.cards:
                    self._log("  [Watchdog] Forcing play on single card.")
                    raw = self.client.call("play", {"cards": [0]})
                    state = parse_gamestate(raw)
                    consecutive_state_actions = 0
                    continue

            # ── State Machine Dispatch ──────────────────────────────────────
            try:
                # Case A: BLIND_SELECT (Play blind, never skip)
                if state.state == GameScreen.BLIND_SELECT:
                    active_b = state.active_blind
                    b_name = active_b.name if active_b else "Blind"
                    target_score = active_b.score if active_b else 300
                    tag_info = f", Tag: {active_b.tag_name}" if (active_b and active_b.tag_name) else ""

                    # Initialize round tracking
                    curr_round_ante = state.ante_num
                    curr_round_num = state.round_num
                    curr_round_blind = b_name
                    curr_round_target = target_score
                    curr_round_hands_played = 0
                    curr_round_discards_used = 0
                    curr_round_hand_types.clear()
                    curr_round_hand_scores.clear()
                    round_recorded = False

                    self._log(f"  [Blind Select] Selecting {b_name} (Ante {state.ante_num}, Goal: {target_score:,} pts{tag_info})")
                    state = select_blind(self.client)
                    self._sleep_delay()
                    continue

                # Case B: SELECTING_HAND (Combat Loop)
                elif state.state == GameScreen.SELECTING_HAND:
                    # 1. Check if any held consumable can be used in combat
                    held_act = manage_held_consumables(state)
                    if held_act and held_act.action_type == "use":
                        self._log(f"  [Combat Consumable] Using {held_act.card_name} | Reason: {held_act.reason}")
                        raw = self.client.call(held_act.rpc_method, held_act.rpc_params)
                        state = parse_gamestate(raw)
                        self._sleep_delay()
                        continue

                    # 2. Check and reorder Jokers if needed before playing
                    if len(state.jokers.cards) > 1:
                        optimal_order = compute_optimal_joker_order(state.jokers.cards, state.ante_num)
                        if optimal_order != list(range(len(state.jokers.cards))):
                            reordered_keys = [state.jokers.cards[idx].key for idx in optimal_order]
                            self._log(f"  [Jokers Rearrange] Reordering jokers -> {' -> '.join(reordered_keys)}")
                            raw = self.client.call("rearrange", {"jokers": optimal_order})
                            state = parse_gamestate(raw)
                            self._sleep_delay()

                    # 3. Combat Decision (Play vs Discard)
                    decision = decide_combat_action(state)
                    c_names = [f"{c.value.rank}{c.value.suit}" for c in decision.cards]
                    chips_before = state.round.chips
                    target_chips = state.active_blind.score if state.active_blind else 300
                    chips_remaining = max(0, target_chips - state.round.chips)

                    if decision.action == "play":
                        htype = decision.candidate.hand_type if decision.candidate else "Hand"
                        self._log(f"  [Combat PLAY] {htype.upper()} ({', '.join(c_names)}) | Need {chips_remaining:,} pts | Hands left: {state.round.hands_left} | {decision.reason}")
                    else:
                        self._log(f"  [Combat DISCARD] {len(c_names)} cards ({', '.join(c_names)}) | Discards left: {state.round.discards_left} | {decision.reason}")

                    raw = self.client.call(decision.action, {"cards": decision.indices})
                    state = parse_gamestate(raw)

                    # Check predicted score vs actual score when playing a hand
                    if decision.action == "play":
                        total_played_hands += 1
                        curr_round_hands_played += 1
                        if decision.candidate:
                            curr_round_hand_types.append(decision.candidate.hand_type)
                        else:
                            curr_round_hand_types.append("Unknown")

                        if state.round.chips > chips_before:
                            actual_score = state.round.chips - chips_before
                        elif state.active_blind and state.active_blind.score > chips_before and state.state in (GameScreen.ROUND_EVAL, "CASH_OUT", GameScreen.SHOP):
                            actual_score = state.active_blind.score - chips_before
                        else:
                            actual_score = max(0, state.round.chips - chips_before)

                        curr_round_hand_scores.append(actual_score)

                        diff = actual_score - decision.predicted_score
                        is_exact = (diff == 0)
                        if is_exact:
                            exact_predictions += 1
                            self._log(f"  [Score Check] Predicted: {decision.predicted_score:,} pts vs Actual: {actual_score:,} pts (MATCH ✓)")
                        else:
                            diff_str = f"+{diff}" if diff > 0 else str(diff)
                            pct_err = (abs(diff) / max(1, actual_score)) * 100.0
                            self._log(f"  [Score Check] Predicted: {decision.predicted_score:,} pts vs Actual: {actual_score:,} pts (DIFF Δ={diff_str} / Err: {pct_err:.1f}%)")
                    else:
                        total_discards_used += 1
                        curr_round_discards_used += 1

                    self._sleep_delay()
                    continue

                # Case C: ROUND_EVAL / CASH_OUT
                elif state.state in (GameScreen.ROUND_EVAL, "CASH_OUT"):
                    self._log(f"  [Round Complete] Beat {curr_round_blind} with {state.round.chips:,} chips! (Bank: ${state.money}) Cashing out to Shop...")

                    # Record round telemetry
                    if not round_recorded:
                        record_round(
                            RoundRecord(
                                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                seed=seed_used,
                                ante=curr_round_ante,
                                round_num=curr_round_num,
                                blind_name=curr_round_blind,
                                target_score=curr_round_target,
                                hands_played=curr_round_hands_played,
                                discards_used=curr_round_discards_used,
                                hand_types_played="|".join(curr_round_hand_types),
                                hand_scores="|".join(str(s) for s in curr_round_hand_scores),
                                total_score=sum(curr_round_hand_scores),
                                outcome="WON",
                            ),
                            self.rounds_csv_path,
                        )
                        round_recorded = True

                    raw = self.client.call("cash_out")
                    state = parse_gamestate(raw)
                    self._sleep_delay()
                    continue

                # Case D: SHOP
                elif state.state == GameScreen.SHOP:
                    shop_act = decide_shop_action(
                        state,
                        exclude_card_indices=failed_shop_cards,
                        exclude_pack_indices=failed_shop_packs,
                        exclude_voucher_indices=failed_shop_vouchers,
                    )
                    last_shop_act = shop_act
                    self._log(f"  [Shop] (Bank: ${state.money}) -> {shop_act.action_type.upper()} {shop_act.target_name} (${shop_act.cost}) | {shop_act.reason}")

                    if shop_act.action_type == ShopActionType.NEXT_ROUND:
                        raw = self.client.call("next_round")
                        state = parse_gamestate(raw)
                        self._sleep_delay()
                        continue

                    raw = self.client.call(shop_act.rpc_method, shop_act.rpc_params)
                    state = parse_gamestate(raw)
                    self._sleep_delay()
                    continue

                # Case E: Booster Packs (SMODS_BOOSTER_OPENED, etc.)
                elif state.state in GameScreen.BOOSTER_STATES:
                    pack_act = decide_pack_action(state)
                    self._log(f"  [Booster] Picked {pack_act.card_name} ({pack_act.action_type.upper()}) | {pack_act.reason}")
                    raw = self.client.call(pack_act.rpc_method, pack_act.rpc_params)
                    state = parse_gamestate(raw)
                    self._sleep_delay()
                    continue

                # Case F: Transient States (Animations, transitions)
                elif state.state in TRANSIENT_STATES:
                    time.sleep(0.3)
                    state = fetch_gamestate(self.client)
                    continue

                # Case G: Unknown / unexpected state fallback
                else:
                    logger.warning(f"Unhandled state '{state.state}'. Refreshing gamestate...")
                    time.sleep(0.5)
                    state = fetch_gamestate(self.client)

            except RPCError as exc:
                self._log(f"  ⚠ [Action Failed] {exc.message}. Resynchronising game state...")
                if state.state == GameScreen.SHOP and last_shop_act:
                    if last_shop_act.action_type == ShopActionType.BUY_CARD and "card" in last_shop_act.rpc_params:
                        failed_shop_cards.add(last_shop_act.rpc_params["card"])
                    elif last_shop_act.action_type == ShopActionType.BUY_PACK and "pack" in last_shop_act.rpc_params:
                        failed_shop_packs.add(last_shop_act.rpc_params["pack"])
                    elif last_shop_act.action_type == ShopActionType.BUY_VOUCHER and "voucher" in last_shop_act.rpc_params:
                        failed_shop_vouchers.add(last_shop_act.rpc_params["voucher"])
                    last_shop_act = None
                time.sleep(0.5)
                state = fetch_gamestate(self.client)
                continue

        if total_played_hands > 0:
            # pyrefly: ignore[division-by-zero]
            exact_prediction_pct = round((exact_predictions / total_played_hands) * 100.0, 1)
        else:
            exact_prediction_pct = 0.0

        duration = round(time.time() - start_time, 1)

        # Hands breakdown summary
        hands_breakdown = ""
        if state.hands:
            breakdown_list = [f"{h}:{info.played}" for h, info in sorted(state.hands.items(), key=lambda x: x[1].played, reverse=True) if info.played > 0]
            hands_breakdown = "|".join(breakdown_list)

        # Return final telemetry record
        return RunRecord(
            timestamp=timestamp,
            seed=seed_used,
            deck=self.deck,
            stake=self.stake,
            win=won,
            max_ante=max_ante,
            final_round=final_round,
            fatal_boss=fatal_boss,
            max_money=max_money,
            most_played_hand=most_played_hand,
            final_jokers=final_jokers,
            duration_sec=duration,
            exact_prediction_pct=exact_prediction_pct,
            total_hands_played=total_played_hands,
            total_discards_used=total_discards_used,
            hands_breakdown=hands_breakdown,
        )
