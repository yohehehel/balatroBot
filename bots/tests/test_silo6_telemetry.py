"""Unit tests for Silo 6: Multi-Run Autonomous Orchestrator, Endless Mode, and Telemetry.

Tests:
1. CSV Creation & Writing:
   - Recording multiple runs (wins and losses) into a temporary CSV file (tmp_path).
   - Verifying columns, types, and aggregate stats (winrate %, average ante, top fatal bosses).
2. Victory Detection in Standard Mode (endless=False):
   - When Ante 8 Boss is defeated (transition to Ante 9+ or state.won), BotRunner stops immediately,
     records win=True, fatal_boss="NONE", and ends run cleanly.
3. Endless Mode (endless=True):
   - When beating Ante 8 and continuing until losing at Ante 11, the run is recorded with
     win=True, max_ante=11, and fatal_boss mentioning the final blind.
4. Watchdog Anti-Infinite Loop:
   - If an identical state (e.g. SHOP) persists beyond safety threshold, the watchdog triggers,
     forces escape (e.g. next_round), and prevents blocking the session.
5. Telemetry Summary Display:
   - Verifying print_summary produces clean console reports with top hands and fatal bosses.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from bots.client_wrapper import RPCError
from bots.runner import BotRunner
from bots.state_parser import GameScreen
from bots.telemetry import (
    CSV_COLUMNS,
    RunRecord,
    compute_stats,
    load_run_history,
    print_summary,
    record_run,
)

# ── Mock Client for Runner Testing ─────────────────────────────────────────

class MockBalatroClient:
    """Mock client that responds with a scripted sequence of gamestate dictionaries or errors."""

    def __init__(self, sequence: list[Any] | None = None) -> None:
        self.sequence = list(sequence) if sequence else []
        self.history: list[tuple[str, dict[str, Any] | None]] = []

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.history.append((method, params))
        if self.sequence:
            item = self.sequence.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        # Default safety fallback
        return {
            "state": GameScreen.GAME_OVER,
            "ante_num": 1,
            "round_num": 1,
            "money": 0,
            "blinds": {"boss": {"name": "The Wall", "status": "CURRENT"}},
        }


# ── Test Suite ────────────────────────────────────────────────────────────

class TestSilo6Telemetry:
    """Test 1: CSV Recording and Aggregate Statistics."""

    def test_csv_creation_and_stats_calculation(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "runs_history.csv"

        # Record 4 runs: 2 wins, 2 losses
        runs = [
            RunRecord(
                timestamp="2026-10-26 20:00:00",
                seed="ABC123",
                deck="Red Deck",
                stake="White Stake",
                win=True,
                max_ante=8,
                final_round=24,
                fatal_boss="NONE",
                max_money=55,
                most_played_hand="Flush",
                final_jokers="j_joker|j_cavendish",
                duration_sec=120.5,
                exact_prediction_pct=100.0,
            ),
            RunRecord(
                timestamp="2026-10-26 20:10:00",
                seed="XYZ789",
                deck="Red Deck",
                stake="White Stake",
                win=False,
                max_ante=4,
                final_round=11,
                fatal_boss="The Wall",
                max_money=22,
                most_played_hand="Pair",
                final_jokers="j_sly",
                duration_sec=65.0,
                exact_prediction_pct=50.0,
            ),
            RunRecord(
                timestamp="2026-10-26 20:20:00",
                seed="RUN003",
                deck="Red Deck",
                stake="White Stake",
                win=True,
                max_ante=9,
                final_round=26,
                fatal_boss="NONE",
                max_money=48,
                most_played_hand="Flush",
                final_jokers="j_blueprint|j_cavendish",
                duration_sec=135.2,
                exact_prediction_pct=75.0,
            ),
            RunRecord(
                timestamp="2026-10-26 20:30:00",
                seed="RUN004",
                deck="Red Deck",
                stake="White Stake",
                win=False,
                max_ante=6,
                final_round=17,
                fatal_boss="The Flint",
                max_money=35,
                most_played_hand="Pair",
                final_jokers="j_banner",
                duration_sec=88.4,
                exact_prediction_pct=0.0,
            ),
        ]

        for r in runs:
            record_run(r, csv_file)

        # 1. Verify file exists and has correct columns
        assert csv_file.exists()
        with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == CSV_COLUMNS
            assert "exact_prediction_pct" in header
            rows = list(reader)
            assert len(rows) == 4

        # 2. Verify reading back records
        loaded = load_run_history(csv_file)
        assert len(loaded) == 4
        assert loaded[0].seed == "ABC123"
        assert loaded[0].win is True
        assert loaded[0].exact_prediction_pct == 100.0
        assert loaded[1].win is False
        assert loaded[1].exact_prediction_pct == 50.0
        assert loaded[1].fatal_boss == "The Wall"

        # 3. Verify statistics
        stats = compute_stats(loaded)
        assert stats["total_runs"] == 4
        assert stats["wins"] == 2
        assert stats["winrate_pct"] == 50.0
        assert stats["avg_max_ante"] == (8 + 4 + 9 + 6) / 4  # 6.75
        assert stats["avg_exact_prediction_pct"] == round((100.0 + 50.0 + 75.0 + 0.0) / 4, 1)  # 56.2
        assert stats["top_hands"] == [("Flush", 2), ("Pair", 2)]
        assert set([b[0] for b in stats["top_fatal_bosses"]]) == {"The Wall", "The Flint"}

    def test_legacy_csv_migration(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "legacy_history.csv"
        # Write legacy 12-column header and a row
        legacy_header = [
            "timestamp", "seed", "deck", "stake", "win", "max_ante",
            "final_round", "fatal_boss", "max_money", "most_played_hand",
            "final_jokers", "duration_sec"
        ]
        legacy_row = [
            "2026-10-25 12:00:00", "OLD123", "Red Deck", "White Stake", "True",
            "8", "24", "NONE", "50", "Straight", "j_joker", "110.0"
        ]
        with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(legacy_header)
            writer.writerow(legacy_row)

        # Loading legacy file should gracefully default exact_prediction_pct to 0.0
        loaded = load_run_history(csv_file)
        assert len(loaded) == 1
        assert loaded[0].seed == "OLD123"
        assert loaded[0].exact_prediction_pct == 0.0

        # Recording a new run should migrate the file header to CSV_COLUMNS
        new_record = RunRecord(
            timestamp="2026-10-26 14:00:00",
            seed="NEW456",
            deck="Red Deck",
            stake="White Stake",
            win=True,
            max_ante=8,
            final_round=24,
            fatal_boss="NONE",
            max_money=60,
            most_played_hand="Flush",
            final_jokers="j_cavendish",
            duration_sec=95.0,
            exact_prediction_pct=100.0,
        )
        record_run(new_record, csv_file)

        # Verify header is migrated and both records are readable
        reloaded = load_run_history(csv_file)
        assert len(reloaded) == 2
        assert reloaded[0].seed == "OLD123"
        assert reloaded[1].seed == "NEW456"
        assert reloaded[1].exact_prediction_pct == 100.0


class TestSilo6VictoryStandardMode:
    """Test 2: Victory detection in standard mode (Ante 8 win stops run)."""

    def test_standard_mode_stops_immediately_on_ante8_win(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test_standard.csv"

        # Mock game sequence:
        # 1. fetch_gamestate -> MENU
        # 2. start_run calls menu -> MENU
        # 3. start_run calls start -> returns Ante 8 state (Boss blind defeated)
        # 4. Beating boss transitions to Ante 9 (won = True)
        mock_sequence = [
            # Initial state
            {"state": GameScreen.MENU, "ante_num": 0, "round_num": 0, "money": 4},
            # After menu
            {"state": GameScreen.MENU, "ante_num": 0, "round_num": 0, "money": 4},
            # After start
            {
                "state": GameScreen.ROUND_EVAL,
                "ante_num": 8,
                "round_num": 24,
                "money": 30,
                "won": True,
                "seed": "WINSEED",
                "jokers": {"cards": [{"key": "j_cavendish", "set": "Joker"}]},
                "hands": {"Flush": {"played": 8, "level": 3}},
            },
        ]

        client = MockBalatroClient(mock_sequence)
        runner = BotRunner(
            client=client,  # type: ignore[arg-type]
            endless=False,
            delay=0.0,
            csv_path=csv_file,
        )

        record = runner.run_single_game(run_index=1)

        assert record.win is True
        assert record.max_ante == 8
        assert record.fatal_boss == "NONE"
        assert record.seed == "WINSEED"
        assert record.most_played_hand == "Flush"
        assert "j_cavendish" in record.final_jokers

    def test_subsequent_run_starts_fresh_after_victory(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test_standard_multi.csv"

        # Run 1:
        # fetch_gamestate -> MENU
        # start_run calls menu -> MENU
        # start_run calls start -> returns Ante 8 won
        # victory -> calls menu
        # Run 2:
        # fetch_gamestate -> returns MENU
        # run_index=2 forces start_run -> calls menu -> calls start -> returns fresh Ante 1 game over
        mock_sequence = [
            # Run 1
            {"state": GameScreen.MENU, "ante_num": 0, "round_num": 0, "money": 4},
            {"state": GameScreen.MENU, "ante_num": 0, "round_num": 0, "money": 4},
            {
                "state": GameScreen.ROUND_EVAL,
                "ante_num": 8,
                "round_num": 24,
                "money": 30,
                "won": True,
                "seed": "WINSEED1",
            },
            {"state": GameScreen.MENU, "ante_num": 0, "round_num": 0, "money": 4},
            # Run 2
            {"state": GameScreen.MENU, "ante_num": 0, "round_num": 0, "money": 4},
            {"state": GameScreen.MENU, "ante_num": 0, "round_num": 0, "money": 4},
            {"state": GameScreen.GAME_OVER, "ante_num": 1, "round_num": 1, "money": 4, "won": False, "seed": "RUN2"},
        ]

        client = MockBalatroClient(mock_sequence)
        runner = BotRunner(
            client=client,  # type: ignore[arg-type]
            endless=False,
            delay=0.0,
            csv_path=csv_file,
        )

        records = runner.run_session(num_runs=2)
        assert len(records) == 2
        assert records[0].win is True
        assert records[0].seed == "WINSEED1"
        assert records[1].win is False
        assert records[1].seed == "RUN2"


class TestSilo6EndlessMode:
    """Test 3: Endless mode continues past Ante 8 until game over."""

    def test_endless_mode_continues_until_game_over(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test_endless.csv"

        mock_sequence = [
            # 1. Initial state
            {"state": GameScreen.MENU, "ante_num": 0, "round_num": 0, "money": 4},
            # 2. After menu
            {"state": GameScreen.MENU, "ante_num": 0, "round_num": 0, "money": 4},
            # 3. After start: Ante 8 beaten (won = True)
            {
                "state": GameScreen.ROUND_EVAL,
                "ante_num": 8,
                "round_num": 24,
                "money": 40,
                "won": True,
                "seed": "ENDLESS_SEED",
                "jokers": {"cards": [{"key": "j_blueprint", "set": "Joker"}]},
            },
            # 4. Cash out to shop at Ante 9
            {
                "state": GameScreen.SHOP,
                "ante_num": 9,
                "round_num": 25,
                "money": 45,
                "won": True,
                "shop": {"cards": []},
                "vouchers": {"cards": []},
                "packs": {"cards": []},
            },
            # 5. Leave shop to Blind select Ante 10
            {
                "state": GameScreen.BLIND_SELECT,
                "ante_num": 10,
                "round_num": 28,
                "money": 45,
                "won": True,
                "blinds": {"small": {"name": "Small Blind", "status": "SELECT"}},
            },
            # 6. Finally Game Over at Ante 11
            {
                "state": GameScreen.GAME_OVER,
                "ante_num": 11,
                "round_num": 31,
                "money": 50,
                "won": True,
                "blinds": {"boss": {"name": "The Violet Vessel", "status": "CURRENT"}},
                "jokers": {"cards": [{"key": "j_blueprint", "set": "Joker"}]},
            },
        ]

        client = MockBalatroClient(mock_sequence)
        runner = BotRunner(
            client=client,  # type: ignore[arg-type]
            endless=True,
            delay=0.0,
            csv_path=csv_file,
        )

        record = runner.run_single_game(run_index=1)

        # In endless mode, win remains True, but max_ante reaches 11
        assert record.win is True
        assert record.max_ante == 11
        assert "The Violet Vessel" in record.fatal_boss
        assert record.seed == "ENDLESS_SEED"


class TestSilo6Watchdog:
    """Test 4: Watchdog prevents infinite loops in stuck states."""

    def test_watchdog_forces_exit_when_stuck_in_shop(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test_watchdog.csv"

        # Create a stuck shop state that repeats indefinitely
        stuck_shop = {
            "state": GameScreen.SHOP,
            "ante_num": 2,
            "round_num": 4,
            "money": 20,
            "shop": {"cards": []},
            "vouchers": {"cards": []},
            "packs": {"cards": []},
            "round": {"reroll_cost": 999},  # Cannot reroll
        }

        # Sequence of 30 identical shop states, followed by GAME_OVER
        sequence: list[dict[str, Any]] = [stuck_shop] * 30
        sequence.append({
            "state": GameScreen.GAME_OVER,
            "ante_num": 2,
            "round_num": 4,
            "money": 20,
            "blinds": {"boss": {"name": "The Pillar", "status": "CURRENT"}},
        })

        client = MockBalatroClient(sequence)
        runner = BotRunner(
            client=client,  # type: ignore[arg-type]
            endless=False,
            delay=0.0,
            csv_path=csv_file,
        )

        record = runner.run_single_game(run_index=1)

        # Check that watchdog forced next_round
        methods_called = [item[0] for item in client.history]
        assert "next_round" in methods_called
        assert record.max_ante == 2


class TestSilo6SummaryReport:
    """Test 5: Console summary report output."""

    def test_print_summary_runs_cleanly(self, capsys: pytest.CaptureFixture[str]) -> None:
        records = [
            RunRecord(
                timestamp="2026-10-26 22:00:00",
                seed="S1",
                deck="Red Deck",
                stake="White Stake",
                win=True,
                max_ante=8,
                final_round=24,
                fatal_boss="NONE",
                max_money=40,
                most_played_hand="Flush",
                final_jokers="j_joker",
                duration_sec=100.0,
            )
        ]

        print_summary(records)
        captured = capsys.readouterr().out

        assert "BALATRO BOT – TELEMETRY & RUN SUMMARY" in captured
        assert "Total Runs       : 1" in captured
        assert "Wins             : 1 / 1 (100.0%)" in captured
        assert "Avg Pred Accuracy: 0.0%" in captured
        assert "Top 3 Hands      : Flush (1)" in captured


class TestSilo6ErrorRecovery:
    """Test 6: Resilience against temporary RPCError (e.g. invalid state)."""

    def test_runner_recovers_gracefully_from_temporary_rpc_error(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test_recovery.csv"

        # Mock sequence:
        # 1. Initial state -> SHOP with a booster pack
        # 2. Buy booster throws RPCError (state transition glitch in Balatro)
        # 3. fetch_gamestate resynchronises -> returns SHOP
        # 4. Next round exits shop -> Blind Select
        # 5. Blind Select -> Victory
        shop_with_pack = {
            "state": GameScreen.SHOP,
            "ante_num": 3,
            "round_num": 9,
            "money": 20,
            "shop": {"cards": []},
            "vouchers": {"cards": []},
            "packs": {"cards": [{"key": "p_celestial_normal_1", "set": "Booster", "cost": {"buy": 4}}]},
        }

        mock_sequence: list[Any] = [
            # 1. fetch_gamestate initial
            shop_with_pack,
            # 2. buy pack raises RPCError (e.g. Method 'buy' requires one of these states: SHOP)
            RPCError(code=-32002, message="Method 'buy' requires one of these states: SHOP"),
            # 3. fetch_gamestate after catch
            {
                "state": GameScreen.SHOP,
                "ante_num": 3,
                "round_num": 9,
                "money": 20,
                "shop": {"cards": []},
                "vouchers": {"cards": []},
                "packs": {"cards": []},  # Now pack is handled or shop updated
            },
            # 4. next_round to exit shop
            {
                "state": GameScreen.ROUND_EVAL,
                "ante_num": 8,
                "round_num": 24,
                "money": 40,
                "won": True,
            },
        ]

        client = MockBalatroClient(mock_sequence)
        runner = BotRunner(
            client=client,  # type: ignore[arg-type]
            endless=False,
            delay=0.0,
            csv_path=csv_file,
        )

        # Must not raise RPCError! Must complete cleanly
        record = runner.run_single_game(run_index=1)
        assert record.win is True
        assert record.max_ante == 8


class TestSilo6CombatScoreCheck:
    """Test 7: Score Check Predicted vs Actual and Exact Prediction Percentage."""

    def test_exact_score_prediction_yields_100_percent(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test_score_check_exact.csv"

        # Combat hand: Pair of Aces (A_H, A_D, K_C, Q_S, J_H)
        # Pair base: 10 chips, 2 mult + Aces (11+11) = 32 chips * 2 mult = 64 pts
        combat_state = {
            "state": GameScreen.SELECTING_HAND,
            "ante_num": 1,
            "round_num": 1,
            "money": 4,
            "hand": {
                "cards": [
                    {"value": {"rank": "A", "suit": "H"}},
                    {"value": {"rank": "A", "suit": "D"}},
                    {"value": {"rank": "K", "suit": "C"}},
                    {"value": {"rank": "Q", "suit": "S"}},
                    {"value": {"rank": "J", "suit": "H"}},
                ]
            },
            "round": {"chips": 0, "hands_left": 4, "discards_left": 0},
            "blinds": {"small": {"name": "Small Blind", "score": 50, "status": "CURRENT"}},
        }

        # Response to play: returns 64 chips scored (exact match)
        post_play_state = {
            "state": GameScreen.ROUND_EVAL,
            "ante_num": 8,
            "round_num": 24,
            "money": 10,
            "won": True,
            "round": {"chips": 64},
        }

        client = MockBalatroClient([combat_state, post_play_state])
        runner = BotRunner(
            client=client,  # type: ignore[arg-type]
            endless=False,
            delay=0.0,
            csv_path=csv_file,
        )

        record = runner.run_single_game(run_index=1)
        assert record.win is True
        assert record.exact_prediction_pct == 100.0

    def test_differing_score_prediction_yields_zero_percent(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test_score_check_diff.csv"

        combat_state = {
            "state": GameScreen.SELECTING_HAND,
            "ante_num": 1,
            "round_num": 1,
            "money": 4,
            "hand": {
                "cards": [
                    {"value": {"rank": "A", "suit": "H"}},
                    {"value": {"rank": "A", "suit": "D"}},
                    {"value": {"rank": "K", "suit": "C"}},
                    {"value": {"rank": "Q", "suit": "S"}},
                    {"value": {"rank": "J", "suit": "H"}},
                ]
            },
            "round": {"chips": 0, "hands_left": 4, "discards_left": 0},
            "blinds": {"small": {"name": "Small Blind", "score": 50, "status": "CURRENT"}},
        }

        # Response to play: returns 200 chips scored (unexpected difference)
        post_play_state = {
            "state": GameScreen.ROUND_EVAL,
            "ante_num": 8,
            "round_num": 24,
            "money": 10,
            "won": True,
            "round": {"chips": 200},
        }

        client = MockBalatroClient([combat_state, post_play_state])
        runner = BotRunner(
            client=client,  # type: ignore[arg-type]
            endless=False,
            delay=0.0,
            csv_path=csv_file,
        )

        record = runner.run_single_game(run_index=1)
        assert record.win is True
        assert record.exact_prediction_pct == 0.0


