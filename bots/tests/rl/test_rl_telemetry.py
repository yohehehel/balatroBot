"""Unit tests for RL Telemetry, CSV Datasets, and TensorBoard Integration."""

import csv
from pathlib import Path
from bots.rl.telemetry_callback import RLTelemetryCallback
from bots.telemetry import (
    PurchaseRecord,
    RoundRecord,
    RunRecord,
    record_purchase,
    record_round,
    record_run,
)


def test_purchase_record_csv_logging(tmp_path: Path):
    csv_file = tmp_path / "test_purchases.csv"
    p1 = PurchaseRecord(
        timestamp="2026-10-28 23:00:00",
        worker_id=1,
        seed="TEST1234",
        ante=2,
        round_num=3,
        money=15,
        action="BUY",
        category="joker",
        item_key="j_half_joker",
        item_name="Half Joker",
    )
    p2 = PurchaseRecord(
        timestamp="2026-10-28 23:01:00",
        worker_id=2,
        seed="TEST1234",
        ante=2,
        round_num=3,
        money=10,
        action="USE",
        category="tarot",
        item_key="c_the_sun",
        item_name="The Sun",
    )
    record_purchase(p1, csv_path=csv_file)
    record_purchase(p2, csv_path=csv_file)

    assert csv_file.exists()
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["category"] == "joker"
        assert reader[0]["item_name"] == "Half Joker"
        assert reader[1]["action"] == "USE"
        assert reader[1]["item_key"] == "c_the_sun"


def test_round_and_run_record_enriched_fields(tmp_path: Path):
    runs_file = tmp_path / "test_runs.csv"
    rounds_file = tmp_path / "test_rounds.csv"

    round_rec = RoundRecord(
        timestamp="2026-10-28 23:00:00",
        seed="TESTSEED",
        ante=2,
        round_num=4,
        blind_name="Small Blind",
        target_score=600,
        hands_played=2,
        discards_used=1,
        hand_types_played="High Card;Flush",
        hand_scores="250;800",
        total_score=800,
        outcome="WON",
        jokers_held="j_joker|j_half",
        money=12,
        hands_left=2,
        discards_left=2,
    )
    record_round(round_rec, csv_path=rounds_file)

    run_rec = RunRecord(
        timestamp="2026-10-28 23:10:00",
        seed="TESTSEED",
        deck="RED",
        stake="WHITE",
        win=False,
        max_ante=3,
        final_round=7,
        fatal_boss="The Wall",
        max_money=25,
        most_played_hand="Flush",
        final_jokers="j_joker|j_half",
        duration_sec=120.5,
        exact_prediction_pct=0.0,
        total_hands_played=14,
        total_discards_used=8,
        hands_breakdown="High Card:4|Flush:10",
        jokers_bought="j_joker|j_half",
        consumables_bought="c_sun|c_earth",
        cards_bought="",
        vouchers_bought="",
        packs_opened="p_arcana",
        consumables_used="The Sun|Earth",
        blinds_played=6,
        blinds_skipped=1,
        shop_rerolls=2,
        total_reward=18.5,
    )
    record_run(run_rec, csv_path=runs_file)

    with open(rounds_file, mode="r", encoding="utf-8") as f:
        r_rows = list(csv.DictReader(f))
        assert len(r_rows) == 1
        assert r_rows[0]["jokers_held"] == "j_joker|j_half"
        assert r_rows[0]["hand_types_played"] == "High Card;Flush"

    with open(runs_file, mode="r", encoding="utf-8") as f:
        run_rows = list(csv.DictReader(f))
        assert len(run_rows) == 1
        assert run_rows[0]["jokers_bought"] == "j_joker|j_half"
        assert run_rows[0]["consumables_used"] == "The Sun|Earth"
        assert float(run_rows[0]["total_reward"]) == 18.5


def test_telemetry_callback_records_metrics():
    from types import SimpleNamespace
    cb = RLTelemetryCallback(rolling_window=5)

    class DummyLogger:
        def __init__(self):
            self.records = {}
        def record(self, key, value):
            self.records[key] = value

    dummy_logger = DummyLogger()
    dummy_model = SimpleNamespace(logger=dummy_logger)
    cb.init_callback(dummy_model)
    cb.locals = {
        "infos": [
            {
                "round_summary": {
                    "round_won": True,
                    "hands_played": 3,
                    "discards_used": 1,
                    "score_obtained": 1200,
                },
                "episode_summary": {
                    "win": False,
                    "max_ante": 4,
                    "final_money": 18,
                    "max_money": 32,
                    "total_reward": 14.2,
                    "hands_played": 12,
                    "discards_used": 6,
                    "blinds_played": 5,
                    "blinds_skipped": 2,
                    "rerolls": 3,
                    "jokers_bought": 4,
                    "consumables_bought": 3,
                    "packs_opened": 2,
                    "consumables_used": 3,
                    "hands_breakdown": {"Pair": 8, "Flush": 4},
                },
            }
        ]
    }

    assert cb._on_step() is True
    assert cb.logger.records["combat/hands_per_round"] == 3
    assert cb.logger.records["combat/round_win_rate"] == 1.0
    assert cb.logger.records["game/max_ante"] == 4
    assert cb.logger.records["shop/jokers_bought"] == 4
    assert cb.logger.records["hands/Pair_pct"] > 0
    assert cb.logger.records["hands/Flush_pct"] > 0


def test_execute_macro_action_shop_transactions():
    from bots.rl.actions import ActionID, execute_macro_action
    from bots.state_parser import AreaInfo, Card, GameScreen, ParsedGameState

    class DummyClient:
        def __init__(self):
            self.calls = []
        def call(self, method, params=None):
            self.calls.append((method, params))
            return {"ok": True}

    state = ParsedGameState(
        state=GameScreen.SHOP,
        money=20,
        shop=AreaInfo(cards=[Card(key="j_joker", label="Joker", set="Joker")]),
        vouchers=AreaInfo(cards=[Card(key="v_overstock_norm", label="Overstock", set="Voucher")]),
        packs=AreaInfo(cards=[Card(key="p_arcana_normal", label="Arcana Pack", set="Booster")]),
    )

    client = DummyClient()

    # Test BUY_SHOP_CARD_0
    res_card = execute_macro_action(ActionID.BUY_SHOP_CARD_0, state, client)
    assert res_card["success"] is True
    assert res_card["transaction"]["action"] == "BUY"
    assert res_card["transaction"]["category"] == "joker"
    assert res_card["transaction"]["item_key"] == "j_joker"

    # Test BUY_VOUCHER
    res_vouch = execute_macro_action(ActionID.BUY_VOUCHER, state, client)
    assert res_vouch["success"] is True
    assert res_vouch["transaction"]["action"] == "BUY"
    assert res_vouch["transaction"]["category"] == "voucher"
    assert res_vouch["transaction"]["item_key"] == "v_overstock_norm"

    # Test BUY_BOOSTER_0
    res_pack = execute_macro_action(ActionID.BUY_BOOSTER_0, state, client)
    assert res_pack["success"] is True
    assert res_pack["transaction"]["action"] == "BUY"
    assert res_pack["transaction"]["category"] == "pack"
    assert res_pack["transaction"]["item_key"] == "p_arcana_normal"

