"""Telemetry and Session Analytics (Silo 6).

Responsibilities:
- Record run data to CSV (bots/runs_history.csv) with columns:
  timestamp,seed,deck,stake,win,max_ante,final_round,fatal_boss,max_money,most_played_hand,final_jokers,duration_sec,
  exact_prediction_pct,total_hands_played,total_discards_used,hands_breakdown
- Record round-level detailed data to CSV (bots/rounds_history.csv) with columns:
  timestamp,seed,ante,round_num,blind_name,target_score,hands_played,discards_used,hand_types_played,hand_scores,total_score,outcome
- Calculate aggregate statistics:
  Total Runs, Wins, Winrate %, Average Max Ante, Average Max Money, Average Duration,
  Top 3 Most Played Hands, Top 3 Fatal Bosses.
- Display a rich, formatted summary report in console.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import threading
from typing import Any

# Default path to runs_history.csv, rounds_history.csv and purchases_history.csv
DEFAULT_CSV_PATH = Path(__file__).parent / "runs_history.csv"
DEFAULT_ROUNDS_CSV_PATH = Path(__file__).parent / "rounds_history.csv"
DEFAULT_PURCHASES_CSV_PATH = Path(__file__).parent / "purchases_history.csv"

_CSV_LOCK = threading.Lock()

CSV_COLUMNS = [
    "timestamp",
    "seed",
    "deck",
    "stake",
    "win",
    "max_ante",
    "final_round",
    "fatal_boss",
    "max_money",
    "most_played_hand",
    "final_jokers",
    "duration_sec",
    "exact_prediction_pct",
    "total_hands_played",
    "total_discards_used",
    "hands_breakdown",
    "jokers_bought",
    "consumables_bought",
    "cards_bought",
    "vouchers_bought",
    "packs_opened",
    "consumables_used",
    "blinds_played",
    "blinds_skipped",
    "shop_rerolls",
    "total_reward",
]

ROUND_CSV_COLUMNS = [
    "timestamp",
    "seed",
    "ante",
    "round_num",
    "blind_name",
    "target_score",
    "hands_played",
    "discards_used",
    "hand_types_played",
    "hand_scores",
    "total_score",
    "outcome",
    "jokers_held",
    "money",
    "hands_left",
    "discards_left",
]

PURCHASE_CSV_COLUMNS = [
    "timestamp",
    "worker_id",
    "seed",
    "ante",
    "round_num",
    "money",
    "action",
    "category",
    "item_key",
    "item_name",
]


@dataclass
class RunRecord:
    """Individual run summary record."""
    timestamp: str
    seed: str
    deck: str
    stake: str
    win: bool
    max_ante: int
    final_round: int
    fatal_boss: str
    max_money: int
    most_played_hand: str
    final_jokers: str
    duration_sec: float
    exact_prediction_pct: float = 0.0
    total_hands_played: int = 0
    total_discards_used: int = 0
    hands_breakdown: str = ""
    jokers_bought: str = ""
    consumables_bought: str = ""
    cards_bought: str = ""
    vouchers_bought: str = ""
    packs_opened: str = ""
    consumables_used: str = ""
    blinds_played: int = 0
    blinds_skipped: int = 0
    shop_rerolls: int = 0
    total_reward: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoundRecord:
    """Detailed round record for fine-grained performance tracking."""
    timestamp: str
    seed: str
    ante: int
    round_num: int
    blind_name: str
    target_score: int
    hands_played: int
    discards_used: int
    hand_types_played: str
    hand_scores: str
    total_score: int
    outcome: str
    jokers_held: str = ""
    money: int = 0
    hands_left: int = 0
    discards_left: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PurchaseRecord:
    """Granular transaction record for purchases, sales, and item usage."""
    timestamp: str
    worker_id: int
    seed: str
    ante: int
    round_num: int
    money: int
    action: str        # BUY, SELL, USE
    category: str      # joker, tarot, planet, spectral, playing_card, voucher, pack
    item_key: str      # e.g. j_half_joker, c_earth, p_arcana_normal
    item_name: str     # e.g. Half Joker, Earth, Arcana Pack

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_run(record: RunRecord, csv_path: Path | str | None = None) -> None:
    """Append a completed run record to the CSV history file.

    Creates the file and writes header if it does not already exist.
    If the file exists with an older header, it migrates the file format safely.
    """
    path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with _CSV_LOCK:
        file_exists = path.exists() and path.stat().st_size > 0

        if file_exists:
            with open(path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, [])
            if header and header != CSV_COLUMNS:
                existing_records = load_run_history(path)
                with open(path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                    writer.writeheader()
                    for rec in existing_records:
                        writer.writerow(rec.to_dict())

        with open(path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record.to_dict())


def record_round(record: RoundRecord, csv_path: Path | str | None = None) -> None:
    """Append a detailed round record to the rounds CSV history file."""
    path = Path(csv_path) if csv_path else DEFAULT_ROUNDS_CSV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with _CSV_LOCK:
        file_exists = path.exists() and path.stat().st_size > 0
        if file_exists:
            with open(path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, [])
            if header and header != ROUND_CSV_COLUMNS:
                with open(path, mode="r", newline="", encoding="utf-8") as f:
                    dict_reader = csv.DictReader(f)
                    existing_rows = list(dict_reader)
                with open(path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=ROUND_CSV_COLUMNS, extrasaction="ignore")
                    writer.writeheader()
                    for r in existing_rows:
                        writer.writerow(r)

        with open(path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ROUND_CSV_COLUMNS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(record.to_dict())


def record_purchase(record: PurchaseRecord, csv_path: Path | str | None = None) -> None:
    """Append a granular purchase/use/sell transaction record to purchases_history.csv."""
    path = Path(csv_path) if csv_path else DEFAULT_PURCHASES_CSV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with _CSV_LOCK:
        file_exists = path.exists() and path.stat().st_size > 0
        with open(path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PURCHASE_CSV_COLUMNS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(record.to_dict())


def load_run_history(csv_path: Path | str | None = None) -> list[RunRecord]:
    """Read and parse all past runs from the CSV history file."""
    path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
    if not path.exists():
        return []

    records: list[RunRecord] = []
    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or not row.get("timestamp"):
                continue
            pct_raw = row.get("exact_prediction_pct")
            if pct_raw is None and None in row and row[None]:
                pct_raw = row[None][0]
            try:
                exact_pct = float(pct_raw) if pct_raw not in (None, "", "N/A") else 0.0
            except (ValueError, TypeError):
                exact_pct = 0.0

            total_hands = 0
            try:
                total_hands = int(row.get("total_hands_played", 0) or 0)
            except (ValueError, TypeError):
                pass

            total_discards = 0
            try:
                total_discards = int(row.get("total_discards_used", 0) or 0)
            except (ValueError, TypeError):
                pass

            breakdown = row.get("hands_breakdown", "") or ""

            records.append(
                RunRecord(
                    timestamp=row.get("timestamp", ""),
                    seed=row.get("seed", ""),
                    deck=row.get("deck", "Red Deck"),
                    stake=row.get("stake", "White Stake"),
                    win=row.get("win", "").strip().lower() in ("true", "1"),
                    max_ante=int(row.get("max_ante", 0)),
                    final_round=int(row.get("final_round", 0)),
                    fatal_boss=row.get("fatal_boss", "NONE"),
                    max_money=int(row.get("max_money", 0)),
                    most_played_hand=row.get("most_played_hand", "None"),
                    final_jokers=row.get("final_jokers", ""),
                    duration_sec=float(row.get("duration_sec", 0.0)),
                    exact_prediction_pct=exact_pct,
                    total_hands_played=total_hands,
                    total_discards_used=total_discards,
                    hands_breakdown=breakdown,
                    jokers_bought=row.get("jokers_bought", "") or "",
                    consumables_bought=row.get("consumables_bought", "") or "",
                    cards_bought=row.get("cards_bought", "") or "",
                    vouchers_bought=row.get("vouchers_bought", "") or "",
                    packs_opened=row.get("packs_opened", "") or "",
                    consumables_used=row.get("consumables_used", "") or "",
                    blinds_played=int(row.get("blinds_played", 0) or 0),
                    blinds_skipped=int(row.get("blinds_skipped", 0) or 0),
                    shop_rerolls=int(row.get("shop_rerolls", 0) or 0),
                    total_reward=float(row.get("total_reward", 0.0) or 0.0),
                )
            )
    return records


def compute_stats(records: list[RunRecord]) -> dict[str, Any]:
    """Calculate aggregate statistics from a list of run records."""
    total_runs = len(records)
    if total_runs == 0:
        return {
            "total_runs": 0,
            "wins": 0,
            "winrate_pct": 0.0,
            "avg_max_ante": 0.0,
            "avg_max_money": 0.0,
            "avg_duration_sec": 0.0,
            "avg_exact_prediction_pct": 0.0,
            "total_hands_played": 0,
            "top_hands": [],
            "top_fatal_bosses": [],
        }

    wins = sum(1 for r in records if r.win)
    winrate_pct = (wins / total_runs) * 100.0

    avg_max_ante = sum(r.max_ante for r in records) / total_runs
    avg_max_money = sum(r.max_money for r in records) / total_runs
    avg_duration_sec = sum(r.duration_sec for r in records) / total_runs
    avg_exact_prediction_pct = sum(r.exact_prediction_pct for r in records) / total_runs
    total_hands = sum(r.total_hands_played for r in records)

    # Top played hands
    hand_counter = Counter(r.most_played_hand for r in records if r.most_played_hand and r.most_played_hand != "None")
    top_hands = hand_counter.most_common(3)

    # Top fatal bosses (excluding NONE)
    boss_counter = Counter(
        r.fatal_boss for r in records
        if r.fatal_boss and r.fatal_boss != "NONE" and not r.fatal_boss.startswith("Endless")
    )
    top_fatal_bosses = boss_counter.most_common(3)

    return {
        "total_runs": total_runs,
        "wins": wins,
        "winrate_pct": winrate_pct,
        "avg_max_ante": round(avg_max_ante, 2),
        "avg_max_money": round(avg_max_money, 1),
        "avg_duration_sec": round(avg_duration_sec, 1),
        "avg_exact_prediction_pct": round(avg_exact_prediction_pct, 1),
        "total_hands_played": total_hands,
        "top_hands": top_hands,
        "top_fatal_bosses": top_fatal_bosses,
    }


def print_summary(
    records: list[RunRecord] | None = None,
    csv_path: Path | str | None = None,
) -> None:
    """Print a formatted console report of run statistics."""
    if records is None:
        records = load_run_history(csv_path)

    stats = compute_stats(records)

    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║             BALATRO BOT – TELEMETRY & RUN SUMMARY              ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    if stats["total_runs"] == 0:
        print("  No runs recorded yet in runs_history.csv.")
        return

    win_color = "✓" if stats["winrate_pct"] >= 50.0 else "⚠"
    print(f"  {win_color} Total Runs       : {stats['total_runs']}")
    print(f"    Wins             : {stats['wins']} / {stats['total_runs']} ({stats['winrate_pct']:.1f}%)")
    print(f"    Avg Max Ante     : {stats['avg_max_ante']:.2f}")
    print(f"    Avg Max Bank     : ${stats['avg_max_money']:.1f}")
    print(f"    Avg Pred Accuracy: {stats['avg_exact_prediction_pct']:.1f}%")
    print(f"    Avg Duration     : {stats['avg_duration_sec']:.1f}s")

    if stats["top_hands"]:
        hands_str = ", ".join([f"{h} ({count})" for h, count in stats["top_hands"]])
        print(f"    Top 3 Hands      : {hands_str}")
    else:
        print("    Top 3 Hands      : N/A")

    if stats["top_fatal_bosses"]:
        bosses_str = ", ".join([f"{b} ({count})" for b, count in stats["top_fatal_bosses"]])
        print(f"    Top Fatal Bosses : {bosses_str}")
    else:
        print("    Top Fatal Bosses : None (100% Winrate or no defeats)")
    print("──────────────────────────────────────────────────────────────────\n")
