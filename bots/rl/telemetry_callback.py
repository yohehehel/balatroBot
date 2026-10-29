"""Live TensorBoard Telemetry Callback for Balatro Macro-RL Training.

Continuously tracks and logs key game performance indicators, economic behavior,
combat efficiency, and poker hand type distributions directly to TensorBoard.
"""

from __future__ import annotations

import collections
import logging
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)

TRACKED_HAND_TYPES = [
    "High Card",
    "Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
    "Five of a Kind",
    "Flush House",
    "Flush Five",
]


class RLTelemetryCallback(BaseCallback):
    """Logs live game, economic, combat, and hand-type telemetry to TensorBoard."""

    def __init__(self, rolling_window: int = 20, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.rolling_window = rolling_window
        self.episode_count = 0
        self.recent_wins = collections.deque(maxlen=rolling_window)
        self.recent_antes = collections.deque(maxlen=rolling_window)
        self.hand_type_counter: collections.Counter[str] = collections.Counter()
        self.total_hands_seen: int = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        for info in infos:
            if not isinstance(info, dict):
                continue

            # 1. Round-level Telemetry (from combat & blind skips)
            if "round_summary" in info:
                r_sum = info["round_summary"]
                self.logger.record("combat/hands_per_round", r_sum.get("hands_played", 0))
                self.logger.record("combat/discards_per_round", r_sum.get("discards_used", 0))
                self.logger.record("combat/round_win_rate", 1.0 if r_sum.get("round_won") else 0.0)
                if r_sum.get("score_obtained", 0) > 0:
                    self.logger.record("combat/score_obtained", r_sum.get("score_obtained", 0))

            # 2. Episode-level Telemetry (from completed runs)
            if "episode_summary" in info:
                ep = info["episode_summary"]
                self.episode_count += 1
                self.recent_wins.append(1.0 if ep.get("win") else 0.0)
                self.recent_antes.append(ep.get("max_ante", 1))

                win_rate = sum(self.recent_wins) / max(1, len(self.recent_wins))
                avg_ante = sum(self.recent_antes) / max(1, len(self.recent_antes))

                # Game Progression
                self.logger.record("game/max_ante", ep.get("max_ante", 1))
                self.logger.record(f"game/avg_ante_rolling{self.rolling_window}", avg_ante)
                self.logger.record(f"game/win_rate_rolling{self.rolling_window}", win_rate)
                self.logger.record("game/final_money", ep.get("final_money", 0))
                self.logger.record("game/max_money", ep.get("max_money", 0))
                self.logger.record("game/total_reward", ep.get("total_reward", 0.0))

                # Economy & Shop Actions
                self.logger.record("shop/jokers_bought", ep.get("jokers_bought", 0))
                self.logger.record("shop/consumables_bought", ep.get("consumables_bought", 0))
                self.logger.record("shop/consumables_used", ep.get("consumables_used", 0))
                self.logger.record("shop/packs_opened", ep.get("packs_opened", 0))
                self.logger.record("shop/rerolls", ep.get("rerolls", 0))
                self.logger.record("shop/blinds_skipped", ep.get("blinds_skipped", 0))

                # Combat Volume
                self.logger.record("combat/total_hands_played", ep.get("hands_played", 0))
                self.logger.record("combat/total_discards_used", ep.get("discards_used", 0))

                # Poker Hand Distribution Percentages
                bd = ep.get("hands_breakdown", {})
                if isinstance(bd, dict):
                    for htype, count in bd.items():
                        self.hand_type_counter[htype] += count
                        self.total_hands_seen += count

                if self.total_hands_seen > 0:
                    for htype in TRACKED_HAND_TYPES:
                        pct = (self.hand_type_counter[htype] / self.total_hands_seen) * 100.0
                        slug = htype.replace(" ", "_")
                        self.logger.record(f"hands/{slug}_pct", pct)

        return True

