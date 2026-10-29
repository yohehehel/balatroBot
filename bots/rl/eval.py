"""Evaluation and Deployment Wrapper for Balatro Macro-RL Agent (Silo RL-5).

Handles:
- Loading trained MaskablePPO model checkpoints.
- Executing single or multi-run inference sessions (run_rl_episode, evaluate_rl_agent).
- Integrating with telemetry.py (Silo 6) to log run records to CSV.
- Displaying performance reports (winrate, max ante, average reward).
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import logging
from pathlib import Path
import sys
import time
from typing import Any

from sb3_contrib import MaskablePPO

from bots.client_wrapper import BalatroClient, ClientConfig
from bots.instance_pool import InstancePool
from bots.rl.env import BalatroMacroEnv
from bots.telemetry import (
    DEFAULT_CSV_PATH,
    RunRecord,
    print_summary,
    record_run,
)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "best_model.zip"
FALLBACK_MODEL_PATH = DEFAULT_MODEL_DIR / "latest_model.zip"
DEFAULT_RL_CSV_PATH = Path(__file__).resolve().parent / "rl_runs_history.csv"


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve model path to an existing checkpoint file, or raise FileNotFoundError."""
    if model_path:
        path = Path(model_path)
        if path.is_file():
            return path
        raise FileNotFoundError(f"Specified RL model checkpoint not found: {path}")

    if DEFAULT_MODEL_PATH.is_file():
        return DEFAULT_MODEL_PATH

    if FALLBACK_MODEL_PATH.is_file():
        return FALLBACK_MODEL_PATH

    # Search for any .zip checkpoint in the checkpoints directory
    if DEFAULT_MODEL_DIR.is_dir():
        checkpoints = sorted(DEFAULT_MODEL_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if checkpoints:
            return checkpoints[0]

    raise FileNotFoundError(
        f"No trained RL checkpoint found in {DEFAULT_MODEL_DIR}.\n"
        "Train a model first with: uv run python -m bots.rl.train\n"
        "Or specify an explicit checkpoint path with --model-path <path>."
    )


def load_model(
    model_path: str | Path | None = None,
    env: Any | None = None,
    device: str = "auto",
) -> MaskablePPO:
    """Load a trained MaskablePPO model from disk."""
    resolved_path = resolve_model_path(model_path)
    logger.info(f"Loading MaskablePPO model from {resolved_path}")
    model = MaskablePPO.load(str(resolved_path), env=env, device=device)
    return model


def run_rl_episode(
    client: BalatroClient,
    model: MaskablePPO,
    max_steps: int = 1000,
    endless: bool = False,
    delay: float = 0.0,
    seed: str | None = None,
    deck: str = "RED",
    stake: str = "WHITE",
    csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute a single complete game run driven by the trained RL agent."""
    env = BalatroMacroEnv(
        client=client,
        instance_id=1,
        deck=deck,
        stake=stake,
        seed=seed,
        max_steps=max_steps,
        delay=delay,
        endless=endless,
    )

    obs, info = env.reset(seed=int(seed) if seed and seed.isdigit() else None)
    total_reward = 0.0
    steps = 0
    max_ante = env.state.ante_num
    max_money = env.state.money
    start_time = time.time()

    terminated = False
    truncated = False

    while not (terminated or truncated):
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1
        max_ante = max(max_ante, env.state.ante_num)
        max_money = max(max_money, env.state.money)

    duration_sec = round(time.time() - start_time, 2)
    win = bool(info.get("is_success", False) or env.state.won or max_ante > 8)

    fatal_boss = "NONE"
    if not win:
        if env.state.active_blind and (env.state.active_blind.type or "").lower() == "boss":
            fatal_boss = env.state.active_blind.name or "Unknown Boss"
        else:
            fatal_boss = f"Ante {max_ante} Combat"

    final_jokers = ", ".join([c.label or c.key for c in env.state.jokers.cards]) if (env.state.jokers and env.state.jokers.cards) else "None"

    # Log record to CSV
    target_csv = csv_path or DEFAULT_CSV_PATH
    record = RunRecord(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        seed=seed or "RANDOM",
        deck=deck,
        stake=stake,
        win=win,
        max_ante=max_ante,
        final_round=env.state.round_num,
        fatal_boss=fatal_boss,
        max_money=max_money,
        most_played_hand="RL-Agent",
        final_jokers=final_jokers,
        duration_sec=duration_sec,
        exact_prediction_pct=0.0,
        total_hands_played=steps,
        total_discards_used=0,
        hands_breakdown=f"steps:{steps};reward:{round(total_reward, 2)}",
    )
    record_run(record, csv_path=target_csv)

    result = {
        "win": win,
        "max_ante": max_ante,
        "final_round": env.state.round_num,
        "max_money": max_money,
        "total_reward": round(total_reward, 2),
        "steps": steps,
        "duration_sec": duration_sec,
        "fatal_boss": fatal_boss,
        "final_jokers": final_jokers,
        "is_success": win,
    }
    return result


def evaluate_rl_agent(
    model_path: str | Path | None = None,
    client: BalatroClient | None = None,
    num_runs: int = 1,
    port: int = 12346,
    delay: float = 0.0,
    seed: str | None = None,
    deck: str = "RED",
    stake: str = "WHITE",
    endless: bool = False,
    max_steps: int = 1000,
    csv_path: str | Path | None = None,
    headless: bool = True,
    platform: str | None = None,
) -> list[dict[str, Any]]:
    """Run an evaluation session of N runs using the RL agent."""
    pool: InstancePool | None = None

    if client is None:
        client_cfg = ClientConfig(base_url=f"http://127.0.0.1:{port}", delay=delay, timeout=30.0)
        client = BalatroClient(config=client_cfg)
        try:
            client.call("health")
        except Exception:
            print(f"[Instance] No active Balatro instance found on port {port}.")
            print(f"[Instance] Auto-launching 1 headless instance (headless={headless}, fast=True) on port {port}...")
            pool = InstancePool(
                num_instances=1,
                start_port=port,
                headless=headless,
                fast=True,
                no_save=True,
                audio=False,
                platform=platform,
            )
            asyncio.run(pool.start())

    try:
        model = load_model(model_path)
        print(f"\n=======================================================")
        print(f"  STARTING RL AGENT EVALUATION ({num_runs} RUN(S))")
        print(f"  Model: {model_path or 'default checkpoint'}")
        print(f"  Port: {port} | Delay: {delay}s | Endless: {endless}")
        print(f"=======================================================\n")

        results: list[dict[str, Any]] = []

        for run_idx in range(1, num_runs + 1):
            print(f"-- Run {run_idx}/{num_runs} Starting (Deck: {deck}, Stake: {stake}, Seed: {seed or 'RANDOM'}) --")
            summary = run_rl_episode(
                client=client,
                model=model,
                max_steps=max_steps,
                endless=endless,
                delay=delay,
                seed=seed,
                deck=deck,
                stake=stake,
                csv_path=csv_path,
            )
            results.append(summary)

            status_str = "VICTORY" if summary["win"] else "DEFEAT"
            print(
                f"-- Run {run_idx}/{num_runs} Finished: {status_str} | "
                f"Ante {summary['max_ante']} | Round {summary['final_round']} | "
                f"Money: ${summary['max_money']} | Reward: {summary['total_reward']} | "
                f"Steps: {summary['steps']} | Duration: {summary['duration_sec']}s --\n"
            )

        # Print summary table
        wins = sum(1 for r in results if r["win"])
        winrate = (wins / len(results)) * 100.0 if results else 0.0
        avg_ante = sum(r["max_ante"] for r in results) / len(results) if results else 0.0
        avg_reward = sum(r["total_reward"] for r in results) / len(results) if results else 0.0

        print("=======================================================")
        print(f"  RL EVALUATION COMPLETED ({len(results)} RUNS)")
        print(f"  Wins: {wins}/{len(results)} ({winrate:.1f}%)")
        print(f"  Average Max Ante: {avg_ante:.2f}")
        print(f"  Average Reward  : {avg_reward:.2f}")
        print("=======================================================\n")

        return results
    finally:
        if pool is not None:
            print("[Instance] Stopping evaluation instance...")
            asyncio.run(pool.stop())


def main() -> None:
    """CLI entry-point for evaluate_rl_agent."""
    parser = argparse.ArgumentParser(description="Evaluate Balatro Macro-RL Agent")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model .zip")
    parser.add_argument("--port", type=int, default=12346, help="Port of running Balatro instance")
    parser.add_argument(
        "--runs",
        "--num-episodes",
        dest="runs",
        type=int,
        default=1,
        help="Number of runs/episodes to execute (default: 1)",
    )
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between actions")
    parser.add_argument("--fast", action="store_true", help="Set delay to 0.0")
    parser.add_argument("--gui", action="store_true", help="Run Balatro in GUI mode if auto-launching an instance")
    parser.add_argument("--endless", action="store_true", help="Continue playing after Ante 8")
    parser.add_argument("--seed", type=str, default=None, help="Seed for runs")
    parser.add_argument("--csv-path", type=str, default=None, help="Path to save telemetry CSV")
    parser.add_argument(
        "--platform",
        choices=["windows", "linux", "darwin", "native"],
        default=None,
        help="Platform launcher (default: auto-detected, use 'native' for Linux Love2D)",
    )

    args = parser.parse_args()
    delay = 0.0 if args.fast else args.delay

    evaluate_rl_agent(
        model_path=args.model_path,
        num_runs=args.runs,
        port=args.port,
        delay=delay,
        seed=args.seed,
        endless=args.endless,
        csv_path=args.csv_path,
        headless=not args.gui,
        platform=args.platform,
    )


if __name__ == "__main__":
    main()

