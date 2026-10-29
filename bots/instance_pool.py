"""Headless Multi-Instance Balatro Pool Orchestrator.

Manages N concurrent Balatro instances (headless, fast, no_save) and runs
parallel bot sessions across them with unified logging and consolidated telemetry.
Supports cross-platform environments (Windows and Linux Native LÖVE).
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys
from typing import Any

from balatrobot.config import Config
from balatrobot.manager import BalatroInstance
from bots.client_wrapper import BalatroClient, ClientConfig
from bots.runner import BotRunner
from bots.telemetry import (
    DEFAULT_CSV_PATH,
    DEFAULT_ROUNDS_CSV_PATH,
    RunRecord,
    print_summary,
    record_run,
)


class InstancePool:
    """Manages a cluster of N concurrent Balatro processes."""

    def __init__(
        self,
        num_instances: int = 1,
        start_port: int = 12346,
        headless: bool = True,
        fast: bool = True,
        no_save: bool = True,
        audio: bool = False,
        platform: str | None = None,
    ) -> None:
        self.num_instances = max(1, num_instances)
        self.start_port = start_port
        self.ports = [start_port + i for i in range(self.num_instances)]
        self.headless = headless
        self.fast = fast
        self.no_save = no_save
        self.audio = audio
        self.platform = platform or os.environ.get("BALATROBOT_PLATFORM")
        self.instances: list[BalatroInstance] = []
        self._is_running = False

    async def start(self) -> None:
        """Start all Balatro instances concurrently and wait for health checks."""
        if self._is_running:
            return

        print(f"\n=======================================================")
        print(f"  LAUNCHING {self.num_instances} BALATRO INSTANCE(S) IN PARALLEL")
        print(f"  Ports: {self.ports}")
        print(f"  Headless: {self.headless} | Fast Mode: {self.fast} | No Disk Save: {self.no_save}")
        if self.platform:
            print(f"  Platform Target: {self.platform}")
        print(f"=======================================================\n")

        self.instances.clear()
        session_id = f"pool-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Clean lovely dump directory beforehand across platforms to prevent file lock races
        dump_candidates: list[Path] = []
        appdata = os.environ.get("APPDATA")
        if appdata:
            dump_candidates.append(Path(appdata) / "Balatro" / "Mods" / "lovely" / "dump")
        # Linux Native Love2D config directory
        dump_candidates.append(Path.home() / ".config" / "love" / "Mods" / "lovely" / "dump")
        # Linux Proton Steam compatdata directory
        dump_candidates.append(
            Path.home()
            / ".local"
            / "share"
            / "Steam"
            / "steamapps"
            / "compatdata"
            / "2379780"
            / "pfx"
            / "drive_c"
            / "users"
            / "steamuser"
            / "AppData"
            / "Roaming"
            / "Balatro"
            / "Mods"
            / "lovely"
            / "dump"
        )

        for lovely_dump in dump_candidates:
            if lovely_dump.exists():
                try:
                    shutil.rmtree(lovely_dump, ignore_errors=True)
                except Exception:
                    pass

        for port in self.ports:
            # Use from_kwargs to inherit system environment variables (BALATROBOT_*)
            kwargs: dict[str, Any] = {
                "port": port,
                "headless": self.headless,
                "fast": self.fast,
                "no_save": self.no_save,
                "audio": self.audio,
            }
            if self.platform:
                kwargs["platform"] = self.platform
            cfg = Config.from_kwargs(**kwargs)
            self.instances.append(BalatroInstance(config=cfg, session_id=session_id, port=port))

        async def _staggered_start(idx: int, inst: BalatroInstance) -> None:
            if idx > 0:
                await asyncio.sleep(idx * 0.6)
            await inst.start()

        try:
            await asyncio.gather(*[_staggered_start(i, inst) for i, inst in enumerate(self.instances)])
            self._is_running = True
            print(f"\n[Pool Ready] All {self.num_instances} instance(s) healthy and listening.\n")
        except Exception as exc:
            print(f"\n[Pool Error] Failed to start all instances: {exc}. Cleaning up...")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop and kill all Balatro instances cleanly."""
        if not self.instances:
            return

        print("\n[Pool Cleanup] Stopping all Balatro instances...")
        tasks = [inst.stop() for inst in self.instances]
        await asyncio.gather(*tasks, return_exceptions=True)
        self.instances.clear()
        self._is_running = False

    async def __aenter__(self) -> InstancePool:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()

    def _run_worker(
        self,
        instance_id: int,
        port: int,
        runs_per_instance: int,
        deck: str,
        stake: str,
        seed: str | None,
        endless: bool,
        delay: float,
        csv_path: Path | str | None,
        rounds_csv_path: Path | str | None,
    ) -> list[RunRecord]:
        """Synchronous worker function executed inside a thread per instance."""
        client_cfg = ClientConfig(base_url=f"http://127.0.0.1:{port}", delay=delay, timeout=30.0)
        client = BalatroClient(config=client_cfg)
        runner = BotRunner(
            client=client,
            endless=endless,
            delay=delay,
            seed=seed,
            deck=deck,
            stake=stake,
            csv_path=csv_path,
            rounds_csv_path=rounds_csv_path,
            instance_id=instance_id,
        )
        return runner.run_session(num_runs=runs_per_instance)

    async def run_parallel_sessions(
        self,
        runs_per_instance: int = 1,
        deck: str = "RED",
        stake: str = "WHITE",
        seed: str | None = None,
        endless: bool = False,
        delay: float = 0.0,
        csv_path: Path | str | None = None,
        rounds_csv_path: Path | str | None = None,
    ) -> list[RunRecord]:
        """Execute parallel bot runs across all pool instances."""
        if not self._is_running:
            raise RuntimeError("Pool is not running. Call start() first or use async context manager.")

        total_runs_targeted = self.num_instances * runs_per_instance
        print(f"── Starting {total_runs_targeted} total run(s) ({runs_per_instance} per instance across {self.num_instances} instances) ──\n")

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=self.num_instances) as executor:
            futures = [
                loop.run_in_executor(
                    executor,
                    self._run_worker,
                    idx + 1,
                    port,
                    runs_per_instance,
                    deck,
                    stake,
                    seed,
                    endless,
                    delay,
                    csv_path,
                    rounds_csv_path,
                )
                for idx, port in enumerate(self.ports)
            ]
            nested_records = await asyncio.gather(*futures)

        all_records: list[RunRecord] = []
        for records in nested_records:
            all_records.extend(records)

        print("\n=======================================================")
        print(f"  ALL PARALLEL SESSIONS COMPLETED ({len(all_records)} RUNS TOTAL)")
        print("=======================================================")
        print_summary(all_records, csv_path)
        return all_records


async def main_async(args: argparse.Namespace) -> None:
    pool = InstancePool(
        num_instances=args.instances,
        start_port=args.start_port,
        headless=not args.headed,
        fast=not args.slow,
        no_save=True,
        audio=False,
        platform=args.platform,
    )

    try:
        async with pool:
            await pool.run_parallel_sessions(
                runs_per_instance=args.runs_per_instance,
                deck=args.deck,
                stake=args.stake,
                seed=args.seed,
                endless=args.endless,
                delay=args.delay,
                csv_path=args.csv or DEFAULT_CSV_PATH,
                rounds_csv_path=args.rounds_csv or DEFAULT_ROUNDS_CSV_PATH,
            )
    except KeyboardInterrupt:
        print("\n[Interrupt] Cancellation requested by user. Terminating pool...")
    except Exception as exc:
        print(f"\n[Error] Execution failed: {exc}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple parallel headless Balatro instances with autonomous bots."
    )
    parser.add_argument(
        "--instances",
        "-n",
        type=int,
        default=1,
        help="Number of concurrent Balatro instances (default: 1)",
    )
    parser.add_argument(
        "--runs-per-instance",
        "-r",
        type=int,
        default=1,
        help="Number of consecutive runs per instance (default: 1)",
    )
    parser.add_argument(
        "--start-port",
        type=int,
        default=12346,
        help="Base port for first instance (default: 12346)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between bot actions (default: 0.0 for maximum throughput)",
    )
    parser.add_argument(
        "--deck",
        type=str,
        default="RED",
        help="Deck type (default: RED)",
    )
    parser.add_argument(
        "--stake",
        type=str,
        default="WHITE",
        help="Stake level (default: WHITE)",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Optional fixed game seed",
    )
    parser.add_argument(
        "--endless",
        action="store_true",
        default=False,
        help="Continue playing past Ante 8 until defeat",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        default=False,
        help="Disable headless mode to show game window(s)",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        default=False,
        help="Disable fast mode",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Custom path for runs history CSV",
    )
    parser.add_argument(
        "--rounds-csv",
        type=str,
        default=None,
        help="Custom path for rounds detailed history CSV",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        choices=["windows", "linux", "darwin", "native"],
        help="Launcher platform (default: auto-detected, use 'native' for Linux Love2D)",
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
