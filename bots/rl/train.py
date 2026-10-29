"""Multi-Instance Training Pipeline for Balatro Macro-RL using MaskablePPO (Silo RL-5).

Supports:
- Parallel multi-instance training on 6 headless Balatro workers (ports 12346..12351).
- Automatic lifecycle orchestration of the headless cluster via InstancePool.
- ActionMasker integration with DummyVecEnv / SubprocVecEnv.
- Custom BalatroFeaturesExtractor (Self-Attention + MLP).
- Checkpointing, TensorBoard logging, and offline smoke testing.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from bots.client_wrapper import BalatroClient, ClientConfig
from bots.instance_pool import InstancePool
from bots.rl.env import BalatroMacroEnv
from bots.rl.network import BalatroFeaturesExtractor
from bots.rl.progress import ProgressDisplay, TrainingProgressCallback
from bots.rl.telemetry_callback import RLTelemetryCallback

logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"
LOGS_DIR = Path(__file__).resolve().parent / "logs"
DEFAULT_BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.zip"
DEFAULT_LATEST_MODEL_PATH = CHECKPOINTS_DIR / "latest_model.zip"


def mask_fn(env: Any) -> np.ndarray:
    """Action masker function returning boolean validity array (26,)."""
    target = env.unwrapped if hasattr(env, "unwrapped") else env
    return target.action_masks()


def make_env_thunk(
    port: int,
    instance_id: int | None = None,
    delay: float = 0.0,
    seed: str | None = None,
) -> Callable[[], gym.Env]:
    """Create a thunk producing a wrapped BalatroMacroEnv instance."""
    def _thunk() -> gym.Env:
        client_cfg = ClientConfig(base_url=f"http://127.0.0.1:{port}", delay=delay, timeout=30.0)
        client = BalatroClient(config=client_cfg)
        env = BalatroMacroEnv(client=client, instance_id=instance_id, delay=delay, seed=seed)
        monitored = Monitor(env)
        return ActionMasker(monitored, mask_fn)
    return _thunk


def create_model(
    env: gym.Env | VecEnv,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.01,
    tensorboard_log: str | None = None,
    seed: int | None = 42,
) -> MaskablePPO:
    """Create a MaskablePPO model configured with the custom BalatroFeaturesExtractor."""
    policy_kwargs: dict[str, Any] = {
        "features_extractor_class": BalatroFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": {"pi": [128], "vf": [128]},
    }

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=seed,
    )
    return model


def train(
    total_timesteps: int = 100_000,
    num_envs: int = 6,
    start_port: int = 12346,
    delay: float = 0.0,
    checkpoint_dir: Path | str | None = None,
    log_dir: Path | str | None = None,
    model_save_path: Path | str | None = None,
    load_model_path: Path | str | None = None,
    save_freq: int = 2048,
    launch_pool: bool = True,
    vec_env_type: str = "dummy",
    env: gym.Env | VecEnv | None = None,
    platform: str | None = None,
    learning_rate: float | None = None,
    clip_range: float | None = None,
    ent_coef: float | None = None,
) -> MaskablePPO:
    """Execute multi-instance Macro-RL training run."""
    ckpt_path = Path(checkpoint_dir or CHECKPOINTS_DIR)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    final_save_path = Path(model_save_path) if model_save_path else DEFAULT_BEST_MODEL_PATH
    final_save_path.parent.mkdir(parents=True, exist_ok=True)

    tb_path = str(Path(log_dir or LOGS_DIR))

    pool: InstancePool | None = None
    vec_env: VecEnv | None = None

    try:
        if env is not None:
            vec_env = env if isinstance(env, VecEnv) else DummyVecEnv([lambda: env])
        else:
            ports = [start_port + i for i in range(num_envs)]
            if launch_pool:
                print(f"[Cluster] Launching {num_envs} headless Balatro instance(s) on ports {ports}...")
                pool = InstancePool(
                    num_instances=num_envs,
                    start_port=start_port,
                    headless=True,
                    fast=True,
                    no_save=True,
                    audio=False,
                    platform=platform,
                )
                asyncio.run(pool.start())

            env_fns = [make_env_thunk(port=p, instance_id=i + 1, delay=delay) for i, p in enumerate(ports)]
            if vec_env_type == "subproc" and num_envs > 1:
                vec_env = SubprocVecEnv(env_fns)
            else:
                vec_env = DummyVecEnv(env_fns)

        # Scale n_steps per env so total rollout buffer is well-proportioned
        n_steps_per_env = max(64, 2048 // max(1, num_envs))
        buffer_size = n_steps_per_env * max(1, num_envs)
        n_rollouts = math.ceil(total_timesteps / buffer_size)
        effective_timesteps = n_rollouts * buffer_size

        from stable_baselines3.common.utils import get_schedule_fn

        reset_timesteps = True
        if load_model_path and Path(load_model_path).exists():
            lr = learning_rate if learning_rate is not None else 1e-4
            clip = clip_range if clip_range is not None else 0.12
            ent = ent_coef if ent_coef is not None else 0.008

            print(f"[Model] Resuming training from checkpoint: {load_model_path}")
            print(f"[Model] Fine-tuning overrides -> LR: {lr}, Clip Range: {clip}, Ent Coef: {ent}")
            custom_objects = {
                "learning_rate": lr,
                "clip_range": clip,
                "ent_coef": ent,
            }
            model = MaskablePPO.load(
                str(load_model_path),
                env=vec_env,
                custom_objects=custom_objects,
                tensorboard_log=tb_path,
            )
            model.n_steps = n_steps_per_env
            if hasattr(model, "rollout_buffer") and model.rollout_buffer is not None:
                model.rollout_buffer.buffer_size = n_steps_per_env
            model.learning_rate = lr
            model.lr_schedule = get_schedule_fn(lr)
            model.clip_range = get_schedule_fn(clip)
            model.ent_coef = ent
            if hasattr(model, "policy") and hasattr(model.policy, "optimizer") and model.policy.optimizer:
                for param_group in model.policy.optimizer.param_groups:
                    param_group["lr"] = lr
            reset_timesteps = False
        else:
            lr = learning_rate if learning_rate is not None else 3e-4
            clip = clip_range if clip_range is not None else 0.2
            ent = ent_coef if ent_coef is not None else 0.01

            model = create_model(
                env=vec_env,
                learning_rate=lr,
                n_steps=n_steps_per_env,
                batch_size=64,
                ent_coef=ent,
                tensorboard_log=tb_path,
            )
            model.clip_range = get_schedule_fn(clip)

        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(ckpt_path),
            name_prefix="macro_ppo_model",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )

        print(f"\n=======================================================")
        print(f"  STARTING MASKABLE PPO TRAINING")
        print(f"  Environments : {num_envs} worker(s)")
        print(f"  Target Steps : {total_timesteps:,} (Completes {n_rollouts} rollouts = {effective_timesteps:,} steps)")
        print(f"  Rollout Size : {buffer_size:,} steps ({n_steps_per_env} steps/env)")
        if not reset_timesteps:
            print(f"  Base Steps   : {model.num_timesteps:,} (Continuing training)")
        print(f"  Target Save  : {final_save_path}")
        print(f"=======================================================\n")

        start_timesteps = model.num_timesteps if not reset_timesteps else 0
        target_total_timesteps = start_timesteps + effective_timesteps

        progress_display = ProgressDisplay(
            total_timesteps=target_total_timesteps,
            initial_timesteps=start_timesteps,
        )
        progress_callback = TrainingProgressCallback(display=progress_display)
        telemetry_callback = RLTelemetryCallback(rolling_window=20)

        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, progress_callback, telemetry_callback],
            reset_num_timesteps=reset_timesteps,
        )
        progress_display.close()

        # Save model to target save path and latest_model.zip
        model.save(str(final_save_path))
        latest_path = ckpt_path / "latest_model.zip"
        model.save(str(latest_path))
        print(f"\n[Training Complete] Model saved to {final_save_path} and {latest_path}")
        return model

    finally:
        if "progress_display" in locals() and progress_display is not None:
            try:
                progress_display.close()
            except Exception:
                pass

        if vec_env is not None and env is None:
            try:
                vec_env.close()
            except Exception:
                pass

        if pool is not None:
            print("[Cluster] Stopping headless Balatro instance pool...")
            try:
                asyncio.run(pool.stop())
            except Exception as e:
                logger.warning(f"Error while stopping instance pool: {e}")


def run_smoke_test(total_timesteps: int = 128) -> MaskablePPO:
    """Execute offline smoke test on a mock environment verifying gradient flow and stability."""
    from bots.tests.rl.test_silo_rl3_env_reward import MockBalatroStateMachine

    def make_mock():
        m_client = MockBalatroStateMachine(initial_state="BLIND_SELECT")
        m_env = BalatroMacroEnv(client=m_client, max_steps=20)
        return ActionMasker(Monitor(m_env), mask_fn)

    vec_env = DummyVecEnv([make_mock, make_mock])

    model = create_model(
        env=vec_env,
        n_steps=64,
        batch_size=64,
        n_epochs=2,
    )

    print(f"Running smoke test on mock multi-env for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    print("Smoke test passed successfully! Multi-env MaskablePPO verified.")
    return model


def main() -> None:
    """CLI entry-point for train.py."""
    parser = argparse.ArgumentParser(description="Train Balatro Macro-RL Agent with Multi-Instance Cluster")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps (default: 100_000)")
    parser.add_argument("--num-envs", type=int, default=6, help="Number of parallel headless Balatro instances (default: 6)")
    parser.add_argument("--start-port", type=int, default=12346, help="Starting port for Balatro cluster (default: 12346)")
    parser.add_argument("--model-save-path", type=str, default=None, help="Custom path to save final model (.zip)")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save periodic checkpoints")
    parser.add_argument("--no-launch-pool", action="store_true", help="Do not launch InstancePool (assume instances already running)")
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy", help="Vectorized environment type")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint (.zip) to resume training and fine-tune from")
    parser.add_argument("--load-model", type=str, default=None, help="Path to existing model checkpoint (.zip) to resume training from")
    parser.add_argument("--resume", action="store_true", help="Automatically resume training from latest_model.zip if available")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate override (default: 3e-4 scratch, 1e-4 fine-tuning)")
    parser.add_argument("--clip-range", type=float, default=None, help="PPO clip range override (default: 0.2 scratch, 0.12 fine-tuning)")
    parser.add_argument("--ent-coef", type=float, default=None, help="Entropy coefficient override (default: 0.01 scratch, 0.008 fine-tuning)")
    parser.add_argument("--smoke-test", action="store_true", help="Run offline smoke test on mock environment")
    parser.add_argument(
        "--platform",
        choices=["windows", "linux", "darwin", "native"],
        default=None,
        help="Platform launcher (default: auto-detected, use 'native' for Linux Love2D)",
    )

    args = parser.parse_args()

    load_path = args.resume_from or args.load_model
    if args.resume and not load_path:
        if DEFAULT_LATEST_MODEL_PATH.exists():
            load_path = str(DEFAULT_LATEST_MODEL_PATH)
            print(f"[CLI] --resume flag detected: auto-loading {load_path}")
        else:
            print(f"[CLI] --resume specified but {DEFAULT_LATEST_MODEL_PATH} does not exist yet. Starting fresh.")

    if args.smoke_test:
        run_smoke_test(total_timesteps=min(128, args.timesteps))
    else:
        train(
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            start_port=args.start_port,
            model_save_path=args.model_save_path,
            checkpoint_dir=args.checkpoint_dir,
            load_model_path=load_path,
            launch_pool=not args.no_launch_pool,
            vec_env_type=args.vec_env,
            platform=args.platform,
            learning_rate=args.learning_rate,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
        )


if __name__ == "__main__":
    main()
