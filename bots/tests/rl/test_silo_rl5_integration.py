"""Unit & Integration Tests for Silo RL-5 (Inference Wrapper, Multi-Env & CLI Integration)."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import numpy as np
import pytest
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from bots.cli import _build_parser
from bots.rl.actions import compute_action_mask
from bots.rl.env import BalatroMacroEnv
from bots.rl.eval import (
    evaluate_rl_agent,
    load_model,
    resolve_model_path,
    run_rl_episode,
)
from bots.rl.train import create_model, mask_fn, train
from bots.telemetry import load_run_history
from bots.tests.rl.test_silo_rl3_env_reward import MockBalatroStateMachine


@pytest.fixture
def mock_client() -> MockBalatroStateMachine:
    return MockBalatroStateMachine(initial_state="BLIND_SELECT")


@pytest.fixture
def mock_vec_env() -> DummyVecEnv:
    def _make():
        client = MockBalatroStateMachine(initial_state="BLIND_SELECT")
        env = BalatroMacroEnv(client=client, max_steps=20)
        return ActionMasker(env, mask_fn)
    return DummyVecEnv([_make, _make])


@pytest.fixture
def saved_test_model(tmp_path: Path, mock_vec_env: DummyVecEnv) -> Path:
    """Create and save a minimal MaskablePPO model for testing inference and evaluation."""
    model = create_model(
        env=mock_vec_env,
        n_steps=64,
        batch_size=64,
        n_epochs=1,
    )
    model_path = tmp_path / "test_model.zip"
    model.save(str(model_path))
    return model_path


# ── 1. CLI Argument Parsing ──────────────────────────────────────────────────

def test_cli_argument_parsing_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.agent == "heuristic"
    assert args.model_path == "bots/rl/checkpoints/best_model.zip"
    assert args.runs == 1
    assert args.port == 12346


def test_cli_argument_parsing_rl_custom() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "--agent", "rl",
        "--model-path", "my_models/custom.zip",
        "--runs", "3",
        "--fast",
        "--endless",
    ])
    assert args.agent == "rl"
    assert args.model_path == "my_models/custom.zip"
    assert args.runs == 3
    assert args.fast is True
    assert args.endless is True


# ── 2. Model Loading & Inference ─────────────────────────────────────────────

def test_resolve_model_path_nonexistent_raises() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_model_path("nonexistent_path_to_model_xyz.zip")


def test_model_loading_and_masked_predict(saved_test_model: Path, mock_client: MockBalatroStateMachine) -> None:
    model = load_model(saved_test_model)
    env = BalatroMacroEnv(client=mock_client)
    obs, info = env.reset()

    mask = env.action_masks()
    assert mask.shape == (26,)
    assert mask.any()

    action, _states = model.predict(obs, action_masks=mask, deterministic=True)
    action_int = int(action)

    assert 0 <= action_int < 26
    assert mask[action_int] is True or mask[action_int] == 1


# ── 3. Episode Execution & Telemetry Integration ─────────────────────────────

def test_run_rl_episode_executes_and_logs_csv(
    saved_test_model: Path,
    mock_client: MockBalatroStateMachine,
    tmp_path: Path,
) -> None:
    model = load_model(saved_test_model)
    csv_file = tmp_path / "test_rl_runs.csv"

    summary = run_rl_episode(
        client=mock_client,
        model=model,
        max_steps=10,
        delay=0.0,
        seed="TEST123",
        csv_path=csv_file,
    )

    # Check returned summary structure
    assert "win" in summary
    assert "max_ante" in summary
    assert "total_reward" in summary
    assert "steps" in summary
    assert summary["steps"] > 0
    assert "duration_sec" in summary
    assert "fatal_boss" in summary
    assert "final_jokers" in summary

    # Verify telemetry CSV output
    assert csv_file.exists()
    records = load_run_history(csv_file)
    assert len(records) == 1
    assert records[0].seed == "TEST123"
    assert records[0].most_played_hand == "RL-Agent"
    assert records[0].total_hands_played == summary["steps"]


def test_evaluate_rl_agent_multi_run(
    saved_test_model: Path,
    mock_client: MockBalatroStateMachine,
    tmp_path: Path,
) -> None:
    csv_file = tmp_path / "eval_session.csv"
    results = evaluate_rl_agent(
        model_path=saved_test_model,
        client=mock_client,
        num_runs=2,
        delay=0.0,
        max_steps=15,
        csv_path=csv_file,
    )

    assert len(results) == 2
    for r in results:
        assert "win" in r
        assert "total_reward" in r
        assert r["steps"] > 0

    assert csv_file.exists()
    records = load_run_history(csv_file)
    assert len(records) == 2


# ── 4. Multi-Env Training Function ───────────────────────────────────────────

def test_train_function_with_mock_env(mock_vec_env: DummyVecEnv, tmp_path: Path) -> None:
    save_path = tmp_path / "custom_best.zip"
    ckpt_dir = tmp_path / "ckpts"

    trained_model = train(
        env=mock_vec_env,
        total_timesteps=64,
        model_save_path=save_path,
        checkpoint_dir=ckpt_dir,
        launch_pool=False,
    )

    assert trained_model is not None
    assert save_path.exists()
    assert (ckpt_dir / "latest_model.zip").exists()

