"""Tests for unique random seed generation and seed propagation across runs."""

import pytest
from bots import BALATRO_SEED_CHARS, generate_random_seed, start_run
from bots.rl.env import BalatroMacroEnv


class DummyClient:
    def __init__(self):
        self.calls: list[tuple[str, dict | None]] = []

    def call(self, method: str, params: dict | None = None) -> dict:
        self.calls.append((method, params))
        if method == "menu":
            return {"state": "MENU"}
        elif method == "start":
            seed_used = (params or {}).get("seed", "")
            return {
                "state": "BLIND_SELECT",
                "seed": seed_used,
                "deck": (params or {}).get("deck", "RED"),
                "stake": (params or {}).get("stake", "WHITE"),
                "ante": 1,
                "round": 1,
                "money": 4,
            }
        elif method == "gamestate":
            return {
                "state": "BLIND_SELECT",
                "ante": 1,
                "round": 1,
                "money": 4,
            }
        return {"state": "UNKNOWN"}


def test_generate_random_seed_format():
    """Verify that generate_random_seed creates valid 8-character uppercase seeds."""
    for _ in range(50):
        seed = generate_random_seed()
        assert len(seed) == 8
        assert all(c in BALATRO_SEED_CHARS for c in seed)
        # Verify 0 and O are excluded
        assert "0" not in seed
        assert "O" not in seed


def test_generate_random_seed_uniqueness():
    """Verify that generating 1,000 seeds yields 1,000 unique seeds (no duplicates)."""
    seeds = {generate_random_seed() for _ in range(1000)}
    assert len(seeds) == 1000


def test_start_run_unseeded_generates_unique_seeds():
    """Verify that start_run() automatically generates a fresh unique seed if omitted."""
    client = DummyClient()

    state1 = start_run(client)
    assert state1.seed != ""
    assert len(state1.seed) == 8

    state2 = start_run(client)
    assert state2.seed != ""
    assert len(state2.seed) == 8

    # The two runs must have different seeds
    assert state1.seed != state2.seed

    # Verify calls sent to client
    start_calls = [c for c in client.calls if c[0] == "start"]
    assert len(start_calls) == 2
    assert "seed" in start_calls[0][1]
    assert "seed" in start_calls[1][1]
    assert start_calls[0][1]["seed"] != start_calls[1][1]["seed"]


def test_start_run_explicit_seed_preserved():
    """Verify that passing an explicit seed is respected."""
    client = DummyClient()

    state = start_run(client, seed="MYSEED12")
    assert state.seed == "MYSEED12"

    start_calls = [c for c in client.calls if c[0] == "start"]
    assert len(start_calls) == 1
    assert start_calls[0][1]["seed"] == "MYSEED12"


def test_macro_env_generates_unique_seeds_on_reset():
    """Verify BalatroMacroEnv generates unique seeds across multiple resets when unseeded."""
    client = DummyClient()
    env = BalatroMacroEnv(client=client, seed=None)

    env.reset()
    seed1 = env.current_run_seed
    assert seed1 is not None
    assert len(seed1) == 8

    env.reset()
    seed2 = env.current_run_seed
    assert seed2 is not None
    assert len(seed2) == 8

    assert seed1 != seed2


def test_macro_env_preserves_explicit_fixed_seed():
    """Verify BalatroMacroEnv preserves a fixed seed when configured with one."""
    client = DummyClient()
    env = BalatroMacroEnv(client=client, seed="FIXED123")

    env.reset()
    assert env.current_run_seed == "FIXED123"

    env.reset()
    assert env.current_run_seed == "FIXED123"

