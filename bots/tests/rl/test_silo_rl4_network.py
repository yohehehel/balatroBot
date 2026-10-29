"""Unit tests for Silo RL-4: Hybrid Entity Self-Attention + MLP & MaskablePPO Pipeline.

Verifies:
1. Forward pass shape (B, 256) and zero NaNs on diverse batches.
2. Robustness to B=1 and completely empty entity slots (zeros / no division by zero).
3. Gradient propagation across self-attention, entity embeddings, context MLP, and fusion layers.
4. Compatibility with sb3_contrib.MaskablePPO:
   - model.predict(obs, action_masks=mask) respecting the mask.
   - model.learn(total_timesteps=64) training loop execution.
"""

from __future__ import annotations

import copy
import gymnasium
import numpy as np
import pytest
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO

from bots.rl.actions import ActionID, NUM_ACTIONS
from bots.rl.constants import (
    D_CTX,
    D_DECK,
    D_ENTITY,
    D_HAND,
    MAX_JOKERS,
    MAX_OFFERS,
)
from bots.rl.env import BalatroMacroEnv
from bots.rl.features import extract_features, get_observation_space
from bots.rl.network import BalatroFeaturesExtractor
from bots.rl.train import create_model
from bots.state_parser import GameScreen, ParsedGameState
from bots.tests.rl.test_silo_rl3_env_reward import MockBalatroStateMachine


def _make_dummy_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    """Create a synthetic torch observation batch matching observation_space."""
    return {
        "context_features": torch.randn(batch_size, D_CTX, dtype=torch.float32),
        "deck_features": torch.rand(batch_size, D_DECK, dtype=torch.float32),
        "hand_levels": torch.rand(batch_size, D_HAND, dtype=torch.float32),
        "held_jokers": torch.cat([
            torch.randint(0, 150, (batch_size, MAX_JOKERS, 1)).float(),
            torch.randint(0, 2, (batch_size, MAX_JOKERS, 1)).float(),
            torch.randn(batch_size, MAX_JOKERS, D_ENTITY - 2),
        ], dim=-1),
        "shop_and_pack_offers": torch.cat([
            torch.randint(0, 150, (batch_size, MAX_OFFERS, 1)).float(),
            torch.randint(0, 2, (batch_size, MAX_OFFERS, 1)).float(),
            torch.randn(batch_size, MAX_OFFERS, D_ENTITY - 2),
        ], dim=-1),
    }


# ── Test 1: Forward Pass & Dimensions ─────────────────────────────────────────

def test_features_extractor_forward_shape_and_no_nan():
    """Verify that BalatroFeaturesExtractor produces output shape (B, 256) without NaNs."""
    obs_space = get_observation_space()
    extractor = BalatroFeaturesExtractor(observation_space=obs_space, features_dim=256)

    batch_size = 4
    batch = _make_dummy_batch(batch_size=batch_size)

    output = extractor(batch)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 256)
    assert not torch.isnan(output).any(), "NaN detected in extractor output"
    assert not torch.isinf(output).any(), "Inf detected in extractor output"


def test_features_extractor_single_sample_and_empty_entities():
    """Verify B=1 with completely zeroed entity slots does not cause NaN or division by zero."""
    obs_space = get_observation_space()
    extractor = BalatroFeaturesExtractor(observation_space=obs_space, features_dim=256)

    batch_zero = {
        "context_features": torch.zeros(1, D_CTX, dtype=torch.float32),
        "deck_features": torch.zeros(1, D_DECK, dtype=torch.float32),
        "hand_levels": torch.zeros(1, D_HAND, dtype=torch.float32),
        "held_jokers": torch.zeros(1, MAX_JOKERS, D_ENTITY, dtype=torch.float32),
        "shop_and_pack_offers": torch.zeros(1, MAX_OFFERS, D_ENTITY, dtype=torch.float32),
    }

    output = extractor(batch_zero)
    assert output.shape == (1, 256)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


# ── Test 2: Gradient Propagation ──────────────────────────────────────────────

def test_gradient_propagation_through_all_submodules():
    """Verify backward() computes non-zero gradients for embedding, attention, MLP, and fusion."""
    obs_space = get_observation_space()
    extractor = BalatroFeaturesExtractor(observation_space=obs_space, features_dim=256)

    batch = _make_dummy_batch(batch_size=4)
    # Ensure at least some entities are present (is_present = 1.0) for attention and embedding grads
    batch["held_jokers"][:, :3, 1] = 1.0
    batch["shop_and_pack_offers"][:, :3, 1] = 1.0

    output = extractor(batch)
    loss = output.sum()
    loss.backward()

    # 1. Check ID Embedding gradients
    assert extractor.id_embedding.weight.grad is not None
    assert torch.abs(extractor.id_embedding.weight.grad).sum() > 0.0

    # 2. Check Attribute Projection gradients
    assert extractor.attr_linear.weight.grad is not None
    assert torch.abs(extractor.attr_linear.weight.grad).sum() > 0.0

    # 3. Check Self-Attention gradients
    assert extractor.self_attn.in_proj_weight.grad is not None
    assert torch.abs(extractor.self_attn.in_proj_weight.grad).sum() > 0.0

    # 4. Check Context MLP gradients
    first_mlp_layer = extractor.ctx_mlp[0]
    assert isinstance(first_mlp_layer, nn.Linear)
    assert first_mlp_layer.weight.grad is not None
    assert torch.abs(first_mlp_layer.weight.grad).sum() > 0.0

    # 5. Check Fusion Layer gradients
    fusion_layer = extractor.fusion[0]
    assert isinstance(fusion_layer, nn.Linear)
    assert fusion_layer.weight.grad is not None
    assert torch.abs(fusion_layer.weight.grad).sum() > 0.0


# ── Test 3: Compatibilité MaskablePPO & Inférence ─────────────────────────────

def test_maskable_ppo_inference_respects_action_mask():
    """Verify model.predict(obs, action_masks=mask) strictly selects valid masked actions."""
    mock_client = MockBalatroStateMachine(initial_state="BLIND_SELECT")
    env = BalatroMacroEnv(client=mock_client)
    obs, info = env.reset()

    model = create_model(env=env, n_steps=64, batch_size=64)

    # In BLIND_SELECT: only PLAY_BLIND (0) and SKIP_BLIND (1) are True
    mask = env.action_masks()
    assert mask[ActionID.PLAY_BLIND]
    assert not any(mask[2:])

    for _ in range(5):
        action, _ = model.predict(obs, action_masks=mask)
        action_int = int(action)
        assert 0 <= action_int < NUM_ACTIONS
        assert mask[action_int], f"Selected action {action_int} was masked False!"


def test_maskable_ppo_learn_step():
    """Verify that model.learn(total_timesteps=64) completes without error on mock environment."""
    mock_client = MockBalatroStateMachine(initial_state="BLIND_SELECT")
    env = BalatroMacroEnv(client=mock_client, max_steps=20)

    model = create_model(env=env, n_steps=64, batch_size=64, n_epochs=2)

    # Execute 64 timesteps of training
    model.learn(total_timesteps=64)
    assert model.num_timesteps >= 64

