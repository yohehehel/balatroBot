"""Hybrid Entity Self-Attention + MLP Features Extractor for Balatro Macro-RL (Silo RL-4).

Processes Balatro's structured Dict observation space:
1. Entity Encoder (held jokers + shop & pack offers) with learned embedding,
   Multi-Head Self-Attention (d_model=64, 4 heads), and masked (Mean + Max) pooling -> dim 128.
2. Context & Deck Encoder (context + deck + hand levels) with LayerNorm + SiLU MLP -> dim 128.
3. Final Fusion: LayerNorm + SiLU projection -> features_dim = 256 for Actor-Critic heads.
"""

from __future__ import annotations

import gymnasium
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from bots.rl.constants import (
    D_CTX,
    D_DECK,
    D_ENTITY,
    D_HAND,
    MAX_JOKERS,
    MAX_OFFERS,
)


class BalatroFeaturesExtractor(BaseFeaturesExtractor):
    """Custom PyTorch Feature Extractor for Balatro Macro-RL.

    Combines self-attention over jokers and store offerings with a deep MLP
    over deck composition and macroeconomic state.
    """

    def __init__(
        self,
        observation_space: gymnasium.spaces.Dict,
        features_dim: int = 256,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_item_embeddings: int = 512,
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)

        self.num_item_embeddings = num_item_embeddings
        self.d_model = embed_dim

        # ── 1. Entity Encoder (Jokers & Offers) ────────────────────────────────
        # In D_ENTITY (30): index 0 is item_id, indices 1..29 are attributes (is_present, type, role, etc.)
        id_embed_dim = embed_dim // 2  # 32
        attr_proj_dim = embed_dim - id_embed_dim  # 32
        num_attrs = D_ENTITY - 1  # 29

        self.id_embedding = nn.Embedding(
            num_embeddings=num_item_embeddings,
            embedding_dim=id_embed_dim,
            padding_idx=0,
        )
        self.attr_linear = nn.Linear(num_attrs, attr_proj_dim)

        self.ent_norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ent_norm2 = nn.LayerNorm(embed_dim)

        # Entity pooling projection: Mean (64) + Max (64) = 128
        self.entity_output_dim = embed_dim * 2  # 128

        # ── 2. Context & Deck Encoder (MLP) ───────────────────────────────────
        context_input_dim = D_CTX + D_DECK + D_HAND  # 45 + 32 + 24 = 101
        mlp_hidden_dim = 128

        self.ctx_mlp = nn.Sequential(
            nn.Linear(context_input_dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.SiLU(),
        )

        # ── 3. Final Fusion Layer ─────────────────────────────────────────────
        # Concatenation of entity representation (128) + context MLP representation (128) = 256
        combined_dim = self.entity_output_dim + mlp_hidden_dim  # 256
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.SiLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass converting Dict observation into a unified 256-dim feature tensor."""
        # 1. Extract inputs
        held_jokers = observations["held_jokers"]  # (B, 10, 30)
        shop_and_pack = observations["shop_and_pack_offers"]  # (B, 12, 30)
        context_f = observations["context_features"]  # (B, 45)
        deck_f = observations["deck_features"]  # (B, 32)
        hand_l = observations["hand_levels"]  # (B, 24)

        # Concatenate all entities along sequence dimension: (B, 22, 30)
        all_entities = torch.cat([held_jokers, shop_and_pack], dim=1)

        # Item IDs (first feature) and attributes (remaining 29 features)
        entity_ids = all_entities[:, :, 0].long().clamp(0, self.num_item_embeddings - 1)
        entity_attrs = all_entities[:, :, 1:]
        is_present = all_entities[:, :, 1:2]  # (B, 22, 1) presence flag

        # Project IDs and continuous/one-hot attributes
        id_repr = self.id_embedding(entity_ids)  # (B, 22, 32)
        attr_repr = self.attr_linear(entity_attrs)  # (B, 22, 32)
        entity_embed = torch.cat([id_repr, attr_repr], dim=-1)  # (B, 22, 64)

        entity_embed = self.ent_norm1(entity_embed)

        # Self-Attention across all jokers and shop/pack offers
        attn_out, _ = self.self_attn(entity_embed, entity_embed, entity_embed)
        entity_repr = self.ent_norm2(entity_embed + attn_out)  # Residual connection

        # Masked Pooling (Mean + Max over present entities)
        mask = is_present  # (B, 22, 1)
        mask_counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)

        # Mean pooling
        mean_pool = (entity_repr * mask).sum(dim=1) / mask_counts  # (B, 64)

        # Max pooling
        masked_entity_repr = entity_repr.masked_fill(mask == 0, -1e4)
        max_pool = masked_entity_repr.max(dim=1)[0]  # (B, 64)
        # Fallback if an entire row has zero present entities
        has_entities = (mask.sum(dim=1) > 0)
        max_pool = torch.where(has_entities, max_pool, torch.zeros_like(max_pool))

        entity_vector = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 128)

        # 2. Context & Deck MLP
        context_input = torch.cat([context_f, deck_f, hand_l], dim=-1)  # (B, 101)
        context_vector = self.ctx_mlp(context_input)  # (B, 128)

        # 3. Final Fusion
        fused = torch.cat([entity_vector, context_vector], dim=-1)  # (B, 256)
        output = self.fusion(fused)  # (B, 256)

        return output

