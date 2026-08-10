"""Transformer-based CTR model (AutoInt-style feature interaction).

Each categorical field's embedding is treated as one token in a length-22
sequence (Avazu has no numeric features). Multi-head self-attention over
that sequence lets the model learn arbitrary-order feature interactions
automatically, in place of FM's hand-derived closed-form second-order term
(see src/models/fm.py):

    tokens = dropout(field_embedding(x) + field_positional_embedding)  # (B, F, d)
    tokens = TransformerEncoder(tokens)                                # (B, F, d)
    logit  = Linear(dropout(mean_pool(tokens)))                        # (B,)

The field positional embedding is necessary because self-attention is
permutation-invariant, but the fields themselves are not interchangeable
(e.g. `hour` and `device_id` carry distinct semantics/vocab).

The head is deliberately a mean-pool + single Linear rather than a
flatten + MLP: flattening gives the head a separate weight block per field
*position*, which is enough extra capacity to memorize specific field-value
combinations instead of learning generalizable interactions (observed
empirically as a much larger train/val AUC gap than FM, even at high
dropout). Mean-pooling removes that per-position weight, forcing all
interaction modeling to happen in the shared self-attention layers.
"""
import numpy as np
import torch
import torch.nn as nn


class TransformerCTR(nn.Module):
    def __init__(
        self,
        vocab_sizes: dict,
        categorical_cols: list,
        embed_dim: int = 16,
        dropout: float = 0.2,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 64,
    ):
        super().__init__()
        field_dims = [vocab_sizes[col] for col in categorical_cols]
        total_dim = sum(field_dims)
        num_fields = len(field_dims)
        offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        self.register_buffer("offsets", torch.from_numpy(offsets))

        self.embedding = nn.Embedding(total_dim, embed_dim)
        self.field_pos_embedding = nn.Parameter(torch.zeros(1, num_fields, embed_dim))
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.field_pos_embedding)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, categorical: torch.Tensor) -> torch.Tensor:
        """categorical: (B, num_fields) long indices, per-field local vocab. Returns (B,) raw logits."""
        x = categorical + self.offsets

        tokens = self.embed_dropout(self.embedding(x) + self.field_pos_embedding)  # (B, num_fields, embed_dim)
        encoded = self.encoder(tokens)  # (B, num_fields, embed_dim)

        pooled = self.head_dropout(encoded.mean(dim=1))  # (B, embed_dim)
        return self.head(pooled).squeeze(-1)
