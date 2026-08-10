"""Factorization Machine (FM) baseline for CTR prediction.

Standard 2-way FM over categorical fields, using a single shared embedding
table per order (order-1 / order-2) indexed via per-field offsets, so an
arbitrary number of fields with different vocab sizes can be looked up with
one nn.Embedding call each:

    logit = bias + sum_i w_i(x_i)                                   # order-1
           + 0.5 * sum_k[ (sum_i v_i(x_i))^2 - sum_i v_i(x_i)^2 ]    # order-2

Avazu has no numeric features, so only categorical fields are modeled.
"""
import numpy as np
import torch
import torch.nn as nn


class FactorizationMachine(nn.Module):
    def __init__(
        self,
        vocab_sizes: dict,
        categorical_cols: list,
        embed_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        field_dims = [vocab_sizes[col] for col in categorical_cols]
        total_dim = sum(field_dims)
        offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        self.register_buffer("offsets", torch.from_numpy(offsets))

        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(total_dim, 1)
        self.embedding = nn.Embedding(total_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.linear.weight)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, categorical: torch.Tensor) -> torch.Tensor:
        """categorical: (B, num_fields) long indices, per-field local vocab. Returns (B,) raw logits."""
        x = categorical + self.offsets

        linear_term = self.linear(x).squeeze(-1).sum(dim=1) + self.bias

        emb = self.dropout(self.embedding(x))  # (B, num_fields, embed_dim)
        square_of_sum = emb.sum(dim=1) ** 2
        sum_of_square = (emb ** 2).sum(dim=1)
        interaction_term = 0.5 * (square_of_sum - sum_of_square).sum(dim=1)

        return linear_term + interaction_term
