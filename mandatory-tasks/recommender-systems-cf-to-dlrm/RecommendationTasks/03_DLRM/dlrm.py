"""
DLRM (Deep Learning Recommendation Model) from scratch.
Naumov et al., Facebook AI Research, 2019. arXiv:1906.00091

Tensor shapes tracked inline so you can trace end-to-end in an interview.
"""

import torch
import torch.nn as nn
from typing import List


def _make_mlp(layer_sizes: List[int], dropout: float = 0.0) -> nn.Sequential:
    """Build a stack of Linear → ReLU (→ optional Dropout) blocks."""
    layers = []
    for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[:-1]):
        pass  # placeholder — real loop below
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:          # no activation after the last linear
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class DLRM(nn.Module):
    """
    DLRM architecture from scratch.

    Args:
        num_numeric      : number of dense/numeric input features (13 for Criteo).
        cat_cardinalities: list of vocabulary sizes for each categorical field (26).
        embed_dim        : embedding dimension `d`. Bottom MLP output must also be `d`.
        bottom_mlp_sizes : hidden sizes for bottom MLP, excluding input and output.
                           Full sizes will be [num_numeric, *bottom_mlp_sizes, embed_dim].
        top_mlp_sizes    : hidden sizes for top MLP, excluding input and the final 1-d output.
                           Full sizes will be [d + n_interactions, *top_mlp_sizes, 1].
        top_dropout      : dropout rate in the top MLP (default 0.0).
    """

    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: List[int],
        embed_dim: int = 16,
        bottom_mlp_sizes: List[int] = None,
        top_mlp_sizes: List[int] = None,
        top_dropout: float = 0.0,
    ):
        super().__init__()
        if bottom_mlp_sizes is None:
            bottom_mlp_sizes = [512, 256]
        if top_mlp_sizes is None:
            top_mlp_sizes = [512, 256]

        self.embed_dim = embed_dim
        self.n_cat = len(cat_cardinalities)

        # ------------------------------------------------------------------
        # 1. Bottom MLP: projects dense features to embed_dim so they can be
        #    dot-producted with embeddings on equal footing.
        #    Shape:  [B, num_numeric]  ->  [B, embed_dim]
        # ------------------------------------------------------------------
        bottom_sizes = [num_numeric] + bottom_mlp_sizes + [embed_dim]
        self.bottom_mlp = _make_mlp(bottom_sizes)

        # ------------------------------------------------------------------
        # 2. Embedding tables: one per categorical field.
        #    Each lookup:  [B]  ->  [B, embed_dim]
        #    Initialised with uniform(-1/sqrt(d), 1/sqrt(d)) — small range
        #    that keeps initial dot products near zero, preventing gradient
        #    saturation in the interaction layer before training begins.
        # ------------------------------------------------------------------
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, embed_dim) for card in cat_cardinalities
        ])
        bound = 1.0 / (embed_dim ** 0.5)
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -bound, bound)

        # Bottom MLP weights: Kaiming uniform (default for nn.Linear) — good
        # for ReLU activations; preserves variance across layers.
        # (PyTorch already uses this by default, stated here for interview clarity.)

        # ------------------------------------------------------------------
        # 3. Interaction: pairwise dot products of all 27 vectors.
        #    Stack:      [B, 27, d]
        #    Z = X @ X^T  ->  [B, 27, 27]
        #    Lower triangle (excluding diagonal): 27*26/2 = 351 values.
        # ------------------------------------------------------------------
        n_vectors = self.n_cat + 1          # 26 embeddings + 1 bottom-MLP output
        n_interactions = n_vectors * (n_vectors - 1) // 2   # 351

        # ------------------------------------------------------------------
        # 4. Top MLP: takes [dense | interactions] and outputs a single logit.
        #    Input size: embed_dim + n_interactions = d + 351
        # ------------------------------------------------------------------
        top_input_dim = embed_dim + n_interactions
        top_sizes = [top_input_dim] + top_mlp_sizes + [1]
        self.top_mlp = _make_mlp(top_sizes, dropout=top_dropout)

        # Store for shape checks
        self.n_interactions = n_interactions

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_num : [B, 13]  float32 standardised numeric features.
            x_cat : [B, 26]  int64  categorical feature indices.

        Returns:
            logits: [B]  raw (pre-sigmoid) prediction scores.
        """
        B = x_num.size(0)

        # --- Bottom MLP ---
        dense_out = self.bottom_mlp(x_num)          # [B, d]

        # --- Embedding lookups ---
        # Each emb(x_cat[:, j]) -> [B, d]; collect into list
        emb_outs = [emb(x_cat[:, j]) for j, emb in enumerate(self.embeddings)]
        # emb_outs: list of 26 tensors, each [B, d]

        # --- Interaction ---
        # Stack dense output and all embeddings into [B, 27, d]
        all_vectors = torch.stack([dense_out] + emb_outs, dim=1)  # [B, 27, d]

        # Pairwise dot products: Z[b,i,j] = dot(all_vectors[b,i], all_vectors[b,j])
        # Z = all_vectors @ all_vectors^T  ->  [B, 27, 27]
        Z = torch.bmm(all_vectors, all_vectors.transpose(1, 2))   # [B, 27, 27]

        # Keep strictly lower triangle (i > j), row by row
        # tril_indices gives the indices where row > col
        rows, cols = torch.tril_indices(
            Z.size(1), Z.size(2), offset=-1, device=Z.device
        )
        interactions = Z[:, rows, cols]   # [B, 351]

        # --- Top MLP ---
        # Concatenate dense residual and interactions
        top_input = torch.cat([dense_out, interactions], dim=1)  # [B, d + 351]
        logit = self.top_mlp(top_input)                          # [B, 1]

        return logit.squeeze(1)                                  # [B]


# ------------------------------------------------------------------
# Unit shape test — run this file directly to verify tensor shapes.
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

    from common.utils import set_seed, count_params
    set_seed(42)

    NUM_NUMERIC = 13
    CAT_CARDS   = [10] * 26          # toy cardinalities for the shape test
    D           = 16
    B           = 4                  # small batch

    model = DLRM(
        num_numeric=NUM_NUMERIC,
        cat_cardinalities=CAT_CARDS,
        embed_dim=D,
        bottom_mlp_sizes=[512, 256],
        top_mlp_sizes=[512, 256],
    )

    x_num = torch.randn(B, NUM_NUMERIC)
    x_cat = torch.randint(0, 10, (B, 26))

    logits = model(x_num, x_cat)

    print("=" * 50)
    print(f"  embed_dim        : {D}")
    print(f"  n_cat            : {model.n_cat}   (26 fields)")
    print(f"  n_interactions   : {model.n_interactions}  (27*26/2)")
    print(f"  top MLP input dim: {D} + {model.n_interactions} = {D + model.n_interactions}")
    print(f"  x_num shape      : {list(x_num.shape)}")
    print(f"  x_cat shape      : {list(x_cat.shape)}")
    print(f"  logits shape     : {list(logits.shape)}  (expected [{B}])")
    print(f"  total params     : {count_params(model):,}")
    print("=" * 50)
    assert logits.shape == (B,), f"Shape mismatch: {logits.shape}"
    print("  Shape test PASSED.")
