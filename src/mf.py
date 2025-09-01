"""
src/mf.py

Implements a simple matrix factorization model with user/item embeddings.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Tuple


class RatingsDataset(Dataset):
    """Stores user–item–rating triples."""

    def __init__(self, user_ids, item_ids, ratings):
        self.u = torch.tensor(user_ids, dtype=torch.long)
        self.i = torch.tensor(item_ids, dtype=torch.long)
        self.r = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.r)

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.r[idx]


class MF(nn.Module):
    """Basic Matrix Factorization with user/item biases."""

    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.user_f = nn.Embedding(n_users, dim)
        self.item_f = nn.Embedding(n_items, dim)
        self.user_b = nn.Embedding(n_users, 1)
        self.item_b = nn.Embedding(n_items, 1)

        # Initialize
        nn.init.normal_(self.user_f.weight, std=0.01)
        nn.init.normal_(self.item_f.weight, std=0.01)
        nn.init.zeros_(self.user_b.weight)
        nn.init.zeros_(self.item_b.weight)

    def forward(self, u, i):
        dot = (self.user_f(u) * self.item_f(i)).sum(dim=1)
        bias = self.user_b(u).squeeze(1) + self.item_b(i).squeeze(1)
        return dot + bias


def build_id_maps(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Build mappings from raw userId/movieId → indices.

    Returns:
        u2idx, i2idx dicts
    """
    unique_users = np.sort(ratings_df["userId"].unique())
    unique_items = np.sort(movies_df["movieId"].unique())
    u2idx = {u: k for k, u in enumerate(unique_users)}
    i2idx = {m: k for k, m in enumerate(unique_items)}
    return u2idx, i2idx


def train_mf(
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    dim: int = 64,
    epochs: int = 5,
    bs: int = 4096,
    lr: float = 1e-2,
    device: str = "cpu",
):
    """
    Train MF model with MSE loss on ratings.

    Returns:
        trained model, u2idx, i2idx
    """
    u2idx, i2idx = build_id_maps(train_df, movies_df)

    train_u = train_df["userId"].map(u2idx).values
    train_i = train_df["movieId"].map(i2idx).values
    train_r = train_df["rating"].values.astype(np.float32)

    ds = RatingsDataset(train_u, train_i, train_r)
    dl = DataLoader(ds, batch_size=bs, shuffle=True)

    n_users, n_items = len(u2idx), len(i2idx)
    model = MF(n_users, n_items, dim).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for u, i, r in dl:
            u, i, r = u.to(device), i.to(device), r.to(device)
            pred = model(u, i)
            loss = loss_fn(pred, r)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(r)
        print(f"Epoch {ep+1}/{epochs} | MSE: {total_loss/len(ds):.4f}")

    return model, u2idx, i2idx