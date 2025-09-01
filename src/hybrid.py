"""
src/hybrid.py

Hybrid recommender: combines CF scores and content similarity.
"""

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple


class HybridRecommender:
    def __init__(self, mf_model, u2idx, i2idx, movie_embeddings: np.ndarray, alpha: float = 0.6, device: str = "cpu"):
        """
        Args:
            mf_model: trained MF model
            u2idx, i2idx: id maps
            movie_embeddings: np.ndarray [n_items, d], normalized
            alpha: weight for CF (vs 1-alpha for content)
            device: "cpu" or "cuda"
        """
        self.mf = mf_model.eval()
        self.u2idx = u2idx
        self.i2idx = i2idx
        self.inv_i = {v: k for k, v in i2idx.items()}
        self.emb = movie_embeddings
        self.alpha = alpha
        self.device = device

    def recommend_for_user(self, raw_user_id: int, topk: int = 20, exclude_seen: List[int] | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recommend top-k movies for a user.

        Args:
            raw_user_id: original MovieLens userId
            topk: number of recommendations
            exclude_seen: list of movieIds to skip

        Returns:
            top_idx, scores arrays
        """
        if raw_user_id not in self.u2idx:
            return np.array([]), np.array([])

        u = torch.tensor([self.u2idx[raw_user_id]], dtype=torch.long, device=self.device)

        with torch.no_grad():
            all_items = torch.arange(len(self.i2idx), dtype=torch.long, device=self.device)
            cf_scores = (self.mf.user_f(u) * self.mf.item_f(all_items)).sum(dim=1)
            cf_scores += self.mf.user_b(u).squeeze(0) + self.mf.item_b(all_items).squeeze(1)

        cf_scores = cf_scores.cpu().numpy()

        # Content-based profile
        content_scores = np.zeros_like(cf_scores)
        if exclude_seen:
            seen_idx = [self.i2idx[m] for m in exclude_seen if m in self.i2idx]
            if seen_idx:
                profile = self.emb[seen_idx].mean(axis=0, keepdims=True)
                content_scores = cosine_similarity(profile, self.emb).ravel()

        scores = self.alpha * cf_scores + (1 - self.alpha) * content_scores

        if exclude_seen:
            for m in exclude_seen:
                if m in self.i2idx:
                    scores[self.i2idx[m]] = -1e9

        top_idx = np.argpartition(-scores, topk)[:topk]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return top_idx, scores[top_idx]