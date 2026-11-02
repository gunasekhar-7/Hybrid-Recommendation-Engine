"""Vectorized Hybrid Recommender Engine.

Fuses Collaborative Filtering (matrix factorization) with Content-Based
Semantic Embeddings (SBERT) using dynamic cold-start weight adaptation
and vectorized BLAS matrix scoring.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .cf import CollaborativeFiltering
from .data import DataProcessor

logger = logging.getLogger(__name__)


class HybridRecommender:
    """Production Hybrid Recommender fusing Collaborative and Content-Based Filtering."""

    def __init__(
        self,
        cf_method: str = "implicit_als",
        cf_n_factors: int = 64,
        cf_reg: float = 0.05,
        cf_epochs: int = 20,
        cb_alpha: float = 0.5,
        cb_beta: float = 0.5,
        adapt_weights: bool = True,
        redis_client: Optional[Any] = None,
        embedding_ttl: int = 86400,
    ) -> None:
        self.cf_method = cf_method
        self.cf_n_factors = cf_n_factors
        self.cf_reg = cf_reg
        self.cf_epochs = cf_epochs

        # Normalized weights (alpha: CF weight, beta: CB weight)
        total = max(1e-6, cb_alpha + cb_beta)
        self.cf_alpha: float = cb_alpha / total
        self.cf_beta: float = cb_beta / total
        self.adapt_weights: bool = adapt_weights

        self.redis_client = redis_client
        self.embedding_ttl = embedding_ttl

        self.data_processor: Optional[DataProcessor] = None
        self.cf_model: Optional[CollaborativeFiltering] = None
        self.is_fitted: bool = False
        self.fit_time: float = 0.0

    def fit(
        self,
        interactions_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
        sbert_batch_size: int = 64,
    ) -> "HybridRecommender":
        """Fit both DataProcessor (semantic embeddings) and CF model.

        Args:
            interactions_df: DataFrame with ['user_id', 'item_id', 'rating']
            item_features_df: DataFrame with ['item_id', 'title', optional 'description']
            sbert_batch_size: Batch size for SBERT inference
        """
        start_time = time.time()

        # 1. Process data & generate embeddings
        self.data_processor = DataProcessor(
            redis_client=self.redis_client,
            embedding_ttl=self.embedding_ttl,
        )
        self.data_processor.fit(
            interactions_df=interactions_df,
            item_features_df=item_features_df,
            batch_size=sbert_batch_size,
        )

        # 2. Extract sparse user-item interaction matrix
        matrix, user_ids, item_ids = self.data_processor.get_cf_matrix()

        # 3. Train Collaborative Filtering model
        self.cf_model = CollaborativeFiltering(
            method=self.cf_method,
            n_factors=self.cf_n_factors,
            regularization=self.cf_reg,
            epochs=self.cf_epochs,
        )
        self.cf_model.fit(
            user_item_matrix=matrix,
            user_ids=user_ids,
            item_ids=item_ids,
            user_interactions=self.data_processor.user_interactions,
        )

        # 4. Adaptive weighting based on dataset sparsity & cold-start ratio
        if self.adapt_weights:
            self._adjust_global_weights(interactions_df)

        self.fit_time = time.time() - start_time
        self.is_fitted = True

        logger.info(
            "HybridRecommender successfully fitted in %.2fs (alpha=%.2f, beta=%.2f)",
            self.fit_time,
            self.cf_alpha,
            self.cf_beta,
        )
        return self

    def _adjust_global_weights(self, interactions_df: pd.DataFrame) -> None:
        """Adjust baseline weights based on user cold-start and interaction distribution."""
        user_counts = interactions_df.groupby("user_id").size()
        cold_user_pct = float((user_counts < 3).mean())

        if cold_user_pct > 0.4:
            # Heavy cold-start: favor content-based semantic matching
            self.cf_beta = 0.70
            self.cf_alpha = 0.30
            logger.info("High cold-start ratio (%.1f%%). Shifted baseline weights to CB=%.2f, CF=%.2f",
                        cold_user_pct * 100, self.cf_beta, self.cf_alpha)
        elif cold_user_pct < 0.15:
            # Rich interactions: favor collaborative filtering
            self.cf_alpha = 0.70
            self.cf_beta = 0.30
            logger.info("Dense interactions detected (cold-start %.1f%%). Shifted baseline to CF=%.2f, CB=%.2f",
                        cold_user_pct * 100, self.cf_alpha, self.cf_beta)
        else:
            self.cf_alpha = 0.50
            self.cf_beta = 0.50

    def _get_user_weights(
        self,
        user_id: str,
        custom_alpha: Optional[float] = None,
        custom_beta: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Determine effective alpha and beta weights for a specific user request."""
        if custom_alpha is not None and custom_beta is not None:
            total = max(1e-6, custom_alpha + custom_beta)
            return custom_alpha / total, custom_beta / total
        elif custom_alpha is not None:
            a = np.clip(custom_alpha, 0.0, 1.0)
            return float(a), float(1.0 - a)
        elif custom_beta is not None:
            b = np.clip(custom_beta, 0.0, 1.0)
            return float(1.0 - b), float(b)

        # Dynamic per-user adaptation:
        if self.adapt_weights and self.data_processor is not None:
            n_interactions = len(self.data_processor.user_interactions.get(str(user_id), {}))
            if n_interactions == 0:
                # 100% cold-start user: pure content-based fallback
                return 0.10, 0.90
            elif n_interactions < 3:
                # Sparse user: strong content bias
                return 0.25, 0.75
            elif n_interactions >= 15:
                # Highly active user: strong collaborative bias
                return 0.70, 0.30

        return self.cf_alpha, self.cf_beta

    def predict_all_scores(
        self,
        user_id: str,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized computation of CF, CB, and Hybrid scores across ALL items.

        Returns:
            Tuple of (hybrid_scores, cf_scores, cb_scores) as np.ndarray of shape (N_items,)
        """
        if not self.is_fitted or self.data_processor is None or self.cf_model is None:
            raise RuntimeError("HybridRecommender must be fitted before generating scores.")

        n_items = len(self.data_processor.item_ids)
        if n_items == 0:
            empty = np.array([], dtype=np.float32)
            return empty, empty, empty

        eff_alpha, eff_beta = self._get_user_weights(user_id, alpha, beta)

        # 1. CF Scores (vectorized)
        cf_scores = self.cf_model.predict_all_items(str(user_id))

        # 2. CB Scores (vectorized BLAS dot product)
        user_profile = self.data_processor.get_user_profile(str(user_id))
        if user_profile is None:
            # Fall back to catalog mean embedding
            user_profile = self.data_processor.global_mean_embedding

        if user_profile is not None and self.data_processor.item_embeddings is not None:
            # (N_items, D) @ (D,) -> (N_items,)
            cos_sim = self.data_processor.item_embeddings @ user_profile
            # Scale cosine similarity from [-1, 1] to [0, 1]
            cb_scores = ((cos_sim + 1.0) / 2.0).astype(np.float32)
        else:
            cb_scores = np.full(n_items, 0.5, dtype=np.float32)

        # 3. Hybrid fusion
        hybrid_scores = (eff_alpha * cf_scores + eff_beta * cb_scores).astype(np.float32)
        return hybrid_scores, cf_scores, cb_scores

    def compute_hybrid_score(
        self,
        user_id: str,
        item_id: str,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Compute hybrid score for a single (user, item) pair."""
        if not self.is_fitted or self.data_processor is None:
            raise RuntimeError("Model is not fitted.")

        i_idx = self.data_processor.item_id_to_idx.get(str(item_id))
        if i_idx is None:
            return 0.5, 0.5, 0.5

        hybrid_all, cf_all, cb_all = self.predict_all_scores(user_id, alpha, beta)
        return float(hybrid_all[i_idx]), float(cf_all[i_idx]), float(cb_all[i_idx])

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        exclude_seen: bool = True,
    ) -> List[Tuple[str, float, float, float]]:
        """Generate top-N hybrid recommendations for a user.

        Returns:
            List of (item_id, hybrid_score, cf_score, cb_score) tuples sorted descending.
        """
        if not self.is_fitted or self.data_processor is None:
            raise RuntimeError("Model must be fitted before generating recommendations.")

        hybrid_scores, cf_scores, cb_scores = self.predict_all_scores(user_id, alpha, beta)
        scores_copy = hybrid_scores.copy()

        # Exclude seen items
        if exclude_seen and str(user_id) in self.data_processor.user_interactions:
            user_seen = self.data_processor.user_interactions[str(user_id)]
            for seen_iid in user_seen.keys():
                idx = self.data_processor.item_id_to_idx.get(seen_iid)
                if idx is not None:
                    scores_copy[idx] = -np.inf

        k = min(n, len(scores_copy))
        if k <= 0:
            return []

        # Fast argpartition for O(N) top-k selection
        top_indices = np.argpartition(-scores_copy, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores_copy[top_indices])]

        results: List[Tuple[str, float, float, float]] = []
        for idx in top_indices:
            if np.isfinite(scores_copy[idx]):
                iid = self.data_processor.item_idx_to_id[idx]
                results.append((
                    iid,
                    float(hybrid_scores[idx]),
                    float(cf_scores[idx]),
                    float(cb_scores[idx]),
                ))

        return results

    def recommend_batch(
        self,
        user_ids: List[str],
        n: int = 10,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        exclude_seen: bool = True,
    ) -> Dict[str, List[Tuple[str, float, float, float]]]:
        """Batch recommendation generation across multiple users."""
        results: Dict[str, List[Tuple[str, float, float, float]]] = {}
        for uid in user_ids:
            results[str(uid)] = self.recommend(
                user_id=str(uid),
                n=n,
                alpha=alpha,
                beta=beta,
                exclude_seen=exclude_seen,
            )
        return results

    def recommend_similar_items(
        self,
        item_id: str,
        n: int = 10,
    ) -> List[Tuple[str, float]]:
        """Item-to-item semantic similarity recommendation."""
        if not self.is_fitted or self.data_processor is None:
            raise RuntimeError("Model must be fitted.")

        target_idx = self.data_processor.item_id_to_idx.get(str(item_id))
        if target_idx is None or self.data_processor.item_embeddings is None:
            return []

        target_emb = self.data_processor.item_embeddings[target_idx]
        sims = self.data_processor.item_embeddings @ target_emb
        sims[target_idx] = -np.inf  # Exclude self

        k = min(n, len(sims) - 1)
        if k <= 0:
            return []

        top_indices = np.argpartition(-sims, k - 1)[:k]
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        return [
            (self.data_processor.item_idx_to_id[idx], float((sims[idx] + 1.0) / 2.0))
            for idx in top_indices
            if np.isfinite(sims[idx])
        ]

    def get_weights(self) -> Dict[str, float]:
        """Return base alpha and beta weights."""
        return {"cf_alpha": self.cf_alpha, "cf_beta": self.cf_beta}

    def get_fit_info(self) -> Dict[str, Any]:
        """Return model metadata and fitting statistics."""
        return {
            "is_fitted": self.is_fitted,
            "fit_time_seconds": round(self.fit_time, 3),
            "cf_method": self.cf_method,
            "cf_n_factors": self.cf_n_factors,
            "cf_reg": self.cf_reg,
            "cf_epochs": self.cf_epochs,
            "adapt_weights": self.adapt_weights,
            "cf_alpha": self.cf_alpha,
            "cf_beta": self.cf_beta,
            "total_items": len(self.data_processor.item_ids) if self.data_processor else 0,
            "total_users": len(self.data_processor.user_ids) if self.data_processor else 0,
        }

    def save(self, directory: str | Path) -> None:
        """Persist entire hybrid recommender artifacts to directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")

        # Save metadata
        meta = {
            "cf_method": self.cf_method,
            "cf_n_factors": self.cf_n_factors,
            "cf_reg": self.cf_reg,
            "cf_epochs": self.cf_epochs,
            "cf_alpha": self.cf_alpha,
            "cf_beta": self.cf_beta,
            "adapt_weights": self.adapt_weights,
            "fit_time": self.fit_time,
            "is_fitted": self.is_fitted,
        }
        with open(dir_path / "metadata.pkl", "wb") as f:
            pickle.dump(meta, f)

        # Save subcomponents
        if self.data_processor:
            self.data_processor.save(dir_path / "data_processor.pkl")
        if self.cf_model:
            self.cf_model.save(dir_path / "cf_model.pkl")

        logger.info("HybridRecommender saved to %s", dir_path)

    @classmethod
    def load(cls, directory: str | Path, redis_client: Optional[Any] = None) -> "HybridRecommender":
        """Load hybrid recommender from disk."""
        dir_path = Path(directory)
        with open(dir_path / "metadata.pkl", "rb") as f:
            meta = pickle.load(f)

        recommender = cls(
            cf_method=meta["cf_method"],
            cf_n_factors=meta["cf_n_factors"],
            cf_reg=meta["cf_reg"],
            cf_epochs=meta["cf_epochs"],
            cb_alpha=meta["cf_alpha"],
            cb_beta=meta["cf_beta"],
            adapt_weights=meta["adapt_weights"],
            redis_client=redis_client,
        )
        recommender.fit_time = meta["fit_time"]
        recommender.is_fitted = meta["is_fitted"]

        recommender.data_processor = DataProcessor.load(
            dir_path / "data_processor.pkl", redis_client=redis_client
        )
        recommender.cf_model = CollaborativeFiltering.load(dir_path / "cf_model.pkl")
        logger.info("Loaded HybridRecommender from %s", dir_path)
        return recommender
